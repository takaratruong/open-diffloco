import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.g1_gradient_audit import (
    E064_REQUIRED_HPARAM_KEYS,
    E064_RUNTIME_HPARAMS,
    SharedTrajectory,
    assert_matching_identity_receipts,
    estimate_e064_shared_gradients,
    estimate_shared_gradients,
    identity_receipt,
    pathwise_negative_objective,
    pytree_shape_signature,
    rollout_one_environment,
    sha256_file,
    stable_pytree_sha256,
    validate_e064_checkpoint_contract,
)
from src.algorithms.shac.gradients import aggregate_per_env_gradients


@flax.struct.dataclass
class FakeState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


@flax.struct.dataclass
class FakeCarriedState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


class SmoothFakeEnv:
    clip_actions = False
    squash_actor_actions = False

    def _apply_obs_noise(self, obs, _rng):
        return obs

    def normalize_actor_obs(self, _normalizer, norm_state, obs):
        return (obs - norm_state["mean"]) / norm_state["scale"]

    def step(self, state, action):
        step = state.info["step"]
        next_obs = state.obs.at[0].add(action[0])
        reward = action[0] + 0.25 * state.obs[0]
        done = step == 1
        return state.replace(
            obs=next_obs,
            reward=reward,
            done=done,
            info={**state.info, "step": step + 1},
        )


def fake_actor(params, normalized_obs):
    return params["gain"] * normalized_obs[:1] + params["bias"]


def fake_state(value=1.0, *, phase=17, seed=3):
    return FakeState(
        obs=jnp.array([value, -2.0], dtype=jnp.float64),
        reward=jnp.array(0.0, dtype=jnp.float64),
        done=jnp.array(False),
        info={
            "rng": jax.random.PRNGKey(seed),
            "phase": jnp.array(phase, dtype=jnp.int32),
            "step": jnp.array(0, dtype=jnp.int32),
        },
    )


class SharedRolloutTest(unittest.TestCase):
    def setUp(self):
        self.env = SmoothFakeEnv()
        self.params = {
            "gain": jnp.array(0.4, dtype=jnp.float32),
            "bias": jnp.array([0.2], dtype=jnp.float32),
        }
        self.norm_state = {
            "mean": jnp.array([0.5, -1.0], dtype=jnp.float64),
            "scale": jnp.array([2.0, 4.0], dtype=jnp.float64),
        }
        self.noise = jnp.array([[1.0], [-2.0], [0.5]], dtype=jnp.float64)

    def test_rollout_materializes_exact_unbounded_actions_and_objective(self):
        trajectory, _ = rollout_one_environment(
            self.params,
            fake_actor,
            self.env,
            normalizer=object(),
            normalizer_state=self.norm_state,
            initial_state=fake_state(),
            action_noise=self.noise,
            sigma=0.1,
        )

        np.testing.assert_allclose(
            trajectory.actions,
            trajectory.means + 0.1 * trajectory.noise,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            trajectory.normalized_observations,
            (trajectory.observations - self.norm_state["mean"])
            / self.norm_state["scale"],
        )
        np.testing.assert_array_equal(
            trajectory.dones, jnp.array([False, True, False])
        )
        self.assertEqual(int(trajectory.initial_phase), 17)

        expected = -(
            trajectory.rewards[0]
            + 0.5 * trajectory.rewards[1]
            + trajectory.rewards[2]
        ) / 3.0
        actual = pathwise_negative_objective(
            trajectory.rewards, trajectory.dones, gamma=0.5
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_initial_state_is_stopped_but_actor_path_remains_differentiable(self):
        def loss_from_initial_observation(observation):
            trajectory, _ = rollout_one_environment(
                self.params,
                fake_actor,
                self.env,
                normalizer=object(),
                normalizer_state=self.norm_state,
                initial_state=fake_state().replace(obs=observation),
                action_noise=self.noise,
                sigma=0.1,
            )
            return pathwise_negative_objective(
                trajectory.rewards, trajectory.dones, gamma=0.5
            )

        initial_gradient = jax.grad(loss_from_initial_observation)(
            fake_state().obs
        )
        np.testing.assert_array_equal(
            initial_gradient, jnp.zeros_like(initial_gradient)
        )

        def loss_from_gain(gain):
            trajectory, _ = rollout_one_environment(
                {**self.params, "gain": gain},
                fake_actor,
                self.env,
                normalizer=object(),
                normalizer_state=self.norm_state,
                initial_state=fake_state(),
                action_noise=self.noise,
                sigma=0.1,
            )
            return pathwise_negative_objective(
                trajectory.rewards, trajectory.dones, gamma=0.5
            )

        self.assertNotEqual(float(jax.grad(loss_from_gain)(self.params["gain"])), 0.0)

    def test_rollout_is_jittable_with_the_frozen_static_sigma(self):
        compiled_rollout = jax.jit(
            lambda params, state, noise, sigma: rollout_one_environment(
                params,
                fake_actor,
                self.env,
                normalizer=object(),
                normalizer_state=self.norm_state,
                initial_state=state,
                action_noise=noise,
                sigma=sigma,
            )
        )

        trajectory, _ = compiled_rollout(
            self.params, fake_state(), self.noise, jnp.array(0.1)
        )

        self.assertEqual(trajectory.actions.shape, (3, 1))

    def test_batched_engine_feeds_identical_materialized_data_to_estimators(self):
        initial_states = jax.vmap(lambda x: fake_state(x, phase=10))(jnp.array([1.0, 1.5]))
        noise = jnp.stack([self.noise, -self.noise])

        result = estimate_shared_gradients(
            self.params,
            fake_actor,
            self.env,
            normalizer=object(),
            normalizer_state=self.norm_state,
            initial_states=initial_states,
            action_noise=noise,
            sigma=0.1,
            gamma=0.5,
            pathwise_clip_norm=1.0,
        )

        for field in (
            "noise",
            "observations",
            "normalized_observations",
            "means",
            "actions",
            "rewards",
            "dones",
            "initial_phase",
        ):
            np.testing.assert_array_equal(
                getattr(result.trajectory, field),
                getattr(result.score_trajectory, field),
            )
        np.testing.assert_allclose(
            result.trajectory.actions,
            result.trajectory.means + 0.1 * result.trajectory.noise,
            rtol=0.0,
            atol=0.0,
        )

        raw_leaves = jax.tree_util.tree_leaves(result.pathwise_raw_gradients)
        clipped_leaves = jax.tree_util.tree_leaves(
            result.pathwise_effective_gradients
        )
        score_leaves = jax.tree_util.tree_leaves(result.score_gradients)
        self.assertTrue(all(leaf.shape[0] == 2 for leaf in raw_leaves))
        self.assertTrue(all(leaf.shape[0] == 2 for leaf in clipped_leaves))
        self.assertTrue(all(leaf.shape[0] == 2 for leaf in score_leaves))
        self.assertTrue(np.all(np.asarray(result.pathwise_raw_norms) > 0.0))
        self.assertTrue(np.all(np.asarray(result.pathwise_clip_scales) <= 1.0))
        production_aggregate, _ = aggregate_per_env_gradients(
            result.pathwise_raw_gradients,
            max_norm=1.0,
        )
        effective_aggregate = jax.tree_util.tree_map(
            lambda leaf: jnp.mean(leaf, axis=0),
            result.pathwise_effective_gradients,
        )
        for actual, expected in zip(
            jax.tree_util.tree_leaves(effective_aggregate),
            jax.tree_util.tree_leaves(production_aggregate),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

        pathwise_receipt = identity_receipt(
            checkpoint_sha256="a" * 64,
            hparams=self._complete_hparams(),
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
            actor_params=self.params,
            normalizer_state=self.norm_state,
            initial_state=initial_states,
            action_noise=noise,
            trajectory=result.trajectory,
        )
        score_receipt = identity_receipt(
            checkpoint_sha256="a" * 64,
            hparams=self._complete_hparams(),
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
            actor_params=self.params,
            normalizer_state=self.norm_state,
            initial_state=initial_states,
            action_noise=noise,
            trajectory=result.score_trajectory,
        )
        assert_matching_identity_receipts(pathwise_receipt, score_receipt)

    @staticmethod
    def _complete_hparams():
        hparams = {key: None for key in E064_REQUIRED_HPARAM_KEYS}
        hparams.update(E064_RUNTIME_HPARAMS)
        return hparams

    def test_rollout_rejects_an_environment_that_clips_actions(self):
        self.env.clip_actions = True
        with self.assertRaisesRegex(ValueError, "unbounded"):
            rollout_one_environment(
                self.params,
                fake_actor,
                self.env,
                normalizer=object(),
                normalizer_state=self.norm_state,
                initial_state=fake_state(),
                action_noise=self.noise,
                sigma=0.1,
            )


class IdentityHashTest(unittest.TestCase):
    def test_hash_is_mapping_order_stable_and_mutation_sensitive(self):
        left = {"b": jnp.array([2, 3]), "a": jnp.array([1.0])}
        right = {"a": jnp.array([1.0]), "b": jnp.array([2, 3])}
        mutated = {"a": jnp.array([1.0]), "b": jnp.array([2, 4])}

        self.assertEqual(stable_pytree_sha256(left), stable_pytree_sha256(right))
        self.assertNotEqual(
            stable_pytree_sha256(left), stable_pytree_sha256(mutated)
        )

    def test_file_hash_and_receipt_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            path.write_bytes(b"E064")
            digest = sha256_file(path)
            self.assertEqual(len(digest), 64)
            path.write_bytes(b"E064-mutated")
            self.assertNotEqual(sha256_file(path), digest)

        complete = SharedRolloutTest._complete_hparams()
        arrays = jnp.array([1.0])
        fake_trajectory = SharedTrajectory(
            noise=arrays,
            observation_rngs=jnp.array([[0, 0]], dtype=jnp.uint32),
            observations=arrays,
            normalized_observations=arrays,
            means=arrays,
            actions=arrays,
            rewards=arrays,
            dones=jnp.array([False]),
            initial_phase=jnp.array(0),
        )
        left = identity_receipt(
            checkpoint_sha256="a" * 64,
            hparams=complete,
            gamma=0.99,
            sigma=0.1,
            pathwise_clip_norm=1.0,
            actor_params={"p": arrays},
            normalizer_state={"mean": arrays},
            initial_state={"obs": arrays},
            action_noise=arrays,
            trajectory=fake_trajectory,
        )
        right = dict(left)
        right["actions"] = "changed"
        with self.assertRaisesRegex(ValueError, "actions"):
            assert_matching_identity_receipts(left, right)


class FrozenCheckpointContractTest(unittest.TestCase):
    def setUp(self):
        self.actor_params = {
            "params": {
                "Dense_0": {
                    "kernel": jnp.zeros((154, 512), dtype=jnp.float32),
                    "bias": jnp.zeros((512,), dtype=jnp.float32),
                },
                "Dense_1": {
                    "kernel": jnp.zeros((512, 512), dtype=jnp.float32),
                    "bias": jnp.zeros((512,), dtype=jnp.float32),
                },
                "Dense_2": {
                    "kernel": jnp.zeros((512, 29), dtype=jnp.float32),
                    "bias": jnp.zeros((29,), dtype=jnp.float32),
                },
            }
        }
        self.normalizer = SimpleNamespace(
            mean=jnp.zeros((154,), dtype=jnp.float32),
            var=jnp.ones((154,), dtype=jnp.float32),
            count=jnp.array(100.0, dtype=jnp.float32),
        )
        self.env_state = {
            "obs": jnp.zeros((64, 154), dtype=jnp.float32),
            "reward": jnp.zeros((64,), dtype=jnp.float32),
            "done": jnp.zeros((64,), dtype=jnp.float32),
            "info": {
                "rng": jnp.zeros((64, 2), dtype=jnp.uint32),
                "phase": jnp.arange(64, dtype=jnp.int32),
                "last_act": jnp.zeros((64, 29), dtype=jnp.float32),
            },
        }
        self.state = SimpleNamespace(
            actor_params=self.actor_params,
            normalizer=self.normalizer,
            env_state=self.env_state,
        )
        self.hparams = {key: None for key in E064_REQUIRED_HPARAM_KEYS}
        self.hparams.update(E064_RUNTIME_HPARAMS)
        self.state_signature = pytree_shape_signature(self.env_state)

    def validate(self, *, hparams=None, state=None, signature=None):
        return validate_e064_checkpoint_contract(
            self.state if state is None else state,
            self.hparams if hparams is None else hparams,
            expected_hparams=self.hparams,
            expected_initial_state_signature=(
                self.state_signature if signature is None else signature
            ),
        )

    def test_accepts_exact_actor_normalizer_state_and_complete_hparams(self):
        validated = self.validate()
        self.assertEqual(validated.population, 64)
        self.assertEqual(validated.horizon, 48)
        self.assertEqual(validated.sigma, 0.1)

    def test_exact_engine_binds_validated_population_horizon_and_action_shape(self):
        contract = self.validate()
        with self.assertRaisesRegex(ValueError, "64, 48, 29"):
            estimate_e064_shared_gradients(
                contract,
                self.actor_params,
                SmoothFakeEnv(),
                normalizer=object(),
                normalizer_state=self.normalizer,
                initial_states=self.env_state,
                action_noise=jnp.zeros((64, 47, 29)),
            )

    def test_accepts_real_flax_attribute_paths_for_carried_state(self):
        carried_state = FakeCarriedState(
            obs=self.env_state["obs"],
            reward=self.env_state["reward"],
            done=self.env_state["done"],
            info=self.env_state["info"],
        )
        state = SimpleNamespace(
            actor_params=self.actor_params,
            normalizer=self.normalizer,
            env_state=carried_state,
        )

        validated = self.validate(
            state=state,
            signature=pytree_shape_signature(carried_state),
        )

        self.assertEqual(validated.population, 64)

    def test_rejects_architecture_normalizer_and_state_shape_mutations(self):
        params = copy.deepcopy(self.actor_params)
        params["params"]["LayerNorm_0"] = {
            "scale": jnp.ones((512,), dtype=jnp.float32)
        }
        with self.assertRaisesRegex(ValueError, "actor parameter shapes"):
            self.validate(
                state=SimpleNamespace(
                    actor_params=params,
                    normalizer=self.normalizer,
                    env_state=self.env_state,
                )
            )

        wrong_normalizer = SimpleNamespace(
            mean=jnp.zeros((153,)), var=jnp.ones((154,)), count=jnp.array(1.0)
        )
        with self.assertRaisesRegex(ValueError, "normalizer"):
            self.validate(
                state=SimpleNamespace(
                    actor_params=self.actor_params,
                    normalizer=wrong_normalizer,
                    env_state=self.env_state,
                )
            )

        mutated_state = {
            **self.env_state,
            "obs": jnp.zeros((64, 153), dtype=jnp.float32),
        }
        with self.assertRaisesRegex(ValueError, "initial-state shape"):
            self.validate(
                state=SimpleNamespace(
                    actor_params=self.actor_params,
                    normalizer=self.normalizer,
                    env_state=mutated_state,
                )
            )

    def test_rejects_incomplete_or_causally_unsafe_hparams(self):
        incomplete = dict(self.hparams)
        incomplete.pop("reference_sha256")
        with self.assertRaisesRegex(ValueError, "complete frozen hparams"):
            self.validate(hparams=incomplete)

        mutations = {
            "terrain": True,
            "push_velocity_range": [-0.1, 0.1],
            "kp_range": [0.9, 1.1],
            "reference_reset_noise_scale": 0.01,
            "observation_noise_mode": "uniform",
            "squash_actor_actions": True,
            "actor_bootstrap_scale": 1.0,
            "num_envs": 63,
            "unroll_length": 47,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                hparams = dict(self.hparams)
                hparams[key] = value
                expected = dict(hparams)
                with self.assertRaisesRegex(ValueError, key):
                    validate_e064_checkpoint_contract(
                        self.state,
                        hparams,
                        expected_hparams=expected,
                        expected_initial_state_signature=self.state_signature,
                    )


if __name__ == "__main__":
    unittest.main()
