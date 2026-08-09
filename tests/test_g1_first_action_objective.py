import unittest
from pathlib import Path
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.g1_gradient_audit import (
    PreparedE064Estimator,
    SharedTrajectory,
    pathwise_negative_objective,
    rollout_batched_environments,
)


@flax.struct.dataclass
class FakeState:
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
        return state.replace(
            obs=state.obs.at[0].add(action[0]),
            reward=action[0] + 0.25 * state.obs[0],
            done=step == 1,
            info={**state.info, "step": step + 1},
        )


def fake_actor(params, normalized_obs):
    return params["gain"] * normalized_obs[:1] + params["bias"]


def fake_state(value, *, phase, seed):
    return FakeState(
        obs=jnp.asarray([value, -2.0], dtype=jnp.float64),
        reward=jnp.asarray(0.0, dtype=jnp.float64),
        done=jnp.asarray(False),
        info={
            "rng": jax.random.PRNGKey(seed),
            "phase": jnp.asarray(phase, dtype=jnp.int32),
            "step": jnp.asarray(0, dtype=jnp.int32),
        },
    )


class FirstActionObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.env = SmoothFakeEnv()
        self.params = {
            "gain": jnp.asarray(0.5, dtype=jnp.float32),
            "bias": jnp.asarray([0.25], dtype=jnp.float32),
        }
        self.normalizer_state = {
            "mean": jnp.asarray([0.5, -1.0], dtype=jnp.float64),
            "scale": jnp.asarray([2.0, 4.0], dtype=jnp.float64),
        }
        self.initial_states = jax.vmap(
            lambda value, phase, seed: fake_state(value, phase=phase, seed=seed)
        )(
            jnp.asarray([1.0, 1.5], dtype=jnp.float64),
            jnp.asarray([10, 20], dtype=jnp.int32),
            jnp.asarray([3, 4], dtype=jnp.uint32),
        )
        self.action_noise = jnp.asarray(
            [[[0.5], [-0.5], [1.0]], [[-1.0], [0.25], [0.5]]],
            dtype=jnp.float64,
        )
        authoritative_core = jax.jit(
            lambda noise: rollout_batched_environments(
                self.params,
                fake_actor,
                self.env,
                normalizer=object(),
                normalizer_state=self.normalizer_state,
                initial_states=self.initial_states,
                action_noise=noise,
                sigma=0.5,
            )
        )
        self.authoritative_shared_trajectory, _ = authoritative_core(
            self.action_noise
        )
        self.prepared = PreparedE064Estimator(
            contract=SimpleNamespace(
                population=2,
                horizon=3,
                sigma=0.5,
                gamma=0.5,
            ),
            actor_params=self.params,
            normalizer_state=self.normalizer_state,
            initial_states=self.initial_states,
            checkpoint_path=Path("unused.pkl"),
            hparams={},
            env=self.env,
            normalizer=object(),
            actor_apply=fake_actor,
            compiled_core=lambda noise: noise,
        )

    def test_nominal_first_action_reproduces_the_frozen_rollout_bit_exactly(self):
        diagnostic = self.prepared.prepare_first_action_objective(
            self.action_noise,
            env_index=1,
            expected_shared_trajectory=self.authoritative_shared_trajectory,
        )

        replay, replay_final_state = diagnostic.rollout(diagnostic.nominal_first_action)

        for actual, expected in zip(
            jax.tree_util.tree_leaves(replay),
            jax.tree_util.tree_leaves(diagnostic.nominal_trajectory),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(replay_final_state),
            jax.tree_util.tree_leaves(diagnostic.nominal_final_state),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            diagnostic.objective(diagnostic.nominal_first_action),
            diagnostic.nominal_objective,
        )
        np.testing.assert_array_equal(
            diagnostic.nominal_objective,
            pathwise_negative_objective(
                diagnostic.nominal_trajectory.rewards,
                diagnostic.nominal_trajectory.dones,
                gamma=0.5,
            ),
        )

    def test_candidate_replaces_only_first_action_before_closed_loop_rollout(self):
        diagnostic = self.prepared.prepare_first_action_objective(
            self.action_noise,
            env_index=1,
            expected_shared_trajectory=self.authoritative_shared_trajectory,
        )
        candidate = diagnostic.nominal_first_action + jnp.asarray(
            [0.25], dtype=diagnostic.nominal_first_action.dtype
        )

        trajectory, _ = diagnostic.rollout(candidate)

        np.testing.assert_array_equal(trajectory.noise, self.action_noise[1])
        np.testing.assert_array_equal(trajectory.actions[0], candidate)
        np.testing.assert_array_equal(
            diagnostic.objective(candidate),
            pathwise_negative_objective(
                trajectory.rewards, trajectory.dones, gamma=0.5
            ),
        )
        gradient = jax.grad(diagnostic.objective)(diagnostic.nominal_first_action)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertNotEqual(float(jnp.linalg.norm(gradient)), 0.0)

    def test_preparation_rejects_invalid_noise_index_and_environment(self):
        invalid_cases = (
            (self.action_noise[:, :-1], 0, "shape"),
            (self.action_noise.at[0, 0, 0].set(jnp.nan), 0, "finite"),
            (self.action_noise, -1, "env_index"),
            (self.action_noise, 2, "env_index"),
            (self.action_noise, True, "env_index"),
        )
        for noise, index, message in invalid_cases:
            with (
                self.subTest(message=message, index=index),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                self.prepared.prepare_first_action_objective(
                    noise,
                    env_index=index,
                    expected_shared_trajectory=self.authoritative_shared_trajectory,
                )

        self.env.clip_actions = True
        with self.assertRaisesRegex(ValueError, "unbounded"):
            self.prepared.prepare_first_action_objective(
                self.action_noise,
                env_index=0,
                expected_shared_trajectory=self.authoritative_shared_trajectory,
            )

    def test_rejects_tampering_in_every_authoritative_trajectory_field(self):
        for field_name in SharedTrajectory._fields:
            source = np.array(
                getattr(self.authoritative_shared_trajectory, field_name), copy=True
            )
            index = (0,) * source.ndim
            if np.issubdtype(source.dtype, np.bool_):
                source[index] = not bool(source[index])
            else:
                source[index] += 1
            tampered = self.authoritative_shared_trajectory._replace(
                **{field_name: jnp.asarray(source)}
            )

            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(
                    ValueError,
                    rf"authoritative E011 shared trajectory {field_name}.*exact",
                ),
            ):
                self.prepared.prepare_first_action_objective(
                    self.action_noise,
                    env_index=0,
                    expected_shared_trajectory=tampered,
                )

    def test_rollout_rejects_invalid_candidate_action(self):
        diagnostic = self.prepared.prepare_first_action_objective(
            self.action_noise,
            env_index=0,
            expected_shared_trajectory=self.authoritative_shared_trajectory,
        )
        with self.assertRaisesRegex(ValueError, "candidate_action.*shape"):
            diagnostic.rollout(jnp.zeros((2,), dtype=jnp.float64))
        with self.assertRaisesRegex(ValueError, "candidate_action.*finite"):
            diagnostic.rollout(jnp.asarray([jnp.nan], dtype=jnp.float64))


if __name__ == "__main__":
    unittest.main()
