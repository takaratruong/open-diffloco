import copy
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import flax
import jax
import jax.numpy as jnp
import numpy as np

import src.algorithms.shac.g1_gradient_audit as g1_runtime
from src.algorithms.shac.g1_gradient_audit import (
    E064_ACTOR_PARAMETERS_SHA256,
    E064_CHECKPOINT_SHA256,
    E064_FROZEN_HPARAMS,
    E064_HPARAMS_SHA256,
    E064_INITIAL_STATE_SHA256,
    E064_NORMALIZER_SHA256,
    E064_REFERENCE_SHA256,
    SharedTrajectory,
    assert_matching_identity_receipts,
    build_and_validate_estimator_receipts,
    estimate_shared_gradients,
    identity_receipt,
    pathwise_negative_objective,
    prepare_compiled_estimator_core,
    prepare_compiled_rollout_core,
    prepare_e064_estimator_engine,
    pytree_shape_signature,
    rollout_one_environment,
    sha256_file,
    stable_mapping_sha256,
    stable_pytree_sha256,
    validate_e064_checkpoint_contract,
    validate_e064_checkpoint_shapes,
    validate_e064_live_semantics,
    validate_identity_observation_handling,
)
from src.algorithms.shac.gradient_audit import discounted_return_to_go
from src.algorithms.shac.gradients import aggregate_per_env_gradients
from src.core.data_structures import Normalizer
from src.envs.g1_tracking.environment import G1TrackingRMR50HzValidatedEnv


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

    def __init__(self, observation_noise_scale=0.0):
        self.observation_noise_scale = observation_noise_scale
        self.observation_noise_calls = 0

    def _apply_obs_noise(self, obs, rng):
        self.observation_noise_calls += 1
        return obs + self.observation_noise_scale * rng[0].astype(obs.dtype)

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

    def test_actor_reconstruction_gate_is_functional_and_fail_closed(self):
        expected = jnp.linspace(-1.0, 1.0, 100, dtype=jnp.float64)
        bounded = expected.at[17].add(1.7e-3)
        evidence = g1_runtime._validate_actor_reconstruction(
            bounded,
            expected,
        )
        self.assertLessEqual(evidence["maximum_absolute_error"], 0.005)
        self.assertLessEqual(evidence["rms_error"], 0.0005)
        self.assertGreaterEqual(evidence["cosine"], 0.9999999)

        with self.assertRaisesRegex(ValueError, "maximum absolute"):
            g1_runtime._validate_actor_reconstruction(
                expected.at[17].add(5.1e-3),
                expected,
            )
        with self.assertRaisesRegex(ValueError, "RMS"):
            g1_runtime._validate_actor_reconstruction(
                expected + 6e-4,
                expected,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            g1_runtime._validate_actor_reconstruction(
                expected[:-1],
                expected,
            )

    def test_score_gradient_gate_is_functional_and_checks_aggregate(self):
        expected = {"w": jnp.array([[100.0], [-99.0]], dtype=jnp.float64)}
        evidence = g1_runtime._validate_score_gradient_reconstruction(
            expected, expected
        )
        self.assertEqual(evidence["score_gradient_relative_l2_error"], 0.0)
        self.assertEqual(evidence["score_mean_gradient_cosine"], 1.0)

        with self.assertRaisesRegex(ValueError, "score gradients"):
            g1_runtime._validate_score_gradient_reconstruction(
                jax.tree_util.tree_map(jnp.zeros_like, expected), expected
            )
        with self.assertRaisesRegex(ValueError, "score gradients"):
            g1_runtime._validate_score_gradient_reconstruction(
                jax.tree_util.tree_map(lambda leaf: 1.006 * leaf, expected),
                expected,
            )
        with self.assertRaisesRegex(ValueError, "aggregate-mean"):
            g1_runtime._validate_score_gradient_reconstruction(
                {"w": jnp.array([[100.1], [-99.0]], dtype=jnp.float64)},
                expected,
            )
        with self.assertRaisesRegex(ValueError, "tree structure"):
            g1_runtime._validate_score_gradient_reconstruction(
                {"other": expected["w"]}, expected
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            g1_runtime._validate_score_gradient_reconstruction(
                {"w": expected["w"].at[0, 0].set(jnp.nan)}, expected
            )

        zeros = jax.tree_util.tree_map(jnp.zeros_like, expected)
        zero_evidence = g1_runtime._validate_score_gradient_reconstruction(
            zeros, zeros
        )
        self.assertEqual(zero_evidence["score_gradient_cosine"], 1.0)

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

    def test_rollout_calls_observation_hook_with_the_recorded_split_key(self):
        env = SmoothFakeEnv(observation_noise_scale=0.01)
        trajectory, _ = rollout_one_environment(
            self.params,
            fake_actor,
            env,
            normalizer=object(),
            normalizer_state=self.norm_state,
            initial_state=fake_state(),
            action_noise=self.noise,
            sigma=0.1,
        )

        expected_first = fake_state().obs + (
            0.01
            * trajectory.observation_rngs[0, 0].astype(fake_state().obs.dtype)
        )
        np.testing.assert_array_equal(
            trajectory.observations[0], expected_first
        )
        self.assertEqual(env.observation_noise_calls, 1)

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
            "raw_observations",
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
        np.testing.assert_array_equal(
            result.score_means, result.trajectory.means
        )

        def reference_score_loss(params, data):
            means = jax.vmap(lambda obs: fake_actor(params, obs))(
                data.normalized_observations
            ).astype(data.means.dtype)
            returns = discounted_return_to_go(
                data.rewards, data.dones, gamma=0.5
            )
            return g1_runtime.detached_gaussian_score_loss(
                means, data.actions, returns, std=0.1
            )

        reference_score_gradients = jax.vmap(
            jax.grad(reference_score_loss), in_axes=(None, 0)
        )(self.params, result.score_trajectory)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(result.score_gradients),
            jax.tree_util.tree_leaves(reference_score_gradients),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

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
            objective_values=result.losses,
            returns_to_go=jax.vmap(
                lambda rewards, dones: discounted_return_to_go(
                    rewards, dones, gamma=0.5
                )
            )(result.trajectory.rewards, result.trajectory.dones),
            score_losses=result.score_losses,
        )
        score_receipt_trajectory = result.score_trajectory._replace(
            means=result.trajectory.means
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
            trajectory=score_receipt_trajectory,
            objective_values=jax.vmap(
                lambda rewards, dones: pathwise_negative_objective(
                    rewards, dones, gamma=0.5
                )
            )(
                result.score_trajectory.rewards,
                result.score_trajectory.dones,
            ),
            returns_to_go=result.score_returns_to_go,
            score_losses=result.score_losses,
        )
        assert_matching_identity_receipts(pathwise_receipt, score_receipt)

        receipts = build_and_validate_estimator_receipts(
            checkpoint_sha256="a" * 64,
            hparams=self._complete_hparams(),
            actor_params=self.params,
            actor_apply=fake_actor,
            normalizer_state=self.norm_state,
            initial_state=initial_states,
            action_noise=noise,
            result=result,
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
        )
        self.assertEqual(receipts[0], receipts[1])
        self.assertEqual(
            receipts[0]["numeric_equivalence_metrics"],
            receipts[1]["numeric_equivalence_metrics"],
        )
        self.assertIn(
            "mean_rms_error",
            receipts[0]["numeric_equivalence_metrics"],
        )
        self.assertEqual(
            receipts[0]["numeric_equivalence_evidence"],
            stable_mapping_sha256(receipts[0]["numeric_equivalence_metrics"]),
        )
        required_numeric_evidence = {
            "mean_rms_error",
            "pathwise_clipping_gate_ulps",
            "score_gradient_relative_l2_error",
            "score_gradient_max_relative_l2_gate",
            "score_gradient_cosine",
            "score_gradient_minimum_cosine",
            "score_mean_gradient_relative_l2_error",
            "score_mean_gradient_max_relative_l2_gate",
            "score_mean_gradient_cosine",
            "score_mean_gradient_minimum_cosine",
        }
        self.assertTrue(
            required_numeric_evidence
            <= receipts[0]["numeric_equivalence_metrics"].keys()
        )
        self.assertTrue(
            all(
                np.isfinite(value)
                for value in receipts[0]["numeric_equivalence_metrics"].values()
            )
        )

        coherent_raw = jax.tree_util.tree_map(
            lambda leaf: 0.75 * leaf, result.pathwise_raw_gradients
        )
        coherent_effective, coherent_norms, coherent_scales = (
            g1_runtime._pathwise_effective_per_environment(
                coherent_raw, max_norm=1.0
            )
        )
        coherent_pathwise = result._replace(
            pathwise_raw_gradients=coherent_raw,
            pathwise_effective_gradients=coherent_effective,
            pathwise_raw_norms=coherent_norms,
            pathwise_clip_scales=coherent_scales,
        )
        coherent_receipts = build_and_validate_estimator_receipts(
            checkpoint_sha256="a" * 64,
            hparams=self._complete_hparams(),
            actor_params=self.params,
            actor_apply=fake_actor,
            normalizer_state=self.norm_state,
            initial_state=initial_states,
            action_noise=noise,
            result=coherent_pathwise,
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
        )
        self.assertNotEqual(
            coherent_receipts[0]["engine_estimator_values"],
            receipts[0]["engine_estimator_values"],
        )

        bounded_score = result._replace(
            score_gradients=jax.tree_util.tree_map(
                lambda leaf: 1.001 * leaf, result.score_gradients
            )
        )
        bounded_score_receipts = build_and_validate_estimator_receipts(
            checkpoint_sha256="a" * 64,
            hparams=self._complete_hparams(),
            actor_params=self.params,
            actor_apply=fake_actor,
            normalizer_state=self.norm_state,
            initial_state=initial_states,
            action_noise=noise,
            result=bounded_score,
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
        )
        self.assertNotEqual(
            bounded_score_receipts[0]["engine_estimator_values"],
            receipts[0]["engine_estimator_values"],
        )

        zero_raw = result._replace(
            pathwise_raw_gradients=jax.tree_util.tree_map(
                jnp.zeros_like, result.pathwise_raw_gradients
            )
        )
        with self.assertRaisesRegex(ValueError, "pathwise clipping"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=zero_raw,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )
        zero_effective = result._replace(
            pathwise_effective_gradients=jax.tree_util.tree_map(
                jnp.zeros_like, result.pathwise_effective_gradients
            )
        )
        with self.assertRaisesRegex(ValueError, "pathwise clipping"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=zero_effective,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )
        zero_clipping_pair = result._replace(
            pathwise_raw_gradients=jax.tree_util.tree_map(
                jnp.zeros_like, result.pathwise_raw_gradients
            ),
            pathwise_effective_gradients=jax.tree_util.tree_map(
                jnp.zeros_like, result.pathwise_effective_gradients
            ),
        )
        with self.assertRaisesRegex(ValueError, "pathwise clipping"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=zero_clipping_pair,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )
        zero_score = result._replace(
            score_gradients=jax.tree_util.tree_map(
                jnp.zeros_like, result.score_gradients
            )
        )
        with self.assertRaisesRegex(ValueError, "score gradients"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=zero_score,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        # The actor fused inside a large differentiable MJX graph and the same
        # actor compiled alone need not be ULP-identical on GPU.  Admit only a
        # small functional discrepancy, well below the 0.01 output-RMS audit
        # step, while retaining exact identity between the two in-engine
        # estimator consumers.
        rounded_receipts = build_and_validate_estimator_receipts(
            checkpoint_sha256="a" * 64,
            hparams=self._complete_hparams(),
            actor_params=self.params,
            actor_apply=lambda params, obs: (
                fake_actor(params, obs)
                + 2.5e-4 * jnp.sin(jnp.sum(obs)).astype(jnp.float32)
            ),
            normalizer_state=self.norm_state,
            initial_state=initial_states,
            action_noise=noise,
            result=result,
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
        )
        self.assertEqual(rounded_receipts[0], rounded_receipts[1])

        original_score_loss = g1_runtime.detached_gaussian_score_loss
        with (
            mock.patch.object(
                g1_runtime,
                "detached_gaussian_score_loss",
                side_effect=lambda *args, **kwargs: (
                    original_score_loss(*args, **kwargs) + 0.1
                ),
            ),
            self.assertRaisesRegex(ValueError, "independent score_losses"),
        ):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=result,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        def raising_actor(*_args):
            raise RuntimeError("independent actor was called")

        with self.assertRaisesRegex(RuntimeError, "independent actor was called"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=raising_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=result,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        with self.assertRaisesRegex(
            ValueError, "independent actor reconstruction"
        ):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=lambda params, obs: fake_actor(params, obs) + 1e-3,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=result,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        altered_noise = result._replace(
            trajectory=result.trajectory._replace(
                noise=result.trajectory.noise.at[0, 0, 0].add(1.0)
            ),
            score_trajectory=result.score_trajectory._replace(
                noise=result.score_trajectory.noise.at[0, 0, 0].add(1.0)
            ),
        )
        with self.assertRaisesRegex(ValueError, "noise"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=altered_noise,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        altered = result._replace(
            score_means=result.score_means.at[0, 0, 0].add(1.0)
        )
        with self.assertRaisesRegex(ValueError, "engine means"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=altered,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        one_ulp = result._replace(
            score_means=result.score_means.at[0, 0, 0].set(
                jnp.nextafter(
                    result.score_means[0, 0, 0],
                    jnp.asarray(jnp.inf, dtype=result.score_means.dtype),
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "engine means"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=one_ulp,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

        altered_objective = result._replace(losses=result.losses + 1.0)
        with self.assertRaisesRegex(ValueError, "objective_values"):
            build_and_validate_estimator_receipts(
                checkpoint_sha256="a" * 64,
                hparams=self._complete_hparams(),
                actor_params=self.params,
                actor_apply=fake_actor,
                normalizer_state=self.norm_state,
                initial_state=initial_states,
                action_noise=noise,
                result=altered_objective,
                gamma=0.5,
                sigma=0.1,
                pathwise_clip_norm=1.0,
            )

    @staticmethod
    def _complete_hparams():
        return dict(E064_FROZEN_HPARAMS)

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

    def test_hash_binds_empty_nodes_and_container_types(self):
        leaf = jnp.array([1.0])

        self.assertNotEqual(
            stable_pytree_sha256({"a": leaf}),
            stable_pytree_sha256({"a": leaf, "empty": {}}),
        )
        self.assertNotEqual(
            stable_pytree_sha256([leaf]), stable_pytree_sha256((leaf,))
        )
        unsupported = jax.tree_util.Partial(lambda value: value, leaf)
        with self.assertRaisesRegex(TypeError, "unsupported static PyTree"):
            stable_pytree_sha256(unsupported)

        class ArrayLookalike:
            def __init__(self):
                self.array = np.array([1], dtype=np.int32)
                self.secret = "unhashed-causal-state"

        with self.assertRaisesRegex(TypeError, "unsupported static PyTree"):
            g1_runtime._stable_aux_schema(ArrayLookalike())

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
            raw_observations=arrays,
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
            objective_values=arrays,
            returns_to_go=arrays,
            score_losses=arrays,
        )
        right = dict(left)
        right["actions"] = "changed"
        with self.assertRaisesRegex(ValueError, "actions"):
            assert_matching_identity_receipts(left, right)

    def test_actual_e064_hashes_repeat_across_fresh_processes(self):
        checkpoint = Path(
            "/home/ubuntu/artifacts/open-diffloco/E-20260808-064/run_root/"
            "training_runs/shac_20260808_145643/policy_final.pkl"
        )
        if not checkpoint.exists():
            self.skipTest("local immutable E064 checkpoint is unavailable")
        script = f"""
import pickle
from src.algorithms.shac.g1_gradient_audit import (
    _normalizer_identity_tree, stable_pytree_sha256,
)
with open({str(checkpoint)!r}, 'rb') as stream:
    state = pickle.load(stream)
print('|'.join((
    stable_pytree_sha256(state.actor_params),
    stable_pytree_sha256(_normalizer_identity_tree(state.normalizer)),
    stable_pytree_sha256(state.env_state),
)))
"""
        environment = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
        }
        receipts = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).parents[1],
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            receipts.append(completed.stdout.strip().splitlines()[-1])

        self.assertEqual(receipts[0], receipts[1])
        self.assertEqual(
            receipts[0],
            (
                f"{E064_ACTOR_PARAMETERS_SHA256}|"
                f"{E064_NORMALIZER_SHA256}|{E064_INITIAL_STATE_SHA256}"
            ),
        )

    def test_actual_e064_hash_and_snapshot_bind_mjx_static_aux_arrays(self):
        checkpoint = Path(
            "/home/ubuntu/artifacts/open-diffloco/E-20260808-064/run_root/"
            "training_runs/shac_20260808_145643/policy_final.pkl"
        )
        if not checkpoint.exists():
            self.skipTest("local immutable E064 checkpoint is unavailable")
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        snapshot = g1_runtime._snapshot_pytree(state.env_state)

        def wrapped_arrays(tree):
            arrays = []

            def inspect_aux(value):
                if type(value).__name__ == "_NumPyArrayHashWrapper":
                    arrays.append(value.array)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        inspect_aux(item)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        inspect_aux(key)
                        inspect_aux(item)

            def visit(tree_definition):
                node_data = tree_definition.node_data()
                if node_data is not None:
                    inspect_aux(node_data[1])
                for child in tree_definition.children():
                    visit(child)

            visit(jax.tree_util.tree_structure(tree))
            return arrays

        original_arrays = wrapped_arrays(state.env_state)
        snapshot_arrays = wrapped_arrays(snapshot)
        self.assertTrue(original_arrays)
        self.assertFalse(
            any(
                original is copied
                for original, copied in zip(
                    original_arrays, snapshot_arrays, strict=True
                )
            )
        )
        original_hash = stable_pytree_sha256(state.env_state)
        snapshot_hash = stable_pytree_sha256(snapshot)
        original_arrays[0].flat[0] += 1
        self.assertNotEqual(stable_pytree_sha256(state.env_state), original_hash)
        self.assertEqual(stable_pytree_sha256(snapshot), snapshot_hash)


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
        self.hparams = dict(E064_FROZEN_HPARAMS)
        self.state_signature = pytree_shape_signature(self.env_state)

    def validate(self, *, hparams=None, state=None, signature=None):
        if hparams is not None:
            raise AssertionError("shape-only validation does not accept hparams")
        return validate_e064_checkpoint_shapes(
            self.state if state is None else state,
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
        self.assertEqual(E064_CHECKPOINT_SHA256, "6b5c6bb208f9acd9f5988fee201915f8aa67cba42c15231d361a4d2ae530a094")
        self.assertEqual(E064_HPARAMS_SHA256, "98499799f221978510ee15b9417a4e408a6a0ae1aff95c9f84b48a4cc88a9c8b")
        self.assertEqual(E064_REFERENCE_SHA256, "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db")

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
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            checkpoint.write_bytes(b"forged")
            with self.assertRaisesRegex(ValueError, "hparams"):
                validate_e064_checkpoint_contract(
                    self.state, incomplete, checkpoint_path=checkpoint
                )

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
                with (
                    self.assertRaisesRegex(ValueError, key),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    checkpoint = Path(directory) / "checkpoint.pkl"
                    checkpoint.write_bytes(b"forged")
                    validate_e064_checkpoint_contract(
                        self.state,
                        hparams,
                        checkpoint_path=checkpoint,
                    )

    def test_caller_cannot_change_actual_and_expected_hparams_together(self):
        forged = dict(E064_FROZEN_HPARAMS)
        forged["best_reward"] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            checkpoint.write_bytes(b"forged")
            with self.assertRaisesRegex(ValueError, "literal frozen E064 hparams"):
                validate_e064_checkpoint_contract(
                    self.state,
                    forged,
                    checkpoint_path=checkpoint,
                )

    def test_actual_e064_checkpoint_hparams_and_state_match_literals(self):
        root = Path(
            "/home/ubuntu/artifacts/open-diffloco/E-20260808-064/run_root/"
            "training_runs/shac_20260808_145643"
        )
        checkpoint = root / "policy_final.pkl"
        if not checkpoint.exists():
            self.skipTest("local immutable E064 checkpoint is unavailable")
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        hparams = json.loads((root / "hparams.json").read_text())

        contract = validate_e064_checkpoint_contract(
            state, hparams, checkpoint_path=checkpoint
        )

        self.assertEqual(
            contract.actor_parameters_sha256,
            E064_ACTOR_PARAMETERS_SHA256,
        )
        self.assertEqual(contract.normalizer_sha256, E064_NORMALIZER_SHA256)
        self.assertEqual(
            contract.initial_state_sha256,
            E064_INITIAL_STATE_SHA256,
        )

    def test_live_semantics_reject_same_shape_normalizer_and_altered_plant(self):
        class AlteredNormalizer(Normalizer):
            def normalize(self, state, values):
                return super().normalize(state, values) + 1.0

        with self.assertRaisesRegex(ValueError, "exact Normalizer"):
            validate_e064_live_semantics(
                SimpleNamespace(reference_sha256=E064_REFERENCE_SHA256),
                object(),
                AlteredNormalizer(154),
                self.normalizer,
                self.env_state,
            )

        env = object.__new__(G1TrackingRMR50HzValidatedEnv)
        env.body_mass_scale = 0.9
        env.xml_path = "/tmp/model"
        env.controller_path = "/tmp/controller"
        with (
            mock.patch.object(
                g1_runtime,
                "sha256_file",
                side_effect=[
                    g1_runtime.E064_XML_SHA256,
                    g1_runtime.E064_CONTROLLER_SHA256,
                ],
            ),
            self.assertRaisesRegex(ValueError, "body_mass_scale"),
        ):
            validate_e064_live_semantics(
                SimpleNamespace(reference_sha256=E064_REFERENCE_SHA256),
                env,
                Normalizer(154),
                self.normalizer,
                self.env_state,
            )

        for name, digests, message in (
            ("xml_path", ["0" * 64], "physical XML"),
            (
                "controller_path",
                [g1_runtime.E064_XML_SHA256, "0" * 64],
                "controller",
            ),
        ):
            with self.subTest(name=name):
                env = object.__new__(G1TrackingRMR50HzValidatedEnv)
                env.xml_path = "/tmp/model"
                env.controller_path = "/tmp/controller"
                with (
                    mock.patch.object(
                        g1_runtime, "sha256_file", side_effect=digests
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    validate_e064_live_semantics(
                        SimpleNamespace(reference_sha256=E064_REFERENCE_SHA256),
                        env,
                        Normalizer(154),
                        self.normalizer,
                        self.env_state,
                    )

    def test_identity_observation_validation_rejects_nonidentity_hook(self):
        states = jax.vmap(lambda value: fake_state(value))(
            jnp.array([1.0, 2.0])
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_identity_observation_handling(
                SmoothFakeEnv(observation_noise_scale=0.01), states
            )

    def test_live_semantics_bind_runtime_arrays_and_mjx_model(self):
        env = G1TrackingRMR50HzValidatedEnv(
            xml_path=E064_FROZEN_HPARAMS["xml_path"],
            reference_path=E064_FROZEN_HPARAMS["reference_path"],
            actor_history_len=1,
            reference_stride=1,
            mass_range=(1.0, 1.0),
            effort_limit_scale=1.0,
            termination_margin_weight=0.0,
            reference_reset_noise_scale=0.0,
        )
        contract = SimpleNamespace(reference_sha256=E064_REFERENCE_SHA256)
        validate_e064_live_semantics(
            contract,
            env,
            Normalizer(154),
            self.normalizer,
            self.env_state,
        )

        original_kp = env.kp
        env.kp = env.kp.at[0].add(1.0)
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            validate_e064_live_semantics(
                contract,
                env,
                Normalizer(154),
                self.normalizer,
                self.env_state,
            )
        env.kp = original_kp

        original_model = env.mjx_model
        env.mjx_model = env.mjx_model.replace(
            body_mass=env.mjx_model.body_mass.at[1].add(1.0)
        )
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            validate_e064_live_semantics(
                contract,
                env,
                Normalizer(154),
                self.normalizer,
                self.env_state,
            )
        env.mjx_model = original_model

        wrapped_arrays = []

        def inspect_aux(value):
            if type(value).__name__ == "_NumPyArrayHashWrapper":
                wrapped_arrays.append(value.array)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    inspect_aux(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    inspect_aux(key)
                    inspect_aux(item)

        def visit(tree_definition):
            node_data = tree_definition.node_data()
            if node_data is not None:
                inspect_aux(node_data[1])
            for child in tree_definition.children():
                visit(child)

        visit(jax.tree_util.tree_structure(env.mjx_model))
        self.assertTrue(wrapped_arrays)
        wrapped_arrays[0].flat[0] += 1
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            validate_e064_live_semantics(
                contract,
                env,
                Normalizer(154),
                self.normalizer,
                self.env_state,
            )


class CompiledCoreBoundaryTest(unittest.TestCase):
    def test_exact_factory_host_validates_before_building_compiled_core(self):
        events = []
        state = SimpleNamespace(
            actor_params={"p": jnp.array(1.0)},
            normalizer=SimpleNamespace(
                mean=jnp.zeros((1,)), var=jnp.ones((1,)), count=jnp.array(1.0)
            ),
            env_state={"obs": jnp.zeros((1, 1))},
        )
        contract = SimpleNamespace(
            sigma=0.1,
            gamma=0.99,
            pathwise_clip_norm=1.0,
            reference_sha256=E064_REFERENCE_SHA256,
            hparams_sha256=E064_HPARAMS_SHA256,
            actor_parameters_sha256=stable_pytree_sha256(state.actor_params),
            normalizer_sha256=stable_pytree_sha256(
                g1_runtime._normalizer_identity_tree(state.normalizer)
            ),
            initial_state_sha256=stable_pytree_sha256(state.env_state),
        )

        def validate_contract(*_args, **_kwargs):
            events.append("literal-validation")
            return contract

        def validate_live(*_args, **_kwargs):
            events.append("live-validation")

        def build_core(*_args, **_kwargs):
            self.assertEqual(
                events, ["literal-validation", "live-validation"]
            )
            events.append("compiled-core")
            return mock.Mock(name="compiled_core")

        with (
            mock.patch.object(
                g1_runtime,
                "validate_e064_checkpoint_contract",
                side_effect=validate_contract,
            ),
            mock.patch.object(
                g1_runtime,
                "validate_e064_live_semantics",
                side_effect=validate_live,
            ),
            mock.patch.object(
                g1_runtime,
                "prepare_compiled_estimator_core",
                side_effect=build_core,
            ),
        ):
            prepared = prepare_e064_estimator_engine(
                state,
                dict(E064_FROZEN_HPARAMS),
                checkpoint_path=Path("checkpoint.pkl"),
                env=SmoothFakeEnv(),
                normalizer=object(),
            )

        self.assertEqual(
            events,
            ["literal-validation", "live-validation", "compiled-core"],
        )
        self.assertIs(prepared.contract, contract)

    def test_host_preparation_returns_reusable_same_shape_jitted_core(self):
        env = SmoothFakeEnv()
        params = {"gain": jnp.array(0.4), "bias": jnp.array([0.2])}
        norm_state = {
            "mean": jnp.array([0.5, -1.0]),
            "scale": jnp.array([2.0, 4.0]),
        }
        states = jax.vmap(lambda value: fake_state(value))(
            jnp.array([1.0, 2.0])
        )
        compiled = prepare_compiled_estimator_core(
            params,
            fake_actor,
            env,
            normalizer=object(),
            normalizer_state=norm_state,
            initial_states=states,
            sigma=0.1,
            gamma=0.5,
            pathwise_clip_norm=1.0,
        )
        first_noise = jnp.zeros((2, 3, 1))
        second_noise = jnp.ones((2, 3, 1))

        first = compiled(first_noise)
        cache_after_first = compiled._cache_size()
        second = compiled(second_noise)

        self.assertEqual(first.losses.shape, (2,))
        self.assertEqual(second.losses.shape, (2,))
        self.assertEqual(cache_after_first, 1)
        self.assertEqual(compiled._cache_size(), 1)
        receipts = build_and_validate_estimator_receipts(
            checkpoint_sha256="a" * 64,
            hparams=dict(E064_FROZEN_HPARAMS),
            actor_params=params,
            actor_apply=fake_actor,
            normalizer_state=norm_state,
            initial_state=states,
            action_noise=first_noise,
            result=first,
            gamma=0.5,
            sigma=0.1,
            pathwise_clip_norm=1.0,
        )
        self.assertEqual(receipts[0], receipts[1])

    def test_preparation_snapshots_mutable_checkpoint_inputs(self):
        actor = {"p": jnp.array([1.0])}
        env_state = {"obs": jnp.array([[2.0]])}
        norm_state = SimpleNamespace(
            mean=jnp.array([3.0]), var=jnp.array([4.0]), count=jnp.array(5.0)
        )
        state = SimpleNamespace(
            actor_params=actor, normalizer=norm_state, env_state=env_state
        )
        contract = SimpleNamespace(
            sigma=0.1,
            gamma=0.99,
            pathwise_clip_norm=1.0,
            reference_sha256=E064_REFERENCE_SHA256,
            hparams_sha256=E064_HPARAMS_SHA256,
            actor_parameters_sha256=stable_pytree_sha256(actor),
            normalizer_sha256=stable_pytree_sha256(
                g1_runtime._normalizer_identity_tree(norm_state)
            ),
            initial_state_sha256=stable_pytree_sha256(env_state),
        )
        captured = {}

        def build_core(actor_params, _actor_apply, _env, **kwargs):
            captured["actor"] = actor_params
            captured["normalizer"] = kwargs["normalizer_state"]
            captured["state"] = kwargs["initial_states"]
            return mock.Mock()

        with (
            mock.patch.object(
                g1_runtime,
                "validate_e064_checkpoint_contract",
                return_value=contract,
            ),
            mock.patch.object(g1_runtime, "validate_e064_live_semantics"),
            mock.patch.object(
                g1_runtime,
                "prepare_compiled_estimator_core",
                side_effect=build_core,
            ),
        ):
            prepare_e064_estimator_engine(
                state,
                dict(E064_FROZEN_HPARAMS),
                checkpoint_path=Path("checkpoint.pkl"),
                env=SmoothFakeEnv(),
                normalizer=object(),
            )

        actor["p"] = jnp.array([9.0])
        env_state["obs"] = jnp.array([[9.0]])
        norm_state.mean = jnp.array([9.0])
        np.testing.assert_array_equal(captured["actor"]["p"], [1.0])
        np.testing.assert_array_equal(captured["state"]["obs"], [[2.0]])
        np.testing.assert_array_equal(captured["normalizer"].mean, [3.0])

    def test_candidate_rollout_core_reuses_frozen_state_across_noise_shards(self):
        env = SmoothFakeEnv()
        params = {"gain": jnp.array(0.4), "bias": jnp.array([0.2])}
        norm_state = {
            "mean": jnp.array([0.5, -1.0]),
            "scale": jnp.array([2.0, 4.0]),
        }
        states = jax.vmap(lambda value: fake_state(value))(
            jnp.array([1.0, 2.0])
        )
        compiled = prepare_compiled_rollout_core(
            params,
            fake_actor,
            env,
            normalizer=object(),
            normalizer_state=norm_state,
            initial_states=states,
            sigma=0.1,
        )

        first, _ = compiled(jnp.zeros((2, 3, 1)))
        second, _ = compiled(jnp.ones((2, 3, 1)))

        np.testing.assert_array_equal(
            first.raw_observations[:, 0], states.obs
        )
        np.testing.assert_array_equal(
            second.raw_observations[:, 0], states.obs
        )
        self.assertFalse(np.array_equal(first.actions, second.actions))
        self.assertEqual(compiled._cache_size(), 1)


if __name__ == "__main__":
    unittest.main()
