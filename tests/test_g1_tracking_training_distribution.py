import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.g1_tracking.training_distribution import (
    ACTOR_NOISE_SLICES,
    PhaseSamplerState,
    corrupt_actor_observation,
    init_phase_sampler,
    phase_sampling_probabilities,
    reset_training_at_phase,
    sample_reset_perturbations,
    sample_training_phase,
    update_phase_sampler,
)


MODEL = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
REFERENCE = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/w02_rmrspec_grounded.npz"
)
CONTROLLER = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)


class PhaseSamplerTest(unittest.TestCase):
    def test_failure_mixture_retains_literal_uniform_floor(self):
        state = PhaseSamplerState(failed_count=jnp.array([0.0, 4.0]))
        probabilities = np.asarray(
            phase_sampling_probabilities(state, uniform_ratio=0.5)
        )
        np.testing.assert_allclose(probabilities, np.array([0.25, 0.75]))

    def test_zero_failure_mixture_is_exactly_uniform(self):
        state = PhaseSamplerState(failed_count=jnp.zeros(5))
        probabilities = np.asarray(
            phase_sampling_probabilities(state, uniform_ratio=0.5)
        )
        np.testing.assert_allclose(probabilities, np.full(5, 0.2))

    def test_starts_uniform_and_moves_toward_failed_bin(self):
        state = init_phase_sampler(reference_length=212)
        before = np.asarray(phase_sampling_probabilities(state))
        np.testing.assert_allclose(
            before, np.full_like(before, 1.0 / len(before))
        )

        updated = update_phase_sampler(
            state,
            phases=jnp.array([10, 10, 10, 170], dtype=jnp.int32),
            terminals=jnp.array([1, 1, 1, 0], dtype=jnp.float32),
            reference_length=212,
        )
        after = np.asarray(phase_sampling_probabilities(updated))
        self.assertGreater(after[0], before[0])
        self.assertAlmostEqual(float(after.sum()), 1.0, places=6)

    def test_update_accumulates_only_terminal_samples(self):
        state = init_phase_sampler(reference_length=212)
        updated = update_phase_sampler(
            state,
            phases=jnp.array([1, 60, 170], dtype=jnp.int32),
            terminals=jnp.array([0, 1, 1], dtype=jnp.float32),
            reference_length=212,
            alpha=0.25,
        )
        np.testing.assert_allclose(
            updated.failed_count,
            np.array([0.0, 0.25, 0.0, 0.0, 0.25]),
        )

    def test_sampled_phases_stay_in_selected_bins_and_reference_range(self):
        state = PhaseSamplerState(
            failed_count=jnp.array([0.0, 0.0, 10.0, 0.0, 0.0])
        )
        keys = jax.random.split(jax.random.PRNGKey(3), 128)
        phases = jax.vmap(
            lambda key: sample_training_phase(
                key, state, reference_length=212
            )
        )(keys)
        phase_array = np.asarray(phases)
        self.assertTrue(np.all(phase_array >= 0))
        self.assertTrue(np.all(phase_array < 211))
        self.assertGreater(np.mean((phase_array >= 84) & (phase_array < 127)), 0.8)

    def test_invalid_sampler_arguments_fail_at_python_boundary(self):
        for reference_length in (0, -1, True):
            with self.subTest(reference_length=reference_length):
                with self.assertRaises(ValueError):
                    init_phase_sampler(reference_length)
        with self.assertRaises(ValueError):
            phase_sampling_probabilities(init_phase_sampler(212), -0.1)


class TrainingNoiseTest(unittest.TestCase):
    def test_unregistered_actor_fields_are_byte_exact(self):
        obs = jnp.arange(154, dtype=jnp.float64)
        noisy = np.asarray(
            corrupt_actor_observation(jax.random.PRNGKey(4), obs)
        )
        registered = np.zeros(154, dtype=bool)
        for region, _ in ACTOR_NOISE_SLICES:
            registered[region] = True
        np.testing.assert_array_equal(
            noisy[~registered], np.asarray(obs)[~registered]
        )

    def test_actor_corruption_respects_each_source_range(self):
        obs = jnp.zeros(154, dtype=jnp.float64)
        noisy = np.asarray(
            corrupt_actor_observation(jax.random.PRNGKey(5), obs)
        )
        for region, bound in ACTOR_NOISE_SLICES:
            values = noisy[region]
            self.assertTrue(np.all(np.abs(values) <= bound))
            self.assertTrue(np.any(values != 0.0))

    def test_corruption_supports_leading_batch_dimensions(self):
        obs = jnp.zeros((3, 154), dtype=jnp.float64)
        noisy = corrupt_actor_observation(jax.random.PRNGKey(6), obs)
        self.assertEqual(noisy.shape, obs.shape)
        self.assertTrue(np.isfinite(np.asarray(noisy)).all())

    def test_reset_perturbation_samples_respect_recorded_ranges(self):
        perturbation = sample_reset_perturbations(
            jax.random.PRNGKey(7), action_dim=29
        )
        position_bounds = np.array([0.02, 0.02, 0.005])
        rotation_bounds = np.array([0.1, 0.1, 0.1])
        linear_bounds = np.array([0.25, 0.25, 0.1])
        angular_bounds = np.array([0.26, 0.26, 0.39])
        for value, bound in (
            (perturbation.root_position, position_bounds),
            (perturbation.root_euler_xyz, rotation_bounds),
            (perturbation.root_linear_velocity, linear_bounds),
            (perturbation.root_angular_velocity, angular_bounds),
        ):
            self.assertTrue(np.all(np.abs(np.asarray(value)) <= bound))
        self.assertTrue(
            np.all(np.abs(np.asarray(perturbation.joint_position)) <= 0.05)
        )


class TrainingResetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzValidatedEnv,
        )

        cls.env = G1TrackingRMR50HzValidatedEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

    def test_perturbed_reset_preserves_phase_and_respects_state_ranges(self):
        phase = 37
        state = reset_training_at_phase(
            self.env,
            jax.random.PRNGKey(11),
            jnp.array(0.0),
            jnp.array(phase),
        )
        qpos = np.asarray(state.data.qpos)
        qvel = np.asarray(state.data.qvel)
        ref_qpos = np.asarray(self.env.reference.qpos[phase])
        ref_qvel = np.asarray(self.env.reference.qvel[phase])

        self.assertEqual(int(state.info["phase"]), phase)
        self.assertTrue(np.isfinite(qpos).all())
        self.assertTrue(np.isfinite(qvel).all())
        np.testing.assert_array_less(
            np.abs(qpos[:3] - ref_qpos[:3]),
            np.array([0.02, 0.02, 0.005]) + 1e-10,
        )
        np.testing.assert_allclose(np.linalg.norm(qpos[3:7]), 1.0, atol=1e-7)
        np.testing.assert_array_less(
            np.abs(qvel[:3] - ref_qvel[:3]),
            np.array([0.25, 0.25, 0.1]) + 1e-10,
        )
        np.testing.assert_array_less(
            np.abs(qvel[3:6] - ref_qvel[3:6]),
            np.array([0.26, 0.26, 0.39]) + 1e-10,
        )
        np.testing.assert_allclose(qvel[6:], ref_qvel[6:], atol=1e-7)
        self.assertTrue(
            np.all(
                qpos[7:] >= np.asarray(self.env.soft_joint_lower) - 1e-10
            )
        )
        self.assertTrue(
            np.all(
                qpos[7:] <= np.asarray(self.env.soft_joint_upper) + 1e-10
            )
        )
        self.assertTrue(
            np.all(np.abs(qpos[7:] - ref_qpos[7:]) <= 0.05 + 1e-10)
        )
        self.assertEqual(state.obs.shape, (154,))
        self.assertEqual(state.info["bootstrap_critic_obs"].shape, (286,))

    def test_reset_is_deterministic_for_caller_key(self):
        args = (
            self.env,
            jax.random.PRNGKey(19),
            jnp.array(0.0),
            jnp.array(53),
        )
        first = reset_training_at_phase(*args)
        second = reset_training_at_phase(*args)
        np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
        np.testing.assert_array_equal(first.data.qvel, second.data.qvel)
        np.testing.assert_array_equal(first.obs, second.obs)


if __name__ == "__main__":
    unittest.main()
