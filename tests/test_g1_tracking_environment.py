import math
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np


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


class G1TrackingEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        cls.env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

    def test_environment_dimensions_and_timebase_match_rmr_port(self):
        env = self.env
        self.assertEqual(env.action_dim, 29)
        self.assertEqual(env.actor_frame_obs_dim, 154)
        self.assertEqual(env.actor_obs_dim, 154)
        self.assertEqual(env.critic_obs_dim, 286)
        self.assertEqual(env.n_frames, 5)
        self.assertAlmostEqual(env.dt, 0.01)
        state = env.reset(jax.random.PRNGKey(2), jnp.array(0.0))
        self.assertEqual(state.obs.shape, (154,))
        self.assertEqual(
            state.info["bootstrap_critic_obs"].shape, (286,)
        )

    def test_canonical_actor_noise_leaves_reference_and_actions_clean(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_observation_noise=True,
        )

        mask = np.asarray(env.actor_noise_mask)
        self.assertEqual(mask.shape, (154,))
        np.testing.assert_array_equal(mask[:58], 0.0)
        np.testing.assert_array_equal(mask[58:64], 0.05)
        np.testing.assert_array_equal(mask[64:67], 0.2)
        np.testing.assert_array_equal(mask[67:96], 0.01)
        np.testing.assert_array_equal(mask[96:125], 0.01)
        np.testing.assert_array_equal(mask[125:154], 0.0)

    def test_canonical_actor_noise_is_bounded_and_tiled_over_history(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=10,
            actor_observation_noise=True,
        )
        obs = jnp.zeros(env.actor_obs_dim)

        noisy = env._apply_obs_noise(obs, jax.random.PRNGKey(19))
        tiled_mask = jnp.tile(env.actor_noise_mask, env.actor_history_len)

        self.assertEqual(env.actor_obs_dim, 10 * env.actor_frame_obs_dim)
        self.assertTrue(bool(jnp.all(jnp.abs(noisy) <= tiled_mask)))
        self.assertGreater(float(jnp.linalg.norm(noisy)), 0.0)
        np.testing.assert_array_equal(noisy[tiled_mask == 0.0], 0.0)

    def test_actor_observation_noise_is_opt_in(self):
        obs = jnp.arange(self.env.actor_obs_dim, dtype=jnp.float64)

        unchanged = self.env._apply_obs_noise(
            obs, jax.random.PRNGKey(20)
        )

        np.testing.assert_array_equal(unchanged, obs)

    def test_fixed_mass_scale_changes_only_non_world_mass_and_inertia(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        shifted = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            mass_range=(1.15, 1.15),
        )

        self.assertEqual(self.env.body_mass_scale, 1.0)
        self.assertEqual(shifted.body_mass_scale, 1.15)
        np.testing.assert_array_equal(
            shifted.mj_model.body_mass[0],
            self.env.mj_model.body_mass[0],
        )
        np.testing.assert_array_equal(
            shifted.mj_model.body_inertia[0],
            self.env.mj_model.body_inertia[0],
        )
        np.testing.assert_allclose(
            shifted.mj_model.body_mass[1:],
            self.env.mj_model.body_mass[1:] * 1.15,
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            shifted.mj_model.body_inertia[1:],
            self.env.mj_model.body_inertia[1:] * 1.15,
            rtol=1e-12,
            atol=0.0,
        )

    def test_mass_scale_rejects_invalid_or_randomized_ranges(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        invalid_ranges = (
            (0.9, 1.1),
            (0.0, 0.0),
            (-1.0, -1.0),
            (float("nan"), float("nan")),
            (1.0,),
        )
        for mass_range in invalid_ranges:
            with self.subTest(mass_range=mass_range):
                with self.assertRaisesRegex(ValueError, "mass_range"):
                    G1TrackingEnv(mass_range=mass_range)

    def test_fixed_effort_limit_scale_changes_only_torque_authority(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        shifted = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            effort_limit_scale=0.7,
        )

        self.assertEqual(self.env.effort_limit_scale, 1.0)
        self.assertEqual(shifted.effort_limit_scale, 0.7)
        np.testing.assert_array_equal(
            self.env.effort_limit,
            self.env.controller.effort_limit,
        )
        np.testing.assert_allclose(
            shifted.effort_limit,
            shifted.controller.effort_limit * 0.7,
            rtol=1e-7,
            atol=0.0,
        )

    def test_effort_limit_scale_rejects_nonpositive_or_nonfinite_values(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "effort_limit_scale",
                ):
                    G1TrackingEnv(effort_limit_scale=scale)

    def test_termination_margin_weight_is_default_off_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        shifted = G1TrackingEnv(termination_margin_weight=0.5)

        self.assertEqual(self.env.termination_margin_weight, 0.0)
        self.assertEqual(shifted.termination_margin_weight, 0.5)
        for weight in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(
                    ValueError,
                    "termination_margin_weight",
                ):
                    G1TrackingEnv(termination_margin_weight=weight)

    def test_open_diffloco_factory_selects_tracking_environment(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv
        from src.envs.go2.environment import get_go2_env_class

        self.assertIs(get_go2_env_class("g1_tracking"), G1TrackingEnv)

    def test_reference_state_initialization_is_exact(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(7), jnp.array(0.0))
        phase = int(state.info["phase"])

        np.testing.assert_allclose(
            state.data.qpos, env.reference.qpos[phase], atol=1e-7
        )
        np.testing.assert_allclose(
            state.data.qvel, env.reference.qvel[phase], atol=1e-7
        )
        reward, components = env._tracking_reward(state.data, state.info)
        self.assertAlmostEqual(float(reward), 5.0, places=5)
        for value in components.values():
            self.assertAlmostEqual(float(value), 1.0, places=5)

    def test_reference_residual_zero_is_phase_target(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_joint_order="source",
            reference_residual_control=True,
            reference_residual_scale=0.5,
        )
        state = env.reset_at_phase(
            jax.random.PRNGKey(29),
            jnp.array(0.0),
            jnp.array(17),
        )

        target = env.position_target(state, jnp.zeros(env.action_dim))

        np.testing.assert_allclose(
            target,
            env.qpos_reference[17, 7:],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(env.squash_actor_actions)

    def test_reference_residual_uses_source_order_and_half_scale(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_joint_order="source",
            reference_residual_control=True,
            reference_residual_scale=0.5,
        )
        state = env.reset_at_phase(
            jax.random.PRNGKey(31),
            jnp.array(0.0),
            jnp.array(17),
        )
        residual = jnp.linspace(-0.8, 0.8, env.action_dim)
        residual_model_order = residual[env.actor_to_model_permutation]

        target = env.position_target(state, residual)
        expected = (
            env.qpos_reference[17, 7:]
            + 0.5 * residual_model_order * env.action_scales
        )

        np.testing.assert_allclose(target, expected, rtol=0.0, atol=1e-12)

    def test_reference_residual_options_are_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        for scale in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "reference_residual_scale",
                ):
                    G1TrackingEnv(
                        reference_residual_control=True,
                        reference_residual_scale=scale,
                    )
        with self.assertRaisesRegex(
            ValueError,
            "reference_residual_control",
        ):
            G1TrackingEnv(reference_residual_control=1)

    def test_reference_reset_noise_is_default_off_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        shifted = G1TrackingEnv(reference_reset_noise_scale=1.0)

        self.assertEqual(self.env.reference_reset_noise_scale, 0.0)
        self.assertEqual(shifted.reference_reset_noise_scale, 1.0)
        for scale in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "reference_reset_noise_scale",
                ):
                    G1TrackingEnv(reference_reset_noise_scale=scale)

    def test_reference_reset_noise_perturbs_only_upstream_rmr_envelope(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(reference_reset_noise_scale=1.0)
        state = env.reset(jax.random.PRNGKey(19), jnp.array(0.0))
        phase = int(state.info["phase"])
        qpos_delta = np.asarray(state.data.qpos - env.qpos_reference[phase])
        qvel_delta = np.asarray(state.data.qvel - env.qvel_reference[phase])

        self.assertGreater(np.max(np.abs(qpos_delta)), 0.0)
        self.assertGreater(np.max(np.abs(qvel_delta)), 0.0)
        np.testing.assert_array_less(
            np.abs(qpos_delta[:3]), [0.020001, 0.020001, 0.005001]
        )
        np.testing.assert_array_less(np.abs(qpos_delta[7:]), 0.050001)
        np.testing.assert_array_less(
            np.abs(qvel_delta[:3]), [0.250001, 0.250001, 0.100001]
        )
        np.testing.assert_array_less(
            np.abs(qvel_delta[3:6]), [0.260001, 0.260001, 0.390001]
        )
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(state.data.qpos[3:7])),
            1.0,
            atol=1e-6,
        )
        self.assertTrue(
            np.all(
                np.asarray(state.data.qpos[7:])
                >= np.asarray(env.soft_joint_lower)
            )
        )
        self.assertTrue(
            np.all(
                np.asarray(state.data.qpos[7:])
                <= np.asarray(env.soft_joint_upper)
            )
        )

    def test_adaptive_resets_bias_phases_without_invalidating_reset_state(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv
        from src.envs.g1_tracking.training_distribution import (
            init_phase_sampler,
        )

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=2,
            adaptive_phase_sampling=True,
            adaptive_phase_uniform_ratio=0.5,
            reference_reset_noise_scale=1.0,
            domain_randomization=True,
            friction_range=(0.8, 1.2),
            mass_range=(0.9, 1.1),
            kp_range=(30.0, 40.0),
            kd_range=(0.4, 0.6),
            com_offset_range=(0.01, 0.01, 0.01),
        )
        failed_count = init_phase_sampler(env.reference_length).failed_count
        target_bin = failed_count.shape[0] - 1
        failed_count = failed_count.at[target_bin].set(100.0)

        adaptive_target_count = 0
        uniform_target_count = 0
        state = None
        for seed in range(6):
            key = jax.random.PRNGKey(seed)
            adaptive_state = env.reset(
                key,
                jnp.array(1.0),
                phase_sampler_failed_count=failed_count,
            )
            uniform_state = self.env.reset(key, jnp.array(0.0))
            if state is None:
                state = adaptive_state
            adaptive_target_count += (
                int(adaptive_state.info["phase"])
                * failed_count.shape[0]
                // env.reference_length
                == target_bin
            )
            uniform_target_count += (
                int(uniform_state.info["phase"])
                * failed_count.shape[0]
                // env.reference_length
                == target_bin
            )

        self.assertGreater(adaptive_target_count, uniform_target_count)
        phase = int(state.info["phase"])
        qpos_delta = np.asarray(state.data.qpos - env.qpos_reference[phase])
        qvel_delta = np.asarray(state.data.qvel - env.qvel_reference[phase])
        self.assertGreater(np.max(np.abs(qpos_delta)), 0.0)
        self.assertGreater(np.max(np.abs(qvel_delta)), 0.0)
        self.assertTrue(np.isfinite(np.asarray(state.data.qpos)).all())
        self.assertTrue(np.isfinite(np.asarray(state.data.qvel)).all())
        np.testing.assert_array_less(
            np.abs(qpos_delta[:3]), [0.020001, 0.020001, 0.005001]
        )
        np.testing.assert_array_less(np.abs(qpos_delta[7:]), 0.050001)
        np.testing.assert_array_less(
            np.abs(qvel_delta[:3]), [0.250001, 0.250001, 0.100001]
        )
        np.testing.assert_array_less(
            np.abs(qvel_delta[3:6]), [0.260001, 0.260001, 0.390001]
        )
        self.assertEqual(state.info["actor_obs_history"].shape, (2, 154))
        for name in (
            "friction_scale",
            "mass_scale",
            "kp_scale",
            "kd_scale",
            "com_offset",
        ):
            self.assertTrue(np.isfinite(np.asarray(state.info[name])).all())
        self.assertGreaterEqual(float(state.info["friction_scale"]), 0.8)
        self.assertLessEqual(float(state.info["friction_scale"]), 1.2)
        self.assertGreaterEqual(float(state.info["mass_scale"]), 0.9)
        self.assertLessEqual(float(state.info["mass_scale"]), 1.1)
        np.testing.assert_array_equal(
            state.info["phase_sampler_failed_count"], failed_count
        )
        expected_rng = jax.random.split(jax.random.PRNGKey(0), 6)[0]
        np.testing.assert_array_equal(state.info["rng"], expected_rng)

    def test_adaptive_sampler_does_not_change_exact_reset_at_phase(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            adaptive_phase_sampling=True,
        )

        phase = env.reference_length // 2
        state = env.reset_at_phase(
            jax.random.PRNGKey(37), jnp.array(0.0), jnp.array(phase)
        )

        self.assertEqual(int(state.info["phase"]), phase)
        np.testing.assert_allclose(
            state.data.qpos,
            env.qpos_reference[phase],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            state.data.qvel, env.qvel_reference[phase]
        )
        np.testing.assert_array_equal(
            state.info["phase_sampler_failed_count"],
            np.zeros_like(state.info["phase_sampler_failed_count"]),
        )

    def test_exact_reset_path_preserves_rng_and_state_when_noise_is_zero(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(reference_reset_noise_scale=0.0)
        key = jax.random.PRNGKey(23)
        expected_rng, phase_key = jax.random.split(key)
        expected_phase = jax.random.randint(
            phase_key,
            (),
            minval=0,
            maxval=env.reference_length - 2,
            dtype=jnp.int32,
        )

        state = env.reset(key, jnp.array(0.0))

        np.testing.assert_array_equal(state.info["rng"], expected_rng)
        self.assertEqual(int(state.info["phase"]), int(expected_phase))
        self.assertNotIn("phase_sampler_failed_count", state.info)
        np.testing.assert_allclose(
            state.data.qpos,
            env.qpos_reference[expected_phase],
            atol=1e-7,
        )

    def test_carried_reset_bank_samples_actual_states_and_phases(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "carried_reset_bank.npz"
            phases = np.array([21, 37], dtype=np.int32)
            qpos = np.asarray(self.env.qpos_reference[phases]).copy()
            qvel = np.asarray(self.env.qvel_reference[phases]).copy()
            qpos[:, 2] -= np.array([0.03, 0.08])
            qvel[:, 2] -= np.array([0.1, 0.2])
            np.savez_compressed(
                bank_path,
                phase=phases,
                qpos=qpos,
                qvel=qvel,
            )
            env = G1TrackingEnv(
                xml_path=str(MODEL),
                reference_path=str(REFERENCE),
                controller_path=str(CONTROLLER),
                actor_history_len=1,
                carried_reset_bank_path=str(bank_path),
                carried_reset_probability=1.0,
                carried_reset_bank_start=1,
            )

            state = env.reset(jax.random.PRNGKey(29), jnp.array(0.0))

        self.assertEqual(int(state.info["phase"]), 37)
        np.testing.assert_allclose(state.data.qpos, qpos[1], atol=1e-7)
        np.testing.assert_allclose(state.data.qvel, qvel[1], atol=1e-7)
        self.assertEqual(env.carried_reset_bank_size, 1)

    def test_carried_reset_bank_is_default_off_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        self.assertIsNone(self.env.carried_reset_bank_path)
        self.assertEqual(self.env.carried_reset_probability, 0.0)
        self.assertEqual(self.env.carried_reset_bank_start, 0)
        for probability in (-0.1, 1.1, float("nan"), True):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError, "carried_reset_probability"
                ):
                    G1TrackingEnv(
                        carried_reset_bank_path="/tmp/missing.npz",
                        carried_reset_probability=probability,
                    )
        with self.assertRaisesRegex(ValueError, "carried_reset_bank_path"):
            G1TrackingEnv(carried_reset_probability=0.5)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            G1TrackingEnv(
                carried_reset_bank_path="/tmp/missing.npz",
                carried_reset_probability=0.5,
                reference_reset_noise_scale=1.0,
            )

    def test_fixed_phase_reset_supports_paired_evaluation(self):
        env = self.env
        state = env.reset_at_phase(
            jax.random.PRNGKey(17), jnp.array(0.0), jnp.array(37)
        )

        self.assertEqual(int(state.info["phase"]), 37)
        np.testing.assert_allclose(
            state.data.qpos, env.reference.qpos[37], atol=1e-7
        )

    def test_actor_observation_begins_with_matching_reference_command(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(11), jnp.array(0.0))
        phase = int(state.info["phase"])
        frame = state.obs.reshape(env.actor_history_len, -1)[-1]

        expected_command = np.concatenate(
            [
                env.reference.qpos[phase, 7:],
                env.reference.qvel[phase, 6:],
            ]
        )
        np.testing.assert_allclose(frame[:58], expected_command, atol=1e-7)
        np.testing.assert_allclose(
            frame[58:64],
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            atol=1e-5,
        )

    def test_actor_observation_appends_multiscale_future_reference(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=10,
            actor_reference_lookahead_steps=(4, 8, 12),
        )
        phase = 37
        state = env.reset_at_phase(
            jax.random.PRNGKey(71),
            jnp.array(0.0),
            jnp.array(phase),
        )
        frames = np.asarray(state.obs).reshape(10, 328)
        actor_order = np.asarray(env.model_to_actor_permutation)
        expected = []
        for offset in (4, 8, 12):
            index = min(
                phase + offset * env.reference_stride,
                env.reference_length - 1,
            )
            expected.extend(
                np.asarray(env.qpos_reference[index, 7:])[actor_order]
            )
            expected.extend(
                np.asarray(env.qvel_reference[index, 6:])[actor_order]
            )

        self.assertEqual(env.actor_reference_lookahead_steps, (4, 8, 12))
        self.assertEqual(env.actor_frame_obs_dim, 328)
        self.assertEqual(env.actor_obs_dim, 3280)
        self.assertEqual(env.critic_obs_dim, 286)
        np.testing.assert_allclose(
            frames[-1, :154],
            np.asarray(self.env._get_actor_obs(state.data, state.info)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            frames[-1, 154:], expected, rtol=0.0, atol=1e-12
        )
        np.testing.assert_array_equal(frames, np.repeat(frames[-1:], 10, axis=0))
        self.assertEqual(state.info["bootstrap_obs"].shape, (3280,))

    def test_actor_observation_supports_delta_future_reference(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_reference_lookahead_steps=(4, 8, 12),
            actor_reference_preview_mode="delta",
        )
        phase = 37
        actual = np.asarray(
            env._future_reference_command(jnp.array(phase))
        ).reshape(3, 58)
        actor_order = np.asarray(env.model_to_actor_permutation)
        current = np.concatenate(
            (
                np.asarray(env.qpos_reference[phase, 7:])[actor_order],
                np.asarray(env.qvel_reference[phase, 6:])[actor_order],
            )
        )
        expected = []
        for offset in (4, 8, 12):
            future_phase = min(
                phase + offset * env.reference_stride,
                env.reference_length - 1,
            )
            expected.append(
                np.concatenate(
                    (
                        np.asarray(
                            env.qpos_reference[future_phase, 7:]
                        )[actor_order],
                        np.asarray(
                            env.qvel_reference[future_phase, 6:]
                        )[actor_order],
                    )
                )
                - current
            )

        self.assertEqual(env.actor_reference_preview_mode, "delta")
        np.testing.assert_allclose(
            actual, np.stack(expected), rtol=0.0, atol=1e-12
        )

    def test_future_reference_preview_mode_is_explicit_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        self.assertEqual(self.env.actor_reference_preview_mode, "absolute")
        for mode in ("relative", True, None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "preview mode"):
                    G1TrackingEnv(
                        actor_reference_lookahead_steps=(4,),
                        actor_reference_preview_mode=mode,
                    )
        with self.assertRaisesRegex(ValueError, "requires lookahead"):
            G1TrackingEnv(actor_reference_preview_mode="delta")

    def test_future_reference_clamps_and_validates_offsets(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_reference_lookahead_steps=(4, 8, 12),
        )
        phase = env.reference_length - 2
        state = env.reset_at_phase(
            jax.random.PRNGKey(73),
            jnp.array(0.0),
            jnp.array(phase),
        )
        suffix = np.asarray(state.obs)[154:].reshape(3, 58)
        final = env.reference_length - 1
        expected = np.concatenate(
            (
                np.asarray(env.qpos_reference[final, 7:]),
                np.asarray(env.qvel_reference[final, 6:]),
            )
        )
        np.testing.assert_allclose(
            suffix, np.repeat(expected[None, :], 3, axis=0), atol=1e-12
        )

        for offsets in ((0,), (-1,), (8, 4), (4, 4), (True,), (4.0,)):
            with self.subTest(offsets=offsets):
                with self.assertRaisesRegex(
                    ValueError, "lookahead steps"
                ):
                    G1TrackingEnv(
                        actor_reference_lookahead_steps=offsets
                    )

    def test_global_anchor_translation_does_not_corrupt_relative_body_pose(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(13), jnp.array(0.0))
        translated_qpos = state.data.qpos.at[0].add(0.3)
        translated = mjx.forward(
            env.mjx_model, state.data.replace(qpos=translated_qpos)
        )

        _, components = env._tracking_reward(translated, state.info)
        self.assertAlmostEqual(
            float(components["anchor_position"]), math.exp(-1.0), places=5
        )
        self.assertAlmostEqual(
            float(components["body_position"]), 1.0, places=5
        )

    def test_one_step_advances_exactly_one_reference_frame(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(3), jnp.array(0.0))
        next_state = env.step(state, jnp.zeros(env.action_dim))

        expected_phase = (int(state.info["phase"]) + 1) % env.reference_length
        self.assertEqual(int(next_state.info["phase"]), expected_phase)
        self.assertTrue(np.isfinite(np.asarray(next_state.data.qpos)).all())
        self.assertTrue(np.isfinite(float(next_state.reward)))

    def test_termination_thresholds_match_upstream_rmr(self):
        from src.envs.g1_tracking.environment import _quat_mul

        env = self.env
        state = env.reset_at_phase(
            jax.random.PRNGKey(19), jnp.array(0.0), jnp.array(0)
        )
        body_pos, body_quat, _, _ = env._body_state(state.data)
        distal_slot = env.distal_body_slots[0]

        within_distal_limit = body_pos.at[distal_slot, 2].add(0.3)
        _, terminal = env._termination(
            state.data, state.info, within_distal_limit, body_quat
        )
        self.assertEqual(float(terminal), 0.0)

        beyond_distal_limit = body_pos.at[distal_slot, 2].add(0.41)
        _, terminal = env._termination(
            state.data, state.info, beyond_distal_limit, body_quat
        )
        self.assertEqual(float(terminal), 1.0)

        within_xy_limit = body_pos.at[0, 0].add(1.2)
        _, terminal = env._termination(
            state.data, state.info, within_xy_limit, body_quat
        )
        self.assertEqual(float(terminal), 0.0)

        beyond_xy_limit = body_pos.at[0, 0].add(1.31)
        _, terminal = env._termination(
            state.data, state.info, beyond_xy_limit, body_quat
        )
        self.assertEqual(float(terminal), 1.0)

        one_radian_roll = jnp.array(
            [jnp.cos(0.5), jnp.sin(0.5), 0.0, 0.0]
        )
        moderate_tilt = body_quat.at[0].set(
            _quat_mul(body_quat[0], one_radian_roll)
        )
        _, terminal = env._termination(
            state.data, state.info, body_pos, moderate_tilt
        )
        self.assertEqual(float(terminal), 0.0)

    def test_termination_errors_are_reusable_for_collocation_barriers(self):
        env = self.env
        state = env.reset_at_phase(
            jax.random.PRNGKey(29), jnp.array(0.0), jnp.array(0)
        )
        body_pos, body_quat, _, _ = env._body_state(state.data)
        shifted = body_pos.at[0, 2].add(0.2)

        errors = env.termination_errors(
            phase=state.info["phase"],
            body_pos=shifted,
            body_quat=body_quat,
        )

        self.assertEqual(
            set(errors),
            {
                "anchor_z_error",
                "anchor_xy_error",
                "gravity_z_error",
                "distal_z_error",
            },
        )
        self.assertAlmostEqual(float(errors["anchor_z_error"]), 0.2, places=6)
        self.assertAlmostEqual(float(errors["anchor_xy_error"]), 0.0, places=6)
        self.assertAlmostEqual(float(errors["gravity_z_error"]), 0.0, places=6)
        self.assertAlmostEqual(float(errors["distal_z_error"]), 0.0, places=6)

    def test_transition_metrics_reserve_all_pre_reset_termination_errors(self):
        expected_keys = {
            "termination_anchor_z_error",
            "termination_anchor_xy_error",
            "termination_gravity_z_error",
            "termination_distal_z_error",
        }

        self.assertTrue(expected_keys.issubset(self.env._init_metrics()))


class G1TrackingRMR50HzEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.environment import G1TrackingRMR50HzEnv

        cls.env = G1TrackingRMR50HzEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.named_50hz_reference = (
            Path(cls.temporary_directory.name) / "named_50hz.npz"
        )
        model_to_source = cls.env.controller.model_to_actor_permutation
        np.savez(
            cls.named_50hz_reference,
            fps=np.asarray([50], dtype=np.int32),
            joint_pos=cls.env.reference.qpos[:3, 7:][:, model_to_source],
            joint_vel=cls.env.reference.qvel[:3, 6:][:, model_to_source],
            body_pos_w=cls.env.reference.body_pos[:3, :1],
            body_quat_w=cls.env.reference.body_quat[:3, :1],
            body_lin_vel_w=cls.env.reference.body_lin_vel[:3, :1],
            body_ang_vel_w=cls.env.reference.body_ang_vel[:3, :1],
            joint_names=np.asarray(cls.env.controller.actor_joint_names),
            root_body_name=np.asarray("pelvis"),
            root_body_index=np.asarray(0, dtype=np.int32),
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_control_reference_and_reward_timebase_match_rmr(self):
        env = self.env
        self.assertEqual(env.n_frames, 10)
        self.assertEqual(env.reference_stride, 2)
        self.assertAlmostEqual(env.dt, 0.02)
        self.assertAlmostEqual(env.reward_scale, 0.02)

        state = env.reset_at_phase(
            jax.random.PRNGKey(23), jnp.array(0.0), jnp.array(0)
        )
        raw_reward, _ = env._tracking_reward(state.data, state.info)
        self.assertAlmostEqual(
            float(raw_reward * env.reward_scale), 0.1, places=6
        )

        next_state = env.step(state, jnp.zeros(env.action_dim))
        self.assertEqual(int(next_state.info["phase"]), 2)

    def test_environment_rejects_reference_timebase_mismatch(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzValidatedEnv,
        )

        with self.assertRaisesRegex(ValueError, "reference timebase"):
            G1TrackingRMR50HzValidatedEnv(
                xml_path=str(MODEL),
                reference_path=str(self.named_50hz_reference),
                controller_path=str(CONTROLLER),
                reference_stride=2,
            )

    def test_factory_selects_50hz_rmr_environment(self):
        from src.envs.g1_tracking.environment import G1TrackingRMR50HzEnv
        from src.envs.go2.environment import get_go2_env_class

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz"),
            G1TrackingRMR50HzEnv,
        )

    def test_unbounded_rmr_variant_preserves_demonstrated_action_support(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzUnboundedEnv,
        )
        from src.envs.go2.environment import get_go2_env_class

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_unbounded"),
            G1TrackingRMR50HzUnboundedEnv,
        )

        unbounded = G1TrackingRMR50HzUnboundedEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )
        action = jnp.arange(unbounded.action_dim, dtype=jnp.float64)
        source_names = unbounded.controller.actor_joint_names
        expected_model_action = np.array(
            [source_names.index(name) for name in unbounded.controller.joint_names]
        )

        np.testing.assert_allclose(
            unbounded._prepare_action(action), expected_model_action
        )
        np.testing.assert_allclose(
            self.env._prepare_action(action),
            np.concatenate(([0.0], np.ones(self.env.action_dim - 1))),
        )
        self.assertFalse(unbounded.squash_actor_actions)

        state = unbounded.reset_at_phase(
            jax.random.PRNGKey(29), jnp.array(0.0), jnp.array(0)
        )
        model_to_actor = unbounded.controller.model_to_actor_permutation
        np.testing.assert_allclose(
            state.obs[:29],
            unbounded.reference.qpos[0, 7:][model_to_actor],
        )

    def test_source_step_variants_keep_our_model_and_match_rmr_timebase(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzSourceStepEnv,
            G1TrackingRMR50HzSourceStepRobustEnv,
            G1TrackingRMR50HzValidatedEnv,
        )
        from src.envs.go2.environment import get_go2_env_class

        source_step = G1TrackingRMR50HzSourceStepEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )
        robust = G1TrackingRMR50HzSourceStepRobustEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )
        validated = G1TrackingRMR50HzValidatedEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_source_step"),
            G1TrackingRMR50HzSourceStepEnv,
        )
        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_source_step_robust"),
            G1TrackingRMR50HzSourceStepRobustEnv,
        )
        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_validated"),
            G1TrackingRMR50HzValidatedEnv,
        )
        for env in (source_step, robust, validated):
            self.assertEqual(env.mj_model.ngeom, self.env.mj_model.ngeom)
            self.assertEqual(env.n_frames, 4)
            self.assertAlmostEqual(env.mj_model.opt.timestep, 0.005)
            self.assertAlmostEqual(env.dt, 0.02)
            self.assertFalse(env.squash_actor_actions)
        self.assertEqual(source_step.mj_model.opt.iterations, 1)
        self.assertEqual(source_step.mj_model.opt.ls_iterations, 5)
        self.assertEqual(robust.mj_model.opt.iterations, 10)
        self.assertEqual(robust.mj_model.opt.ls_iterations, 20)
        self.assertEqual(validated.mj_model.opt.iterations, 4)
        self.assertEqual(validated.mj_model.opt.ls_iterations, 5)


if __name__ == "__main__":
    unittest.main()
