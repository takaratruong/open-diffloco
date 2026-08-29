import copy
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
        self.assertNotIn("bootstrap_jave_obs", state.info)

    def test_jave_observation_reconstructs_the_transition_reward(self):
        env = self.env
        self.assertFalse(env.jave_enabled)
        env.jave_enabled = True
        self.addCleanup(setattr, env, "jave_enabled", False)
        state = env.reset_at_phase(
            jax.random.PRNGKey(211),
            jnp.array(0.0),
            jnp.array(10, dtype=jnp.int32),
        )
        action = jnp.linspace(-0.15, 0.15, env.action_dim)

        current_jave_obs = env._get_jave_obs(state.data, state.info)
        next_state = env.step(state, action)
        bootstrap_jave_obs = next_state.info["bootstrap_jave_obs"]
        reconstructed_reward = env.compute_reward_from_jave_obs(
            current_jave_obs,
            bootstrap_jave_obs,
            action,
        )

        self.assertEqual(env.jave_obs_dim, env.critic_obs_dim + 1)
        self.assertEqual(current_jave_obs.shape, (env.jave_obs_dim,))
        np.testing.assert_array_equal(
            np.asarray(current_jave_obs[: env.critic_obs_dim]),
            np.asarray(env._get_critic_obs(state.data, state.info)),
        )
        np.testing.assert_allclose(
            np.asarray(reconstructed_reward),
            np.asarray(next_state.reward),
            rtol=1e-10,
            atol=1e-10,
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

    def test_tracking_velocity_kernel_is_explicit_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        pseudo_huber = G1TrackingEnv(
            tracking_velocity_kernel="pseudo_huber"
        )

        self.assertEqual(self.env.tracking_velocity_kernel, "exponential")
        self.assertEqual(
            pseudo_huber.tracking_velocity_kernel, "pseudo_huber"
        )
        with self.assertRaisesRegex(ValueError, "tracking_velocity_kernel"):
            G1TrackingEnv(tracking_velocity_kernel="not-a-kernel")

    def test_anchor_position_kernel_is_explicit_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        treatment = G1TrackingEnv(
            tracking_anchor_position_kernel="dual_scale"
        )
        quadratic = G1TrackingEnv(
            tracking_anchor_position_kernel="quadratic"
        )

        self.assertEqual(
            self.env.tracking_anchor_position_kernel, "exponential"
        )
        self.assertEqual(
            treatment.tracking_anchor_position_kernel, "dual_scale"
        )
        self.assertEqual(
            quadratic.tracking_anchor_position_kernel, "quadratic"
        )
        with self.assertRaisesRegex(
            ValueError, "tracking_anchor_position_kernel"
        ):
            G1TrackingEnv(tracking_anchor_position_kernel="not-a-kernel")

    def test_torso_orientation_weight_is_default_off_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        shifted = G1TrackingEnv(tracking_torso_orientation_weight=1.0)

        self.assertEqual(self.env.tracking_torso_orientation_weight, 0.0)
        self.assertEqual(shifted.tracking_torso_orientation_weight, 1.0)
        for weight in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(
                    ValueError, "tracking_torso_orientation_weight"
                ):
                    G1TrackingEnv(tracking_torso_orientation_weight=weight)

    def test_root_velocity_weight_is_default_off_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        shifted = G1TrackingEnv(tracking_root_velocity_weight=1.0)

        self.assertEqual(self.env.tracking_root_velocity_weight, 0.0)
        self.assertEqual(shifted.tracking_root_velocity_weight, 1.0)
        for weight in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(
                    ValueError, "tracking_root_velocity_weight"
                ):
                    G1TrackingEnv(tracking_root_velocity_weight=weight)

    def test_environment_adds_only_explicit_root_velocity_term(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        legacy = G1TrackingEnv(tracking_root_velocity_weight=0.0)
        treatment = G1TrackingEnv(tracking_root_velocity_weight=1.0)
        phase = jnp.array(0)
        body_pos = legacy.body_pos_reference[phase]
        body_quat = legacy.body_quat_reference[phase]
        body_lin_vel = legacy.body_lin_vel_reference[phase].at[0, 0].add(1.0)
        body_ang_vel = legacy.body_ang_vel_reference[phase].at[0, 2].add(math.pi)

        legacy_reward, legacy_components = legacy._tracking_reward_from_body_state(
            {"phase": phase}, body_pos, body_quat, body_lin_vel, body_ang_vel
        )
        treatment_reward, treatment_components = (
            treatment._tracking_reward_from_body_state(
                {"phase": phase}, body_pos, body_quat, body_lin_vel, body_ang_vel
            )
        )

        expected = 2.0 - math.sqrt(3.0)
        self.assertNotIn("root_linear_velocity", legacy_components)
        self.assertNotIn("root_angular_velocity", legacy_components)
        self.assertAlmostEqual(
            float(treatment_components["root_linear_velocity"]), expected, places=6
        )
        self.assertAlmostEqual(
            float(treatment_components["root_angular_velocity"]), expected, places=6
        )
        self.assertAlmostEqual(
            float(treatment_reward - legacy_reward), expected, places=6
        )

    def test_environment_adds_only_the_direct_torso_orientation_term(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv, _quat_mul

        legacy = G1TrackingEnv(tracking_torso_orientation_weight=0.0)
        treatment = G1TrackingEnv(tracking_torso_orientation_weight=1.0)
        phase = jnp.array(0)
        body_pos = legacy.body_pos_reference[phase]
        body_quat = legacy.body_quat_reference[phase]
        body_lin_vel = legacy.body_lin_vel_reference[phase]
        body_ang_vel = legacy.body_ang_vel_reference[phase]
        pitched = body_quat.at[7].set(
            _quat_mul(
                jnp.array([math.cos(0.2), 0.0, math.sin(0.2), 0.0]),
                body_quat[7],
            )
        )

        legacy_reward, legacy_components = legacy._tracking_reward_from_body_state(
            {"phase": phase}, body_pos, pitched, body_lin_vel, body_ang_vel
        )
        treatment_reward, treatment_components = (
            treatment._tracking_reward_from_body_state(
                {"phase": phase}, body_pos, pitched, body_lin_vel, body_ang_vel
            )
        )

        expected = 2.0 - math.sqrt(3.0)
        self.assertNotIn("torso_orientation", legacy_components)
        self.assertAlmostEqual(
            float(treatment_components["torso_orientation"]), expected, places=5
        )
        self.assertAlmostEqual(
            float(treatment_reward - legacy_reward), expected, places=5
        )

    def test_environment_forwards_pseudo_huber_velocity_kernel(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(tracking_velocity_kernel="pseudo_huber")
        phase = jnp.array(0)
        target_body_pos = env.body_pos_reference[phase]
        target_body_quat = env.body_quat_reference[phase]
        actual_body_lin_vel = env.body_lin_vel_reference[phase].at[0, 0].add(1.0)

        _, components = env._tracking_reward_from_body_state(
            {"phase": phase},
            target_body_pos,
            target_body_quat,
            actual_body_lin_vel,
            env.body_ang_vel_reference[phase],
        )

        expected = 2.0 - math.sqrt(1.0 + 2.0 / len(env.body_ids))
        self.assertAlmostEqual(
            float(components["body_linear_velocity"]), expected, places=6
        )

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

    def test_reference_residual_can_use_unbounded_full_rmr_action_scale(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_joint_order="source",
            clip_actions=False,
            squash_actor_actions_override=False,
            reference_residual_control=True,
            reference_residual_scale=1.0,
        )
        state = env.reset_at_phase(
            jax.random.PRNGKey(32),
            jnp.array(0.0),
            jnp.array(17),
        )
        raw_action = jnp.linspace(-2.0, 2.0, env.action_dim)
        model_action = raw_action[env.actor_to_model_permutation]

        np.testing.assert_allclose(
            env._prepare_action(raw_action),
            model_action,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            env.position_target(state, raw_action),
            env.qpos_reference[17, 7:] + model_action * env.action_scales,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertFalse(env.squash_actor_actions)

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

    def test_root_reset_noise_treatment_is_default_off_and_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        explicit_default = G1TrackingEnv(
            reference_reset_noise_scale=1.0,
            reference_root_reset_noise_multiplier=1.0,
            reference_root_reset_noise_probability=0.0,
        )
        implicit_default = G1TrackingEnv(reference_reset_noise_scale=1.0)
        key = jax.random.PRNGKey(31)

        explicit_state = explicit_default.reset(key, jnp.array(0.0))
        implicit_state = implicit_default.reset(key, jnp.array(0.0))

        np.testing.assert_array_equal(
            explicit_state.data.qpos, implicit_state.data.qpos
        )
        np.testing.assert_array_equal(
            explicit_state.data.qvel, implicit_state.data.qvel
        )
        for kwargs in (
            {"reference_root_reset_noise_multiplier": 0.99},
            {"reference_root_reset_noise_multiplier": float("nan")},
            {"reference_root_reset_noise_multiplier": True},
            {"reference_root_reset_noise_probability": -0.1},
            {"reference_root_reset_noise_probability": 1.1},
            {"reference_root_reset_noise_probability": float("nan")},
            {"reference_root_reset_noise_probability": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "root reset noise"):
                    G1TrackingEnv(**kwargs)

    def test_root_reset_noise_multiplier_changes_only_root_envelope(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(reference_reset_noise_scale=1.0)
        phase = jnp.array(17, dtype=jnp.int32)
        pose_key, velocity_key, joint_key = jax.random.split(
            jax.random.PRNGKey(32), 3
        )
        baseline_qpos, baseline_qvel = env._noisy_reference_state(
            phase, pose_key, velocity_key, joint_key, root_multiplier=1.0
        )
        recovery_qpos, recovery_qvel = env._noisy_reference_state(
            phase, pose_key, velocity_key, joint_key, root_multiplier=2.0
        )
        reference_qpos = env.qpos_reference[phase]
        reference_qvel = env.qvel_reference[phase]

        np.testing.assert_array_equal(
            recovery_qpos[7:] - reference_qpos[7:],
            baseline_qpos[7:] - reference_qpos[7:],
        )
        np.testing.assert_array_equal(
            recovery_qvel[6:], baseline_qvel[6:]
        )
        np.testing.assert_allclose(
            recovery_qpos[:3] - reference_qpos[:3],
            2.0 * (baseline_qpos[:3] - reference_qpos[:3]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            recovery_qvel[:6] - reference_qvel[:6],
            2.0 * (baseline_qvel[:6] - reference_qvel[:6]),
            rtol=0.0,
            atol=1e-7,
        )

    def test_root_reset_noise_probability_one_uses_registered_bounds(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            reference_reset_noise_scale=1.0,
            reference_root_reset_noise_multiplier=2.0,
            reference_root_reset_noise_probability=1.0,
        )
        state = env.reset(jax.random.PRNGKey(33), jnp.array(0.0))
        phase = int(state.info["phase"])
        qpos_delta = np.asarray(state.data.qpos - env.qpos_reference[phase])
        qvel_delta = np.asarray(state.data.qvel - env.qvel_reference[phase])

        np.testing.assert_array_less(
            np.abs(qpos_delta[:3]), [0.040001, 0.040001, 0.010001]
        )
        np.testing.assert_array_less(np.abs(qpos_delta[7:]), 0.050001)
        np.testing.assert_array_less(
            np.abs(qvel_delta[:3]), [0.500001, 0.500001, 0.200001]
        )
        np.testing.assert_array_less(
            np.abs(qvel_delta[3:6]), [0.520001, 0.520001, 0.780001]
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

    def test_carried_bank_preserves_noisy_reference_fallback(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        phase = np.array([37], dtype=np.int32)
        qpos = np.asarray(self.env.qpos_reference[phase]).copy()
        qvel = np.asarray(self.env.qvel_reference[phase]).copy()
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "carried_reset_bank.npz"
            np.savez_compressed(
                bank_path,
                phase=phase,
                qpos=qpos,
                qvel=qvel,
            )
            probability = float(np.nextafter(0.0, 1.0))
            for domain_randomization in (False, True):
                with self.subTest(
                    domain_randomization=domain_randomization
                ):
                    env = G1TrackingEnv(
                        xml_path=str(MODEL),
                        reference_path=str(REFERENCE),
                        controller_path=str(CONTROLLER),
                        actor_history_len=1,
                        domain_randomization=domain_randomization,
                        reference_reset_noise_scale=1.0,
                        carried_reset_bank_path=str(bank_path),
                        carried_reset_probability=probability,
                    )
                    key = jax.random.PRNGKey(31)
                    state = env.reset(key, jnp.array(0.0))
                    reset_phase = int(state.info["phase"])
                    self.assertFalse(
                        np.array_equal(
                            np.asarray(state.data.qpos),
                            np.asarray(env.qpos_reference[reset_phase]),
                        )
                    )
                    self.assertFalse(
                        np.array_equal(
                            np.asarray(state.data.qvel),
                            np.asarray(env.qvel_reference[reset_phase]),
                        )
                    )

    def test_carried_reset_bank_restores_complete_actor_context(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        phases = np.array([21, 37], dtype=np.int32)
        template = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=10,
            actor_reference_lookahead_steps=(4, 8, 12),
            actor_reference_preview_mode="delta",
        )
        qpos = np.asarray(template.qpos_reference[phases]).copy()
        qvel = np.asarray(template.qvel_reference[phases]).copy()
        last_act = np.stack(
            (
                np.linspace(-0.4, 0.4, template.action_dim),
                np.linspace(0.3, -0.3, template.action_dim),
            )
        )
        histories = []
        for index, phase in enumerate(phases):
            reference_state = template.reset_at_phase(
                jax.random.PRNGKey(80 + index),
                jnp.array(0.0),
                jnp.array(phase),
            )
            info = {
                **reference_state.info,
                "last_act": jnp.asarray(last_act[index]),
            }
            current_frame = np.asarray(
                template._get_actor_obs(reference_state.data, info)
            )
            history = np.repeat(current_frame[None, :], 10, axis=0)
            history[:-1, 0] += np.linspace(0.01, 0.09, 9)
            histories.append(history)
        actor_obs_history = np.stack(histories)

        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "context_carried_reset_bank.npz"
            np.savez_compressed(
                bank_path,
                phase=phases,
                qpos=qpos,
                qvel=qvel,
                last_act=last_act,
                actor_obs_history=actor_obs_history,
            )
            env = G1TrackingEnv(
                xml_path=str(MODEL),
                reference_path=str(REFERENCE),
                controller_path=str(CONTROLLER),
                actor_history_len=10,
                actor_reference_lookahead_steps=(4, 8, 12),
                actor_reference_preview_mode="delta",
                reference_reset_noise_scale=1.0,
                carried_reset_bank_path=str(bank_path),
                carried_reset_probability=1.0,
                carried_reset_bank_start=1,
            )

            state = env.reset(jax.random.PRNGKey(29), jnp.array(0.0))

        self.assertTrue(env.carried_reset_restores_actor_context)
        self.assertEqual(int(state.info["phase"]), 37)
        np.testing.assert_allclose(
            state.info["last_act"], last_act[1], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            state.info["actor_obs_history"],
            actor_obs_history[1],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            state.obs,
            actor_obs_history[1].reshape(-1),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            state.info["bootstrap_obs"],
            actor_obs_history[1].reshape(-1),
            rtol=0.0,
            atol=0.0,
        )

    def test_carried_reset_bank_rejects_partial_or_invalid_actor_context(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        phase = np.array([21], dtype=np.int32)
        qpos = np.asarray(self.env.qpos_reference[phase]).copy()
        qvel = np.asarray(self.env.qvel_reference[phase]).copy()
        base = {"phase": phase, "qpos": qpos, "qvel": qvel}
        invalid_contexts = (
            {"last_act": np.zeros((1, 29))},
            {"actor_obs_history": np.zeros((1, 10, 154))},
            {
                "last_act": np.zeros((1, 28)),
                "actor_obs_history": np.zeros((1, 10, 154)),
            },
            {
                "last_act": np.zeros((1, 29)),
                "actor_obs_history": np.full((1, 10, 154), np.nan),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, context in enumerate(invalid_contexts):
                with self.subTest(index=index):
                    bank_path = Path(directory) / f"invalid_{index}.npz"
                    np.savez_compressed(bank_path, **base, **context)
                    with self.assertRaisesRegex(
                        ValueError, "carried reset bank actor context"
                    ):
                        G1TrackingEnv(
                            xml_path=str(MODEL),
                            reference_path=str(REFERENCE),
                            controller_path=str(CONTROLLER),
                            actor_history_len=10,
                            carried_reset_bank_path=str(bank_path),
                            carried_reset_probability=1.0,
                        )

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

    def test_motion_anchor_position_observation_is_default_off(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        default_env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )
        explicit_false_env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_observe_motion_anchor_position=False,
        )
        default_state = default_env.reset_at_phase(
            jax.random.PRNGKey(79), jnp.array(0.0), jnp.array(37)
        )
        explicit_false_state = explicit_false_env.reset_at_phase(
            jax.random.PRNGKey(79), jnp.array(0.0), jnp.array(37)
        )

        self.assertFalse(default_env.actor_observe_motion_anchor_position)
        self.assertFalse(explicit_false_env.actor_observe_motion_anchor_position)
        self.assertEqual(default_env.actor_frame_obs_dim, 154)
        self.assertEqual(default_env.actor_obs_dim, 154)
        np.testing.assert_array_equal(
            default_state.obs, explicit_false_state.obs
        )
        np.testing.assert_array_equal(
            default_env.actor_noise_mask, explicit_false_env.actor_noise_mask
        )

    def test_motion_anchor_position_observation_has_expected_order_and_shape(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingEnv,
            _quat_apply,
            _quat_inv,
            _rotation_6d,
        )

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=10,
            actor_reference_lookahead_steps=(4, 8, 12),
            actor_observe_motion_anchor_position=True,
        )
        phase = jnp.array(37)
        state = env.reset_at_phase(
            jax.random.PRNGKey(83), jnp.array(0.0), phase
        )
        translated_data = mjx.forward(
            env.mjx_model,
            state.data.replace(qpos=state.data.qpos.at[:3].add(
                jnp.array([0.3, -0.2, 0.1])
            )),
        )
        anchor_position, anchor_orientation = env._anchor_relative_reference(
            translated_data, phase
        )
        frame = np.asarray(env._get_actor_obs(translated_data, state.info))
        actor_order = np.asarray(env.model_to_actor_permutation)
        expected_command = np.concatenate(
            (
                np.asarray(env.qpos_reference[phase, 7:])[actor_order],
                np.asarray(env.qvel_reference[phase, 6:])[actor_order],
            )
        )

        self.assertTrue(env.actor_observe_motion_anchor_position)
        self.assertEqual(env.actor_frame_obs_dim, 331)
        self.assertEqual(env.actor_obs_dim, 3310)
        self.assertEqual(state.obs.shape, (3310,))
        np.testing.assert_allclose(frame[:58], expected_command, atol=1e-12)
        np.testing.assert_allclose(frame[58:61], anchor_position, atol=1e-12)
        np.testing.assert_allclose(
            frame[61:67],
            np.asarray(_rotation_6d(anchor_orientation)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            frame[67:70],
            np.asarray(
                _quat_apply(
                    _quat_inv(translated_data.qpos[3:7]),
                    translated_data.qvel[3:6],
                )
            ),
            atol=1e-12,
        )

    def test_motion_anchor_position_noise_mask_and_option_are_validated(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_observation_noise=True,
            actor_observe_motion_anchor_position=True,
        )

        mask = np.asarray(env.actor_noise_mask)
        self.assertEqual(mask.shape, (157,))
        np.testing.assert_array_equal(mask[:61], 0.0)
        np.testing.assert_allclose(mask[61:67], 0.05, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(mask[67:70], 0.2, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(mask[70:99], 0.01, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(mask[99:128], 0.01, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(mask[128:157], 0.0)
        for value in (1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "actor_observe_motion_anchor_position"
                ):
                    G1TrackingEnv(actor_observe_motion_anchor_position=value)

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
        self.assertNotIn(
            "determinism_mjx_substep_fingerprint", next_state.info
        )
        self.assertNotIn(
            "determinism_mjx_control_step_fingerprint", next_state.info
        )

    def test_probe_step_exposes_raw_mjx_fingerprints(self):
        env = copy.copy(self.env)
        env.determinism_probe = True
        state = env.reset(jax.random.PRNGKey(301), jnp.array(0.0))

        next_state = env.step(state, jnp.zeros(env.action_dim))

        for name in (
            "determinism_mjx_substep_fingerprint",
            "determinism_mjx_control_step_fingerprint",
            "determinism_mjx_substep_integrated_state_fingerprint",
            "determinism_mjx_substep_acceleration_state_fingerprint",
            "determinism_mjx_substep_constraint_force_fingerprint",
            "determinism_mjx_substep_contact_state_fingerprint",
            "determinism_mjx_substep_field_time_fingerprint",
            "determinism_mjx_substep_field_qpos_fingerprint",
            "determinism_mjx_substep_field_qvel_fingerprint",
            "determinism_mjx_substep_field_qacc_fingerprint",
            "determinism_mjx_substep_field_qacc_smooth_fingerprint",
            "determinism_mjx_substep_field_qacc_warmstart_fingerprint",
            "determinism_mjx_substep_field_qfrc_applied_fingerprint",
            "determinism_mjx_substep_field_qfrc_smooth_fingerprint",
            "determinism_mjx_substep_field_qfrc_constraint_fingerprint",
            "determinism_mjx_substep_field_efc_force_fingerprint",
            "determinism_mjx_substep_field_contact_fingerprint",
        ):
            fingerprint = next_state.info[name]
            self.assertEqual(fingerprint.shape, (4,))
            self.assertEqual(fingerprint.dtype, jnp.uint32)

    def test_environment_exposes_grouped_left_right_support(self):
        env = self.env
        state = env.reset_at_phase(
            jax.random.PRNGKey(41), jnp.array(0.0), jnp.array(0)
        )

        support = env.foot_support_signature(state.data)

        self.assertEqual(support.shape, (2,))
        self.assertEqual(support.dtype, jnp.bool_)
        np.testing.assert_array_equal(
            env._support_foot_body_ids,
            np.asarray([env.body_ids[3], env.body_ids[6]]),
        )

    def test_step_persists_nonreset_contact_topology_event(self):
        env = self.env
        state = env.reset_at_phase(
            jax.random.PRNGKey(42), jnp.array(0.0), jnp.array(0)
        )
        before = env.contact_pair_signature(state.data)

        next_state = env.step(state, jnp.zeros(env.action_dim))
        after = env.contact_pair_signature(next_state.data)

        expected = bool(jnp.any(before != after))
        self.assertEqual(
            bool(next_state.info["transition_contact_topology_event"]),
            expected,
        )

    def test_terminal_reset_is_not_reported_as_contact_topology_event(self):
        env = self.env
        final_phase = env.reference_length - 1
        state = env.reset_at_phase(
            jax.random.PRNGKey(43),
            jnp.array(0.0),
            jnp.array(final_phase),
        )

        next_state = env.step(state, jnp.zeros(env.action_dim))

        self.assertTrue(bool(next_state.done))
        self.assertFalse(
            bool(next_state.info["transition_contact_topology_event"])
        )

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

    def test_parity_variant_is_explicit_and_legacy_residual_stays_squashed(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzActionParityEnv,
            G1TrackingRMR50HzSourceStepEnv,
        )
        from src.envs.go2.environment import get_go2_env_class

        legacy = G1TrackingRMR50HzSourceStepEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_observation_noise=True,
            reference_residual_control=True,
        )
        parity = G1TrackingRMR50HzActionParityEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            actor_observation_noise=True,
            reference_residual_control=True,
        )

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_action_parity"),
            G1TrackingRMR50HzActionParityEnv,
        )
        self.assertTrue(legacy.squash_actor_actions)
        self.assertFalse(parity.squash_actor_actions)
        np.testing.assert_array_equal(
            np.asarray(legacy.actor_noise_mask)[96:125], 0.01
        )
        np.testing.assert_array_equal(
            np.asarray(parity.actor_noise_mask)[96:125], 0.5
        )

        state = parity.reset_at_phase(
            jax.random.PRNGKey(43), jnp.array(0.0), jnp.array(17)
        )
        raw_action = jnp.linspace(-4.8, 4.8, parity.action_dim)
        gradient = jax.grad(
            lambda value: jnp.sum(parity.position_target(state, value))
        )(raw_action)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertTrue(np.all(np.abs(np.asarray(gradient)) > 0.0))

    def test_decoupled_exploration_bounds_mean_but_not_sample(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzDecoupledExplorationEnv,
        )
        from src.envs.go2.environment import get_go2_env_class

        env = G1TrackingRMR50HzDecoupledExplorationEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_decoupled_exploration"),
            G1TrackingRMR50HzDecoupledExplorationEnv,
        )
        self.assertTrue(env.squash_actor_mean)
        self.assertFalse(env.clip_sampled_actor_actions)
        self.assertFalse(env.squash_actor_actions)

    def test_upstream_boundary_bounds_mean_and_sample(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzUpstreamBoundaryEnv,
        )
        from src.envs.go2.environment import get_go2_env_class

        env = G1TrackingRMR50HzUpstreamBoundaryEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_upstream_boundary"),
            G1TrackingRMR50HzUpstreamBoundaryEnv,
        )
        self.assertTrue(env.squash_actor_mean)
        self.assertTrue(env.clip_sampled_actor_actions)
        self.assertFalse(env.squash_actor_actions)

    def test_upstream_action_penalty_variant_matches_quadruped_weight(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzUpstreamActionPenaltyEnv,
        )
        from src.envs.go2.environment import get_go2_env_class

        env = G1TrackingRMR50HzUpstreamActionPenaltyEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

        self.assertIs(
            get_go2_env_class("g1_tracking_rmr_50hz_upstream_action_penalty"),
            G1TrackingRMR50HzUpstreamActionPenaltyEnv,
        )
        self.assertEqual(env.action_magnitude_weight, 0.05)
        self.assertTrue(env.clip_sampled_actor_actions)

    def test_parity_randomization_targets_torso_com_not_pelvis(self):
        from src.envs.g1_tracking.environment import (
            G1TrackingRMR50HzActionParityEnv,
            G1TrackingRMR50HzSourceStepEnv,
        )

        common = dict(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
            domain_randomization=True,
            friction_range=(1.0, 1.0),
            com_offset_range=(0.025, 0.05, 0.05),
        )
        legacy = G1TrackingRMR50HzSourceStepEnv(**common)
        parity = G1TrackingRMR50HzActionParityEnv(**common)
        sample = parity._nominal_randomization()
        sample = {
            **sample,
            "com_offset": jnp.asarray([0.01, -0.02, 0.03]),
        }

        randomized = parity._get_randomized_model(sample)
        legacy_randomized = legacy._get_randomized_model(sample)

        legacy_zero = legacy._sample_randomization(
            jax.random.PRNGKey(44), jnp.asarray(0.0)
        )
        parity_zero = parity._sample_randomization(
            jax.random.PRNGKey(44), jnp.asarray(0.0)
        )

        self.assertEqual(
            parity.randomization_com_body_id,
            parity.mj_model.body("torso_link").id,
        )
        np.testing.assert_allclose(
            randomized.body_ipos[parity.randomization_com_body_id],
            parity.base_ipos[parity.randomization_com_body_id]
            + sample["com_offset"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            randomized.body_ipos[parity.pelvis_body_id],
            parity.base_ipos[parity.pelvis_body_id],
        )
        np.testing.assert_array_equal(legacy_zero["com_offset"], jnp.zeros(3))
        self.assertGreater(
            np.linalg.norm(np.asarray(parity_zero["com_offset"])), 0.0
        )

        np.testing.assert_allclose(
            legacy_randomized.body_ipos[legacy.pelvis_body_id],
            legacy.base_ipos[legacy.pelvis_body_id] + sample["com_offset"],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
