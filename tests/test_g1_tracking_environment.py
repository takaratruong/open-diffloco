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
