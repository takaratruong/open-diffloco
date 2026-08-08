import unittest
import hashlib

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.envs.g1_tracking.failure_collocation import (
    FailureWindow,
    TML_ACTION_SCALE,
    TML_BODY_NAMES,
    TML_DEFAULT_JOINT_POS,
    TML_JOINT_NAMES,
    corrected_episode_mapping,
    feasibility_merit,
    multiple_shooting_equalities,
    physical_state_defect,
    rollout_segment,
    select_failure_window,
    world_body_kinematics,
)


class G1FailureCollocationTest(unittest.TestCase):
    def test_default_failure_window_has_fixed_small_dimensions(self):
        window = FailureWindow()

        self.assertEqual(window.transitions, 24)
        self.assertEqual(window.segments, 12)
        self.assertEqual(window.knot_phases, tuple(range(111, 136, 2)))
        self.assertEqual(window.decision_size, 1548)
        self.assertEqual(window.equality_size, 852)

    def test_failure_window_rejects_nondivisible_or_empty_ranges(self):
        for kwargs in (
            {"start_phase": 135, "end_phase": 135},
            {"start_phase": 111, "end_phase": 134},
            {"segment_steps": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    FailureWindow(**kwargs)

    def test_state_defect_is_zero_for_equal_states_and_quaternion_sign(self):
        qpos = jnp.linspace(-0.2, 0.3, 36, dtype=jnp.float64)
        qpos = qpos.at[3:7].set(
            jnp.array([0.5, -0.5, 0.5, -0.5], dtype=jnp.float64)
        )
        qvel = jnp.linspace(-1.0, 1.0, 35, dtype=jnp.float64)
        sign_flipped = qpos.at[3:7].multiply(-1.0)

        identical = physical_state_defect(qpos, qvel, qpos, qvel)
        sign_equivalent = physical_state_defect(
            qpos, qvel, sign_flipped, qvel
        )

        self.assertEqual(identical.shape, (70,))
        np.testing.assert_allclose(identical, 0.0, atol=1e-12)
        np.testing.assert_allclose(sign_equivalent, 0.0, atol=1e-12)

    def test_select_failure_window_aligns_knots_and_between_knot_actions(self):
        phases = np.arange(100, 141, dtype=np.int32)
        qpos = np.zeros((len(phases), 36), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[:, 0] = phases
        qvel = np.zeros((len(phases), 35), dtype=np.float64)
        qvel[:, 0] = phases
        action = np.zeros((len(phases), 29), dtype=np.float64)
        action[:, 0] = phases

        selected = select_failure_window(
            {
                "phase": phases,
                "qpos": qpos,
                "qvel": qvel,
                "action": action,
            },
            FailureWindow(),
        )

        np.testing.assert_array_equal(
            selected["knot_phase"], np.arange(111, 136, 2)
        )
        np.testing.assert_array_equal(
            selected["action_phase"], np.arange(111, 135)
        )
        np.testing.assert_array_equal(
            selected["knot_qpos"][:, 0], np.arange(111, 136, 2)
        )
        np.testing.assert_array_equal(
            selected["actions"][:, 0], np.arange(111, 135)
        )
        self.assertEqual(selected["knot_qpos"].shape, (13, 36))
        self.assertEqual(selected["knot_qvel"].shape, (13, 35))
        self.assertEqual(selected["actions"].shape, (24, 29))

    def test_select_failure_window_rejects_missing_phase(self):
        phases = np.delete(np.arange(111, 136, dtype=np.int32), 10)
        arrays = {
            "phase": phases,
            "qpos": np.pad(
                np.ones((len(phases), 1)), ((0, 0), (3, 32))
            ),
            "qvel": np.zeros((len(phases), 35)),
            "action": np.zeros((len(phases), 29)),
        }

        with self.assertRaisesRegex(ValueError, "every phase"):
            select_failure_window(arrays, FailureWindow())

    def test_generated_segment_has_zero_multiple_shooting_defect(self):
        qpos = jnp.zeros(36, dtype=jnp.float64).at[3].set(1.0)
        qvel = jnp.zeros(35, dtype=jnp.float64)
        actions = jnp.zeros((2, 29), dtype=jnp.float64)
        actions = actions.at[:, 0].set(jnp.array([0.1, -0.03]))
        actions = actions.at[:, 1].set(jnp.array([0.02, 0.04]))

        def step_fn(current_qpos, current_qvel, action):
            next_qpos = current_qpos.at[0].add(action[0])
            next_qvel = current_qvel.at[0].add(action[1])
            return next_qpos, next_qvel

        next_qpos, next_qvel = rollout_segment(
            step_fn, qpos, qvel, actions
        )
        equalities = multiple_shooting_equalities(
            step_fn,
            jnp.stack((qpos, next_qpos)),
            jnp.stack((qvel, next_qvel)),
            actions,
            segment_steps=2,
        )

        self.assertEqual(equalities.shape, (71,))
        np.testing.assert_allclose(equalities, 0.0, atol=1e-12)

    def test_quadratic_merit_gradient_is_finite(self):
        decision = jnp.array([0.2, -0.3], dtype=jnp.float64)

        def merit(value):
            return feasibility_merit(
                objective=jnp.sum(jnp.square(value)),
                equalities=value[:1],
                slacks=1.0 - jnp.abs(value),
            )

        value, gradient = jax.value_and_grad(merit)(decision)

        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertGreater(float(value), 0.0)

    def test_corrected_episode_mapping_matches_frozen_tml_raw_contract(self):
        rows = 13
        qpos = np.zeros((rows, 36), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[:, 0] = np.arange(rows)
        qvel = np.zeros((rows, 35), dtype=np.float64)
        qvel[:, 3:6] = np.arange(rows)[:, None]
        body_pos = np.zeros((rows, 30, 3), dtype=np.float64)
        body_rot = np.zeros((rows, 30, 4), dtype=np.float64)
        body_rot[..., 0] = 1.0
        body_lin_vel = np.zeros((rows, 30, 3), dtype=np.float64)
        root_ang_vel = np.full((rows, 3), 0.25, dtype=np.float64)
        actions = np.full((rows, 29), 0.1, dtype=np.float64)

        episode = corrected_episode_mapping(
            qpos=qpos,
            qvel=qvel,
            root_ang_vel=root_ang_vel,
            body_pos=body_pos,
            body_rot=body_rot,
            body_lin_vel=body_lin_vel,
            actions=actions,
            joint_names=TML_JOINT_NAMES,
            body_names=TML_BODY_NAMES,
            default_joint_pos=np.zeros(29),
            action_scale=np.ones(29),
            clip_name="dance1_subject2_f122_422_50hz",
            env_origin=np.zeros(3),
            checkpoint_sha256="d" * 64,
            config_sha256="e" * 64,
            checkpoint_path="/artifacts/actor.pkl",
            config_path="/artifacts/hparams.json",
            motion_asset_sha256="f" * 64,
            terrain_asset_sha256="1" * 64,
            motion_asset_path="/artifacts/dance.npz",
            terrain_asset_path="/artifacts/g1.xml",
            grail_commit="2" * 40,
            correction_method="failure_centered_multiple_shooting",
            correction_run_id="smoke-001",
            correction_source_sha256="a" * 64,
            correction_code_commit="b" * 40,
            dynamics_model_sha256="c" * 64,
            dynamics_backend="mujoco-mjx-3.9-solver4-ls5",
            episode_weight=1.0,
        )

        self.assertEqual(
            episode["schema_version"], "sonic_grail_rollout_npz_v1"
        )
        self.assertEqual(episode["root_pos"].shape, (rows, 3))
        self.assertEqual(episode["root_rot"].shape, (rows, 4))
        self.assertEqual(episode["root_ang_vel"].shape, (rows, 3))
        self.assertEqual(episode["body_pos"].shape, (rows, 30, 3))
        self.assertEqual(episode["body_rot"].shape, (rows, 30, 4))
        self.assertEqual(episode["body_lin_vel"].shape, (rows, 30, 3))
        self.assertEqual(episode["joint_pos"].shape, (rows, 29))
        self.assertEqual(episode["joint_vel"].shape, (rows, 29))
        self.assertEqual(episode["action"].shape, (rows, 29))
        np.testing.assert_array_equal(episode["root_pos"], qpos[:, :3])
        np.testing.assert_array_equal(episode["root_ang_vel"], root_ang_vel)
        np.testing.assert_array_equal(episode["joint_vel"], qvel[:, 6:])
        np.testing.assert_array_equal(
            episode["joint_names"], np.asarray(TML_JOINT_NAMES)
        )
        np.testing.assert_array_equal(
            episode["body_names"], np.asarray(TML_BODY_NAMES)
        )
        np.testing.assert_array_equal(
            episode["default_joint_pos"], TML_DEFAULT_JOINT_POS
        )
        np.testing.assert_array_equal(
            episode["action_scale"], TML_ACTION_SCALE
        )
        np.testing.assert_allclose(
            episode["default_joint_pos"]
            + episode["action_scale"] * episode["action"][0],
            np.full(29, 0.1),
            atol=1e-7,
        )
        self.assertEqual(
            hashlib.sha256(
                np.ascontiguousarray(
                    episode["default_joint_pos"], dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "58995c0aa7385a7325f53b98e9bcc9e4ca5abaadbf688bc5f15799621cd5806a",
        )
        self.assertEqual(
            hashlib.sha256(
                np.ascontiguousarray(
                    episode["action_scale"], dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "15b40285f67a8d10e7a82c11d4b03a13c2501e861dab88c8ddec6385fb696f52",
        )
        self.assertEqual(episode["sim_dt"], 0.005)
        self.assertEqual(episode["control_dt"], 0.02)
        self.assertEqual(episode["decimation"], 4)
        self.assertEqual(episode["trajectory_source"], "diffsim_corrected")
        self.assertEqual(episode["quaternion_convention"], "WXYZ")
        self.assertEqual(
            episode["root_angular_velocity_frame"], "world"
        )
        self.assertEqual(episode["checkpoint_sha256"], "d" * 64)
        self.assertEqual(episode["grail_commit"], "2" * 40)
        self.assertEqual(
            episode["action_semantics"],
            "raw_sonic_action_pd_target_equals_default_plus_g1_model12_scale",
        )

    def test_corrected_episode_mapping_hard_fails_missing_action_row(self):
        rows = 13
        qpos = np.zeros((rows, 36), dtype=np.float64)
        qpos[:, 3] = 1.0
        body_rot = np.zeros((rows, 30, 4), dtype=np.float64)
        body_rot[..., 0] = 1.0
        with self.assertRaisesRegex(ValueError, "action"):
            corrected_episode_mapping(
                qpos=qpos,
                qvel=np.zeros((rows, 35)),
                root_ang_vel=np.zeros((rows, 3)),
                body_pos=np.zeros((rows, 30, 3)),
                body_rot=body_rot,
                body_lin_vel=np.zeros((rows, 30, 3)),
                actions=np.zeros((rows - 1, 29)),
                joint_names=TML_JOINT_NAMES,
                body_names=TML_BODY_NAMES,
                default_joint_pos=np.zeros(29),
                action_scale=np.ones(29),
                clip_name="dance",
                env_origin=np.zeros(3),
                checkpoint_sha256="d" * 64,
                config_sha256="e" * 64,
                checkpoint_path="/actor.pkl",
                config_path="/hparams.json",
                motion_asset_sha256="f" * 64,
                terrain_asset_sha256="1" * 64,
                motion_asset_path="/dance.npz",
                terrain_asset_path="/g1.xml",
                grail_commit="2" * 40,
                correction_method="smoke",
                correction_run_id="smoke-001",
                correction_source_sha256="a" * 64,
                correction_code_commit="b" * 40,
                dynamics_model_sha256="c" * 64,
                dynamics_backend="mjx",
                episode_weight=1.0,
            )

    def test_world_body_kinematics_rotates_free_joint_angular_velocity(self):
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='root'>"
            "<freejoint/><geom type='sphere' size='.1' mass='1'/>"
            "</body></worldbody></mujoco>"
        )
        qpos = np.array(
            [[0.0, 0.0, 0.0, 2**-0.5, 0.0, 0.0, 2**-0.5]]
        )
        qvel = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

        _, _, _, body_ang_vel = world_body_kinematics(
            model, qpos, qvel, (1,)
        )

        np.testing.assert_allclose(
            body_ang_vel[0, 0], [0.0, 1.0, 0.0], atol=1e-12
        )
        self.assertFalse(
            np.allclose(body_ang_vel[0, 0], qvel[0, 3:6], atol=1e-12)
        )


if __name__ == "__main__":
    unittest.main()
