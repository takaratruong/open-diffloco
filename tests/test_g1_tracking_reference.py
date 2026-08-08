import tempfile
import unittest
from pathlib import Path

import mujoco
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


class G1TrackingReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.reference import (
            RMR_G1_BODY_NAMES,
            load_mujoco_reference,
        )
        from src.envs.g1_tracking.controller import load_rmr_controller

        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))
        cls.controller = load_rmr_controller(cls.model, CONTROLLER)
        cls.reference = load_mujoco_reference(
            cls.model, REFERENCE, RMR_G1_BODY_NAMES
        )

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_body_order_matches_upstream_rmr_g1_task(self):
        from src.envs.g1_tracking.reference import RMR_G1_BODY_NAMES

        self.assertEqual(
            RMR_G1_BODY_NAMES,
            (
                "pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",
            ),
        )

    def test_reference_shapes_and_values_are_finite(self):
        reference = self.reference

        self.assertEqual(reference.qpos.shape, (121, self.model.nq))
        self.assertEqual(reference.qvel.shape, (121, self.model.nv))
        self.assertEqual(reference.body_pos.shape, (121, 14, 3))
        self.assertEqual(reference.body_quat.shape, (121, 14, 4))
        self.assertEqual(reference.body_lin_vel.shape, (121, 14, 3))
        self.assertEqual(reference.body_ang_vel.shape, (121, 14, 3))
        for array in (
            reference.qpos,
            reference.qvel,
            reference.body_pos,
            reference.body_quat,
            reference.body_lin_vel,
            reference.body_ang_vel,
        ):
            self.assertTrue(np.isfinite(array).all())

    def test_precomputed_pose_and_velocity_match_mujoco(self):
        reference = self.reference
        frame = 37
        data = mujoco.MjData(self.model)
        data.qpos[:] = reference.qpos[frame]
        data.qvel[:] = reference.qvel[frame]
        mujoco.mj_forward(self.model, data)

        for slot, body_id in enumerate(reference.body_ids):
            jacp = np.empty((3, self.model.nv))
            jacr = np.empty((3, self.model.nv))
            mujoco.mj_jacBody(self.model, data, jacp, jacr, int(body_id))
            np.testing.assert_allclose(
                reference.body_pos[frame, slot], data.xpos[body_id], atol=1e-7
            )
            np.testing.assert_allclose(
                reference.body_quat[frame, slot], data.xquat[body_id], atol=1e-7
            )
            np.testing.assert_allclose(
                reference.body_ang_vel[frame, slot],
                jacr @ data.qvel,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                reference.body_lin_vel[frame, slot],
                jacp @ data.qvel,
                atol=1e-7,
            )

    def test_body_quaternions_are_unit_length(self):
        norms = np.linalg.norm(self.reference.body_quat, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_named_rmr_reference_maps_joints_and_recovers_root_velocity(self):
        from src.envs.g1_tracking.reference import load_mujoco_reference

        frames = 5
        source_joint_pos = (
            np.arange(frames * 29, dtype=np.float32).reshape(frames, 29)
            / 1000.0
        )
        source_joint_vel = source_joint_pos / 10.0
        root_pos = np.zeros((frames, 1, 3), dtype=np.float32)
        root_pos[:, 0, 2] = 0.8
        root_quat = np.zeros((frames, 1, 4), dtype=np.float32)
        root_quat[..., 0] = 1.0
        root_lin_vel = np.tile(
            np.asarray([0.2, -0.1, 0.05], dtype=np.float32),
            (frames, 1, 1),
        )
        root_ang_vel = np.tile(
            np.asarray([0.1, 0.2, -0.15], dtype=np.float32),
            (frames, 1, 1),
        )
        fixture = Path(self.temporary_directory.name) / "named_rmr.npz"
        np.savez(
            fixture,
            fps=np.asarray([50], dtype=np.int32),
            joint_pos=source_joint_pos,
            joint_vel=source_joint_vel,
            body_pos_w=root_pos,
            body_quat_w=root_quat,
            body_lin_vel_w=root_lin_vel,
            body_ang_vel_w=root_ang_vel,
            joint_names=np.asarray(self.controller.actor_joint_names),
            root_body_name=np.asarray("pelvis"),
            root_body_index=np.asarray(0, dtype=np.int32),
        )

        reference = load_mujoco_reference(
            self.model,
            fixture,
            controller=self.controller,
        )

        self.assertEqual(reference.fps, 50.0)
        source_to_model = self.controller.actor_to_model_permutation
        np.testing.assert_allclose(
            reference.qpos[:, 7:], source_joint_pos[:, source_to_model], atol=0.0
        )
        np.testing.assert_allclose(
            reference.qvel[:, 6:], source_joint_vel[:, source_to_model], atol=0.0
        )
        np.testing.assert_allclose(
            reference.body_pos[:, 0], root_pos[:, 0], atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            reference.body_lin_vel[:, 0],
            root_lin_vel[:, 0],
            atol=2e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            reference.body_ang_vel[:, 0],
            root_ang_vel[:, 0],
            atol=2e-5,
            rtol=0.0,
        )

    def test_legacy_xv_reference_remains_unchanged(self):
        from src.envs.g1_tracking.reference import load_mujoco_reference

        actual = load_mujoco_reference(self.model, REFERENCE)
        with np.load(REFERENCE, allow_pickle=False) as archive:
            expected_qpos = np.array(archive["X"], dtype=np.float64, copy=True)
            expected_qpos[:, 3:7] /= np.linalg.norm(
                expected_qpos[:, 3:7], axis=-1, keepdims=True
            )
            np.testing.assert_array_equal(actual.qpos, expected_qpos)
            np.testing.assert_array_equal(
                actual.qvel,
                np.asarray(archive["V"], dtype=np.float64),
            )
        self.assertIsNone(actual.fps)


if __name__ == "__main__":
    unittest.main()
