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


class G1TrackingReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.reference import (
            RMR_G1_BODY_NAMES,
            load_mujoco_reference,
        )

        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))
        cls.reference = load_mujoco_reference(
            cls.model, REFERENCE, RMR_G1_BODY_NAMES
        )

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


if __name__ == "__main__":
    unittest.main()
