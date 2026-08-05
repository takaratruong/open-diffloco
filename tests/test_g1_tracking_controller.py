import unittest
from pathlib import Path

import mujoco
import numpy as np


MODEL = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
CONTROLLER = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)


class G1TrackingControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.controller import load_rmr_controller

        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))
        cls.controller = load_rmr_controller(cls.model, CONTROLLER)

    def test_controller_is_in_model_joint_order(self):
        expected_names = tuple(
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            for joint_id in range(1, self.model.njnt)
        )
        self.assertEqual(self.controller.joint_names, expected_names)
        for array in (
            self.controller.kp,
            self.controller.kd,
            self.controller.effort_limit,
            self.controller.action_scale,
        ):
            self.assertEqual(array.shape, (29,))
            self.assertTrue(np.isfinite(array).all())
            self.assertTrue((array >= 0.0).all())
        self.assertEqual(self.controller.default_joint_pos.shape, (29,))
        self.assertTrue(np.isfinite(self.controller.default_joint_pos).all())

    def test_inferred_action_scale_reconstructs_logged_rmr_targets(self):
        with np.load(CONTROLLER, allow_pickle=False) as archive:
            log_names = tuple(map(str, archive["joint_names"]))
            perm = np.array(
                [log_names.index(name) for name in self.controller.joint_names]
            )
            actions = archive["action"][:, perm]
            expected_targets = archive["pos_target"][:, perm]

        actual_targets = (
            self.controller.default_joint_pos
            + actions * self.controller.action_scale
        )
        np.testing.assert_allclose(actual_targets, expected_targets, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
