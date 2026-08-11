"""Unit tests for the frozen evaluation-only torso-wrench oracle."""

import math
import unittest

import jax.numpy as jp
import mujoco
import numpy as np


class G1TorsoWrenchOracleTest(unittest.TestCase):
    def setUp(self):
        from src.evaluation.g1_torso_wrench_oracle import (
            TorsoWrenchParameters,
        )

        self.parameters = TorsoWrenchParameters(
            nominal_total_mass=1.0,
            gravity_magnitude=10.0,
        )
        self.identity = jp.array([1.0, 0.0, 0.0, 0.0])
        self.zero = jp.zeros(3)

    def _wrench(
        self,
        *,
        actual_position=None,
        actual_quaternion=None,
        reference_position=None,
        reference_quaternion=None,
        actual_linear_velocity=None,
        reference_linear_velocity=None,
        actual_angular_velocity=None,
        reference_angular_velocity=None,
        scale=1.0,
    ):
        from src.evaluation.g1_torso_wrench_oracle import compute_torso_wrench

        return compute_torso_wrench(
            parameters=self.parameters,
            actual_position=(
                self.zero if actual_position is None else actual_position
            ),
            actual_quaternion=(
                self.identity
                if actual_quaternion is None
                else actual_quaternion
            ),
            actual_linear_velocity=(
                self.zero
                if actual_linear_velocity is None
                else actual_linear_velocity
            ),
            actual_angular_velocity=(
                self.zero
                if actual_angular_velocity is None
                else actual_angular_velocity
            ),
            reference_position=(
                self.zero if reference_position is None else reference_position
            ),
            reference_quaternion=(
                self.identity
                if reference_quaternion is None
                else reference_quaternion
            ),
            reference_linear_velocity=(
                self.zero
                if reference_linear_velocity is None
                else reference_linear_velocity
            ),
            reference_angular_velocity=(
                self.zero
                if reference_angular_velocity is None
                else reference_angular_velocity
            ),
            scale=scale,
        )

    def test_zero_scale_returns_an_exact_zero_wrench(self):
        wrench = self._wrench(
            reference_position=jp.array([1.0, -2.0, 3.0]),
            reference_quaternion=jp.array([0.0, 1.0, 0.0, 0.0]),
            reference_linear_velocity=jp.array([4.0, 5.0, 6.0]),
            reference_angular_velocity=jp.array([-3.0, 2.0, 1.0]),
            scale=0.0,
        )

        np.testing.assert_array_equal(np.asarray(wrench), np.zeros(6))

    def test_quaternion_error_uses_the_shortest_rotation(self):
        from src.evaluation.g1_torso_wrench_oracle import (
            shortest_quaternion_rotation_vector,
        )

        negative_quarter_turn_about_z = jp.array(
            [-math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]
        )
        rotation = shortest_quaternion_rotation_vector(
            target_quaternion=negative_quarter_turn_about_z,
            actual_quaternion=self.identity,
        )

        np.testing.assert_allclose(
            np.asarray(rotation),
            np.array([0.0, 0.0, math.pi / 2.0]),
            atol=1e-6,
        )

    def test_wrench_is_computed_in_yaw_frame_and_returned_in_world_frame(self):
        yaw_ninety_degrees = jp.array(
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        )
        wrench = self._wrench(
            actual_quaternion=yaw_ninety_degrees,
            reference_position=jp.array([0.01, 0.0, 0.0]),
            reference_quaternion=yaw_ninety_degrees,
        )

        expected_force = np.array(
            [self.parameters.translational_kp * 0.01, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            np.asarray(wrench[:3]), expected_force, atol=1e-6
        )
        np.testing.assert_allclose(np.asarray(wrench[3:]), np.zeros(3), atol=1e-6)

    def test_named_torso_row_receives_force_then_torque_channels(self):
        from src.evaluation.g1_torso_wrench_oracle import (
            resolve_torso_body_id,
            write_torso_wrench,
        )

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='torso_link'/></worldbody></mujoco>"
        )
        torso_body_id = resolve_torso_body_id(model)
        applied = write_torso_wrench(
            jp.zeros((model.nbody, 6)),
            torso_body_id=torso_body_id,
            world_wrench=jp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )

        np.testing.assert_array_equal(
            np.asarray(applied[torso_body_id]), np.arange(1.0, 7.0)
        )
        np.testing.assert_array_equal(np.asarray(applied[0]), np.zeros(6))

    def test_parameters_and_torso_id_are_resolved_from_the_environment(self):
        from src.evaluation.g1_torso_wrench_oracle import (
            torso_wrench_parameters_from_environment,
        )

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='torso_link'/></worldbody></mujoco>"
        )

        class Environment:
            mj_model = model
            nominal_total_mass = 42.0
            base_gravity_mag = 9.81

        torso_body_id, parameters = torso_wrench_parameters_from_environment(
            Environment()
        )

        self.assertEqual(torso_body_id, 1)
        self.assertEqual(parameters.nominal_total_mass, 42.0)
        self.assertEqual(parameters.gravity_magnitude, 9.81)

    def test_writing_another_policy_step_overwrites_stale_torso_wrench(self):
        from src.evaluation.g1_torso_wrench_oracle import write_torso_wrench

        previous = jp.full((4, 6), 7.0)
        applied = write_torso_wrench(
            previous,
            torso_body_id=2,
            world_wrench=jp.zeros(6),
        )

        np.testing.assert_array_equal(np.asarray(applied[2]), np.zeros(6))
        np.testing.assert_array_equal(np.asarray(applied[1]), np.full(6, 7.0))

    def test_force_and_torque_are_finite_and_norm_limited(self):
        wrench = self._wrench(
            reference_position=jp.array([1e20, -1e20, 1e20]),
            reference_quaternion=jp.array([0.0, 1.0, 0.0, 0.0]),
            reference_linear_velocity=jp.array([1e20, 1e20, -1e20]),
            reference_angular_velocity=jp.array([1e20, -1e20, 1e20]),
        )

        self.assertTrue(np.isfinite(np.asarray(wrench)).all())
        self.assertLessEqual(
            float(jp.linalg.norm(wrench[:3])),
            self.parameters.force_cap + 1e-5,
        )
        self.assertLessEqual(
            float(jp.linalg.norm(wrench[3:])),
            self.parameters.torque_cap + 1e-5,
        )


if __name__ == "__main__":
    unittest.main()
