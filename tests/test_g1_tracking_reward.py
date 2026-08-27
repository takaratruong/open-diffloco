import math
import unittest

import jax
import jax.numpy as jnp


class RMRTrackingRewardTest(unittest.TestCase):
    def setUp(self):
        self.body_pos = jnp.zeros((2, 3))
        self.body_quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        self.body_lin_vel = jnp.zeros((2, 3))
        self.body_ang_vel = jnp.zeros((2, 3))

    def reward(self, **overrides):
        from src.envs.g1_tracking.reward import rmr_tracking_reward

        values = {
            "target_anchor_pos": jnp.zeros(3),
            "actual_anchor_pos": jnp.zeros(3),
            "target_anchor_quat": jnp.array([1.0, 0.0, 0.0, 0.0]),
            "actual_anchor_quat": jnp.array([1.0, 0.0, 0.0, 0.0]),
            "target_body_pos": self.body_pos,
            "actual_body_pos": self.body_pos,
            "target_body_quat": self.body_quat,
            "actual_body_quat": self.body_quat,
            "target_body_lin_vel": self.body_lin_vel,
            "actual_body_lin_vel": self.body_lin_vel,
            "target_body_ang_vel": self.body_ang_vel,
            "actual_body_ang_vel": self.body_ang_vel,
        }
        values.update(overrides)
        return rmr_tracking_reward(**values)

    def test_exact_reference_has_maximum_six_term_reward(self):
        reward, components = self.reward()

        self.assertAlmostEqual(float(reward), 5.0, places=6)
        self.assertEqual(
            set(components),
            {
                "anchor_position",
                "anchor_orientation",
                "body_position",
                "body_orientation",
                "body_linear_velocity",
                "body_angular_velocity",
            },
        )
        for value in components.values():
            self.assertAlmostEqual(float(value), 1.0, places=6)

    def test_components_match_rmr_weights_and_stds(self):
        half_turn = jnp.array(
            [math.cos(0.5), 0.0, 0.0, math.sin(0.5)]
        )
        actual_body_pos = self.body_pos.at[0, 0].set(0.3)
        actual_body_lin_vel = self.body_lin_vel.at[0, 1].set(1.0)
        reward, components = self.reward(
            actual_anchor_pos=jnp.array([0.3, 0.0, 0.0]),
            actual_anchor_quat=half_turn,
            actual_body_pos=actual_body_pos,
            actual_body_lin_vel=actual_body_lin_vel,
        )

        self.assertAlmostEqual(
            float(components["anchor_position"]), math.exp(-1.0), places=6
        )
        self.assertAlmostEqual(
            float(components["anchor_orientation"]),
            math.exp(-(1.0**2) / (0.4**2)),
            places=5,
        )
        self.assertAlmostEqual(
            float(components["body_position"]), math.exp(-0.5), places=6
        )
        self.assertAlmostEqual(
            float(components["body_linear_velocity"]), math.exp(-0.5), places=6
        )
        expected = (
            0.5 * components["anchor_position"]
            + 0.5 * components["anchor_orientation"]
            + components["body_position"]
            + components["body_orientation"]
            + components["body_linear_velocity"]
            + components["body_angular_velocity"]
        )
        self.assertAlmostEqual(float(reward), float(expected), places=6)

    def test_reward_gradient_is_finite_and_points_back_to_reference(self):
        def scalar_reward(anchor_x):
            reward, _ = self.reward(
                actual_anchor_pos=jnp.array([anchor_x, 0.0, 0.0])
            )
            return reward

        gradient = jax.grad(scalar_reward)(jnp.array(0.1))
        self.assertTrue(bool(jnp.isfinite(gradient)))
        self.assertLess(float(gradient), 0.0)

    def test_dual_scale_anchor_position_preserves_peak_and_registered_mix(self):
        _, at_reference = self.reward(anchor_position_kernel="dual_scale")
        _, displaced = self.reward(
            actual_anchor_pos=jnp.array([0.3, 0.0, 0.0]),
            anchor_position_kernel="dual_scale",
        )

        self.assertAlmostEqual(float(at_reference["anchor_position"]), 1.0)
        expected = 0.75 * math.exp(-1.0) + 0.25 * math.exp(
            -(0.3 / 0.8) ** 2
        )
        self.assertAlmostEqual(
            float(displaced["anchor_position"]), expected, places=6
        )

    def test_dual_scale_anchor_position_has_more_far_error_gradient(self):
        def component(error, kernel):
            _, components = self.reward(
                actual_anchor_pos=jnp.array([error, 0.0, 0.0]),
                anchor_position_kernel=kernel,
            )
            return components["anchor_position"]

        legacy = jax.grad(lambda error: component(error, "exponential"))(
            jnp.array(0.74)
        )
        treatment = jax.grad(lambda error: component(error, "dual_scale"))(
            jnp.array(0.74)
        )

        self.assertTrue(bool(jnp.isfinite(treatment)))
        self.assertLess(float(treatment), 0.0)
        self.assertGreater(abs(float(treatment)), 7.0 * abs(float(legacy)))

    def test_quadratic_anchor_position_matches_exponential_locally_without_far_saturation(self):
        def component(error, kernel):
            _, components = self.reward(
                actual_anchor_pos=jnp.array([error, 0.0, 0.0]),
                anchor_position_kernel=kernel,
            )
            return components["anchor_position"]

        at_reference = component(jnp.array(0.0), "quadratic")
        at_scale = component(jnp.array(0.3), "quadratic")
        far_gradient = jax.grad(
            lambda error: component(error, "quadratic")
        )(jnp.array(0.74))

        self.assertAlmostEqual(float(at_reference), 1.0)
        self.assertAlmostEqual(float(at_scale), 0.0, places=6)
        self.assertAlmostEqual(
            float(far_gradient), -2.0 * 0.74 / 0.3**2, places=4
        )

    def test_tracking_reward_rejects_unknown_anchor_position_kernel(self):
        with self.assertRaisesRegex(ValueError, "anchor position kernel"):
            self.reward(anchor_position_kernel="not-a-kernel")

    def test_pseudo_huber_velocity_kernel_matches_registered_formula(self):
        actual_body_lin_vel = self.body_lin_vel.at[0, 1].set(1.0)

        reward, components = self.reward(
            actual_body_lin_vel=actual_body_lin_vel,
            velocity_kernel="pseudo_huber",
        )

        expected_linear_velocity = 2.0 - math.sqrt(2.0)
        self.assertAlmostEqual(
            float(components["body_linear_velocity"]),
            expected_linear_velocity,
            places=6,
        )
        self.assertAlmostEqual(
            float(components["body_angular_velocity"]), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(reward), 4.0 + expected_linear_velocity, places=6
        )

    def test_pseudo_huber_velocity_gradient_remains_informative_far_from_target(self):
        def velocity_component(error, kernel):
            _, components = self.reward(
                actual_body_lin_vel=self.body_lin_vel.at[0, 0].set(error),
                velocity_kernel=kernel,
            )
            return components["body_linear_velocity"]

        pseudo_huber_gradient = jax.grad(
            lambda error: velocity_component(error, "pseudo_huber")
        )(jnp.array(10.0))
        exponential_gradient = jax.grad(
            lambda error: velocity_component(error, "exponential")
        )(jnp.array(10.0))

        self.assertTrue(bool(jnp.isfinite(pseudo_huber_gradient)))
        self.assertLess(float(pseudo_huber_gradient), -0.1)
        self.assertLess(abs(float(exponential_gradient)), 1e-10)

    def test_tracking_reward_rejects_unknown_velocity_kernel(self):
        with self.assertRaisesRegex(ValueError, "velocity kernel"):
            self.reward(velocity_kernel="not-a-kernel")

    def test_torso_orientation_reward_matches_registered_pseudo_huber(self):
        from src.envs.g1_tracking.reward import torso_orientation_tracking_reward

        target = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (14, 1))
        actual = target.at[7].set(
            jnp.array([math.cos(0.2), 0.0, math.sin(0.2), 0.0])
        )

        value = torso_orientation_tracking_reward(target, actual)

        self.assertAlmostEqual(float(value), 2.0 - math.sqrt(3.0), places=5)

    def test_torso_orientation_reward_has_finite_nonsaturating_gradient(self):
        from src.envs.g1_tracking.reward import torso_orientation_tracking_reward

        target = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (14, 1))

        def reward(angle):
            actual = target.at[7].set(
                jnp.array([jnp.cos(angle / 2), 0.0, jnp.sin(angle / 2), 0.0])
            )
            return torso_orientation_tracking_reward(target, actual)

        gradient = jax.grad(reward)(jnp.array(1.2))

        self.assertTrue(bool(jnp.isfinite(gradient)))
        self.assertLess(float(gradient), -1.0)

    def test_termination_margin_penalty_activates_before_hard_limits(self):
        from src.envs.g1_tracking.reward import termination_margin_penalty

        inactive = termination_margin_penalty(
            anchor_z_error=jnp.array(0.125),
            anchor_xy_error=jnp.array(0.65),
            gravity_z_error=jnp.array(0.4),
            distal_z_error=jnp.array(0.2),
        )
        anchor_at_limit = termination_margin_penalty(
            anchor_z_error=jnp.array(0.25),
            anchor_xy_error=jnp.array(0.0),
            gravity_z_error=jnp.array(0.0),
            distal_z_error=jnp.array(0.0),
        )

        self.assertAlmostEqual(float(inactive), 0.0, places=7)
        self.assertAlmostEqual(float(anchor_at_limit), -1.0, places=7)

        bounded_explosion = termination_margin_penalty(
            anchor_z_error=jnp.array(1e9),
            anchor_xy_error=jnp.array(1e9),
            gravity_z_error=jnp.array(1e9),
            distal_z_error=jnp.array(1e9),
        )
        self.assertAlmostEqual(float(bounded_explosion), -4.0, places=7)

    def test_termination_margin_penalty_has_height_recovery_gradient(self):
        from src.envs.g1_tracking.reward import termination_margin_penalty

        gradient = jax.grad(
            lambda anchor_z_error: termination_margin_penalty(
                anchor_z_error=anchor_z_error,
                anchor_xy_error=jnp.array(0.0),
                gravity_z_error=jnp.array(0.0),
                distal_z_error=jnp.array(0.0),
            )
        )(jnp.array(0.2))

        self.assertTrue(math.isfinite(float(gradient)))
        self.assertLess(float(gradient), 0.0)

    def test_upstream_differentiable_regularizers_match_rmr_weights(self):
        from src.envs.g1_tracking.reward import rmr_regularization_reward

        penalty, components = rmr_regularization_reward(
            action=jnp.array([1.0, -1.0, 0.0]),
            previous_action=jnp.zeros(3),
            joint_pos=jnp.array([-1.2, 0.0, 1.4]),
            soft_joint_lower=jnp.array([-1.0, -1.0, -1.0]),
            soft_joint_upper=jnp.array([1.0, 1.0, 1.0]),
        )

        self.assertAlmostEqual(float(components["action_rate"]), -0.2)
        self.assertAlmostEqual(float(components["joint_limit"]), -6.0)
        self.assertAlmostEqual(float(penalty), -6.2, places=6)

    def test_regularizers_are_zero_for_steady_in_limit_commands(self):
        from src.envs.g1_tracking.reward import rmr_regularization_reward

        penalty, components = rmr_regularization_reward(
            action=jnp.array([0.2, -0.1]),
            previous_action=jnp.array([0.2, -0.1]),
            joint_pos=jnp.array([0.0, 0.5]),
            soft_joint_lower=jnp.array([-1.0, -1.0]),
            soft_joint_upper=jnp.array([1.0, 1.0]),
        )

        self.assertAlmostEqual(float(penalty), 0.0)
        self.assertTrue(all(float(value) == 0.0 for value in components.values()))

    def test_optional_action_magnitude_matches_upstream_weight(self):
        from src.envs.g1_tracking.reward import rmr_regularization_reward

        penalty, components = rmr_regularization_reward(
            action=jnp.array([1.0, -1.0, 0.5]),
            previous_action=jnp.array([1.0, -1.0, 0.5]),
            joint_pos=jnp.zeros(3),
            soft_joint_lower=-jnp.ones(3),
            soft_joint_upper=jnp.ones(3),
            action_magnitude_weight=0.05,
        )

        self.assertAlmostEqual(float(components["action_magnitude"]), -0.1125)
        self.assertAlmostEqual(float(penalty), -0.1125)

    def test_joint_limit_penalty_caps_solver_explosion_without_changing_weight(self):
        from src.envs.g1_tracking.reward import rmr_regularization_reward

        penalty, components = rmr_regularization_reward(
            action=jnp.zeros(2),
            previous_action=jnp.zeros(2),
            joint_pos=jnp.array([-1e9, 1e9]),
            soft_joint_lower=jnp.array([-1.0, -1.0]),
            soft_joint_upper=jnp.array([1.0, 1.0]),
        )

        self.assertAlmostEqual(float(components["joint_limit"]), -10.0)
        self.assertAlmostEqual(float(penalty), -10.0)


if __name__ == "__main__":
    unittest.main()
