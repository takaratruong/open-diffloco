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
        self.assertAlmostEqual(float(penalty), -6.2)

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


if __name__ == "__main__":
    unittest.main()
