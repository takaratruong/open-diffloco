import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from tools.evaluate_g1_tracking import (
    configure_jax,
    make_evaluation_env,
    scale_policy_action,
)


class G1TrackingEvaluatorTest(unittest.TestCase):
    def test_evaluator_enables_training_precision(self):
        with mock.patch.object(jax.config, "update") as update:
            configure_jax()

        update.assert_called_once_with("jax_enable_x64", True)

    def test_unbounded_native_timebase_is_available_for_action_tape_control(self):
        env = make_evaluation_env("g1_tracking_rmr_50hz_unbounded")

        self.assertAlmostEqual(env.dt, 0.02)
        self.assertEqual(env.reference_stride, 2)
        self.assertFalse(env.squash_actor_actions)

    def test_action_gain_scales_policy_without_changing_direction(self):
        action = jnp.array([-0.8, 0.2, 1.0])
        np.testing.assert_allclose(
            scale_policy_action(action, 0.25),
            np.array([-0.2, 0.05, 0.25]),
        )

    def test_action_gain_must_interpolate_zero_and_learned_policy(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            scale_policy_action(jnp.zeros(3), 1.1)


if __name__ == "__main__":
    unittest.main()
