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

    def test_native_timebase_evaluation_uses_strict_rmr_termination(self):
        env = make_evaluation_env("g1_tracking_rmr_50hz")

        self.assertAlmostEqual(env.dt, 0.02)
        self.assertEqual(env.reference_stride, 2)
        self.assertEqual(env.termination_grace_steps, 0)

    def test_training_grace_variant_is_forbidden_in_evaluation(self):
        with self.assertRaisesRegex(ValueError, "evaluation environment"):
            make_evaluation_env("g1_tracking_rmr_50hz_grace")

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
