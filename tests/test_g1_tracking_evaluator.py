import unittest

import jax.numpy as jnp
import numpy as np

from tools.evaluate_g1_tracking import scale_policy_action


class G1TrackingEvaluatorTest(unittest.TestCase):
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
