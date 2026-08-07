import unittest

import jax.numpy as jnp

from src.algorithms.shac.algorithm import actor_bootstrap_scale_at_step


class ActorBootstrapScheduleTest(unittest.TestCase):
    def test_zero_delay_preserves_existing_scale_from_step_zero(self):
        scale = actor_bootstrap_scale_at_step(jnp.array(0), 0.75, 0)
        self.assertEqual(float(scale), 0.75)
        self.assertEqual(scale.dtype, jnp.dtype("float32"))

    def test_positive_delay_is_zero_before_boundary_and_full_at_boundary(self):
        self.assertEqual(
            float(actor_bootstrap_scale_at_step(jnp.array(61_439), 1.0, 61_440)),
            0.0,
        )
        self.assertEqual(
            float(actor_bootstrap_scale_at_step(jnp.array(61_440), 1.0, 61_440)),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
