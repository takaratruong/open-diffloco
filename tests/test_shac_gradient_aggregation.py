import unittest

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.gradients import aggregate_per_env_gradients


class PerEnvironmentGradientAggregationTest(unittest.TestCase):
    def test_extreme_rollout_cannot_determine_aggregate_direction(self):
        gradients = {
            "w": jnp.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1_000_000.0],
                ]
            )
        }

        aggregated, stats = aggregate_per_env_gradients(
            gradients, max_norm=1.0
        )

        np.testing.assert_allclose(
            aggregated["w"], np.array([2.0 / 3.0, 1.0 / 3.0]), atol=1e-6
        )
        self.assertAlmostEqual(float(stats["raw_norm_median"]), 1.0)
        self.assertAlmostEqual(float(stats["raw_norm_max"]), 1_000_000.0)

    def test_nonfinite_rollout_is_removed_as_a_whole(self):
        gradients = {
            "w": jnp.array([[2.0, 0.0], [jnp.nan, 4.0]]),
            "b": jnp.array([[0.0], [3.0]]),
        }

        aggregated, stats = aggregate_per_env_gradients(
            gradients, max_norm=1.0
        )

        np.testing.assert_allclose(aggregated["w"], np.array([0.5, 0.0]))
        np.testing.assert_allclose(aggregated["b"], np.array([0.0]))
        self.assertAlmostEqual(float(stats["finite_fraction"]), 0.5)

    def test_rejects_mismatched_environment_axes(self):
        with self.assertRaisesRegex(ValueError, "leading env axis"):
            aggregate_per_env_gradients(
                {"w": jnp.zeros((2, 3)), "b": jnp.zeros((3,))},
                max_norm=1.0,
            )


if __name__ == "__main__":
    unittest.main()
