import unittest

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.gradients import (
    aggregate_per_env_gradients,
    per_env_gradient_statistics,
)


class PerEnvironmentGradientAggregationTest(unittest.TestCase):
    def test_statistics_report_nonfinite_rollouts_without_aggregating(self):
        gradients = {
            "w": jnp.array([[3.0, 4.0], [jnp.nan, 9.0]]),
            "b": jnp.array([[0.0], [2.0]]),
        }

        stats = per_env_gradient_statistics(gradients)

        self.assertAlmostEqual(float(stats["finite_fraction"]), 0.5)
        np.testing.assert_array_equal(
            stats["finite_by_env"], np.array([True, False])
        )
        np.testing.assert_allclose(
            stats["raw_norm_by_env"], np.array([5.0, np.sqrt(85.0)])
        )

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

    def test_shard_statistics_reconstruct_full_population(self):
        gradients = {
            "w": jnp.array(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [jnp.nan, 1.0],
                    [0.0, 4.0],
                    [0.0, 5.0],
                    [0.0, 6.0],
                ]
            )
        }

        full, full_stats = aggregate_per_env_gradients(
            gradients, max_norm=2.0
        )
        shard_results = [
            aggregate_per_env_gradients(
                {"w": gradients["w"][start : start + 3]},
                max_norm=2.0,
            )
            for start in (0, 3)
        ]
        shard_means = jnp.stack(
            [result[0]["w"] for result in shard_results]
        )
        norms = jnp.concatenate(
            [result[1]["raw_norm_by_env"] for result in shard_results]
        )
        finite = jnp.concatenate(
            [result[1]["finite_by_env"] for result in shard_results]
        )

        np.testing.assert_allclose(jnp.mean(shard_means, axis=0), full["w"])
        self.assertAlmostEqual(
            float(jnp.mean(finite.astype(jnp.float32))),
            float(full_stats["finite_fraction"]),
        )
        self.assertAlmostEqual(
            float(jnp.median(norms)), float(full_stats["raw_norm_median"])
        )
        self.assertAlmostEqual(
            float(jnp.max(norms)), float(full_stats["raw_norm_max"])
        )


if __name__ == "__main__":
    unittest.main()
