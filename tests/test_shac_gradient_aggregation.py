import unittest

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.gradients import (
    aggregate_per_env_gradients,
    grouped_gradient_population_statistics,
    per_env_gradient_statistics,
    rollout_terminal_mode_indices,
    support_contact_mode_indices,
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

    def test_statistics_measure_clipped_population_cancellation(self):
        gradients = {
            "w": jnp.array(
                [
                    [10.0, 0.0],
                    [-10.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        }

        stats = per_env_gradient_statistics(gradients, max_norm=1.0)

        np.testing.assert_allclose(
            stats["effective_norm_by_env"], np.ones(4), atol=1e-6
        )
        self.assertAlmostEqual(float(stats["population_mean_norm"]), 0.5)
        self.assertAlmostEqual(float(stats["population_rms_norm"]), 1.0)
        self.assertAlmostEqual(float(stats["population_variance_trace"]), 0.75)
        self.assertAlmostEqual(float(stats["population_cancellation_ratio"]), 0.5)
        self.assertAlmostEqual(float(stats["population_gradient_noise_scale"]), 3.0)
        self.assertAlmostEqual(float(stats["population_esnr"]), 4.0 / 3.0)

    def test_grouped_statistics_partition_within_and_between_variance(self):
        stats = grouped_gradient_population_statistics(
            group_mean_gradients={
                "w": jnp.array([[0.0, 0.0], [0.0, 1.0]])
            },
            population_mean_gradient={"w": jnp.array([0.0, 0.5])},
            effective_norm_by_env=jnp.ones((4,)),
            group_indices=jnp.array([0, 0, 1, 1]),
            group_count=2,
        )

        np.testing.assert_array_equal(stats["group_counts"], np.array([2, 2]))
        np.testing.assert_allclose(stats["group_mean_norms"], [0.0, 1.0])
        np.testing.assert_allclose(stats["group_rms_norms"], [1.0, 1.0])
        np.testing.assert_allclose(stats["group_variance_traces"], [1.0, 0.0])
        np.testing.assert_allclose(stats["group_cancellation_ratios"], [0.0, 1.0])
        self.assertAlmostEqual(float(stats["within_group_variance_trace"]), 0.5)
        self.assertAlmostEqual(float(stats["between_group_variance_trace"]), 0.25)
        self.assertAlmostEqual(float(stats["total_variance_trace"]), 0.75)
        self.assertAlmostEqual(float(stats["within_group_variance_fraction"]), 2.0 / 3.0)

    def test_support_contact_modes_encode_none_left_right_both(self):
        modes = support_contact_mode_indices(
            jnp.array(
                [
                    [False, False],
                    [True, False],
                    [False, True],
                    [True, True],
                ]
            )
        )

        np.testing.assert_array_equal(modes, np.arange(4))

    def test_terminal_modes_encode_survival_and_early_middle_late(self):
        terminals = jnp.zeros((4, 24), dtype=bool)
        terminals = terminals.at[1, 0].set(True)
        terminals = terminals.at[2, 8].set(True)
        terminals = terminals.at[3, 23].set(True)

        modes = rollout_terminal_mode_indices(terminals)

        np.testing.assert_array_equal(modes, np.arange(4))

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
