import unittest

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.microbatch import (
    flatten_population,
    mean_shard_trees,
    reshape_population,
    summarize_shard_stats,
)


class MicrobatchTreeTest(unittest.TestCase):
    def test_population_reshape_round_trip_preserves_order(self):
        population = {
            "obs": jnp.arange(24, dtype=jnp.float32).reshape(8, 3),
            "info": {"step": jnp.arange(8, dtype=jnp.int32)},
        }

        sharded = reshape_population(
            population, accumulation_steps=2, microbatch_size=4
        )
        restored = flatten_population(sharded)

        self.assertEqual(sharded["obs"].shape, (2, 4, 3))
        np.testing.assert_array_equal(restored["obs"], population["obs"])
        np.testing.assert_array_equal(
            restored["info"]["step"], population["info"]["step"]
        )

    def test_population_reshape_rejects_wrong_leading_size(self):
        with self.assertRaisesRegex(ValueError, "effective leading size"):
            reshape_population(
                {"x": jnp.zeros((7, 2))},
                accumulation_steps=2,
                microbatch_size=4,
            )

    def test_mean_shard_trees_averages_only_shard_axis(self):
        shards = {
            "w": jnp.array([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]),
            "b": jnp.array([[2.0], [4.0], [6.0]]),
        }

        mean = mean_shard_trees(shards)

        np.testing.assert_allclose(mean["w"], jnp.array([3.0, 5.0]))
        np.testing.assert_allclose(mean["b"], jnp.array([4.0]))

    def test_shard_statistics_use_complete_environment_population(self):
        shard_stats = {
            "finite_by_env": jnp.array(
                [[True, True, False], [True, True, True]]
            ),
            "raw_norm_by_env": jnp.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 100.0]]
            ),
        }

        summary = summarize_shard_stats(shard_stats)

        self.assertAlmostEqual(float(summary["finite_fraction"]), 5.0 / 6.0)
        self.assertAlmostEqual(float(summary["raw_norm_median"]), 3.5)
        self.assertAlmostEqual(float(summary["raw_norm_max"]), 100.0)
        np.testing.assert_array_equal(
            summary["finite_by_env"],
            jnp.array([True, True, False, True, True, True]),
        )
        np.testing.assert_array_equal(
            summary["raw_norm_by_env"],
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]),
        )


if __name__ == "__main__":
    unittest.main()
