import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.batch_gradients import (
    batch_mean_value_and_grad,
    finalize_tree_moments,
    init_tree_moments,
    tree_cosine,
    tree_dot,
    tree_norm,
    update_tree_moments,
)


class BatchGradientTest(unittest.TestCase):
    def test_batch_mean_gradient_equals_explicit_per_example_mean(self):
        params = {"w": jnp.array([0.5, -1.0])}
        x = jnp.array([[1.0, 2.0], [-2.0, 3.0], [4.0, -1.0]])

        def loss_one(value, sample):
            return jnp.square(jnp.dot(value["w"], sample))

        value, direct = batch_mean_value_and_grad(loss_one, params, x)
        individual_values = jax.vmap(loss_one, in_axes=(None, 0))(params, x)
        explicit = jax.tree_util.tree_map(
            lambda leaf: jnp.mean(leaf, axis=0),
            jax.vmap(jax.grad(loss_one), in_axes=(None, 0))(params, x),
        )

        np.testing.assert_allclose(value, jnp.mean(individual_values), atol=1e-7)
        np.testing.assert_allclose(direct["w"], explicit["w"], atol=1e-7)

    def test_supports_multiple_batched_inputs(self):
        params = {"scale": jnp.array(2.0)}
        x = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 3.0, 5.0])

        def loss_one(value, sample, expected):
            return jnp.square(value["scale"] * sample - expected)

        value, gradient = batch_mean_value_and_grad(
            loss_one, params, x, target
        )
        self.assertAlmostEqual(float(value), 0.75, places=6)
        self.assertAlmostEqual(
            float(gradient["scale"]), 11.0 / 3.0, places=6
        )


class TreeGeometryTest(unittest.TestCase):
    def setUp(self):
        self.left = {
            "a": jnp.array([1.0, 2.0]),
            "b": (jnp.array([-3.0]),),
        }
        self.right = {
            "a": jnp.array([4.0, -2.0]),
            "b": (jnp.array([1.0]),),
        }

    def test_tree_dot_norm_and_cosine_match_flat_vectors(self):
        left = np.array([1.0, 2.0, -3.0])
        right = np.array([4.0, -2.0, 1.0])
        expected_dot = float(left @ right)
        expected_norm = float(np.linalg.norm(left))
        expected_cosine = expected_dot / (
            np.linalg.norm(left) * np.linalg.norm(right)
        )

        self.assertAlmostEqual(float(tree_dot(self.left, self.right)), expected_dot)
        self.assertAlmostEqual(
            float(tree_norm(self.left)), expected_norm, places=6
        )
        self.assertAlmostEqual(
            float(tree_cosine(self.left, self.right)), expected_cosine
        )

    def test_zero_tree_has_finite_zero_cosine(self):
        zero = jax.tree_util.tree_map(jnp.zeros_like, self.left)
        self.assertEqual(float(tree_norm(zero)), 0.0)
        self.assertEqual(float(tree_cosine(zero, self.left)), 0.0)


class TreeMomentsTest(unittest.TestCase):
    @staticmethod
    def _tree(vector):
        return {
            "first": jnp.asarray(vector[:2]),
            "second": (jnp.asarray(vector[2:]),),
        }

    def test_online_moments_match_explicit_population_statistics(self):
        vectors = np.array(
            [
                [1.0, 2.0, -1.0],
                [3.0, 0.0, 2.0],
                [-1.0, 4.0, 1.0],
            ]
        )
        state = init_tree_moments(self._tree(vectors[0]))
        for vector in vectors:
            state = update_tree_moments(state, self._tree(vector))
        summary = finalize_tree_moments(state)

        mean = vectors.mean(axis=0)
        trace_variance = np.mean(
            np.sum(np.square(vectors - mean), axis=1)
        )
        expected_snr = np.linalg.norm(mean) / np.sqrt(trace_variance)
        self.assertEqual(int(state.count), len(vectors))
        self.assertAlmostEqual(
            float(summary.mean_norm), np.linalg.norm(mean), places=6
        )
        self.assertAlmostEqual(
            float(summary.trace_variance), trace_variance, places=6
        )
        self.assertAlmostEqual(float(summary.snr), expected_snr, places=6)

    def test_identical_gradients_have_infinite_snr(self):
        tree = self._tree(np.array([1.0, -2.0, 3.0]))
        state = init_tree_moments(tree)
        state = update_tree_moments(state, tree)
        state = update_tree_moments(state, tree)
        summary = finalize_tree_moments(state)
        self.assertEqual(float(summary.trace_variance), 0.0)
        self.assertTrue(math.isinf(float(summary.snr)))

    def test_zero_mean_with_nonzero_variance_has_zero_snr(self):
        positive = self._tree(np.array([1.0, -2.0, 3.0]))
        negative = self._tree(np.array([-1.0, 2.0, -3.0]))
        state = init_tree_moments(positive)
        state = update_tree_moments(state, positive)
        state = update_tree_moments(state, negative)
        summary = finalize_tree_moments(state)
        self.assertEqual(float(summary.mean_norm), 0.0)
        self.assertGreater(float(summary.trace_variance), 0.0)
        self.assertEqual(float(summary.snr), 0.0)

    def test_empty_moments_cannot_be_finalized(self):
        state = init_tree_moments(self._tree(np.zeros(3)))
        with self.assertRaises(ValueError):
            finalize_tree_moments(state)


if __name__ == "__main__":
    unittest.main()
