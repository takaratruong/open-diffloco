"""Focused tests for randomly initialized RMR training networks."""

import math
import unittest

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from src.core.rmr_training_policy import (
    RMR_HIDDEN_DIMS,
    apply_rmr_mlp,
    gaussian_entropy,
    init_gaussian_rmr_actor,
    init_rmr_critic,
    rmr_mlp_parameter_count,
    sample_rmr_action,
)


class RmrTrainingPolicyTest(unittest.TestCase):
    def test_exact_actor_architecture_and_nonzero_output_head(self):
        params = init_gaussian_rmr_actor(jax.random.PRNGKey(42))

        self.assertEqual(
            tuple(weight.shape for weight in params.mlp.weights),
            (
                (2048, 154),
                (2048, 2048),
                (1024, 2048),
                (1024, 1024),
                (512, 1024),
                (512, 512),
                (29, 512),
            ),
        )
        self.assertEqual(RMR_HIDDEN_DIMS, (2048, 2048, 1024, 1024, 512, 512))
        self.assertEqual(rmr_mlp_parameter_count(params.mlp), 8_463_901)
        self.assertTrue(np.any(np.asarray(params.mlp.weights[-1]) != 0.0))
        np.testing.assert_array_equal(np.asarray(params.log_std), np.zeros(29))

    def test_exact_critic_architecture_and_parameter_count(self):
        params = init_rmr_critic(jax.random.PRNGKey(42))

        self.assertEqual(
            tuple(weight.shape for weight in params.weights),
            (
                (2048, 286),
                (2048, 2048),
                (1024, 2048),
                (1024, 1024),
                (512, 1024),
                (512, 512),
                (1, 512),
            ),
        )
        self.assertEqual(rmr_mlp_parameter_count(params), 8_719_873)

    def test_every_linear_parameter_respects_pytorch_default_bounds(self):
        actor = init_gaussian_rmr_actor(jax.random.PRNGKey(7))
        critic = init_rmr_critic(jax.random.PRNGKey(8))

        for params in (actor.mlp, critic):
            for weight, bias in zip(params.weights, params.biases):
                bound = 1.0 / math.sqrt(weight.shape[1])
                self.assertLessEqual(
                    float(np.max(np.abs(np.asarray(weight)))),
                    bound,
                )
                self.assertLessEqual(
                    float(np.max(np.abs(np.asarray(bias)))),
                    bound,
                )

    def test_application_matches_explicit_elu_reference(self):
        params = init_gaussian_rmr_actor(jax.random.PRNGKey(9))
        observations = jax.random.normal(
            jax.random.PRNGKey(10),
            (3, 154),
            dtype=jnp.float32,
        )

        expected = observations
        for index, (weight, bias) in enumerate(
            zip(params.mlp.weights, params.mlp.biases)
        ):
            expected = (
                jnp.matmul(
                    expected,
                    weight.T,
                    precision=lax.Precision.HIGHEST,
                )
                + bias
            )
            if index != len(params.mlp.weights) - 1:
                expected = jax.nn.elu(expected)

        actual = apply_rmr_mlp(params.mlp, observations)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_sampling_uses_caller_owned_epsilon_exactly(self):
        params = init_gaussian_rmr_actor(jax.random.PRNGKey(1))
        observations = jnp.zeros((3, 154), dtype=jnp.float32)
        epsilon = jnp.arange(87, dtype=jnp.float32).reshape(3, 29) / 100.0
        mean = apply_rmr_mlp(params.mlp, observations)

        np.testing.assert_allclose(
            sample_rmr_action(params, observations, epsilon),
            mean + epsilon,
            rtol=0.0,
            atol=1e-7,
        )

    def test_entropy_matches_unit_gaussian(self):
        log_std = jnp.zeros(29, dtype=jnp.float32)
        expected = 29 * 0.5 * (1.0 + math.log(2.0 * math.pi))
        self.assertAlmostEqual(float(gaussian_entropy(log_std)), expected, places=4)

    def test_every_actor_layer_and_log_std_receives_gradient(self):
        params = init_gaussian_rmr_actor(jax.random.PRNGKey(2))
        observations = jax.random.normal(
            jax.random.PRNGKey(3),
            (2, 154),
            dtype=jnp.float32,
        )
        epsilon = jnp.ones((2, 29), dtype=jnp.float32)

        gradients = jax.grad(
            lambda value: jnp.sum(
                sample_rmr_action(value, observations, epsilon)
            )
        )(params)

        for gradient in (
            *gradients.mlp.weights,
            *gradients.mlp.biases,
            gradients.log_std,
        ):
            array = np.asarray(gradient)
            self.assertTrue(np.isfinite(array).all())
            self.assertGreater(float(np.linalg.norm(array)), 0.0)

    def test_invalid_dimensions_fail_closed(self):
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    init_gaussian_rmr_actor(
                        jax.random.PRNGKey(4),
                        input_dim=invalid,
                    )


if __name__ == "__main__":
    unittest.main()
