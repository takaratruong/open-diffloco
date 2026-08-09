import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.gradient_audit import (
    detached_gaussian_score_loss,
    discounted_return_to_go,
)


class DiscountedReturnToGoTest(unittest.TestCase):
    def test_accumulates_normal_transitions_to_fragment_end(self):
        returns = discounted_return_to_go(
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([False, False, False]),
            gamma=0.5,
        )

        np.testing.assert_array_equal(returns, jnp.array([2.75, 3.5, 3.0]))

    def test_done_cuts_off_future_episode_rewards(self):
        returns = discounted_return_to_go(
            jnp.array([1.0, 2.0, 100.0, 4.0]),
            jnp.array([False, True, False, False]),
            gamma=0.5,
        )

        np.testing.assert_array_equal(returns, jnp.array([2.0, 2.0, 102.0, 4.0]))

    def test_final_terminal_keeps_its_immediate_reward(self):
        returns = discounted_return_to_go(
            jnp.array([1.0, 7.0]),
            jnp.array([False, True]),
            gamma=0.99,
        )

        np.testing.assert_array_equal(returns, jnp.array([7.93, 7.0]))


class DetachedGaussianScoreLossTest(unittest.TestCase):
    def test_gradient_has_analytic_gaussian_score_sign(self):
        mean = jnp.array([[0.25]])
        action = jnp.array([[0.75]])
        returns = jnp.array([2.0])

        gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                value, action, returns, std=0.5
            )
        )(mean)

        np.testing.assert_array_equal(gradient, jnp.array([[-4.0]]))

    def test_detaching_reparameterized_action_prevents_gradient_cancellation(
        self,
    ):
        mean = jnp.array([[0.25]])
        action = mean + 0.5 * jnp.array([[1.0]])
        returns = jnp.array([2.0])

        detached_gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                value, action, returns, std=0.5
            )
        )(mean)

        def incorrectly_attached_loss(value):
            reparameterized_action = value + 0.5 * jnp.array([[1.0]])
            log_probability = -0.5 * jnp.square(
                (reparameterized_action - value) / 0.5
            )
            return -jnp.sum(returns[:, None] * log_probability)

        attached_gradient = jax.grad(incorrectly_attached_loss)(mean)

        np.testing.assert_array_equal(detached_gradient, jnp.array([[-4.0]]))
        np.testing.assert_array_equal(attached_gradient, jnp.zeros_like(mean))

    def test_ratio_one_ppo_clipped_surrogate_gradient_equals_score_loss(self):
        mean = jnp.array([[-0.25], [0.75]])
        actions = jnp.array([[0.25], [0.25]])
        returns = jnp.array([2.0, -3.0])
        std = 0.5

        direct_gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                value, actions, returns, std=std
            )
        )(mean)

        stopped_actions = jax.lax.stop_gradient(actions)
        stopped_returns = jax.lax.stop_gradient(returns)

        def log_probability(value):
            squared_normalized_error = jnp.square(
                (stopped_actions - value) / std
            )
            return -0.5 * jnp.sum(
                squared_normalized_error, axis=-1
            )

        old_log_probability = jax.lax.stop_gradient(log_probability(mean))

        def ratio_one_ppo_loss(value):
            ratio = jnp.exp(log_probability(value) - old_log_probability)
            unclipped = ratio * stopped_returns
            clipped = jnp.clip(ratio, 0.8, 1.2) * stopped_returns
            return -jnp.mean(jnp.minimum(unclipped, clipped))

        ratio = jnp.exp(log_probability(mean) - old_log_probability)
        ppo_gradient = jax.grad(ratio_one_ppo_loss)(mean)

        np.testing.assert_array_equal(ratio, jnp.ones_like(ratio))
        np.testing.assert_array_equal(ppo_gradient, direct_gradient)


if __name__ == "__main__":
    unittest.main()
