import math

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.ppo.algorithm import (
    compute_gae,
    gaussian_entropy,
    gaussian_log_prob,
    init_ppo_actor,
    ppo_loss,
)


def test_gaussian_log_prob_matches_closed_form() -> None:
    mean = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    action = jnp.array([[1.0, -1.0]], dtype=jnp.float32)
    log_std = jnp.log(jnp.array([0.5, 2.0], dtype=jnp.float32))

    observed = gaussian_log_prob(action, mean, log_std)
    expected = -0.5 * (
        ((1.0 - 0.0) / 0.5) ** 2
        + ((-1.0 - 1.0) / 2.0) ** 2
        + 2.0 * math.log(0.5)
        + 2.0 * math.log(2.0)
        + 2.0 * math.log(2.0 * math.pi)
    )

    np.testing.assert_allclose(observed, [expected], rtol=1e-6, atol=1e-6)


def test_compute_gae_bootstraps_clip_end_but_not_true_terminal() -> None:
    rewards = jnp.array([[1.0], [1.0]], dtype=jnp.float32)
    values = jnp.array([[0.25], [0.25]], dtype=jnp.float32)
    bootstrap_values = jnp.array([[2.0], [2.0]], dtype=jnp.float32)
    dones = jnp.ones((2, 1), dtype=jnp.float32)
    terminals = jnp.array([[1.0], [0.0]], dtype=jnp.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        bootstrap_values=bootstrap_values,
        dones=dones,
        terminals=terminals,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_returns = np.array([[1.0], [1.0 + 0.99 * 2.0]])
    np.testing.assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        advantages,
        expected_returns - np.asarray(values),
        rtol=1e-6,
        atol=1e-6,
    )


def test_ppo_loss_uses_clipped_surrogate_and_value_objective() -> None:
    old_log_prob = jnp.zeros(2, dtype=jnp.float32)
    new_log_prob = jnp.log(jnp.array([1.5, 0.5], dtype=jnp.float32))
    advantages = jnp.array([1.0, -1.0], dtype=jnp.float32)
    values = jnp.array([2.0, -2.0], dtype=jnp.float32)
    old_values = jnp.zeros(2, dtype=jnp.float32)
    returns = jnp.array([1.0, -1.0], dtype=jnp.float32)

    loss, metrics = ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        values=values,
        old_values=old_values,
        returns=returns,
        entropy=jnp.array([2.0, 2.0], dtype=jnp.float32),
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    # Ratios 1.5 and 0.5 both hit their adverse clipped bound, yielding
    # surrogate terms 1.2 and -0.8.  The clipped value prediction is +/-0.2.
    expected_policy_loss = -np.mean([1.2, -0.8])
    # PPO takes the larger of unclipped and clipped value errors.
    expected_value_loss = 0.5 * np.mean([1.0**2, (-1.0) ** 2])
    expected = expected_policy_loss + 0.5 * expected_value_loss - 0.01 * 2.0
    np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metrics["clip_fraction"], 1.0)


def test_initial_actor_standard_deviation_is_exactly_point_two() -> None:
    params = init_ppo_actor(
        jax.random.PRNGKey(4),
        input_dim=154,
        action_dim=29,
        initial_std=0.2,
    )

    np.testing.assert_allclose(
        np.exp(np.asarray(params.log_std)),
        np.full(29, 0.2, dtype=np.float32),
        rtol=1e-7,
        atol=1e-7,
    )


def test_ppo_objective_has_finite_gradients() -> None:
    def objective(new_log_prob, values, log_std):
        entropy = jnp.broadcast_to(
            gaussian_entropy(log_std), new_log_prob.shape
        )
        loss, _ = ppo_loss(
            new_log_prob=new_log_prob,
            old_log_prob=jnp.zeros(3),
            advantages=jnp.array([-1.0, 0.5, 2.0]),
            values=values,
            old_values=jnp.zeros(3),
            returns=jnp.array([-0.5, 1.0, 1.5]),
            entropy=entropy,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )
        return loss

    gradients = jax.grad(objective, argnums=(0, 1, 2))(
        jnp.array([-0.1, 0.0, 0.1]),
        jnp.array([-0.25, 0.5, 1.25]),
        jnp.log(jnp.full(29, 0.2)),
    )

    assert all(np.all(np.isfinite(np.asarray(gradient))) for gradient in gradients)
