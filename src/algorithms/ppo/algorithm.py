"""PPO primitives for the exact G1 MJX tracking positive control."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from src.core.rmr_training_policy import (
    GaussianRmrActorParams,
    gaussian_entropy as _gaussian_entropy,
    init_gaussian_rmr_actor,
)


def init_ppo_actor(
    key: jax.Array,
    *,
    input_dim: int,
    action_dim: int,
    initial_std: float,
) -> GaussianRmrActorParams:
    """Initializes the RMR actor with an explicit learned Gaussian scale."""

    if not math.isfinite(initial_std) or initial_std <= 0.0:
        raise ValueError("initial_std must be positive and finite")
    params = init_gaussian_rmr_actor(
        key,
        input_dim=input_dim,
        action_dim=action_dim,
        dtype=jnp.float32,
    )
    return params._replace(
        log_std=jnp.full(
            (action_dim,), jnp.log(jnp.float32(initial_std)), dtype=jnp.float32
        )
    )


def gaussian_log_prob(
    action: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Returns the joint log probability of a diagonal Gaussian action."""

    action = jnp.asarray(action)
    mean = jnp.asarray(mean, dtype=action.dtype)
    log_std = jnp.asarray(log_std, dtype=action.dtype)
    inverse_variance_error = (action - mean) * jnp.exp(-log_std)
    per_coordinate = -0.5 * (
        jnp.square(inverse_variance_error)
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    return jnp.sum(per_coordinate, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Returns summed entropy for a diagonal Gaussian."""

    return _gaussian_entropy(log_std)


def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    dones: jax.Array,
    terminals: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes GAE while distinguishing true terminals from truncations."""

    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values, dtype=rewards.dtype)
    bootstrap_values = jnp.asarray(bootstrap_values, dtype=rewards.dtype)
    dones = jnp.asarray(dones, dtype=rewards.dtype)
    terminals = jnp.asarray(terminals, dtype=rewards.dtype)
    if not (
        rewards.shape
        == values.shape
        == bootstrap_values.shape
        == dones.shape
        == terminals.shape
    ):
        raise ValueError("GAE inputs must have identical shapes")

    deltas = (
        rewards
        + gamma * (1.0 - terminals) * bootstrap_values
        - values
    )

    def backward(carry, inputs):
        delta, done = inputs
        advantage = delta + gamma * gae_lambda * (1.0 - done) * carry
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(values[0]),
        (deltas[::-1], dones[::-1]),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def ppo_loss(
    *,
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    entropy: jax.Array,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Returns the standard clipped PPO actor-critic objective."""

    ratio = jnp.exp(new_log_prob - old_log_prob)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -jnp.mean(
        jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    )

    clipped_values = old_values + jnp.clip(
        values - old_values, -clip_epsilon, clip_epsilon
    )
    value_errors = jnp.square(values - returns)
    clipped_value_errors = jnp.square(clipped_values - returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_errors, clipped_value_errors))
    mean_entropy = jnp.mean(entropy)
    total = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )
    approximate_kl = jnp.mean(old_log_prob - new_log_prob)
    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
    )
    return total, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": mean_entropy,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
    }
