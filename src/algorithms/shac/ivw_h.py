"""Phase-local IVW-H action-gradient primitives for SHAC diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jp


def _host_array(value: Any) -> np.ndarray | None:
    """Return a host array when validation is outside a traced computation."""

    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(value)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return None


def _require_finite(name: str, *values: Any) -> None:
    for value in values:
        array = _host_array(value)
        if array is not None and not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")


def _require_phase_replicas(phases: Any, *, minimum: int = 2) -> None:
    array = _host_array(phases)
    if array is None:
        return
    _, counts = np.unique(array, return_counts=True)
    if counts.size == 0 or np.any(counts < minimum):
        label = "two" if minimum == 2 else str(minimum)
        raise ValueError(f"every phase requires at least {label} replicas")


def discounted_reward_to_go(reward: Any, done: Any, *, gamma: float) -> jax.Array:
    """Compute discounted within-episode returns for a batched finite horizon."""

    reward = jp.asarray(reward)
    done = jp.asarray(done, dtype=jp.bool_)
    if reward.ndim != 2 or done.shape != reward.shape:
        raise ValueError("reward and done must have shape (N, H)")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError("gamma must be a finite scalar in [0, 1]")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be a finite scalar in [0, 1]")
    _require_finite("reward", reward)

    def backward(carry, values):
        reward_t, done_t = values
        value = reward_t + gamma * jp.where(done_t, 0.0, carry)
        return value, value

    _, reversed_values = jax.lax.scan(
        backward,
        jp.zeros(reward.shape[0], dtype=reward.dtype),
        (reward.T[::-1], done.T[::-1]),
    )
    return reversed_values[::-1].T


def leave_one_out_phase_advantages(returns: Any, phases: Any) -> jax.Array:
    """Center returns using only other replicas with the same start phase."""

    returns = jp.asarray(returns)
    phases = jp.asarray(phases)
    if returns.ndim != 2 or phases.shape != returns.shape[:1]:
        raise ValueError("returns/phases must have shapes (N, H) and (N,)")
    _require_finite("returns", returns)
    _require_phase_replicas(phases)

    same_phase = phases[:, None] == phases[None, :]
    peers = same_phase.astype(returns.dtype) - jp.eye(
        returns.shape[0], dtype=returns.dtype
    )
    peer_count = jp.sum(peers, axis=1)
    baseline = (
        jp.einsum("ij,jt->it", peers, returns) / peer_count[:, None]
    )
    return returns - baseline


def gaussian_mean_score_gradients(
    means: Any,
    sampled_actions: Any,
    advantages: Any,
    sigma: Any,
    *,
    horizon: int,
) -> jax.Array:
    """Return negative-return score samples with respect to Gaussian means."""

    means = jp.asarray(means)
    sampled_actions = jp.asarray(sampled_actions)
    advantages = jp.asarray(advantages)
    sigma = jp.asarray(sigma)
    if means.ndim != 3 or sampled_actions.shape != means.shape:
        raise ValueError("means/actions must have shape (N, H, A)")
    if advantages.shape != means.shape[:2] or sigma.shape != means.shape[2:]:
        raise ValueError("advantage/sigma shapes are incompatible")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    _require_finite(
        "score-gradient inputs", means, sampled_actions, advantages, sigma
    )
    host_sigma = _host_array(sigma)
    if host_sigma is not None and np.any(host_sigma <= 0.0):
        raise ValueError("sigma must be finite and positive")

    return (
        advantages[..., None]
        / horizon
        * (means - jax.lax.stop_gradient(sampled_actions))
        / jp.square(sigma)
    )


def phase_step_action_ivw(
    score_grad: Any,
    pathwise_grad: Any,
    phases: Any,
) -> jax.Array:
    """Compute published IVW-H pathwise weights within each phase group."""

    score_grad = jp.asarray(score_grad)
    pathwise_grad = jp.asarray(pathwise_grad)
    phases = jp.asarray(phases)
    if score_grad.ndim != 3 or pathwise_grad.shape != score_grad.shape:
        raise ValueError("gradient samples must share shape (N, H, A)")
    if phases.shape != score_grad.shape[:1]:
        raise ValueError("phases must have shape (N,)")
    _require_finite("gradient samples", score_grad, pathwise_grad)
    _require_phase_replicas(phases)

    same_phase = (phases[:, None] == phases[None, :]).astype(score_grad.dtype)
    count = jp.sum(same_phase, axis=1)

    def sample_variance(values):
        mean = (
            jp.einsum("ij,jta->ita", same_phase, values)
            / count[:, None, None]
        )
        sum_squares = jp.einsum("ij,jta->ita", same_phase, jp.square(values))
        centered_sum_squares = jp.maximum(
            sum_squares - count[:, None, None] * jp.square(mean),
            0.0,
        )
        return centered_sum_squares / (count[:, None, None] - 1.0)

    score_variance = sample_variance(score_grad)
    pathwise_variance = sample_variance(pathwise_grad)
    total_variance = score_variance + pathwise_variance
    return jp.where(
        total_variance > 0.0,
        score_variance / total_variance,
        0.0,
    )


def fuse_action_gradients(
    pathwise_grad: Any,
    score_grad: Any,
    pathwise_weight: Any,
) -> jax.Array:
    """Fuse action-gradient samples with the IVW-H pathwise weight."""

    pathwise_grad = jp.asarray(pathwise_grad)
    score_grad = jp.asarray(score_grad)
    pathwise_weight = jp.asarray(pathwise_weight)
    if (
        pathwise_grad.shape != score_grad.shape
        or pathwise_weight.shape != pathwise_grad.shape
    ):
        raise ValueError("fusion inputs must share shape")
    _require_finite(
        "fusion inputs", pathwise_grad, score_grad, pathwise_weight
    )
    host_weight = _host_array(pathwise_weight)
    if host_weight is not None and np.any(
        (host_weight < 0.0) | (host_weight > 1.0)
    ):
        raise ValueError("pathwise weights must be in [0, 1]")
    return (
        pathwise_weight * pathwise_grad
        + (1.0 - pathwise_weight) * score_grad
    )
