"""Detached phase-robust weighting for SHAC actor gradients."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jp


PyTree = Any


class PhaseWeighting(NamedTuple):
    """Per-bin and per-environment detached actor-loss weights."""

    bin_counts: jax.Array
    bin_losses: jax.Array
    bin_weights: jax.Array
    env_weights: jax.Array
    unweighted_loss: jax.Array
    weighted_loss: jax.Array
    valid: jax.Array


def phase_bin_indices(
    phases: jax.Array, *, phase_count: int, bin_count: int
) -> jax.Array:
    """Map valid reference phases to contiguous timeline bins."""
    if phase_count < 1 or bin_count < 1:
        raise ValueError("phase_count and bin_count must be positive")
    phases = jp.asarray(phases, dtype=jp.int32)
    return jp.minimum((phases * bin_count) // phase_count, bin_count - 1)


def phase_robust_weights(
    losses: jax.Array,
    phases: jax.Array,
    *,
    phase_count: int,
    bin_count: int,
    robust_fraction: float,
) -> PhaseWeighting:
    """Return detached bounded weights derived from occupied-bin losses."""
    if not 0.0 <= robust_fraction <= 1.0:
        raise ValueError("robust_fraction must be in [0, 1]")
    losses = jp.asarray(losses)
    phases = jp.asarray(phases)
    if losses.ndim != 1 or phases.shape != losses.shape:
        raise ValueError("losses and phases must be matching vectors")

    detached_losses = jax.lax.stop_gradient(losses)
    bins = phase_bin_indices(
        phases, phase_count=phase_count, bin_count=bin_count
    )
    counts = jp.zeros((bin_count,), dtype=jp.int32).at[bins].add(1)
    occupied = counts > 0
    sums = jp.zeros((bin_count,), dtype=losses.dtype).at[bins].add(
        detached_losses
    )
    bin_losses = jp.where(occupied, sums / jp.maximum(counts, 1), jp.nan)
    all_finite = jp.all(jp.isfinite(detached_losses))

    occupied_count = jp.sum(occupied).astype(losses.dtype)
    safe_losses = jp.where(
        occupied & jp.isfinite(bin_losses), bin_losses, 0.0
    )
    mean = jp.sum(safe_losses) / jp.maximum(occupied_count, 1.0)
    variance = (
        jp.sum(jp.where(occupied, jp.square(safe_losses - mean), 0.0))
        / jp.maximum(occupied_count, 1.0)
    )
    std = jp.sqrt(variance)
    standardized = jp.where(
        std >= 1e-6, (safe_losses - mean) / std, 0.0
    )
    probabilities = jax.nn.softmax(
        jp.where(occupied, standardized, -jp.inf)
    )
    raw_bin_weights = (
        (1.0 - robust_fraction)
        + robust_fraction * occupied_count * probabilities
    )
    raw_bin_weights = jp.where(occupied, raw_bin_weights, 0.0)
    raw_bin_weights = jp.where(
        all_finite,
        raw_bin_weights,
        occupied.astype(losses.dtype),
    )
    env_weights = raw_bin_weights[bins]
    env_weights = env_weights / jp.mean(env_weights)
    env_weights = jax.lax.stop_gradient(env_weights)
    raw_bin_weights = jax.lax.stop_gradient(raw_bin_weights)
    return PhaseWeighting(
        bin_counts=counts,
        bin_losses=bin_losses,
        bin_weights=raw_bin_weights,
        env_weights=env_weights,
        unweighted_loss=jp.mean(detached_losses),
        weighted_loss=(
            jp.sum(detached_losses * env_weights) / jp.sum(env_weights)
        ),
        valid=all_finite,
    )


def aggregate_phase_weighted_gradients(
    per_env_grads: PyTree, env_weights: jax.Array
) -> PyTree:
    """Take an elementwise finite-aware weighted environment mean."""
    leaves = jax.tree_util.tree_leaves(per_env_grads)
    env_weights = jp.asarray(env_weights)
    if not leaves or env_weights.ndim != 1:
        raise ValueError("gradient tree and vector weights are required")
    num_envs = env_weights.shape[0]
    if any(
        leaf.ndim < 1 or leaf.shape[0] != num_envs for leaf in leaves
    ):
        raise ValueError("all gradient leaves must share the weight axis")

    def weighted_mean(leaf):
        shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        weights = env_weights.reshape(shape)
        finite = jp.isfinite(leaf) & jp.isfinite(weights)
        numerator = jp.sum(
            jp.where(finite, leaf * weights, 0.0), axis=0
        )
        denominator = jp.sum(jp.where(finite, weights, 0.0), axis=0)
        return jp.where(denominator > 0.0, numerator / denominator, 0.0)

    return jax.tree_util.tree_map(weighted_mean, per_env_grads)
