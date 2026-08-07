"""Robust aggregation helpers for differentiable-simulation policy gradients."""

from typing import Any

import jax
import jax.numpy as jp


def aggregate_per_env_gradients(
    per_env_grads: Any,
    *,
    max_norm: float,
) -> tuple[Any, dict[str, jax.Array]]:
    """Sanitizes and norm-clips each environment before taking its mean.

    Differentiable contact can produce a finite but extreme gradient for one
    rollout.  Clipping only the already-averaged gradient bounds the optimizer
    step but does not stop that rollout from determining the update direction.
    """
    leaves = jax.tree_util.tree_leaves(per_env_grads)
    if not leaves:
        raise ValueError("per_env_grads must contain at least one array")

    num_envs = leaves[0].shape[0]
    finite_by_env = jp.ones((num_envs,), dtype=jp.bool_)
    squared_norm = jp.zeros((num_envs,), dtype=jp.float32)
    for leaf in leaves:
        if leaf.shape[0] != num_envs:
            raise ValueError("all gradient leaves must share the leading env axis")
        reduction_axes = tuple(range(1, leaf.ndim))
        finite_by_env = finite_by_env & jp.all(
            jp.isfinite(leaf), axis=reduction_axes
        )
        safe_leaf = jp.where(jp.isfinite(leaf), leaf, 0.0)
        squared_norm = squared_norm + jp.sum(
            jp.square(safe_leaf.astype(jp.float32)), axis=reduction_axes
        )

    raw_norm = jp.sqrt(squared_norm)
    scale = jp.minimum(1.0, max_norm / jp.maximum(raw_norm, 1e-12))
    scale = jp.where(finite_by_env, scale, 0.0)

    def clip_and_average(leaf):
        safe_leaf = jp.where(jp.isfinite(leaf), leaf, 0.0)
        broadcast_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        return jp.mean(safe_leaf * scale.reshape(broadcast_shape), axis=0)

    aggregated = jax.tree_util.tree_map(clip_and_average, per_env_grads)
    stats = {
        "finite_fraction": jp.mean(finite_by_env.astype(jp.float32)),
        "raw_norm_median": jp.median(raw_norm),
        "raw_norm_max": jp.max(raw_norm),
        "finite_by_env": finite_by_env,
        "raw_norm_by_env": raw_norm,
    }
    return aggregated, stats
