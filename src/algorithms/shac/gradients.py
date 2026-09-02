"""Robust aggregation helpers for differentiable-simulation policy gradients."""

from typing import Any

import jax
import jax.numpy as jp


def support_contact_mode_indices(foot_support: jax.Array) -> jax.Array:
    """Encode no, left, right, and double support as indices zero to three."""

    support = jp.asarray(foot_support, dtype=jp.bool_)
    if support.ndim != 2 or support.shape[1] != 2:
        raise ValueError("foot_support must have shape (population, 2)")
    return support[:, 0].astype(jp.int32) + 2 * support[:, 1].astype(
        jp.int32
    )


def rollout_terminal_mode_indices(terminals: jax.Array) -> jax.Array:
    """Encode survival or terminal occurrence in the early, middle, or late third."""

    terminal = jp.asarray(terminals, dtype=jp.bool_)
    if terminal.ndim != 2 or terminal.shape[1] < 1:
        raise ValueError("terminals must have shape (population, horizon)")
    horizon = terminal.shape[1]
    terminated = jp.any(terminal, axis=1)
    first_terminal = jp.argmax(terminal, axis=1)
    terminal_third = jp.minimum((first_terminal * 3) // horizon, 2)
    return jp.where(terminated, terminal_third + 1, 0).astype(jp.int32)


def gradient_population_statistics(
    population_mean_gradient: Any,
    effective_norm_by_env: jax.Array,
) -> dict[str, jax.Array]:
    """Summarize cancellation and noise around one population mean gradient."""

    leaves = jax.tree_util.tree_leaves(population_mean_gradient)
    norms = jp.asarray(effective_norm_by_env, dtype=jp.float32)
    if not leaves:
        raise ValueError("population_mean_gradient must contain at least one array")
    if norms.ndim != 1 or norms.shape[0] < 1:
        raise ValueError("effective_norm_by_env must be a nonempty vector")

    population_mean_squared_norm = jp.asarray(0.0, dtype=jp.float32)
    for leaf in leaves:
        population_mean_squared_norm = population_mean_squared_norm + jp.sum(
            jp.square(jp.asarray(leaf, dtype=jp.float32))
        )
    population_mean_squared_norm = jp.maximum(
        population_mean_squared_norm, 0.0
    )
    population_mean_squared_sample_norm = jp.mean(jp.square(norms))
    population_variance_trace = jp.maximum(
        population_mean_squared_sample_norm - population_mean_squared_norm,
        0.0,
    )
    epsilon = jp.asarray(1e-12, dtype=jp.float32)
    return {
        "population_mean_norm": jp.sqrt(population_mean_squared_norm),
        "population_rms_norm": jp.sqrt(population_mean_squared_sample_norm),
        "population_variance_trace": population_variance_trace,
        "population_cancellation_ratio": jp.sqrt(
            population_mean_squared_norm
            / jp.maximum(population_mean_squared_sample_norm, epsilon)
        ),
        "population_gradient_noise_scale": (
            population_variance_trace
            / jp.maximum(population_mean_squared_norm, epsilon)
        ),
        "population_esnr": (
            norms.shape[0]
            * population_mean_squared_norm
            / jp.maximum(population_variance_trace, epsilon)
        ),
    }


def grouped_gradient_population_statistics(
    *,
    group_mean_gradients: Any,
    population_mean_gradient: Any,
    effective_norm_by_env: jax.Array,
    group_indices: jax.Array,
    group_count: int,
) -> dict[str, jax.Array]:
    """Partition gradient trace variance into fixed categorical groups."""

    group_leaves = jax.tree_util.tree_leaves(group_mean_gradients)
    population_leaves = jax.tree_util.tree_leaves(population_mean_gradient)
    norms = jp.asarray(effective_norm_by_env, dtype=jp.float32)
    indices = jp.asarray(group_indices, dtype=jp.int32)
    if not group_leaves or not population_leaves:
        raise ValueError("group and population gradients must be nonempty")
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    if norms.ndim != 1 or indices.shape != norms.shape:
        raise ValueError("gradient norms and group indices must be matching vectors")
    if any(leaf.ndim < 1 or leaf.shape[0] != group_count for leaf in group_leaves):
        raise ValueError("group gradient leaves must share the complete group axis")

    group_mean_squared_norms = jp.zeros((group_count,), dtype=jp.float32)
    for leaf in group_leaves:
        axes = tuple(range(1, leaf.ndim))
        group_mean_squared_norms = group_mean_squared_norms + jp.sum(
            jp.square(jp.asarray(leaf, dtype=jp.float32)), axis=axes
        )
    population_mean_squared_norm = jp.asarray(0.0, dtype=jp.float32)
    for leaf in population_leaves:
        population_mean_squared_norm = population_mean_squared_norm + jp.sum(
            jp.square(jp.asarray(leaf, dtype=jp.float32))
        )

    group_counts = jp.zeros((group_count,), dtype=jp.int32).at[indices].add(1)
    group_squared_norm_sums = jp.zeros((group_count,), dtype=jp.float32).at[
        indices
    ].add(jp.square(norms))
    safe_counts = jp.maximum(group_counts, 1).astype(jp.float32)
    group_second_moments = group_squared_norm_sums / safe_counts
    group_variance_traces = jp.maximum(
        group_second_moments - group_mean_squared_norms, 0.0
    )
    total_count = jp.maximum(jp.sum(group_counts), 1).astype(jp.float32)
    group_weights = group_counts.astype(jp.float32) / total_count
    within_group_variance_trace = jp.sum(
        group_weights * group_variance_traces
    )
    between_group_variance_trace = jp.maximum(
        jp.sum(group_weights * group_mean_squared_norms)
        - population_mean_squared_norm,
        0.0,
    )
    total_variance_trace = (
        within_group_variance_trace + between_group_variance_trace
    )
    epsilon = jp.asarray(1e-12, dtype=jp.float32)
    return {
        "group_counts": group_counts,
        "group_mean_norms": jp.sqrt(jp.maximum(group_mean_squared_norms, 0.0)),
        "group_rms_norms": jp.sqrt(jp.maximum(group_second_moments, 0.0)),
        "group_variance_traces": group_variance_traces,
        "group_cancellation_ratios": jp.sqrt(
            group_mean_squared_norms
            / jp.maximum(group_second_moments, epsilon)
        ),
        "group_gradient_noise_scales": (
            group_variance_traces
            / jp.maximum(group_mean_squared_norms, epsilon)
        ),
        "group_esnr": (
            group_counts.astype(jp.float32)
            * group_mean_squared_norms
            / jp.maximum(group_variance_traces, epsilon)
        ),
        "within_group_variance_trace": within_group_variance_trace,
        "between_group_variance_trace": between_group_variance_trace,
        "total_variance_trace": total_variance_trace,
        "within_group_variance_fraction": (
            within_group_variance_trace
            / jp.maximum(total_variance_trace, epsilon)
        ),
        "between_group_variance_fraction": (
            between_group_variance_trace
            / jp.maximum(total_variance_trace, epsilon)
        ),
    }


def per_env_gradient_statistics(
    per_env_grads: Any,
    *,
    max_norm: float | None = None,
) -> dict[str, jax.Array]:
    """Measure raw and effective rollout-gradient population statistics.

    ``max_norm`` applies the same whole-environment clipping and non-finite
    removal used by :func:`aggregate_per_env_gradients`.  The population
    metrics therefore describe the distribution that actually reaches the
    mean update rather than the unbounded raw gradients alone.
    """
    leaves = jax.tree_util.tree_leaves(per_env_grads)
    if not leaves:
        raise ValueError("per_env_grads must contain at least one array")
    if max_norm is not None and max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

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
    scale = jp.ones_like(raw_norm)
    if max_norm is not None:
        scale = jp.minimum(max_norm / jp.maximum(raw_norm, 1e-12), 1.0)
    scale = jp.where(finite_by_env, scale, 0.0)

    effective_squared_norm = jp.zeros_like(squared_norm)
    effective_mean_leaves = []
    for leaf in leaves:
        safe_leaf = jp.where(jp.isfinite(leaf), leaf, 0.0).astype(jp.float32)
        broadcast_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        effective_leaf = safe_leaf * scale.reshape(broadcast_shape)
        reduction_axes = tuple(range(1, leaf.ndim))
        effective_squared_norm = effective_squared_norm + jp.sum(
            jp.square(effective_leaf), axis=reduction_axes
        )
        effective_mean_leaves.append(jp.mean(effective_leaf, axis=0))
    effective_mean = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(per_env_grads), effective_mean_leaves
    )
    effective_norm_by_env = jp.sqrt(effective_squared_norm)
    population_stats = gradient_population_statistics(
        effective_mean, effective_norm_by_env
    )
    return {
        "finite_fraction": jp.mean(finite_by_env.astype(jp.float32)),
        "raw_norm_median": jp.median(raw_norm),
        "raw_norm_max": jp.max(raw_norm),
        "finite_by_env": finite_by_env,
        "raw_norm_by_env": raw_norm,
        "effective_norm_by_env": effective_norm_by_env,
        **population_stats,
    }


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
    stats = per_env_gradient_statistics(per_env_grads, max_norm=max_norm)
    finite_by_env = stats["finite_by_env"]
    raw_norm = stats["raw_norm_by_env"]
    num_envs = raw_norm.shape[0]
    scale = jp.minimum(1.0, max_norm / jp.maximum(raw_norm, 1e-12))
    scale = jp.where(finite_by_env, scale, 0.0)

    def clip_and_average(leaf):
        safe_leaf = jp.where(jp.isfinite(leaf), leaf, 0.0)
        broadcast_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        return jp.mean(safe_leaf * scale.reshape(broadcast_shape), axis=0)

    aggregated = jax.tree_util.tree_map(clip_and_average, per_env_grads)
    return aggregated, stats
