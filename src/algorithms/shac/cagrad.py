"""Pure JAX phase-bin reduction and CAGrad gradient combination."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jp

from src.algorithms.shac.gradients import per_env_gradient_statistics

PyTree = Any


class PhaseGradientAccumulator(NamedTuple):
    """Finite-aware per-bin gradient sums and contributor counts."""

    sums: PyTree
    finite_counts: PyTree
    env_counts: jax.Array


class CAGradResult(NamedTuple):
    """Combined gradient and fixed-solver diagnostics."""

    combined_gradient: PyTree
    weights: jax.Array
    gram_matrix: jax.Array
    cosine_matrix: jax.Array
    objective: jax.Array
    dual_gap: jax.Array
    uniform_combined_cosine: jax.Array
    valid: jax.Array


def accumulate_phase_gradients(
    per_env_gradients: PyTree,
    phases: jax.Array,
    *,
    phase_count: int,
    bin_count: int,
    per_env_max_norm: float | None = None,
) -> PhaseGradientAccumulator:
    """Accumulate finite gradient elements by reference-phase bin."""
    if phase_count < 1 or bin_count < 1:
        raise ValueError("phase_count and bin_count must be positive")
    phases = jp.asarray(phases, dtype=jp.int32)
    leaves = jax.tree_util.tree_leaves(per_env_gradients)
    if not leaves or phases.ndim != 1:
        raise ValueError("gradient tree and phase vector are required")
    num_envs = phases.shape[0]
    if any(
        leaf.ndim < 1 or leaf.shape[0] != num_envs for leaf in leaves
    ):
        raise ValueError("all gradient leaves must share the phase axis")

    finite_by_env = None
    if per_env_max_norm is not None:
        if per_env_max_norm <= 0.0:
            raise ValueError("per-env gradient max norm must be positive")
        stats = per_env_gradient_statistics(per_env_gradients)
        finite_by_env = stats["finite_by_env"]
        raw_norm = stats["raw_norm_by_env"]
        scale = jp.minimum(
            1.0, per_env_max_norm / jp.maximum(raw_norm, 1e-12)
        )
        scale = jp.where(finite_by_env, scale, 0.0)

        def sanitize_and_clip(leaf):
            broadcast_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
            return jp.where(jp.isfinite(leaf), leaf, 0.0) * scale.reshape(
                broadcast_shape
            )

        per_env_gradients = jax.tree_util.tree_map(
            sanitize_and_clip, per_env_gradients
        )

    bins = jp.minimum((phases * bin_count) // phase_count, bin_count - 1)

    def finite_sum(leaf):
        finite = jp.isfinite(leaf)
        shape = (bin_count,) + leaf.shape[1:]
        return jp.zeros(shape, dtype=leaf.dtype).at[bins].add(
            jp.where(finite, leaf, 0.0)
        )

    def finite_count(leaf):
        finite = (
            jp.broadcast_to(
                finite_by_env.reshape(
                    (num_envs,) + (1,) * (leaf.ndim - 1)
                ),
                leaf.shape,
            )
            if finite_by_env is not None
            else jp.isfinite(leaf)
        )
        shape = (bin_count,) + leaf.shape[1:]
        return jp.zeros(shape, dtype=jp.int32).at[bins].add(
            finite.astype(jp.int32)
        )

    sums = jax.tree_util.tree_map(finite_sum, per_env_gradients)
    finite_counts = jax.tree_util.tree_map(finite_count, per_env_gradients)
    env_contributors = (
        finite_by_env.astype(jp.int32)
        if finite_by_env is not None
        else jp.ones((num_envs,), dtype=jp.int32)
    )
    env_counts = jp.zeros((bin_count,), dtype=jp.int32).at[bins].add(
        env_contributors
    )
    return PhaseGradientAccumulator(sums, finite_counts, env_counts)


def merge_phase_gradient_accumulators(
    first: PhaseGradientAccumulator,
    second: PhaseGradientAccumulator,
) -> PhaseGradientAccumulator:
    """Add corresponding fields from two population shards."""
    return PhaseGradientAccumulator(
        sums=jax.tree_util.tree_map(jp.add, first.sums, second.sums),
        finite_counts=jax.tree_util.tree_map(
            jp.add, first.finite_counts, second.finite_counts
        ),
        env_counts=first.env_counts + second.env_counts,
    )


def finalize_phase_gradients(
    accumulator: PhaseGradientAccumulator,
) -> tuple[PyTree, jax.Array, jax.Array]:
    """Form finite-aware task means and report occupied phase bins."""

    def finite_mean(sums, counts):
        return jp.where(counts > 0, sums / jp.maximum(counts, 1), 0.0)

    task_gradients = jax.tree_util.tree_map(
        finite_mean, accumulator.sums, accumulator.finite_counts
    )
    valid = jp.all(accumulator.env_counts > 0)
    return task_gradients, accumulator.env_counts, valid


def combine_cagrad(
    task_gradients: PyTree,
    *,
    alpha: float,
    iterations: int,
    eps: float = 1e-8,
) -> CAGradResult:
    """Combine task gradients with fixed-iteration CAGrad."""
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    leaves = jax.tree_util.tree_leaves(task_gradients)
    if not leaves or any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("task gradients must have a leading task axis")
    task_count = leaves[0].shape[0]
    if task_count < 1 or any(leaf.shape[0] != task_count for leaf in leaves):
        raise ValueError("all gradient leaves must share the task axis")

    flat_gradients = jp.concatenate(
        [leaf.reshape((task_count, -1)) for leaf in leaves], axis=1
    )
    gram_matrix = flat_gradients @ flat_gradients.T
    uniform = jp.full(
        (task_count,), 1.0 / task_count, dtype=gram_matrix.dtype
    )
    gram_uniform = gram_matrix @ uniform
    uniform_norm = jp.sqrt(jp.maximum(uniform @ gram_uniform, 0.0))
    coefficient = jp.asarray(alpha, gram_matrix.dtype) * uniform_norm

    def objective_gradient(weights):
        gram_weights = gram_matrix @ weights
        weighted_norm = jp.sqrt(
            jp.maximum(weights @ gram_weights, 0.0) + eps
        )
        return gram_uniform + coefficient * gram_weights / weighted_norm

    def frank_wolfe_step(index, weights):
        gradient = objective_gradient(weights)
        vertex = jax.nn.one_hot(
            jp.argmin(gradient), task_count, dtype=weights.dtype
        )
        gamma = jp.asarray(2.0, weights.dtype) / (
            index.astype(weights.dtype) + 2.0
        )
        return weights + gamma * (vertex - weights)

    weights = jax.lax.fori_loop(
        0, iterations, frank_wolfe_step, uniform
    )
    gram_weights = gram_matrix @ weights
    weighted_squared_norm = jp.maximum(weights @ gram_weights, 0.0)
    weighted_norm = jp.sqrt(weighted_squared_norm)
    objective = (
        weights @ gram_uniform
        + coefficient * jp.sqrt(weighted_squared_norm + eps)
    )
    objective_grad = objective_gradient(weights)
    dual_gap = weights @ objective_grad - jp.min(objective_grad)

    uniform_gradient = jax.tree_util.tree_map(
        lambda leaf: jp.mean(leaf, axis=0), task_gradients
    )
    weighted_gradient = jax.tree_util.tree_map(
        lambda leaf: jp.tensordot(weights, leaf, axes=1), task_gradients
    )
    scale = coefficient / (weighted_norm + eps)
    denominator = 1.0 + alpha**2
    combined_gradient = jax.tree_util.tree_map(
        lambda mean, weighted: (mean + scale * weighted) / denominator,
        uniform_gradient,
        weighted_gradient,
    )

    task_norms = jp.sqrt(jp.maximum(jp.diag(gram_matrix), 0.0))
    cosine_matrix = gram_matrix / (
        task_norms[:, None] * task_norms[None, :] + eps
    )
    uniform_flat = jp.mean(flat_gradients, axis=0)
    combined_flat = jp.concatenate(
        [leaf.reshape(-1) for leaf in jax.tree_util.tree_leaves(combined_gradient)]
    )
    uniform_combined_cosine = jp.vdot(uniform_flat, combined_flat) / (
        jp.linalg.norm(uniform_flat) * jp.linalg.norm(combined_flat) + eps
    )

    combined_finite = jp.all(
        jp.stack(
            [jp.all(jp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(combined_gradient)]
        )
    )
    simplex_valid = (
        jp.all(weights >= 0.0)
        & jp.all(weights <= 1.0)
        & (jp.abs(jp.sum(weights) - 1.0) <= 1e-6)
    )
    valid = (
        jp.all(jp.isfinite(flat_gradients))
        & jp.all(jp.isfinite(gram_matrix))
        & jp.all(jp.isfinite(cosine_matrix))
        & jp.all(jp.isfinite(weights))
        & jp.isfinite(objective)
        & jp.isfinite(dual_gap)
        & jp.isfinite(uniform_combined_cosine)
        & combined_finite
        & simplex_valid
    )
    return CAGradResult(
        combined_gradient=combined_gradient,
        weights=weights,
        gram_matrix=gram_matrix,
        cosine_matrix=cosine_matrix,
        objective=objective,
        dual_gap=dual_gap,
        uniform_combined_cosine=uniform_combined_cosine,
        valid=valid,
    )
