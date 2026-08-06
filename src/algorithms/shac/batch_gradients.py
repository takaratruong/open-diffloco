"""Memory-safe gradient aggregation and geometry diagnostics."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jp


PyTree = Any


class TreeMoments(NamedTuple):
    """Online tree mean and Welford sum of squared deviations."""

    count: jax.Array
    mean: PyTree
    mean_squared_norm: jax.Array


class TreeMomentSummary(NamedTuple):
    """Population geometry derived from online gradient moments."""

    mean_norm: jax.Array
    trace_variance: jax.Array
    snr: jax.Array


def batch_mean_value_and_grad(
    single_loss: Callable, params: PyTree, *batched_inputs: jax.Array
) -> tuple[jax.Array, PyTree]:
    """Differentiates a batch mean without materializing per-example grads."""
    if not batched_inputs:
        raise ValueError("at least one batched input is required")
    batch_size = batched_inputs[0].shape[0]
    if any(value.shape[0] != batch_size for value in batched_inputs[1:]):
        raise ValueError("all batched inputs must share their leading size")

    def mean_loss(value):
        losses = jax.vmap(
            single_loss,
            in_axes=(None,) + (0,) * len(batched_inputs),
        )(value, *batched_inputs)
        return jp.mean(losses)

    return jax.value_and_grad(mean_loss)(params)


def tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    """Returns the Euclidean inner product across matching pytree leaves."""
    products = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda left_leaf, right_leaf: jp.vdot(
                left_leaf, right_leaf
            ).real,
            left,
            right,
        )
    )
    if not products:
        return jp.array(0.0)
    return sum(products[1:], products[0])


def tree_norm(tree: PyTree) -> jax.Array:
    """Returns the aggregate Euclidean norm of a pytree."""
    return jp.sqrt(jp.maximum(tree_dot(tree, tree), 0.0))


def tree_cosine(left: PyTree, right: PyTree) -> jax.Array:
    """Returns a finite cosine, defining zero-vector comparisons as zero."""
    numerator = tree_dot(left, right)
    denominator = tree_norm(left) * tree_norm(right)
    tiny = jp.finfo(denominator.dtype).tiny
    return jp.where(
        denominator > 0.0,
        numerator / jp.maximum(denominator, tiny),
        jp.zeros_like(numerator),
    )


def init_tree_moments(tree: PyTree) -> TreeMoments:
    """Initializes an empty accumulator with the supplied tree structure."""
    zeros = jax.tree_util.tree_map(jp.zeros_like, tree)
    return TreeMoments(
        count=jp.array(0, dtype=jp.int32),
        mean=zeros,
        mean_squared_norm=jp.array(0.0, dtype=tree_dot(tree, tree).dtype),
    )


def update_tree_moments(
    state: TreeMoments, tree: PyTree
) -> TreeMoments:
    """Adds one gradient tree using Welford's numerically stable update."""
    next_count = state.count + 1
    delta = jax.tree_util.tree_map(
        lambda value, mean: value - mean, tree, state.mean
    )
    next_mean = jax.tree_util.tree_map(
        lambda mean, difference: mean + difference / next_count,
        state.mean,
        delta,
    )
    delta_after = jax.tree_util.tree_map(
        lambda value, mean: value - mean, tree, next_mean
    )
    squared_deviation_sum = (
        state.mean_squared_norm + tree_dot(delta, delta_after)
    )
    return TreeMoments(
        count=next_count,
        mean=next_mean,
        mean_squared_norm=squared_deviation_sum,
    )


def finalize_tree_moments(state: TreeMoments) -> TreeMomentSummary:
    """Returns population trace variance and aggregate gradient SNR."""
    if int(state.count) == 0:
        raise ValueError("cannot finalize empty tree moments")
    mean_norm = tree_norm(state.mean)
    trace_variance = jp.maximum(
        state.mean_squared_norm / state.count, 0.0
    )
    standard_deviation = jp.sqrt(trace_variance)
    snr = jp.where(
        standard_deviation > 0.0,
        mean_norm / standard_deviation,
        jp.where(
            mean_norm > 0.0,
            jp.array(jp.inf, dtype=mean_norm.dtype),
            jp.zeros_like(mean_norm),
        ),
    )
    return TreeMomentSummary(
        mean_norm=mean_norm,
        trace_variance=trace_variance,
        snr=snr,
    )
