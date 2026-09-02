"""Pure pytree operations for memory-bounded SHAC microbatches."""

from typing import Any

import jax
import jax.numpy as jp


PyTree = Any


def reshape_population(
    tree: PyTree,
    *,
    accumulation_steps: int,
    microbatch_size: int,
) -> PyTree:
    """Splits one ordered population into equal leading-axis shards."""
    effective_size = accumulation_steps * microbatch_size
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("population tree must contain at least one array")
    if any(value.ndim < 1 or value.shape[0] != effective_size for value in leaves):
        raise ValueError(
            "every population leaf must use the effective leading size"
        )
    return jax.tree_util.tree_map(
        lambda value: value.reshape(
            (accumulation_steps, microbatch_size) + value.shape[1:]
        ),
        tree,
    )


def flatten_population(tree: PyTree) -> PyTree:
    """Merges shard and microbatch axes without changing population order."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("sharded population tree must contain an array")
    if any(value.ndim < 2 for value in leaves):
        raise ValueError("every sharded population leaf needs two leading axes")
    leading_shape = leaves[0].shape[:2]
    if any(value.shape[:2] != leading_shape for value in leaves[1:]):
        raise ValueError("all sharded population leaves must share leading axes")
    return jax.tree_util.tree_map(
        lambda value: value.reshape(
            (value.shape[0] * value.shape[1],) + value.shape[2:]
        ),
        tree,
    )


def mean_shard_trees(shard_trees: PyTree) -> PyTree:
    """Averages equal-sized shard aggregates over their leading axis."""
    leaves = jax.tree_util.tree_leaves(shard_trees)
    if not leaves or any(value.ndim < 1 for value in leaves):
        raise ValueError("shard tree leaves need a leading shard axis")
    shard_count = leaves[0].shape[0]
    if any(value.shape[0] != shard_count for value in leaves[1:]):
        raise ValueError("all shard tree leaves must share the shard axis")
    return jax.tree_util.tree_map(
        lambda value: jp.mean(value, axis=0), shard_trees
    )


def summarize_shard_stats(shard_stats: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Reduces per-shard gradient diagnostics over the full population."""
    finite_by_env = shard_stats["finite_by_env"].reshape(-1)
    raw_norm_by_env = shard_stats["raw_norm_by_env"].reshape(-1)
    effective_norm_by_env = shard_stats.get(
        "effective_norm_by_env", shard_stats["raw_norm_by_env"]
    ).reshape(-1)
    if not (
        finite_by_env.shape
        == raw_norm_by_env.shape
        == effective_norm_by_env.shape
    ):
        raise ValueError(
            "finite flags and gradient norms must share population shape"
        )
    return {
        "finite_fraction": jp.mean(finite_by_env.astype(jp.float32)),
        "raw_norm_median": jp.median(raw_norm_by_env),
        "raw_norm_max": jp.max(raw_norm_by_env),
        "finite_by_env": finite_by_env,
        "raw_norm_by_env": raw_norm_by_env,
        "effective_norm_by_env": effective_norm_by_env,
    }
