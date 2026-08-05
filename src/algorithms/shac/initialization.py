"""Canonical JIT input initialization for SHAC."""

import jax
import jax.numpy as jp


def canonicalize_normalizer_dtype(state, dtype):
    """Moves all normalizer statistics to the rollout observation dtype."""
    dtype = jp.dtype(dtype)
    return jax.tree_util.tree_map(
        lambda value: jp.asarray(value, dtype=dtype),
        state,
    )


def canonicalize_step_dtype(step):
    """Returns a non-weak device scalar for a stable train-step signature."""
    dtype = jp.int64 if jax.config.x64_enabled else jp.int32
    return jp.asarray(step, dtype=dtype)
