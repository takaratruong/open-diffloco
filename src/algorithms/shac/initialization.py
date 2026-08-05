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


def commit_tree_to_local_device(tree):
    """Strongly types and commits JIT inputs before warm-up."""

    def strong_type(value):
        if isinstance(value, jax.Array) and value.weak_type:
            return jp.asarray(value, dtype=value.dtype)
        return value

    tree = jax.tree_util.tree_map(strong_type, tree)
    return jax.device_put(tree, jax.local_devices()[0])


def canonicalize_tree_like(current_tree, template_tree):
    """Matches array dtype, weak typing, device, and sharding to a template."""

    def canonicalize(current, template):
        if isinstance(template, jax.Array):
            value = jp.asarray(current, dtype=template.dtype)
            return jax.device_put(value, template.sharding)
        return current

    return jax.tree_util.tree_map(
        canonicalize,
        current_tree,
        template_tree,
    )
