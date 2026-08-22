"""Backward-only compliant-contact utilities for frozen SHAC audits."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


@jax.custom_vjp
def _hard_primal_compliant_vjp(hard: jax.Array, compliant: jax.Array) -> jax.Array:
    return hard


def _hard_primal_compliant_vjp_fwd(hard, compliant):
    del compliant
    return hard, None


def _hard_primal_compliant_vjp_bwd(_residual, cotangent):
    return jnp.zeros_like(cotangent), cotangent


_hard_primal_compliant_vjp.defvjp(
    _hard_primal_compliant_vjp_fwd,
    _hard_primal_compliant_vjp_bwd,
)


def backward_from_compliant(hard: Any, compliant: Any) -> Any:
    """Return the exact hard pytree while routing floating VJPs through compliant."""

    hard_structure = jax.tree_util.tree_structure(hard)
    compliant_structure = jax.tree_util.tree_structure(compliant)
    if hard_structure != compliant_structure:
        raise ValueError("hard and compliant states must share tree structure")

    hard_leaves = jax.tree_util.tree_leaves(hard)
    compliant_leaves = jax.tree_util.tree_leaves(compliant)
    for hard_leaf, compliant_leaf in zip(hard_leaves, compliant_leaves):
        if (
            hard_leaf.shape != compliant_leaf.shape
            or hard_leaf.dtype != compliant_leaf.dtype
        ):
            raise ValueError("hard and compliant leaves must share shape and dtype")

    def combine(hard_leaf, compliant_leaf):
        if jnp.issubdtype(hard_leaf.dtype, jnp.inexact):
            return _hard_primal_compliant_vjp(hard_leaf, compliant_leaf)
        return jax.lax.stop_gradient(hard_leaf)

    return jax.tree_util.tree_map(combine, hard, compliant)


def with_contact_time_constant(model: Any, time_constant: float) -> Any:
    """Return a model copy with only positive-format contact time constants changed."""

    if (
        isinstance(time_constant, bool)
        or not math.isfinite(float(time_constant))
        or float(time_constant) <= 0.0
    ):
        raise ValueError("contact time constant must be finite and positive")
    solref = jnp.asarray(model.geom_solref)
    if solref.ndim != 2 or solref.shape[1] != 2:
        raise ValueError("geom_solref must have shape (ngeom, 2)")
    return model.replace(
        geom_solref=solref.at[:, 0].set(jnp.asarray(time_constant, solref.dtype))
    )
