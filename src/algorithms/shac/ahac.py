"""Pure Adaptive Horizon Actor-Critic utilities.

The functions in this module intentionally know nothing about the G1
environment or the SHAC trainer.  This keeps the contact-stiffness proxy and
dual update independently testable before they enter a compiled rollout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


ROOT_DOF_COUNT = 6


class HorizonDualUpdate(NamedTuple):
    """Result of one projected AHAC horizon/dual update."""

    horizon: jax.Array
    dual: jax.Array
    valid: jax.Array


def contact_stiffness(
    qfrc_constraint: jax.Array,
    qacc: jax.Array,
) -> jax.Array:
    """Return AHAC's force-over-modified-acceleration contact proxy.

    MuJoCo exposes matching six-dimensional floating-base constraint force and
    acceleration vectors.  Using only these root coordinates avoids mixing
    unrelated joint-space dimensions and is the direct MJX analogue of the
    spatial force/acceleration normalization in the AHAC reference code.
    """

    constraint = jp.asarray(qfrc_constraint)
    acceleration = jp.asarray(qacc)
    if constraint.shape != acceleration.shape:
        raise ValueError("constraint force and acceleration shapes must be matching")
    if not constraint.shape or constraint.shape[-1] < ROOT_DOF_COUNT:
        raise ValueError("contact stiffness inputs require at least six coordinates")
    root_force = constraint[..., :ROOT_DOF_COUNT]
    root_acceleration = acceleration[..., :ROOT_DOF_COUNT]
    modified_acceleration = jp.maximum(jp.abs(root_acceleration), 1.0)
    return jp.linalg.norm(root_force / modified_acceleration, axis=-1)


def active_horizon_mask(horizon: jax.Array, maximum: int) -> jax.Array:
    """Return the static-scan mask for a rounded global AHAC horizon."""

    if maximum < 1:
        raise ValueError("maximum horizon must be positive")
    rounded = jp.floor(jp.asarray(horizon) + 0.5).astype(jp.int32)
    return jp.arange(maximum, dtype=jp.int32) < rounded


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over active leading-axis entries without inactive contamination."""

    values = jp.asarray(values)
    mask = jp.asarray(mask, dtype=jp.bool_)
    if values.shape[0] != mask.shape[0]:
        raise ValueError("mask must match the values leading axis")
    expanded_mask = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
    weights = expanded_mask.astype(values.dtype)
    denominator = jp.maximum(jp.sum(weights), jp.asarray(1.0, values.dtype))
    return jp.sum(jp.where(expanded_mask, values, 0.0)) / denominator


def update_horizon_dual(
    *,
    horizon: jax.Array,
    dual: jax.Array,
    contact_by_step: jax.Array,
    active_mask: jax.Array,
    threshold: float | jax.Array,
    learning_rate: float | jax.Array,
    minimum: int,
    maximum: int,
) -> HorizonDualUpdate:
    """Apply projected dual ascent and the resulting bounded horizon update."""

    dual = jp.asarray(dual)
    contact_by_step = jp.asarray(contact_by_step, dtype=dual.dtype)
    active_mask = jp.asarray(active_mask, dtype=jp.bool_)
    if dual.shape != contact_by_step.shape or dual.shape != active_mask.shape:
        raise ValueError("dual, contact, and active-mask shapes must match")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid AHAC horizon bounds")

    threshold_array = jp.asarray(threshold, dtype=dual.dtype)
    rate = jp.asarray(learning_rate, dtype=dual.dtype)
    violation = contact_by_step - threshold_array
    proposed_dual = jp.maximum(dual + rate * violation, 0.0)
    new_dual = jp.where(active_mask, proposed_dual, 0.0)
    new_horizon = jp.clip(
        jp.asarray(horizon, dtype=dual.dtype) + rate * jp.sum(new_dual),
        minimum,
        maximum,
    )
    valid = (
        jp.all(jp.isfinite(contact_by_step))
        & jp.all(jp.isfinite(dual))
        & jp.isfinite(jp.asarray(horizon))
        & jp.isfinite(threshold_array)
        & (threshold_array > 0.0)
        & jp.isfinite(rate)
        & (rate > 0.0)
        & jp.all(jp.isfinite(new_dual))
        & jp.isfinite(new_horizon)
    )
    return HorizonDualUpdate(
        horizon=new_horizon,
        dual=new_dual,
        valid=valid,
    )


def critic_convergence(
    loss_history: jax.Array,
    tolerance: float | jax.Array,
) -> jax.Array:
    """Whether five consecutive finite critic losses have stabilized."""

    losses = jp.asarray(loss_history)
    if losses.ndim != 1 or losses.shape[0] < 5:
        return jp.asarray(False)
    recent = losses[-5:]
    tolerance_array = jp.asarray(tolerance, dtype=recent.dtype)
    return (
        jp.all(jp.isfinite(recent))
        & jp.isfinite(tolerance_array)
        & (tolerance_array > 0.0)
        & (jp.mean(jp.abs(jp.diff(recent))) < tolerance_array)
    )
