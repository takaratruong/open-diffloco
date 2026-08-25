"""Continuous physical-deviation gate for a frozen recovery residual."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np


class DeviationGate(NamedTuple):
    """Fixed lower and upper body-position-error thresholds in metres."""

    lower: float
    upper: float


REGISTERED_DEVIATION_GATE = DeviationGate(lower=0.10, upper=0.20)


def _validate_contract(contract: DeviationGate) -> DeviationGate:
    if not isinstance(contract, DeviationGate):
        raise ValueError("deviation gate contract is required")
    if (
        isinstance(contract.lower, bool)
        or isinstance(contract.upper, bool)
        or not math.isfinite(float(contract.lower))
        or not math.isfinite(float(contract.upper))
        or float(contract.lower) < 0.0
        or float(contract.upper) <= float(contract.lower)
    ):
        raise ValueError("deviation gate thresholds are invalid")
    return DeviationGate(float(contract.lower), float(contract.upper))


def deviation_recovery_gate(
    error: jax.Array,
    contract: DeviationGate = REGISTERED_DEVIATION_GATE,
) -> jax.Array:
    """Return a smooth gate from aligned mean body-position error."""
    contract = _validate_contract(contract)
    values = jp.asarray(error)
    if not isinstance(values, jax.core.Tracer) and not np.isfinite(
        np.asarray(values)
    ).all():
        raise ValueError("body-position error must be finite")
    ratio = jp.clip(
        (values - contract.lower) / (contract.upper - contract.lower),
        0.0,
        1.0,
    )
    smooth = ratio * ratio * (3.0 - 2.0 * ratio)
    gate = jp.where(
        values <= contract.lower,
        jp.zeros_like(smooth),
        jp.where(values >= contract.upper, jp.ones_like(smooth), smooth),
    )
    return jp.where(jp.isfinite(values), gate, jp.asarray(jp.nan, gate.dtype))


def compose_deviation_gated_recovery(
    parent_action: jax.Array,
    residual_action: jax.Array,
    error: jax.Array,
    contract: DeviationGate = REGISTERED_DEVIATION_GATE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compose a frozen parent and recovery residual through the fixed gate."""
    parent = jp.asarray(parent_action)
    residual = jp.asarray(residual_action)
    if parent.shape != residual.shape:
        raise ValueError("parent and residual action shapes must match")
    if parent.ndim < 1:
        raise ValueError("actions must have a coordinate axis")
    gate = deviation_recovery_gate(error, contract)
    if gate.shape != parent.shape[:-1]:
        raise ValueError("error shape must match action batch axes")
    gated_residual = gate[..., None] * residual
    action = jp.where(
        gate[..., None] == 0.0,
        parent,
        parent + gated_residual,
    )
    return action, gated_residual, gate
