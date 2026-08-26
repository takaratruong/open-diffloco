"""Pure registered four-step centroidal propulsion objective."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jp

from src.envs.g1_tracking.centroidal_momentum import yaw_frame_momentum


class CentroidalWindowResult(NamedTuple):
    """Loss and fixed-shape diagnostics for one rollout population."""

    loss: jax.Array
    valid_count: jax.Array
    error: jax.Array
    normalized_error: jax.Array
    valid: jax.Array
    p99_forward_abs: jax.Array


def pseudo_huber(value: jax.Array, delta: float) -> jax.Array:
    """Return elementwise pseudo-Huber values at a fixed positive delta."""
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("pseudo-Huber delta must be positive and finite")
    values = jp.asarray(value)
    scale = jp.asarray(delta, dtype=values.dtype)
    return scale * scale * (
        jp.sqrt(1.0 + jp.square(values / scale)) - 1.0
    )


def _window_validity(
    done: jax.Array, active: jax.Array, window: int
) -> jax.Array:
    invalid = jp.asarray(done, dtype=bool) | ~jp.asarray(active, dtype=bool)
    prefix = jp.concatenate(
        (
            jp.zeros((1,), dtype=jp.int32),
            jp.cumsum(invalid.astype(jp.int32)),
        )
    )
    return (prefix[window:] - prefix[:-window]) == 0


def centroidal_window_objective(
    actual: jax.Array,
    reference: jax.Array,
    root_quaternion: jax.Array,
    *,
    done: jax.Array,
    active: jax.Array,
    window: int,
    linear_scale: float,
    angular_scale: float,
    delta: float,
) -> CentroidalWindowResult:
    """Compare four-step robot/reference momentum changes in start-yaw frames."""
    if window != 4:
        raise ValueError("registered centroidal objective requires window=4")
    if delta != 0.1:
        raise ValueError("registered centroidal objective requires delta=0.1")
    if not math.isfinite(linear_scale) or linear_scale <= 0.0:
        raise ValueError("linear momentum scale must be positive and finite")
    if not math.isfinite(angular_scale) or angular_scale <= 0.0:
        raise ValueError("angular momentum scale must be positive and finite")

    actual_values = jp.asarray(actual)
    reference_values = jp.asarray(reference)
    quaternions = jp.asarray(root_quaternion)
    terminal = jp.asarray(done)
    enabled = jp.asarray(active)
    if (
        actual_values.ndim != 2
        or actual_values.shape[1] != 6
        or reference_values.shape != actual_values.shape
        or quaternions.shape != (actual_values.shape[0], 4)
        or terminal.shape != (actual_values.shape[0] - 1,)
        or enabled.shape != terminal.shape
        or terminal.shape[0] < window
    ):
        raise ValueError("centroidal rollout arrays do not align")

    raw_error = (
        actual_values[window:] - actual_values[:-window]
    ) - (
        reference_values[window:] - reference_values[:-window]
    )
    error = jax.vmap(yaw_frame_momentum)(
        raw_error, quaternions[:-window]
    )
    scales = jp.asarray(
        [linear_scale] * 3 + [angular_scale] * 3,
        dtype=error.dtype,
    )
    normalized_error = error / scales
    valid = _window_validity(terminal, enabled, window)
    valid_count = jp.sum(valid.astype(jp.int32))
    denominator = jp.maximum(valid_count, 1).astype(error.dtype) * 6.0
    loss = jp.sum(
        jp.where(
            valid[:, None],
            pseudo_huber(normalized_error, delta),
            jp.zeros_like(normalized_error),
        )
    ) / denominator

    sorted_forward = jp.sort(
        jp.where(valid, jp.abs(normalized_error[:, 0]), jp.inf)
    )
    percentile_index = jp.clip(
        jp.ceil(0.99 * valid_count.astype(jp.float32)).astype(jp.int32) - 1,
        0,
        sorted_forward.shape[0] - 1,
    )
    p99_forward_abs = jp.where(
        valid_count > 0,
        sorted_forward[percentile_index],
        jp.asarray(0.0, dtype=error.dtype),
    )
    return CentroidalWindowResult(
        loss=loss,
        valid_count=valid_count,
        error=error,
        normalized_error=normalized_error,
        valid=valid,
        p99_forward_abs=p99_forward_abs,
    )
