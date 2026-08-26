"""Pure masked capture-point tracking objective."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jp

from src.algorithms.shac.centroidal_objective import pseudo_huber


class CapturePointResult(NamedTuple):
    """Loss and fixed-shape diagnostics for one rollout population."""

    loss: jax.Array
    valid_count: jax.Array
    error: jax.Array
    normalized_error: jax.Array
    valid: jax.Array
    p99_norm: jax.Array


def capture_state_validity(
    active: jax.Array, done: jax.Array
) -> jax.Array:
    """Return H scan-state masks plus the non-reset final-state mask."""
    active_values = jp.asarray(active, dtype=bool)
    done_values = jp.asarray(done, dtype=bool)
    if (
        active_values.ndim != 1
        or done_values.shape != active_values.shape
        or active_values.shape[0] < 1
    ):
        raise ValueError("active/done transition arrays do not align")
    final_valid = active_values[-1] & ~done_values[-1]
    return jp.concatenate((active_values, final_valid[None]), axis=0)


def capture_point_objective(
    actual: jax.Array,
    reference: jax.Array,
    *,
    active: jax.Array,
    standing_height: float,
    delta: float,
) -> CapturePointResult:
    """Track planar robot/reference capture points with a robust dense loss."""
    if not math.isfinite(standing_height) or standing_height <= 0.0:
        raise ValueError("standing_height must be positive and finite")
    if delta != 0.1:
        raise ValueError("registered capture-point objective requires delta=0.1")
    actual_values = jp.asarray(actual)
    reference_values = jp.asarray(reference)
    valid = jp.asarray(active, dtype=bool)
    if (
        actual_values.ndim != 2
        or actual_values.shape[1] != 2
        or reference_values.shape != actual_values.shape
        or valid.shape != (actual_values.shape[0],)
    ):
        raise ValueError("capture-point rollout arrays do not align")

    error = actual_values - reference_values
    normalized_error = error / jp.asarray(
        standing_height, dtype=error.dtype
    )
    valid_count = jp.sum(valid.astype(jp.int32))
    denominator = jp.maximum(valid_count, 1).astype(error.dtype) * 2.0
    loss = jp.sum(
        jp.where(
            valid[:, None],
            pseudo_huber(normalized_error, delta),
            jp.zeros_like(normalized_error),
        )
    ) / denominator

    norm = jp.linalg.norm(normalized_error, axis=-1)
    sorted_norm = jp.sort(jp.where(valid, norm, jp.inf))
    percentile_index = jp.clip(
        jp.ceil(0.99 * valid_count.astype(jp.float32)).astype(jp.int32) - 1,
        0,
        sorted_norm.shape[0] - 1,
    )
    p99_norm = jp.where(
        valid_count > 0,
        sorted_norm[percentile_index],
        jp.asarray(0.0, dtype=error.dtype),
    )
    return CapturePointResult(
        loss=loss,
        valid_count=valid_count,
        error=error,
        normalized_error=normalized_error,
        valid=valid,
        p99_norm=p99_norm,
    )
