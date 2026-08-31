"""Pure registered four-step centroidal propulsion objective."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from src.envs.g1_tracking.centroidal_momentum import yaw_frame_momentum


class CentroidalWindowResult(NamedTuple):
    """Loss and fixed-shape diagnostics for one rollout population."""

    loss: jax.Array
    valid_count: jax.Array
    error: jax.Array
    normalized_error: jax.Array
    valid: jax.Array
    p99_forward_abs: jax.Array


class SupportAwareImpulseTarget(NamedTuple):
    """Dense phase tables for one primary and one held-out target replica."""

    primary_by_phase: jax.Array
    duplicate_by_phase: jax.Array
    phase_valid: jax.Array
    component_scales: jax.Array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_support_aware_impulse_target(
    path: str | Path,
    *,
    expected_sha256: str,
    reference_length: int,
    expected_component_scales: np.ndarray,
) -> tuple[SupportAwareImpulseTarget, dict[str, object]]:
    """Load the immutable E002 target into phase-indexed dense tables."""
    resolved = Path(path).expanduser().resolve()
    if (
        not resolved.is_file()
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or _sha256_file(resolved) != expected_sha256
    ):
        raise ValueError("support-aware target artifact hash mismatch")
    required = {
        "window_start_transitions",
        "window_end_transitions_inclusive",
        "component_scales",
        "support_projected_full_a",
        "support_projected_full_b",
        "support_projection_feasible_full_a",
        "support_projection_feasible_full_b",
    }
    with np.load(resolved, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("support-aware target artifact keys do not match")
        arrays = {name: np.asarray(archive[name]) for name in required}

    starts = np.asarray(arrays["window_start_transitions"], dtype=np.int64)
    ends = np.asarray(arrays["window_end_transitions_inclusive"], dtype=np.int64)
    scales = np.asarray(arrays["component_scales"], dtype=np.float64)
    expected_scales = np.asarray(expected_component_scales, dtype=np.float64)
    primary = np.asarray(arrays["support_projected_full_a"], dtype=np.float64)
    duplicate = np.asarray(arrays["support_projected_full_b"], dtype=np.float64)
    feasible_primary = np.asarray(
        arrays["support_projection_feasible_full_a"], dtype=bool
    )
    feasible_duplicate = np.asarray(
        arrays["support_projection_feasible_full_b"], dtype=bool
    )
    if (
        isinstance(reference_length, bool)
        or not isinstance(reference_length, int)
        or reference_length <= 125
        or starts.shape != (125,)
        or not np.array_equal(starts, np.arange(1, 126, dtype=np.int64))
        or not np.array_equal(ends, starts + 3)
        or scales.shape != (6,)
        or expected_scales.shape != (6,)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
        or not np.allclose(scales, expected_scales, rtol=0.0, atol=1e-12)
        or primary.shape != (125, 6)
        or duplicate.shape != (125, 6)
        or not np.isfinite(primary).all()
        or not np.isfinite(duplicate).all()
        or feasible_primary.shape != (125,)
        or feasible_duplicate.shape != (125,)
        or not feasible_primary.all()
        or not feasible_duplicate.all()
    ):
        raise ValueError("support-aware target artifact contract is invalid")

    primary_dense = np.zeros((reference_length, 6), dtype=np.float64)
    duplicate_dense = np.zeros((reference_length, 6), dtype=np.float64)
    phase_valid = np.zeros((reference_length,), dtype=bool)
    primary_dense[starts] = primary
    duplicate_dense[starts] = duplicate
    phase_valid[starts] = True
    target = SupportAwareImpulseTarget(
        primary_by_phase=jp.asarray(primary_dense),
        duplicate_by_phase=jp.asarray(duplicate_dense),
        phase_valid=jp.asarray(phase_valid),
        component_scales=jp.asarray(scales),
    )
    return target, {
        "valid": True,
        "artifact_path": str(resolved),
        "artifact_sha256": expected_sha256,
        "primary_replica": "full-a",
        "heldout_replica": "full-b",
        "window": 4,
        "phase_first": 1,
        "phase_last": 125,
        "window_count": 125,
        "component_scales": scales.tolist(),
    }


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


def support_aware_impulse_objective(
    actual: jax.Array,
    root_quaternion: jax.Array,
    phases: jax.Array,
    target_by_phase: jax.Array,
    target_phase_valid: jax.Array,
    *,
    done: jax.Array,
    active: jax.Array,
    gravity_impulse: jax.Array,
    component_scales: jax.Array,
    window: int,
    reference_stride: int,
    delta: float,
) -> CentroidalWindowResult:
    """Match phase-indexed non-gravity impulse in each start-yaw frame."""
    if window != 4:
        raise ValueError("registered support-aware objective requires window=4")
    if reference_stride != 1:
        raise ValueError("registered support-aware objective requires stride=1")
    if delta != 0.1:
        raise ValueError("registered support-aware objective requires delta=0.1")

    actual_values = jp.asarray(actual)
    quaternions = jp.asarray(root_quaternion)
    phase_values = jp.asarray(phases, dtype=jp.int32)
    targets = jp.asarray(target_by_phase, dtype=actual_values.dtype)
    target_valid = jp.asarray(target_phase_valid, dtype=bool)
    terminal = jp.asarray(done, dtype=bool)
    enabled = jp.asarray(active, dtype=bool)
    gravity = jp.asarray(gravity_impulse, dtype=actual_values.dtype)
    scales = jp.asarray(component_scales, dtype=actual_values.dtype)
    transition_count = actual_values.shape[0] - 1
    if (
        actual_values.ndim != 2
        or actual_values.shape[1] != 6
        or quaternions.shape != (actual_values.shape[0], 4)
        or phase_values.shape != (transition_count,)
        or terminal.shape != (transition_count,)
        or enabled.shape != terminal.shape
        or transition_count < window
        or targets.ndim != 2
        or targets.shape[1] != 6
        or target_valid.shape != (targets.shape[0],)
        or gravity.shape != (3,)
        or scales.shape != (6,)
    ):
        raise ValueError("support-aware rollout arrays do not align")

    window_count = transition_count - window + 1
    start_phases = phase_values[:window_count]
    phase_contiguous = jp.ones((window_count,), dtype=bool)
    for offset in range(window):
        phase_contiguous = phase_contiguous & (
            phase_values[offset : offset + window_count]
            == start_phases + offset * reference_stride
        )
    phase_in_bounds = (start_phases >= 0) & (start_phases < targets.shape[0])
    safe_phases = jp.clip(start_phases, 0, targets.shape[0] - 1)
    selected_target = targets[safe_phases]

    contact_impulse = actual_values[window:] - actual_values[:-window]
    contact_impulse = contact_impulse.at[:, :3].add(-gravity)
    raw_error = contact_impulse - selected_target
    error = jax.vmap(yaw_frame_momentum)(raw_error, quaternions[:-window])
    normalized_error = error / scales
    valid = (
        _window_validity(terminal, enabled, window)
        & phase_contiguous
        & phase_in_bounds
        & target_valid[safe_phases]
    )
    valid_count = jp.sum(valid.astype(jp.int32))
    denominator = jp.maximum(valid_count, 1).astype(error.dtype) * 6.0
    loss = (
        jp.sum(
            jp.where(
                valid[:, None],
                pseudo_huber(normalized_error, delta),
                jp.zeros_like(normalized_error),
            )
        )
        / denominator
    )

    sorted_forward = jp.sort(jp.where(valid, jp.abs(normalized_error[:, 0]), jp.inf))
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
