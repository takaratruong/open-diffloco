"""Compact-support recovery correction over an immutable parent actor."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
    current_treatment_frame,
)


class RecoverySupport(NamedTuple):
    """Frozen positive anchors and compact phase/state support."""

    anchors: jax.Array
    radius: jax.Array
    phase_min: int
    phase_max: int
    taper: int


def _smoothstep(values: jax.Array) -> jax.Array:
    clipped = jp.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _state_gate_from_distances(
    distances: jax.Array, radius: jax.Array
) -> jax.Array:
    ratio = distances / radius
    return jp.where(
        distances < radius,
        jp.square(1.0 - jp.square(ratio)),
        jp.zeros_like(distances),
    )


def compact_recovery_gate(
    normalized_observations: jax.Array,
    phases: jax.Array,
    support: RecoverySupport,
) -> jax.Array:
    """Evaluate a smooth gate that is exactly zero outside frozen support."""
    observations = jp.asarray(normalized_observations)
    phases = jp.asarray(phases)
    anchors = jp.asarray(support.anchors, dtype=observations.dtype)
    radius = jp.asarray(support.radius, dtype=observations.dtype)
    if observations.ndim < 1 or anchors.ndim != 2:
        raise ValueError("recovery support observations must have a feature axis")
    if observations.shape[-1] != anchors.shape[-1]:
        raise ValueError("recovery support observation width does not match anchors")
    if phases.shape != observations.shape[:-1]:
        raise ValueError("recovery support phases must match observation batch")

    delta = observations[..., None, :] - anchors
    squared_distance = jp.min(jp.sum(jp.square(delta), axis=-1), axis=-1)
    distance = jp.sqrt(squared_distance)
    state_gate = _state_gate_from_distances(distance, radius)

    phase_values = phases.astype(observations.dtype)
    taper = jp.asarray(support.taper, dtype=observations.dtype)
    left = _smoothstep(
        (phase_values - float(support.phase_min - support.taper)) / taper
    )
    right = _smoothstep(
        (float(support.phase_max + support.taper) - phase_values) / taper
    )
    return state_gate * left * right


def build_recovery_support(
    positive_frames: np.ndarray,
    negative_frames: np.ndarray,
    positive_phases: np.ndarray,
    *,
    taper: int = 4,
    minimum_positive_coverage: int = 20,
) -> tuple[RecoverySupport, dict[str, object]]:
    """Build deterministic half-margin support from positive/negative corpora."""
    positives = np.asarray(positive_frames)
    negatives = np.asarray(negative_frames)
    phases = np.asarray(positive_phases)
    if positives.ndim != 2 or negatives.ndim != 2:
        raise ValueError("recovery support corpora must be rank-two")
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        raise ValueError("recovery support corpora must be nonempty")
    if positives.shape[1] != negatives.shape[1]:
        raise ValueError("positive and negative support width must match")
    if phases.ndim != 1 or phases.shape[0] != positives.shape[0]:
        raise ValueError("positive phases must match positive support rows")
    if not np.issubdtype(phases.dtype, np.integer):
        raise ValueError("positive phases must be integers")
    if not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ValueError("recovery support corpora must be finite")
    if (
        isinstance(taper, bool)
        or not isinstance(taper, int)
        or taper < 1
    ):
        raise ValueError("recovery support taper must be a positive integer")
    if (
        isinstance(minimum_positive_coverage, bool)
        or not isinstance(minimum_positive_coverage, int)
        or minimum_positive_coverage < 1
        or minimum_positive_coverage > positives.shape[0]
    ):
        raise ValueError("minimum positive coverage is invalid")
    if positives.shape[0] < 2:
        raise ValueError("positive support requires at least two rows")

    negative_delta = negatives[:, None, :] - positives[None, :, :]
    negative_distances = np.sqrt(
        np.min(np.sum(np.square(negative_delta), axis=-1), axis=1)
    )
    minimum_negative_distance = float(np.min(negative_distances))
    radius = 0.5 * minimum_negative_distance
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("recovery support radius must be positive and finite")

    positive_delta = positives[:, None, :] - positives[None, :, :]
    positive_pairwise = np.sqrt(np.sum(np.square(positive_delta), axis=-1))
    np.fill_diagonal(positive_pairwise, np.inf)
    leave_one_out = np.min(positive_pairwise, axis=1)
    positive_coverage = int(np.sum(leave_one_out < radius))
    if positive_coverage < minimum_positive_coverage:
        raise ValueError("positive leave-one-out support coverage is insufficient")

    negative_state_gate = np.where(
        negative_distances < radius,
        np.square(1.0 - np.square(negative_distances / radius)),
        0.0,
    )
    if not np.array_equal(negative_state_gate, np.zeros_like(negative_state_gate)):
        raise ValueError("protected negatives are active inside recovery support")

    support = RecoverySupport(
        anchors=jp.asarray(positives, dtype=jp.float32),
        radius=jp.asarray(radius, dtype=jp.float32),
        phase_min=int(np.min(phases)),
        phase_max=int(np.max(phases)),
        taper=taper,
    )
    report = {
        "valid": True,
        "positive_rows": int(positives.shape[0]),
        "protected_negative_rows": int(negatives.shape[0]),
        "frame_width": int(positives.shape[1]),
        "minimum_protected_negative_distance": minimum_negative_distance,
        "radius": radius,
        "phase_min": support.phase_min,
        "phase_max": support.phase_max,
        "taper": taper,
        "positive_leave_one_out_coverage": positive_coverage,
        "minimum_positive_coverage": minimum_positive_coverage,
        "positive_leave_one_out_distances": leave_one_out.tolist(),
        "protected_negative_distances": negative_distances.tolist(),
        "protected_negative_max_gate": float(np.max(negative_state_gate)),
    }
    return support, report


def apply_state_gated_recovery(
    parent_actor,
    residual_actor: PreviewResidualAdapter,
    params: FrozenPreviewResidualParams,
    normalized_observations: jax.Array,
    phases: jax.Array,
    support: RecoverySupport,
    *,
    history_len: int,
    treatment_frame_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply an ELU residual only inside registered phase/state support."""
    _, parent_action, residual_action = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        params,
        normalized_observations,
        history_len=history_len,
        treatment_frame_dim=treatment_frame_dim,
    )
    current_frame = current_treatment_frame(
        normalized_observations,
        history_len=history_len,
        treatment_frame_dim=treatment_frame_dim,
    )
    gate = compact_recovery_gate(current_frame, phases, support)
    gated_residual = gate[..., None] * residual_action
    candidate = jp.where(
        gate[..., None] == 0.0,
        parent_action,
        parent_action + gated_residual,
    )
    return candidate, parent_action, gated_residual, gate
