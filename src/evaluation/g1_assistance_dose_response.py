"""Pure contract and analysis for the fixed G1 assistance dose response."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


PHASES = (0, 100, 200, 300, 400)
ASSISTANCE_SCALES = (0.0, 0.1, 0.25, 0.5, 1.0)
CHECKPOINT_LABELS = ("parent", "midpoint", "assistance_end", "final")


def condition_is_valid(
    summary: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scale: float,
) -> bool:
    """Validate one rollout and its bounded wrench telemetry."""
    try:
        steps = int(summary["steps"])
        remaining = int(summary["remaining_reference_transitions"])
        terminal = bool(summary["terminal"])
        completed = bool(summary["completed_reference_suffix"])
        telemetry_steps = int(telemetry["steps"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_completion = steps == remaining and not terminal
    return bool(
        1 <= steps <= remaining
        and telemetry_steps == steps
        and completed == expected_completion
        and telemetry.get("finite") is True
        and telemetry.get("force_cap_compliant") is True
        and telemetry.get("torque_cap_compliant") is True
        and (scale != 0.0 or telemetry.get("exact_zero_wrench") is True)
    )


def required_scale(
    records: Sequence[Mapping[str, Any]], *, scales: Sequence[float]
) -> float | None:
    """Return the smallest registered scale completing one phase suffix."""
    expected = tuple(float(scale) for scale in scales)
    if tuple(sorted(expected)) != expected or len(set(expected)) != len(expected):
        raise ValueError("scales must be unique and increasing")
    by_scale = {float(record["scale"]): record for record in records}
    if len(by_scale) != len(records) or set(by_scale) != set(expected):
        raise ValueError("records must cover the exact registered scale grid")
    for scale in expected:
        record = by_scale[scale]
        if record.get("valid") is not True:
            raise ValueError(f"invalid dose-response record at scale {scale}")
        completed = record.get("completed_reference_suffix")
        if not isinstance(completed, bool):
            raise ValueError("completion flags must be boolean")
        if completed:
            return scale
    return None


def _validated_thresholds(
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[int, list[float]]:
    labels = tuple(checkpoint.get("label") for checkpoint in checkpoints)
    if labels != CHECKPOINT_LABELS:
        raise ValueError(f"checkpoint labels must be {CHECKPOINT_LABELS}")
    trajectories = {phase: [] for phase in PHASES}
    expected_phase_keys = {str(phase) for phase in PHASES}
    for checkpoint in checkpoints:
        values = checkpoint.get("required_scales")
        if not isinstance(values, Mapping) or set(values) != expected_phase_keys:
            raise ValueError("required scales must contain the exact phase grid")
        for phase in PHASES:
            value = values[str(phase)]
            if value is None:
                trajectories[phase].append(math.inf)
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric not in ASSISTANCE_SCALES:
                raise ValueError("required scale is outside the registered grid")
            trajectories[phase].append(numeric)
    return trajectories


def classify_threshold_trajectory(
    checkpoints: Sequence[Mapping[str, Any]],
) -> str:
    """Classify whether assistance requirements fall through training time."""
    trajectories = _validated_thresholds(checkpoints)
    monotonic = all(
        all(later <= earlier for earlier, later in zip(values, values[1:]))
        for values in trajectories.values()
    )
    strict = any(
        any(later < earlier for earlier, later in zip(values, values[1:]))
        for values in trajectories.values()
    )
    if monotonic and strict:
        return "assistance-requirement-decreases"
    any_reduction_from_parent = any(
        any(value < values[0] for value in values[1:])
        for values in trajectories.values()
    )
    if any_reduction_from_parent:
        return "mixed-threshold-transfer"
    return "assistance-dependent-no-transfer"
