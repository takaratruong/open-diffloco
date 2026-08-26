"""Classify the preregistered E026/E004/E005 centroidal discriminator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALID = "propulsion-signal-valid"
NOT_DISCRIMINATING = "propulsion-signal-not-discriminating"
INVALID = "invalid-execution"


@dataclass(frozen=True)
class DiscriminatorMetrics:
    """Fixed numerical and derivative gates for one controller."""

    p99_forward_abs: float
    component_rms: tuple[float, float, float, float, float, float]
    values_finite: bool
    action_gradient_finite: bool
    action_gradient_norm: float
    derivative_agreement: bool


def _metrics_valid(metrics: DiscriminatorMetrics) -> bool:
    scalars = np.asarray(
        (metrics.p99_forward_abs, metrics.action_gradient_norm, *metrics.component_rms),
        dtype=np.float64,
    )
    return bool(
        metrics.values_finite
        and metrics.action_gradient_finite
        and metrics.derivative_agreement
        and np.isfinite(scalars).all()
        and metrics.action_gradient_norm > 0.0
        and metrics.p99_forward_abs >= 0.0
        and np.all(scalars[2:] >= 0.0)
    )


def classify_discriminator(
    *,
    assisted: DiscriminatorMetrics,
    e026: DiscriminatorMetrics,
    e005: DiscriminatorMetrics,
) -> str:
    """Apply the immutable separation and derivative gates."""
    if not all(_metrics_valid(value) for value in (assisted, e026, e005)):
        return INVALID
    forward_separates = assisted.p99_forward_abs <= 0.8 * min(
        e026.p99_forward_abs, e005.p99_forward_abs
    )
    other_components_safe = all(
        assisted.component_rms[index]
        <= 1.05
        * max(e026.component_rms[index], e005.component_rms[index])
        for index in range(1, 6)
    )
    return VALID if forward_separates and other_components_safe else NOT_DISCRIMINATING


def sum_external_impulse(
    left_impulse: np.ndarray, right_impulse: np.ndarray
) -> np.ndarray:
    """Sum per-foot six-dimensional impulses without identity dependence."""
    left = np.asarray(left_impulse)
    right = np.asarray(right_impulse)
    if left.shape != right.shape or left.shape[-1:] != (6,):
        raise ValueError("left/right external impulses must have matching (..., 6) shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("external impulses must be finite")
    return left + right


def validate_common_prefix(traces: dict[str, np.ndarray]) -> None:
    """Require the exact three-controller, 106-transition comparison."""
    if set(traces) != {"e026", "e004", "e005"}:
        raise ValueError("common prefix requires e026, e004, and e005")
    for value in traces.values():
        array = np.asarray(value)
        if array.shape != (106, 6) or not np.isfinite(array).all():
            raise ValueError("common prefix must contain 106 finite six-vectors")
