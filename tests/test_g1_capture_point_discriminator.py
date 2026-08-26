from __future__ import annotations

import pytest

from tools.evaluate_g1_capture_point_discriminator import (
    CaptureMetrics,
    classify_capture_discriminator,
)


def _metrics(p99: float, final: float) -> CaptureMetrics:
    return CaptureMetrics(
        rms=0.2,
        p99=p99,
        final=final,
        component_rms=(0.1, 0.1),
    )


def test_capture_discriminator_requires_assisted_separation() -> None:
    assert classify_capture_discriminator(
        assisted=_metrics(0.4, 0.4),
        e026=_metrics(1.0, 1.0),
        e005=_metrics(1.5, 1.5),
    ) == "capture-signal-valid"
    assert classify_capture_discriminator(
        assisted=_metrics(0.9, 0.4),
        e026=_metrics(1.0, 1.0),
        e005=_metrics(1.5, 1.5),
    ) == "capture-signal-not-discriminating"


def test_capture_discriminator_fails_closed_on_nonfinite_metrics() -> None:
    with pytest.raises(ValueError):
        classify_capture_discriminator(
            assisted=_metrics(float("nan"), 0.4),
            e026=_metrics(1.0, 1.0),
            e005=_metrics(1.5, 1.5),
        )

