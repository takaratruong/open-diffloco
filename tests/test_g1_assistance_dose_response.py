from __future__ import annotations

import pytest

from src.evaluation.g1_assistance_dose_response import (
    ASSISTANCE_SCALES,
    PHASES,
    classify_threshold_trajectory,
    required_scale,
)


def _records(*completed_scales: float) -> list[dict[str, object]]:
    return [
        {
            "scale": scale,
            "completed_reference_suffix": scale in completed_scales,
            "valid": True,
        }
        for scale in ASSISTANCE_SCALES
    ]


def _checkpoint(label: str, values: tuple[float | None, ...]) -> dict:
    return {
        "label": label,
        "required_scales": {
            str(phase): value for phase, value in zip(PHASES, values, strict=True)
        },
    }


def test_required_scale_returns_smallest_completion() -> None:
    assert required_scale(
        _records(0.1, 0.25, 0.5, 1.0), scales=ASSISTANCE_SCALES
    ) == pytest.approx(0.1)


def test_required_scale_returns_none_when_full_assistance_fails() -> None:
    assert required_scale(_records(), scales=ASSISTANCE_SCALES) is None


def test_required_scale_rejects_incomplete_or_invalid_grid() -> None:
    with pytest.raises(ValueError, match="exact registered scale grid"):
        required_scale(_records()[:-1], scales=ASSISTANCE_SCALES)

    records = _records(1.0)
    records[-1]["valid"] = False
    with pytest.raises(ValueError, match="invalid dose-response record"):
        required_scale(records, scales=ASSISTANCE_SCALES)


def test_classify_threshold_trajectory_detects_monotonic_decrease() -> None:
    checkpoints = [
        _checkpoint("parent", (1.0, None, 1.0, None, 0.5)),
        _checkpoint("midpoint", (0.5, None, 1.0, 1.0, 0.5)),
        _checkpoint("assistance_end", (0.5, 1.0, 0.5, 1.0, 0.25)),
        _checkpoint("final", (0.25, 1.0, 0.5, 0.5, 0.25)),
    ]
    assert (
        classify_threshold_trajectory(checkpoints)
        == "assistance-requirement-decreases"
    )


def test_classify_threshold_trajectory_detects_mixed_transfer() -> None:
    checkpoints = [
        _checkpoint("parent", (1.0, 1.0, 1.0, 1.0, 1.0)),
        _checkpoint("midpoint", (0.25, 1.0, 1.0, 1.0, 1.0)),
        _checkpoint("assistance_end", (0.5, 1.0, 1.0, 1.0, 1.0)),
        _checkpoint("final", (0.5, 1.0, 1.0, 1.0, 1.0)),
    ]
    assert (
        classify_threshold_trajectory(checkpoints)
        == "mixed-threshold-transfer"
    )


def test_classify_threshold_trajectory_detects_no_transfer() -> None:
    checkpoints = [
        _checkpoint("parent", (0.5, 1.0, None, 0.5, 1.0)),
        _checkpoint("midpoint", (0.5, 1.0, None, 1.0, 1.0)),
        _checkpoint("assistance_end", (1.0, None, None, 1.0, 1.0)),
        _checkpoint("final", (1.0, None, None, 1.0, None)),
    ]
    assert (
        classify_threshold_trajectory(checkpoints)
        == "assistance-dependent-no-transfer"
    )


def test_classify_threshold_trajectory_rejects_wrong_checkpoint_order() -> None:
    checkpoints = [
        _checkpoint("parent", (1.0,) * 5),
        _checkpoint("final", (0.5,) * 5),
    ]
    with pytest.raises(ValueError, match="checkpoint labels"):
        classify_threshold_trajectory(checkpoints)
