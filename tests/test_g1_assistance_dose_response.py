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


def test_worker_parser_freezes_the_registered_grid() -> None:
    from tools.evaluate_g1_assistance_dose_response import build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint",
            "/artifacts/checkpoint.pkl",
            "--checkpoint-label",
            "parent",
            "--checkpoint-sha256",
            "a" * 64,
            "--reference-path",
            "/artifacts/reference.npz",
            "--code-commit",
            "b" * 40,
            "--physical-gpu-uuid",
            "GPU-test",
            "--output",
            "/evidence/worker.json",
        ]
    )
    assert args.phases == PHASES
    assert args.assistance_scales == ASSISTANCE_SCALES
    assert args.seed == 0
    assert args.solver_profile == "g1-4x5"


def test_registered_conditions_cover_each_phase_scale_once() -> None:
    from tools.evaluate_g1_assistance_dose_response import registered_conditions

    assert registered_conditions() == tuple(
        (phase, scale) for phase in PHASES for scale in ASSISTANCE_SCALES
    )


def test_condition_validity_requires_safe_wrench_and_exact_zero() -> None:
    from tools.evaluate_g1_assistance_dose_response import condition_is_valid

    summary = {
        "steps": 10,
        "remaining_reference_transitions": 20,
        "completed_reference_suffix": False,
        "terminal": True,
    }
    telemetry = {
        "steps": 10,
        "finite": True,
        "force_cap_compliant": True,
        "torque_cap_compliant": True,
        "exact_zero_wrench": True,
    }
    assert condition_is_valid(summary, telemetry, scale=0.0)
    assert not condition_is_valid(
        summary, {**telemetry, "exact_zero_wrench": False}, scale=0.0
    )
    assert condition_is_valid(
        summary, {**telemetry, "exact_zero_wrench": False}, scale=0.1
    )


def test_worker_document_rejects_missing_condition() -> None:
    from tools.evaluate_g1_assistance_dose_response import build_worker_document

    conditions = [
        {
            "phase": phase,
            "scale": scale,
            "valid": True,
            "completed_reference_suffix": False,
        }
        for phase, scale in tuple(
            (phase, scale)
            for phase in PHASES
            for scale in ASSISTANCE_SCALES
        )[:-1]
    ]
    with pytest.raises(ValueError, match="exact phase/scale grid"):
        build_worker_document(
            checkpoint_label="parent",
            provenance={"checkpoint_sha256": "a" * 64},
            conditions=conditions,
            device={"platform": "gpu", "device_count": 1},
        )
