from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_g1_counterfactual_wrench_feasibility import (
    FEASIBILITY_PHASES,
    FEASIBILITY_THRESHOLD,
    bounded_damped_projection,
    build_parser,
    classify_feasibility,
    leg_residual_bounds,
    publish_feasibility_artifacts,
    validate_feasibility_artifacts,
)


def _rows(
    *,
    phases: tuple[int, ...] = FEASIBILITY_PHASES,
    normalized_residual: float = 0.49,
) -> list[dict[str, object]]:
    return [
        {
            "phase": phase,
            "target_norm": 1.0,
            "jacobian_rank": 12,
            "normalized_residual": normalized_residual,
            "action_rms": 0.1,
            "action_max": 0.2,
            "bound_fraction": 0.0,
            "finite": True,
        }
        for phase in phases
    ]


def test_projection_recovers_reachable_target() -> None:
    jacobian = np.eye(12)
    target = np.ones(12) * 0.1

    report = bounded_damped_projection(
        jacobian,
        target,
        lower=np.full(12, -1.0),
        upper=np.full(12, 1.0),
        damping=1e-6,
    )

    assert report["rank"] == 12
    assert report["normalized_residual"] < 1e-5
    np.testing.assert_allclose(report["achieved"], target, atol=1e-6)
    assert report["bound_fraction"] == 0.0


def test_projection_reports_bounded_unreachable_target() -> None:
    report = bounded_damped_projection(
        np.eye(12),
        np.ones(12),
        lower=np.full(12, -0.1),
        upper=np.full(12, 0.1),
        damping=1e-6,
    )

    assert report["normalized_residual"] == pytest.approx(0.9, abs=1e-6)
    assert report["bound_fraction"] == 1.0
    assert report["action_max"] == pytest.approx(0.1)


def test_leg_residual_bounds_are_head_bounds_not_total_action_bounds() -> None:
    lower, upper = leg_residual_bounds(np.linspace(-0.9, 0.9, 29))

    np.testing.assert_array_equal(lower, np.full(12, -1.0))
    np.testing.assert_array_equal(upper, np.full(12, 1.0))


@pytest.mark.parametrize(
    ("rows", "outcome"),
    [
        (_rows(normalized_residual=0.49), "leg-counterfactual-feasible"),
        (_rows(normalized_residual=0.51), "leg-counterfactual-not-feasible"),
        (_rows(phases=(0, 25, 50, 75)), "leg-counterfactual-not-feasible"),
    ],
)
def test_gate_requires_every_phase_and_median_at_most_half(rows, outcome) -> None:
    report = classify_feasibility(rows)

    assert report["outcome"] == outcome
    assert report["threshold"] == FEASIBILITY_THRESHOLD


def test_gate_fails_closed_on_nonfinite_or_zero_target() -> None:
    rows = _rows()
    rows[0]["target_norm"] = 0.0
    assert classify_feasibility(rows)["valid"] is False
    rows = _rows()
    rows[0]["normalized_residual"] = float("nan")
    assert classify_feasibility(rows)["valid"] is False


def test_parser_pins_phases_threshold_and_requires_assets() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--teacher-checkpoint", "teacher.pkl",
            "--reference-path", "reference.npz",
            "--model-path", "g1.xml",
            "--controller-path", "controller.npz",
            "--solver-profile", "g1-4x5",
            "--code-commit", "a" * 40,
            "--output-dir", "out",
        ]
    )

    assert tuple(args.phases) == FEASIBILITY_PHASES
    assert args.max_states_per_phase == 24
    assert not any(action.dest == "threshold" for action in parser._actions)


def test_atomic_artifacts_bind_npz_and_reject_tampering(tmp_path: Path) -> None:
    arrays = {
        "phase": np.asarray(FEASIBILITY_PHASES, dtype=np.int32),
        "jacobian": np.repeat(np.eye(12)[None], 5, axis=0),
        "target": np.ones((5, 12)),
    }
    payload = {
        "valid": True,
        "outcome": "leg-counterfactual-feasible",
        "phases": list(FEASIBILITY_PHASES),
        "threshold": FEASIBILITY_THRESHOLD,
    }

    report_path = publish_feasibility_artifacts(tmp_path, arrays, payload)
    validated = validate_feasibility_artifacts(report_path)

    assert validated["valid"] is True
    assert validated["npz_sha256"]
    npz_path = report_path.with_name(validated["npz_file"])
    npz_path.write_bytes(npz_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_feasibility_artifacts(report_path)


def test_artifact_validator_rejects_drifted_protocol(tmp_path: Path) -> None:
    report = tmp_path / "counterfactual_wrench_feasibility.json"
    report.write_text(json.dumps({"valid": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        validate_feasibility_artifacts(report)
