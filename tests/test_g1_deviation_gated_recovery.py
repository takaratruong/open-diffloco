import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.algorithms.shac.deviation_gated_recovery import deviation_recovery_gate
from tools.evaluate_g1_deviation_gated_recovery import (
    ARMS,
    PHASES,
    atomic_json,
    atomic_npz,
    build_parser,
    build_completion_manifest,
    classify_deviation_gate,
    validate_completion_manifest,
    validate_raw_rollout,
)


def _summary(
    survival,
    body_ratio=1.0,
    orientation_ratio=1.0,
    metric_mask=(True, True, True, True, True),
):
    body_ratios = (
        [body_ratio] * 5 if np.isscalar(body_ratio) else list(body_ratio)
    )
    orientation_ratios = (
        [orientation_ratio] * 5
        if np.isscalar(orientation_ratio)
        else list(orientation_ratio)
    )
    return {
        "survival": list(survival),
        "body_position_error_ratio": body_ratios,
        "body_orientation_error_ratio": orientation_ratios,
        "tracking_metric_mask": list(metric_mask),
    }


def test_classification_requires_componentwise_preservation():
    outcome = classify_deviation_gate(
        parent=_summary([116, 99, 67, 49, 24]),
        global_arm=_summary([124, 99, 74, 49, 24]),
        gated=_summary([124, 98, 74, 49, 24]),
        final_body_position_error=np.linspace(0.05, 0.04, 10),
    )

    assert outcome == "useful-correction-not-localizable"


def test_short_clip_solution_requires_nonincreasing_tail():
    outcome = classify_deviation_gate(
        parent=_summary([116, 99, 67, 49, 24]),
        global_arm=_summary([124, 99, 74, 49, 24]),
        gated=_summary([124, 99, 74, 49, 24]),
        final_body_position_error=np.linspace(0.05, 0.25, 10),
    )

    assert outcome == "deviation-gating-advances"


def test_short_clip_solution_requires_tracking_metric_gate():
    outcome = classify_deviation_gate(
        parent=_summary([116, 99, 67, 49, 24]),
        global_arm=_summary([124, 99, 74, 49, 24]),
        gated=_summary([124, 99, 74, 49, 24], body_ratio=1.06),
        final_body_position_error=np.linspace(0.05, 0.04, 10),
    )

    assert outcome == "useful-correction-not-localizable"


def test_short_clip_solution_passes_all_registered_gates():
    outcome = classify_deviation_gate(
        parent=_summary([116, 99, 67, 49, 24]),
        global_arm=_summary([124, 99, 74, 49, 24]),
        gated=_summary([124, 99, 74, 49, 24]),
        final_body_position_error=np.linspace(0.05, 0.04, 10),
    )

    assert outcome == "deviation-gating-solves-short-clip"


def test_tracking_metric_gate_ignores_parent_incomplete_suffixes():
    outcome = classify_deviation_gate(
        parent=_summary([116, 99, 67, 49, 24]),
        global_arm=_summary([124, 99, 74, 49, 24]),
        gated=_summary(
            [124, 99, 74, 49, 24],
            body_ratio=(2.0, 1.0, 2.0, 1.0, 1.0),
            metric_mask=(False, True, False, True, True),
        ),
        final_body_position_error=np.linspace(0.05, 0.04, 10),
    )

    assert outcome == "deviation-gating-solves-short-clip"


def _raw_arrays(arm: str, rows: int = 3, phase: int = 0):
    error = np.linspace(0.0, 0.2, rows)
    gate = np.asarray(deviation_recovery_gate(error))
    parent = np.arange(rows * 2, dtype=np.float64).reshape(rows, 2)
    residual = np.full((rows, 2), 0.2, dtype=np.float64)
    if arm == "parent":
        candidate = parent
        gated = np.zeros_like(residual)
    elif arm == "global":
        candidate = parent + residual
        gated = residual
    else:
        gated = gate[:, None] * residual
        candidate = parent + gated
    values = np.zeros((rows, 16), dtype=np.float64)
    values[:, 0] = np.arange(rows)
    values[:, 1] = phase + np.arange(rows)
    values[:, 7] = np.linspace(0.03, 0.01, rows)
    values[:, 8] = np.linspace(0.02, 0.01, rows)
    return {
        "columns": np.asarray(
            [
                "step",
                "phase",
                "reward",
                "done",
                "terminal",
                "anchor_position_error",
                "anchor_orientation_error",
                "body_position_error",
                "body_orientation_error",
                "body_linear_velocity_error",
                "body_angular_velocity_error",
                "transition_phase",
                "termination_anchor_z_error",
                "termination_anchor_xy_error",
                "termination_gravity_z_error",
                "termination_distal_z_error",
            ]
        ),
        "values": values,
        "pre_body_position_error": error,
        "gate": gate,
        "parent_action": parent,
        "raw_residual_action": residual,
        "gated_residual_action": gated,
        "candidate_action": candidate,
        "sampled_action": candidate,
        "effective_action": candidate,
        "qpos": np.zeros((rows, 36)),
        "qvel": np.zeros((rows, 35)),
    }


@pytest.mark.parametrize("arm", ARMS)
def test_raw_rollout_recomputes_registered_composition(tmp_path, arm):
    path = tmp_path / f"{arm}.npz"
    atomic_npz(path, **_raw_arrays(arm))

    result = validate_raw_rollout(path, arm=arm, phase=0)

    assert result["rows"] == 3
    assert result["valid"] is True


def test_raw_rollout_rejects_tampered_candidate_action(tmp_path):
    arrays = _raw_arrays("gated")
    arrays["candidate_action"] = arrays["candidate_action"].copy()
    arrays["candidate_action"][1, 0] += 0.1
    path = tmp_path / "gated.npz"
    atomic_npz(path, **arrays)

    with pytest.raises(ValueError, match="composition"):
        validate_raw_rollout(path, arm="gated", phase=0)


def _publish_fixture(root: Path) -> Path:
    records = []
    survival = {
        "parent": [116, 99, 67, 49, 24],
        "global": [124, 99, 74, 49, 24],
        "gated": [124, 99, 74, 49, 24],
    }
    for arm in ARMS:
        for phase_index, phase in enumerate(PHASES):
            rows = survival[arm][phase_index]
            path = root / "arms" / arm / f"phase_{phase:03d}.npz"
            arrays = _raw_arrays(arm, rows=rows, phase=phase)
            atomic_npz(path, **arrays)
            summary = path.with_suffix(".json")
            atomic_json(
                summary,
                {
                    "arm": arm,
                    "phase": phase,
                    "steps": rows,
                    "terminal": False,
                    "mean_body_position_error": float(
                        np.mean(arrays["values"][:, 7])
                    ),
                    "mean_body_orientation_error": float(
                        np.mean(arrays["values"][:, 8])
                    ),
                },
            )
            records.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "path": str(path),
                    "summary_path": str(summary),
                }
            )
    selection = root / "selection.json"
    atomic_json(
        selection,
        {
            "valid": True,
            "outcome": "deviation-gating-solves-short-clip",
            "summaries": {
                "parent": {"survival": survival["parent"]},
                "global": {"survival": survival["global"]},
                "gated": {
                    "survival": survival["gated"],
                    "body_position_error_ratio": [1.0] * 5,
                    "body_orientation_error_ratio": [1.0] * 5,
                    "tracking_metric_mask": [False, True, False, True, True],
                },
            },
            "final_body_position_error": np.linspace(0.03, 0.01, 10).tolist(),
        },
    )
    manifest = root / "completion.json"
    build_completion_manifest(
        manifest,
        records=records,
        selection_path=selection,
        provenance={"checkpoint_sha256": "a" * 64, "code_commit": "b" * 40},
    )
    return manifest


def test_manifest_rejects_tampered_raw_evidence(tmp_path):
    manifest = _publish_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    raw = Path(payload["records"][0]["path"])
    raw.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        validate_completion_manifest(manifest)


def test_manifest_accepts_complete_fifteen_rollout_fixture(tmp_path):
    manifest = _publish_fixture(tmp_path)

    payload = validate_completion_manifest(manifest)

    assert payload["valid"] is True
    assert len(payload["records"]) == 15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_rejects_summary_that_disagrees_with_raw_evidence(tmp_path):
    manifest = _publish_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    record = payload["records"][0]
    summary_path = Path(record["summary_path"])
    summary = json.loads(summary_path.read_text())
    summary["mean_body_position_error"] = 99.0
    atomic_json(summary_path, summary)
    record["summary_sha256"] = _sha256(summary_path)
    atomic_json(manifest, payload)

    with pytest.raises(ValueError, match="summary"):
        validate_completion_manifest(manifest)


def test_manifest_rejects_selection_that_disagrees_with_raw_evidence(tmp_path):
    manifest = _publish_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    selection_path = Path(payload["selection"]["path"])
    selection = json.loads(selection_path.read_text())
    selection["outcome"] = "correction-intrinsically-insufficient"
    atomic_json(selection_path, selection)
    payload["selection"]["sha256"] = _sha256(selection_path)
    atomic_json(manifest, payload)

    with pytest.raises(ValueError, match="selection"):
        validate_completion_manifest(manifest)


def test_parser_requires_exact_scientific_inputs(tmp_path):
    parser = build_parser()

    args = parser.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pkl"),
            "--reference-path",
            str(tmp_path / "reference.npz"),
            "--repository",
            str(tmp_path / "repo"),
            "--code-commit",
            "a" * 40,
            "--output-directory",
            str(tmp_path / "output"),
        ]
    )

    assert args.seed == 0
    assert args.solver_profile == "g1-4x5"
    assert args.render is True
