import json

import pytest


PHASES = [0, 100, 200, 300, 400]


def _records():
    return [
        {
            "step": 1_376_256,
            "actor_cagrad_bin_counts": [100, 100, 100, 100, 100],
            "actor_cagrad_bin_losses": [5, 4, 3, 2, 1],
            "actor_preview_valid": True,
        },
        {
            "step": 1_572_864,
            "actor_cagrad_bin_counts": [100, 100, 100, 100, 100],
            "actor_cagrad_bin_losses": [4, 3, 2, 1, 0],
            "actor_preview_valid": True,
        },
    ]


def _summary(survival, checkpoint_sha):
    return {
        "phases": PHASES,
        "steps": {
            str(phase): steps
            for phase, steps in zip(PHASES, survival, strict=True)
        },
        "checkpoint_sha256": checkpoint_sha,
        "reference_sha256": "reference-sha",
        "solver_profile": "g1-4x5",
    }


def test_audit_pairs_registered_bins_and_reports_rank_agreement():
    from tools.evaluate_preview_objective_survival import (
        build_objective_survival_audit,
    )

    summaries = {
        1_376_256: _summary([10, 20, 30, 40, 50], "mid-sha"),
        1_572_864: _summary([20, 30, 40, 50, 60], "final-sha"),
    }

    audit = build_objective_survival_audit(
        _records(), summaries, phase_count=499
    )

    assert len(audit["cases"]) == 10
    assert audit["loss_survival_spearman"] < 0.0
    assert audit["checkpoint_steps"] == [1_376_256, 1_572_864]
    assert audit["phases"] == PHASES
    assert audit["valid"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda records, summaries: records.append(records[0]),
            "duplicate",
        ),
        (
            lambda records, summaries: summaries.pop(1_572_864),
            "steps must match",
        ),
        (
            lambda records, summaries: summaries[1_376_256].update(
                phases=[0, 50, 100, 200, 400]
            ),
            "phases",
        ),
        (
            lambda records, summaries: records[0].update(
                actor_cagrad_bin_losses=[1, 2, 3, 4, float("nan")]
            ),
            "finite",
        ),
    ],
)
def test_audit_rejects_invalid_or_mismatched_inputs(mutate, message):
    from tools.evaluate_preview_objective_survival import (
        build_objective_survival_audit,
    )

    records = _records()
    summaries = {
        1_376_256: _summary([10, 20, 30, 40, 50], "mid-sha"),
        1_572_864: _summary([20, 30, 40, 50, 60], "final-sha"),
    }
    mutate(records, summaries)

    with pytest.raises(ValueError, match=message):
        build_objective_survival_audit(records, summaries, phase_count=499)


def test_cli_writes_source_hashes_atomically(tmp_path):
    from tools.evaluate_preview_objective_survival import main

    metrics = tmp_path / "checkpoint_phase_metrics.json"
    midpoint = tmp_path / "midpoint.json"
    final = tmp_path / "final.json"
    output = tmp_path / "objective_survival_audit.json"
    metrics.write_text(json.dumps(_records()))
    midpoint.write_text(
        json.dumps(_summary([10, 20, 30, 40, 50], "mid-sha"))
    )
    final.write_text(
        json.dumps(_summary([20, 30, 40, 50, 60], "final-sha"))
    )

    main(
        [
            "--checkpoint-phase-metrics",
            str(metrics),
            "--phase-grid",
            "1376256",
            str(midpoint),
            "--phase-grid",
            "1572864",
            str(final),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["valid"]
    assert len(payload["source_sha256"]) == 3
    assert not list(tmp_path.glob(".*.tmp"))
