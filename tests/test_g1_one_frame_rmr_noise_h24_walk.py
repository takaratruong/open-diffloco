from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


CONTROL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}


def test_builder_changes_only_history_and_execution_budget() -> None:
    from tools.run_g1_one_frame_rmr_noise_h24_walk import (
        TOTAL_STEPS,
        build_one_frame_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_rmr_noise_h24_walk import (
        TOTAL_STEPS as E023_TOTAL_STEPS,
        build_rmr_noise_h24_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    parent = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_one_frame_kwargs("g1-4x5", reference, 0)
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }

    assert changed == {"actor_history_len", "total_steps"}
    assert treatment["actor_history_len"] == 1
    assert treatment["total_steps"] == TOTAL_STEPS == 393_216
    assert treatment["checkpoint_interval"] == 196_608
    assert treatment["action_noise_schedule_steps"] == E023_TOTAL_STEPS == 1_572_864
    assert expected_checkpoint_steps() == (196_608, 393_216)
    assert treatment["actor_reference_lookahead_steps"] == (4, 8, 12)
    assert treatment["actor_reference_preview_mode"] == "delta"


@pytest.mark.parametrize(
    ("treatment", "expected"),
    [
        (
            {16: (43, 36, 48, 49, 24), 32: CONTROL[32]},
            "one-frame-early-advances",
        ),
        (CONTROL, "one-frame-early-parity"),
        (
            {16: (46, 32, 48, 49, 24), 32: (45, 46, 53, 49, 24)},
            "one-frame-early-mixed",
        ),
        (
            {16: (38, 34, 45, 46, 24), 32: (41, 46, 49, 45, 24)},
            "one-frame-early-underperforms",
        ),
    ],
)
def test_classifier_covers_registered_outcomes(treatment, expected) -> None:
    from tools.run_g1_one_frame_rmr_noise_h24_walk import (
        classify_history_ablation,
    )

    assert classify_history_ablation(treatment) == expected


def test_advancement_requires_preserving_phase_100() -> None:
    from tools.run_g1_one_frame_rmr_noise_h24_walk import (
        classify_history_ablation,
    )

    treatment = {16: (43, 36, 48, 49, 23), 32: (45, 50, 53, 49, 23)}

    assert classify_history_ablation(treatment) == "one-frame-early-parity"


@pytest.mark.parametrize(
    "treatment",
    [
        {16: CONTROL[16]},
        {16: CONTROL[16], 32: (45, 50, 53, 49)},
        {16: CONTROL[16], 32: (45, 50, np.nan, 49, 24)},
        {16: CONTROL[16], 32: (45.0, 50, 53, 49, 24)},
        {16: CONTROL[16], 32: (125, 50, 53, 49, 24)},
    ],
)
def test_classifier_fails_closed_on_invalid_evidence(treatment) -> None:
    from tools.run_g1_one_frame_rmr_noise_h24_walk import (
        classify_history_ablation,
    )

    with pytest.raises(ValueError):
        classify_history_ablation(treatment)


def test_selection_uses_first_four_phase_key_and_earliest_tie() -> None:
    from tools.run_g1_one_frame_rmr_noise_h24_walk import (
        select_history_checkpoint,
    )

    assert select_history_checkpoint(CONTROL) == 32
    tied = {16: CONTROL[16], 32: CONTROL[16]}
    assert select_history_checkpoint(tied) == 16


def test_preflight_records_one_scientific_delta(monkeypatch) -> None:
    import tools.run_g1_one_frame_rmr_noise_h24_walk as runner

    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"valid": True, "protocol": "parent"},
    )
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="abc",
    )

    assert report["valid"] is True
    assert report["scientific_delta"] == ["actor_history_len"]
    assert report["actor_history_len"] == 1
    assert report["actor_input_dim"] == 328
    assert report["total_updates"] == 32
    assert report["action_noise_schedule_steps"] == 1_572_864


def test_parser_requires_code_commit() -> None:
    from tools.run_g1_one_frame_rmr_noise_h24_walk import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
