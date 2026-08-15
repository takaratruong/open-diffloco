from __future__ import annotations

from pathlib import Path

import pytest


def test_lafan_recovery_changes_reference_with_explicit_authority(tmp_path: Path):
    from tools.run_g1_e023_lafan_anchored_carried_recovery import (
        CONTINUATION_END_STEP,
        build_lafan_recovery_kwargs,
        expected_checkpoint_steps,
    )

    kwargs = build_lafan_recovery_kwargs(
        "g1-4x5",
        tmp_path / "lafan.npz",
        0,
        tmp_path / "checkpoint_step_1572864.pkl",
        tmp_path / "bank.npz",
    )
    assert kwargs["allow_resume_reference_path_change"] is True
    assert kwargs["actor_residual_preview_adapter"] is True
    assert kwargs["allow_resume_actor_residual_preview_adapter_start"] is True
    assert kwargs["carried_reset_probability"] == 0.25
    assert kwargs["actor_policy_anchor_weight"] == 1.0
    assert kwargs["total_steps"] == CONTINUATION_END_STEP == 2_359_296
    assert expected_checkpoint_steps() == (
        1_671_168,
        1_769_472,
        1_867_776,
        1_966_080,
        2_064_384,
        2_162_688,
        2_260_992,
        2_359_296,
    )


def test_lafan_selector_preserves_every_parent_phase_and_requires_clean_suffix():
    from tools.run_g1_e023_lafan_anchored_carried_recovery import (
        select_checkpoint,
    )

    result = select_checkpoint(
        [
            {
                "update": 8,
                "survival": [117, 63, 49, 39, 46],
                "completed_suffix": [False] * 5,
            },
            {
                "update": 16,
                "survival": [118, 65, 51, 40, 47],
                "completed_suffix": [False] * 5,
            },
            {
                "update": 32,
                "survival": [499, 399, 299, 199, 99],
                "completed_suffix": [True, True, False, True, True],
            },
        ]
    )
    assert result["selected_update"] == 32
    assert result["eligible_updates"] == [16, 32]
    assert result["outcome"] == "lafan-carried-recovery-advances"

    solved = select_checkpoint(
        [
            {
                "update": 64,
                "survival": [499, 399, 299, 199, 99],
                "completed_suffix": [True] * 5,
            }
        ]
    )
    assert solved["outcome"] == "lafan-carried-recovery-solves"


def test_reference_migration_requires_exact_old_and_new_reference_hashes():
    from tools.run_g1_e023_lafan_anchored_carried_recovery import (
        EXPECTED_LAFAN_REFERENCE_SHA256,
        EXPECTED_PARENT_REFERENCE_SHA256,
        validate_reference_migration,
    )

    report = {
        "protocol": "g1-reference-path-migration-v1",
        "valid": True,
        "previous_reference_sha256": EXPECTED_PARENT_REFERENCE_SHA256,
        "requested_reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "environment_state_reinitialized": True,
    }
    assert validate_reference_migration(report)["valid"] is True
    with pytest.raises(ValueError, match="reference migration"):
        validate_reference_migration(
            {**report, "environment_state_reinitialized": False}
        )


def test_lafan_runner_parser_requires_pinned_bank_and_parent():
    from tools.run_g1_e023_lafan_anchored_carried_recovery import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
