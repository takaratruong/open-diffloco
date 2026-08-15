from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_recovery_builder_changes_only_registered_treatment(tmp_path: Path):
    from tools.run_g1_e023_anchored_carried_recovery import (
        CONTINUATION_END_STEP,
        build_anchored_carried_recovery_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = tmp_path / "walk.npz"
    checkpoint = tmp_path / "checkpoint_step_1572864.pkl"
    bank = tmp_path / "carried.npz"
    parent = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_anchored_carried_recovery_kwargs(
        "g1-4x5", reference, 0, checkpoint, bank
    )
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }
    assert changed == {
        "actor_policy_anchor_weight",
        "actor_residual_preview_adapter",
        "allow_resume_actor_residual_preview_adapter_start",
        "allow_resume_carried_reset_change",
        "carried_reset_bank_path",
        "carried_reset_probability",
        "checkpoint_interval",
        "resume_from",
        "total_steps",
    }
    assert treatment.get("actor_residual_preview_hidden", 256) == 256
    assert treatment.get("actor_residual_preview_optimizer", "adam") == "adam"
    assert treatment["carried_reset_probability"] == 0.25
    assert treatment["actor_policy_anchor_weight"] == 1.0
    assert treatment["reference_reset_noise_scale"] == 0.0
    assert treatment["domain_randomization"] is False
    assert treatment["unroll_length"] == 24
    assert treatment["gradient_accumulation_steps"] == 2
    assert treatment["action_noise_schedule_steps"] == 1_572_864
    assert CONTINUATION_END_STEP == 2_359_296
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


def _migration_report() -> dict[str, object]:
    return {
        "valid": True,
        "parent_parameters_exact": True,
        "parent_mu_exact": True,
        "parent_nu_exact": True,
        "optimizer_count_exact": True,
        "optimizer_outer_state_exact": True,
        "adapter_parameters_finite": True,
        "adapter_mu_zero": True,
        "adapter_nu_zero": True,
        "residual_action_zero": True,
        "reconstructed_parent_exact": True,
        "max_action_absolute_error": 0.0,
        "max_action_relative_error": 0.0,
    }


def _telemetry_rows() -> list[dict[str, object]]:
    from tools.run_g1_e023_anchored_carried_recovery import (
        expected_checkpoint_steps,
    )

    return [
        {
            "step": step,
            "actor_preview_frozen_parameter_drift_max_abs": 0.0,
            "actor_preview_frozen_moment_drift_max_abs": 0.0,
            "actor_preview_normalizer_drift_max_abs": 0.0,
            "actor_preview_gradient_norm": 0.1,
            "actor_preview_update_norm": 0.01,
            "actor_preview_valid": True,
            "actor_policy_anchor_weight": 1.0,
            "actor_policy_anchor_squared_error": 0.001,
            "actor_policy_anchor_valid": True,
        }
        for step in expected_checkpoint_steps()
    ]


def test_recovery_telemetry_requires_exact_migration_and_frozen_parent():
    from tools.run_g1_e023_anchored_carried_recovery import (
        validate_recovery_telemetry,
    )

    result = validate_recovery_telemetry(_telemetry_rows(), _migration_report())
    assert result["valid"] is True
    assert result["checkpoint_count"] == 8

    rows = _telemetry_rows()
    rows[-1]["actor_preview_normalizer_drift_max_abs"] = 1e-7
    with pytest.raises(ValueError, match="frozen drift"):
        validate_recovery_telemetry(rows, _migration_report())
    duplicate = _telemetry_rows()
    duplicate.append(dict(duplicate[-1]))
    with pytest.raises(ValueError, match="exact checkpoint grid"):
        validate_recovery_telemetry(duplicate, _migration_report())
    report = _migration_report()
    report["residual_action_zero"] = False
    with pytest.raises(ValueError, match="migration"):
        validate_recovery_telemetry(_telemetry_rows(), report)


def test_componentwise_selector_never_trades_away_e023_phase():
    from tools.run_g1_e023_anchored_carried_recovery import select_checkpoint

    records = [
        {
            "update": 8,
            "survival": [120, 99, 60, 49, 24],
            "completed_suffix": [False, True, False, True, True],
        },
        {
            "update": 16,
            "survival": [116, 99, 68, 49, 24],
            "completed_suffix": [False, True, False, True, True],
        },
        {
            "update": 32,
            "survival": [117, 99, 68, 49, 24],
            "completed_suffix": [False, True, False, True, True],
        },
        {
            "update": 64,
            "survival": [124, 99, 74, 49, 24],
            "completed_suffix": [True, True, True, True, True],
        },
    ]
    selected = select_checkpoint(records)
    assert selected["outcome"] == "anchored-carried-solves-walk"
    assert selected["selected_update"] == 64
    assert selected["eligible_updates"] == [16, 32, 64]

    insufficient = select_checkpoint(
        [
            {
                "update": 8,
                "survival": [123, 98, 74, 49, 24],
                "completed_suffix": [False, False, True, True, True],
            }
        ]
    )
    assert insufficient["outcome"] == "anchored-carried-insufficient"
    assert insufficient["selected_update"] is None

    final_transition_terminal = select_checkpoint(
        [
            {
                "update": 16,
                "survival": [124, 99, 74, 49, 24],
                "completed_suffix": [True, True, False, True, True],
            }
        ]
    )
    assert final_transition_terminal["outcome"] == "anchored-carried-advances"


def test_recovery_preflight_pins_parent_and_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import tools.run_g1_e023_anchored_carried_recovery as runner

    checkpoint = tmp_path / "checkpoint_step_1572864.pkl"
    hparams = tmp_path / "hparams.json"
    bank = tmp_path / "bank.npz"
    summary = tmp_path / "bank.json"
    for path in (checkpoint, hparams, bank):
        path.write_bytes(b"x")
    summary.write_text(
        json.dumps(
            {
                "valid": True,
                "protocol": "g1-e023-history-carried-reset-bank-v1",
                "rows": 48,
                "rows_per_source": [24, 24],
                "source_phases": [0, 50],
                "checkpoint_sha256": runner.EXPECTED_RESUME_SHA256,
                "hparams_sha256": runner.EXPECTED_RESUME_HPARAMS_SHA256,
                "bank_sha256": "c" * 64,
                "code_commit": "d" * 40,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"valid": True, "protocol": "parent"},
    )
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: {
            checkpoint: runner.EXPECTED_RESUME_SHA256,
            hparams: runner.EXPECTED_RESUME_HPARAMS_SHA256,
            bank: "c" * 64,
            summary: "e" * 64,
        }[path.resolve()],
    )
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        resume_from=checkpoint,
        carried_bank=bank,
        carried_bank_summary=summary,
        carried_bank_sha256="c" * 64,
        carried_bank_summary_sha256="e" * 64,
        code_commit="d" * 40,
    )
    assert report["valid"] is True
    assert report["scientific_delta"] == [
        "resume_from",
        "total_steps",
        "checkpoint_interval",
        "actor_residual_preview_adapter",
        "actor_policy_anchor_weight",
        "carried_reset_bank_path",
        "carried_reset_probability",
    ]
    assert report["bank_rows"] == 48
