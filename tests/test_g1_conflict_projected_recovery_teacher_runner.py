from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_runner_changes_only_registered_teacher_treatment():
    from tools.run_g1_conflict_projected_recovery_teacher import (
        RECOVERY_TEACHER_SHA256,
        build_conflict_projected_recovery_teacher_kwargs,
    )
    from tools.run_g1_zero_head_feature_transfer import (
        EXPERT_SHA256,
        build_zero_head_feature_transfer_kwargs,
    )

    args = ("g1-4x5", "/tmp/ref.npz", 0, "/tmp/parent.pkl", "/tmp/bank.npz")
    baseline = build_zero_head_feature_transfer_kwargs(
        *args, expert_path="/tmp/expert.pkl", expert_sha256=EXPERT_SHA256
    )
    candidate = build_conflict_projected_recovery_teacher_kwargs(
        *args,
        expert_path="/tmp/expert.pkl",
        expert_sha256=EXPERT_SHA256,
        teacher_path="/tmp/teacher.npz",
        teacher_sha256=RECOVERY_TEACHER_SHA256,
    )
    delta = {
        "actor_recovery_teacher_dataset_path",
        "actor_recovery_teacher_dataset_sha256",
        "actor_recovery_teacher_gradient_ratio",
        "allow_resume_actor_recovery_teacher_change",
    }
    assert set(candidate) == set(baseline) | delta
    for key, value in baseline.items():
        if hasattr(value, "shape"):
            np.testing.assert_array_equal(candidate[key], value)
        else:
            assert candidate[key] == value
    assert candidate["actor_recovery_teacher_gradient_ratio"] == 0.5
    assert candidate["allow_resume_actor_recovery_teacher_change"] is True


def _telemetry_rows():
    from tools.run_g1_e023_anchored_carried_recovery import (
        expected_checkpoint_steps,
    )

    return [
        {
            "step": step,
            "actor_recovery_teacher_loss": 0.2,
            "actor_recovery_teacher_raw_gradient_norm": 4.0,
            "actor_recovery_teacher_projected_gradient_norm": 3.0,
            "actor_recovery_teacher_applied_gradient_norm": 1.0,
            "actor_recovery_teacher_physics_gradient_norm": 2.0,
            "actor_recovery_teacher_combined_gradient_norm": 2.5,
            "actor_recovery_teacher_physics_dot": -0.5,
            "actor_recovery_teacher_physics_cosine": -0.25,
            "actor_recovery_teacher_applied_scale": 1.0 / 3.0,
            "actor_recovery_teacher_parent_gradient_max_abs": 0.0,
            "actor_recovery_teacher_valid": True,
        }
        for step in expected_checkpoint_steps()
    ]


def test_teacher_telemetry_requires_every_checkpoint_and_norm_cap():
    from tools.run_g1_conflict_projected_recovery_teacher import (
        validate_recovery_teacher_telemetry,
    )

    result = validate_recovery_teacher_telemetry(_telemetry_rows())
    assert result["valid"] is True
    assert result["checkpoint_count"] == 8

    missing = _telemetry_rows()[:-1]
    with pytest.raises(ValueError, match="checkpoint grid"):
        validate_recovery_teacher_telemetry(missing)
    over_cap = _telemetry_rows()
    over_cap[3]["actor_recovery_teacher_applied_gradient_norm"] = 1.01
    with pytest.raises(ValueError, match="norm cap"):
        validate_recovery_teacher_telemetry(over_cap)
    nonfinite = _telemetry_rows()
    nonfinite[0]["actor_recovery_teacher_loss"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_recovery_teacher_telemetry(nonfinite)


def test_training_validation_requires_exact_teacher_hparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import tools.run_g1_conflict_projected_recovery_teacher as runner

    teacher = str((tmp_path / "teacher.npz").resolve())
    kwargs = {
        "actor_recovery_teacher_dataset_path": teacher,
        "actor_recovery_teacher_dataset_sha256": runner.RECOVERY_TEACHER_SHA256,
        "actor_recovery_teacher_gradient_ratio": 0.5,
        "allow_resume_actor_recovery_teacher_change": True,
    }
    (tmp_path / "hparams.json").write_text(
        json.dumps(
            {
                "actor_recovery_teacher_enabled": True,
                **kwargs,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_phase_metrics.json").write_text(
        json.dumps(_telemetry_rows()), encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "validate_zero_head_training_artifacts",
        lambda *_args, **_kwargs: {"valid": True, "protocol": "base"},
    )
    result = runner.validate_training_artifacts(
        tmp_path, expected_kwargs=kwargs
    )
    assert result["protocol"] == (
        "g1-conflict-projected-recovery-teacher-training-v1"
    )

    hparams = json.loads((tmp_path / "hparams.json").read_text())
    hparams["actor_recovery_teacher_gradient_ratio"] = 0.4
    (tmp_path / "hparams.json").write_text(json.dumps(hparams))
    with pytest.raises(ValueError, match="hparams"):
        runner.validate_training_artifacts(tmp_path, expected_kwargs=kwargs)
