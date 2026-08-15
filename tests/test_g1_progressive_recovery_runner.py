from __future__ import annotations

import pytest

from tools.run_g1_progressive_recovery_expert import (
    END_STEP,
    EXPECTED_CHECKPOINT_STEPS,
    build_progressive_recovery_kwargs,
    validate_recovery_training_rows,
)


def test_progressive_runner_builds_only_registered_treatment_delta(tmp_path):
    kwargs = build_progressive_recovery_kwargs(
        "g1-4x5",
        tmp_path / "reference.npz",
        0,
        tmp_path / "e023.pkl",
        tmp_path / "targeted.npz",
        tmp_path / "support.npz",
        "a" * 64,
    )

    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == EXPECTED_CHECKPOINT_STEPS
    assert kwargs["unroll_length"] == 24
    assert kwargs["num_envs"] == 256
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["actor_lr"] == 1e-3
    assert kwargs["actor_per_env_grad_clip"] == 1.0
    assert kwargs["actor_policy_anchor_weight"] == 1.0
    assert kwargs["actor_bootstrap_scale"] == 0.0
    assert kwargs["actor_residual_preview_adapter"] is True
    assert kwargs["allow_resume_actor_residual_preview_adapter_start"] is True
    assert kwargs["actor_state_gated_recovery_support_path"] == str(
        (tmp_path / "support.npz").resolve()
    )
    assert kwargs["actor_state_gated_recovery_support_sha256"] == "a" * 64
    assert kwargs["allow_resume_actor_state_gated_recovery_start"] is True
    assert kwargs["actor_cagrad"] is False
    assert kwargs["allow_resume_actor_cagrad_change"] is True
    assert kwargs["carried_reset_probability"] == 0.25
    assert kwargs["reference_reset_noise_scale"] == 0.0
    assert kwargs["actor_observation_noise"] is False
    assert kwargs["domain_randomization"] is False
    assert kwargs["push_velocity_range"] == (0.0, 0.0)
    assert kwargs["friction_range"] == (1.0, 1.0)
    assert kwargs["torso_wrench_assistance"] is False


def _row(step):
    return {
        "step": step,
        "actor_preview_gradient_norm": 1.0,
        "actor_preview_update_norm": 0.1,
        "actor_preview_frozen_parameter_drift_max_abs": 0.0,
        "actor_preview_frozen_moment_drift_max_abs": 0.0,
        "actor_preview_normalizer_drift_max_abs": 0.0,
        "actor_preview_valid": True,
        "actor_recovery_gate_activation_fraction": 0.1,
        "actor_recovery_gate_max": 0.8,
        "actor_recovery_carried_activation_fraction": 0.3,
        "actor_recovery_reference_activation_fraction": 0.02,
        "actor_recovery_gated_residual_rms": 0.05,
        "actor_recovery_gated_residual_max_abs": 0.2,
        "actor_recovery_valid": True,
    }


def test_training_row_validator_requires_exact_finite_grid():
    report = validate_recovery_training_rows(
        [_row(step) for step in EXPECTED_CHECKPOINT_STEPS]
    )
    assert report["valid"] is True
    assert report["checkpoint_steps"] == list(EXPECTED_CHECKPOINT_STEPS)

    bad = [_row(step) for step in EXPECTED_CHECKPOINT_STEPS]
    bad[0]["actor_recovery_gate_max"] = float("nan")
    with pytest.raises(ValueError, match="recovery telemetry"):
        validate_recovery_training_rows(bad)


def test_training_row_validator_rejects_cagrad_or_wrong_grid():
    bad = [_row(step) for step in EXPECTED_CHECKPOINT_STEPS]
    bad[0]["actor_cagrad_valid"] = True
    with pytest.raises(ValueError, match="CAGrad"):
        validate_recovery_training_rows(bad)
    with pytest.raises(ValueError, match="checkpoint grid"):
        validate_recovery_training_rows(bad[:-1])


def test_runner_source_does_not_hide_dynamic_sweeps():
    source = __import__(
        "inspect"
    ).getsource(build_progressive_recovery_kwargs)
    assert "for " not in source
    assert "sweep" not in source.lower()
