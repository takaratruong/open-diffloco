import json

import numpy as np
import pytest


def _cagrad_row(step: int) -> dict[str, object]:
    return {
        "step": step,
        "actor_bootstrap_scale_current": 0.0,
        "torso_wrench_assistance_scale_current": 0.0,
        "torso_wrench_assistance_active_fraction": 0.0,
        "torso_wrench_assistance_max_force": 0.0,
        "torso_wrench_assistance_max_torque": 0.0,
        "torso_wrench_assistance_valid": True,
        "actor_preview_frozen_parameter_drift_max_abs": 0.0,
        "actor_preview_frozen_moment_drift_max_abs": 0.0,
        "actor_preview_normalizer_drift_max_abs": 0.0,
        "actor_preview_gradient_norm": 0.1,
        "actor_preview_update_norm": 0.2,
        "actor_preview_valid": True,
        "actor_cagrad_bin_counts": [1.0] * 5,
        "actor_cagrad_bin_gradient_norms": [1.0] * 5,
        "actor_cagrad_bin_losses": [1.0] * 5,
        "actor_cagrad_weights": [0.2] * 5,
        "actor_cagrad_gram_matrix": np.eye(5).tolist(),
        "actor_cagrad_cosine_matrix": np.eye(5).tolist(),
        "actor_cagrad_objective": 1.0,
        "actor_cagrad_dual_gap": 0.0,
        "actor_cagrad_uniform_combined_cosine": 0.5,
        "actor_cagrad_combined_norm": 1.0,
        "actor_cagrad_valid": True,
    }


def test_matched_scalar_is_exact_rmr_vector_rms():
    from src.core.rmr_action_noise import RMR_ACTION_STD
    from tools.run_g1_matched_rms_action_noise_continuation import (
        MATCHED_RMS_ACTION_NOISE_STD,
    )

    expected = float(
        np.sqrt(np.mean(np.square(RMR_ACTION_STD.astype(np.float64))))
    )
    assert MATCHED_RMS_ACTION_NOISE_STD == expected
    assert MATCHED_RMS_ACTION_NOISE_STD == 0.25027265203867416


def test_matched_scalar_changes_only_registered_continuation_fields(tmp_path):
    from tools.run_g1_matched_rms_action_noise_continuation import (
        MATCHED_RMS_ACTION_NOISE_STD,
        build_matched_rms_action_noise_kwargs,
    )
    from tools.run_g1_rmr_action_noise_continuation import RMR_NOISE_END_STEP
    from tools.run_g1_zero_bootstrap_continuation import (
        build_zero_bootstrap_kwargs,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e008-selected.pkl"
    parent = build_zero_bootstrap_kwargs("g1-4x5", reference, 0, checkpoint)
    candidate = build_matched_rms_action_noise_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    changed_keys = {
        key
        for key in set(parent) | set(candidate)
        if not np.array_equal(
            np.asarray(parent.get(key)), np.asarray(candidate.get(key))
        )
    }
    assert changed_keys == {
        "total_steps",
        "action_noise_std_start",
        "action_noise_std_end",
        "action_noise_schedule_steps",
        "allow_resume_action_noise_change",
    }
    assert candidate["total_steps"] == RMR_NOISE_END_STEP
    assert candidate["action_noise_std_start"] == MATCHED_RMS_ACTION_NOISE_STD
    assert candidate["action_noise_std_end"] == MATCHED_RMS_ACTION_NOISE_STD


def test_matched_scalar_parser_rejects_scientific_overrides():
    from tools.run_g1_matched_rms_action_noise_continuation import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e008-selected.pkl",
        "--code-commit",
        "0" * 40,
    ]
    build_parser().parse_args(required)
    for override in (
        ["--seed", "1"],
        ["--total-steps", "3000000"],
        ["--action-noise-std", "0.3"],
        ["--num-envs", "512"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])


def test_matched_scalar_training_validator_requires_exact_scalar(tmp_path):
    from tools.run_g1_matched_rms_action_noise_continuation import (
        MATCHED_RMS_ACTION_NOISE_STD,
        validate_training_artifacts,
    )
    from tools.run_g1_rmr_action_noise_continuation import (
        RMR_NOISE_END_STEP,
        expected_checkpoint_steps,
    )

    run = tmp_path / "shac_run"
    run.mkdir()
    hparams = {
        "total_steps": RMR_NOISE_END_STEP,
        "action_noise_std_start": MATCHED_RMS_ACTION_NOISE_STD,
        "action_noise_std_end": MATCHED_RMS_ACTION_NOISE_STD,
        "action_noise_schedule_steps": RMR_NOISE_END_STEP,
        "allow_resume_action_noise_change": True,
        "actor_bootstrap_scale": 0.0,
        "allow_resume_actor_bootstrap_scale_change": True,
        "torso_wrench_assistance": True,
        "torso_wrench_assistance_start_step": 1_327_104,
        "torso_wrench_assistance_end_step": 1_622_016,
        "torso_wrench_assistance_zero_fraction": 0.25,
        "reference_reset_noise_scale": 1.0,
        "domain_randomization": True,
        "actor_cagrad": True,
        "actor_residual_preview_adapter": True,
        "gradient_accumulation_steps": 2,
        "unroll_length": 12,
        "num_envs": 256,
        "effective_num_envs": 512,
        "seed": 0,
    }
    (run / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
    rows = []
    for step in expected_checkpoint_steps():
        (run / f"checkpoint_step_{step}.pkl").write_bytes(b"checkpoint")
        rows.append(_cagrad_row(step))
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    validation = validate_training_artifacts(run)
    assert validation["valid"] is True
    assert validation["protocol"] == (
        "g1-matched-rms-action-noise-continuation-training-v1"
    )
    hparams["action_noise_std_end"] = np.nextafter(
        MATCHED_RMS_ACTION_NOISE_STD, 1.0
    )
    (run / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
    with pytest.raises(ValueError, match="action_noise_std_end"):
        validate_training_artifacts(run)
