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


def test_rmr_noise_changes_only_the_registered_continuation_fields(tmp_path):
    from src.core.rmr_action_noise import RMR_ACTION_STD
    from tools.run_g1_rmr_action_noise_continuation import (
        RMR_NOISE_END_STEP,
        build_rmr_action_noise_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_zero_bootstrap_continuation import (
        build_zero_bootstrap_kwargs,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e008-selected.pkl"
    parent = build_zero_bootstrap_kwargs("g1-4x5", reference, 0, checkpoint)
    candidate = build_rmr_action_noise_kwargs("g1-4x5", reference, 0, checkpoint)
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
    assert candidate["action_noise_schedule_steps"] == RMR_NOISE_END_STEP
    assert candidate["allow_resume_action_noise_change"] is True
    np.testing.assert_array_equal(candidate["action_noise_std_start"], RMR_ACTION_STD)
    np.testing.assert_array_equal(candidate["action_noise_std_end"], RMR_ACTION_STD)
    assert expected_checkpoint_steps() == (
        1_916_928,
        1_966_080,
        2_015_232,
        2_064_384,
    )


def test_rmr_noise_provenance_pins_selected_e008_and_source_vector_contract():
    from src.core.rmr_action_noise import (
        RMR_ACTION_STD,
        RMR_ACTION_STD_JOINT_NAMES,
    )
    from tools.run_g1_rmr_action_noise_continuation import (
        E008_SELECTED_CHECKPOINT_SHA256,
        E008_SELECTED_HPARAMS_SHA256,
        E008_SELECTED_STEP,
        RMR_SOURCE_CHECKPOINT_SHA256,
        RMR_SOURCE_JOINT_NAMES,
        RMR_SOURCE_STD,
    )

    assert E008_SELECTED_STEP == 1_867_776
    assert E008_SELECTED_CHECKPOINT_SHA256 == (
        "2de4af6d78cd5250c87577397c048b06e60c5b8a7b272c0f8966b8bf589b4474"
    )
    assert E008_SELECTED_HPARAMS_SHA256 == (
        "e0b78f2185d91e7d2edadff0afb4f470e70d38f1f7716c304cf866380e594dba"
    )
    assert RMR_SOURCE_CHECKPOINT_SHA256 == (
        "5174a0f1dc8c83ef9ea45769c3b0f19383e5aeeafea2171433f8e7bb88b21746"
    )
    assert RMR_SOURCE_JOINT_NAMES == RMR_ACTION_STD_JOINT_NAMES
    np.testing.assert_array_equal(RMR_SOURCE_STD, RMR_ACTION_STD)


def test_parser_rejects_scientific_overrides():
    from tools.run_g1_rmr_action_noise_continuation import build_parser

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
        ["--action-noise-std-start", "0.5"],
        ["--action-noise-std-end", "0.5"],
        ["--action-noise-schedule-steps", "3000000"],
        ["--allow-resume-action-noise-change", "false"],
        ["--actor-bootstrap-scale", "0.5"],
        ["--num-envs", "512"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])


def test_noise_schedule_endpoint_change_requires_noise_resume_authority():
    from src.algorithms.shac.algorithm import resolve_action_noise_schedule_steps

    saved = {"total_steps": 1_867_776, "action_noise_schedule_steps": 1_867_776}
    with pytest.raises(ValueError, match="allow_resume_action_noise_change"):
        resolve_action_noise_schedule_steps(
            total_steps=2_064_384,
            resumed_step=1_867_776,
            resumed_hparams=saved,
            requested_schedule_steps=2_064_384,
            allow_resume_action_noise_change=False,
        )
    assert (
        resolve_action_noise_schedule_steps(
            total_steps=2_064_384,
            resumed_step=1_867_776,
            resumed_hparams=saved,
            requested_schedule_steps=2_064_384,
            allow_resume_action_noise_change=True,
        )
        == 2_064_384
    )
    assert (
        resolve_action_noise_schedule_steps(
            total_steps=2_064_384,
            resumed_step=1_867_776,
            resumed_hparams=saved,
            requested_schedule_steps=None,
            allow_resume_action_noise_change=False,
        )
        == 1_867_776
    )
    for invalid_schedule in (True, 1.5, "2064384"):
        with pytest.raises(ValueError, match="schedule steps must be positive"):
            resolve_action_noise_schedule_steps(
                total_steps=2_064_384,
                resumed_step=1_867_776,
                resumed_hparams={
                    "total_steps": 1_867_776,
                    "action_noise_schedule_steps": invalid_schedule,
                },
            )


def test_training_validation_requires_32_updates_fixed_noise_and_finite_cagrad(
    tmp_path,
):
    from src.core.rmr_action_noise import RMR_ACTION_STD, action_noise_std_hparam
    from tools.run_g1_rmr_action_noise_continuation import (
        CHECKPOINT_INTERVAL,
        RMR_NOISE_END_STEP,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    run = tmp_path / "shac_run"
    run.mkdir()
    hparams = {
        "total_steps": RMR_NOISE_END_STEP,
        "action_noise_std_start": action_noise_std_hparam(RMR_ACTION_STD),
        "action_noise_std_end": action_noise_std_hparam(RMR_ACTION_STD),
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
    assert validation["update_count"] == 32
    assert validation["checkpoint_interval"] == CHECKPOINT_INTERVAL
    assert validation["checkpoint_steps"] == list(expected_checkpoint_steps())

    hparams["action_noise_std_end"] = 0.32
    (run / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
    with pytest.raises(ValueError, match="action_noise_std_end"):
        validate_training_artifacts(run)

    hparams["action_noise_std_end"] = action_noise_std_hparam(RMR_ACTION_STD)
    (run / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
    rows[-1]["actor_preview_frozen_parameter_drift_max_abs"] = 1e-6
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen parent state"):
        validate_training_artifacts(run)

    rows[-1]["actor_preview_frozen_parameter_drift_max_abs"] = 0.0
    rows[-1]["actor_cagrad_weights"] = [float("nan")] * 5
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid CAGrad telemetry"):
        validate_training_artifacts(run)

    rows[-1]["actor_cagrad_weights"] = [0.2] * 5
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps([*rows, rows[-1]]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly four"):
        validate_training_artifacts(run)

    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    extra_step = expected_checkpoint_steps()[-1] + CHECKPOINT_INTERVAL
    (run / f"checkpoint_step_{extra_step}.pkl").write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="exactly four"):
        validate_training_artifacts(run)
