import json

import pytest


def test_zero_bootstrap_changes_only_endpoint_scale_and_resume_authority(tmp_path):
    from tools.run_g1_frozen_residual_assistance_curriculum import (
        build_frozen_residual_assistance_kwargs,
    )
    from tools.run_g1_zero_bootstrap_continuation import (
        ZERO_BOOTSTRAP_END_STEP,
        build_zero_bootstrap_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e012-selected.pkl"
    parent = build_frozen_residual_assistance_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_zero_bootstrap_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }
    assert delta == {
        "total_steps": ZERO_BOOTSTRAP_END_STEP,
        "actor_bootstrap_scale": 0.0,
        "allow_resume_actor_bootstrap_scale_change": True,
    }
    assert expected_checkpoint_steps() == (
        1_720_320,
        1_769_472,
        1_818_624,
        1_867_776,
    )


def test_zero_bootstrap_parser_exposes_no_scientific_overrides():
    from tools.run_g1_zero_bootstrap_continuation import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e012-selected.pkl",
        "--code-commit",
        "0" * 40,
    ]
    build_parser().parse_args(required)
    for override in (
        ["--seed", "1"],
        ["--actor-bootstrap-scale", "0.5"],
        ["--num-envs", "512"],
        ["--total-steps", "3000000"],
        ["--unroll-length", "24"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])


def test_training_validation_requires_zero_bootstrap_and_dense_grid(tmp_path):
    from tools.run_g1_zero_bootstrap_continuation import (
        ZERO_BOOTSTRAP_END_STEP,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    run = tmp_path / "shac_run"
    run.mkdir()
    hparams = {
        "total_steps": ZERO_BOOTSTRAP_END_STEP,
        "checkpoint_interval": 49_152,
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
        rows.append(
            {
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
                "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
                "actor_cagrad_bin_gradient_norms": [1, 1, 1, 1, 1],
                "actor_cagrad_bin_losses": [1, 2, 3, 4, 5],
                "actor_cagrad_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                "actor_cagrad_gram_matrix": [[1, 0, 0, 0, 0]] * 5,
                "actor_cagrad_cosine_matrix": [[1, 0, 0, 0, 0]] * 5,
                "actor_cagrad_objective": 1.0,
                "actor_cagrad_dual_gap": 0.0,
                "actor_cagrad_uniform_combined_cosine": 0.5,
                "actor_cagrad_combined_norm": 1.0,
                "actor_cagrad_valid": True,
            }
        )
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    assert validate_training_artifacts(run)["valid"] is True

    rows[-1]["actor_bootstrap_scale_current"] = 0.1
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="zero actor bootstrap"):
        validate_training_artifacts(run)

    rows[-1]["actor_bootstrap_scale_current"] = 0.0
    rows[-1]["actor_cagrad_valid"] = False
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid CAGrad telemetry"):
        validate_training_artifacts(run)

    rows[-1]["actor_cagrad_valid"] = True
    rows[-1]["actor_cagrad_bin_counts"] = [float("nan"), 1, 1, 1, 1]
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid CAGrad coverage"):
        validate_training_artifacts(run)


def test_checkpoint_telemetry_builder_persists_bootstrap_and_cagrad_contract():
    import numpy as np

    from src.algorithms.shac.algorithm import build_checkpoint_cagrad_telemetry

    metrics = {
        "actor_bootstrap_scale_current": 0.0,
        "actor_cagrad_bin_counts": np.ones(5),
        "actor_cagrad_bin_gradient_norms": np.ones(5),
        "actor_cagrad_bin_losses": np.arange(5.0),
        "actor_cagrad_weights": np.full(5, 0.2),
        "actor_cagrad_gram_matrix": np.eye(5),
        "actor_cagrad_cosine_matrix": np.eye(5),
        "actor_cagrad_objective": 1.0,
        "actor_cagrad_dual_gap": 0.0,
        "actor_cagrad_uniform_combined_cosine": 0.5,
        "actor_cagrad_combined_norm": 1.0,
        "actor_cagrad_valid": True,
    }

    row = build_checkpoint_cagrad_telemetry(metrics)

    assert row["actor_bootstrap_scale_current"] == 0.0
    assert row["actor_cagrad_valid"] is True
    assert row["actor_cagrad_bin_counts"] == [1.0] * 5
    assert len(row["actor_cagrad_gram_matrix"]) == 5
