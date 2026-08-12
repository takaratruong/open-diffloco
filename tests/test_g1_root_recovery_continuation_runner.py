import json
from pathlib import Path

import pytest


def test_root_recovery_preflight_hashes_runtime_assets(tmp_path: Path) -> None:
    from src.envs.g1_tracking.environment import (
        DEFAULT_CONTROLLER_PATH,
        DEFAULT_MODEL_PATH,
    )
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets

    assets = validate_runtime_assets(
        Path(DEFAULT_MODEL_PATH), Path(DEFAULT_CONTROLLER_PATH)
    )

    assert assets["model_sha256"] == (
        "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
    )
    assert assets["controller_sha256"] == (
        "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
    )
    wrong = tmp_path / "wrong.xml"
    wrong.write_text("<mujoco/>", encoding="utf-8")
    with pytest.raises(ValueError, match="model SHA-256"):
        validate_runtime_assets(wrong, Path(DEFAULT_CONTROLLER_PATH))


def test_root_recovery_changes_only_registered_distribution_and_endpoint(
    tmp_path: Path,
) -> None:
    from tools.run_g1_frozen_residual_assistance_curriculum import (
        build_frozen_residual_assistance_kwargs,
    )
    from tools.run_g1_root_recovery_continuation import (
        ROOT_RECOVERY_MULTIPLIER,
        ROOT_RECOVERY_PROBABILITY,
        build_root_recovery_continuation_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e012-final.pkl"
    parent = build_frozen_residual_assistance_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_root_recovery_continuation_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }

    assert delta == {
        "total_steps": 2_113_536,
        "reference_root_reset_noise_multiplier": ROOT_RECOVERY_MULTIPLIER,
        "reference_root_reset_noise_probability": ROOT_RECOVERY_PROBABILITY,
        "allow_resume_reference_root_reset_noise_change": True,
    }
    assert expected_checkpoint_steps() == (
        1_769_472,
        1_818_624,
        1_867_776,
        1_916_928,
        1_966_080,
        2_015_232,
        2_064_384,
        2_113_536,
    )


def test_root_recovery_parser_has_no_scientific_overrides() -> None:
    from tools.run_g1_root_recovery_continuation import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e012-final.pkl",
        "--code-commit",
        "0" * 40,
    ]
    args = build_parser().parse_args(required)
    assert args.seed == 0
    for override in (
        ["--root-multiplier", "3.0"],
        ["--root-probability", "1.0"],
        ["--num-envs", "512"],
        ["--total-steps", "3000000"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])


def test_root_recovery_postvalidation_requires_exact_treatment(
    tmp_path: Path,
) -> None:
    from tools.run_g1_root_recovery_continuation import (
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    run = tmp_path / "shac_run"
    run.mkdir()
    hparams = {
        "total_steps": 2_113_536,
        "torso_wrench_assistance": True,
        "torso_wrench_assistance_start_step": 1_327_104,
        "torso_wrench_assistance_end_step": 1_622_016,
        "torso_wrench_assistance_zero_fraction": 0.25,
        "reference_reset_noise_scale": 1.0,
        "reference_root_reset_noise_multiplier": 2.0,
        "reference_root_reset_noise_probability": 0.5,
        "allow_resume_reference_root_reset_noise_change": True,
        "domain_randomization": True,
        "actor_cagrad": True,
        "actor_residual_preview_adapter": True,
        "gradient_accumulation_steps": 2,
        "unroll_length": 12,
        "num_envs": 256,
        "effective_num_envs": 512,
    }
    (run / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
    rows = []
    for step in expected_checkpoint_steps():
        (run / f"checkpoint_step_{step}.pkl").write_bytes(b"checkpoint")
        rows.append(
            {
                "step": step,
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
            }
        )
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    assert validate_training_artifacts(run)["valid"] is True

    hparams["reference_root_reset_noise_probability"] = 0.25
    (run / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
    with pytest.raises(ValueError, match="probability"):
        validate_training_artifacts(run)
