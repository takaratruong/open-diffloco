import json
from pathlib import Path

import pytest


def test_consolidation_changes_only_absolute_endpoint(tmp_path: Path) -> None:
    from tools.run_g1_frozen_residual_assistance_curriculum import (
        build_frozen_residual_assistance_kwargs,
    )
    from tools.run_g1_zero_assistance_consolidation import (
        CONSOLIDATION_END_STEP,
        build_zero_assistance_consolidation_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e012-final.pkl"
    parent = build_frozen_residual_assistance_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_zero_assistance_consolidation_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }

    assert delta == {"total_steps": CONSOLIDATION_END_STEP}
    assert candidate["torso_wrench_assistance_end_step"] == 1_622_016
    assert candidate["reference_reset_noise_scale"] == 1.0
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


def test_consolidation_parser_has_no_scientific_overrides() -> None:
    from tools.run_g1_zero_assistance_consolidation import build_parser

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
    assert args.resume_from == Path("/tmp/e012-final.pkl")

    for override in (
        ["--solver-profile", "upstream-1x5"],
        ["--reference-reset-noise-scale", "2.0"],
        ["--assistance-end-step", "1700000"],
        ["--num-envs", "512"],
        ["--total-steps", "3000000"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])


def test_validate_training_artifacts_requires_exact_zero_dense_grid(
    tmp_path: Path,
) -> None:
    from tools.run_g1_zero_assistance_consolidation import (
        CONSOLIDATION_END_STEP,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    run = tmp_path / "shac_run"
    run.mkdir()
    hparams = {
        "total_steps": CONSOLIDATION_END_STEP,
        "checkpoint_interval": 49_152,
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

    validation = validate_training_artifacts(run)

    assert validation["valid"] is True
    assert validation["checkpoint_steps"] == list(expected_checkpoint_steps())

    rows[-1]["torso_wrench_assistance_scale_current"] = 0.01
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact-zero assistance"):
        validate_training_artifacts(run)
