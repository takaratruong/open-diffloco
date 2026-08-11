from pathlib import Path

import pytest


def test_assistance_continuation_changes_only_fixed_curriculum_contract(
    tmp_path: Path,
) -> None:
    from tools.run_g1_frozen_residual_assistance_curriculum import (
        build_frozen_residual_assistance_kwargs,
    )
    from tools.run_g1_frozen_residual_preview_continuation import (
        build_frozen_residual_preview_kwargs,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e008.pkl"
    parent = build_frozen_residual_preview_kwargs("g1-4x5", reference, 0, checkpoint)
    candidate = build_frozen_residual_assistance_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    changed = {
        "total_steps",
        "torso_wrench_assistance",
        "torso_wrench_assistance_start_step",
        "torso_wrench_assistance_end_step",
        "torso_wrench_assistance_zero_fraction",
        "allow_resume_torso_wrench_assistance_change",
    }

    assert candidate["total_steps"] == 1_720_320
    assert candidate["checkpoint_interval"] == 49_152
    assert candidate["torso_wrench_assistance"] is True
    assert candidate["torso_wrench_assistance_start_step"] == 1_327_104
    assert candidate["torso_wrench_assistance_end_step"] == 1_622_016
    assert candidate["torso_wrench_assistance_zero_fraction"] == 0.25
    assert candidate["allow_resume_torso_wrench_assistance_change"] is True
    assert candidate["actor_residual_preview_adapter"] is True
    assert candidate["actor_residual_preview_hidden"] == 256
    assert candidate.get("actor_residual_preview_optimizer", "adam") == "adam"
    assert candidate["actor_cagrad"] is True
    assert candidate["gradient_accumulation_steps"] == 2
    assert candidate["unroll_length"] == 12
    assert candidate.get("carried_reset_probability", 0.0) == 0.0
    assert {key: value for key, value in candidate.items() if key not in changed} == {
        key: value for key, value in parent.items() if key not in changed
    }


def test_assistance_continuation_parser_has_no_scientific_overrides() -> None:
    from tools.run_g1_frozen_residual_assistance_curriculum import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e008.pkl",
    ]
    args = build_parser().parse_args(required)
    assert args.resume_from == Path("/tmp/e008.pkl")

    for override in (
        ["--assistance-end-step", "1600000"],
        ["--zero-fraction", "0.5"],
        ["--num-envs", "512"],
        ["--unroll-length", "24"],
        ["--total-steps", "2000000"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])
