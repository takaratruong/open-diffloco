from pathlib import Path

import pytest


def test_confirmation_changes_only_resume_randomness(tmp_path: Path) -> None:
    from tools.run_g1_frozen_residual_assistance_confirmation import (
        build_frozen_residual_assistance_confirmation_kwargs,
    )
    from tools.run_g1_frozen_residual_assistance_curriculum import (
        build_frozen_residual_assistance_kwargs,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e008.pkl"
    baseline = build_frozen_residual_assistance_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    confirmation = build_frozen_residual_assistance_confirmation_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    keys = set(baseline) | set(confirmation)
    delta = {
        key: confirmation.get(key)
        for key in keys
        if baseline.get(key) != confirmation.get(key)
    }

    assert delta == {"resume_random_seed": 1}


def test_confirmation_parser_has_no_scientific_overrides() -> None:
    from tools.run_g1_frozen_residual_assistance_confirmation import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e008.pkl",
    ]
    args = build_parser().parse_args(required)
    assert args.seed == 0
    assert args.resume_from == Path("/tmp/e008.pkl")

    for override in (
        ["--resume-random-seed", "2"],
        ["--assistance-end-step", "1600000"],
        ["--num-envs", "512"],
        ["--total-steps", "2000000"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])
