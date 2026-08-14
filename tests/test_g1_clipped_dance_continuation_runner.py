from pathlib import Path


def test_clipped_dance_continuation_changes_only_endpoint_and_clip(
    tmp_path: Path,
) -> None:
    from tools.run_g1_frozen_residual_assistance_curriculum import (
        build_frozen_residual_assistance_kwargs,
    )
    from tools.run_g1_clipped_dance_continuation import (
        CONTINUATION_END_STEP,
        build_clipped_dance_continuation_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e012-selected.pkl"
    parent = build_frozen_residual_assistance_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_clipped_dance_continuation_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }

    assert delta == {
        "actor_per_env_grad_clip": 1.0,
        "allow_resume_actor_per_env_grad_clip_change": True,
        "total_steps": CONTINUATION_END_STEP,
    }
    assert expected_checkpoint_steps() == (1_720_320, 1_769_472)


def test_clipped_dance_parser_has_no_scientific_overrides() -> None:
    import pytest

    from tools.run_g1_clipped_dance_continuation import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e012-selected.pkl",
        "--code-commit",
        "0" * 40,
    ]
    args = build_parser().parse_args(required)
    assert args.resume_from == Path("/tmp/e012-selected.pkl")

    for override in (
        ["--actor-per-env-grad-clip", "0.5"],
        ["--num-envs", "512"],
        ["--total-steps", "2000000"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])
