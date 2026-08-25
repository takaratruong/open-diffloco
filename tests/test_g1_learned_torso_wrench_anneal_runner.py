from __future__ import annotations

from pathlib import Path


def test_anneal_grid_is_continuous_then_has_zero_tail():
    from tools.run_g1_learned_torso_wrench_anneal import (
        ANNEAL_END_STEP,
        END_STEP,
        START_STEP,
        expected_checkpoint_steps,
    )

    steps = expected_checkpoint_steps()
    assert steps[0] > START_STEP
    assert ANNEAL_END_STEP in steps
    assert steps[-1] == END_STEP
    assert END_STEP > ANNEAL_END_STEP


def test_anneal_kwargs_change_only_registered_learned_wrench_continuation(
    tmp_path: Path,
):
    from tools.run_g1_learned_torso_wrench_anneal import (
        ANNEAL_END_STEP,
        END_STEP,
        START_STEP,
        build_anneal_kwargs,
        expected_checkpoint_steps,
    )

    kwargs = build_anneal_kwargs(
        "g1-4x5", tmp_path / "reference.npz", 0, tmp_path / "parent.pkl"
    )
    assert kwargs["actor_learned_torso_wrench"] is True
    assert kwargs["allow_resume_actor_learned_torso_wrench_start"] is False
    assert kwargs["allow_resume_actor_learned_torso_wrench_change"] is True
    assert kwargs["actor_learned_torso_wrench_scale"] == 1.0
    assert kwargs["actor_learned_torso_wrench_scale_end"] == 0.0
    assert kwargs["actor_learned_torso_wrench_scale_start_step"] == START_STEP
    assert kwargs["actor_learned_torso_wrench_scale_end_step"] == ANNEAL_END_STEP
    assert kwargs["actor_learned_torso_wrench_condition_on_scale"] is True
    assert kwargs["actor_learned_torso_wrench_train_controller"] is True
    assert kwargs["actor_learned_torso_wrench_penalty"] == 0.01
    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == expected_checkpoint_steps()


def test_parser_requires_pinned_inputs():
    from tools.run_g1_learned_torso_wrench_anneal import build_parser

    args = build_parser().parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--reference-path",
            "/tmp/reference.npz",
            "--resume-from",
            "/tmp/parent.pkl",
            "--output-root",
            "/tmp/output",
            "--code-commit",
            "a" * 40,
        ]
    )
    assert args.seed == 0
