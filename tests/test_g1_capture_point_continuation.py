from __future__ import annotations

import inspect

from tools.run_g1_capture_point_continuation import (
    CHECKPOINT_INTERVAL,
    END_STEP,
    START_STEP,
    build_capture_point_kwargs,
    expected_checkpoint_steps,
)


def test_capture_point_continuation_has_exact_update_grid() -> None:
    assert CHECKPOINT_INTERVAL == 8 * 512 * 24
    assert END_STEP == START_STEP + 32 * 512 * 24
    assert expected_checkpoint_steps() == tuple(
        START_STEP + update * 512 * 24 for update in (8, 16, 24, 32)
    )


def test_capture_point_kwargs_change_only_registered_treatment() -> None:
    kwargs = build_capture_point_kwargs(
        "g1-4x5",
        "/tmp/reference.npz",
        0,
        "/tmp/checkpoint.pkl",
        capture_weight=0.25,
    )

    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == expected_checkpoint_steps()
    assert kwargs["actor_frozen_controller_residual"] is True
    assert kwargs["actor_frozen_controller_residual_hidden"] == 256
    assert kwargs["allow_resume_actor_frozen_controller_residual_start"] is True
    assert kwargs["actor_capture_point_tracking"] is True
    assert kwargs["actor_capture_point_delta"] == 0.1
    assert kwargs["actor_capture_point_weight"] == 0.25
    assert kwargs["allow_resume_actor_capture_point_tracking_start"] is True
    assert kwargs["actor_learned_torso_wrench"] is False
    assert kwargs["torso_wrench_assistance"] is False
    assert kwargs["actor_cagrad"] is True
    assert kwargs["carried_reset_probability"] == 0.0
    assert kwargs["reference_reset_noise_scale"] == 0.0


def test_runner_invokes_exact_solver_and_training_validator() -> None:
    from tools import run_g1_capture_point_continuation as runner

    source = inspect.getsource(runner.main)
    assert "validate_e026_preflight(" in source
    assert "solver_context(get_solver_profile(args.solver_profile))" in source
    assert "validate_training_artifacts(" in source
