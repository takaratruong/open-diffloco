import pytest


def test_per_env_grad_clip_resume_preserves_saved_value_by_default() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_actor_per_env_grad_clip_resume_setting,
    )

    assert (
        resolve_actor_per_env_grad_clip_resume_setting(
            {"actor_per_env_grad_clip": None},
            requested=None,
            allow_change=False,
        )
        is None
    )
    assert resolve_actor_per_env_grad_clip_resume_setting(
        {"actor_per_env_grad_clip": 1.0},
        requested=None,
        allow_change=False,
    ) == pytest.approx(1.0)


def test_per_env_grad_clip_resume_requires_explicit_change_authority() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_actor_per_env_grad_clip_resume_setting,
    )

    with pytest.raises(ValueError, match="explicit resume authority"):
        resolve_actor_per_env_grad_clip_resume_setting(
            {"actor_per_env_grad_clip": None},
            requested=1.0,
            allow_change=False,
        )

    assert resolve_actor_per_env_grad_clip_resume_setting(
        {"actor_per_env_grad_clip": None},
        requested=1.0,
        allow_change=True,
    ) == pytest.approx(1.0)


def test_per_env_grad_clip_resume_fails_closed_on_missing_metadata() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_actor_per_env_grad_clip_resume_setting,
    )

    with pytest.raises(ValueError, match="complete resume metadata"):
        resolve_actor_per_env_grad_clip_resume_setting(
            {},
            requested=1.0,
            allow_change=True,
        )


def test_per_env_grad_clip_resume_authority_must_be_boolean() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_actor_per_env_grad_clip_resume_setting,
    )

    with pytest.raises(ValueError, match="must be boolean"):
        resolve_actor_per_env_grad_clip_resume_setting(
            {"actor_per_env_grad_clip": None},
            requested=1.0,
            allow_change=1,
        )
