import jax
import jax.numpy as jp
import numpy as np
import pytest


def test_assistance_scale_has_exact_fixed_schedule_endpoints() -> None:
    from src.algorithms.shac.torso_wrench_curriculum import (
        assistance_scale_at_step,
    )

    scale = jax.jit(
        lambda step: assistance_scale_at_step(step, start_step=100, end_step=200)
    )

    assert float(scale(jp.array(50))) == 1.0
    assert float(scale(jp.array(100))) == 1.0
    assert float(scale(jp.array(150))) == 0.5
    assert float(scale(jp.array(200))) == 0.0
    assert float(scale(jp.array(250))) == 0.0


def test_assistance_sampling_is_deterministic_bounded_and_exactly_zeroable() -> None:
    from src.algorithms.shac.torso_wrench_curriculum import (
        sample_assistance_scales,
    )

    key = jax.random.PRNGKey(17)
    first = sample_assistance_scales(
        key,
        num_envs=1024,
        scheduled_scale=jp.array(0.75),
        zero_fraction=0.25,
    )
    second = sample_assistance_scales(
        key,
        num_envs=1024,
        scheduled_scale=jp.array(0.75),
        zero_fraction=0.25,
    )

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert set(np.unique(np.asarray(first))) == {0.0, 0.75}
    assert 0.20 < float(jp.mean(first == 0.0)) < 0.30
    np.testing.assert_array_equal(
        np.asarray(
            sample_assistance_scales(
                key,
                num_envs=32,
                scheduled_scale=jp.array(0.0),
                zero_fraction=0.25,
            )
        ),
        np.zeros(32),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"enabled": "yes"},
            "torso_wrench_assistance must be boolean",
        ),
        (
            {"start_step": True},
            "start_step must be a non-negative integer",
        ),
        (
            {"start_step": 20, "end_step": 20},
            "end_step must be greater than start_step",
        ),
        (
            {"zero_fraction": float("nan")},
            r"zero_fraction must be finite and in \[0, 1\]",
        ),
        (
            {"zero_fraction": 1.1},
            r"zero_fraction must be finite and in \[0, 1\]",
        ),
        (
            {"env_variant": "go2"},
            "requires a G1 tracking environment",
        ),
    ],
)
def test_assistance_configuration_rejects_invalid_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    from src.algorithms.shac.torso_wrench_curriculum import (
        validate_torso_wrench_assistance_configuration,
    )

    settings = {
        "enabled": True,
        "start_step": 10,
        "end_step": 20,
        "zero_fraction": 0.25,
        "env_variant": "g1_tracking_rmr_50hz_source_step",
    }
    settings.update(kwargs)

    with pytest.raises(ValueError, match=message):
        validate_torso_wrench_assistance_configuration(**settings)


def test_assistance_resume_requires_authority_and_restores_active_defaults() -> None:
    from src.algorithms.shac.torso_wrench_curriculum import (
        resolve_torso_wrench_assistance_resume_settings,
    )

    legacy = {"total_steps": 100}
    requested = {
        "requested_enabled": True,
        "requested_start_step": 100,
        "requested_end_step": 200,
        "requested_zero_fraction": 0.25,
    }
    with pytest.raises(ValueError, match="must match the checkpoint"):
        resolve_torso_wrench_assistance_resume_settings(
            legacy, **requested, allow_change=False
        )
    assert resolve_torso_wrench_assistance_resume_settings(
        legacy, **requested, allow_change=True
    ) == (True, 100, 200, 0.25)

    active = {
        "torso_wrench_assistance": True,
        "torso_wrench_assistance_start_step": 100,
        "torso_wrench_assistance_end_step": 200,
        "torso_wrench_assistance_zero_fraction": 0.25,
    }
    assert resolve_torso_wrench_assistance_resume_settings(
        active,
        requested_enabled=False,
        requested_start_step=0,
        requested_end_step=1,
        requested_zero_fraction=0.0,
        allow_change=False,
    ) == (True, 100, 200, 0.25)


def test_assistance_resume_rejects_incomplete_active_metadata() -> None:
    from src.algorithms.shac.torso_wrench_curriculum import (
        resolve_torso_wrench_assistance_resume_settings,
    )

    with pytest.raises(ValueError, match="complete resume metadata"):
        resolve_torso_wrench_assistance_resume_settings(
            {"torso_wrench_assistance": True},
            requested_enabled=False,
            requested_start_step=0,
            requested_end_step=1,
            requested_zero_fraction=0.0,
            allow_change=False,
        )
