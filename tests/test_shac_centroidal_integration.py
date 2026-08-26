from __future__ import annotations

import inspect

import pytest


def test_centroidal_and_frozen_controller_flags_are_default_off() -> None:
    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters
    assert parameters["actor_frozen_controller_residual"].default is False
    assert parameters["actor_frozen_controller_residual_hidden"].default == 256
    assert (
        parameters["allow_resume_actor_frozen_controller_residual_start"].default
        is False
    )
    assert parameters["actor_centroidal_propulsion"].default is False
    assert parameters["actor_centroidal_window"].default == 4
    assert parameters["actor_centroidal_delta"].default == 0.1
    assert parameters["actor_centroidal_weight"].default == 1.0
    assert (
        parameters["allow_resume_actor_centroidal_propulsion_start"].default
        is False
    )


def test_frozen_controller_upgrade_requires_explicit_resume_authority() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_frozen_controller_residual_resume_setting,
    )

    legacy = {"actor_residual_preview_adapter": True}
    with pytest.raises(ValueError, match="explicit authority"):
        resolve_frozen_controller_residual_resume_setting(
            legacy,
            requested=True,
            requested_hidden=256,
            allow_start=False,
            is_resume=True,
        )
    assert resolve_frozen_controller_residual_resume_setting(
        legacy,
        requested=True,
        requested_hidden=256,
        allow_start=True,
        is_resume=True,
    ) == (True, 256, True)


def test_centroidal_upgrade_requires_exact_registered_settings() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_centroidal_propulsion_resume_settings,
    )

    legacy = {"actor_centroidal_propulsion": False}
    with pytest.raises(ValueError, match="explicit authority"):
        resolve_centroidal_propulsion_resume_settings(
            legacy,
            requested_enabled=True,
            requested_window=4,
            requested_delta=0.1,
            requested_weight=1.0,
            allow_start=False,
            is_resume=True,
        )
    assert resolve_centroidal_propulsion_resume_settings(
        legacy,
        requested_enabled=True,
        requested_window=4,
        requested_delta=0.1,
        requested_weight=1.0,
        allow_start=True,
        is_resume=True,
    ) == (True, 4, 0.1, 1.0, True)


def test_train_wires_nested_controller_and_window_objective() -> None:
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)
    assert "migrate_frozen_controller_residual(" in source
    assert "apply_frozen_controller_residual(" in source
    assert "update_frozen_controller_residual(" in source
    assert "mjx_centroidal_momentum(" in source
    assert "centroidal_window_objective(" in source
    assert '"actor_centroidal_propulsion"' in source
    assert '"actor_centroidal_valid_window_count"' in source


def test_centroidal_treatment_rejects_assistance_and_unfrozen_parent() -> None:
    from src.algorithms.shac.algorithm import (
        validate_centroidal_propulsion_configuration,
    )

    valid = {
        "enabled": True,
        "window": 4,
        "delta": 0.1,
        "weight": 1.0,
        "frozen_controller_residual": True,
        "actor_residual_preview_adapter": True,
        "torso_wrench_assistance": False,
        "actor_learned_torso_wrench": False,
        "unroll_length": 24,
        "env_variant": "g1_tracking_rmr50_shac",
    }
    validate_centroidal_propulsion_configuration(**valid)
    for change in (
        {"frozen_controller_residual": False},
        {"torso_wrench_assistance": True},
        {"actor_learned_torso_wrench": True},
        {"unroll_length": 3},
    ):
        with pytest.raises(ValueError):
            validate_centroidal_propulsion_configuration(
                **{**valid, **change}
            )
