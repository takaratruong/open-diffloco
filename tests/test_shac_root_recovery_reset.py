import inspect

import pytest


def test_root_reset_resume_treats_missing_metadata_as_legacy_baseline() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_root_reset_noise_resume_settings,
    )

    with pytest.raises(ValueError, match="allow_resume_reference_root"):
        resolve_root_reset_noise_resume_settings(
            {},
            requested_multiplier=2.0,
            requested_probability=0.5,
            allow_change=False,
        )

    assert resolve_root_reset_noise_resume_settings(
        {},
        requested_multiplier=2.0,
        requested_probability=0.5,
        allow_change=True,
    ) == (2.0, 0.5)


def test_root_reset_resume_restores_or_explicitly_changes_checkpoint() -> None:
    from src.algorithms.shac.algorithm import (
        resolve_root_reset_noise_resume_settings,
    )

    hparams = {
        "reference_root_reset_noise_multiplier": 1.5,
        "reference_root_reset_noise_probability": 0.25,
    }

    assert resolve_root_reset_noise_resume_settings(
        hparams,
        requested_multiplier=1.0,
        requested_probability=0.0,
        allow_change=False,
    ) == (1.5, 0.25)
    with pytest.raises(ValueError, match="allow_resume_reference_root"):
        resolve_root_reset_noise_resume_settings(
            hparams,
            requested_multiplier=2.0,
            requested_probability=0.5,
            allow_change=False,
        )
    assert resolve_root_reset_noise_resume_settings(
        hparams,
        requested_multiplier=2.0,
        requested_probability=0.5,
        allow_change=True,
    ) == (2.0, 0.5)


def test_root_reset_train_signature_exposes_default_off_treatment() -> None:
    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters

    assert parameters["reference_root_reset_noise_multiplier"].default == 1.0
    assert parameters["reference_root_reset_noise_probability"].default == 0.0
    assert (
        parameters["allow_resume_reference_root_reset_noise_change"].default
        is False
    )
