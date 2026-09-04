from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.g1_hard_contact_substep_derivative_discriminator.run import (
    classify_substep_discriminator,
    set_one_physics_substep,
)


def _report(gradient: list[bool], finite_difference: list[bool] | None = None):
    return {
        "case_pass": gradient,
        "gradient_agreement": gradient,
        "finite_difference_agreement": (
            gradient if finite_difference is None else finite_difference
        ),
    }


def test_one_substep_treatment_changes_only_n_frames() -> None:
    marker = object()
    env = SimpleNamespace(n_frames=4, dt=0.02, reference_stride=1, marker=marker)

    result = set_one_physics_substep(env)

    assert result is env
    assert env.n_frames == 1
    assert env.dt == 0.02
    assert env.reference_stride == 1
    assert env.marker is marker
    with pytest.raises(ValueError, match="exactly four"):
        set_one_physics_substep(env)


def test_classifier_localizes_first_substep_full_partial_and_late_failures() -> None:
    failed = [False] * 10
    partial = [True] * 6 + [False] * 4
    passed = [True] * 10

    result = classify_substep_discriminator(
        measurement_valid=True,
        reset_contact_present=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(failed),
    )
    assert result["outcome"] == "first-substep-systematically-ad-inconsistent"

    result = classify_substep_discriminator(
        measurement_valid=True,
        reset_contact_present=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(partial),
    )
    assert result["outcome"] == "first-substep-partially-ad-inconsistent"
    assert result["treatment_gradient_agreement_count"] == 6

    result = classify_substep_discriminator(
        measurement_valid=True,
        reset_contact_present=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(passed),
    )
    assert result["outcome"] == "ad-inconsistency-emerges-after-first-substep"


def test_classifier_fails_closed_without_valid_contact_measurement() -> None:
    failed = [False] * 10
    passed = [True] * 10
    result = classify_substep_discriminator(
        measurement_valid=False,
        reset_contact_present=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(passed),
    )
    assert result["outcome"] == "invalid-measurement"
    assert not result["scientifically_interpretable"]

    result = classify_substep_discriminator(
        measurement_valid=True,
        reset_contact_present=False,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(passed),
    )
    assert result["outcome"] == "invalid-measurement"


def test_classifier_rejects_wrong_case_vectors() -> None:
    with pytest.raises(ValueError, match="vector"):
        classify_substep_discriminator(
            measurement_valid=True,
            reset_contact_present=True,
            control_smooth_report=_report([False] * 9),
            treatment_smooth_report=_report([True] * 10),
        )
    with pytest.raises(ValueError, match="vector"):
        classify_substep_discriminator(
            measurement_valid=True,
            reset_contact_present=True,
            control_smooth_report=_report([False] * 10),
            treatment_smooth_report=_report([np.bool_(True)] * 10),
        )


def test_runner_reuses_e009_control_and_runs_one_treatment_probe_twice() -> None:
    from experiments.g1_hard_contact_substep_derivative_discriminator import run

    source = inspect.getsource(run)
    runner = inspect.getsource(run._run)
    assert "source_e009_raw" in source
    assert "source_e009_audit" in source
    assert runner.count("_build_compiled_probe(env, direction)") == 1
    assert runner.count("compiled_probe(states, actions)") == 2
    assert "disable_contact_dynamics" not in source
    assert "load_source_actor_policy" not in source
    assert "train(" not in source
    assert '"policy_evaluation_computed": False' in source
    assert '"optimizer_update_retained": False' in source
