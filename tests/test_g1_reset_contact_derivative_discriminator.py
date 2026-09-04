from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np


class _FakeOpt:
    def __init__(self, disableflags: int):
        self.disableflags = disableflags

    def replace(self, **updates):
        return _FakeOpt(updates.get("disableflags", self.disableflags))


class _FakeModel:
    def __init__(self, opt: _FakeOpt, marker: object):
        self.opt = opt
        self.marker = marker

    def replace(self, **updates):
        return _FakeModel(updates.get("opt", self.opt), self.marker)


def test_contact_disable_changes_only_cpu_and_mjx_disable_flags() -> None:
    from experiments.g1_reset_contact_derivative_discriminator.run import (
        disable_contact_dynamics,
    )

    marker = object()
    env = SimpleNamespace(
        mj_model=SimpleNamespace(opt=SimpleNamespace(disableflags=8)),
        mjx_model=_FakeModel(_FakeOpt(8), marker),
    )

    result = disable_contact_dynamics(env)

    assert result is env
    assert env.mj_model.opt.disableflags != 8
    assert env.mjx_model.opt.disableflags == env.mj_model.opt.disableflags
    assert env.mjx_model.marker is marker


def _report(
    values: list[bool],
    *,
    gradient: list[bool] | None = None,
    finite_difference: list[bool] | None = None,
) -> dict[str, object]:
    return {
        "case_pass": values,
        "gradient_agreement": values if gradient is None else gradient,
        "finite_difference_agreement": (
            values if finite_difference is None else finite_difference
        ),
    }


def test_classifier_distinguishes_contact_cause_contribution_and_persistence() -> None:
    from experiments.g1_reset_contact_derivative_discriminator.run import (
        classify_contact_discriminator,
    )

    failed = [False] * 10
    partly_valid = [True] * 6 + [False] * 4
    valid = [True] * 10

    result = classify_contact_discriminator(
        measurement_valid=True,
        contact_absent=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(valid),
    )
    assert result["outcome"] == "contact-dynamics-cause-reset-derivative-failure"

    result = classify_contact_discriminator(
        measurement_valid=True,
        contact_absent=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(
            failed, gradient=valid, finite_difference=failed
        ),
    )
    assert (
        result["outcome"]
        == "contact-removes-ad-disagreement-finite-difference-unresolved"
    )

    result = classify_contact_discriminator(
        measurement_valid=True,
        contact_absent=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(partly_valid),
    )
    assert (
        result["outcome"] == "contact-dynamics-contribute-to-reset-derivative-failure"
    )

    result = classify_contact_discriminator(
        measurement_valid=True,
        contact_absent=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(failed),
    )
    assert result["outcome"] == "contact-removal-does-not-rescue-reset-derivatives"

    result = classify_contact_discriminator(
        measurement_valid=False,
        contact_absent=True,
        control_smooth_report=_report(failed),
        treatment_smooth_report=_report(valid),
    )
    assert result["outcome"] == "invalid-measurement"
    assert result["scientifically_interpretable"] is False


def test_classifier_rejects_wrong_case_vector_and_non_boolean_values() -> None:
    from experiments.g1_reset_contact_derivative_discriminator.run import (
        classify_contact_discriminator,
    )

    with np.testing.assert_raises(ValueError):
        classify_contact_discriminator(
            measurement_valid=True,
            contact_absent=True,
            control_smooth_report=_report([False] * 9),
            treatment_smooth_report=_report([True] * 10),
        )
    with np.testing.assert_raises(ValueError):
        classify_contact_discriminator(
            measurement_valid=True,
            contact_absent=True,
            control_smooth_report=_report([False] * 10),
            treatment_smooth_report=_report([1] * 10),
        )


def test_runner_reuses_e009_control_and_runs_only_one_treatment_probe() -> None:
    from experiments.g1_reset_contact_derivative_discriminator import run

    source = inspect.getsource(run._run)
    assert "source_e009_raw" in source
    assert "source_e009_audit" in source
    assert source.count("_build_compiled_probe(") == 1
    assert source.count("compiled_probe(states, actions)") == 2
    assert "load_source_actor_policy" not in inspect.getsource(run)
    assert "train(" not in inspect.getsource(run)
    assert '"policy_evaluation_computed": False' in inspect.getsource(run)
    assert '"optimizer_update_retained": False' in inspect.getsource(run)
