from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from tools.evaluate_g1_centroidal_propulsion_discriminator import (
    DiscriminatorMetrics,
    classify_discriminator,
    sum_external_impulse,
    validate_common_prefix,
)


def _metrics(*, p99: float, rms=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    return DiscriminatorMetrics(
        p99_forward_abs=p99,
        component_rms=tuple(rms),
        values_finite=True,
        action_gradient_finite=True,
        action_gradient_norm=1.0,
        derivative_agreement=True,
    )


def test_classifier_requires_twenty_percent_forward_separation() -> None:
    assisted = _metrics(p99=0.79)
    e026 = _metrics(p99=1.0)
    e005 = _metrics(p99=1.1)

    assert (
        classify_discriminator(assisted=assisted, e026=e026, e005=e005)
        == "propulsion-signal-valid"
    )
    assert (
        classify_discriminator(
            assisted=replace(assisted, p99_forward_abs=0.81),
            e026=e026,
            e005=e005,
        )
        == "propulsion-signal-not-discriminating"
    )


def test_classifier_rejects_other_component_regression_and_derivative_failure() -> None:
    e026 = _metrics(p99=1.0)
    e005 = _metrics(p99=1.1)
    regressed = _metrics(p99=0.7, rms=(1.0, 1.06, 1.0, 1.0, 1.0, 1.0))
    invalid = replace(_metrics(p99=0.7), derivative_agreement=False)

    assert classify_discriminator(assisted=regressed, e026=e026, e005=e005) == "propulsion-signal-not-discriminating"
    assert classify_discriminator(assisted=invalid, e026=e026, e005=e005) == "invalid-execution"


def test_summed_external_impulse_is_exactly_permutation_invariant() -> None:
    left = np.arange(6, dtype=np.float64)
    right = np.arange(6, dtype=np.float64) + 10.0

    np.testing.assert_array_equal(
        sum_external_impulse(left, right),
        sum_external_impulse(right, left),
    )


def test_common_prefix_requires_exactly_106_transitions() -> None:
    traces = {
        "e026": np.zeros((106, 6)),
        "e004": np.zeros((106, 6)),
        "e005": np.zeros((106, 6)),
    }
    validate_common_prefix(traces)
    traces["e005"] = np.zeros((105, 6))
    with pytest.raises(ValueError, match="106"):
        validate_common_prefix(traces)


def test_tracking_evaluator_persists_centroidal_state_trace() -> None:
    from tools import evaluate_g1_tracking

    source = inspect.getsource(evaluate_g1_tracking.main)
    for key in (
        "centroidal_momentum",
        "reference_centroidal_momentum",
        "centroidal_root_quaternion",
        "centroidal_window_error",
        "centroidal_window_error_normalized",
        "centroidal_window_valid",
    ):
        assert f"{key}=" in source
