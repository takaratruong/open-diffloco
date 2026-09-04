from __future__ import annotations

import numpy as np
import pytest

from experiments.g1_contact_free_phase100_smoothness.run import (
    EPSILONS,
    FINE_EPSILONS,
    SOURCE_E011_RAW_SHA256,
    _validate_e011_sources,
    classify_slope_limit,
    directional_slopes,
)


def test_epsilon_grid_and_fine_window_are_frozen() -> None:
    assert np.array_equal(
        EPSILONS,
        np.asarray(
            [
                1e-2,
                3e-3,
                1e-3,
                3e-4,
                1e-4,
                3e-5,
                1e-5,
                3e-6,
                1e-6,
                3e-7,
                1e-7,
            ],
            dtype=np.float64,
        ),
    )
    assert np.array_equal(FINE_EPSILONS, np.asarray([1e-6, 3e-7, 1e-7]))
    assert np.all(np.diff(EPSILONS) < 0.0)


def test_directional_slopes_use_signed_one_sided_convention() -> None:
    base = 2.0
    derivative = -0.25
    curvature = 3.0
    plus = base + derivative * EPSILONS + curvature * EPSILONS**2
    minus = base - derivative * EPSILONS + curvature * EPSILONS**2

    result = directional_slopes(base, plus, minus, EPSILONS)

    assert np.allclose(result["positive"], derivative + curvature * EPSILONS)
    assert np.allclose(result["negative"], derivative - curvature * EPSILONS)
    assert np.allclose(result["central"], derivative)


def test_classification_accepts_common_fine_ad_limit() -> None:
    slopes = np.full(EPSILONS.shape, -0.0125)
    result = classify_slope_limit(
        measurement_valid=True,
        ad_slope=-0.0125,
        positive=slopes,
        negative=slopes,
        central=slopes,
    )

    assert result["outcome"] == "fine-window-ad-consistent"
    assert result["fine_positive_matches_ad"]
    assert result["fine_negative_matches_ad"]
    assert result["fine_central_matches_ad"]
    assert result["selected_side"] is None


def test_classification_detects_ad_selected_one_sided_branch() -> None:
    positive = np.full(EPSILONS.shape, 0.2)
    negative = np.full(EPSILONS.shape, -0.4)
    central = 0.5 * (positive + negative)
    result = classify_slope_limit(
        measurement_valid=True,
        ad_slope=0.2,
        positive=positive,
        negative=negative,
        central=central,
    )

    assert result["outcome"] == "ad-selects-one-sided-branch"
    assert result["selected_side"] == "positive"
    assert result["fine_sides_separated"]


def test_classification_detects_directional_kink_without_ad_branch() -> None:
    positive = np.full(EPSILONS.shape, 0.2)
    negative = np.full(EPSILONS.shape, -0.4)
    central = 0.5 * (positive + negative)
    result = classify_slope_limit(
        measurement_valid=True,
        ad_slope=0.8,
        positive=positive,
        negative=negative,
        central=central,
    )

    assert result["outcome"] == "one-sided-directional-kink"
    assert result["selected_side"] is None


def test_classification_detects_stable_fd_limit_disagreeing_with_ad() -> None:
    slopes = np.full(EPSILONS.shape, 0.5)
    result = classify_slope_limit(
        measurement_valid=True,
        ad_slope=0.7,
        positive=slopes,
        negative=slopes,
        central=slopes,
    )

    assert result["outcome"] == "stable-fd-limit-disagrees-with-ad"
    assert result["fine_positive_stable"]
    assert result["fine_negative_stable"]
    assert result["fine_central_stable"]


def test_classification_detects_no_stable_fine_limit() -> None:
    positive = np.zeros(EPSILONS.shape)
    negative = np.zeros(EPSILONS.shape)
    central = np.zeros(EPSILONS.shape)
    positive[-3:] = [0.1, 0.3, -0.2]
    negative[-3:] = [-0.1, 0.4, 0.25]
    central[-3:] = [0.0, 0.35, 0.025]
    result = classify_slope_limit(
        measurement_valid=True,
        ad_slope=0.2,
        positive=positive,
        negative=negative,
        central=central,
    )

    assert result["outcome"] == "no-stable-fd-limit"


def test_classification_fails_closed_on_invalid_measurement() -> None:
    slopes = np.full(EPSILONS.shape, 0.2)
    result = classify_slope_limit(
        measurement_valid=False,
        ad_slope=0.2,
        positive=slopes,
        negative=slopes,
        central=slopes,
    )

    assert result["outcome"] == "invalid-measurement"
    assert not result["scientifically_interpretable"]


def test_e011_guard_uses_final_persisted_protocol() -> None:
    case_pass = [True] * 9 + [False]
    report = {
        "protocol": "g1-reset-contact-derivative-classification-v1",
        "outcome": "contact-removes-ad-disagreement-finite-difference-unresolved",
        "raw_npz_sha256": SOURCE_E011_RAW_SHA256,
        "treatment": {
            "smooth_reference_state": {"case_pass": case_pass},
            "e002_h1_reward": {"case_pass": case_pass},
        },
    }
    audit = {
        "protocol": "g1-reset-contact-derivative-independent-audit-v1",
        "experiment": "E-20260904-011",
        "valid": True,
        "measurement_valid": True,
        "scientifically_interpretable": True,
        "outcome": "contact-removes-ad-disagreement-finite-difference-unresolved",
        "checks_passed": 21,
        "checks_total": 21,
        "contact_absent": True,
        "control_smooth_pass_count": 0,
        "treatment_smooth_pass_count": 9,
        "treatment_gradient_agreement_count": 10,
        "treatment_finite_difference_agreement_count": 9,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }

    _validate_e011_sources(report, audit)
    report["protocol"] = "g1-reset-contact-derivative-report-v1"
    with pytest.raises(ValueError, match="persisted report"):
        _validate_e011_sources(report, audit)
