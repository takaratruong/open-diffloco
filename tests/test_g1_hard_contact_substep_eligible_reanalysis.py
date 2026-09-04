import numpy as np
import pytest

from experiments.g1_hard_contact_substep_eligible_reanalysis.run import (
    classify_eligible_substeps,
    primal_eligibility,
)


def test_primal_eligibility_uses_only_finite_direct_and_ad_primals():
    source = np.ones((10, 2), dtype=np.float64)
    reverse = source.copy()
    forward = source.copy()
    reverse[9, 0] += 1e-3
    forward[9, 0] += 1e-3

    result = primal_eligibility(source, reverse, forward)

    assert result.tolist() == [True] * 9 + [False]


def test_primal_eligibility_rejects_nonfinite_values():
    source = np.ones((10, 2), dtype=np.float64)
    reverse = source.copy()
    forward = source.copy()
    source[3, 1] = np.nan

    result = primal_eligibility(source, reverse, forward)

    assert result.tolist() == [True, True, True, False] + [True] * 6


@pytest.mark.parametrize(
    ("treatment_passes", "expected"),
    [
        (0, "first-substep-systematic-on-eligible-cases"),
        (7, "first-and-later-substeps-both-contribute-on-eligible-cases"),
        (9, "ad-inconsistency-emerges-after-first-substep-on-eligible-cases"),
    ],
)
def test_classification_follows_frozen_eligible_count(treatment_passes, expected):
    eligible = np.asarray([True] * 9 + [False])
    control = np.asarray([False] * 10)
    treatment = np.asarray(
        [True] * treatment_passes + [False] * (9 - treatment_passes) + [False]
    )

    result = classify_eligible_substeps(
        measurement_valid=True,
        eligible=eligible,
        control_gradient_agreement=control,
        treatment_gradient_agreement=treatment,
    )

    assert result["outcome"] == expected
    assert result["scientifically_interpretable"] is True
    assert result["eligible_case_count"] == 9
    assert result["control_agreement_count"] == 0
    assert result["treatment_agreement_count"] == treatment_passes


def test_classification_fails_closed_on_measurement_or_mask_change():
    eligible = np.asarray([True] * 9 + [False])
    control = np.asarray([False] * 10)
    treatment = np.asarray([True] * 7 + [False] * 3)

    invalid = classify_eligible_substeps(
        measurement_valid=False,
        eligible=eligible,
        control_gradient_agreement=control,
        treatment_gradient_agreement=treatment,
    )
    wrong_mask = classify_eligible_substeps(
        measurement_valid=True,
        eligible=np.asarray([True] * 8 + [False] * 2),
        control_gradient_agreement=control,
        treatment_gradient_agreement=treatment,
    )

    assert invalid["outcome"] == "invalid-measurement"
    assert wrong_mask["outcome"] == "invalid-measurement"
    assert invalid["scientifically_interpretable"] is False
    assert wrong_mask["scientifically_interpretable"] is False


def test_classification_requires_zero_control_agreements():
    eligible = np.asarray([True] * 9 + [False])
    control = np.asarray([True] + [False] * 9)
    treatment = np.asarray([True] * 7 + [False] * 3)

    result = classify_eligible_substeps(
        measurement_valid=True,
        eligible=eligible,
        control_gradient_agreement=control,
        treatment_gradient_agreement=treatment,
    )

    assert result["outcome"] == "invalid-measurement"
    assert result["scientifically_interpretable"] is False
