import numpy as np

from experiments.g1_hard_contact_action_interpolation_reanalysis.run import (
    ALPHAS,
    CASE_COUNT,
    PHASE_CASES,
    PROBE_NAMES,
    classify_action_regimes,
    correct_nonselected_outputs,
    primal_eligibility,
    transition_records,
)


def test_correct_nonselected_outputs_preserves_alpha_then_case_axes() -> None:
    raw: dict[str, np.ndarray] = {}
    for output_index, name in enumerate(PROBE_NAMES):
        baseline = np.arange(CASE_COUNT * 2, dtype=np.float64).reshape(CASE_COUNT, 2)
        baseline += output_index * 100.0
        values = np.broadcast_to(
            baseline, (len(PHASE_CASES), ALPHAS.size, *baseline.shape)
        ).copy()
        for slot, (_, selected_index) in enumerate(PHASE_CASES):
            values[slot, :, selected_index] += np.arange(ALPHAS.size)[:, None]
        raw[f"baseline_first_{name}"] = baseline
        raw[f"first_{name}"] = values

    valid, by_output = correct_nonselected_outputs(raw)
    assert valid
    assert all(by_output.values())

    raw["first_source_primal"][0, 3, 6, 0] += 1.0
    valid, by_output = correct_nonselected_outputs(raw)
    assert not valid
    assert not by_output["source_primal"]


def test_primal_eligibility_uses_both_objectives_and_all_three_primals() -> None:
    source = np.ones((2, ALPHAS.size, 2), dtype=np.float64)
    reverse = source.copy()
    forward = source.copy()

    assert np.all(primal_eligibility(source, reverse, forward))

    reverse[0, 2, 1] += 1e-4
    forward[1, 4, 0] = np.nan
    expected = np.ones((2, ALPHAS.size), dtype=bool)
    expected[0, 2] = False
    expected[1, 4] = False
    assert np.array_equal(primal_eligibility(source, reverse, forward), expected)


def test_transition_records_preserve_direction_and_fixed_brackets() -> None:
    mask = np.asarray([True, True, False, False, True, False, False, True, False])

    records = transition_records(mask, ALPHAS)

    assert records == [
        {"left_alpha": 0.125, "right_alpha": 0.25, "from": "pass", "to": "fail"},
        {"left_alpha": 0.375, "right_alpha": 0.5, "from": "fail", "to": "pass"},
        {"left_alpha": 0.5, "right_alpha": 0.625, "from": "pass", "to": "fail"},
        {"left_alpha": 0.75, "right_alpha": 0.875, "from": "fail", "to": "pass"},
        {"left_alpha": 0.875, "right_alpha": 1.0, "from": "pass", "to": "fail"},
    ]


def test_classification_selects_multiple_regimes_for_e016_masks() -> None:
    masks = np.asarray(
        [
            [True, True, False, False, False, True, False, True, False],
            [True, True, True, False, False, True, True, True, False],
        ]
    )

    result = classify_action_regimes(
        measurement_valid=True,
        primal_valid=np.ones_like(masks),
        smooth_agreement=masks,
        alphas=ALPHAS,
    )

    assert result["valid"]
    assert result["scientifically_interpretable"]
    assert result["outcome"] == "multiple-ad-regimes-along-action-segment"
    assert result["transition_counts"] == [5, 3]
    assert result["pass_to_fail_counts"] == [3, 2]
    assert result["recovery_counts"] == [2, 1]


def test_classification_distinguishes_single_transition_and_primal_boundary() -> None:
    single = np.asarray(
        [
            [True, True, True, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False, False],
        ]
    )
    result = classify_action_regimes(
        measurement_valid=True,
        primal_valid=np.ones_like(single),
        smooth_agreement=single,
        alphas=ALPHAS,
    )
    assert result["outcome"] == "single-ad-transition-bracketed-in-both-phases"

    primal_valid = np.ones_like(single)
    primal_valid[0, 4] = False
    result = classify_action_regimes(
        measurement_valid=True,
        primal_valid=primal_valid,
        smooth_agreement=single,
        alphas=ALPHAS,
    )
    assert result["valid"]
    assert result["outcome"] == "transform-primal-boundary-along-action-segment"


def test_classification_fails_closed_on_measurement_or_endpoint_contract() -> None:
    masks = np.ones((2, ALPHAS.size), dtype=bool)
    for measurement_valid in (False, True):
        result = classify_action_regimes(
            measurement_valid=measurement_valid,
            primal_valid=np.ones_like(masks),
            smooth_agreement=masks,
            alphas=ALPHAS,
        )
        assert not result["valid"]
        assert result["outcome"] == "invalid-measurement"
