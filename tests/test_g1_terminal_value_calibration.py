import math

import numpy as np
import pytest

from tools.evaluate_g1_terminal_value_calibration import (
    build_payload,
    calibration_metrics,
    classify_calibration,
    discounted_terminal_returns,
    h12_boundary_records,
    validate_code_provenance,
)


def test_discounted_terminal_returns_uses_realized_terminal_tail():
    actual = discounted_terminal_returns([1.0, 2.0, 4.0], gamma=0.5)
    np.testing.assert_allclose(actual, [3.0, 4.0, 4.0])


def test_calibration_metrics_report_scale_free_error_and_order():
    metrics = calibration_metrics([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["rank_correlation"] == pytest.approx(1.0)
    assert metrics["nrmse"] == pytest.approx(math.sqrt(14.0 / 3.0) / 4.0)
    assert metrics["bias"] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "values,returns",
    [([], []), ([1.0], [1.0, 2.0]), ([1.0, np.nan], [1.0, 2.0])],
)
def test_calibration_metrics_reject_invalid_inputs(values, returns):
    with pytest.raises(ValueError):
        calibration_metrics(values, returns)


def test_h12_boundary_records_selects_index_after_twelve_transitions():
    traces = [
        {
            "phase": 100,
            "values": list(range(15)),
            "realized_returns": [float(i + 1) for i in range(15)],
        }
    ]
    assert h12_boundary_records(traces) == [
        {
            "phase": 100,
            "transition": 12,
            "value": 12.0,
            "realized_return": 13.0,
            "relative_error": 1.0 / 13.0,
        }
    ]


def test_classification_requires_all_scalar_calibration_gates():
    assert (
        classify_calibration(
            {"rank_correlation": 0.81, "nrmse": 0.24},
            [{"relative_error": 0.25}] * 5,
        )
        == "terminal-value-calibration-adequate"
    )
    assert (
        classify_calibration(
            {"rank_correlation": 0.79, "nrmse": 0.10},
            [{"relative_error": 0.01}] * 5,
        )
        == "terminal-value-miscalibrated"
    )


def test_payload_requires_exact_registered_phase_order():
    traces = []
    for phase in (0, 100, 200, 300, 400):
        traces.append(
            {
                "phase": phase,
                "values": [0.0 for _ in range(13)],
                "realized_returns": [float(i + 1) for i in range(13)],
            }
        )
    payload = build_payload(traces, provenance={"checkpoint_sha256": "a" * 64})
    assert payload["protocol"] == "g1-terminal-value-calibration-v1"
    assert payload["outcome"] == "terminal-value-miscalibrated"
    traces[0]["phase"] = 1
    with pytest.raises(ValueError):
        build_payload(traces, provenance={})


def test_code_provenance_fails_closed_on_head_or_dirty_mismatch():
    commit = "a" * 40
    assert validate_code_provenance(
        expected_commit=commit, actual_commit=commit, dirty=False
    ) == {"code_commit": commit, "dirty": False}
    with pytest.raises(ValueError):
        validate_code_provenance(
            expected_commit=commit, actual_commit="b" * 40, dirty=False
        )
    with pytest.raises(ValueError):
        validate_code_provenance(
            expected_commit=commit, actual_commit=commit, dirty=True
        )
    assert (
        classify_calibration(
            {"rank_correlation": 0.99, "nrmse": 0.10},
            [{"relative_error": 0.26}] + [{"relative_error": 0.01}] * 4,
        )
        == "terminal-value-miscalibrated"
    )
