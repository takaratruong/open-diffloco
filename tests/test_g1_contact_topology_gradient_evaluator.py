import json
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_g1_contact_topology_gradients import (
    ACTORS,
    ACTION_DIM,
    HORIZON,
    PHASES,
    POPULATION,
    build_fixed_phase_population,
    classify_contact_topology_gradient_audit,
    compare_solver_gradients,
    validate_completion,
)


def _actor_evidence(**overrides):
    value = {
        "ordinary_solver_cosine": 0.8,
        "truncated_solver_cosine": 0.9,
        "ordinary_truncated_cosine": 0.9,
        "truncated_to_ordinary_norm_ratio": 1.0,
        "phase_solver_cosine_delta": [0.01] * 5,
        "event_bins": 3,
        "event_count": 24,
        "finite": True,
    }
    value.update(overrides)
    return value


def _evidence(**actor_overrides):
    return {
        "valid": True,
        "actors": {
            actor: _actor_evidence(**actor_overrides.get(actor, {}))
            for actor in ACTORS
        },
    }


def test_fixed_population_has_exact_phase_counts_and_noise() -> None:
    population = build_fixed_phase_population(0)

    assert population["phase"].shape == (POPULATION,)
    assert population["noise"].shape == (POPULATION, HORIZON, ACTION_DIM)
    assert population["noise"].dtype == np.float32
    np.testing.assert_array_equal(
        np.unique(population["phase"], return_counts=True)[0], PHASES
    )
    np.testing.assert_array_equal(
        np.unique(population["phase"], return_counts=True)[1], 24
    )


def test_fixed_population_rejects_unregistered_seed() -> None:
    with pytest.raises(ValueError, match="seed must be zero"):
        build_fixed_phase_population(1)


def test_solver_comparison_reports_relative_angular_error_reduction() -> None:
    comparison = compare_solver_gradients(
        ordinary_4x5=np.asarray([1.0, 0.0]),
        ordinary_10x20=np.asarray([0.8, 0.6]),
        truncated_4x5=np.asarray([1.0, 0.0]),
        truncated_10x20=np.asarray([0.9, np.sqrt(0.19)]),
    )

    assert comparison["ordinary_solver_cosine"] == pytest.approx(0.8)
    assert comparison["truncated_solver_cosine"] == pytest.approx(0.9)
    assert comparison["angular_error_reduction"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    (
        ({"valid": False, "actors": {}}, "invalid-execution"),
        (
            _evidence(fresh={"ordinary_truncated_cosine": 0.79}),
            "contact-truncation-destructive",
        ),
        (_evidence(), "contact-truncation-robust"),
        (
            _evidence(e023={"truncated_solver_cosine": 0.83}),
            "contact-truncation-neutral",
        ),
    ),
)
def test_classifier_uses_registered_precedence(evidence, expected) -> None:
    assert classify_contact_topology_gradient_audit(evidence) == expected


def test_classifier_rejects_insufficient_event_coverage() -> None:
    assert (
        classify_contact_topology_gradient_audit(
            _evidence(fresh={"event_bins": 2})
        )
        == "invalid-execution"
    )
    assert (
        classify_contact_topology_gradient_audit(
            _evidence(e023={"event_count": 23})
        )
        == "invalid-execution"
    )


def test_completion_reopens_and_rejects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "gradient_summary.json"
    artifact.write_text('{"valid": true}\n', encoding="utf-8")
    from tools.prepare_g1_rmr_reference import sha256_file

    completion = {
        "valid": True,
        "protocol": "g1-contact-topology-gradient-v1",
        "artifacts": {
            "gradient_summary.json": sha256_file(artifact),
        },
    }
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert validate_completion(completion_path)["valid"] is True
    artifact.write_text('{"valid": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        validate_completion(completion_path)
