from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.objective_direction_audit import (
    aggregate_audit_direction,
    classify_objective_direction_audit,
    normalized_descent_proposal,
    ordinary_componentwise_safe,
    select_carried_safe_candidate,
    tree_cosine,
    tree_dot,
    tree_l2_norm,
)


def test_tree_geometry_handles_nested_numeric_pytrees() -> None:
    first = {"a": jnp.asarray([1.0, 0.0]), "b": (jnp.asarray([2.0]),)}
    aligned = {"a": jnp.asarray([2.0, 0.0]), "b": (jnp.asarray([4.0]),)}
    opposed = {"a": jnp.asarray([-1.0, 0.0]), "b": (jnp.asarray([-2.0]),)}
    orthogonal = {"a": jnp.asarray([0.0, 1.0]), "b": (jnp.asarray([0.0]),)}

    assert float(tree_l2_norm(first)) == pytest.approx(np.sqrt(5.0))
    assert float(tree_dot(first, aligned)) == pytest.approx(10.0)
    assert float(tree_cosine(first, aligned)) == pytest.approx(1.0)
    assert float(tree_cosine(first, opposed)) == pytest.approx(-1.0)
    assert float(tree_cosine(first, orthogonal)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "gradient",
    [
        {"p": jnp.zeros((2,))},
        {"p": jnp.asarray([jnp.nan, 1.0])},
    ],
)
def test_direction_geometry_fails_closed_on_invalid_gradient(gradient) -> None:
    params = {"p": jnp.asarray([1.0, 2.0])}
    with pytest.raises(ValueError):
        normalized_descent_proposal(params, gradient, displacement=0.25)


def test_normalized_descent_proposal_has_exact_requested_norm() -> None:
    params = {"p": jnp.asarray([1.0, 2.0]), "q": jnp.asarray([-3.0])}
    gradient = {"p": jnp.asarray([3.0, 4.0]), "q": jnp.asarray([0.0])}

    proposal = normalized_descent_proposal(
        params, gradient, displacement=0.25
    )
    delta = {key: proposal[key] - params[key] for key in params}

    assert float(tree_l2_norm(delta)) == pytest.approx(0.25)
    assert float(tree_dot(delta, gradient)) < 0.0


def test_aggregate_audit_direction_clips_before_five_bin_cagrad() -> None:
    gradients = {"p": jnp.asarray([[10.0], [1.0], [1.0], [1.0], [1.0]])}
    phases = jnp.asarray([0, 20, 40, 60, 80], dtype=jnp.int32)

    result = aggregate_audit_direction(
        gradients,
        phases,
        phase_count=100,
        clip_norm=1.0,
        alpha=0.5,
        iterations=32,
    )

    assert result.valid is True
    assert result.env_counts.tolist() == [1, 1, 1, 1, 1]
    np.testing.assert_allclose(np.asarray(result.task_gradients["p"]), 1.0)
    assert float(tree_l2_norm(result.combined_gradient)) > 0.0
    assert np.isfinite(np.asarray(result.cosine_matrix)).all()


def test_select_carried_safe_candidate_uses_registered_ordering() -> None:
    baseline = [3, 3, 3]
    rows = [
        {"multiplier": 0.5, "candidate_survival": [4, 3, 3]},
        {"multiplier": 0.25, "candidate_survival": [4, 4, 3]},
        {"multiplier": 0.125, "candidate_survival": [4, 4, 3]},
        {"multiplier": 1.0, "candidate_survival": [5, 2, 3]},
    ]

    selected = select_carried_safe_candidate(
        rows, baseline_survival=baseline
    )

    assert selected is not None
    assert selected["multiplier"] == 0.125
    assert selected["improved_count"] == 2
    assert selected["regressed_count"] == 0


def test_select_carried_safe_candidate_rejects_regression_and_no_gain() -> None:
    assert (
        select_carried_safe_candidate(
            [
                {"multiplier": 0.25, "candidate_survival": [3, 3]},
                {"multiplier": 0.5, "candidate_survival": [4, 2]},
            ],
            baseline_survival=[3, 3],
        )
        is None
    )


def test_ordinary_componentwise_safe_requires_preservation() -> None:
    assert ordinary_componentwise_safe([116, 64, 49, 39, 47])
    assert not ordinary_componentwise_safe([117, 63, 49, 39, 46])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"execution_valid": False}, "invalid-execution"),
        (
            {"h24_tape_cosine": 0.49},
            "stochastic-gradient-inconsistent",
        ),
        (
            {"full_gate_by_direction": {"h24": True}},
            "current-h24-direction-useful",
        ),
        (
            {"full_gate_by_direction": {"h48": True}},
            "short-horizon-credit-misaligned",
        ),
        (
            {"full_gate_by_direction": {"bootstrap": True}},
            "terminal-bootstrap-useful",
        ),
        (
            {"h24_h48_phase_cosines": [-0.1, -0.2, 0.4, 0.5, 0.6]},
            "cross-state-objective-conflict",
        ),
        (
            {
                "aggregate_pairwise_cosines": [0.81, 0.9, 0.85],
                "h24_h48_phase_cosines": [0.9] * 5,
                "every_direction_mixed": False,
            },
            "aligned-local-step-insufficient",
        ),
        (
            {
                "aggregate_pairwise_cosines": [0.7, 0.6, 0.5],
                "h24_h48_phase_cosines": [0.4] * 5,
                "every_direction_mixed": False,
            },
            "direction-audit-inconclusive",
        ),
    ],
)
def test_classification_follows_registered_precedence(overrides, expected) -> None:
    evidence = {
        "execution_valid": True,
        "h24_tape_cosine": 0.75,
        "h24_tape_phase_cosines": [0.75] * 5,
        "h24_h48_cosine": 0.5,
        "h24_h48_phase_cosines": [0.5] * 5,
        "aggregate_pairwise_cosines": [0.5, 0.5, 0.5],
        "full_gate_by_direction": {
            "h24": False,
            "h48": False,
            "bootstrap": False,
        },
        "h48_carried_strictly_better": False,
        "every_direction_mixed": True,
    }
    evidence.update(overrides)
    if "full_gate_by_direction" in overrides:
        evidence["full_gate_by_direction"] = {
            "h24": False,
            "h48": False,
            "bootstrap": False,
            **overrides["full_gate_by_direction"],
        }

    assert classify_objective_direction_audit(**evidence) == expected
