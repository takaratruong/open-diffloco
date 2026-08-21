from __future__ import annotations

import numpy as np
import pytest


def test_interpolate_actor_tree_preserves_endpoints_and_midpoint():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        interpolate_actor_tree,
    )

    source = {"w": np.array([0.0, 2.0], dtype=np.float32)}
    proposal = {"w": np.array([2.0, 4.0], dtype=np.float32)}

    midpoint = interpolate_actor_tree(source, proposal, alpha=0.5)
    endpoint = interpolate_actor_tree(source, proposal, alpha=1.0)

    np.testing.assert_array_equal(midpoint["w"], [1.0, 3.0])
    np.testing.assert_array_equal(endpoint["w"], proposal["w"])


def test_interpolate_actor_tree_rejects_structure_nonfinite_and_unregistered_alpha():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        interpolate_actor_tree,
    )

    source = {"w": np.array([0.0, 2.0], dtype=np.float32)}
    proposal = {"w": np.array([2.0, 4.0], dtype=np.float32)}

    with pytest.raises(ValueError, match="structure"):
        interpolate_actor_tree(source, {"x": proposal["w"]}, alpha=0.5)
    with pytest.raises(ValueError, match="finite"):
        interpolate_actor_tree(
            source,
            {"w": np.array([np.nan, 4.0], dtype=np.float32)},
            alpha=0.5,
        )
    with pytest.raises(ValueError, match="registered"):
        interpolate_actor_tree(source, proposal, alpha=0.2)


def test_gaussian_proximity_matches_closed_form_and_zero_identity():
    from tools.evaluate_g1_e023_kl_behavioral_trust import gaussian_proximity

    old = np.zeros((2, 2), dtype=np.float64)
    new = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    std = np.array([2.0, 4.0], dtype=np.float64)
    epsilon = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float64)

    report = gaussian_proximity(old, new, std, epsilon)
    identity = gaussian_proximity(old, old, std, epsilon)

    np.testing.assert_allclose(
        report["kl_by_state_action"], [[0.125, 0.0], [0.0, 0.125]]
    )
    np.testing.assert_allclose(report["kl_by_state"], [0.125, 0.125])
    assert report["mean_kl"] == pytest.approx(0.0625)
    assert report["p95_state_kl"] == pytest.approx(0.125)
    assert report["max_state_kl"] == pytest.approx(0.125)
    assert np.isfinite(report["log_ratio_by_state"]).all()
    assert identity["mean_kl"] == 0.0
    assert identity["p95_state_kl"] == 0.0
    assert identity["max_state_kl"] == 0.0
    np.testing.assert_array_equal(identity["log_ratio_by_state"], 0.0)


@pytest.mark.parametrize(
    ("std", "match"),
    [
        (np.array([1.0]), "shape"),
        (np.array([1.0, 0.0]), "positive"),
        (np.array([1.0, np.nan]), "finite"),
    ],
)
def test_gaussian_proximity_rejects_invalid_standard_deviation(std, match):
    from tools.evaluate_g1_e023_kl_behavioral_trust import gaussian_proximity

    values = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=match):
        gaussian_proximity(values, values, std, values)


def _row(alpha, *, kl=True, survival=(116, 99, 67, 49, 24)):
    return {
        "alpha": alpha,
        "kl_feasible": kl,
        "survival": list(survival),
        "valid": True,
    }


def test_selector_chooses_largest_safe_alpha_and_marks_advancement():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    rows = [_row(alpha, kl=alpha <= 0.25) for alpha in CANDIDATE_ALPHAS]
    rows[2]["survival"] = [117, 99, 67, 49, 24]

    result = select_kl_behavioral_candidate(rows)

    assert result["outcome"] == "kl-trust-advances"
    assert result["selected_alpha"] == 0.25


def test_selector_does_not_compensate_for_any_phase_regression():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    rows = [_row(alpha, kl=False) for alpha in CANDIDATE_ALPHAS]
    rows[-1] = _row(
        CANDIDATE_ALPHAS[-1],
        kl=True,
        survival=(124, 99, 74, 49, 23),
    )

    result = select_kl_behavioral_candidate(rows)

    assert result["outcome"] == "kl-feasible-behaviorally-unsafe"
    assert result["selected_alpha"] is None


def test_selector_distinguishes_preservation_and_no_kl_candidate():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    preserving = [_row(alpha, kl=alpha == 0.125) for alpha in CANDIDATE_ALPHAS]
    absent = [_row(alpha, kl=False) for alpha in CANDIDATE_ALPHAS]

    assert (
        select_kl_behavioral_candidate(preserving)["outcome"]
        == "kl-trust-preserves-only"
    )
    assert (
        select_kl_behavioral_candidate(absent)["outcome"]
        == "no-nonzero-kl-candidate"
    )


def test_selector_rejects_incomplete_or_reordered_candidate_grid():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    rows = [_row(alpha) for alpha in CANDIDATE_ALPHAS]
    with pytest.raises(ValueError, match="candidate grid"):
        select_kl_behavioral_candidate(rows[:-1])
    with pytest.raises(ValueError, match="candidate grid"):
        select_kl_behavioral_candidate(list(reversed(rows)))
