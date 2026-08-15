from __future__ import annotations

import pytest

from tools.evaluate_g1_progressive_recovery_pair import (
    classify_outcome,
    select_progressive_candidate,
)


PARENT = ((116, 63, 49, 39, 47), (115, 63, 50, 39, 47))


def test_selector_enforces_two_repeats_and_componentwise_gate():
    candidates = {
        8: ((119, 61, 47, 37, 45), (118, 61, 48, 37, 45)),
        16: ((120, 63, 49, 39, 47), (119, 63, 50, 39, 47)),
        32: ((121, 60, 49, 39, 47), (120, 63, 50, 39, 47)),
    }

    selection = select_progressive_candidate(PARENT, candidates)

    assert selection["selected_update"] == 16
    assert selection["candidates"]["8"]["eligible"] is True
    assert selection["candidates"]["32"]["eligible"] is False
    assert selection["candidates"]["32"]["reason"] == "protected-phase-regression"


def test_selector_ranks_worst_phase_zero_then_full_vector_then_earlier():
    candidates = {
        8: ((120, 63, 49, 39, 47), (119, 63, 50, 39, 47)),
        16: ((121, 62, 49, 39, 47), (120, 63, 50, 39, 47)),
        32: ((121, 63, 49, 39, 47), (120, 63, 50, 39, 47)),
    }
    assert select_progressive_candidate(PARENT, candidates)[
        "selected_update"
    ] == 32

    tied = {8: candidates[32], 16: candidates[32]}
    assert select_progressive_candidate(PARENT, tied)["selected_update"] == 8


def test_selector_rejects_missing_repeat_or_malformed_grid():
    with pytest.raises(ValueError, match="two paired repeats"):
        select_progressive_candidate(PARENT[:1], {8: PARENT})
    with pytest.raises(ValueError, match="five finite survivals"):
        select_progressive_candidate(PARENT, {8: ((1, 2), (1, 2))})


def test_registered_outcome_map_is_complete():
    assert classify_outcome(None, support_separable=False) == "support-not-separable"
    assert classify_outcome(None, execution_valid=False) == "invalid-execution"
    assert classify_outcome(None) == "gated-recovery-insufficient"
    assert classify_outcome(
        {"selected_vector": [120, 63, 50, 40, 48]}
    ) == "gated-recovery-advances"
    assert classify_outcome(
        {"selected_vector": [499, 399, 299, 199, 99]}
    ) == "gated-recovery-solves"
