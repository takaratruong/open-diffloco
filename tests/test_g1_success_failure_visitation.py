import numpy as np
import pytest

from experiments.g1_success_failure_visitation.run import (
    TRANSITION_METRICS,
    compare_traces,
    selected_boundary_indices,
    summarize_trace,
    validate_source_grids,
)


def _trace(length: int, *, qpos_offset: float = 0.0, terminal: bool = True):
    metrics = np.zeros((length, len(TRANSITION_METRICS)), dtype=np.float64)
    return {
        "phase": np.arange(10, 10 + length),
        "qpos": np.full((length, 3), qpos_offset),
        "qvel": np.zeros((length, 2)),
        "model_action": np.zeros((length, 2)),
        "position_target": np.zeros((length, 2)),
        "last_action": np.zeros((length, 2)),
        "foot_support": np.zeros((length, 2), dtype=bool),
        "contact_pairs": np.zeros((length, 4), dtype=bool),
        "constraint_force_root": np.zeros((length, 2)),
        "reward": np.ones(length),
        "done": np.asarray([False] * (length - 1) + [True]),
        "terminal": np.asarray([False] * (length - 1) + [terminal]),
        "metrics": metrics,
    }


def test_selected_boundaries_cover_quartiles_and_events() -> None:
    assert selected_boundary_indices(9, (3, 7, None)) == [0, 2, 3, 4, 6, 7, 8]
    with pytest.raises(ValueError, match="at least one"):
        selected_boundary_indices(0, ())


def test_summarize_trace_classifies_terminal_and_complete_suffix() -> None:
    failed = summarize_trace(_trace(4), phase=10, remaining=6)
    assert failed["steps"] == 4
    assert failed["terminal"] is True
    assert failed["completed_suffix"] is False

    complete = summarize_trace(_trace(4, terminal=False), phase=10, remaining=4)
    assert complete["terminal"] is False
    assert complete["completed_suffix"] is True


def test_compare_traces_localizes_first_state_and_contact_divergence() -> None:
    ppo = _trace(5, terminal=False)
    diffsim = _trace(4)
    ppo["qpos"][1:, 0] = 0.2
    ppo["contact_pairs"][2:, 1] = True
    comparison, selected = compare_traces(ppo, diffsim, phase=10)
    assert comparison["overlap_steps"] == 4
    assert comparison["first_state_divergence_phase"] == 11
    assert comparison["first_contact_pair_divergence_phase"] == 12
    assert 1 in selected and 2 in selected


def test_source_grid_validation_rejects_wrong_survival() -> None:
    ppo = {
        "source": {
            "summary": {
                "phases": [0, 25, 50, 75, 100],
                "survival": [271, 246, 221, 196, 171],
                "completed_suffix": [True] * 5,
            }
        }
    }
    diffsim = {
        "summary": {
            "phases": [0, 25, 50, 75, 100],
            "survival": [136, 144, 84, 90, 79],
            "completed_suffix": [False] * 5,
        }
    }
    validate_source_grids(ppo, diffsim)
    diffsim["summary"]["survival"][0] = 135
    with pytest.raises(ValueError, match="outcomes changed"):
        validate_source_grids(ppo, diffsim)
