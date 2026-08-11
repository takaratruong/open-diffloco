import pytest


def _result(phase, steps, terminal=False):
    return {
        "phase": phase,
        "steps": steps,
        "terminal": terminal,
        "mean_reward": 1.0,
        "mean_anchor_position_error": 0.1,
        "mean_anchor_orientation_error": 0.2,
        "mean_body_position_error": 0.3,
        "mean_body_orientation_error": 0.4,
        "mean_body_linear_velocity_error": 0.5,
        "mean_body_angular_velocity_error": 0.6,
    }


def test_phase_grid_marks_exact_completed_suffixes_and_robust_statistics():
    from tools.evaluate_g1_rmr_phase_grid import build_phase_grid_summary

    phases = (0, 24, 48, 72, 96)
    results = [
        _result(phase, 120 - phase, terminal=False) for phase in phases
    ]
    summary = build_phase_grid_summary(
        results,
        phases=phases,
        reference_transitions=120,
    )

    assert summary["survival"] == [120, 96, 72, 48, 24]
    assert summary["completed_suffix"] == [True] * 5
    assert summary["minimum_survival"] == 24
    assert summary["median_survival"] == 72
    assert summary["mean_survival"] == 72.0


def test_phase_grid_rejects_duplicate_or_invalid_phases():
    from tools.evaluate_g1_rmr_phase_grid import build_phase_grid_summary

    with pytest.raises(ValueError, match="five unique"):
        build_phase_grid_summary(
            [_result(0, 1)] * 5,
            phases=(0, 0, 1, 2, 3),
            reference_transitions=120,
        )

