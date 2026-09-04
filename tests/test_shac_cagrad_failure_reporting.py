from __future__ import annotations

import json

import numpy as np
import pytest


def _metrics() -> dict[str, object]:
    gradient_finite = np.ones(500, dtype=bool)
    gradient_finite[250] = False
    phases = np.repeat(np.asarray([10, 60, 120, 180, 230]), 100)
    phases[250] = 137
    support_modes = np.zeros(500, dtype=np.int32)
    support_modes[250] = 1
    terminal_modes = np.zeros(500, dtype=np.int32)
    terminal_modes[250] = 3
    losses = np.ones(500)
    losses[250] = 0.25
    return {
        "actor_cagrad_bin_counts": np.asarray([100, 100, 99, 100, 100]),
        "actor_cagrad_loss_bin_counts": np.asarray([100, 100, 100, 100, 100]),
        "actor_cagrad_bin_gradient_norms": np.arange(1.0, 6.0),
        "actor_cagrad_bin_losses": np.arange(5.0),
        "actor_cagrad_weights": np.full(5, 0.2),
        "actor_cagrad_gram_matrix": np.eye(5),
        "actor_cagrad_cosine_matrix": np.eye(5),
        "actor_cagrad_objective": 1.0,
        "actor_cagrad_dual_gap": 0.0,
        "actor_cagrad_uniform_combined_cosine": 0.5,
        "actor_cagrad_combined_norm": 1.0,
        "actor_cagrad_gradient_bins_occupied": True,
        "actor_cagrad_solver_valid": True,
        "actor_cagrad_bin_gradient_norms_finite": True,
        "actor_cagrad_reduction_valid": False,
        "actor_cagrad_loss_bins_occupied": True,
        "actor_cagrad_losses_finite": True,
        "actor_cagrad_loss_valid": True,
        "actor_cagrad_counts_match": False,
        "actor_cagrad_valid": False,
        "actor_grad_finite_fraction": 499.0 / 500.0,
        "actor_grad_raw_median": 2.0,
        "actor_grad_raw_max": np.inf,
        "actor_cagrad_gradient_finite_by_env": gradient_finite,
        "actor_cagrad_loss_finite_by_env": np.ones(500, dtype=bool),
        "actor_cagrad_start_phases": phases,
        "actor_cagrad_start_support_modes": support_modes,
        "actor_cagrad_terminal_modes": terminal_modes,
        "actor_cagrad_losses_by_env": losses,
        "actor_bootstrap_scale_current": 1.0,
        "ahac_horizon_before_update": 8.0,
        "ahac_horizon": 9.0,
        "ahac_actor_constraint_penalty": 0.0,
        "ahac_critic_head_disagreement": 0.0,
        "ahac_horizon_valid": True,
    }


def test_cagrad_failure_report_localizes_nonfinite_environment_gradients() -> None:
    from src.algorithms.shac.algorithm import build_cagrad_failure_report

    report = build_cagrad_failure_report(
        _metrics(), input_step=1_867_776, computed_output_step=1_880_064
    )

    assert report["protocol"] == "shac-cagrad-failure-v1"
    assert report["report_valid"] is True
    assert report["actor_cagrad_valid"] is False
    assert report["failure_class"] == "nonfinite-per-environment-gradients"
    assert report["phase_bins"]["missing_gradient_contributors"] == [0, 0, 1, 0, 0]
    assert report["phase_bins"]["loss_counts"] == [100] * 5
    assert report["phase_bins"]["gradient_counts"] == [100, 100, 99, 100, 100]
    assert report["actor_gradient"]["raw_norm_max"] is None
    assert report["population"]["size"] == 500
    assert report["population"]["nonfinite_gradient_environments"] == [
        {
            "index": 250,
            "loss": 0.25,
            "loss_finite": True,
            "phase": 137,
            "start_support_mode": 1,
            "terminal_mode": 3,
        }
    ]
    assert report["population"]["nonfinite_loss_environments"] == []
    assert len(report["population"]["gradient_finite_by_env"]) == 500
    assert report["population"]["gradient_finite_by_env"][250] is False
    assert report["population"]["loss_finite_by_env"] == [True] * 500
    assert report["population"]["start_phases"][250] == 137
    assert report["population"]["start_support_modes"][250] == 1
    assert report["population"]["terminal_modes"][250] == 3
    assert report["population"]["losses_by_env"][250] == 0.25
    assert report["computed_candidate_state_persisted"] is False


def test_cagrad_failure_report_is_strict_create_only_json(tmp_path) -> None:
    from src.algorithms.shac.algorithm import (
        build_cagrad_failure_report,
        persist_cagrad_failure_report,
    )

    metrics = _metrics()
    gram = np.eye(5)
    gram[2, 3] = np.nan
    metrics.update(
        actor_cagrad_bin_counts=np.full(5, 100),
        actor_cagrad_gram_matrix=gram,
        actor_cagrad_solver_valid=False,
        actor_cagrad_reduction_valid=False,
        actor_cagrad_counts_match=True,
        actor_grad_finite_fraction=1.0,
        actor_cagrad_gradient_finite_by_env=np.ones(500, dtype=bool),
    )
    report = build_cagrad_failure_report(
        metrics, input_step=10, computed_output_step=20
    )

    assert report["failure_class"] == "cagrad-solver-invalid"
    assert report["solver"]["gram_matrix_finite"] is False
    assert report["solver"]["gram_matrix"][2][3] is None

    path = persist_cagrad_failure_report(tmp_path, report)
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == report
    with pytest.raises(FileExistsError, match="already exists"):
        persist_cagrad_failure_report(tmp_path, report)
