from __future__ import annotations

import numpy as np
import pytest


def _failure_report() -> dict[str, object]:
    from src.algorithms.shac.algorithm import build_cagrad_failure_report

    gradient_finite = np.ones(512, dtype=bool)
    gradient_finite[101] = False
    phases = np.arange(512, dtype=np.int32) % 272
    phases[101] = 60
    metrics = {
        "actor_cagrad_bin_counts": np.asarray([101, 98, 102, 99, 111]),
        "actor_cagrad_loss_bin_counts": np.asarray([101, 99, 102, 99, 111]),
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
        "actor_cagrad_reduction_valid": True,
        "actor_cagrad_loss_bins_occupied": True,
        "actor_cagrad_losses_finite": True,
        "actor_cagrad_loss_valid": True,
        "actor_cagrad_counts_match": False,
        "actor_cagrad_valid": False,
        "actor_grad_finite_fraction": 511.0 / 512.0,
        "actor_grad_raw_median": 2.0,
        "actor_grad_raw_max": np.inf,
        "actor_cagrad_gradient_finite_by_env": gradient_finite,
        "actor_cagrad_loss_finite_by_env": np.ones(512, dtype=bool),
        "actor_cagrad_start_phases": phases,
        "actor_cagrad_start_support_modes": np.zeros(512, dtype=np.int32),
        "actor_cagrad_terminal_modes": np.zeros(512, dtype=np.int32),
        "actor_cagrad_losses_by_env": np.ones(512),
        "actor_bootstrap_scale_current": 1.0,
        "ahac_horizon_before_update": 8.0,
        "ahac_horizon": 8.0,
        "ahac_actor_constraint_penalty": 0.0,
        "ahac_critic_head_disagreement": 0.0,
        "ahac_horizon_valid": True,
    }
    return build_cagrad_failure_report(
        metrics,
        input_step=1_867_776,
        computed_output_step=1_880_064,
    )


def test_failure_artifact_validation_selects_exact_localization() -> None:
    from experiments.g1_ahac_cagrad_failure_diagnostic.run import (
        validate_failure_artifact,
    )

    validation = validate_failure_artifact(_failure_report())

    assert validation["valid"] is True
    assert validation["outcome"] == "nonfinite-per-environment-gradients"
    assert validation["missing_gradient_contributor_count"] == 1
    assert validation["computed_optimizer_updates"] == 1
    assert validation["persisted_optimizer_updates"] == 0
    assert validation["policy_retained"] is False


def test_failure_artifact_validation_rejects_persisted_candidate() -> None:
    from experiments.g1_ahac_cagrad_failure_diagnostic.run import (
        validate_failure_artifact,
    )

    report = _failure_report()
    report["computed_candidate_state_persisted"] = True

    with pytest.raises(ValueError, match="no-retention"):
        validate_failure_artifact(report)
