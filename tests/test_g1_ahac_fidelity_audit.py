from __future__ import annotations

import math
import types

import numpy as np
import pytest


def _parity(steps: int) -> dict[str, np.ndarray]:
    return {
        "action_exact": np.ones((steps,), dtype=bool),
        "qpos_exact": np.ones((steps,), dtype=bool),
        "qvel_exact": np.ones((steps,), dtype=bool),
        "obs_exact": np.ones((steps,), dtype=bool),
        "reward_exact": np.ones((steps,), dtype=bool),
        "done_exact": np.ones((steps,), dtype=bool),
        "phase_exact": np.ones((steps,), dtype=bool),
        "rng_exact": np.ones((steps,), dtype=bool),
        "action_max_abs_delta": np.zeros((steps,), dtype=np.float64),
        "qpos_max_abs_delta": np.zeros((steps,), dtype=np.float64),
        "qvel_max_abs_delta": np.zeros((steps,), dtype=np.float64),
        "obs_max_abs_delta": np.zeros((steps,), dtype=np.float64),
        "reward_max_abs_delta": np.zeros((steps,), dtype=np.float64),
    }


def test_contact_summary_uses_population_step_means_for_ahac_threshold() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        summarize_contact_population,
    )

    root = np.tile(np.array([[10.0, 20.0], [30.0, 40.0]]), (12, 1))
    spatial = np.tile(
        np.array([[100.0, 300.0], [200.0, 1_000.0]]), (12, 1)
    )

    summary = summarize_contact_population(root, spatial, parity=_parity(24))

    assert summary["classification"] == "official-C500-active-in-mjx-scale"
    assert summary["official_threshold_individual_exceedance_count"] == 12
    assert summary["official_threshold_step_mean_exceedance_count"] == 12
    assert summary["all_body_spatial"]["step_means"][:2] == [200.0, 600.0]
    proposal = summary["aggregation_matched_calibration_proposal"]
    assert proposal["threshold"] == 400.0
    assert proposal["statistic"] == "median-of-per-step-effective-512-means"
    assert summary["individual_sample_median"]["threshold"] == 250.0
    assert "aggregation-mismatched" in summary["individual_sample_median"]["status"]


def test_official_threshold_can_be_individually_exceeded_but_population_inert() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        summarize_contact_population,
    )

    root = np.full((24, 4), 25.0)
    spatial = np.tile(
        np.array(
            [[1_000.0, 100.0, 100.0, 100.0], [900.0, 100.0, 100.0, 100.0]]
        ),
        (12, 1),
    )

    summary = summarize_contact_population(root, spatial, parity=_parity(24))

    assert summary["official_threshold_individual_exceedance_count"] == 24
    assert summary["official_threshold_step_mean_exceedance_count"] == 0
    assert summary["classification"] == "official-C500-inert-in-mjx-scale"


def test_stationary_horizon_response_stops_before_unmeasured_slot() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        stationary_projected_horizon_response,
    )

    response = stationary_projected_horizon_response(
        np.full((24,), 125.0),
        threshold=1.0,
        initial_horizon=24,
        update_budget=1_000,
    )

    assert response["measurement_support_exhausted"] is True
    assert response["simulated_updates"] < 1_000
    assert response["end_rounded_horizon"] == 25
    assert response["first_rounded_extension_update"] == response["simulated_updates"]


def test_stationary_official_scale_is_exactly_inert() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        stationary_projected_horizon_response,
    )

    response = stationary_projected_horizon_response(
        np.linspace(107.0, 125.0, 24),
        threshold=500.0,
        initial_horizon=8,
        update_budget=2_000,
    )

    assert response["simulated_updates"] == 2_000
    assert response["measurement_support_exhausted"] is False
    assert response["end_horizon"] == 8.0
    assert response["dual_max"] == 0.0
    assert response["first_rounded_extension_update"] is None


def test_summary_rejects_any_metric_dependent_trajectory_change() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        summarize_contact_population,
    )

    parity = _parity(2)
    parity["qpos_exact"][1] = False
    parity["qpos_max_abs_delta"][1] = 1e-12

    with pytest.raises(ValueError, match="changed the paired trajectory"):
        summarize_contact_population(
            np.ones((2, 2)), np.ones((2, 2)), parity=parity
        )


def test_contact_action_gradient_summary_closes_population_moments() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        summarize_contact_action_gradients,
    )

    gradients = np.array([[3.0, 4.0], [0.0, 5.0]])
    summary = summarize_contact_action_gradients(
        np.array([10.0, 20.0]),
        gradients,
        trace_values=np.array([10.0, 20.0]),
    )

    expected_mean_norm = np.linalg.norm(np.mean(gradients, axis=0))
    expected_rms_norm = np.sqrt(np.mean(np.sum(gradients**2, axis=1)))
    assert summary["valid"] is True
    assert summary["finite_fraction"] == 1.0
    assert summary["population_mean_norm"] == pytest.approx(expected_mean_norm)
    assert summary["population_rms_norm"] == pytest.approx(expected_rms_norm)
    assert summary["population_cancellation_ratio"] == pytest.approx(
        expected_mean_norm / expected_rms_norm
    )


def test_metric_scans_compile_as_two_independent_graphs() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        execute_separately_compiled_metric_scans,
    )

    compiled = []

    def fake_jit(function):
        compiled.append(function)
        return function

    def root_scan(initial, noise):
        return initial + noise, {"contact": initial - noise}

    def spatial_scan(initial, noise):
        return initial * noise, {"contact": initial + noise}

    root_result, spatial_result = execute_separately_compiled_metric_scans(
        root_scan,
        spatial_scan,
        3,
        2,
        compile_fn=fake_jit,
    )

    assert compiled == [root_scan, spatial_scan]
    assert root_result == (5, {"contact": 1})
    assert spatial_result == (6, {"contact": 5})


def test_upstream_semantics_audit_detects_missing_equation_10_actor_term() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        inspect_upstream_ahac_semantics,
    )

    source = """
class AHAC:
    def compute_actor_loss(self):
        return self.cfs.sum()

    def train(self):
        self.lambd = self.lambd + self.cfs - self.C
        self.H = self.H + self.lambd.sum()
"""
    summary = inspect_upstream_ahac_semantics(source)

    assert summary["upstream_actor_reads_contact_signal"] is True
    assert summary["upstream_actor_reads_dual"] is False
    assert summary["upstream_actor_reads_contact_threshold"] is False
    assert summary["upstream_train_updates_dual_and_horizon"] is True
    assert summary["paper_source_actor_objective_divergence"] is True


def test_evaluation_environment_exposes_metric_selector() -> None:
    import inspect

    from tools.evaluate_g1_tracking import make_evaluation_env

    signature = inspect.signature(make_evaluation_env)
    assert signature.parameters["contact_stiffness_metric"].default == (
        "root_generalized"
    )


def test_audit_constants_bind_h64_tape_without_training() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        EFFECTIVE_NUM_ENVS,
        LOCAL_UPDATE_BUDGET,
        MEASURED_STEPS,
        OFFICIAL_CONTACT_THRESHOLD,
        PROSPECTIVE_MAX_HORIZON,
        START_STEP,
        UPSTREAM_DEFAULT_UPDATE_BUDGET,
    )

    assert START_STEP == 1_867_776
    assert EFFECTIVE_NUM_ENVS == 512
    assert MEASURED_STEPS == 24
    assert PROSPECTIVE_MAX_HORIZON == 64
    assert OFFICIAL_CONTACT_THRESHOLD == 500.0
    assert LOCAL_UPDATE_BUDGET == 128
    assert UPSTREAM_DEFAULT_UPDATE_BUDGET == 2_000
    assert UPSTREAM_DEFAULT_UPDATE_BUDGET > LOCAL_UPDATE_BUDGET
    assert math.isclose(UPSTREAM_DEFAULT_UPDATE_BUDGET / LOCAL_UPDATE_BUDGET, 15.625)


def test_paper_critic_migration_discards_legacy_delayed_target() -> None:
    from experiments.g1_ahac_fidelity_audit.run import (
        critic_migration_receipt,
    )

    state = types.SimpleNamespace(
        critic_params={"params": {"Dense_0": {"kernel": np.array([1.0])}}},
        target_critic_params={
            "params": {"Dense_0": {"kernel": np.array([2.0])}}
        },
    )

    receipt = critic_migration_receipt(state)

    assert receipt["valid"] is True
    assert receipt["uses_delayed_target_critic"] is False
    assert receipt["legacy_target_would_be_discarded"] is True
    assert receipt["critic_head_sha256"] == receipt["target_critic_head_sha256"]
    assert receipt["source_target_critic_sha256"] != receipt[
        "source_critic_sha256"
    ]
