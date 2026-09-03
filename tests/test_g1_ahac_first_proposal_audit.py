from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)


def _actor(adapter_value: float) -> FrozenControllerResidualParams:
    return FrozenControllerResidualParams(
        parent={"frozen": jnp.asarray([7.0, 8.0])},
        adapter={"trainable": jnp.asarray([adapter_value, adapter_value + 2.0])},
    )


def test_interpolation_preserves_parent_and_has_exact_endpoints() -> None:
    from experiments.g1_ahac_first_proposal_audit.run import (
        interpolate_actor_params,
    )

    source = _actor(0.0)
    proposal = _actor(4.0)

    midpoint = interpolate_actor_params(source, proposal, alpha=0.5)
    endpoint = interpolate_actor_params(source, proposal, alpha=1.0)

    np.testing.assert_array_equal(midpoint.parent["frozen"], source.parent["frozen"])
    np.testing.assert_array_equal(midpoint.adapter["trainable"], [2.0, 4.0])
    np.testing.assert_array_equal(endpoint.adapter["trainable"], [4.0, 6.0])


def test_selection_requires_both_carried_and_phase_preservation() -> None:
    from experiments.g1_ahac_first_proposal_audit.run import classify_candidates

    source_phase = [136, 144, 84, 90, 79]
    source_carried = {
        "mean_return": 1.0,
        "survival_p10": 6.0,
        "survival_median": 35.0,
        "survival_mean": 38.0,
    }
    rows = [
        {
            "alpha": 1.0,
            "phase_survival": [137, 144, 84, 90, 78],
            "carried": {
                "mean_return": 1.1,
                "survival_p10": 7.0,
                "survival_median": 36.0,
                "survival_mean": 39.0,
            },
        },
        {
            "alpha": 0.5,
            "phase_survival": [136, 144, 84, 90, 79],
            "carried": {
                "mean_return": 0.9,
                "survival_p10": 7.0,
                "survival_median": 36.0,
                "survival_mean": 39.0,
            },
        },
        {
            "alpha": 0.25,
            "phase_survival": [136, 145, 84, 90, 79],
            "carried": {
                "mean_return": 1.01,
                "survival_p10": 6.0,
                "survival_median": 35.0,
                "survival_mean": 38.0,
            },
        },
        {
            "alpha": 0.125,
            "phase_survival": [135, 144, 84, 90, 79],
            "carried": {
                "mean_return": 0.8,
                "survival_p10": 5.0,
                "survival_median": 34.0,
                "survival_mean": 37.0,
            },
        },
        {
            "alpha": 0.0625,
            "phase_survival": [135, 144, 84, 90, 79],
            "carried": {
                "mean_return": 0.8,
                "survival_p10": 5.0,
                "survival_median": 34.0,
                "survival_mean": 37.0,
            },
        },
        {
            "alpha": 0.03125,
            "phase_survival": [135, 144, 84, 90, 79],
            "carried": {
                "mean_return": 0.8,
                "survival_p10": 5.0,
                "survival_median": 34.0,
                "survival_mean": 37.0,
            },
        },
    ]

    result = classify_candidates(
        rows,
        source_phase_survival=source_phase,
        source_carried=source_carried,
    )

    assert result["outcome"] == "ahac-proposal-safe-advances"
    assert result["selected_alpha"] == 0.25
    assert result["candidates"][0]["phase_safe"] is False
    assert result["candidates"][1]["carried_safe"] is False
    assert result["candidates"][2]["eligible"] is True
    assert result["policy_retained"] is False


def test_json_array_loader_accepts_checkpoint_metric_rows(tmp_path) -> None:
    from experiments.g1_ahac_first_proposal_audit.run import _read_json_array

    path = tmp_path / "checkpoint_phase_metrics.json"
    path.write_text(json.dumps([{"step": 1_880_064}]), encoding="utf-8")

    assert _read_json_array(path) == [{"step": 1_880_064}]


def test_proposal_kwargs_change_only_explicit_ahac_contract() -> None:
    from experiments.g1_ahac_first_proposal_audit.run import (
        CONTACT_THRESHOLD,
        END_STEP,
        build_proposal_kwargs,
    )

    kwargs = build_proposal_kwargs(
        "g1-4x5",
        "/tmp/reference.npz",
        0,
        "/tmp/checkpoint.pkl",
    )

    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == (END_STEP,)
    assert kwargs["ahac"] is True
    assert kwargs["ahac_semantics"] == "paper_equation_10_no_target"
    assert kwargs["ahac_contact_metric"] == "all_body_spatial"
    assert kwargs["ahac_contact_threshold"] == CONTACT_THRESHOLD
    assert kwargs["actor_bootstrap_scale"] == 1.0
    assert kwargs["allow_resume_actor_bootstrap_scale_change"] is True
    assert kwargs["use_lr_decay"] is False
    assert kwargs["allow_resume_core_optimizer_change"] is False
