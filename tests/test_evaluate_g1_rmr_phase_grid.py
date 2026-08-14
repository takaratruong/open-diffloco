import pytest
import jax.numpy as jnp
from pathlib import Path

from src.core.rmr_policy import RmrPolicy


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


def _policy(input_dim: int) -> RmrPolicy:
    return RmrPolicy(
        mean=jnp.zeros(input_dim),
        std=jnp.ones(input_dim),
        weights=(jnp.zeros((29, input_dim)),),
        biases=(jnp.zeros(29),),
    )


def test_candidate_observation_uses_checkpoint_input_width():
    from tools.evaluate_g1_rmr_phase_grid import (
        select_rmr_policy_observation,
    )

    observation = jnp.arange(174, dtype=jnp.float32)
    assert select_rmr_policy_observation(_policy(154), observation).shape == (
        154,
    )
    assert select_rmr_policy_observation(_policy(174), observation).shape == (
        174,
    )
    with pytest.raises(ValueError, match="exceeds evaluator observation"):
        select_rmr_policy_observation(_policy(175), observation)


def test_phase_grid_routes_rollouts_through_one_compiled_step():
    source = Path("tools/evaluate_g1_rmr_phase_grid.py").read_text()

    assert "compiled_step = build_compiled_step(env)" in source
    assert source.count("step_fn=compiled_step") == 2


def test_rmr_policy_interpolation_preserves_normalizer_and_blends_network():
    from tools.evaluate_g1_rmr_phase_grid import interpolate_rmr_policy

    source = _policy(3)
    candidate = source._replace(
        weights=(jnp.full((29, 3), 4.0),),
        biases=(jnp.full(29, 2.0),),
    )

    midpoint = interpolate_rmr_policy(source, candidate, alpha=0.25)

    assert jnp.array_equal(midpoint.mean, source.mean)
    assert jnp.array_equal(midpoint.std, source.std)
    assert jnp.all(midpoint.weights[0] == 1.0)
    assert jnp.all(midpoint.biases[0] == 0.5)


def test_rmr_policy_interpolation_fails_closed_on_contract_drift():
    from tools.evaluate_g1_rmr_phase_grid import interpolate_rmr_policy

    source = _policy(3)
    with pytest.raises(ValueError, match="normalization"):
        interpolate_rmr_policy(
            source,
            source._replace(mean=jnp.ones(3)),
            alpha=0.5,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        interpolate_rmr_policy(source, source, alpha=0.0)


def test_rmr_policy_interpolation_alpha_one_is_bit_exact_candidate():
    from tools.evaluate_g1_rmr_phase_grid import interpolate_rmr_policy

    source = _policy(3)._replace(
        weights=(jnp.full((29, 3), 1e10, dtype=jnp.float32),)
    )
    candidate = _policy(3)._replace(
        weights=(jnp.ones((29, 3), dtype=jnp.float32),)
    )

    selected = interpolate_rmr_policy(source, candidate, alpha=1.0)

    assert jnp.array_equal(selected.weights[0], candidate.weights[0])
