from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest

from experiments.g1_hard_contact_action_interpolation_derivatives.run import (
    ALPHAS,
    PHASE_CASES,
    PROBE_OUTPUT_NAMES,
    _all_nonselected_outputs_match_baseline,
    baseline_replay_gate,
    classify_action_interpolation,
    execute_interpolation_sweeps,
    initial_pd_diagnostics,
    interpolate_case_actions,
)

jax.config.update("jax_enable_x64", True)


def probe_output(offset: float = 0.0) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in PROBE_OUTPUT_NAMES:
        if name in {"direct_done", "direct_terminal", "direct_contact_stiffness"}:
            result[name] = np.full((10,), offset, dtype=np.float64)
        elif name in {"forward_jacobian", "reverse_jacobian"}:
            result[name] = np.full((10, 2, 29), offset, dtype=np.float64)
        else:
            result[name] = np.full((10, 2), offset, dtype=np.float64)
    return result


def test_interpolate_case_actions_changes_only_diffsim_row_and_preserves_endpoints() -> (
    None
):
    actions = np.arange(10 * 29, dtype=np.float64).reshape(10, 29) / 100.0

    ppo = np.asarray(
        interpolate_case_actions(actions, ppo_index=4, diffsim_index=5, alpha=0.0)
    )
    midpoint = np.asarray(
        interpolate_case_actions(actions, ppo_index=4, diffsim_index=5, alpha=0.25)
    )
    diffsim = np.asarray(
        interpolate_case_actions(actions, ppo_index=4, diffsim_index=5, alpha=1.0)
    )

    assert np.array_equal(ppo[5], actions[4])
    assert np.array_equal(midpoint[5], 0.75 * actions[4] + 0.25 * actions[5])
    assert np.array_equal(diffsim, actions)
    assert np.array_equal(midpoint[:5], actions[:5])
    assert np.array_equal(midpoint[6:], actions[6:])
    with pytest.raises(ValueError, match="alpha"):
        interpolate_case_actions(actions, ppo_index=4, diffsim_index=5, alpha=1.1)
    with pytest.raises(ValueError, match="paired"):
        interpolate_case_actions(actions, ppo_index=5, diffsim_index=4, alpha=0.5)


def test_baseline_gate_requires_complete_repeat_and_both_e014_invocations() -> None:
    expected_first = probe_output()
    expected_second = probe_output()
    actual_first = probe_output()
    actual_second = probe_output()

    gate = baseline_replay_gate(
        actual_first, actual_second, expected_first, expected_second
    )

    assert gate == {
        "output_names_exact": True,
        "repeat_exact": True,
        "first_matches_e014": True,
        "second_matches_e014": True,
        "valid": True,
    }
    actual_second["forward_primal"] = actual_second["forward_primal"].copy()
    actual_second["forward_primal"][7, 0] = 1.0
    assert not baseline_replay_gate(
        actual_first, actual_second, expected_first, expected_second
    )["valid"]


def test_nonselected_output_gate_preserves_alpha_then_case_axis_order() -> None:
    baseline = probe_output()
    for values in baseline.values():
        values[...] = np.arange(values.size, dtype=np.float64).reshape(values.shape)
    sweep = {
        f"first_{name}": np.broadcast_to(
            values, (len(PHASE_CASES), ALPHAS.size, *values.shape)
        ).copy()
        for name, values in baseline.items()
    }

    assert _all_nonselected_outputs_match_baseline(sweep, baseline)

    sweep["first_source_primal"][0, 3, 6, 0] += 1.0
    assert not _all_nonselected_outputs_match_baseline(sweep, baseline)


def test_interpolation_sweeps_reuse_one_callable_and_repeat_complete_outputs() -> None:
    actions = np.arange(10 * 29, dtype=np.float64).reshape(10, 29) / 100.0
    calls: list[np.ndarray] = []

    def compiled_probe(_states: object, candidate_actions: object):
        candidate = np.asarray(candidate_actions, dtype=np.float64)
        calls.append(candidate.copy())
        output = probe_output()
        output["source_primal"] = np.stack(
            (candidate[:, 0], 2.0 * candidate[:, 0]), axis=1
        )
        return output

    result = execute_interpolation_sweeps(
        compiled_probe,
        states=object(),
        actions=actions,
        alphas=np.asarray([0.0, 0.5, 1.0]),
        phase_cases=((4, 5), (6, 7)),
    )

    assert len(calls) == 2 * 2 * 3
    assert result["repeat_exact"]
    assert result["first_source_primal"].shape == (2, 3, 10, 2)
    assert np.array_equal(result["first_source_primal"], result["second_source_primal"])
    for repeat in range(2):
        for phase_slot, (ppo_index, diffsim_index) in enumerate(((4, 5), (6, 7))):
            base = (repeat * 2 + phase_slot) * 3
            assert np.array_equal(calls[base][diffsim_index], actions[ppo_index])
            assert np.array_equal(calls[base + 2], actions)


@pytest.mark.parametrize(
    ("primal_valid", "gradient_agreement", "expected"),
    [
        (
            np.ones((2, 5), dtype=bool),
            np.asarray([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=bool),
            "single-ad-transition-bracketed-in-both-phases",
        ),
        (
            np.ones((2, 5), dtype=bool),
            np.asarray([[1, 0, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=bool),
            "multiple-ad-regimes-along-action-segment",
        ),
        (
            np.asarray([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]], dtype=bool),
            np.asarray([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=bool),
            "transform-primal-boundary-along-action-segment",
        ),
    ],
)
def test_classification_distinguishes_single_multiple_and_primal_boundaries(
    primal_valid: np.ndarray,
    gradient_agreement: np.ndarray,
    expected: str,
) -> None:
    result = classify_action_interpolation(
        measurement_valid=True,
        alphas=np.linspace(0.0, 1.0, 5),
        primal_valid=primal_valid,
        gradient_agreement=gradient_agreement,
    )

    assert result["valid"]
    assert result["scientifically_interpretable"]
    assert result["outcome"] == expected


def test_classification_fails_closed_when_endpoints_do_not_reproduce_e015() -> None:
    result = classify_action_interpolation(
        measurement_valid=True,
        alphas=ALPHAS,
        primal_valid=np.ones((2, ALPHAS.size), dtype=bool),
        gradient_agreement=np.ones((2, ALPHAS.size), dtype=bool),
    )

    assert not result["valid"]
    assert result["outcome"] == "invalid-measurement"


def test_initial_pd_diagnostics_reports_margin_without_modifying_torque() -> None:
    result = initial_pd_diagnostics(
        kp=np.asarray([10.0, 20.0]),
        kd=np.asarray([1.0, 2.0]),
        effort_limit=np.asarray([5.0, 10.0]),
        position_target=np.asarray([[0.3, -0.1], [0.2, 0.3]]),
        joint_position=np.zeros((2, 2)),
        joint_velocity=np.asarray([[0.5, 0.0], [0.0, 1.0]]),
    )

    assert np.array_equal(result["raw_torque"], np.asarray([[2.5, -2.0], [2.0, 4.0]]))
    assert np.array_equal(result["clipped_torque"], result["raw_torque"])
    assert np.array_equal(result["clipped"], np.zeros((2, 2), dtype=bool))
    assert np.all(result["effort_margin"] > 0.0)


def test_runner_reuses_exact_e014_probe_and_contains_no_training_path() -> None:
    from experiments.g1_hard_contact_action_interpolation_derivatives import run

    source = inspect.getsource(run)
    runner_source = inspect.getsource(run._run)
    assert "_build_compiled_probe" in source
    assert runner_source.count("_build_compiled_probe(env, direction)") == 1
    assert "_build_compiled_interpolation" not in source
    assert "load_source_actor_policy" not in source
    assert "train(" not in source
    assert '"policy_evaluation_computed": False' in source
    assert '"optimizer_update_retained": False' in source
    assert PHASE_CASES == ((4, 5), (6, 7))
