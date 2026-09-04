from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest

from experiments.g1_contact_free_phase100_exact_replay_smoothness.run import (
    CASE_INDEX,
    PROBE_OUTPUT_NAMES,
    _case_source_primal_or_nan,
    _extract_e011_probe_output,
    baseline_replay_gate,
    execute_perturbation_sweeps,
    perturb_case_actions,
)

jax.config.update("jax_enable_x64", True)


def _probe_output(offset: float = 0.0) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in PROBE_OUTPUT_NAMES:
        if name in {"direct_done", "direct_terminal", "direct_contact_stiffness"}:
            result[name] = np.full((10,), offset, dtype=np.float64)
        elif name in {"forward_jacobian", "reverse_jacobian"}:
            result[name] = np.full((10, 2, 29), offset, dtype=np.float64)
        else:
            result[name] = np.full((10, 2), offset, dtype=np.float64)
    return result


def test_extract_e011_probe_output_requires_the_complete_persisted_mapping() -> None:
    expected = _probe_output()
    raw = {f"treatment_first_{name}": value for name, value in expected.items()}

    actual = _extract_e011_probe_output(raw, "first")

    assert set(actual) == set(PROBE_OUTPUT_NAMES)
    assert all(np.array_equal(actual[name], expected[name]) for name in expected)
    raw.pop("treatment_first_source_primal")
    with pytest.raises(ValueError, match="complete E011 first probe output"):
        _extract_e011_probe_output(raw, "first")


def test_baseline_replay_gate_requires_repeat_and_both_e011_invocations() -> None:
    expected_first = _probe_output()
    expected_second = _probe_output()
    actual_first = _probe_output()
    actual_second = _probe_output()

    gate = baseline_replay_gate(
        actual_first,
        actual_second,
        expected_first,
        expected_second,
    )

    assert gate == {
        "output_names_exact": True,
        "repeat_exact": True,
        "first_matches_e011": True,
        "second_matches_e011": True,
        "valid": True,
    }
    actual_second["source_primal"] = actual_second["source_primal"].copy()
    actual_second["source_primal"][CASE_INDEX, 0] = 1.0
    gate = baseline_replay_gate(
        actual_first,
        actual_second,
        expected_first,
        expected_second,
    )
    assert not gate["repeat_exact"]
    assert not gate["second_matches_e011"]
    assert not gate["valid"]


def test_invalid_baseline_can_still_publish_a_strict_json_safe_result() -> None:
    assert np.all(np.isnan(_case_source_primal_or_nan({})))
    assert np.all(
        np.isnan(
            _case_source_primal_or_nan(
                {"source_primal": np.zeros((1, 2), dtype=np.float64)}
            )
        )
    )
    result = _case_source_primal_or_nan(_probe_output())
    assert result.shape == (2,)
    assert np.array_equal(result, np.zeros((2,), dtype=np.float64))


def test_perturb_case_actions_changes_only_the_selected_full_batch_row() -> None:
    actions = np.arange(10 * 29, dtype=np.float64).reshape(10, 29) / 1000.0
    direction = np.linspace(-1.0, 1.0, 29, dtype=np.float64)

    perturbed = np.asarray(
        perturb_case_actions(actions, direction, epsilon=3e-4, sign=-1)
    )

    assert np.array_equal(perturbed[:CASE_INDEX], actions[:CASE_INDEX])
    assert np.array_equal(perturbed[CASE_INDEX + 1 :], actions[CASE_INDEX + 1 :])
    assert np.allclose(perturbed[CASE_INDEX], actions[CASE_INDEX] - 3e-4 * direction)
    with pytest.raises(ValueError, match="sign"):
        perturb_case_actions(actions, direction, epsilon=3e-4, sign=0)


def test_perturbation_sweeps_reuse_one_callable_and_repeat_exactly() -> None:
    actions = np.zeros((10, 29), dtype=np.float64)
    direction = np.ones((29,), dtype=np.float64) / np.sqrt(29.0)
    epsilons = np.asarray([1e-3, 1e-4], dtype=np.float64)
    calls: list[np.ndarray] = []

    def compiled_probe(_states: object, candidate_actions: object):
        candidate = np.asarray(candidate_actions, dtype=np.float64)
        calls.append(candidate.copy())
        source_primal = np.stack((candidate[:, 0], 2.0 * candidate[:, 0]), axis=1)
        return {
            "source_primal": source_primal,
            "direct_done": np.zeros((10,), dtype=np.float64),
            "direct_terminal": np.zeros((10,), dtype=np.float64),
        }

    result = execute_perturbation_sweeps(
        compiled_probe,
        states=object(),
        actions=actions,
        direction=direction,
        epsilons=epsilons,
    )

    assert len(calls) == 2 * 2 * len(epsilons)
    assert result["repeat_exact"]
    assert np.array_equal(result["first_plus_primal"], result["second_plus_primal"])
    assert np.array_equal(result["first_minus_primal"], result["second_minus_primal"])
    for candidate in calls:
        assert np.array_equal(candidate[:CASE_INDEX], actions[:CASE_INDEX])
        assert np.array_equal(candidate[CASE_INDEX + 1 :], actions[CASE_INDEX + 1 :])


def test_runner_imports_the_exact_e011_probe_and_contains_no_training_path() -> None:
    from experiments.g1_contact_free_phase100_exact_replay_smoothness import run

    source = inspect.getsource(run)
    runner_source = inspect.getsource(run._run)
    assert "_build_compiled_probe" in source
    assert runner_source.count("_build_compiled_probe(env, direction)") == 1
    assert "_build_compiled_sweep" not in source
    assert "load_source_actor_policy" not in source
    assert "train(" not in source
    assert '"policy_evaluation_computed": False' in source
    assert '"optimizer_update_retained": False' in source
