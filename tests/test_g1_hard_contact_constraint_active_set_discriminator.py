import inspect

import numpy as np

from experiments.g1_hard_contact_constraint_active_set_discriminator.run import (
    ALPHAS,
    PHASE_CASES,
    TELEMETRY_NAMES,
    _all_nonselected_telemetry_match_baseline,
    classify_active_set_discriminator,
    direct_replay_gate,
    execute_telemetry_sweeps,
    reconstruct_pyramidal_active_set,
    summarize_active_set_transitions,
)


SMOOTH_MASK = np.asarray(
    [
        [True, True, False, False, False, True, False, True, False],
        [True, True, True, False, False, True, True, True, False],
    ]
)


def active_from_mask(mask: np.ndarray, row: int, *, width: int = 5) -> np.ndarray:
    active = np.zeros((*mask.shape, width), dtype=bool)
    active[..., row] = mask
    return active


def telemetry_output(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        name: np.full((10, 2) if name == "source_primal" else (10, 3), offset)
        for name in TELEMETRY_NAMES
    }


def test_direct_replay_gate_requires_every_legacy_direct_output_exact() -> None:
    expected = {
        "source_primal": np.arange(40, dtype=np.float64).reshape(2, 1, 10, 2),
        "direct_done": np.zeros((2, 1, 10), dtype=np.float64),
        "direct_terminal": np.zeros((2, 1, 10), dtype=np.float64),
        "direct_contact_stiffness": np.ones((2, 1, 10), dtype=np.float64),
    }
    actual = {name: value.copy() for name, value in expected.items()}

    gate = direct_replay_gate(actual, expected)

    assert gate["valid"]
    assert all(gate["by_output"].values())
    actual["source_primal"][1, 0, 7, 1] += 1.0
    assert not direct_replay_gate(actual, expected)["valid"]


def test_telemetry_sweeps_reuse_full_batch_and_repeat_exactly() -> None:
    actions = np.arange(10 * 29, dtype=np.float64).reshape(10, 29) / 100.0
    calls: list[np.ndarray] = []

    def compiled_probe(_states: object, candidate_actions: object):
        candidate = np.asarray(candidate_actions)
        calls.append(candidate.copy())
        output = telemetry_output()
        for name in TELEMETRY_NAMES:
            output[name][:, 0] = candidate[:, 0]
        return output

    result = execute_telemetry_sweeps(
        compiled_probe,
        states=object(),
        actions=actions,
        alphas=np.asarray([0.0, 0.5, 1.0]),
        phase_cases=PHASE_CASES,
    )

    assert len(calls) == 12
    assert result["repeat_exact"]
    assert result["first_source_primal"].shape == (2, 3, 10, 2)
    assert np.array_equal(
        result["candidate_actions"], np.asarray(calls[:6]).reshape(2, 3, 10, 29)
    )


def test_nonselected_telemetry_gate_preserves_alpha_then_case_axis_order() -> None:
    baseline = telemetry_output()
    for values in baseline.values():
        values[...] = np.arange(values.size).reshape(values.shape)
    sweep = {
        f"first_{name}": np.broadcast_to(
            values, (len(PHASE_CASES), ALPHAS.size, *values.shape)
        ).copy()
        for name, values in baseline.items()
    }

    assert _all_nonselected_telemetry_match_baseline(sweep, baseline)
    sweep["first_source_primal"][0, 3, 6, 0] += 1.0
    assert not _all_nonselected_telemetry_match_baseline(sweep, baseline)


def test_host_active_reconstruction_matches_equality_friction_and_unilateral_rules() -> (
    None
):
    efc_type = np.asarray([0, 1, 1, 3, 6], dtype=np.int64)
    jaref = np.broadcast_to(
        np.asarray([1.0, 0.02, 0.06, -0.01, 0.03]), (2, ALPHAS.size, 5)
    ).copy()
    frictionloss = np.zeros_like(jaref)
    frictionloss[..., 1:3] = 0.1
    efc_d = np.full_like(jaref, 2.0)

    active = reconstruct_pyramidal_active_set(
        jaref=jaref,
        efc_type=efc_type,
        efc_frictionloss=frictionloss,
        efc_d=efc_d,
    )

    assert np.all(active == np.asarray([True, True, False, True, False]))


def test_transition_summary_separates_friction_contact_limit_and_other_rows() -> None:
    active = np.zeros((2, ALPHAS.size, 5), dtype=bool)
    active[:, 1:, 0] = True
    active[:, 2:, 1] = True
    active[:, 3:, 2] = True
    active[:, 4:, 3] = True
    active[:, 5:, 4] = True
    efc_type = np.asarray([1, 6, 3, 0, 99], dtype=np.int64)

    result = summarize_active_set_transitions(active, efc_type)

    assert result["any_change"].shape == (2, ALPHAS.size - 1)
    assert result["friction_change"][0].tolist() == [True] + [False] * 7
    assert result["contact_change"][0].tolist() == [False, True] + [False] * 6
    assert result["limit_change"][0].tolist() == [False, False, True] + [False] * 5
    assert (
        result["other_change"][0].tolist() == [False] * 3 + [True, True] + [False] * 3
    )


def test_classifier_selects_frictionloss_when_only_friction_tracks_mask() -> None:
    efc_type = np.asarray([1, 1, 6], dtype=np.int64)
    active = active_from_mask(SMOOTH_MASK, 0, width=3)

    result = classify_active_set_discriminator(
        measurement_valid=True,
        smooth_agreement=SMOOTH_MASK,
        active=active,
        efc_type=efc_type,
    )

    assert result["valid"]
    assert result["outcome"] == "frictionloss-active-set-exactly-tracks-ad-regimes"
    assert result["active_change_matches_mask_transition"]
    assert result["changed_constraint_categories"] == ["frictionloss"]


def test_classifier_selects_contact_when_only_contact_tracks_mask() -> None:
    efc_type = np.asarray([1, 6, 6], dtype=np.int64)
    active = active_from_mask(SMOOTH_MASK, 1, width=3)

    result = classify_active_set_discriminator(
        measurement_valid=True,
        smooth_agreement=SMOOTH_MASK,
        active=active,
        efc_type=efc_type,
    )

    assert result["outcome"] == "contact-active-set-exactly-tracks-ad-regimes"
    assert result["changed_constraint_categories"] == ["contact"]


def test_classifier_selects_coupled_when_exact_transitions_use_both_categories() -> (
    None
):
    efc_type = np.asarray([1, 6], dtype=np.int64)
    active = np.zeros((*SMOOTH_MASK.shape, 2), dtype=bool)
    active[..., 0] = SMOOTH_MASK
    active[..., 1] = SMOOTH_MASK

    result = classify_active_set_discriminator(
        measurement_valid=True,
        smooth_agreement=SMOOTH_MASK,
        active=active,
        efc_type=efc_type,
    )

    assert result["outcome"] == "coupled-active-set-exactly-tracks-ad-regimes"
    assert result["changed_constraint_categories"] == ["contact", "frictionloss"]


def test_classifier_reports_nontracking_when_active_changes_extra_or_miss_transition() -> (
    None
):
    efc_type = np.asarray([1, 6], dtype=np.int64)
    active = active_from_mask(SMOOTH_MASK, 0, width=2)
    active[0, 3:, 1] = True

    result = classify_active_set_discriminator(
        measurement_valid=True,
        smooth_agreement=SMOOTH_MASK,
        active=active,
        efc_type=efc_type,
    )

    assert result["valid"]
    assert result["outcome"] == "final-active-set-does-not-track-ad-regimes"
    assert not result["active_change_matches_mask_transition"]


def test_classifier_fails_closed_on_invalid_measurement_or_shape() -> None:
    active = active_from_mask(SMOOTH_MASK, 0, width=2)
    efc_type = np.asarray([1, 6], dtype=np.int64)
    for measurement_valid, smooth in (
        (False, SMOOTH_MASK),
        (True, SMOOTH_MASK[:, :-1]),
    ):
        result = classify_active_set_discriminator(
            measurement_valid=measurement_valid,
            smooth_agreement=smooth,
            active=active,
            efc_type=efc_type,
        )
        assert not result["valid"]
        assert result["outcome"] == "invalid-measurement"


def test_runner_uses_direct_context_probe_and_contains_no_derivative_or_training_path() -> (
    None
):
    from experiments.g1_hard_contact_constraint_active_set_discriminator import run

    source = inspect.getsource(run)
    runner_source = inspect.getsource(run._run)
    assert runner_source.count("_build_compiled_telemetry_probe(env)") == 1
    assert "mjx_solver.Context.create(model, data, grad=False)" in source
    assert "mjx_support.contact_force_dim" in source
    assert "compute_two_objective_derivatives" not in source
    assert "jax.grad" not in source
    assert "load_source_actor_policy" not in source
    assert "train(" not in source
    assert '"derivative_computed": False' in source
    assert '"policy_evaluation_computed": False' in source
