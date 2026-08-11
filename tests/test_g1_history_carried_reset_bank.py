import copy

import numpy as np
import pytest

from tools.build_g1_history_carried_reset_bank import (
    select_preterminal_indices,
    source_rollout_step_limit,
    validate_observed_survival,
    validate_history_bank,
)


SOURCE_PHASES = (0, 100, 200, 300, 400)
SURVIVAL = (75, 63, 94, 74, 45)
HISTORY_LEN = 10
FRAME_DIM = 328


def _valid_arrays() -> dict[str, np.ndarray]:
    source_start_phase = []
    source_step = []
    transitions_to_terminal = []
    phase = []
    for start, steps in zip(SOURCE_PHASES, SURVIVAL, strict=True):
        indices = select_preterminal_indices(steps)
        source_start_phase.extend([start] * len(indices))
        source_step.extend(indices.tolist())
        transitions_to_terminal.extend((steps - indices).tolist())
        phase.extend((start + indices).tolist())
    rows = len(phase)
    qpos = np.zeros((rows, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    history = np.zeros((rows, HISTORY_LEN, FRAME_DIM), dtype=np.float64)
    fresh_frame = np.zeros((rows, FRAME_DIM), dtype=np.float64)
    return {
        "qpos": qpos,
        "qvel": np.zeros((rows, 35), dtype=np.float64),
        "phase": np.asarray(phase, dtype=np.int32),
        "last_act": np.zeros((rows, 29), dtype=np.float64),
        "actor_obs_history": history,
        "fresh_actor_frame": fresh_frame,
        "action": np.zeros((rows, 29), dtype=np.float64),
        "source_start_phase": np.asarray(
            source_start_phase, dtype=np.int32
        ),
        "source_step": np.asarray(source_step, dtype=np.int32),
        "transitions_to_terminal": np.asarray(
            transitions_to_terminal, dtype=np.int32
        ),
        "terminal": np.zeros(rows, dtype=np.float64),
        "termination_errors": np.zeros((rows, 4), dtype=np.float64),
        "termination_thresholds": np.array(
            [0.25, 1.3, 0.8, 0.4], dtype=np.float64
        ),
    }


def _validate(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return validate_history_bank(
        arrays,
        expected_source_phases=SOURCE_PHASES,
        expected_survival=SURVIVAL,
        history_len=HISTORY_LEN,
        frame_dim=FRAME_DIM,
    )


def test_selects_exact_twenty_four_state_preterminal_band() -> None:
    indices = select_preterminal_indices(70)

    np.testing.assert_array_equal(indices, np.arange(41, 65))
    np.testing.assert_array_equal(70 - indices, np.arange(29, 5, -1))


def test_source_rollout_observes_full_suffix_instead_of_expected_count() -> None:
    assert source_rollout_step_limit(499, 0) == 499
    assert source_rollout_step_limit(499, 400) == 99
    with pytest.raises(ValueError, match="source phase"):
        source_rollout_step_limit(499, 499)


def test_observed_survival_only_requires_the_fixed_preterminal_band() -> None:
    observed = (63, 63, 95, 69, 44)

    assert validate_observed_survival(observed, source_count=5) == observed
    with pytest.raises(ValueError, match="at least 29"):
        validate_observed_survival((28, 63, 95, 69, 44), source_count=5)
    with pytest.raises(ValueError, match="5 source"):
        validate_observed_survival((63, 63), source_count=5)


def test_valid_history_bank_preserves_exact_source_contract() -> None:
    summary = _validate(_valid_arrays())

    assert summary == {
        "valid": True,
        "protocol": "g1-history-carried-reset-bank-v1",
        "rows": 120,
        "rows_per_source": [24, 24, 24, 24, 24],
        "source_phases": list(SOURCE_PHASES),
        "source_survival": list(SURVIVAL),
        "minimum_transitions_to_terminal": 6,
        "maximum_transitions_to_terminal": 29,
        "minimum_hard_limit_clearance": 1.0,
        "maximum_history_frame_error": 0.0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda arrays: arrays["source_step"].__setitem__(0, 42),
            "source step",
        ),
        (
            lambda arrays: arrays["terminal"].__setitem__(0, 1.0),
            "nonterminal",
        ),
        (
            lambda arrays: arrays["termination_errors"].__setitem__(
                (0, 0), 0.25
            ),
            "hard termination limits",
        ),
        (
            lambda arrays: arrays["qvel"].__setitem__((0, 0), np.nan),
            "finite",
        ),
        (
            lambda arrays: arrays["qpos"].__setitem__((0, 3), 0.5),
            "quaternions",
        ),
        (
            lambda arrays: arrays["fresh_actor_frame"].__setitem__(
                (0, 0), 0.01
            ),
            "history frame",
        ),
    ),
)
def test_history_bank_rejects_invalid_scientific_rows(
    mutation, message: str
) -> None:
    arrays = copy.deepcopy(_valid_arrays())
    mutation(arrays)

    with pytest.raises(ValueError, match=message):
        _validate(arrays)


def test_history_bank_rejects_misaligned_context_shape() -> None:
    arrays = _valid_arrays()
    arrays["actor_obs_history"] = arrays["actor_obs_history"][:, :9]

    with pytest.raises(ValueError, match="actor_obs_history shape"):
        _validate(arrays)
