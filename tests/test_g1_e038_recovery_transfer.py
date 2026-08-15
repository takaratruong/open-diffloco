from __future__ import annotations

import numpy as np
import pytest


SOURCE_PHASES = (0, 100, 200, 300, 400)
ROWS_PER_SOURCE = 24
HORIZON = 32


def _source_start_phase() -> np.ndarray:
    return np.repeat(np.asarray(SOURCE_PHASES, dtype=np.int32), ROWS_PER_SOURCE)


def test_validate_bank_layout_requires_exact_five_ordered_24_row_bands():
    from tools.evaluate_g1_e038_recovery_transfer import validate_bank_layout

    arrays = {
        "source_start_phase": _source_start_phase(),
        "qpos": np.zeros((120, 36), dtype=np.float64),
    }

    normalized = validate_bank_layout(arrays)

    assert set(normalized) == set(arrays)
    np.testing.assert_array_equal(
        normalized["source_start_phase"], arrays["source_start_phase"]
    )


@pytest.mark.parametrize(
    "source_start_phase",
    (
        np.repeat(np.asarray(SOURCE_PHASES, dtype=np.int32), (24, 24, 24, 24, 23)),
        np.concatenate(
            (
                np.full(24, 0),
                np.full(24, 100),
                np.full(24, 200),
                np.full(23, 300),
                np.full(25, 400),
            )
        ),
        np.repeat(np.asarray((100, 0, 200, 300, 400), dtype=np.int32), 24),
    ),
)
def test_validate_bank_layout_rejects_malformed_source_groups(source_start_phase):
    from tools.evaluate_g1_e038_recovery_transfer import validate_bank_layout

    with pytest.raises(ValueError, match="source.*groups"):
        validate_bank_layout({"source_start_phase": source_start_phase})


def test_validate_bank_layout_rejects_non_row_aligned_state_arrays():
    from tools.evaluate_g1_e038_recovery_transfer import validate_bank_layout

    with pytest.raises(ValueError, match="qpos.*120 rows"):
        validate_bank_layout(
            {
                "source_start_phase": _source_start_phase(),
                "qpos": np.zeros((119, 36), dtype=np.float64),
            }
        )


def test_survival_from_terminals_uses_first_terminal_or_full_horizon():
    from tools.evaluate_g1_e038_recovery_transfer import survival_from_terminals

    terminals = np.zeros((3, HORIZON), dtype=bool)
    terminals[0, 0] = True
    terminals[1, 17] = True

    assert survival_from_terminals(terminals) == [0, 17, HORIZON]


def _survival(*, phase_zero_successes: int, untouched_successes: int) -> list[int]:
    return (
        [HORIZON] * phase_zero_successes
        + [HORIZON - 1] * (ROWS_PER_SOURCE - phase_zero_successes)
        + [HORIZON] * untouched_successes
        + [HORIZON - 1] * (96 - untouched_successes)
    )


def test_classify_transfer_covers_every_registered_outcome():
    from tools.evaluate_g1_e038_recovery_transfer import classify_transfer

    phases = _source_start_phase()
    parent = [HORIZON - 1] * 120

    assert (
        classify_transfer(parent, parent, phases, execution_valid=False)
        == "invalid-execution"
    )
    assert (
        classify_transfer(parent, _survival(phase_zero_successes=12, untouched_successes=10), phases, True)
        == "recovery-expert-generalizes"
    )
    assert (
        classify_transfer(parent, _survival(phase_zero_successes=12, untouched_successes=9), phases, True)
        == "recovery-expert-local-only"
    )
    assert (
        classify_transfer(parent, _survival(phase_zero_successes=11, untouched_successes=0), phases, True)
        == "recovery-expert-mixed-transfer"
    )
    assert (
        classify_transfer(parent, _survival(phase_zero_successes=9, untouched_successes=0), phases, True)
        == "recovery-expert-destructive"
    )


def test_classify_transfer_marks_untouched_median_regression_destructive():
    from tools.evaluate_g1_e038_recovery_transfer import classify_transfer

    parent = [HORIZON - 1] * 120
    expert = [HORIZON] * 12 + [HORIZON - 1] * 12 + [0] * 96

    assert (
        classify_transfer(parent, expert, _source_start_phase(), execution_valid=True)
        == "recovery-expert-destructive"
    )


def test_classify_transfer_marks_improvements_and_regressions_mixed():
    from tools.evaluate_g1_e038_recovery_transfer import classify_transfer

    parent = [HORIZON - 1] * 120
    expert = (
        [HORIZON] * 12
        + [HORIZON - 1] * 12
        + [HORIZON - 2]
        + [HORIZON - 1] * 95
    )

    assert (
        classify_transfer(parent, expert, _source_start_phase(), execution_valid=True)
        == "recovery-expert-mixed-transfer"
    )


def test_classify_transfer_marks_unmatched_regression_destructive():
    from tools.evaluate_g1_e038_recovery_transfer import classify_transfer

    parent = [HORIZON] * 24 + [HORIZON - 1] * 96
    expert = (
        [HORIZON] * 12
        + [HORIZON - 1] * 12
        + [HORIZON - 2]
        + [HORIZON - 1] * 95
    )

    assert (
        classify_transfer(parent, expert, _source_start_phase(), execution_valid=True)
        == "recovery-expert-destructive"
    )


def test_zero_seed_is_enforced():
    from tools.evaluate_g1_e038_recovery_transfer import _zero_seed, build_parser

    assert _zero_seed("0") == 0
    assert build_parser().parse_args([]).seed == 0
    with pytest.raises(Exception, match="exactly zero"):
        _zero_seed("1")
