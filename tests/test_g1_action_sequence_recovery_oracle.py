from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tools.run_g1_action_sequence_recovery_oracle import (
    build_parser,
    phase_tape_correction,
    recovery_oracle_outcome,
)


def test_phase_tape_correction_is_exact_zero_outside_registered_window():
    tape = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    phases = jnp.asarray([9, 10, 12, 13, 14])

    correction = phase_tape_correction(tape, phases, phase_min=10)

    np.testing.assert_array_equal(correction[0], np.zeros(3))
    np.testing.assert_array_equal(correction[1], tape[0])
    np.testing.assert_array_equal(correction[2], tape[2])
    np.testing.assert_array_equal(correction[3], tape[3])
    np.testing.assert_array_equal(correction[4], np.zeros(3))


def test_oracle_outcome_requires_every_carried_start_to_cross_horizon():
    assert recovery_oracle_outcome(
        baseline_survival=[29, 6],
        candidate_survival=[40, 32],
        horizon=32,
        execution_valid=True,
    ) == "action-sequence-recoverable"
    assert recovery_oracle_outcome(
        baseline_survival=[29, 6],
        candidate_survival=[40, 31],
        horizon=32,
        execution_valid=True,
    ) == "action-sequence-insufficient"
    assert recovery_oracle_outcome(
        baseline_survival=[29, 6],
        candidate_survival=[40, 32],
        horizon=32,
        execution_valid=False,
    ) == "invalid-execution"


def test_parser_accepts_independent_state_tape_treatment():
    required = [
        "--checkpoint",
        "/tmp/e023.pkl",
        "--hparams",
        "/tmp/hparams.json",
        "--reference-path",
        "/tmp/reference.npz",
        "--source-bank",
        "/tmp/bank.npz",
        "--output-directory",
        "/tmp/output",
        "--code-commit",
        "a" * 40,
    ]
    assert build_parser().parse_args(
        [*required, "--independent-tapes", "--worst-margin-objective"]
    ).independent_tapes is True
    assert build_parser().parse_args(
        [*required, "--independent-tapes", "--worst-margin-objective"]
    ).worst_margin_objective is True
