from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

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
    assert build_parser().parse_args(
        [
            "--checkpoint",
            "/tmp/parent.pkl",
            "--hparams",
            "/tmp/hparams.json",
            "--reference-path",
            "/tmp/reference.npz",
            "--source-bank",
            "/tmp/bank.npz",
            "--expert-checkpoint",
            "/tmp/expert.pkl",
            "--output-directory",
            "/tmp/output",
            "--code-commit",
            "a" * 40,
            "--solver-profile",
            "g1-4x5",
        ]
    ).seed == 0
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(Exception, match="exactly zero"):
        _zero_seed("1")


def _paired_evidence() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "source_start_phase": _source_start_phase(),
        "initial_qpos": np.zeros((120, 36), dtype=np.float64),
        "initial_qvel": np.zeros((120, 35), dtype=np.float64),
        "initial_phase": np.zeros(120, dtype=np.int32),
        "initial_last_act": np.zeros((120, 29), dtype=np.float64),
        "initial_actor_obs_history": np.zeros((120, 10, 328), dtype=np.float64),
        "initial_rng_key": np.zeros((120, 2), dtype=np.uint32),
    }
    for arm in ("parent", "expert"):
        arrays.update(
            {
                f"{arm}_qpos": np.zeros((120, HORIZON, 36), dtype=np.float64),
                f"{arm}_phase": np.zeros((120, HORIZON), dtype=np.int32),
                f"{arm}_parent_action": np.zeros(
                    (120, HORIZON, 29), dtype=np.float64
                ),
                f"{arm}_correction": np.zeros(
                    (120, HORIZON, 29), dtype=np.float64
                ),
                f"{arm}_raw_action": np.zeros(
                    (120, HORIZON, 29), dtype=np.float64
                ),
                f"{arm}_effective_action": np.zeros(
                    (120, HORIZON, 29), dtype=np.float64
                ),
                f"{arm}_alive": np.ones((120, HORIZON), dtype=bool),
                f"{arm}_terminal": np.zeros((120, HORIZON), dtype=bool),
                f"{arm}_reward": np.zeros((120, HORIZON), dtype=np.float64),
                f"{arm}_normalized_termination_errors": np.zeros(
                    (120, HORIZON, 4), dtype=np.float64
                ),
            }
        )
    return arrays


def test_paired_evidence_requires_exact_tensors_and_action_boundary():
    from tools.evaluate_g1_e038_recovery_transfer import validate_paired_evidence

    arrays = _paired_evidence()
    arrays["expert_correction"][0, 0, 0] = 1.5
    arrays["expert_raw_action"][0, 0, 0] = 1.5
    arrays["expert_effective_action"][0, 0, 0] = 1.0

    validation = validate_paired_evidence(arrays)

    assert validation["parent_survival"] == [HORIZON] * 120
    assert validation["expert_survival"] == [HORIZON] * 120
    assert validation["source_rows"] == [24] * 5
    assert validation["phase_zero_expert_successes"] == 24
    assert validation["untouched_newly_recovered"] == 0
    assert validation["any_survival_regression"] is False

    arrays["expert_effective_action"][0, 0, 0] = 0.9999999999999999
    with pytest.raises(ValueError, match="effective action"):
        validate_paired_evidence(arrays)


def test_paired_evidence_rejects_unpaired_state_or_nonfinite_tensor():
    from tools.evaluate_g1_e038_recovery_transfer import validate_paired_evidence

    arrays = _paired_evidence()
    arrays["expert_qpos"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="initial qpos"):
        validate_paired_evidence(arrays)

    arrays = _paired_evidence()
    arrays["parent_reward"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_paired_evidence(arrays)


def test_paired_evidence_requires_exact_initial_bank_state_and_evidence_dtypes():
    from tools.evaluate_g1_e038_recovery_transfer import validate_paired_evidence

    arrays = _paired_evidence()
    arrays["initial_qpos"][0, 0] = 1.0
    with pytest.raises(ValueError, match="initial qpos.*bank"):
        validate_paired_evidence(arrays)

    arrays = _paired_evidence()
    arrays["parent_alive"] = arrays["parent_alive"].astype(np.uint8)
    with pytest.raises(ValueError, match="alive.*boolean"):
        validate_paired_evidence(arrays)

    arrays = _paired_evidence()
    arrays["expert_phase"] = arrays["expert_phase"].astype(np.float64)
    with pytest.raises(ValueError, match="phase.*integer"):
        validate_paired_evidence(arrays)


def test_publish_evaluation_writes_hash_bound_manifest_after_evidence(tmp_path):
    from tools.evaluate_g1_e038_recovery_transfer import publish_evaluation
    from tools.prepare_g1_rmr_reference import sha256_file

    manifest = publish_evaluation(
        output_directory=tmp_path,
        arrays=_paired_evidence(),
        provenance={
            "code_commit": "abc123",
            "input_sha256": {
                "parent_checkpoint": "a" * 64,
                "hparams": "b" * 64,
                "reference": "c" * 64,
                "source_bank": "d" * 64,
                "expert_checkpoint": "e" * 64,
                "model": "2" * 64,
                "controller": "3" * 64,
            },
            "parameter_sha256_before": {
                "parent": "f" * 64,
                "normalizer": "0" * 64,
                "expert": "1" * 64,
            },
            "parameter_sha256_after": {
                "parent": "f" * 64,
                "normalizer": "0" * 64,
                "expert": "1" * 64,
            },
        },
    )

    evidence_path = tmp_path / "paired_rollouts.npz"
    summary_path = tmp_path / "summary.json"
    assert evidence_path.is_file()
    assert summary_path.is_file()
    assert summary_path.stat().st_mtime_ns >= evidence_path.stat().st_mtime_ns
    assert manifest["paired_rollouts_sha256"] == sha256_file(evidence_path)
    assert manifest["parent_survival"] == [HORIZON] * 120
    assert manifest["expert_survival"] == [HORIZON] * 120
    assert json.loads(summary_path.read_text(encoding="utf-8")) == manifest


def test_parameter_immutability_is_bit_exact():
    from tools.evaluate_g1_e038_recovery_transfer import (
        parameter_tree_sha256,
        validate_parameter_immutability,
    )

    before = {"layer": {"weight": np.asarray([1.0, 2.0])}}
    unchanged = {"layer": {"weight": np.asarray([1.0, 2.0])}}
    changed = {"layer": {"weight": np.asarray([1.0, 2.000000000000001])}}

    digest = parameter_tree_sha256(before)
    hashes = {"parent": digest, "normalizer": digest, "expert": digest}
    assert validate_parameter_immutability(hashes, hashes)
    with pytest.raises(ValueError, match="parameters changed"):
        validate_parameter_immutability(
            hashes,
            {
                "parent": digest,
                "normalizer": digest,
                "expert": parameter_tree_sha256(changed),
            },
        )
    assert digest == parameter_tree_sha256(unchanged)


def test_parameter_hash_distinguishes_jax_pytree_structure_and_named_state():
    from collections import namedtuple

    from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256

    NormState = namedtuple("NormState", ("mean", "var"))
    leaves = (np.asarray([1.0]), np.asarray([2.0]))

    assert parameter_tree_sha256(NormState(*leaves)) != parameter_tree_sha256(leaves)


def test_exact_e023_hashes_and_parent_checkpoint_identity_are_enforced(monkeypatch):
    from tools import evaluate_g1_e038_recovery_transfer as evaluator

    paths = {
        "parent_checkpoint": Path("/tmp/parent.pkl"),
        "hparams": Path("/tmp/hparams.json"),
        "reference": Path("/tmp/reference.npz"),
        "source_bank": Path("/tmp/bank.npz"),
        "expert_checkpoint": Path("/tmp/expert.pkl"),
    }
    expected = {
        "parent_checkpoint": evaluator.EXPECTED_PARENT_CHECKPOINT_SHA256,
        "hparams": evaluator.EXPECTED_PARENT_HPARAMS_SHA256,
        "reference": evaluator.EXPECTED_REFERENCE_SHA256,
        "source_bank": evaluator.EXPECTED_SOURCE_BANK_SHA256,
        "expert_checkpoint": evaluator.EXPECTED_EXPERT_CHECKPOINT_SHA256,
    }
    monkeypatch.setattr(
        evaluator,
        "sha256_file",
        lambda path: expected[next(name for name, value in paths.items() if value == path)],
    )
    assert evaluator.validate_exact_input_hashes(paths) == expected

    layers = {
        name: {"kernel": np.zeros((1, 1)), "bias": np.zeros(1)}
        for name in (
            "Dense_0",
            "Dense_1",
            "Dense_2",
            "Dense_3",
            "LayerNorm_0",
            "LayerNorm_1",
            "LayerNorm_2",
        )
    }
    evaluator.validate_parent_checkpoint(
        SimpleNamespace(
            step=1_572_864,
            actor_params={"params": layers},
        )
    )
    with pytest.raises(ValueError, match="step"):
        evaluator.validate_parent_checkpoint(
            SimpleNamespace(step=1, actor_params={"params": layers})
        )


def test_run_evaluation_exposes_all_immutable_inputs_and_output_directory():
    from tools.evaluate_g1_e038_recovery_transfer import run_evaluation

    assert set(inspect.signature(run_evaluation).parameters) == {
        "checkpoint_path",
        "hparams_path",
        "reference_path",
        "bank_path",
        "expert_checkpoint_path",
        "output_directory",
        "seed",
        "code_commit",
    }
