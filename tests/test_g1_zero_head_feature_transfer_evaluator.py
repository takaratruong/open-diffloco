from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def _record(update: int, carried: list[int], ordinary: list[int]):
    return {
        "update": update,
        "carried_survival": carried,
        "ordinary_survival": ordinary,
    }


def test_selector_requires_componentwise_carried_preservation():
    from tools.evaluate_g1_zero_head_feature_transfer import select_checkpoint

    parent = [10] * 120
    records = [
        _record(8, [11] * 119 + [9], [117, 63, 49, 39, 47]),
        _record(16, [11] * 120, [116, 63, 49, 39, 47]),
        _record(32, [12] * 120, [116, 63, 49, 39, 47]),
    ]

    selected = select_checkpoint(records, parent_survival=parent)

    assert selected["outcome"] == "zero-head-features-advance"
    assert selected["eligible_updates"] == [16, 32]
    assert selected["selected_update"] == 32
    assert selected["selected_carried_survival"] == [12] * 120


def test_selector_reports_solve_insufficient_and_rejects_malformed():
    from tools.evaluate_g1_zero_head_feature_transfer import select_checkpoint

    parent = [10] * 120
    solved = select_checkpoint(
        [_record(8, [32] * 120, [499, 399, 299, 199, 99])],
        parent_survival=parent,
    )
    assert solved["outcome"] == "zero-head-features-solve"

    insufficient = select_checkpoint(
        [_record(8, [10] * 119 + [9], [116, 63, 49, 39, 47])],
        parent_survival=parent,
    )
    assert insufficient["outcome"] == "zero-head-features-insufficient"
    assert insufficient["selected_update"] is None

    with pytest.raises(ValueError, match="selection record"):
        select_checkpoint(
            [_record(8, [10] * 119, [116, 63, 49, 39, 47])],
            parent_survival=parent,
        )


def test_parser_requires_candidate_hash_step_and_all_pinned_inputs():
    from tools.evaluate_g1_zero_head_feature_transfer import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--parent-checkpoint", "/tmp/parent.pkl",
            "--parent-hparams", "/tmp/parent.json",
            "--candidate-checkpoint", "/tmp/candidate.pkl",
            "--candidate-sha256", "a" * 64,
            "--candidate-step", "1671168",
            "--reference-path", "/tmp/ref.npz",
            "--source-bank", "/tmp/bank.npz",
            "--output-directory", "/tmp/out",
            "--code-commit", "b" * 40,
            "--solver-profile", "g1-4x5",
        ]
    )
    assert args.seed == 0
    assert args.candidate_step == 1_671_168
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--parent-checkpoint", "/tmp/parent.pkl",
                "--parent-hparams", "/tmp/parent.json",
                "--candidate-checkpoint", "/tmp/candidate.pkl",
                "--candidate-sha256", "a" * 64,
                "--candidate-step", "1671168",
                "--reference-path", "/tmp/ref.npz",
                "--source-bank", "/tmp/bank.npz",
                "--output-directory", "/tmp/out",
                "--code-commit", "b" * 40,
                "--solver-profile", "g1-4x5",
                "--seed", "1",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--parent-checkpoint", "/tmp/parent.pkl",
                "--parent-hparams", "/tmp/parent.json",
                "--candidate-checkpoint", "/tmp/candidate.pkl",
                "--candidate-sha256", "a" * 64,
                "--candidate-step", "1867776",
                "--reference-path", "/tmp/ref.npz",
                "--source-bank", "/tmp/bank.npz",
                "--output-directory", "/tmp/out",
                "--code-commit", "b" * 40,
                "--solver-profile", "g1-4x5",
            ]
        )


def _candidate_state(*, step: int = 1_671_168):
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
    )

    parent = {"parent": jnp.asarray([1.0], dtype=jnp.float32)}
    normalizer = {"mean": jnp.zeros((328,), dtype=jnp.float32)}
    adapter = {
        "params": {
            "Dense_0": {
                "kernel": jnp.zeros((328, 256), dtype=jnp.float32),
                "bias": jnp.zeros((256,), dtype=jnp.float32),
            },
            "Dense_1": {
                "kernel": jnp.zeros((256, 29), dtype=jnp.float32),
                "bias": jnp.zeros((29,), dtype=jnp.float32),
            },
        }
    }
    return (
        SimpleNamespace(
            step=step,
            actor_params=FrozenPreviewResidualParams(parent, adapter),
            normalizer=normalizer,
        ),
        parent,
        normalizer,
    )


def test_candidate_checkpoint_requires_registered_selection_step_and_frozen_state():
    from tools.evaluate_g1_zero_head_feature_transfer import (
        validate_candidate_checkpoint,
    )

    candidate, parent, normalizer = _candidate_state()
    report = validate_candidate_checkpoint(
        candidate,
        candidate_step=1_671_168,
        parent_params=parent,
        parent_normalizer=normalizer,
    )
    assert report["valid"] is True
    assert report["candidate_update"] == 8

    candidate.step = 1_867_776
    with pytest.raises(ValueError, match="checkpoint grid"):
        validate_candidate_checkpoint(
            candidate,
            candidate_step=1_867_776,
            parent_params=parent,
            parent_normalizer=normalizer,
        )

    candidate, parent, normalizer = _candidate_state()
    candidate.actor_params.adapter["params"]["Dense_1"]["bias"] = jnp.full(
        (29,), jnp.nan
    )
    with pytest.raises(ValueError, match="frozen-state contract"):
        validate_candidate_checkpoint(
            candidate,
            candidate_step=1_671_168,
            parent_params=parent,
            parent_normalizer=normalizer,
        )


def _paired_evidence() -> dict[str, np.ndarray]:
    source_phases = np.repeat(
        np.asarray((0, 100, 200, 300, 400), dtype=np.int32), 24
    )
    arrays: dict[str, np.ndarray] = {
        "source_start_phase": source_phases,
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
                f"{arm}_qpos": np.zeros((120, 32, 36), dtype=np.float64),
                f"{arm}_phase": np.zeros((120, 32), dtype=np.int32),
                f"{arm}_parent_action": np.zeros((120, 32, 29)),
                f"{arm}_correction": np.zeros((120, 32, 29)),
                f"{arm}_raw_action": np.zeros((120, 32, 29)),
                f"{arm}_effective_action": np.zeros((120, 32, 29)),
                f"{arm}_alive": np.ones((120, 32), dtype=bool),
                f"{arm}_terminal": np.zeros((120, 32), dtype=bool),
                f"{arm}_reward": np.zeros((120, 32)),
                f"{arm}_normalized_termination_errors": np.zeros((120, 32, 4)),
            }
        )
    return arrays


def _provenance() -> dict[str, object]:
    return {
        "candidate_step": 1_671_168,
        "code_provenance": {
            "repository": "/tmp/repository",
            "code_commit": "a" * 40,
            "dirty_patch_sha256": "0" * 64,
        },
        "input_sha256": {
            "parent_checkpoint": "1" * 64,
            "parent_hparams": "2" * 64,
            "candidate_checkpoint": "3" * 64,
            "reference": "4" * 64,
            "source_bank": "5" * 64,
            "model": "6" * 64,
            "controller": "7" * 64,
        },
        "candidate_validation": {
            "valid": True,
            "candidate_step": 1_671_168,
            "candidate_update": 8,
            "parent_parameters_exact": True,
            "normalizer_exact": True,
            "adapter_shape_exact": True,
            "parameters_finite": True,
        },
    }


def test_publication_requires_exact_provenance_and_is_hash_bound(tmp_path: Path):
    from tools.evaluate_g1_zero_head_feature_transfer import publish_evaluation
    from tools.prepare_g1_rmr_reference import sha256_file

    manifest = publish_evaluation(
        output_directory=tmp_path,
        arrays=_paired_evidence(),
        provenance=_provenance(),
    )

    assert manifest["candidate_update"] == 8
    assert manifest["paired_rollouts_sha256"] == sha256_file(
        tmp_path / "paired_rollouts.npz"
    )
    assert json.loads((tmp_path / "summary.json").read_text()) == manifest

    malformed = _provenance()
    malformed["input_sha256"] = {"candidate_checkpoint": "3" * 64}
    with pytest.raises(ValueError, match="input hash manifest"):
        publish_evaluation(
            output_directory=tmp_path / "bad",
            arrays=_paired_evidence(),
            provenance=malformed,
        )
