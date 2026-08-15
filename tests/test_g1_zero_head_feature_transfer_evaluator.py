from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def _record(
    update: int,
    carried: list[int],
    ordinary: list[int],
    completed: list[bool] | None = None,
):
    return {
        "update": update,
        "carried_survival": carried,
        "ordinary_survival": ordinary,
        "ordinary_completed": completed if completed is not None else [False] * 5,
    }


def test_selector_requires_componentwise_carried_preservation():
    from tools.evaluate_g1_zero_head_feature_transfer import select_checkpoint

    parent = [10] * 120
    records = [
        _record(8, [11] * 119 + [9], [117, 63, 49, 39, 47]),
        _record(16, [11] * 120, [116, 63, 49, 39, 47]),
        _record(32, [12] * 120, [116, 63, 49, 39, 47]),
        _record(64, [10] * 120, [116, 63, 49, 39, 47]),
    ]

    selected = select_checkpoint(records, parent_survival=parent)

    assert selected["outcome"] == "zero-head-features-advance"
    assert selected["eligible_updates"] == [16, 32, 64]
    assert selected["selected_update"] == 32
    assert selected["selected_carried_survival"] == [12] * 120


def test_selector_reports_solve_insufficient_and_rejects_malformed():
    from tools.evaluate_g1_zero_head_feature_transfer import select_checkpoint

    parent = [10] * 120
    solved = select_checkpoint(
        [
            _record(
                8,
                [32] * 120,
                [499, 399, 299, 199, 99],
                [True] * 5,
            ),
            _record(16, [10] * 119 + [9], [116, 63, 49, 39, 47]),
            _record(32, [10] * 119 + [9], [116, 63, 49, 39, 47]),
            _record(64, [10] * 119 + [9], [116, 63, 49, 39, 47]),
        ],
        parent_survival=parent,
    )
    assert solved["outcome"] == "zero-head-features-solve"

    terminal_at_end = select_checkpoint(
        [
            _record(update, [32] * 120, [499, 399, 299, 199, 99])
            for update in (8, 16, 32, 64)
        ],
        parent_survival=parent,
    )
    assert terminal_at_end["outcome"] == "zero-head-features-advance"

    insufficient = select_checkpoint(
        [
            _record(update, [10] * 119 + [9], [116, 63, 49, 39, 47])
            for update in (8, 16, 32, 64)
        ],
        parent_survival=parent,
    )
    assert insufficient["outcome"] == "zero-head-features-insufficient"
    assert insufficient["selected_update"] is None

    with pytest.raises(ValueError, match="selection record"):
        select_checkpoint(
            [
                _record(8, [10] * 119, [116, 63, 49, 39, 47]),
                _record(16, [10] * 120, [116, 63, 49, 39, 47]),
                _record(32, [10] * 120, [116, 63, 49, 39, 47]),
                _record(64, [10] * 120, [116, 63, 49, 39, 47]),
            ],
            parent_survival=parent,
        )

    with pytest.raises(ValueError, match="exact updates"):
        select_checkpoint(
            [
                _record(update, [10] * 120, [116, 63, 49, 39, 47])
                for update in (8, 16, 32, 999)
            ],
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
    import optax

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        initialize_residual_adapter_optimizer,
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
    composite = FrozenPreviewResidualParams(parent, adapter)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1.0e-3)
    )
    parent_optimizer = optimizer.init(parent)
    candidate_optimizer = initialize_residual_adapter_optimizer(
        optimizer,
        parent_optimizer_state=parent_optimizer,
        composite_params=composite,
    )
    common = {
        "key": jnp.zeros((2,), dtype=jnp.uint32),
        "env_state": {"state": jnp.asarray(0.0)},
        "critic_params": {"critic": jnp.asarray(0.0)},
        "target_critic_params": {"critic": jnp.asarray(0.0)},
        "critic_opt": {"critic": jnp.asarray(0.0)},
        "critic_normalizer": {"mean": jnp.asarray(0.0)},
        "ldm_params": None,
        "ldm_opt": None,
        "replay_buffer": None,
    }
    parent_state = SimpleNamespace(
        step=1_572_864,
        actor_params=parent,
        normalizer=normalizer,
        actor_opt=parent_optimizer,
        **common,
    )
    candidate_state = SimpleNamespace(
        step=step,
        actor_params=composite,
        normalizer=normalizer,
        actor_opt=candidate_optimizer,
        **common,
    )
    return candidate_state, parent_state


def test_candidate_checkpoint_requires_registered_selection_step_and_frozen_state():
    from tools.evaluate_g1_zero_head_feature_transfer import (
        validate_candidate_checkpoint,
    )

    candidate, parent = _candidate_state()
    report = validate_candidate_checkpoint(
        candidate,
        candidate_step=1_671_168,
        parent_state=parent,
    )
    assert report["valid"] is True
    assert report["candidate_update"] == 8

    candidate.step = 1_867_776
    with pytest.raises(ValueError, match="checkpoint grid"):
        validate_candidate_checkpoint(
            candidate,
            candidate_step=1_867_776,
            parent_state=parent,
        )

    candidate, parent = _candidate_state()
    candidate.actor_params.adapter["params"]["Dense_1"]["bias"] = jnp.full(
        (29,), jnp.nan
    )
    with pytest.raises(ValueError, match="frozen-state contract"):
        validate_candidate_checkpoint(
            candidate,
            candidate_step=1_671_168,
            parent_state=parent,
        )

    candidate, parent = _candidate_state()
    del candidate.actor_opt
    with pytest.raises(ValueError, match="complete TrainState"):
        validate_candidate_checkpoint(
            candidate,
            candidate_step=1_671_168,
            parent_state=parent,
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


def _write_selection_fixture(root: Path) -> Path:
    from tools.evaluate_g1_rmr_phase_grid import build_phase_grid_summary
    from tools.evaluate_g1_zero_head_feature_transfer import (
        EVALUATION_CHECKPOINTS,
        EXPECTED_LAFAN_REFERENCE_SHA256,
        TRAINING_CHECKPOINT_STEPS,
    )
    from tools.prepare_g1_rmr_reference import sha256_file

    training_run = root / "training-run"
    training_run.mkdir()
    checkpoint_hashes: dict[str, str] = {}
    code_provenance = {
        "repository": "/tmp/repository",
        "code_commit": "c" * 40,
        "dirty_patch_sha256": "0" * 64,
    }
    for step in TRAINING_CHECKPOINT_STEPS:
        checkpoint_path = training_run / f"checkpoint_step_{step}.pkl"
        checkpoint_path.write_bytes(f"checkpoint-{step}".encode())
        checkpoint_hashes[str(step)] = sha256_file(checkpoint_path)
    for step, update in EVALUATION_CHECKPOINTS.items():
        directory = root / f"update{update:03d}"
        carried_directory = directory / "carried"
        ordinary_directory = directory / "ordinary"
        carried_directory.mkdir(parents=True)
        ordinary_directory.mkdir(parents=True)
        checkpoint_path = training_run / f"checkpoint_step_{step}.pkl"
        checkpoint_hash = sha256_file(checkpoint_path)
        evidence_path = carried_directory / "paired_rollouts.npz"
        arrays = _paired_evidence()
        for arm in ("parent", "expert"):
            terminal_step = 11 if arm == "expert" and update == 32 else 10
            arrays[f"{arm}_terminal"][:, terminal_step] = True
            arrays[f"{arm}_alive"][:, terminal_step + 1 :] = False
        np.savez_compressed(evidence_path, **arrays)
        parent = [10] * 120
        candidate = [11] * 120 if update == 32 else parent
        carried = {
            "valid": True,
            "protocol": "g1-zero-head-feature-transfer-carried-evaluation-v1",
            "candidate_step": step,
            "candidate_update": update,
            "parent_survival": parent,
            "candidate_survival": candidate,
            "carried_no_regression": True,
            "carried_improvement_count": 120 if update == 32 else 0,
            "carried_regression_count": 0,
            "seed": 0,
            "solver_profile": "g1-4x5",
            "code_provenance": code_provenance,
            "input_sha256": {
                "parent_checkpoint": "1" * 64,
                "parent_hparams": "2" * 64,
                "candidate_checkpoint": checkpoint_hash,
                "reference": EXPECTED_LAFAN_REFERENCE_SHA256,
                "source_bank": "5" * 64,
                "model": "6" * 64,
                "controller": "7" * 64,
            },
            "paired_rollouts_path": str(evidence_path),
            "paired_rollouts_sha256": sha256_file(evidence_path),
        }
        (carried_directory / "summary.json").write_text(json.dumps(carried))
        survival = [116, 63, 49, 39, 47]
        ordinary_results = [
            {"phase": phase, "steps": steps, "terminal": True}
            for phase, steps in zip(
                (0, 100, 200, 300, 400), survival, strict=True
            )
        ]
        ordinary = {
            "protocol": "g1-flax-dance-replay-free-five-phase-v1",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
            "reference_transitions": 499,
            "solver_profile": "g1-4x5",
            "seed": 0,
            "post_policy_action_clip": False,
            "code_provenance": code_provenance,
            "actor_residual_preview_adapter": True,
            "actor_residual_preview_hidden": 256,
            "results": ordinary_results,
            "summary": build_phase_grid_summary(
                ordinary_results,
                phases=(0, 100, 200, 300, 400),
                reference_transitions=499,
            ),
        }
        (ordinary_directory / "phase_grid_summary.json").write_text(
            json.dumps(ordinary)
        )
    training = root / "training_validation.json"
    training.write_text(
        json.dumps(
            {
                "valid": True,
                "protocol": "g1-zero-head-feature-transfer-training-v1",
                "run_directory": str(training_run),
                "checkpoint_steps": list(TRAINING_CHECKPOINT_STEPS),
                "checkpoint_sha256_by_step": checkpoint_hashes,
            }
        )
    )
    return training


def test_aggregate_selection_requires_all_hash_bound_carried_and_ordinary_evidence(
    tmp_path: Path,
    monkeypatch,
):
    from tools import evaluate_g1_zero_head_feature_transfer as evaluator

    monkeypatch.setattr(
        evaluator,
        "validate_code_provenance",
        lambda commit: {
            "repository": "/tmp/repository",
            "code_commit": commit,
            "dirty_patch_sha256": "0" * 64,
        },
    )

    training = _write_selection_fixture(tmp_path)
    output = tmp_path / "selection.json"
    manifest = evaluator.aggregate_selection(
        evaluation_root=tmp_path,
        training_validation_path=training,
        output_path=output,
        expected_code_commit="c" * 40,
    )

    assert manifest["valid"] is True
    assert manifest["outcome"] == "zero-head-features-advance"
    assert manifest["selected_update"] == 32
    assert set(manifest["evaluation_sha256"]) == {"8", "16", "32", "64"}
    assert json.loads(output.read_text()) == manifest

    phase_path = tmp_path / "update016/ordinary/phase_grid_summary.json"
    phase = json.loads(phase_path.read_text())
    phase["checkpoint_sha256"] = "f" * 64
    phase_path.write_text(json.dumps(phase))
    with pytest.raises(ValueError, match="phase-grid provenance"):
        evaluator.aggregate_selection(
            evaluation_root=tmp_path,
            training_validation_path=training,
            output_path=tmp_path / "bad.json",
            expected_code_commit="c" * 40,
        )


def test_selection_cli_requires_training_evaluations_and_output():
    from tools.select_g1_zero_head_feature_transfer import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--evaluation-root",
            "/tmp/evaluations",
            "--training-validation",
            "/tmp/training.json",
            "--output",
            "/tmp/selection.json",
            "--code-commit",
            "d" * 40,
        ]
    )
    assert args.output == Path("/tmp/selection.json")
