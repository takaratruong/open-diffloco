from __future__ import annotations

import inspect

import numpy as np
import pytest


SOURCE_PHASES = (0, 100, 200, 300, 400)
SURVIVAL = (116, 63, 49, 39, 47)


def _arrays(
    survival: tuple[int, ...] = SURVIVAL,
) -> dict[str, np.ndarray]:
    from tools.build_g1_history_carried_reset_bank import (
        select_preterminal_indices,
    )

    chunks = []
    for phase, steps in zip(SOURCE_PHASES, survival, strict=True):
        indices = select_preterminal_indices(steps)
        chunks.append((phase, steps, indices))
    rows = 120
    qpos = np.zeros((rows, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    history = np.zeros((rows, 10, 328), dtype=np.float64)
    return {
        "qpos": qpos,
        "qvel": np.zeros((rows, 35), dtype=np.float64),
        "phase": np.concatenate(
            [phase + indices for phase, _, indices in chunks]
        ),
        "last_act": np.zeros((rows, 29), dtype=np.float64),
        "actor_obs_history": history,
        "fresh_actor_frame": history[:, -1].copy(),
        "action": np.zeros((rows, 29), dtype=np.float64),
        "source_start_phase": np.concatenate(
            [np.full(indices.size, phase) for phase, _, indices in chunks]
        ),
        "source_step": np.concatenate(
            [indices for _, _, indices in chunks]
        ),
        "transitions_to_terminal": np.concatenate(
            [steps - indices for _, steps, indices in chunks]
        ),
        "terminal": np.zeros(rows, dtype=np.float64),
        "termination_errors": np.zeros((rows, 4), dtype=np.float64),
        "termination_thresholds": np.asarray(
            [0.25, 1.3, 0.8, 0.4], dtype=np.float64
        ),
    }


def test_lafan_bank_requires_five_exact_preterminal_bands():
    from tools.build_g1_e023_lafan_carried_reset_bank import (
        LAFAN_SOURCE_PHASES,
        build_lafan_bank_summary,
    )

    assert LAFAN_SOURCE_PHASES == SOURCE_PHASES
    summary = build_lafan_bank_summary(
        _arrays(), observed_survival=SURVIVAL, frame_dim=328
    )
    assert summary["valid"] is True
    assert summary["rows"] == 120
    assert summary["rows_per_source"] == [24] * 5
    assert summary["source_survival"] == list(SURVIVAL)

    tolerant = (115, 63, 49, 39, 46)
    tolerant_summary = build_lafan_bank_summary(
        _arrays(tolerant), observed_survival=tolerant, frame_dim=328
    )
    assert tolerant_summary["source_survival"] == list(tolerant)

    with pytest.raises(ValueError, match="zero-shot baseline"):
        build_lafan_bank_summary(
            _arrays((113, 63, 49, 39, 47)),
            observed_survival=(113, 63, 49, 39, 47),
            frame_dim=328,
        )


def test_lafan_bank_parser_requires_parent_hparams_and_clean_code():
    from tools.build_g1_e023_lafan_carried_reset_bank import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "/tmp/policy.pkl",
            "--checkpoint-sha256",
            "a" * 64,
            "--hparams",
            "/tmp/hparams.json",
            "--hparams-sha256",
            "b" * 64,
            "--reference-path",
            "/tmp/reference.npz",
            "--reference-sha256",
            "c" * 64,
            "--code-commit",
            "d" * 40,
            "--output-npz",
            "/tmp/bank.npz",
            "--output-json",
            "/tmp/bank.json",
        ]
    )
    assert args.code_commit == "d" * 40
    assert args.seed == 0


def test_e023_collection_uses_the_registered_compiled_step_boundary():
    from tools import build_g1_e023_carried_reset_bank as collector

    source = inspect.getsource(collector.collect_e023_bank)
    assert "build_compiled_step(env)" in source
    assert "step_fn=compiled_step" in source
    assert "_load_policy(" in source
    assert "env, checkpoint_path, seed" in source
