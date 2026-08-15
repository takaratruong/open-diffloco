from __future__ import annotations

import numpy as np
import pytest


def _e023_hparams() -> dict[str, object]:
    return {
        "total_steps": 1_572_864,
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "actor_hidden": [512, 256, 128],
        "actor_layer_norm": True,
        "actor_zero_output": True,
        "actor_history_len": 10,
        "actor_reference_lookahead_steps": [4, 8, 12],
        "actor_reference_preview_mode": "delta",
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "reference_reset_noise_scale": 0.0,
        "domain_randomization": False,
        "actor_observation_noise": False,
        "squash_actor_mean": False,
        "clip_sampled_actor_actions": False,
        "solver_profile": "g1-4x5",
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
        "reference_stride": 1,
    }


def _bank_arrays(*, frame_dim: int = 328) -> dict[str, np.ndarray]:
    from tools.build_g1_history_carried_reset_bank import (
        select_preterminal_indices,
    )

    survival = (40, 40)
    source_phases = (0, 50)
    chunks = []
    for source_phase, steps in zip(source_phases, survival, strict=True):
        indices = select_preterminal_indices(steps)
        chunks.append(
            {
                "phase": source_phase + indices,
                "source_start_phase": np.full(indices.size, source_phase),
                "source_step": indices,
                "transitions_to_terminal": steps - indices,
            }
        )
    rows = 48
    qpos = np.zeros((rows, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    history = np.zeros((rows, 10, frame_dim), dtype=np.float64)
    return {
        "qpos": qpos,
        "qvel": np.zeros((rows, 35), dtype=np.float64),
        "phase": np.concatenate([chunk["phase"] for chunk in chunks]),
        "last_act": np.zeros((rows, 29), dtype=np.float64),
        "actor_obs_history": history,
        "fresh_actor_frame": history[:, -1].copy(),
        "action": np.zeros((rows, 29), dtype=np.float64),
        "source_start_phase": np.concatenate(
            [chunk["source_start_phase"] for chunk in chunks]
        ),
        "source_step": np.concatenate(
            [chunk["source_step"] for chunk in chunks]
        ),
        "transitions_to_terminal": np.concatenate(
            [chunk["transitions_to_terminal"] for chunk in chunks]
        ),
        "terminal": np.zeros(rows, dtype=np.float64),
        "termination_errors": np.zeros((rows, 4), dtype=np.float64),
        "termination_thresholds": np.asarray(
            [0.25, 1.3, 0.8, 0.4], dtype=np.float64
        ),
    }


def test_e023_contract_is_exact_and_rejects_drift():
    from tools.build_g1_e023_carried_reset_bank import (
        E023_SOURCE_PHASES,
        validate_e023_hparams,
    )

    assert E023_SOURCE_PHASES == (0, 50)
    contract = validate_e023_hparams(_e023_hparams())
    assert contract["actor_hidden"] == [512, 256, 128]
    assert contract["actor_reference_lookahead_steps"] == [4, 8, 12]
    assert contract["clip_sampled_actor_actions"] is False
    assert contract["reference_reset_noise_scale"] == 0.0

    for key, wrong in (
        ("env_variant", "g1_tracking_rmr_50hz_source_step"),
        ("total_steps", 1_572_863),
        ("reference_reset_noise_scale", 1.0),
        ("domain_randomization", True),
        ("clip_sampled_actor_actions", True),
    ):
        with pytest.raises(ValueError, match="E023 hparams"):
            validate_e023_hparams({**_e023_hparams(), key: wrong})


def test_e023_bank_summary_requires_two_complete_failure_bands():
    from tools.build_g1_e023_carried_reset_bank import build_e023_bank_summary

    summary = build_e023_bank_summary(
        _bank_arrays(), observed_survival=(40, 40), frame_dim=328
    )
    assert summary["valid"] is True
    assert summary["source_phases"] == [0, 50]
    assert summary["rows"] == 48
    assert summary["rows_per_source"] == [24, 24]
    assert summary["minimum_transitions_to_terminal"] == 6
    assert summary["maximum_transitions_to_terminal"] == 29

    with pytest.raises(ValueError, match="exactly two"):
        build_e023_bank_summary(
            _bank_arrays(), observed_survival=(40,), frame_dim=328
        )


def test_e023_bank_parser_requires_pinned_inputs():
    from tools.build_g1_e023_carried_reset_bank import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
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
    assert args.seed == 0
    assert args.code_commit == "d" * 40
