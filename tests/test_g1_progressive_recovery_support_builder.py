from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _source_bank() -> dict[str, np.ndarray]:
    phases = np.repeat(np.asarray([0, 100, 200, 300, 400]), 24)
    rows = phases.size
    return {
        "qpos": np.arange(rows * 36, dtype=np.float64).reshape(rows, 36),
        "qvel": np.arange(rows * 35, dtype=np.float64).reshape(rows, 35),
        "phase": phases + np.tile(np.arange(24), 5),
        "last_act": np.zeros((rows, 29), dtype=np.float64),
        "actor_obs_history": np.zeros((rows, 10, 328), dtype=np.float64),
        "fresh_actor_frame": np.zeros((rows, 328), dtype=np.float64),
        "action": np.zeros((rows, 29), dtype=np.float64),
        "source_start_phase": phases,
        "source_step": np.tile(np.arange(24), 5),
        "transitions_to_terminal": np.tile(np.arange(24, 0, -1), 5),
        "terminal": np.zeros(rows, dtype=np.float64),
        "termination_errors": np.zeros((rows, 4), dtype=np.float64),
        "termination_thresholds": np.ones(4, dtype=np.float64),
    }


def test_targeted_bank_selects_exact_phase_zero_rows():
    from tools.build_g1_progressive_recovery_support import build_targeted_bank

    source = _source_bank()
    targeted = build_targeted_bank(source, source_phase=0)

    assert targeted["qpos"].shape == (24, 36)
    np.testing.assert_array_equal(targeted["qpos"], source["qpos"][:24])
    np.testing.assert_array_equal(targeted["source_start_phase"], 0)
    np.testing.assert_array_equal(
        targeted["termination_thresholds"], source["termination_thresholds"]
    )


def test_targeted_bank_rejects_noncanonical_source_layout():
    from tools.build_g1_progressive_recovery_support import build_targeted_bank

    source = _source_bank()
    source["source_start_phase"][24] = 0

    with pytest.raises(ValueError, match="five exact 24-row bands"):
        build_targeted_bank(source, source_phase=0)


def test_support_payload_binds_exact_compact_gate():
    from tools.build_g1_progressive_recovery_support import (
        build_support_artifact,
    )

    positives = np.stack(
        [np.asarray([0.05 * index, 0.0]) for index in range(24)]
    ).astype(np.float32)
    negatives = np.asarray([[5.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    phases = np.arange(80, 104, dtype=np.int32)

    arrays, summary = build_support_artifact(
        positives,
        negatives,
        phases,
        source_bank_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        reference_sha256="c" * 64,
    )

    assert summary["valid"] is True
    assert summary["positive_rows"] == 24
    assert summary["protected_negative_rows"] == 2
    assert summary["protected_negative_max_gate"] == 0.0
    assert arrays["anchors"].shape == (24, 2)
    assert float(arrays["radius"]) == pytest.approx(1.925)
    assert int(arrays["phase_min"]) == 80
    assert int(arrays["phase_max"]) == 103


def test_atomic_publication_round_trips_without_pickle(tmp_path: Path):
    from tools.build_g1_progressive_recovery_support import publish_artifacts

    targeted = _source_bank()
    targeted = {
        key: (value[:24] if value.shape and value.shape[0] == 120 else value)
        for key, value in targeted.items()
    }
    support = {
        "anchors": np.zeros((24, 328), dtype=np.float32),
        "radius": np.asarray(1.0, dtype=np.float32),
        "phase_min": np.asarray(80, dtype=np.int32),
        "phase_max": np.asarray(103, dtype=np.int32),
        "taper": np.asarray(4, dtype=np.int32),
        "positive_leave_one_out_distances": np.ones(24, dtype=np.float32),
        "protected_negative_distances": np.ones(3, dtype=np.float32) * 3.0,
    }
    summary = {"valid": True, "protocol": "test"}

    published = publish_artifacts(
        output_directory=tmp_path,
        targeted_arrays=targeted,
        support_arrays=support,
        support_summary=summary,
    )

    with np.load(published["targeted_bank_path"], allow_pickle=False) as archive:
        assert archive["qpos"].shape == (24, 36)
    with np.load(published["support_path"], allow_pickle=False) as archive:
        assert archive["anchors"].shape == (24, 328)
    manifest = json.loads(
        Path(published["support_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["support_sha256"] == published["support_sha256"]
    assert manifest["targeted_bank_sha256"] == published["targeted_bank_sha256"]


def test_parser_requires_all_pinned_inputs():
    from tools.build_g1_progressive_recovery_support import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pkl",
            "--checkpoint-sha256",
            "a" * 64,
            "--hparams",
            "hparams.json",
            "--hparams-sha256",
            "b" * 64,
            "--reference-path",
            "reference.npz",
            "--reference-sha256",
            "c" * 64,
            "--source-bank",
            "bank.npz",
            "--source-bank-sha256",
            "d" * 64,
            "--code-commit",
            "e" * 40,
            "--output-directory",
            "output",
        ]
    )
    assert parsed.seed == 0
