from __future__ import annotations

import numpy as np
import pytest

from tools.build_g1_e034_recovery_teacher_dataset import (
    E034_SURVIVAL,
    build_parser,
    publish_teacher_dataset,
    validate_teacher_arrays,
)


def _valid_arrays() -> dict[str, np.ndarray]:
    terminal = np.zeros((24, 32), dtype=bool)
    alive = np.ones((24, 32), dtype=bool)
    for row, survival in enumerate(E034_SURVIVAL):
        if survival < 32:
            terminal[row, survival] = True
            alive[row, survival + 1 :] = False
    raw = np.zeros((24, 32, 29), dtype=np.float64)
    raw[14:, :, 0] = 1.5
    return {
        "actor_obs": np.zeros((24, 32, 3280), dtype=np.float32),
        "phase": np.broadcast_to(np.arange(32), (24, 32)).copy(),
        "parent_action": np.zeros((24, 32, 29), dtype=np.float64),
        "correction": raw.copy(),
        "raw_action": raw,
        "effective_action": np.clip(raw, -1.0, 1.0),
        "alive": alive,
        "terminal": terminal,
        "reward": np.zeros((24, 32), dtype=np.float64),
        "normalized_termination_errors": np.zeros(
            (24, 32, 4), dtype=np.float64
        ),
    }


def test_teacher_dataset_requires_exact_e034_replay_and_reports_clipping():
    summary = validate_teacher_arrays(_valid_arrays())

    assert summary["survival"] == list(E034_SURVIVAL)
    assert summary["successful_starts"] == 13
    assert summary["teacher_rows"] == 13 * 32
    assert summary["failed_clip_fraction"] > summary["recovered_clip_fraction"]


@pytest.mark.parametrize(
    "mutation",
    ("shape", "survival", "clip", "nonfinite"),
)
def test_teacher_dataset_fails_closed_on_invalid_evidence(mutation: str):
    arrays = _valid_arrays()
    if mutation == "shape":
        arrays["actor_obs"] = arrays["actor_obs"][:, :, :-1]
    elif mutation == "survival":
        arrays["terminal"][14, 26] = False
    elif mutation == "clip":
        arrays["effective_action"][0, 0, 0] = 0.25
    else:
        arrays["reward"][0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_teacher_arrays(arrays)


def test_teacher_dataset_publication_is_hash_bound_and_manifest_last(tmp_path):
    arrays = _valid_arrays()

    manifest = publish_teacher_dataset(
        output_directory=tmp_path,
        arrays=arrays,
        provenance={"code_commit": "a" * 40},
    )

    assert manifest["valid"] is True
    assert len(manifest["dataset_sha256"]) == 64
    assert (tmp_path / "e034_recovery_teacher_dataset.npz").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_teacher_dataset_parser_requires_all_pinned_inputs():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pkl",
            "--hparams",
            "hparams.json",
            "--reference-path",
            "reference.npz",
            "--source-bank",
            "bank.npz",
            "--oracle-evidence",
            "oracle.npz",
            "--output-directory",
            "output",
            "--code-commit",
            "a" * 40,
        ]
    )

    assert args.seed == 0
    assert args.oracle_evidence.name == "oracle.npz"
