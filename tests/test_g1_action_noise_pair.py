from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from src.core.rmr_action_noise import RMR_ACTION_STD, RMR_ACTION_STD_JOINT_NAMES


def _write_arm(
    root: Path,
    arm: str,
    *,
    zero: bool,
    reset: str = "r" * 64,
    rows: int = 2,
    terminal: bool = True,
) -> dict:
    from tools.evaluate_g1_action_noise_pair import RECORD_COLUMNS, epsilon_tape

    directory = root / arm
    directory.mkdir(parents=True)
    std = np.zeros(29) if zero else np.asarray(RMR_ACTION_STD, dtype=np.float64)
    epsilon = epsilon_tape(seed=0, steps=rows)
    noise = np.zeros((rows, 29), dtype=np.float64) if zero else epsilon * std
    mean = np.full((rows, 29), 0.2)
    action = np.clip(mean + noise, -1.0, 1.0)
    values = np.zeros((rows, len(RECORD_COLUMNS)), dtype=np.float64)
    values[:, 0] = np.arange(rows)
    values[:, 1] = np.arange(rows)
    values[:, 11] = np.arange(1, rows + 1)
    if terminal:
        values[-1, 4] = 1.0
    np.savez_compressed(
        directory / "evaluation.npz",
        columns=np.asarray(RECORD_COLUMNS),
        values=values,
        action_mean=mean,
        epsilon=epsilon,
        action_noise=noise,
        action=action,
        joint_names=np.asarray(RMR_ACTION_STD_JOINT_NAMES),
        xfrc_applied=np.zeros((rows, 31, 6), dtype=np.float64),
        xfrc_body_count=np.asarray(31, dtype=np.int64),
        remaining_reference_transitions=np.asarray(499, dtype=np.int64),
        requested_step_limit=np.asarray(-1, dtype=np.int64),
    )
    summary = {
        "steps": rows,
        "terminal": terminal,
        "evaluation_start_phase": 0,
        "remaining_reference_transitions": 499,
        "completed_reference_suffix": False,
        "intermediate_reset_occurred": terminal,
        "paired_reset_state_sha256": reset,
        "action_noise_exact_zero": zero,
        "assistance_exact_zero": True,
        "noise_seed": 0,
        "noise_joint_names": list(RMR_ACTION_STD_JOINT_NAMES),
        "requested_step_limit": None,
        "artificially_truncated": False,
        "checkpoint_sha256": "c" * 64,
        "reference_sha256": "d" * 64,
        "mean_reward": 0.0,
        "max_anchor_z_error": 0.0,
        "max_anchor_xy_error": 0.0,
        "max_gravity_z_error": 0.0,
        "max_distal_z_error": 0.0,
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    imageio.mimsave(directory / "evaluation.mp4", [frame], fps=1)
    imageio.imwrite(directory / "contact_sheet.png", frame)
    return summary


def _provenance() -> dict:
    return {
        "checkpoint_sha256": "c" * 64,
        "reference_sha256": "d" * 64,
        "seed": 0,
        "action_noise_joint_names": list(RMR_ACTION_STD_JOINT_NAMES),
        "rmr_action_std": np.asarray(RMR_ACTION_STD, dtype=np.float32).tolist(),
        "phase": 0,
        "expected_remaining_reference_transitions": 499,
    }


def test_paired_reset_calls_reset_once_and_branches_the_identical_state() -> None:
    from tools.evaluate_g1_action_noise_pair import paired_reset

    class Environment:
        def __init__(self):
            self.calls = []

        def reset_at_phase(self, key, zero, phase):
            self.calls.append((key, zero, phase))
            return SimpleNamespace(token="same-state")

    environment = Environment()
    deterministic, noisy = paired_reset(environment, phase=7, seed=0)
    assert len(environment.calls) == 1
    assert deterministic is noisy


def test_deterministic_noise_tape_is_exact_zero_and_rmr_tape_is_seeded() -> None:
    from tools.evaluate_g1_action_noise_pair import action_noise_tape

    zero = action_noise_tape(seed=0, steps=3, action_std=np.zeros(29))
    first = action_noise_tape(seed=0, steps=3, action_std=RMR_ACTION_STD)
    assert np.array_equal(zero, np.zeros((3, 29)))
    assert np.array_equal(
        first, action_noise_tape(seed=0, steps=3, action_std=RMR_ACTION_STD)
    )
    assert not np.array_equal(first, np.zeros((3, 29)))
    assert not np.signbit(zero).any()


def test_manifest_reopens_real_artifacts_and_hash_binds_them(tmp_path: Path) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(tmp_path, "deterministic", zero=True),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False),
    }
    manifest = build_pair_manifest(
        output_dir=tmp_path, provenance=_provenance(), arms=arms
    )
    assert manifest["valid"] is True
    assert set(manifest["artifact_sha256"]) == {"deterministic", "rmr-noisy"}
    assert len(manifest["artifact_sha256"]["rmr-noisy"]) == 4


def test_manifest_accepts_natural_unequal_terminal_lengths_with_shared_prefix(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(tmp_path, "deterministic", zero=True, rows=2),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False, rows=1),
    }
    manifest = build_pair_manifest(
        output_dir=tmp_path, provenance=_provenance(), arms=arms
    )
    assert manifest["arms"]["rmr-noisy"]["terminal"] is True


def test_manifest_rejects_hidden_truncation_and_cochanged_constants(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(
            tmp_path, "deterministic", zero=True, terminal=False
        ),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False, terminal=False),
    }
    with pytest.raises(ValueError, match="complete"):
        build_pair_manifest(output_dir=tmp_path, provenance=_provenance(), arms=arms)
    bad = _provenance()
    bad.update(phase=1, expected_remaining_reference_transitions=498)
    with pytest.raises(ValueError, match="immutable"):
        build_pair_manifest(output_dir=tmp_path, provenance=bad, arms=arms)


def test_manifest_rejects_self_consistent_wrong_wrench_body_count(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(tmp_path, "deterministic", zero=True),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False),
    }
    for arm in arms:
        path = tmp_path / arm / "evaluation.npz"
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["xfrc_applied"] = np.zeros((2, 1, 6))
        arrays["xfrc_body_count"] = np.asarray(1)
        np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="wrench"):
        build_pair_manifest(output_dir=tmp_path, provenance=_provenance(), arms=arms)


@pytest.mark.parametrize(
    "tamper",
    (
        "summary",
        "noise",
        "action",
        "joint-order",
        "assistance",
        "reset",
        "epsilon",
        "negative-zero",
        "xfrc",
        "phase",
        "remaining",
        "sentinel",
        "xfrc-shape",
    ),
)
def test_manifest_rejects_tampered_or_cross_arm_artifacts(
    tmp_path: Path, tamper: str
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(tmp_path, "deterministic", zero=True),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False),
    }
    path = tmp_path / "rmr-noisy"
    if tamper in {"summary", "assistance", "reset"}:
        summary = json.loads((path / "summary.json").read_text())
        if tamper == "summary":
            summary["steps"] = 1
        if tamper == "assistance":
            summary["assistance_exact_zero"] = False
        if tamper == "reset":
            summary["paired_reset_state_sha256"] = "x" * 64
        (path / "summary.json").write_text(json.dumps(summary))
    elif tamper in {"summary", "assistance", "reset"}:
        pass
    else:
        with np.load(path / "evaluation.npz") as source:
            arrays = {key: source[key] for key in source.files}
        if tamper == "noise":
            arrays["action_noise"][0, 0] += 1.0
        if tamper == "action":
            arrays["action"][0, 0] += 0.1
        if tamper == "joint-order":
            arrays["joint_names"] = arrays["joint_names"][::-1]
        if tamper == "epsilon":
            arrays["epsilon"][0, 0] += 1.0
        if tamper == "negative-zero":
            arrays["action_noise"][0, 0] = -0.0
        if tamper == "xfrc":
            arrays["xfrc_applied"][0, 0, 0] = -0.0
        if tamper == "phase":
            arrays["values"][0, 1] = 1
        if tamper == "remaining":
            arrays["remaining_reference_transitions"] = np.asarray(498)
        if tamper == "sentinel":
            arrays["requested_step_limit"] = np.asarray(1)
        if tamper == "xfrc-shape":
            arrays["xfrc_applied"] = np.zeros((2,))
        np.savez_compressed(path / "evaluation.npz", **arrays)
    with pytest.raises(ValueError):
        build_pair_manifest(output_dir=tmp_path, provenance=_provenance(), arms=arms)


def test_manifest_rejects_stale_supplied_summary_and_truncated_publication(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(tmp_path, "deterministic", zero=True),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False),
    }
    stale = {**arms, "deterministic": {**arms["deterministic"], "steps": 99}}
    with pytest.raises(ValueError, match="supplied arm summary"):
        build_pair_manifest(output_dir=tmp_path, provenance=_provenance(), arms=stale)
    summary = json.loads((tmp_path / "deterministic" / "summary.json").read_text())
    summary.update(requested_step_limit=1, artificially_truncated=True)
    (tmp_path / "deterministic" / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="artificially truncated"):
        build_pair_manifest(output_dir=tmp_path, provenance=_provenance(), arms=arms)


def test_manifest_rejects_unpinned_vector_and_wrong_media_frame_count(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    arms = {
        "deterministic": _write_arm(tmp_path, "deterministic", zero=True),
        "rmr-noisy": _write_arm(tmp_path, "rmr-noisy", zero=False),
    }
    bad = _provenance()
    bad["rmr_action_std"][0] += 1e-10
    with pytest.raises(ValueError, match="pinned"):
        build_pair_manifest(output_dir=tmp_path, provenance=bad, arms=arms)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    imageio.mimsave(tmp_path / "rmr-noisy" / "evaluation.mp4", [frame, frame], fps=1)
    with pytest.raises(ValueError, match="frame count"):
        build_pair_manifest(output_dir=tmp_path, provenance=_provenance(), arms=arms)


def test_rollout_noise_uses_exact_positive_zero_tape() -> None:
    from tools.evaluate_g1_action_noise_pair import rollout_noise

    noise = rollout_noise(seed=0, steps=3, action_std=np.zeros(29))
    assert np.array_equal(noise, np.zeros((3, 29)))
    assert not np.signbit(noise).any()


def test_output_directory_must_be_fresh_and_staging_failure_publishes_nothing(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import prepare_output_staging

    output = tmp_path / "published"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_output_staging(output)
    assert not (tmp_path / ".published.staging").exists()


def test_provenance_hashes_checkpoint_reference_and_runtime_assets(
    tmp_path: Path,
) -> None:
    from tools.evaluate_g1_action_noise_pair import build_provenance

    checkpoint, reference, model, controller = (
        tmp_path / name
        for name in ("checkpoint.pkl", "reference.npz", "g1.xml", "controller.npz")
    )
    for path, content in (
        (checkpoint, b"checkpoint"),
        (reference, b"reference"),
        (model, b"model"),
        (controller, b"controller"),
    ):
        path.write_bytes(content)
    (tmp_path / "hparams.json").write_bytes(b"hparams")
    provenance = build_provenance(
        checkpoint=checkpoint,
        reference=reference,
        model_path=model,
        controller_path=controller,
        seed=0,
        solver_profile="g1-4x5",
    )
    assert provenance["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
