from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_paired_reset_calls_reset_once_and_branches_the_identical_state() -> None:
    from tools.evaluate_g1_action_noise_pair import paired_reset

    class Environment:
        def __init__(self) -> None:
            self.calls = []

        def reset_at_phase(self, key, zero, phase):
            self.calls.append((key, zero, phase))
            return SimpleNamespace(token="same-state")

    environment = Environment()
    deterministic, noisy = paired_reset(environment, phase=7, seed=0)

    assert len(environment.calls) == 1
    assert deterministic is noisy
    assert deterministic.token == "same-state"


def test_deterministic_noise_tape_is_exact_zero_and_rmr_tape_is_seeded() -> None:
    from tools.evaluate_g1_action_noise_pair import action_noise_tape
    from src.core.rmr_action_noise import RMR_ACTION_STD

    zero = action_noise_tape(seed=0, steps=3, action_std=np.zeros(29))
    first = action_noise_tape(seed=0, steps=3, action_std=RMR_ACTION_STD)
    repeated = action_noise_tape(seed=0, steps=3, action_std=RMR_ACTION_STD)

    assert zero.shape == (3, 29)
    assert np.array_equal(zero, np.zeros((3, 29)))
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, np.zeros((3, 29)))


def test_noisy_action_uses_pinned_joint_order_and_clips_after_noise() -> None:
    from tools.evaluate_g1_action_noise_pair import noisy_action
    from src.core.rmr_action_noise import RMR_ACTION_STD_JOINT_NAMES

    mean = np.full(29, 0.95)
    epsilon = np.ones(29)
    std = np.linspace(0.01, 0.29, 29)
    result = noisy_action(
        mean,
        epsilon,
        std,
        actor_joint_names=RMR_ACTION_STD_JOINT_NAMES,
    )

    assert np.allclose(result, np.clip(mean + std, -1.0, 1.0), atol=1e-7)
    with pytest.raises(ValueError, match="pinned RMR actor joint order"):
        noisy_action(mean, epsilon, std, actor_joint_names=tuple(reversed(RMR_ACTION_STD_JOINT_NAMES)))


def test_provenance_hashes_checkpoint_reference_and_runtime_assets(tmp_path: Path) -> None:
    from tools.evaluate_g1_action_noise_pair import build_provenance

    checkpoint = tmp_path / "checkpoint.pkl"
    reference = tmp_path / "reference.npz"
    model = tmp_path / "g1.xml"
    controller = tmp_path / "controller.npz"
    for path, content in ((checkpoint, b"checkpoint"), (reference, b"reference"), (model, b"model"), (controller, b"controller")):
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
    assert provenance["reference_sha256"] == hashlib.sha256(b"reference").hexdigest()
    assert provenance["runtime_assets"]["model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert provenance["runtime_assets"]["controller_sha256"] == hashlib.sha256(b"controller").hexdigest()


def test_pair_aggregation_requires_complete_matching_arm_artifacts(tmp_path: Path) -> None:
    from tools.evaluate_g1_action_noise_pair import build_pair_manifest

    for arm in ("deterministic", "rmr-noisy"):
        directory = tmp_path / arm
        directory.mkdir()
        for name in ("summary.json", "evaluation.npz", "evaluation.mp4", "contact_sheet.png"):
            (directory / name).write_bytes(b"artifact")

    manifest = build_pair_manifest(
        output_dir=tmp_path,
        provenance={"checkpoint_sha256": "a" * 64},
        arms={
            "deterministic": {"steps": 2, "action_noise_exact_zero": True},
            "rmr-noisy": {"steps": 2, "action_noise_exact_zero": False},
        },
    )
    assert manifest["valid"] is True
    (tmp_path / "rmr-noisy" / "evaluation.mp4").unlink()
    with pytest.raises(ValueError, match="missing required artifact"):
        build_pair_manifest(
            output_dir=tmp_path,
            provenance={"checkpoint_sha256": "a" * 64},
            arms={
                "deterministic": {"steps": 2, "action_noise_exact_zero": True},
                "rmr-noisy": {"steps": 2, "action_noise_exact_zero": False},
            },
        )
