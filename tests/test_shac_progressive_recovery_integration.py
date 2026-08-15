from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.algorithms.shac.algorithm import (
    load_recovery_support_artifact,
    resolve_recovery_support_resume_setting,
    train,
)


def _write_support(path: Path) -> str:
    np.savez_compressed(
        path,
        anchors=np.zeros((24, 328), dtype=np.float32),
        radius=np.asarray(1.0, dtype=np.float32),
        phase_min=np.asarray(80, dtype=np.int32),
        phase_max=np.asarray(103, dtype=np.int32),
        taper=np.asarray(4, dtype=np.int32),
        positive_leave_one_out_distances=np.ones(24, dtype=np.float32) * 0.5,
        protected_negative_distances=np.ones(8, dtype=np.float32) * 3.0,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_loader_binds_sha_and_exact_contract(tmp_path: Path):
    path = tmp_path / "support.npz"
    digest = _write_support(path)

    support, report = load_recovery_support_artifact(path, expected_sha256=digest)

    assert support.anchors.shape == (24, 328)
    assert float(support.radius) == 1.0
    assert support.phase_min == 80
    assert support.phase_max == 103
    assert support.taper == 4
    assert report["sha256"] == digest
    assert report["protected_negative_max_gate"] == 0.0


def test_support_loader_rejects_hash_or_protected_distance(tmp_path: Path):
    path = tmp_path / "support.npz"
    digest = _write_support(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_recovery_support_artifact(path, expected_sha256="0" * 64)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["protected_negative_distances"] = np.asarray([0.5], dtype=np.float32)
    np.savez_compressed(path, **arrays)
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="protected negatives"):
        load_recovery_support_artifact(path, expected_sha256=changed)
    assert digest != changed


def test_resume_setting_allows_one_explicit_legacy_start():
    path, digest = "/tmp/support.npz", "a" * 64

    assert resolve_recovery_support_resume_setting(
        {"actor_state_gated_recovery": False},
        requested_path=path,
        requested_sha256=digest,
        allow_start=True,
        is_resume=True,
    ) == (path, digest)

    with pytest.raises(ValueError, match="explicit start authority"):
        resolve_recovery_support_resume_setting(
            {"actor_state_gated_recovery": False},
            requested_path=path,
            requested_sha256=digest,
            allow_start=False,
            is_resume=True,
        )


def test_resume_setting_rejects_missing_or_changed_treated_metadata():
    requested = "/tmp/support.npz"
    saved = {
        "actor_state_gated_recovery": True,
        "actor_state_gated_recovery_support_path": requested,
        "actor_state_gated_recovery_support_sha256": "a" * 64,
    }
    assert resolve_recovery_support_resume_setting(
        saved,
        requested_path=requested,
        requested_sha256="a" * 64,
        allow_start=False,
        is_resume=True,
    ) == (requested, "a" * 64)
    with pytest.raises(ValueError, match="must match the checkpoint"):
        resolve_recovery_support_resume_setting(
            saved,
            requested_path=requested,
            requested_sha256="b" * 64,
            allow_start=False,
            is_resume=True,
        )
    with pytest.raises(ValueError, match="metadata is missing"):
        resolve_recovery_support_resume_setting(
            None,
            requested_path=requested,
            requested_sha256="a" * 64,
            allow_start=False,
            is_resume=True,
        )


def test_train_exposes_support_and_authority_arguments():
    parameters = inspect.signature(train).parameters
    assert parameters["actor_state_gated_recovery_support_path"].default is None
    assert parameters["actor_state_gated_recovery_support_sha256"].default is None
    assert parameters["allow_resume_actor_state_gated_recovery_start"].default is False


def test_actor_loss_uses_pre_step_phase_for_gated_recovery():
    source = inspect.getsource(train)
    call = source.index("apply_state_gated_recovery(")
    step = source.index("next_state = env.step(state, noisy_action)")
    assert call < step
    assert 'state.info["phase"]' in source[call:step]
    assert 'transition["recovery_gate"]' in source


def test_recovery_telemetry_splits_carried_and_reference_activation():
    source = inspect.getsource(train)
    assert 'transition["reset_was_carried"]' in source
    assert '"actor_recovery_carried_activation_fraction"' in source
    assert '"actor_recovery_reference_activation_fraction"' in source
