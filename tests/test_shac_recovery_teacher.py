from __future__ import annotations

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


E034_SUCCESS_MASK = np.asarray(
    [True] * 12 + [False, True] + [False] * 10,
    dtype=bool,
)


def _write_teacher(path: Path) -> str:
    parent = np.zeros((24, 32, 29), dtype=np.float64)
    correction = np.full((24, 32, 29), 0.25, dtype=np.float64)
    np.savez_compressed(
        path,
        actor_obs=np.zeros((24, 32, 3280), dtype=np.float64),
        parent_action=parent,
        correction=correction,
        effective_action=np.clip(parent + correction, -1.0, 1.0),
        success_mask=E034_SUCCESS_MASK,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_teacher_batch_selects_exact_success_rows_and_reproduces_loss(
    tmp_path: Path,
):
    from src.algorithms.shac.recovery_teacher import (
        load_recovery_teacher_batch,
        recovery_teacher_imitation_loss,
    )

    path = tmp_path / "teacher.npz"
    digest = _write_teacher(path)
    batch = load_recovery_teacher_batch(path, expected_sha256=digest)

    assert batch.actor_obs.shape == (416, 3280)
    assert batch.parent_action.shape == (416, 29)
    assert batch.teacher_correction.shape == (416, 29)
    assert batch.teacher_effective_action.shape == (416, 29)
    loss = recovery_teacher_imitation_loss(
        jnp.asarray(batch.teacher_correction),
        jnp.asarray(batch.parent_action),
        jnp.asarray(batch.teacher_correction),
        jnp.asarray(batch.teacher_effective_action),
    )
    np.testing.assert_allclose(loss, 0.0, rtol=0, atol=1e-12)


def test_teacher_batch_fails_closed_on_hash_mask_shape_and_boundary(tmp_path: Path):
    from src.algorithms.shac.recovery_teacher import load_recovery_teacher_batch

    path = tmp_path / "teacher.npz"
    _write_teacher(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_recovery_teacher_batch(path, expected_sha256="0" * 64)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["success_mask"] = np.roll(arrays["success_mask"], 1)
    wrong_mask = tmp_path / "wrong-mask.npz"
    np.savez_compressed(wrong_mask, **arrays)
    wrong_mask_digest = hashlib.sha256(wrong_mask.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="success mask"):
        load_recovery_teacher_batch(
            wrong_mask, expected_sha256=wrong_mask_digest
        )

    arrays["success_mask"] = E034_SUCCESS_MASK
    arrays["actor_obs"] = arrays["actor_obs"][:, :, :-1]
    wrong_shape = tmp_path / "wrong-shape.npz"
    np.savez_compressed(wrong_shape, **arrays)
    wrong_shape_digest = hashlib.sha256(wrong_shape.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="actor_obs shape"):
        load_recovery_teacher_batch(
            wrong_shape, expected_sha256=wrong_shape_digest
        )

    arrays["actor_obs"] = np.zeros((24, 32, 3280), dtype=np.float64)
    arrays["effective_action"] = arrays["effective_action"].copy()
    arrays["effective_action"][0, 0, 0] += 0.1
    wrong_boundary = tmp_path / "wrong-boundary.npz"
    np.savez_compressed(wrong_boundary, **arrays)
    boundary_digest = hashlib.sha256(wrong_boundary.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="effective action"):
        load_recovery_teacher_batch(
            wrong_boundary, expected_sha256=boundary_digest
        )


def test_conflict_projection_removes_opposition_and_caps_teacher_norm():
    from src.algorithms.shac.recovery_teacher import (
        mix_conflict_projected_teacher_gradient,
    )

    result = mix_conflict_projected_teacher_gradient(
        {"w": jnp.asarray([1.0, 0.0])},
        {"w": jnp.asarray([-1.0, 2.0])},
        max_ratio=0.5,
    )

    np.testing.assert_allclose(result.projected_teacher_gradient["w"], [0.0, 2.0])
    np.testing.assert_allclose(result.applied_teacher_gradient["w"], [0.0, 0.5])
    np.testing.assert_allclose(result.combined_gradient["w"], [1.0, 0.5])
    np.testing.assert_allclose(result.physics_teacher_dot, -1.0)
    np.testing.assert_allclose(result.applied_scale, 0.25)
    np.testing.assert_allclose(result.applied_teacher_norm, 0.5)
    assert bool(result.valid)


def test_gradient_mix_handles_aligned_zero_and_nonfinite_trees():
    from src.algorithms.shac.recovery_teacher import (
        mix_conflict_projected_teacher_gradient,
    )

    aligned = mix_conflict_projected_teacher_gradient(
        {"w": jnp.asarray([1.0, 0.0])},
        {"w": jnp.asarray([2.0, 0.0])},
        max_ratio=0.5,
    )
    np.testing.assert_allclose(aligned.combined_gradient["w"], [1.5, 0.0])
    np.testing.assert_allclose(aligned.applied_scale, 0.25)

    zero = mix_conflict_projected_teacher_gradient(
        {"w": jnp.zeros(2)}, {"w": jnp.ones(2)}, max_ratio=0.5
    )
    np.testing.assert_array_equal(zero.combined_gradient["w"], np.zeros(2))
    np.testing.assert_allclose(zero.applied_scale, 0.0)
    assert bool(zero.valid)

    nonfinite = mix_conflict_projected_teacher_gradient(
        {"w": jnp.asarray([1.0, 0.0])},
        {"w": jnp.asarray([jnp.nan, 0.0])},
        max_ratio=0.5,
    )
    assert not bool(nonfinite.valid)
    assert np.isfinite(np.asarray(nonfinite.combined_gradient["w"])).all()


def test_resume_settings_require_complete_metadata_and_explicit_authority():
    from src.algorithms.shac.recovery_teacher import (
        resolve_recovery_teacher_resume_settings,
    )

    requested = dict(
        requested_path="/tmp/teacher.npz",
        requested_sha256="a" * 64,
        requested_ratio=0.5,
    )
    fresh = resolve_recovery_teacher_resume_settings(
        **requested, resumed_hparams=None, is_resume=False, allow_change=False
    )
    assert fresh == ("/tmp/teacher.npz", "a" * 64, 0.5)

    with pytest.raises(ValueError, match="metadata"):
        resolve_recovery_teacher_resume_settings(
            **requested, resumed_hparams={}, is_resume=True, allow_change=False
        )
    with pytest.raises(ValueError, match="metadata"):
        resolve_recovery_teacher_resume_settings(
            **requested,
            resumed_hparams={"actor_recovery_teacher_dataset_path": "/tmp/x"},
            is_resume=True,
            allow_change=False,
        )

    disabled = {
        "actor_recovery_teacher_enabled": False,
        "actor_recovery_teacher_dataset_path": None,
        "actor_recovery_teacher_dataset_sha256": None,
        "actor_recovery_teacher_gradient_ratio": 0.0,
    }
    with pytest.raises(ValueError, match="explicit authority"):
        resolve_recovery_teacher_resume_settings(
            **requested,
            resumed_hparams=disabled,
            is_resume=True,
            allow_change=False,
        )
    assert resolve_recovery_teacher_resume_settings(
        **requested,
        resumed_hparams=disabled,
        is_resume=True,
        allow_change=True,
    ) == ("/tmp/teacher.npz", "a" * 64, 0.5)


def test_legacy_resume_without_teacher_metadata_preserves_disabled_treatment():
    from src.algorithms.shac.recovery_teacher import (
        resolve_recovery_teacher_resume_settings,
    )

    assert resolve_recovery_teacher_resume_settings(
        requested_path=None,
        requested_sha256=None,
        requested_ratio=0.0,
        resumed_hparams={},
        is_resume=True,
        allow_change=False,
    ) == (None, None, 0.0)
