"""Immutable recovery-teacher data and conflict-projected gradient mixing."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


_SOURCE_SHAPES = {
    "actor_obs": (24, 32, 3280),
    "parent_action": (24, 32, 29),
    "correction": (24, 32, 29),
    "effective_action": (24, 32, 29),
    "success_mask": (24,),
}
_E034_SUCCESS_MASK = np.asarray(
    [True] * 12 + [False, True] + [False] * 10,
    dtype=bool,
)
_RESUME_KEYS = (
    "actor_recovery_teacher_enabled",
    "actor_recovery_teacher_dataset_path",
    "actor_recovery_teacher_dataset_sha256",
    "actor_recovery_teacher_gradient_ratio",
)


class RecoveryTeacherBatch(NamedTuple):
    """The exact 416 successful E036 transitions."""

    actor_obs: np.ndarray
    parent_action: np.ndarray
    teacher_correction: np.ndarray
    teacher_effective_action: np.ndarray


class TeacherGradientMix(NamedTuple):
    """Combined gradient and scalar evidence for one teacher mix."""

    combined_gradient: Any
    projected_teacher_gradient: Any
    applied_teacher_gradient: Any
    physics_norm: jax.Array
    raw_teacher_norm: jax.Array
    projected_teacher_norm: jax.Array
    applied_teacher_norm: jax.Array
    combined_norm: jax.Array
    physics_teacher_dot: jax.Array
    physics_teacher_cosine: jax.Array
    applied_scale: jax.Array
    valid: jax.Array


class RecoveryTeacherGradientResult(NamedTuple):
    """Teacher loss, adapter-only gradient, and its bounded physics mix."""

    loss: jax.Array
    teacher_gradient: Any
    mix: TeacherGradientMix
    parent_gradient_max_abs: jax.Array
    enabled: jax.Array
    valid: jax.Array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recovery_teacher_batch(
    path: str | Path, *, expected_sha256: str
) -> RecoveryTeacherBatch:
    """Load and validate the immutable successful E036 supervision rows."""
    source = Path(path).resolve()
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise ValueError("recovery teacher SHA-256 does not match")
    with np.load(source, allow_pickle=False) as archive:
        if any(name not in archive for name in _SOURCE_SHAPES):
            raise ValueError("recovery teacher dataset is incomplete")
        arrays = {name: np.asarray(archive[name]) for name in _SOURCE_SHAPES}
    for name, shape in _SOURCE_SHAPES.items():
        value = arrays[name]
        if value.shape != shape:
            raise ValueError(f"recovery teacher {name} shape does not match")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"recovery teacher {name} must be finite")
    success_mask = np.asarray(arrays["success_mask"], dtype=bool)
    if not np.array_equal(success_mask, _E034_SUCCESS_MASK):
        raise ValueError("recovery teacher success mask does not match E034")
    expected_effective = np.clip(
        arrays["parent_action"] + arrays["correction"], -1.0, 1.0
    )
    if not np.allclose(
        expected_effective,
        arrays["effective_action"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("recovery teacher effective action is inconsistent")

    def selected(name: str) -> np.ndarray:
        value = arrays[name][success_mask]
        return np.asarray(value.reshape((-1, value.shape[-1]))).copy()

    batch = RecoveryTeacherBatch(
        actor_obs=selected("actor_obs"),
        parent_action=selected("parent_action"),
        teacher_correction=selected("correction"),
        teacher_effective_action=selected("effective_action"),
    )
    if batch.actor_obs.shape[0] != 416:
        raise ValueError("recovery teacher must contain exactly 416 transitions")
    return batch


def recovery_teacher_imitation_loss(
    predicted_correction: jax.Array,
    parent_action: jax.Array,
    teacher_correction: jax.Array,
    teacher_effective_action: jax.Array,
) -> jax.Array:
    """Match both the oracle correction and its real clipped action."""
    correction_error = jnp.mean(
        jnp.square(predicted_correction - teacher_correction)
    )
    predicted_effective = jnp.clip(
        parent_action + predicted_correction, -1.0, 1.0
    )
    effective_error = jnp.mean(
        jnp.square(predicted_effective - teacher_effective_action)
    )
    return correction_error + effective_error


def _tree_sum_squares(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return sum(
        (jnp.sum(jnp.square(value)) for value in leaves),
        jnp.asarray(0.0),
    )


def _tree_dot(left: Any, right: Any) -> jax.Array:
    return sum(
        (
            jnp.sum(left_leaf * right_leaf)
            for left_leaf, right_leaf in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
                strict=True,
            )
        ),
        jnp.asarray(0.0),
    )


def _tree_finite(tree: Any) -> jax.Array:
    return jnp.all(
        jnp.stack(
            [jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)]
        )
    )


def mix_conflict_projected_teacher_gradient(
    physics_gradient: Any,
    teacher_gradient: Any,
    *,
    max_ratio: float,
    epsilon: float = 1e-12,
) -> TeacherGradientMix:
    """Add a bounded teacher direction without opposing the physics direction."""
    if jax.tree_util.tree_structure(physics_gradient) != jax.tree_util.tree_structure(
        teacher_gradient
    ):
        raise ValueError("physics and teacher gradient trees do not match")
    safe_physics = jax.tree_util.tree_map(
        lambda value: jnp.where(jnp.isfinite(value), value, 0.0), physics_gradient
    )
    safe_teacher = jax.tree_util.tree_map(
        lambda value: jnp.where(jnp.isfinite(value), value, 0.0), teacher_gradient
    )
    ratio = jnp.asarray(max_ratio)
    eps = jnp.asarray(epsilon)
    valid = (
        _tree_finite(physics_gradient)
        & _tree_finite(teacher_gradient)
        & jnp.isfinite(ratio)
        & (ratio >= 0.0)
        & jnp.isfinite(eps)
        & (eps > 0.0)
    )
    physics_sq = _tree_sum_squares(safe_physics)
    teacher_sq = _tree_sum_squares(safe_teacher)
    dot = _tree_dot(safe_physics, safe_teacher)
    coefficient = jnp.minimum(dot, 0.0) / (physics_sq + eps)
    projected = jax.tree_util.tree_map(
        lambda teacher, physics: teacher - coefficient * physics,
        safe_teacher,
        safe_physics,
    )
    projected_sq = _tree_sum_squares(projected)
    physics_norm = jnp.sqrt(physics_sq)
    raw_teacher_norm = jnp.sqrt(teacher_sq)
    projected_norm = jnp.sqrt(projected_sq)
    scale = jnp.where(
        (physics_norm > 0.0) & (projected_norm > 0.0) & valid,
        jnp.minimum(1.0, ratio * physics_norm / (projected_norm + eps)),
        0.0,
    )
    applied = jax.tree_util.tree_map(lambda value: scale * value, projected)
    combined = jax.tree_util.tree_map(
        lambda physics, teacher: physics + teacher,
        safe_physics,
        applied,
    )
    applied_norm = jnp.sqrt(_tree_sum_squares(applied))
    combined_norm = jnp.sqrt(_tree_sum_squares(combined))
    cosine = jnp.where(
        (physics_norm > 0.0) & (raw_teacher_norm > 0.0),
        dot / (physics_norm * raw_teacher_norm + eps),
        0.0,
    )
    valid = (
        valid
        & jnp.isfinite(dot)
        & jnp.isfinite(cosine)
        & jnp.isfinite(projected_norm)
        & jnp.isfinite(applied_norm)
        & jnp.isfinite(combined_norm)
        & (applied_norm <= ratio * physics_norm + 1e-7)
    )
    return TeacherGradientMix(
        combined_gradient=combined,
        projected_teacher_gradient=projected,
        applied_teacher_gradient=applied,
        physics_norm=physics_norm,
        raw_teacher_norm=raw_teacher_norm,
        projected_teacher_norm=projected_norm,
        applied_teacher_norm=applied_norm,
        combined_norm=combined_norm,
        physics_teacher_dot=dot,
        physics_teacher_cosine=cosine,
        applied_scale=scale,
        valid=valid,
    )


def mix_recovery_teacher_actor_gradient(
    physics_gradient: Any,
    actor_params: Any,
    *,
    residual_actor: Any,
    teacher_frames: jax.Array | None,
    parent_action: jax.Array | None,
    teacher_correction: jax.Array | None,
    teacher_effective_action: jax.Array | None,
    max_ratio: float,
) -> RecoveryTeacherGradientResult:
    """Compute the adapter teacher gradient and safely mix it with physics."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, actor_params)
    if teacher_frames is None:
        mix = mix_conflict_projected_teacher_gradient(
            physics_gradient, zeros, max_ratio=0.0
        )
        zero = jnp.asarray(0.0, dtype=mix.physics_norm.dtype)
        return RecoveryTeacherGradientResult(
            loss=zero,
            teacher_gradient=zeros,
            mix=mix,
            parent_gradient_max_abs=zero,
            enabled=jnp.asarray(False),
            valid=mix.valid,
        )
    if any(
        value is None
        for value in (
            parent_action,
            teacher_correction,
            teacher_effective_action,
        )
    ):
        raise ValueError("enabled recovery teacher tensors are incomplete")
    if not hasattr(actor_params, "adapter") or not hasattr(actor_params, "parent"):
        raise ValueError("recovery teacher requires composite residual parameters")

    def teacher_loss_fn(params):
        predicted = residual_actor.apply(params.adapter, teacher_frames)
        return recovery_teacher_imitation_loss(
            predicted,
            parent_action,
            teacher_correction,
            teacher_effective_action,
        )

    loss, teacher_gradient = jax.value_and_grad(teacher_loss_fn)(actor_params)
    mix = mix_conflict_projected_teacher_gradient(
        physics_gradient, teacher_gradient, max_ratio=max_ratio
    )
    parent_leaves = jax.tree_util.tree_leaves(teacher_gradient.parent)
    parent_max = (
        jnp.max(
            jnp.stack(
                [jnp.max(jnp.abs(value)) for value in parent_leaves]
            )
        )
        if parent_leaves
        else jnp.asarray(0.0)
    )
    valid = (
        jnp.isfinite(loss)
        & (parent_max == 0.0)
        & mix.valid
    )
    return RecoveryTeacherGradientResult(
        loss=loss,
        teacher_gradient=teacher_gradient,
        mix=mix,
        parent_gradient_max_abs=parent_max,
        enabled=jnp.asarray(True),
        valid=valid,
    )


def _validate_requested(
    path: str | None, sha256: str | None, ratio: float
) -> tuple[str | None, str | None, float]:
    enabled = path is not None
    if enabled:
        resolved = str(Path(path).resolve())
        if sha256 is None or len(sha256) != 64:
            raise ValueError("recovery teacher SHA-256 is invalid")
        try:
            int(sha256, 16)
        except ValueError as error:
            raise ValueError("recovery teacher SHA-256 is invalid") from error
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("recovery teacher gradient ratio is invalid")
        return resolved, sha256, float(ratio)
    if sha256 is not None or ratio != 0.0:
        raise ValueError("disabled recovery teacher settings are inconsistent")
    return None, None, 0.0


def resolve_recovery_teacher_resume_settings(
    *,
    requested_path: str | None,
    requested_sha256: str | None,
    requested_ratio: float,
    resumed_hparams: Mapping[str, object] | None,
    is_resume: bool,
    allow_change: bool,
) -> tuple[str | None, str | None, float]:
    """Resolve teacher treatment while failing closed on resume drift."""
    requested = _validate_requested(
        requested_path, requested_sha256, requested_ratio
    )
    if not is_resume:
        return requested
    metadata = resumed_hparams or {}
    present = [key in metadata for key in _RESUME_KEYS]
    if any(present) and not all(present):
        raise ValueError("recovery teacher resume metadata is incomplete")
    if not any(present):
        saved = (None, None, 0.0)
        if requested == saved:
            return requested
        if not allow_change:
            raise ValueError("recovery teacher resume metadata is missing")
    else:
        saved_enabled = metadata["actor_recovery_teacher_enabled"]
        if not isinstance(saved_enabled, bool):
            raise ValueError("recovery teacher resume metadata is invalid")
        saved = _validate_requested(
            metadata["actor_recovery_teacher_dataset_path"]
            if saved_enabled
            else None,
            metadata["actor_recovery_teacher_dataset_sha256"]
            if saved_enabled
            else None,
            float(metadata["actor_recovery_teacher_gradient_ratio"]),
        )
    if requested != saved and not allow_change:
        raise ValueError("recovery teacher change requires explicit authority")
    return requested
