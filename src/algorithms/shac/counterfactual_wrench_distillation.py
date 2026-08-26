"""Leg-only credit from a frozen assisted counterfactual transition."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jp
import numpy as np

from src.core.rmr_action_noise import RMR_ACTION_STD_JOINT_NAMES


LEG_ACTION_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

_BLOCK_NAMES = (
    "base_linear",
    "base_angular",
    "centroidal_linear",
    "centroidal_angular",
)


def resolve_leg_action_indices(actor_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Resolve the twelve legs in the exact registered 29-action order."""
    names = tuple(map(str, actor_joint_names))
    if len(names) != 29:
        raise ValueError("leg residual requires 29 canonical actor joints")
    if len(set(names)) != len(names):
        raise ValueError("actor joint names must be unique")
    if names != RMR_ACTION_STD_JOINT_NAMES:
        raise ValueError("actor joints must use the canonical order")
    indices = tuple(names.index(name) for name in LEG_ACTION_NAMES)
    if len(set(indices)) != 12:
        raise ValueError("leg residual indices must be unique")
    return indices


def scatter_leg_residual(
    residual: jax.Array,
    indices: Sequence[int],
    *,
    action_dim: int = 29,
) -> jax.Array:
    """Scatter twelve leg corrections into an otherwise exact-zero action."""
    values = jp.asarray(residual)
    static_indices = tuple(int(index) for index in indices)
    if values.ndim < 1 or values.shape[-1] != 12 or len(static_indices) != 12:
        raise ValueError("leg residual requires exactly twelve actions")
    if (
        action_dim != 29
        or len(set(static_indices)) != 12
        or min(static_indices) < 0
        or max(static_indices) >= action_dim
    ):
        raise ValueError("leg residual indices are invalid")
    values = _finite_or_nan(values, "leg residual values must be finite")
    output = jp.zeros(values.shape[:-1] + (action_dim,), dtype=values.dtype)
    return output.at[..., jp.asarray(static_indices, dtype=jp.int32)].set(values)


def counterfactual_target_change(
    before: jax.Array,
    after: jax.Array,
) -> jax.Array:
    """Return one finite 12-D local dynamics change."""
    before_values = jp.asarray(before)
    after_values = jp.asarray(after, dtype=before_values.dtype)
    if before_values.shape != (12,) or after_values.shape != (12,):
        raise ValueError("counterfactual change requires two 12-vectors")
    change = after_values - before_values
    return _finite_or_nan(change, "counterfactual change must be finite")


def counterfactual_transition_loss(
    student_change: jax.Array,
    teacher_change: jax.Array,
    target_rms: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare student and frozen-teacher local dynamics in four blocks."""
    student = jp.asarray(student_change)
    teacher = jp.asarray(teacher_change, dtype=student.dtype)
    rms = jp.asarray(target_rms, dtype=student.dtype)
    if student.shape != (12,) or teacher.shape != (12,) or rms.shape != (12,):
        raise ValueError("counterfactual loss requires three 12-vectors")
    valid = (
        jp.all(jp.isfinite(student))
        & jp.all(jp.isfinite(teacher))
        & jp.all(jp.isfinite(rms))
        & jp.all(rms >= 0.0)
    )
    safe_rms = jp.maximum(rms, jp.asarray(1e-3, dtype=student.dtype))
    normalized_error = (student - teacher) / safe_rms
    delta = jp.asarray(0.1, dtype=student.dtype)
    element_loss = jp.square(delta) * (
        jp.sqrt(1.0 + jp.square(normalized_error / delta)) - 1.0
    )
    block_losses = jp.mean(element_loss.reshape(4, 3), axis=-1)
    loss = jp.mean(block_losses)
    student_norm = jp.linalg.norm(student)
    teacher_norm = jp.linalg.norm(teacher)
    cosine = jp.vdot(student, teacher) / jp.maximum(
        student_norm * teacher_norm,
        jp.asarray(1e-12, dtype=student.dtype),
    )
    telemetry = {
        f"{name}_loss": block_losses[index]
        for index, name in enumerate(_BLOCK_NAMES)
    }
    telemetry.update(
        cosine=cosine,
        student_rms=jp.sqrt(jp.mean(jp.square(student))),
        teacher_rms=jp.sqrt(jp.mean(jp.square(teacher))),
        normalized_error_rms=jp.sqrt(jp.mean(jp.square(normalized_error))),
        valid=valid.astype(student.dtype),
    )
    failed = jp.asarray(jp.nan, dtype=student.dtype)
    return jp.where(valid, loss, failed), {
        name: jp.where(valid, value, failed) for name, value in telemetry.items()
    }


def _finite_or_nan(values: jax.Array, message: str) -> jax.Array:
    """Raise eagerly and propagate NaN under JIT for fail-closed execution."""
    if not isinstance(values, jax.core.Tracer):
        if not bool(np.all(np.isfinite(np.asarray(values)))):
            raise ValueError(message)
        return values
    valid = jp.all(jp.isfinite(values))
    return jp.where(valid, values, jp.full_like(values, jp.nan))
