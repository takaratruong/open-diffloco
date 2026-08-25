"""Zero-effect policy head for bounded training-only torso wrenches."""

from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jp


PyTree = Any


class FrozenControllerWrenchParams(NamedTuple):
    """An immutable joint controller plus a trainable torso-wrench head."""

    controller: PyTree
    wrench: PyTree


class LearnedTorsoWrenchHead(nn.Module):
    """Predict one normalized yaw-frame force/torque from a policy frame."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, frame: jax.Array) -> jax.Array:
        hidden = nn.elu(nn.Dense(self.hidden_dim)(frame))
        logits = nn.Dense(
            6,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(hidden)
        return jp.tanh(logits)


def apply_learned_torso_wrench(
    head: LearnedTorsoWrenchHead,
    params: FrozenControllerWrenchParams,
    frame: jax.Array,
) -> jax.Array:
    """Apply only the trainable normalized wrench head."""
    if not isinstance(params, FrozenControllerWrenchParams):
        raise ValueError("learned wrench parameters require controller and wrench trees")
    values = head.apply(params.wrench, frame)
    if values.shape != frame.shape[:-1] + (6,):
        raise ValueError("learned torso wrench head must return six values")
    return values


def build_learned_wrench_mask(
    params: FrozenControllerWrenchParams,
) -> FrozenControllerWrenchParams:
    """Select every wrench-head scalar and no controller scalar."""
    if not isinstance(params, FrozenControllerWrenchParams):
        raise ValueError("learned wrench parameters require controller and wrench trees")
    return FrozenControllerWrenchParams(
        controller=jax.tree_util.tree_map(
            lambda value: jp.zeros(value.shape, dtype=bool), params.controller
        ),
        wrench=jax.tree_util.tree_map(
            lambda value: jp.ones(value.shape, dtype=bool), params.wrench
        ),
    )


def normalized_yaw_wrench_to_world(
    normalized_wrench: jax.Array,
    *,
    root_quaternion: jax.Array,
    force_cap: float | jax.Array,
    torque_cap: float | jax.Array,
    scale: float | jax.Array,
) -> jax.Array:
    """Bound a normalized yaw-frame wrench and rotate it into world axes."""
    values = jp.asarray(normalized_wrench)
    quaternion = jp.asarray(root_quaternion, dtype=values.dtype)
    if values.shape != (6,):
        raise ValueError("normalized_wrench must have shape (6,)")
    if quaternion.shape != (4,):
        raise ValueError("root_quaternion must have shape (4,)")

    force_cap = jp.asarray(force_cap, dtype=values.dtype)
    torque_cap = jp.asarray(torque_cap, dtype=values.dtype)
    scale = jp.asarray(scale, dtype=values.dtype)
    valid = (
        jp.all(jp.isfinite(values))
        & jp.all(jp.isfinite(quaternion))
        & jp.isfinite(force_cap)
        & jp.isfinite(torque_cap)
        & jp.isfinite(scale)
        & (force_cap > 0.0)
        & (torque_cap > 0.0)
        & (scale >= 0.0)
        & (scale <= 1.0)
    )

    force_yaw = _norm_bounded(values[:3], force_cap)
    torque_yaw = _norm_bounded(values[3:], torque_cap)
    yaw = _yaw_quaternion(quaternion)
    world = jp.concatenate(
        (_quaternion_apply(yaw, force_yaw), _quaternion_apply(yaw, torque_yaw))
    )
    scaled = world * scale
    failed = jp.full((6,), jp.nan, dtype=values.dtype)
    return jp.where(scale == 0.0, jp.zeros_like(scaled), jp.where(valid, scaled, failed))


def _norm_bounded(vector: jax.Array, maximum: jax.Array) -> jax.Array:
    vector = vector * maximum
    # sqrt(sum(x**2)) has an undefined derivative at the zero-output
    # initialization.  The tiny squared floor preserves the identity Jacobian
    # there while remaining negligible relative to physical wrench caps.
    norm = jp.sqrt(jp.sum(jp.square(vector)) + jp.asarray(1e-12, vector.dtype))
    factor = jp.minimum(jp.asarray(1.0, vector.dtype), maximum / jp.maximum(norm, 1e-12))
    return vector * factor


def _yaw_quaternion(quaternion: jax.Array) -> jax.Array:
    quaternion = quaternion / jp.maximum(jp.linalg.norm(quaternion), 1e-12)
    w, x, y, z = quaternion
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return jp.asarray([jp.cos(half), 0.0, 0.0, jp.sin(half)], dtype=quaternion.dtype)


def _quaternion_apply(quaternion: jax.Array, vector: jax.Array) -> jax.Array:
    scalar = quaternion[0]
    axis = quaternion[1:]
    return (
        2.0 * jp.dot(axis, vector) * axis
        + (scalar * scalar - jp.dot(axis, axis)) * vector
        + 2.0 * scalar * jp.cross(axis, vector)
    )
