"""Zero-effect policy head for bounded training-only torso wrenches."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jp

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    build_residual_adapter_mask,
)


PyTree = Any


class FrozenControllerWrenchParams(NamedTuple):
    """An immutable joint controller plus a trainable torso-wrench head."""

    controller: PyTree
    wrench: PyTree


class LearnedTorsoWrenchHead(nn.Module):
    """Predict one normalized yaw-frame force/torque from a policy frame."""

    hidden_dim: int = 256
    condition_on_scale: bool = False

    @nn.compact
    def __call__(
        self,
        frame: jax.Array,
        assistance_scale: jax.Array | None = None,
    ) -> jax.Array:
        if self.condition_on_scale:
            if assistance_scale is None:
                raise ValueError("conditioned wrench head requires assistance scale")
            scale = jp.asarray(assistance_scale, dtype=frame.dtype)
            if scale.shape not in ((), frame.shape[:-1]):
                raise ValueError("assistance scale shape must match wrench frames")
            scale = jp.broadcast_to(scale, frame.shape[:-1])
            valid = jp.isfinite(scale) & (scale >= 0.0) & (scale <= 1.0)
            scale = jp.where(valid, scale, jp.asarray(jp.nan, frame.dtype))
            frame = jp.concatenate((frame, scale[..., None]), axis=-1)
        elif assistance_scale is not None:
            raise ValueError("legacy wrench head cannot consume assistance scale")
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
    assistance_scale: jax.Array | None = None,
) -> jax.Array:
    """Apply only the trainable normalized wrench head."""
    if not isinstance(params, FrozenControllerWrenchParams):
        raise ValueError("learned wrench parameters require controller and wrench trees")
    if head.condition_on_scale:
        if assistance_scale is None:
            raise ValueError("conditioned wrench head requires assistance scale")
        scale = jp.asarray(assistance_scale, dtype=frame.dtype)
        if scale.shape not in ((), frame.shape[:-1]):
            raise ValueError("assistance scale shape must match wrench frames")
        scale = jp.broadcast_to(scale, frame.shape[:-1])
        valid = jp.isfinite(scale) & (scale >= 0.0) & (scale <= 1.0)
        scale = jp.where(valid, scale, jp.asarray(jp.nan, frame.dtype))
        layers = params.wrench["params"]
        dense0 = layers["Dense_0"]
        dense1 = layers["Dense_1"]
        if dense0["kernel"].shape[0] != frame.shape[-1] + 1:
            raise ValueError("conditioned wrench head input width is invalid")
        # Keep the legacy matmul bit-exact and add only the new scalar row.
        hidden = jp.matmul(frame, dense0["kernel"][:-1]) + dense0["bias"]
        hidden = hidden + scale[..., None] * dense0["kernel"][-1]
        hidden = nn.elu(hidden)
        values = jp.tanh(jp.matmul(hidden, dense1["kernel"]) + dense1["bias"])
    else:
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


def build_learned_wrench_anneal_mask(
    params: FrozenControllerWrenchParams,
) -> FrozenControllerWrenchParams:
    """Train the joint residual adapter and wrench head, never their parent."""
    if not isinstance(params, FrozenControllerWrenchParams):
        raise ValueError("learned wrench parameters require controller and wrench trees")
    if not isinstance(params.controller, FrozenPreviewResidualParams):
        raise ValueError("annealing requires a frozen-preview residual controller")
    return FrozenControllerWrenchParams(
        controller=build_residual_adapter_mask(params.controller),
        wrench=jax.tree_util.tree_map(
            lambda value: jp.ones(value.shape, dtype=bool), params.wrench
        ),
    )


def migrate_learned_wrench_scale_conditioning(
    params: FrozenControllerWrenchParams,
) -> FrozenControllerWrenchParams:
    """Append an exact-zero scalar row to a legacy wrench head input."""
    if not isinstance(params, FrozenControllerWrenchParams):
        raise ValueError("learned wrench parameters require controller and wrench trees")
    if not isinstance(params.wrench, (dict, FrozenDict)):
        raise ValueError("learned wrench parameters must be a mapping")
    mutable = unfreeze(params.wrench)
    try:
        kernel = mutable["params"]["Dense_0"]["kernel"]
    except (KeyError, TypeError) as error:
        raise ValueError("learned wrench head has an invalid first layer") from error
    if kernel.ndim != 2:
        raise ValueError("learned wrench first-layer kernel must be rank two")
    mutable["params"]["Dense_0"]["kernel"] = jp.concatenate(
        (kernel, jp.zeros((1, kernel.shape[1]), dtype=kernel.dtype)), axis=0
    )
    migrated = freeze(mutable) if isinstance(params.wrench, FrozenDict) else mutable
    return FrozenControllerWrenchParams(
        controller=params.controller,
        wrench=migrated,
    )


def learned_wrench_scale_at_step(
    step: jax.Array | int,
    *,
    start_step: int,
    end_step: int,
    start_scale: float,
    end_scale: float,
) -> jax.Array:
    """Return a clipped continuous wrench-cap schedule with exact endpoints."""
    if end_step <= start_step:
        raise ValueError("wrench schedule end_step must exceed start_step")
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (start_scale, end_scale)
    ):
        raise ValueError("wrench schedule scales must be finite and in [0, 1]")
    value = jp.asarray(step, dtype=jp.float32)
    progress = jp.clip(
        (value - float(start_step)) / float(end_step - start_step),
        0.0,
        1.0,
    )
    interpolated = start_scale + progress * (end_scale - start_scale)
    return jp.where(
        value <= start_step,
        jp.asarray(start_scale, dtype=jp.float32),
        jp.where(
            value >= end_step,
            jp.asarray(end_scale, dtype=jp.float32),
            interpolated,
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
