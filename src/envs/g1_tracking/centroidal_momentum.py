"""Shared differentiable centroidal-momentum measurements for G1 tracking."""

from __future__ import annotations

import math

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np


def _validate_concrete_finite(name: str, value: jax.Array) -> None:
    try:
        concrete = np.asarray(value)
    except jax.errors.TracerArrayConversionError:
        return
    if not np.isfinite(concrete).all():
        raise ValueError(f"{name} must be finite")


def _unit_quaternion(quaternion: jax.Array) -> jax.Array:
    value = jp.asarray(quaternion)
    if value.shape != (4,):
        raise ValueError("root quaternion must have shape (4,)")
    _validate_concrete_finite("root quaternion", value)
    norm = jp.linalg.norm(value)
    try:
        concrete_norm = float(np.asarray(norm))
    except jax.errors.TracerArrayConversionError:
        concrete_norm = 1.0
    if concrete_norm <= 1e-12:
        raise ValueError("root quaternion must have nonzero norm")
    return value / jp.maximum(norm, 1e-12)


def yaw_frame_vector(
    world_vector: jax.Array, root_quaternion: jax.Array
) -> jax.Array:
    """Rotate one world-frame vector into the scalar-first root-yaw frame."""
    vector = jp.asarray(world_vector)
    if vector.shape != (3,):
        raise ValueError("world vector must have shape (3,)")
    _validate_concrete_finite("world vector", vector)
    w, x, y, z = _unit_quaternion(root_quaternion)
    yaw = jp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cosine = jp.cos(yaw)
    sine = jp.sin(yaw)
    return jp.asarray(
        (
            cosine * vector[0] + sine * vector[1],
            -sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        dtype=vector.dtype,
    )


def yaw_frame_momentum(
    world_momentum: jax.Array, root_quaternion: jax.Array
) -> jax.Array:
    """Rotate linear and angular centroidal momentum into one yaw frame."""
    value = jp.asarray(world_momentum)
    if value.shape != (6,):
        raise ValueError("centroidal momentum must have shape (6,)")
    _validate_concrete_finite("centroidal momentum", value)
    return jp.concatenate(
        (
            yaw_frame_vector(value[:3], root_quaternion),
            yaw_frame_vector(value[3:], root_quaternion),
        )
    )


def capture_point_position(
    com_world: jax.Array,
    linear_momentum_world: jax.Array,
    *,
    total_mass: float,
    gravity_magnitude: float,
) -> jax.Array:
    """Return the planar divergent component of motion in world coordinates."""
    com = jp.asarray(com_world)
    momentum = jp.asarray(linear_momentum_world)
    if com.shape != (3,) or momentum.shape != (3,):
        raise ValueError("COM and linear momentum must each have shape (3,)")
    _validate_concrete_finite("COM", com)
    _validate_concrete_finite("linear momentum", momentum)
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total_mass must be positive and finite")
    if not math.isfinite(gravity_magnitude) or gravity_magnitude <= 0.0:
        raise ValueError("gravity_magnitude must be positive and finite")
    try:
        concrete_height = float(np.asarray(com[2]))
    except jax.errors.TracerArrayConversionError:
        concrete_height = 1.0
    if concrete_height <= 0.0:
        raise ValueError("COM height must be positive")
    omega = jp.sqrt(
        jp.asarray(gravity_magnitude, dtype=com.dtype) / com[2]
    )
    velocity_xy = momentum[:2] / jp.asarray(total_mass, dtype=momentum.dtype)
    return com[:2] + velocity_xy / omega


def mjx_centroidal_momentum(
    model: mjx.Model,
    data: mjx.Data,
    root_body_id: int,
    total_mass: float,
) -> jax.Array:
    """Return `[linear, angular]` subtree momentum from differentiable MJX."""
    if root_body_id < 1 or root_body_id >= model.nbody:
        raise ValueError("root_body_id must identify a non-world body")
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total_mass must be positive and finite")
    measured = mjx.subtree_vel(model, data)
    linear = (
        jp.asarray(total_mass, dtype=data.qvel.dtype)
        * measured._impl.subtree_linvel[root_body_id]
    )
    angular = measured._impl.subtree_angmom[root_body_id]
    return jp.concatenate((linear, angular))


def mjx_capture_point(
    model: mjx.Model,
    data: mjx.Data,
    root_body_id: int,
    total_mass: float,
    gravity_magnitude: float,
) -> jax.Array:
    """Return the differentiable planar capture point for one MJX state."""
    momentum = mjx_centroidal_momentum(
        model, data, root_body_id, total_mass
    )
    return capture_point_position(
        data.subtree_com[root_body_id],
        momentum[:3],
        total_mass=total_mass,
        gravity_magnitude=gravity_magnitude,
    )


def cpu_centroidal_momentum(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    root_body_id: int,
) -> np.ndarray:
    """Return the matching MuJoCo CPU centroidal momentum."""
    if root_body_id < 1 or root_body_id >= model.nbody:
        raise ValueError("root_body_id must identify a non-world body")
    position = np.asarray(qpos, dtype=np.float64)
    velocity = np.asarray(qvel, dtype=np.float64)
    if position.shape != (model.nq,) or velocity.shape != (model.nv,):
        raise ValueError("qpos/qvel shapes do not match the model")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("qpos/qvel must be finite")
    data = mujoco.MjData(model)
    data.qpos[:] = position
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    return np.concatenate(
        (
            model.body_subtreemass[root_body_id]
            * data.subtree_linvel[root_body_id],
            data.subtree_angmom[root_body_id],
        )
    )


def cpu_capture_point(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    root_body_id: int,
) -> np.ndarray:
    """Return the matching MuJoCo CPU planar capture point."""
    position = np.asarray(qpos, dtype=np.float64)
    velocity = np.asarray(qvel, dtype=np.float64)
    if position.shape != (model.nq,) or velocity.shape != (model.nv,):
        raise ValueError("qpos/qvel shapes do not match the model")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("qpos/qvel must be finite")
    if root_body_id < 1 or root_body_id >= model.nbody:
        raise ValueError("root_body_id must identify a non-world body")
    data = mujoco.MjData(model)
    data.qpos[:] = position
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    total_mass = float(model.body_subtreemass[root_body_id])
    momentum = total_mass * data.subtree_linvel[root_body_id]
    gravity_magnitude = float(np.linalg.norm(model.opt.gravity))
    return np.asarray(
        capture_point_position(
            data.subtree_com[root_body_id],
            momentum,
            total_mass=total_mass,
            gravity_magnitude=gravity_magnitude,
        )
    )


def reference_centroidal_momentum(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    root_body_id: int,
) -> np.ndarray:
    """Precompute one CPU centroidal-momentum row per reference frame."""
    positions = np.asarray(qpos, dtype=np.float64)
    velocities = np.asarray(qvel, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != model.nq
        or velocities.shape != (positions.shape[0], model.nv)
        or positions.shape[0] < 1
    ):
        raise ValueError("reference qpos/qvel arrays do not align")
    return np.stack(
        tuple(
            cpu_centroidal_momentum(model, position, velocity, root_body_id)
            for position, velocity in zip(positions, velocities)
        ),
        axis=0,
    )


def standing_com_height(
    model: mujoco.MjModel, qpos: np.ndarray, root_body_id: int
) -> float:
    """Return the pinned standing subtree-COM height above world ground."""
    position = np.asarray(qpos, dtype=np.float64)
    if position.shape != (model.nq,) or not np.isfinite(position).all():
        raise ValueError("standing qpos must be a finite model configuration")
    if root_body_id < 1 or root_body_id >= model.nbody:
        raise ValueError("root_body_id must identify a non-world body")
    data = mujoco.MjData(model)
    data.qpos[:] = position
    mujoco.mj_forward(model, data)
    height = float(data.subtree_com[root_body_id, 2])
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("standing COM height must be positive and finite")
    return height
