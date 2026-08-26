"""Differentiable G1 floating-base propulsion diagnostics."""

from __future__ import annotations

import math

import jax
import jax.numpy as jp
import numpy as np

from src.envs.g1_tracking.centroidal_momentum import yaw_frame_vector


def _validate_host_array(name: str, value: jax.Array) -> None:
    """Validate concrete values while remaining compatible with JAX tracing."""
    try:
        concrete = np.asarray(value)
    except jax.errors.TracerArrayConversionError:
        return
    if not np.isfinite(concrete).all():
        raise ValueError(f"{name} must be finite")


def constraint_propulsion_sample(
    *,
    qfrc_constraint: jax.Array,
    root_quaternion: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Return yaw-frame floating-base constraint force and interval impulse."""
    generalized_force = jp.asarray(qfrc_constraint)
    if generalized_force.ndim != 1 or generalized_force.shape[0] < 3:
        raise ValueError("constraint force must be a vector with at least 3 rows")
    _validate_host_array("constraint force", generalized_force)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    force = yaw_frame_vector(generalized_force[:3], root_quaternion)
    return force, force * jp.asarray(dt, dtype=force.dtype)


def reference_required_force(
    *,
    reference_root_velocity: jax.Array,
    phase: int | jax.Array,
    stride: int,
    dt: float,
    total_mass: float,
    root_quaternion: jax.Array,
) -> jax.Array:
    """Return reference pelvis-acceleration force in the current yaw frame."""
    velocities = jp.asarray(reference_root_velocity)
    if velocities.ndim != 2 or velocities.shape[1] != 3 or velocities.shape[0] < 1:
        raise ValueError("reference root velocity must have shape (frames, 3)")
    _validate_host_array("reference root velocity", velocities)
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be finite and positive")
    current = jp.asarray(phase, dtype=jp.int32)
    next_phase = jp.minimum(current + stride, velocities.shape[0] - 1)
    acceleration_world = (velocities[next_phase] - velocities[current]) / dt
    return total_mass * yaw_frame_vector(acceleration_world, root_quaternion)


def summarize_propulsion(
    actual_forward: np.ndarray, required_forward: np.ndarray
) -> dict[str, float]:
    """Summarize aligned host-side forward-force evidence."""
    actual = np.asarray(actual_forward, dtype=np.float64)
    required = np.asarray(required_forward, dtype=np.float64)
    if (
        actual.ndim != 1
        or required.shape != actual.shape
        or actual.size < 1
        or not np.isfinite(actual).all()
        or not np.isfinite(required).all()
    ):
        raise ValueError("propulsion rows must be aligned finite vectors")
    return {
        "propulsion_forward_error_rms": float(
            np.sqrt(np.mean(np.square(actual - required)))
        ),
        "propulsion_forward_force_peak_abs": float(np.max(np.abs(actual))),
    }
