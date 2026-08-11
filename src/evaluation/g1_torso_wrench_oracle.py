"""Pure bounded torso-wrench control for frozen G1 evaluation."""

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jp
import mujoco


TORSO_BODY_NAME = "torso_link"
ORACLE_FREQUENCY_HZ = 2.0
ORACLE_LEVER_ARM_METERS = 0.3
ORACLE_TORQUE_CAP_FRACTION = 0.3


@dataclass(frozen=True)
class TorsoWrenchParameters:
    """Fixed physical parameters for the six-dimensional analytic oracle."""

    nominal_total_mass: float
    gravity_magnitude: float
    frequency_hz: float = ORACLE_FREQUENCY_HZ
    lever_arm_meters: float = ORACLE_LEVER_ARM_METERS
    torque_cap_fraction: float = ORACLE_TORQUE_CAP_FRACTION

    def __post_init__(self) -> None:
        values = (
            self.nominal_total_mass,
            self.gravity_magnitude,
            self.frequency_hz,
            self.lever_arm_meters,
            self.torque_cap_fraction,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("torso-wrench parameters must be finite and positive")

    @property
    def omega(self) -> float:
        """Critical-damping angular frequency in radians per second."""
        return 2.0 * math.pi * self.frequency_hz

    @property
    def translational_kp(self) -> float:
        """Translational spring gain for the nominal total mass."""
        return self.nominal_total_mass * self.omega**2

    @property
    def translational_kd(self) -> float:
        """Translational critical-damping gain for the nominal total mass."""
        return 2.0 * self.nominal_total_mass * self.omega

    @property
    def rotational_inertia(self) -> float:
        """Point-mass rotational approximation used by the frozen oracle."""
        return self.nominal_total_mass * self.lever_arm_meters**2

    @property
    def rotational_kp(self) -> float:
        """Rotational spring gain for the effective torso inertia."""
        return self.rotational_inertia * self.omega**2

    @property
    def rotational_kd(self) -> float:
        """Rotational critical-damping gain for the effective torso inertia."""
        return 2.0 * self.rotational_inertia * self.omega

    @property
    def body_weight(self) -> float:
        """Nominal robot body weight in newtons."""
        return self.nominal_total_mass * self.gravity_magnitude

    @property
    def force_cap(self) -> float:
        """One-body-weight force norm cap in newtons."""
        return self.body_weight

    @property
    def torque_cap(self) -> float:
        """Frozen torso torque norm cap in newton metres."""
        return (
            self.body_weight
            * self.lever_arm_meters
            * self.torque_cap_fraction
        )


def resolve_torso_body_id(model: mujoco.MjModel) -> int:
    """Resolve the required torso body by its stable model name."""
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME
    )
    if body_id < 0:
        raise ValueError(f"model is missing required body {TORSO_BODY_NAME!r}")
    return int(body_id)


def torso_wrench_parameters_from_environment(
    environment: Any,
) -> tuple[int, TorsoWrenchParameters]:
    """Read nominal physical values and the named torso body from an environment."""
    try:
        nominal_total_mass = float(environment.nominal_total_mass)
        gravity_magnitude = float(environment.base_gravity_mag)
        model = environment.mj_model
    except AttributeError as error:
        raise ValueError(
            "environment must provide nominal_total_mass, base_gravity_mag, "
            "and mj_model"
        ) from error
    return (
        resolve_torso_body_id(model),
        TorsoWrenchParameters(
            nominal_total_mass=nominal_total_mass,
            gravity_magnitude=gravity_magnitude,
        ),
    )


def shortest_quaternion_rotation_vector(
    *, target_quaternion: jax.Array, actual_quaternion: jax.Array
) -> jax.Array:
    """Return the shortest WXYZ quaternion error as a world-frame rotvec."""
    error = _quaternion_multiply(
        _unit_quaternion(target_quaternion),
        _quaternion_conjugate(_unit_quaternion(actual_quaternion)),
    )
    error = jp.where(error[0] < 0.0, -error, error)
    sine_half_angle = _safe_norm(error[1:])
    angle = 2.0 * jp.arctan2(sine_half_angle, error[0])
    denominator = jp.where(sine_half_angle > 1e-8, sine_half_angle, 1.0)
    return _finite_vector(angle * error[1:] / denominator)


def compute_torso_wrench(
    *,
    parameters: TorsoWrenchParameters,
    actual_position: jax.Array,
    actual_quaternion: jax.Array,
    actual_linear_velocity: jax.Array,
    actual_angular_velocity: jax.Array,
    reference_position: jax.Array,
    reference_quaternion: jax.Array,
    reference_linear_velocity: jax.Array,
    reference_angular_velocity: jax.Array,
    scale: float | jax.Array = 1.0,
) -> jax.Array:
    """Compute one capped world-frame force/torque wrench without side effects."""
    yaw_quaternion = _yaw_quaternion(actual_quaternion)
    yaw_inverse = _quaternion_conjugate(yaw_quaternion)
    position_error, position_scale = _scale_stable_quaternion_difference(
        yaw_inverse,
        reference_position,
        actual_position,
    )
    linear_velocity_error, linear_velocity_scale = (
        _scale_stable_quaternion_difference(
            yaw_inverse,
            reference_linear_velocity,
            actual_linear_velocity,
        )
    )
    orientation_error, orientation_scale = _scale_stable_quaternion_apply(
        yaw_inverse,
        shortest_quaternion_rotation_vector(
            target_quaternion=reference_quaternion,
            actual_quaternion=actual_quaternion,
        ),
    )
    angular_velocity_error, angular_velocity_scale = (
        _scale_stable_quaternion_difference(
            yaw_inverse,
            reference_angular_velocity,
            actual_angular_velocity,
        )
    )
    force_yaw = _bounded_pd_response(
        proportional_error=position_error,
        proportional_scale=position_scale,
        derivative_error=linear_velocity_error,
        derivative_scale=linear_velocity_scale,
        proportional_gain=parameters.translational_kp,
        derivative_gain=parameters.translational_kd,
        maximum_norm=parameters.force_cap,
    )
    torque_yaw = _bounded_pd_response(
        proportional_error=orientation_error,
        proportional_scale=orientation_scale,
        derivative_error=angular_velocity_error,
        derivative_scale=angular_velocity_scale,
        proportional_gain=parameters.rotational_kp,
        derivative_gain=parameters.rotational_kd,
        maximum_norm=parameters.torque_cap,
    )
    force_world = _quaternion_apply(yaw_quaternion, force_yaw)
    torque_world = _quaternion_apply(yaw_quaternion, torque_yaw)
    bounded = jp.concatenate((force_world, torque_world))
    bounded_scale = jp.clip(_finite_vector(jp.asarray(scale)), 0.0, 1.0)
    zero = jp.zeros(6, dtype=bounded.dtype)
    return jp.where(bounded_scale == 0.0, zero, bounded * bounded_scale)


def compute_environment_torso_wrench(
    environment: Any,
    state: Any,
    *,
    torso_slot: int,
    parameters: TorsoWrenchParameters,
    scale: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the aligned torso wrench and return its actual velocities."""
    positions, quaternions, linear_velocities, angular_velocities = (
        environment._body_state(state.data)
    )
    phase = state.info["phase"]
    reference_positions, reference_quaternions = (
        environment._aligned_reference_body_targets(
            positions[0], quaternions[0], phase
        )
    )
    reference_anchor = environment.body_quat_reference[phase, 0]
    yaw_delta = _yaw_quaternion(
        _quaternion_multiply(
            quaternions[0], _quaternion_conjugate(reference_anchor)
        )
    )
    reference_linear_velocity = _quaternion_apply(
        yaw_delta, environment.body_lin_vel_reference[phase, torso_slot]
    )
    reference_angular_velocity = _quaternion_apply(
        yaw_delta, environment.body_ang_vel_reference[phase, torso_slot]
    )
    wrench = compute_torso_wrench(
        parameters=parameters,
        actual_position=positions[torso_slot],
        actual_quaternion=quaternions[torso_slot],
        actual_linear_velocity=linear_velocities[torso_slot],
        actual_angular_velocity=angular_velocities[torso_slot],
        reference_position=reference_positions[torso_slot],
        reference_quaternion=reference_quaternions[torso_slot],
        reference_linear_velocity=reference_linear_velocity,
        reference_angular_velocity=reference_angular_velocity,
        scale=scale,
    )
    return (
        wrench,
        linear_velocities[torso_slot],
        angular_velocities[torso_slot],
    )


def write_torso_wrench(
    xfrc_applied: jax.Array,
    *,
    torso_body_id: int,
    world_wrench: jax.Array,
) -> jax.Array:
    """Overwrite the torso's MuJoCo force/torque row for one policy step."""
    applied = jp.asarray(xfrc_applied)
    wrench = _finite_vector(jp.asarray(world_wrench))
    if applied.ndim != 2 or applied.shape[1] != 6:
        raise ValueError("xfrc_applied must have shape (body_count, 6)")
    if wrench.shape != (6,):
        raise ValueError("world_wrench must have shape (6,)")
    if torso_body_id < 0 or torso_body_id >= applied.shape[0]:
        raise ValueError("torso_body_id lies outside xfrc_applied")
    return applied.at[torso_body_id].set(wrench)


def _finite_vector(vector: jax.Array) -> jax.Array:
    return jp.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_norm(vector: jax.Array) -> jax.Array:
    finite_vector = _finite_vector(vector)
    maximum = jp.max(jp.abs(finite_vector))
    denominator = jp.where(maximum > 0.0, maximum, 1.0)
    return maximum * jp.sqrt(jp.sum(jp.square(finite_vector / denominator)))


def _bounded_pd_response(
    *,
    proportional_error: jax.Array,
    proportional_scale: jax.Array,
    derivative_error: jax.Array,
    derivative_scale: jax.Array,
    proportional_gain: float,
    derivative_gain: float,
    maximum_norm: float,
) -> jax.Array:
    """Compute a capped PD response without forming an overflowing demand."""
    proportional_error = _finite_vector(proportional_error)
    derivative_error = _finite_vector(derivative_error)
    error_scale = jp.maximum(
        proportional_scale,
        derivative_scale,
    )
    denominator = jp.where(error_scale > 0.0, error_scale, 1.0)
    normalized_demand = (
        proportional_gain * proportional_error * (proportional_scale / denominator)
        + derivative_gain * derivative_error * (derivative_scale / denominator)
    )
    demand_norm = _safe_norm(normalized_demand)
    direction = normalized_demand / jp.where(demand_norm > 0.0, demand_norm, 1.0)
    response_norm = jp.minimum(maximum_norm, error_scale * demand_norm)
    return _finite_vector(direction * response_norm)


def _scale_stable_quaternion_difference(
    quaternion: jax.Array,
    reference: jax.Array,
    actual: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Rotate a finite difference without constructing an overflowing vector."""
    reference = _finite_vector(reference)
    actual = _finite_vector(actual)
    scale = jp.maximum(jp.max(jp.abs(reference)), jp.max(jp.abs(actual)))
    denominator = jp.where(scale > 0.0, scale, 1.0)
    normalized_difference = reference / denominator - actual / denominator
    return _quaternion_apply(quaternion, normalized_difference), scale


def _scale_stable_quaternion_apply(
    quaternion: jax.Array, vector: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rotate a finite vector while keeping its magnitude as a separate scale."""
    vector = _finite_vector(vector)
    scale = jp.max(jp.abs(vector))
    denominator = jp.where(scale > 0.0, scale, 1.0)
    return _quaternion_apply(quaternion, vector / denominator), scale


def _unit_quaternion(quaternion: jax.Array) -> jax.Array:
    finite_quaternion = _finite_vector(jp.asarray(quaternion))
    norm = _safe_norm(finite_quaternion)
    normalized = finite_quaternion / jp.where(norm > 1e-8, norm, 1.0)
    identity = jp.array([1.0, 0.0, 0.0, 0.0], dtype=normalized.dtype)
    return jp.where(norm > 1e-8, normalized, identity)


def _quaternion_conjugate(quaternion: jax.Array) -> jax.Array:
    return quaternion * jp.array([1.0, -1.0, -1.0, -1.0])


def _quaternion_multiply(first: jax.Array, second: jax.Array) -> jax.Array:
    first_w, first_x, first_y, first_z = first
    second_w, second_x, second_y, second_z = second
    return jp.array(
        (
            first_w * second_w
            - first_x * second_x
            - first_y * second_y
            - first_z * second_z,
            first_w * second_x
            + first_x * second_w
            + first_y * second_z
            - first_z * second_y,
            first_w * second_y
            - first_x * second_z
            + first_y * second_w
            + first_z * second_x,
            first_w * second_z
            + first_x * second_y
            - first_y * second_x
            + first_z * second_w,
        )
    )


def _quaternion_apply(quaternion: jax.Array, vector: jax.Array) -> jax.Array:
    quaternion_vector = quaternion[1:]
    scalar = quaternion[0]
    return (
        2.0 * jp.dot(quaternion_vector, vector) * quaternion_vector
        + (scalar**2 - jp.dot(quaternion_vector, quaternion_vector)) * vector
        + 2.0 * scalar * jp.cross(quaternion_vector, vector)
    )


def _yaw_quaternion(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = _unit_quaternion(quaternion)
    yaw = jp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    half_yaw = 0.5 * yaw
    return jp.array([jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)])
