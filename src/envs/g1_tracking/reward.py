"""RMR/BeyondMimic rigid-body motion-tracking reward."""

from collections.abc import Mapping

import jax
import jax.numpy as jp


def quaternion_error_magnitude(
    target: jax.Array, actual: jax.Array
) -> jax.Array:
    """Returns the shortest rotation angle between WXYZ quaternions."""
    target = target / jp.maximum(jp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    actual = actual / jp.maximum(jp.linalg.norm(actual, axis=-1, keepdims=True), 1e-8)
    absolute_dot = jp.abs(jp.sum(target * actual, axis=-1))
    return 2.0 * jp.arccos(jp.clip(absolute_dot, 0.0, 1.0))


def _position_error(target: jax.Array, actual: jax.Array) -> jax.Array:
    return jp.sum(jp.square(target - actual), axis=-1)


def _velocity_tracking_kernel(
    normalized_mean_squared_error: jax.Array,
    kernel: str,
) -> jax.Array:
    if kernel == "exponential":
        return jp.exp(-normalized_mean_squared_error)
    if kernel == "pseudo_huber":
        # Matches exp(-x)'s value and first derivative at x=0, while its
        # residual gradient approaches a nonzero constant for large errors.
        return 2.0 - jp.sqrt(1.0 + 2.0 * normalized_mean_squared_error)
    raise ValueError(f"unknown tracking velocity kernel: {kernel}")


def _anchor_position_tracking_kernel(
    squared_error: jax.Array,
    kernel: str,
) -> jax.Array:
    if kernel == "exponential":
        return jp.exp(-squared_error / 0.3**2)
    if kernel == "dual_scale":
        return 0.75 * jp.exp(-squared_error / 0.3**2) + 0.25 * jp.exp(
            -squared_error / 0.8**2
        )
    if kernel == "quadratic":
        # Matches exp(-x) through first order in normalized squared error,
        # while retaining corrective gradient far from the reference.
        return 1.0 - squared_error / 0.3**2
    raise ValueError(f"unknown anchor position kernel: {kernel}")


def torso_orientation_tracking_reward(
    target_body_quat: jax.Array,
    actual_body_quat: jax.Array,
    *,
    torso_body_slot: int = 7,
) -> jax.Array:
    """Return a non-saturating reference-relative torso orientation reward."""
    angle = quaternion_error_magnitude(
        target_body_quat[..., torso_body_slot, :],
        actual_body_quat[..., torso_body_slot, :],
    )
    normalized_squared_error = jp.square(angle) / 0.4**2
    return 2.0 - jp.sqrt(1.0 + 2.0 * normalized_squared_error)


def root_velocity_tracking_reward(
    target_body_lin_vel: jax.Array,
    actual_body_lin_vel: jax.Array,
    target_body_ang_vel: jax.Array,
    actual_body_ang_vel: jax.Array,
    *,
    root_body_slot: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Return non-saturating pelvis linear and angular velocity rewards."""
    linear_error = _position_error(
        target_body_lin_vel[..., root_body_slot, :],
        actual_body_lin_vel[..., root_body_slot, :],
    )
    angular_error = _position_error(
        target_body_ang_vel[..., root_body_slot, :],
        actual_body_ang_vel[..., root_body_slot, :],
    )
    return (
        _velocity_tracking_kernel(linear_error / 1.0**2, "pseudo_huber"),
        _velocity_tracking_kernel(angular_error / jp.pi**2, "pseudo_huber"),
    )


def termination_margin_penalty(
    *,
    anchor_z_error: jax.Array,
    anchor_xy_error: jax.Array,
    gravity_z_error: jax.Array,
    distal_z_error: jax.Array,
) -> jax.Array:
    """Returns a differentiable surrogate for the four hard RMR limits.

    The penalty is zero inside half of every termination threshold, then grows
    quadratically to minus one when any individual threshold is reached and is
    capped beyond it. Its optional environment weight defaults to zero, so the
    exact upstream RMR reward remains the default task.
    """
    thresholds = jp.array([0.25, 1.3, 0.8, 0.4])
    errors = jp.stack(
        (
            anchor_z_error,
            anchor_xy_error,
            gravity_z_error,
            distal_z_error,
        ),
        axis=-1,
    )
    normalized_excess = jp.clip(
        (errors / thresholds - 0.5) / 0.5,
        0.0,
        1.0,
    )
    return -jp.sum(jp.square(normalized_excess), axis=-1)


def rmr_regularization_reward(
    *,
    action: jax.Array,
    previous_action: jax.Array,
    joint_pos: jax.Array,
    soft_joint_lower: jax.Array,
    soft_joint_upper: jax.Array,
    action_magnitude_weight: float = 0.0,
) -> tuple[jax.Array, Mapping[str, jax.Array]]:
    """Computes RMR's differentiable action-rate and joint-limit terms."""
    action_rate = -0.1 * jp.sum(jp.square(action - previous_action), axis=-1)
    action_magnitude = -action_magnitude_weight * jp.sum(
        jp.square(action), axis=-1
    )
    below_limit = jp.maximum(soft_joint_lower - joint_pos, 0.0)
    above_limit = jp.maximum(joint_pos - soft_joint_upper, 0.0)
    # Upstream PhysX trajectories do not exhibit MJX's rare finite solver
    # explosions. Preserve the exact local slope and weight while preventing
    # an invalid transition from dominating critic targets by orders of
    # magnitude.
    total_violation = jp.minimum(
        jp.sum(below_limit + above_limit, axis=-1), 1.0
    )
    joint_limit = -10.0 * total_violation
    components = {
        "action_rate": action_rate,
        "action_magnitude": action_magnitude,
        "joint_limit": joint_limit,
    }
    if action_magnitude_weight == 0.0:
        return action_rate + joint_limit, components
    return action_rate + action_magnitude + joint_limit, components


def rmr_tracking_reward(
    *,
    target_anchor_pos: jax.Array,
    actual_anchor_pos: jax.Array,
    target_anchor_quat: jax.Array,
    actual_anchor_quat: jax.Array,
    target_body_pos: jax.Array,
    actual_body_pos: jax.Array,
    target_body_quat: jax.Array,
    actual_body_quat: jax.Array,
    target_body_lin_vel: jax.Array,
    actual_body_lin_vel: jax.Array,
    target_body_ang_vel: jax.Array,
    actual_body_ang_vel: jax.Array,
    velocity_kernel: str = "exponential",
    anchor_position_kernel: str = "exponential",
) -> tuple[jax.Array, Mapping[str, jax.Array]]:
    """Computes the six default RMR tracking terms and their weighted sum.

    Inputs may be unbatched ``(B, D)`` arrays or carry arbitrary leading batch
    dimensions. Body errors are averaged over the selected RMR body set before
    applying each exponential, matching the upstream task.
    """
    components = {
        "anchor_position": _anchor_position_tracking_kernel(
            _position_error(target_anchor_pos, actual_anchor_pos),
            anchor_position_kernel,
        ),
        "anchor_orientation": jp.exp(
            -jp.square(
                quaternion_error_magnitude(target_anchor_quat, actual_anchor_quat)
            )
            / 0.4**2
        ),
        "body_position": jp.exp(
            -jp.mean(_position_error(target_body_pos, actual_body_pos), axis=-1)
            / 0.3**2
        ),
        "body_orientation": jp.exp(
            -jp.mean(
                jp.square(
                    quaternion_error_magnitude(target_body_quat, actual_body_quat)
                ),
                axis=-1,
            )
            / 0.4**2
        ),
        "body_linear_velocity": _velocity_tracking_kernel(
            jp.mean(
                _position_error(target_body_lin_vel, actual_body_lin_vel), axis=-1
            )
            / 1.0**2,
            velocity_kernel,
        ),
        "body_angular_velocity": _velocity_tracking_kernel(
            jp.mean(
                _position_error(target_body_ang_vel, actual_body_ang_vel), axis=-1
            )
            / 3.14**2,
            velocity_kernel,
        ),
    }
    reward = (
        0.5 * components["anchor_position"]
        + 0.5 * components["anchor_orientation"]
        + components["body_position"]
        + components["body_orientation"]
        + components["body_linear_velocity"]
        + components["body_angular_velocity"]
    )
    return reward, components
