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
) -> tuple[jax.Array, Mapping[str, jax.Array]]:
    """Computes the six default RMR tracking terms and their weighted sum.

    Inputs may be unbatched ``(B, D)`` arrays or carry arbitrary leading batch
    dimensions. Body errors are averaged over the selected RMR body set before
    applying each exponential, matching the upstream task.
    """
    components = {
        "anchor_position": jp.exp(
            -_position_error(target_anchor_pos, actual_anchor_pos) / 0.3**2
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
        "body_linear_velocity": jp.exp(
            -jp.mean(
                _position_error(target_body_lin_vel, actual_body_lin_vel), axis=-1
            )
            / 1.0**2
        ),
        "body_angular_velocity": jp.exp(
            -jp.mean(
                _position_error(target_body_ang_vel, actual_body_ang_vel), axis=-1
            )
            / 3.14**2
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
