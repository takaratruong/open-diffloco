"""Pinned per-joint action-noise provenance for the RMR G1 controller."""

import jax
import jax.numpy as jp
import numpy as np

RMR_ACTION_STD_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# Exact float32 values from model_4999.pt, SHA-256
# 5174a0f1dc8c83ef9ea45769c3b0f19383e5aeeafea2171433f8e7bb88b21746.
RMR_ACTION_STD = jp.array(
    (
        0.240585923,
        0.241455898,
        0.170482919,
        0.199771196,
        0.210099757,
        0.265766233,
        0.178219095,
        0.179201528,
        0.261124432,
        0.218309745,
        0.227685556,
        0.277082771,
        0.279196233,
        0.233728826,
        0.249337018,
        0.255797625,
        0.247496039,
        0.209423468,
        0.209507123,
        0.185532168,
        0.189358085,
        0.237778917,
        0.234698176,
        0.223405078,
        0.221006036,
        0.358317554,
        0.35941571,
        0.373313069,
        0.34844774,
    ),
    dtype=jp.float32,
)

ActionNoiseStd = float | int | jax.Array


def validate_action_noise_std(
    value, *, action_dim: int, actor_joint_names
) -> ActionNoiseStd:
    """Return a finite nonnegative scalar or exact-order action vector."""
    array = np.asarray(value)
    if array.ndim not in (0, 1):
        raise ValueError("action noise std must be a scalar or a rank-one vector")
    if not np.isfinite(array).all():
        raise ValueError("action noise std must be finite")
    if (array < 0.0).any():
        raise ValueError("action noise std must be non-negative")
    if array.ndim == 0:
        if isinstance(value, jax.Array):
            return value
        return array.item()
    if array.ndim == 1:
        if action_dim != len(RMR_ACTION_STD_JOINT_NAMES):
            raise ValueError("action noise vector action dimension must be 29")
        if array.shape != (len(RMR_ACTION_STD_JOINT_NAMES),):
            raise ValueError(
                "action noise std vector must have shape (29,)"
            )
        if tuple(actor_joint_names) != RMR_ACTION_STD_JOINT_NAMES:
            raise ValueError(
                "action noise vector requires the pinned RMR actor joint order"
            )
    return jp.asarray(value, dtype=jp.float32)


def action_noise_std_hparam(value) -> float | list[float]:
    """Convert action noise to JSON-safe scalar or lossless vector metadata."""
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.ndim == 1:
        return array.tolist()
    raise ValueError("action noise std must be a scalar or a rank-one vector")


def resolve_action_noise_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    is_resume: bool,
    requested_start,
    requested_end,
    allow_change: bool,
    action_dim: int,
    actor_joint_names,
) -> tuple[ActionNoiseStd, ActionNoiseStd]:
    """Restore exact action noise unless its treatment change is authorized."""
    if not isinstance(is_resume, bool):
        raise ValueError("is_resume must be boolean")
    if not isinstance(allow_change, bool):
        raise ValueError("allow_resume_action_noise_change must be boolean")
    requested_start = validate_action_noise_std(
        requested_start,
        action_dim=action_dim,
        actor_joint_names=actor_joint_names,
    )
    requested_end = validate_action_noise_std(
        requested_end,
        action_dim=action_dim,
        actor_joint_names=actor_joint_names,
    )
    if resumed_hparams is None:
        if is_resume and not allow_change:
            raise ValueError("resume metadata requires complete action noise metadata")
        return requested_start, requested_end
    required = {"action_noise_std_start", "action_noise_std_end"}
    if not required.issubset(resumed_hparams):
        if allow_change:
            return requested_start, requested_end
        raise ValueError("resume metadata requires complete action noise metadata")

    saved_start = validate_action_noise_std(
        resumed_hparams["action_noise_std_start"],
        action_dim=action_dim,
        actor_joint_names=actor_joint_names,
    )
    saved_end = validate_action_noise_std(
        resumed_hparams["action_noise_std_end"],
        action_dim=action_dim,
        actor_joint_names=actor_joint_names,
    )
    changed = not (
        np.array_equal(np.asarray(saved_start), np.asarray(requested_start))
        and np.array_equal(np.asarray(saved_end), np.asarray(requested_end))
    )
    if changed and not allow_change:
        raise ValueError(
            "action noise settings must match the checkpoint unless "
            "allow_resume_action_noise_change is enabled"
        )
    if allow_change:
        return requested_start, requested_end
    return saved_start, saved_end
