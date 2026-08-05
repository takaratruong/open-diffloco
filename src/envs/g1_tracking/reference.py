"""Pinned reference-motion loading and MuJoCo rigid-body preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


RMR_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class MujocoReference:
    """Generalized and rigid-body state for every reference frame."""

    qpos: np.ndarray
    qvel: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    body_lin_vel: np.ndarray
    body_ang_vel: np.ndarray
    body_ids: tuple[int, ...]
    body_names: tuple[str, ...]


def _checked_body_ids(
    model: mujoco.MjModel, body_names: Sequence[str]
) -> tuple[int, ...]:
    body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    )
    missing = [name for name, body_id in zip(body_names, body_ids) if body_id < 0]
    if missing:
        raise ValueError(f"reference body names missing from model: {missing}")
    return body_ids


def load_mujoco_reference(
    model: mujoco.MjModel,
    reference_path: str | Path,
    body_names: Sequence[str] = RMR_G1_BODY_NAMES,
) -> MujocoReference:
    """Loads qpos/qvel and precomputes RMR body state through MuJoCo FK."""
    with np.load(reference_path, allow_pickle=False) as archive:
        qpos = np.array(archive["X"], dtype=np.float64, copy=True)
        qvel = np.array(archive["V"], dtype=np.float64, copy=True)

    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(
            f"reference X must have shape (T, {model.nq}), got {qpos.shape}"
        )
    if qvel.shape != (qpos.shape[0], model.nv):
        raise ValueError(
            f"reference V must have shape ({qpos.shape[0]}, {model.nv}), "
            f"got {qvel.shape}"
        )
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("reference qpos/qvel must be finite")
    quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    if np.any(quaternion_norm < 1e-8):
        raise ValueError("reference root quaternion must be nonzero")
    # MJX normalizes free-joint quaternions during forward. Canonicalize once
    # here so RSI qpos, precomputed body state, and the live data agree exactly.
    qpos[:, 3:7] /= quaternion_norm

    body_names = tuple(body_names)
    body_ids = _checked_body_ids(model, body_names)
    num_frames = qpos.shape[0]
    num_bodies = len(body_ids)
    body_pos = np.empty((num_frames, num_bodies, 3), dtype=np.float64)
    body_quat = np.empty((num_frames, num_bodies, 4), dtype=np.float64)
    body_lin_vel = np.empty((num_frames, num_bodies, 3), dtype=np.float64)
    body_ang_vel = np.empty((num_frames, num_bodies, 3), dtype=np.float64)

    data = mujoco.MjData(model)
    jacp = np.empty((3, model.nv), dtype=np.float64)
    jacr = np.empty((3, model.nv), dtype=np.float64)
    for frame in range(num_frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        for slot, body_id in enumerate(body_ids):
            body_pos[frame, slot] = data.xpos[body_id]
            body_quat[frame, slot] = data.xquat[body_id]
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
            body_lin_vel[frame, slot] = jacp @ data.qvel
            body_ang_vel[frame, slot] = jacr @ data.qvel

    arrays = (
        qpos,
        qvel,
        body_pos,
        body_quat,
        body_lin_vel,
        body_ang_vel,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("precomputed reference body state must be finite")
    for array in arrays:
        array.setflags(write=False)

    return MujocoReference(
        qpos=qpos,
        qvel=qvel,
        body_pos=body_pos,
        body_quat=body_quat,
        body_lin_vel=body_lin_vel,
        body_ang_vel=body_ang_vel,
        body_ids=body_ids,
        body_names=body_names,
    )
