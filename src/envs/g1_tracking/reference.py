"""Pinned reference-motion loading and MuJoCo rigid-body preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from src.envs.g1_tracking.controller import RMRController


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
    fps: float | None


_RMR_REFERENCE_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "root_body_name",
    "root_body_index",
}


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


def _root_generalized_velocity(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    root_body_id: int,
    linear_velocity_world: np.ndarray,
    angular_velocity_world: np.ndarray,
) -> np.ndarray:
    """Recovers free-joint qvel from the root body's logged world velocity."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    jacp = np.empty((3, model.nv), dtype=np.float64)
    jacr = np.empty((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, root_body_id)
    root_jacobian = np.concatenate((jacp[:, :6], jacr[:, :6]), axis=0)
    target_velocity = np.concatenate(
        (linear_velocity_world, angular_velocity_world)
    )
    try:
        return np.linalg.solve(root_jacobian, target_velocity)
    except np.linalg.LinAlgError as error:
        raise ValueError("root free-joint velocity Jacobian is singular") from error


def _load_named_rmr_state(
    model: mujoco.MjModel,
    archive: np.lib.npyio.NpzFile,
    controller: RMRController | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray | int]]:
    if controller is None:
        raise ValueError("named RMR reference requires an RMR controller")
    missing = sorted(_RMR_REFERENCE_KEYS.difference(archive.files))
    if missing:
        raise ValueError(f"named RMR reference missing arrays: {missing}")

    stored_joint_names = tuple(map(str, archive["joint_names"]))
    if stored_joint_names != controller.actor_joint_names:
        raise ValueError("named RMR joint order does not match the controller source order")
    root_body_name = str(np.asarray(archive["root_body_name"]).item())
    root_body_index = int(np.asarray(archive["root_body_index"]).item())
    if root_body_name != "pelvis" or root_body_index != 0:
        raise ValueError("named RMR root must be pelvis at rigid-body index zero")

    fps_values = np.asarray(archive["fps"], dtype=np.float64).reshape(-1)
    if fps_values.size != 1 or fps_values[0] != 50.0:
        raise ValueError("named RMR reference fps must equal 50")
    fps = float(fps_values[0])
    joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(archive["joint_vel"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != model.nq - 7:
        raise ValueError(
            f"joint_pos must have shape (T, {model.nq - 7}), got {joint_pos.shape}"
        )
    frames = int(joint_pos.shape[0])
    if frames <= 0 or joint_vel.shape != (frames, model.nv - 6):
        raise ValueError(
            f"joint_vel must have shape ({frames}, {model.nv - 6}), "
            f"got {joint_vel.shape}"
        )
    permutation = np.asarray(controller.actor_to_model_permutation)
    if permutation.shape != (model.nq - 7,):
        raise ValueError("controller permutation does not match the MuJoCo model")

    body_arrays = {
        "body_pos_w": np.asarray(archive["body_pos_w"], dtype=np.float64),
        "body_quat_w": np.asarray(archive["body_quat_w"], dtype=np.float64),
        "body_lin_vel_w": np.asarray(
            archive["body_lin_vel_w"], dtype=np.float64
        ),
        "body_ang_vel_w": np.asarray(
            archive["body_ang_vel_w"], dtype=np.float64
        ),
    }
    body_count: int | None = None
    for key, trailing_size in (
        ("body_pos_w", 3),
        ("body_quat_w", 4),
        ("body_lin_vel_w", 3),
        ("body_ang_vel_w", 3),
    ):
        value = body_arrays[key]
        if value.ndim != 3 or value.shape[0] != frames or value.shape[2] != trailing_size:
            raise ValueError(f"{key} must have shape (T, B, {trailing_size})")
        if body_count is None:
            body_count = int(value.shape[1])
        elif value.shape[1] != body_count:
            raise ValueError("named RMR rigid-body arrays have inconsistent body counts")
    if body_count is None or not 0 <= root_body_index < body_count:
        raise ValueError("named RMR root_body_index is out of range")

    carried_arrays = (joint_pos, joint_vel, *body_arrays.values())
    if not all(np.isfinite(array).all() for array in carried_arrays):
        raise ValueError("named RMR state arrays must be finite")

    qpos = np.zeros((frames, model.nq), dtype=np.float64)
    qvel = np.zeros((frames, model.nv), dtype=np.float64)
    qpos[:, :3] = body_arrays["body_pos_w"][:, root_body_index]
    qpos[:, 3:7] = body_arrays["body_quat_w"][:, root_body_index]
    qpos[:, 7:] = joint_pos[:, permutation]
    qvel[:, 6:] = joint_vel[:, permutation]

    root_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
    )
    if root_body_id < 0:
        raise ValueError(f"named RMR root body missing from model: {root_body_name}")
    for frame in range(frames):
        qvel[frame, :6] = _root_generalized_velocity(
            model,
            qpos[frame],
            root_body_id,
            body_arrays["body_lin_vel_w"][frame, root_body_index],
            body_arrays["body_ang_vel_w"][frame, root_body_index],
        )
    validation_state: dict[str, np.ndarray | int] = {
        **body_arrays,
        "root_body_index": root_body_index,
    }
    return qpos, qvel, fps, validation_state


def _validate_named_rmr_root(
    reference_body_pos: np.ndarray,
    reference_body_quat: np.ndarray,
    reference_body_lin_vel: np.ndarray,
    reference_body_ang_vel: np.ndarray,
    validation_state: dict[str, np.ndarray | int],
) -> None:
    root_index = int(validation_state["root_body_index"])
    expected_pos = np.asarray(validation_state["body_pos_w"])[:, root_index]
    expected_quat = np.asarray(validation_state["body_quat_w"])[:, root_index]
    expected_lin_vel = np.asarray(validation_state["body_lin_vel_w"])[
        :, root_index
    ]
    expected_ang_vel = np.asarray(validation_state["body_ang_vel_w"])[
        :, root_index
    ]
    np.testing.assert_allclose(
        reference_body_pos[:, 0], expected_pos, atol=2e-5, rtol=0.0
    )
    quaternion_error = np.minimum(
        np.linalg.norm(reference_body_quat[:, 0] - expected_quat, axis=-1),
        np.linalg.norm(reference_body_quat[:, 0] + expected_quat, axis=-1),
    )
    if np.any(quaternion_error > 2e-5):
        raise ValueError(
            "named RMR root orientation does not reconstruct in MuJoCo "
            f"(max error {float(np.max(quaternion_error))})"
        )
    np.testing.assert_allclose(
        reference_body_lin_vel[:, 0], expected_lin_vel, atol=2e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        reference_body_ang_vel[:, 0], expected_ang_vel, atol=2e-5, rtol=0.0
    )


def load_mujoco_reference(
    model: mujoco.MjModel,
    reference_path: str | Path,
    body_names: Sequence[str] = RMR_G1_BODY_NAMES,
    controller: RMRController | None = None,
) -> MujocoReference:
    """Loads qpos/qvel and precomputes RMR body state through MuJoCo FK."""
    with np.load(reference_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        is_legacy = {"X", "V"}.issubset(keys)
        is_named_rmr = _RMR_REFERENCE_KEYS.issubset(keys)
        if is_legacy and is_named_rmr:
            raise ValueError("reference archive has ambiguous legacy and named schemas")
        if is_legacy:
            qpos = np.array(archive["X"], dtype=np.float64, copy=True)
            qvel = np.array(archive["V"], dtype=np.float64, copy=True)
            fps = None
            named_validation = None
        elif is_named_rmr:
            qpos, qvel, fps, named_validation = _load_named_rmr_state(
                model, archive, controller
            )
        else:
            raise ValueError(
                "reference archive must use legacy X/V or named RMR schema; "
                f"found keys {sorted(keys)}"
            )

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
    if named_validation is not None:
        try:
            _validate_named_rmr_root(
                body_pos,
                body_quat,
                body_lin_vel,
                body_ang_vel,
                named_validation,
            )
        except AssertionError as error:
            raise ValueError(
                f"named RMR root state does not reconstruct in MuJoCo: {error}"
            ) from error
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
        fps=fps,
    )
