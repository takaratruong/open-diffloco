"""RMR G1 joint controller provenance and ordering."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class RMRController:
    """PD controller arrays in MuJoCo hinge-joint order."""

    joint_names: tuple[str, ...]
    actor_joint_names: tuple[str, ...]
    actor_to_model_permutation: np.ndarray
    model_to_actor_permutation: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray


def _model_hinge_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    )


def load_rmr_controller(
    model: mujoco.MjModel, controller_path: str | Path
) -> RMRController:
    """Loads controller arrays and recovers the exact logged RMR action scale."""
    with np.load(controller_path, allow_pickle=False) as archive:
        source_names = tuple(map(str, archive["joint_names"]))
        model_names = _model_hinge_names(model)
        if len(model_names) != model.nv - 6:
            raise ValueError("expected one hinge joint for every actuated G1 DoF")
        missing = [name for name in model_names if name not in source_names]
        if missing:
            raise ValueError(f"controller joints missing from model order: {missing}")
        permutation = np.array(
            [source_names.index(name) for name in model_names], dtype=np.int32
        )
        inverse_permutation = np.argsort(permutation).astype(np.int32)

        def ordered(key: str) -> np.ndarray:
            return np.array(
                archive[key][permutation], dtype=np.float64, copy=True
            )

        kp = ordered("kp")
        kd = ordered("kd")
        effort_limit = ordered("effort_limit")
        default_joint_pos = ordered("default_joint_pos")
        actions = np.asarray(
            archive["action"][:, permutation], dtype=np.float64
        )
        position_targets = np.asarray(
            archive["pos_target"][:, permutation], dtype=np.float64
        )

    scales = np.empty(len(model_names), dtype=np.float64)
    for joint in range(len(model_names)):
        supported = np.abs(actions[:, joint]) > 1e-4
        if not supported.any():
            raise ValueError(
                f"cannot infer action scale for {model_names[joint]}: "
                "logged actions are all zero"
            )
        ratios = (
            position_targets[supported, joint] - default_joint_pos[joint]
        ) / actions[supported, joint]
        scales[joint] = np.median(ratios)

    reconstructed = default_joint_pos + actions * scales
    max_error = float(np.max(np.abs(reconstructed - position_targets)))
    if max_error > 2e-6:
        raise ValueError(
            f"logged RMR action scale is inconsistent (max error {max_error})"
        )

    arrays = (
        permutation,
        inverse_permutation,
        kp,
        kd,
        effort_limit,
        default_joint_pos,
        scales,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("controller arrays must be finite")
    if not all((array > 0.0).all() for array in (kp, kd, effort_limit, scales)):
        raise ValueError("controller gains, effort limits, and scales must be positive")
    for array in arrays:
        array.setflags(write=False)

    return RMRController(
        joint_names=model_names,
        actor_joint_names=source_names,
        actor_to_model_permutation=permutation,
        model_to_actor_permutation=inverse_permutation,
        kp=kp,
        kd=kd,
        effort_limit=effort_limit,
        default_joint_pos=default_joint_pos,
        action_scale=scales,
    )
