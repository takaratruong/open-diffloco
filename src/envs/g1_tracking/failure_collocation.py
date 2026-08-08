"""Small failure-centered multiple-shooting primitives for G1 LAFAN."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


QPOS_DIM = 36
QVEL_DIM = 35
ACTION_DIM = 29
STATE_DEFECT_DIM = 70
TERMINATION_LIMITS = jnp.array([0.25, 1.3, 0.8, 0.4])
TML_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
TML_JOINT_NAMES = (
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
TML_DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
TML_DEFAULT_JOINT_POS[[0, 1]] = -0.312
TML_DEFAULT_JOINT_POS[[9, 10]] = 0.669
TML_DEFAULT_JOINT_POS[[13, 14]] = -0.363
TML_DEFAULT_JOINT_POS[[11, 12]] = 0.2
TML_DEFAULT_JOINT_POS[15] = 0.2
TML_DEFAULT_JOINT_POS[16] = -0.2
TML_DEFAULT_JOINT_POS[[21, 22]] = 0.6
TML_ACTION_SCALE = np.asarray(
    [
        0.3506614565849304,
        0.3506614565849304,
        0.5475464463233948,
        0.3506614565849304,
        0.3506614565849304,
        0.4385773241519928,
        0.5475464463233948,
        0.5475464463233948,
        0.4385773241519928,
        0.3506614565849304,
        0.3506614565849304,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.4385773241519928,
        0.07450087368488312,
        0.07450087368488312,
        0.07450087368488312,
        0.07450087368488312,
    ],
    dtype=np.float32,
)
TML_DEFAULT_JOINT_POS.setflags(write=False)
TML_ACTION_SCALE.setflags(write=False)


@dataclass(frozen=True)
class FailureWindow:
    """Fixed topology around the selected actor's delayed collapse."""

    start_phase: int = 111
    end_phase: int = 135
    segment_steps: int = 2

    def __post_init__(self) -> None:
        values = (self.start_phase, self.end_phase, self.segment_steps)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("failure window values must be integers")
        if self.start_phase < 0 or self.end_phase <= self.start_phase:
            raise ValueError("failure window must contain positive phase extent")
        if self.segment_steps < 1:
            raise ValueError("segment_steps must be positive")
        if self.transitions % self.segment_steps:
            raise ValueError("failure window must contain complete shooting segments")

    @property
    def transitions(self) -> int:
        return self.end_phase - self.start_phase

    @property
    def segments(self) -> int:
        return self.transitions // self.segment_steps

    @property
    def knot_phases(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.start_phase,
                self.end_phase + 1,
                self.segment_steps,
            )
        )

    @property
    def decision_size(self) -> int:
        free_state_values = self.segments * (QPOS_DIM + QVEL_DIM)
        action_values = self.transitions * ACTION_DIM
        return free_state_values + action_values

    @property
    def equality_size(self) -> int:
        return self.segments * (STATE_DEFECT_DIM + 1)


def _quaternion_inverse(quaternion: jax.Array) -> jax.Array:
    return quaternion * jnp.array([1.0, -1.0, -1.0, -1.0])


def _quaternion_product(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def physical_state_defect(
    predicted_qpos: jax.Array,
    predicted_qvel: jax.Array,
    knot_qpos: jax.Array,
    knot_qvel: jax.Array,
) -> jax.Array:
    """Return a 70D physical defect with sign-invariant root orientation."""
    predicted_qpos = jnp.asarray(predicted_qpos)
    predicted_qvel = jnp.asarray(predicted_qvel)
    knot_qpos = jnp.asarray(knot_qpos)
    knot_qvel = jnp.asarray(knot_qvel)
    if predicted_qpos.shape != (QPOS_DIM,) or knot_qpos.shape != (QPOS_DIM,):
        raise ValueError("qpos defects require width 36")
    if predicted_qvel.shape != (QVEL_DIM,) or knot_qvel.shape != (QVEL_DIM,):
        raise ValueError("qvel defects require width 35")

    relative = _quaternion_product(
        predicted_qpos[3:7], _quaternion_inverse(knot_qpos[3:7])
    )
    relative = jnp.where(relative[0] < 0.0, -relative, relative)
    return jnp.concatenate(
        (
            predicted_qpos[:3] - knot_qpos[:3],
            2.0 * relative[1:],
            predicted_qpos[7:] - knot_qpos[7:],
            predicted_qvel - knot_qvel,
        )
    )


def rollout_segment(
    step_fn: Callable[
        [jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ],
    qpos: jax.Array,
    qvel: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Roll one deliberately short differentiable shooting segment."""
    qpos = jnp.asarray(qpos)
    qvel = jnp.asarray(qvel)
    actions = jnp.asarray(actions)
    if qpos.shape != (QPOS_DIM,) or qvel.shape != (QVEL_DIM,):
        raise ValueError("segment state must have qpos/qvel widths 36/35")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError("segment actions must have width 29")
    if actions.shape[0] < 1:
        raise ValueError("segment must contain at least one action")
    for action in actions:
        qpos, qvel = step_fn(qpos, qvel, action)
    return qpos, qvel


def multiple_shooting_equalities(
    step_fn: Callable[
        [jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ],
    knot_qpos: jax.Array,
    knot_qvel: jax.Array,
    actions: jax.Array,
    *,
    segment_steps: int,
) -> jax.Array:
    """Return explicit segment defects and free-knot quaternion equalities."""
    knot_qpos = jnp.asarray(knot_qpos)
    knot_qvel = jnp.asarray(knot_qvel)
    actions = jnp.asarray(actions)
    if isinstance(segment_steps, bool) or not isinstance(segment_steps, int):
        raise ValueError("segment_steps must be a positive integer")
    if segment_steps < 1:
        raise ValueError("segment_steps must be a positive integer")
    if knot_qpos.ndim != 2 or knot_qpos.shape[1] != QPOS_DIM:
        raise ValueError("knot_qpos must have width 36")
    if knot_qvel.shape != (knot_qpos.shape[0], QVEL_DIM):
        raise ValueError("knot_qvel must align with qpos at width 35")
    segments = knot_qpos.shape[0] - 1
    if segments < 1:
        raise ValueError("multiple shooting requires at least two knots")
    if actions.shape != (segments * segment_steps, ACTION_DIM):
        raise ValueError("actions do not align with shooting segments")

    residuals = []
    for segment in range(segments):
        action_start = segment * segment_steps
        predicted_qpos, predicted_qvel = rollout_segment(
            step_fn,
            knot_qpos[segment],
            knot_qvel[segment],
            actions[action_start : action_start + segment_steps],
        )
        defect = physical_state_defect(
            predicted_qpos,
            predicted_qvel,
            knot_qpos[segment + 1],
            knot_qvel[segment + 1],
        )
        quaternion_norm = jnp.reshape(
            jnp.sum(jnp.square(knot_qpos[segment + 1, 3:7])) - 1.0,
            (1,),
        )
        residuals.append(jnp.concatenate((defect, quaternion_norm)))
    return jnp.concatenate(residuals)


def feasibility_merit(
    *,
    objective: jax.Array,
    equalities: jax.Array,
    slacks: jax.Array,
    defect_weight: float = 100.0,
    constraint_weight: float = 100.0,
) -> jax.Array:
    """Combine inspectable objective, defect, and violation terms."""
    if defect_weight <= 0.0 or constraint_weight <= 0.0:
        raise ValueError("merit weights must be positive")
    equalities = jnp.asarray(equalities)
    slacks = jnp.asarray(slacks)
    return (
        jnp.asarray(objective)
        + defect_weight * jnp.mean(jnp.square(equalities))
        + constraint_weight * jnp.mean(jnp.square(jnp.minimum(slacks, 0.0)))
    )


def stateless_physics_step(
    env,
    qpos: jax.Array,
    qvel: jax.Array,
    action: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Advance qpos/qvel through a fresh validated MJX physical state."""
    data = mjx.make_data(env.mjx_model).replace(qpos=qpos, qvel=qvel)
    data = mjx.forward(env.mjx_model, data)
    data, _, _ = env.advance_physics(data, action)
    return data.qpos, data.qvel


def failure_objective(
    env,
    knot_qpos: jax.Array,
    knot_qvel: jax.Array,
    knot_phases: jax.Array,
    actions: jax.Array,
    nominal_actions: jax.Array,
) -> jax.Array:
    """Small objective centered on pelvis-height recovery."""
    knot_qpos = jnp.asarray(knot_qpos)
    knot_qvel = jnp.asarray(knot_qvel)
    knot_phases = jnp.asarray(knot_phases, dtype=jnp.int32)
    actions = jnp.asarray(actions)
    nominal_actions = jnp.asarray(nominal_actions)
    if knot_qpos.ndim != 2 or knot_qpos.shape[1] != QPOS_DIM:
        raise ValueError("objective knot_qpos must have width 36")
    if knot_qvel.shape != (knot_qpos.shape[0], QVEL_DIM):
        raise ValueError("objective knot_qvel must align at width 35")
    if knot_phases.shape != (knot_qpos.shape[0],):
        raise ValueError("objective phases must align with knots")
    if actions.shape != nominal_actions.shape or actions.ndim != 2:
        raise ValueError("objective actions and nominal actions must align")
    if actions.shape[1] != ACTION_DIM:
        raise ValueError("objective actions must have width 29")

    pelvis_height_loss = jnp.mean(
        jnp.square(
            knot_qpos[:, 2] - env.qpos_reference[knot_phases, 2]
        )
    )
    joint_tracking_loss = jnp.mean(
        jnp.square(
            knot_qpos[:, 7:] - env.qpos_reference[knot_phases, 7:]
        )
    )
    correction_loss = jnp.mean(jnp.square(actions - nominal_actions))
    rate_loss = jnp.mean(jnp.square(actions[1:] - actions[:-1]))
    return (
        pelvis_height_loss
        + 1e-3 * joint_tracking_loss
        + correction_loss
        + 1e-3 * rate_loss
    )


def anchor_xy_squared_slack(
    reference_xy: jax.Array,
    actual_xy: jax.Array,
    *,
    limit: float = 1.3,
) -> jax.Array:
    """Return an exact-feasible-set squared XY-distance slack."""
    reference_xy = jnp.asarray(reference_xy)
    actual_xy = jnp.asarray(actual_xy)
    if reference_xy.shape != (2,) or actual_xy.shape != (2,):
        raise ValueError("anchor XY positions must have shape (2,)")
    if limit <= 0.0:
        raise ValueError("anchor XY limit must be positive")
    delta = reference_xy - actual_xy
    return jnp.asarray(limit) ** 2 - jnp.sum(jnp.square(delta))


def physical_path_slack_components(
    env,
    knot_qpos: jax.Array,
    knot_qvel: jax.Array,
    knot_phases: jax.Array,
    actions: jax.Array,
    *,
    segment_steps: int,
    contact_penetration_allowance: float = 0.005,
) -> dict[str, jax.Array]:
    """Return terminal, action, torque, and contact slack components."""
    knot_qpos = jnp.asarray(knot_qpos)
    knot_qvel = jnp.asarray(knot_qvel)
    knot_phases = jnp.asarray(knot_phases, dtype=jnp.int32)
    actions = jnp.asarray(actions)
    if contact_penetration_allowance < 0.0:
        raise ValueError("contact penetration allowance must be non-negative")
    segments = knot_qpos.shape[0] - 1
    if knot_qvel.shape != (segments + 1, QVEL_DIM):
        raise ValueError("slack knots must align at qvel width 35")
    if knot_qpos.shape != (segments + 1, QPOS_DIM):
        raise ValueError("slack knots must align at qpos width 36")
    if knot_phases.shape != (segments + 1,):
        raise ValueError("slack phases must align with knots")
    if actions.shape != (segments * segment_steps, ACTION_DIM):
        raise ValueError("slack actions do not align with segments")

    terminal_slacks = []
    action_slacks = []
    torque_slacks = []
    contact_slacks = []
    for knot in range(segments + 1):
        data = mjx.make_data(env.mjx_model).replace(
            qpos=knot_qpos[knot], qvel=knot_qvel[knot]
        )
        kinematic_data = mjx.kinematics(env.mjx_model, data)
        body_pos = jnp.stack(
            tuple(kinematic_data.xpos[body_id] for body_id in env.body_ids)
        )
        body_quat = jnp.stack(
            tuple(kinematic_data.xquat[body_id] for body_id in env.body_ids)
        )
        errors = env.termination_errors(
            phase=knot_phases[knot],
            body_pos=body_pos,
            body_quat=body_quat,
        )
        anchor_xy_slack = anchor_xy_squared_slack(
            env.body_pos_reference[knot_phases[knot], 0, :2],
            body_pos[0, :2],
            limit=1.3,
        )
        terminal_slacks.append(
            jnp.stack(
                (
                    TERMINATION_LIMITS[0] - errors["anchor_z_error"],
                    anchor_xy_slack,
                    TERMINATION_LIMITS[2] - errors["gravity_z_error"],
                    TERMINATION_LIMITS[3] - errors["distal_z_error"],
                )
            )
        )
        contact_data = mjx.forward(env.mjx_model, data)
        contact_slacks.append(
            jnp.reshape(
                jnp.min(contact_data.contact.dist)
                + contact_penetration_allowance,
                (1,),
            )
        )

    for segment in range(segments):
        data = mjx.make_data(env.mjx_model).replace(
            qpos=knot_qpos[segment], qvel=knot_qvel[segment]
        )
        data = mjx.forward(env.mjx_model, data)
        action_start = segment * segment_steps
        for action in actions[action_start : action_start + segment_steps]:
            data, prepared_action, raw_torque = env.advance_physics(
                data, action
            )
            action_slacks.append(1.0 - jnp.abs(prepared_action))
            torque_slacks.append(env.effort_limit - jnp.abs(raw_torque))
    return {
        "terminal": jnp.concatenate(terminal_slacks),
        "action": jnp.concatenate(action_slacks),
        "torque": jnp.concatenate(torque_slacks),
        "contact": jnp.concatenate(contact_slacks),
    }


def physical_path_slack_groups(
    env,
    knot_qpos: jax.Array,
    knot_qvel: jax.Array,
    knot_phases: jax.Array,
    actions: jax.Array,
    *,
    segment_steps: int,
    contact_penetration_allowance: float = 0.005,
) -> tuple[jax.Array, jax.Array]:
    """Separate candidate-solve hard slacks from contact diagnostics."""
    components = physical_path_slack_components(
        env,
        knot_qpos,
        knot_qvel,
        knot_phases,
        actions,
        segment_steps=segment_steps,
        contact_penetration_allowance=contact_penetration_allowance,
    )
    differentiable = jnp.concatenate(
        (
            components["terminal"],
            components["action"],
            components["torque"],
        )
    )
    return differentiable, components["contact"]


def physical_path_slacks(
    env,
    knot_qpos: jax.Array,
    knot_qvel: jax.Array,
    knot_phases: jax.Array,
    actions: jax.Array,
    *,
    segment_steps: int,
    contact_penetration_allowance: float = 0.005,
) -> jax.Array:
    """Evaluate all path slacks, retaining contact as a value diagnostic."""
    differentiable, contact = physical_path_slack_groups(
        env,
        knot_qpos,
        knot_qvel,
        knot_phases,
        actions,
        segment_steps=segment_steps,
        contact_penetration_allowance=contact_penetration_allowance,
    )
    return jnp.concatenate((differentiable, contact))


def world_body_kinematics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    body_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate body pose and velocities in MuJoCo's world frame."""
    qpos_array = np.asarray(qpos, dtype=np.float64)
    qvel_array = np.asarray(qvel, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != model.nq:
        raise ValueError("world kinematics qpos shape does not match model")
    if qvel_array.shape != (qpos_array.shape[0], model.nv):
        raise ValueError("world kinematics qvel shape does not match model")
    if not np.isfinite(qpos_array).all() or not np.isfinite(qvel_array).all():
        raise ValueError("world kinematics state must be finite")
    body_ids = tuple(body_ids)
    if (
        not body_ids
        or len(set(body_ids)) != len(body_ids)
        or any(body_id <= 0 or body_id >= model.nbody for body_id in body_ids)
    ):
        raise ValueError("body_ids must be unique non-world model bodies")

    rows = qpos_array.shape[0]
    positions = np.empty((rows, len(body_ids), 3), dtype=np.float64)
    rotations = np.empty((rows, len(body_ids), 4), dtype=np.float64)
    linear_velocities = np.empty((rows, len(body_ids), 3), dtype=np.float64)
    angular_velocities = np.empty((rows, len(body_ids), 3), dtype=np.float64)
    data = mujoco.MjData(model)
    jacobian_position = np.empty((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.empty((3, model.nv), dtype=np.float64)
    for row in range(rows):
        data.qpos[:] = qpos_array[row]
        data.qvel[:] = qvel_array[row]
        mujoco.mj_forward(model, data)
        for slot, body_id in enumerate(body_ids):
            positions[row, slot] = data.xpos[body_id]
            rotations[row, slot] = data.xquat[body_id]
            mujoco.mj_jacBody(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                body_id,
            )
            linear_velocities[row, slot] = jacobian_position @ data.qvel
            angular_velocities[row, slot] = jacobian_rotation @ data.qvel
    return positions, rotations, linear_velocities, angular_velocities


def corrected_episode_mapping(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    root_ang_vel: np.ndarray,
    body_pos: np.ndarray,
    body_rot: np.ndarray,
    body_lin_vel: np.ndarray,
    actions: np.ndarray,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    clip_name: str,
    env_origin: np.ndarray,
    checkpoint_sha256: str,
    config_sha256: str,
    checkpoint_path: str,
    config_path: str,
    motion_asset_sha256: str,
    terrain_asset_sha256: str,
    motion_asset_path: str,
    terrain_asset_path: str,
    grail_commit: str,
    correction_method: str,
    correction_run_id: str,
    correction_source_sha256: str,
    correction_code_commit: str,
    dynamics_model_sha256: str,
    dynamics_backend: str,
    episode_weight: float,
) -> dict[str, object]:
    """Map a dense DiffSim correction to the frozen TML raw contract."""

    def finite_array(
        value: np.ndarray, name: str, shape: tuple[int, ...]
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array.copy()

    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != QPOS_DIM:
        raise ValueError("qpos must be a matrix with width 36")
    rows = qpos_array.shape[0]
    if rows < 13:
        raise ValueError("corrected episode must contain at least 13 rows")
    if not np.isfinite(qpos_array).all():
        raise ValueError("qpos must be finite")
    qpos_array = qpos_array.copy()
    qvel_array = finite_array(qvel, "qvel", (rows, QVEL_DIM))
    root_ang_vel_array = finite_array(
        root_ang_vel, "root_ang_vel", (rows, 3)
    )
    action_array = finite_array(actions, "action", (rows, ACTION_DIM))
    body_pos_array = finite_array(body_pos, "body_pos", (rows, 30, 3))
    body_rot_array = finite_array(body_rot, "body_rot", (rows, 30, 4))
    body_lin_vel_array = finite_array(
        body_lin_vel, "body_lin_vel", (rows, 30, 3)
    )
    default_array = finite_array(
        default_joint_pos, "default_joint_pos", (ACTION_DIM,)
    )
    scale_array = finite_array(action_scale, "action_scale", (ACTION_DIM,))
    if np.any(scale_array <= 0.0):
        raise ValueError("action_scale must be positive")
    if not np.allclose(
        np.linalg.norm(qpos_array[:, 3:7], axis=1),
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError("root_rot must contain normalized WXYZ quaternions")
    if not np.allclose(
        np.linalg.norm(body_rot_array, axis=-1),
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError("body_rot must contain normalized WXYZ quaternions")

    def canonical_order(
        values: tuple[str, ...],
        name: str,
        expected_names: tuple[str, ...],
    ) -> np.ndarray:
        names = tuple(values)
        if (
            len(names) != len(expected_names)
            or len(set(names)) != len(expected_names)
            or any(not isinstance(value, str) or not value for value in names)
        ):
            raise ValueError(
                f"{name} must contain {len(expected_names)} unique names"
            )
        if set(names) != set(expected_names):
            raise ValueError(f"{name} must match the canonical G1 names")
        lookup = {value: index for index, value in enumerate(names)}
        return np.asarray([lookup[value] for value in expected_names])

    joint_order = canonical_order(
        joint_names, "joint_names", TML_JOINT_NAMES
    )
    body_order = canonical_order(body_names, "body_names", TML_BODY_NAMES)
    source_default = default_array[joint_order]
    source_scale = scale_array[joint_order]
    physical_target = (
        source_default[None, :]
        + source_scale[None, :] * action_array[:, joint_order]
    )
    tml_action = (
        physical_target - TML_DEFAULT_JOINT_POS[None, :]
    ) / TML_ACTION_SCALE[None, :]
    origin = finite_array(env_origin, "env_origin", (3,))

    def require_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be nonempty text")
        return value

    def require_hash(value: str, name: str, width: int) -> str:
        if not isinstance(value, str) or re.fullmatch(
            rf"[0-9a-f]{{{width}}}", value
        ) is None:
            raise ValueError(f"{name} must be {width} lowercase hex characters")
        return value

    if (
        isinstance(episode_weight, bool)
        or not np.isfinite(episode_weight)
        or episode_weight <= 0.0
    ):
        raise ValueError("episode_weight must be positive and finite")

    return {
        "schema_version": "sonic_grail_rollout_npz_v1",
        "root_pos": qpos_array[:, :3] - origin[None, :],
        "root_rot": qpos_array[:, 3:7],
        "root_ang_vel": root_ang_vel_array,
        "body_pos": body_pos_array[:, body_order] - origin[None, None, :],
        "body_rot": body_rot_array[:, body_order],
        "body_lin_vel": body_lin_vel_array[:, body_order],
        "joint_pos": qpos_array[:, 7:][:, joint_order],
        "joint_vel": qvel_array[:, 6:][:, joint_order],
        "action": tml_action,
        "joint_names": np.asarray(TML_JOINT_NAMES),
        "body_names": np.asarray(TML_BODY_NAMES),
        "default_joint_pos": TML_DEFAULT_JOINT_POS.copy(),
        "action_scale": TML_ACTION_SCALE.copy(),
        "quaternion_convention": "WXYZ",
        "root_position_frame": "world_relative_to_env_origin",
        "root_rotation_frame": "world",
        "body_position_frame": "world_relative_to_env_origin",
        "body_rotation_frame": "world",
        "body_linear_velocity_frame": "world",
        "root_angular_velocity_frame": "world",
        "sim_dt": 0.005,
        "control_dt": 0.02,
        "decimation": 4,
        "state_sample_phase": "pre_action",
        "action_sample_phase": "applied_to_transition_t_to_t_plus_1",
        "action_semantics": (
            "raw_sonic_action_pd_target_equals_default_plus_g1_model12_scale"
        ),
        "trajectory_source": "diffsim_corrected",
        "episode_weight": float(episode_weight),
        "clip_name": require_text(clip_name, "clip_name"),
        "env_origin": origin,
        "checkpoint_sha256": require_hash(
            checkpoint_sha256, "checkpoint_sha256", 64
        ),
        "config_sha256": require_hash(config_sha256, "config_sha256", 64),
        "checkpoint_path": require_text(checkpoint_path, "checkpoint_path"),
        "config_path": require_text(config_path, "config_path"),
        "motion_asset_sha256": require_hash(
            motion_asset_sha256, "motion_asset_sha256", 64
        ),
        "terrain_asset_sha256": require_hash(
            terrain_asset_sha256, "terrain_asset_sha256", 64
        ),
        "motion_asset_path": require_text(
            motion_asset_path, "motion_asset_path"
        ),
        "terrain_asset_path": require_text(
            terrain_asset_path, "terrain_asset_path"
        ),
        "grail_commit": require_hash(grail_commit, "grail_commit", 40),
        "correction_method": require_text(
            correction_method, "correction_method"
        ),
        "correction_run_id": require_text(
            correction_run_id, "correction_run_id"
        ),
        "correction_source_sha256": require_hash(
            correction_source_sha256, "correction_source_sha256", 64
        ),
        "correction_code_commit": require_hash(
            correction_code_commit, "correction_code_commit", 40
        ),
        "dynamics_model_sha256": require_hash(
            dynamics_model_sha256, "dynamics_model_sha256", 64
        ),
        "dynamics_backend": require_text(
            dynamics_backend, "dynamics_backend"
        ),
    }


def select_failure_window(
    arrays: Mapping[str, np.ndarray],
    window: FailureWindow,
) -> dict[str, np.ndarray]:
    """Select aligned actor knots and between-knot actions by exact phase."""
    required = {
        "phase": (None,),
        "qpos": (QPOS_DIM,),
        "qvel": (QVEL_DIM,),
        "action": (ACTION_DIM,),
    }
    missing = sorted(set(required) - set(arrays))
    if missing:
        raise ValueError(f"rollout arrays missing fields: {missing}")

    phase_raw = np.asarray(arrays["phase"])
    if phase_raw.ndim != 1:
        raise ValueError("phase must be a vector")
    phase = phase_raw.astype(np.int32)
    if not np.array_equal(phase_raw, phase):
        raise ValueError("phase must be integer-valued")
    if len(np.unique(phase)) != len(phase):
        raise ValueError("phase values must be unique")

    rows = len(phase)
    normalized: dict[str, np.ndarray] = {"phase": phase}
    for name, trailing_shape in required.items():
        if name == "phase":
            continue
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.shape != (rows, *trailing_shape):
            raise ValueError(
                f"{name} must have shape {(rows, *trailing_shape)}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        normalized[name] = value

    expected_phases = np.arange(
        window.start_phase, window.end_phase + 1, dtype=np.int32
    )
    phase_to_row = {int(value): index for index, value in enumerate(phase)}
    if any(int(value) not in phase_to_row for value in expected_phases):
        raise ValueError("rollout must contain every phase in the failure window")

    knot_phase = np.asarray(window.knot_phases, dtype=np.int32)
    action_phase = np.arange(
        window.start_phase, window.end_phase, dtype=np.int32
    )
    knot_rows = np.asarray([phase_to_row[int(value)] for value in knot_phase])
    action_rows = np.asarray([phase_to_row[int(value)] for value in action_phase])
    knot_qpos = normalized["qpos"][knot_rows].copy()
    quaternion_norm = np.linalg.norm(knot_qpos[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("selected knot root quaternions must be normalized")

    return {
        "knot_phase": knot_phase,
        "action_phase": action_phase,
        "knot_qpos": knot_qpos,
        "knot_qvel": normalized["qvel"][knot_rows].copy(),
        "actions": normalized["action"][action_rows].copy(),
    }
