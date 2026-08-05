"""Differentiable MJX implementation of the RMR G1 tracking task."""

import functools
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from src.core.data_structures import EnvState
from src.envs.g1_tracking.controller import load_rmr_controller
from src.envs.g1_tracking.reference import (
    RMR_G1_BODY_NAMES,
    load_mujoco_reference,
)
from src.envs.g1_tracking.reward import (
    quaternion_error_magnitude,
    rmr_regularization_reward,
    rmr_tracking_reward,
)


DEFAULT_MODEL_PATH = (
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
DEFAULT_REFERENCE_PATH = (
    "/home/ubuntu/projects/diffsim2real/outputs/w02_rmrspec_grounded.npz"
)
DEFAULT_CONTROLLER_PATH = (
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)


def _quat_inv(q: jax.Array) -> jax.Array:
    return q * jp.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = jp.moveaxis(a, -1, 0)
    bw, bx, by, bz = jp.moveaxis(b, -1, 0)
    return jp.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_apply(q: jax.Array, vector: jax.Array) -> jax.Array:
    q_vector = q[..., 1:]
    scalar = q[..., :1]
    return (
        2.0 * jp.sum(q_vector * vector, axis=-1, keepdims=True) * q_vector
        + (jp.square(scalar) - jp.sum(jp.square(q_vector), axis=-1, keepdims=True))
        * vector
        + 2.0 * scalar * jp.cross(q_vector, vector)
    )


def _yaw_quaternion(q: jax.Array) -> jax.Array:
    w, x, y, z = jp.moveaxis(q, -1, 0)
    yaw = jp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    zeros = jp.zeros_like(yaw)
    return jp.stack(
        (jp.cos(0.5 * yaw), zeros, zeros, jp.sin(0.5 * yaw)), axis=-1
    )


def _rotation_6d(q: jax.Array) -> jax.Array:
    """Matches upstream matrix_from_quat(q)[..., :2].reshape(...)."""
    w, x, y, z = jp.moveaxis(q, -1, 0)
    matrix = jp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return matrix[..., :2, :].reshape(q.shape[:-1] + (6,))


class G1TrackingEnv:
    """RMR task geometry with first-order MJX dynamics and SHAC interfaces."""

    def __init__(
        self,
        xml_path: str = DEFAULT_MODEL_PATH,
        reference_path: str = DEFAULT_REFERENCE_PATH,
        controller_path: str = DEFAULT_CONTROLLER_PATH,
        actor_history_len: int = 1,
        **_unused_go2_options,
    ):
        if actor_history_len < 1:
            raise ValueError("actor_history_len must be at least one")

        self.xml_path = str(Path(xml_path))
        self.reference_path = str(Path(reference_path))
        self.controller_path = str(Path(controller_path))
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        if (self.mj_model.nq, self.mj_model.nv) != (36, 35):
            raise ValueError(
                "RMR G1 model must expose nq=36 and nv=35, got "
                f"{self.mj_model.nq}/{self.mj_model.nv}"
            )

        # The working Open-DiffLoco path uses a fixed, fully differentiable
        # solver budget. Preserve the G1 2 ms physics step and advance five
        # substeps per 100 Hz reference frame.
        self.mj_model.opt.iterations = 1
        self.mj_model.opt.ls_iterations = 5
        self.mj_model.geom_margin[:] = 0.0
        self.mjx_model = mjx.put_model(self.mj_model)

        self.reference = load_mujoco_reference(
            self.mj_model, self.reference_path, RMR_G1_BODY_NAMES
        )
        self.controller = load_rmr_controller(
            self.mj_model, self.controller_path
        )
        self.reference_length = self.reference.qpos.shape[0]
        self.body_ids = tuple(self.reference.body_ids)
        self.anchor_body_id = self.body_ids[0]
        self.distal_body_slots = (3, 6, 10, 13)

        self.qpos_reference = jp.asarray(self.reference.qpos)
        self.qvel_reference = jp.asarray(self.reference.qvel)
        self.body_pos_reference = jp.asarray(self.reference.body_pos)
        self.body_quat_reference = jp.asarray(self.reference.body_quat)
        self.body_lin_vel_reference = jp.asarray(
            self.reference.body_lin_vel
        )
        self.body_ang_vel_reference = jp.asarray(
            self.reference.body_ang_vel
        )
        self.kp = jp.asarray(self.controller.kp)
        self.kd = jp.asarray(self.controller.kd)
        self.effort_limit = jp.asarray(self.controller.effort_limit)
        self.default_joints = jp.asarray(
            self.controller.default_joint_pos
        )
        self.action_scales = jp.asarray(self.controller.action_scale)
        joint_ranges = np.asarray(self.mj_model.jnt_range[1:])
        joint_limited = np.asarray(self.mj_model.jnt_limited[1:], dtype=bool)
        if joint_ranges.shape != (29, 2):
            raise ValueError("RMR G1 model must expose 29 scalar joint ranges")
        joint_centers = np.mean(joint_ranges, axis=-1)
        soft_lower = joint_centers + 0.9 * (
            joint_ranges[:, 0] - joint_centers
        )
        soft_upper = joint_centers + 0.9 * (
            joint_ranges[:, 1] - joint_centers
        )
        self.soft_joint_lower = jp.asarray(
            np.where(joint_limited, soft_lower, -np.inf)
        )
        self.soft_joint_upper = jp.asarray(
            np.where(joint_limited, soft_upper, np.inf)
        )

        self.n_frames = 5
        self.dt = float(self.mj_model.opt.timestep * self.n_frames)
        self.action_dim = 29
        self.actor_history_len = actor_history_len
        self.actor_frame_obs_dim = 154
        self.actor_obs_dim = self.actor_frame_obs_dim * actor_history_len
        self.critic_obs_dim = 286

        # Compatibility fields consumed by the unchanged Open-DiffLoco SHAC
        # rollout. Terrain/disturbance amplitudes remain zero in the registered
        # first task discriminator.
        left_foot = self.body_ids[3]
        right_foot = self.body_ids[6]
        self._foot_body_ids = jp.array(
            [left_foot, right_foot, left_foot, right_foot], dtype=jp.int32
        )
        self.nominal_total_mass = float(jp.sum(self.mjx_model.body_mass))
        self.base_gravity_mag = float(jp.linalg.norm(self.mjx_model.opt.gravity))

    def _body_state(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        positions = []
        quaternions = []
        linear_velocities = []
        angular_velocities = []
        for body_id in self.body_ids:
            root_id = int(self.mj_model.body_rootid[body_id])
            angular_velocity = data.cvel[body_id, :3]
            offset = data.xpos[body_id] - data.subtree_com[root_id]
            positions.append(data.xpos[body_id])
            quaternions.append(data.xquat[body_id])
            linear_velocities.append(
                data.cvel[body_id, 3:]
                - jp.cross(offset, angular_velocity)
            )
            angular_velocities.append(angular_velocity)
        return (
            jp.stack(positions),
            jp.stack(quaternions),
            jp.stack(linear_velocities),
            jp.stack(angular_velocities),
        )

    def _aligned_reference_body_targets(
        self,
        actual_anchor_pos: jax.Array,
        actual_anchor_quat: jax.Array,
        phase: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        reference_anchor_pos = self.body_pos_reference[phase, 0]
        reference_anchor_quat = self.body_quat_reference[phase, 0]
        yaw_delta = _yaw_quaternion(
            _quat_mul(actual_anchor_quat, _quat_inv(reference_anchor_quat))
        )
        aligned_anchor = actual_anchor_pos.at[2].set(reference_anchor_pos[2])
        positions = aligned_anchor + _quat_apply(
            yaw_delta,
            self.body_pos_reference[phase] - reference_anchor_pos,
        )
        quaternions = _quat_mul(
            jp.broadcast_to(
                yaw_delta, self.body_quat_reference[phase].shape
            ),
            self.body_quat_reference[phase],
        )
        return positions, quaternions

    def _tracking_reward(self, data: mjx.Data, info: dict):
        body_pos, body_quat, body_lin_vel, body_ang_vel = self._body_state(data)
        return self._tracking_reward_from_body_state(
            info, body_pos, body_quat, body_lin_vel, body_ang_vel
        )

    def _tracking_reward_from_body_state(
        self,
        info: dict,
        body_pos: jax.Array,
        body_quat: jax.Array,
        body_lin_vel: jax.Array,
        body_ang_vel: jax.Array,
    ):
        phase = info["phase"]
        target_body_pos, target_body_quat = (
            self._aligned_reference_body_targets(
                body_pos[0], body_quat[0], phase
            )
        )
        return rmr_tracking_reward(
            target_anchor_pos=self.body_pos_reference[phase, 0],
            actual_anchor_pos=body_pos[0],
            target_anchor_quat=self.body_quat_reference[phase, 0],
            actual_anchor_quat=body_quat[0],
            target_body_pos=target_body_pos,
            actual_body_pos=body_pos,
            target_body_quat=target_body_quat,
            actual_body_quat=body_quat,
            target_body_lin_vel=self.body_lin_vel_reference[phase],
            actual_body_lin_vel=body_lin_vel,
            target_body_ang_vel=self.body_ang_vel_reference[phase],
            actual_body_ang_vel=body_ang_vel,
        )

    def _anchor_relative_reference(
        self, data: mjx.Data, phase: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        actual_pos = data.xpos[self.anchor_body_id]
        actual_quat = data.xquat[self.anchor_body_id]
        inverse_actual = _quat_inv(actual_quat)
        position = _quat_apply(
            inverse_actual,
            self.body_pos_reference[phase, 0] - actual_pos,
        )
        orientation = _quat_mul(
            inverse_actual, self.body_quat_reference[phase, 0]
        )
        return position, orientation

    def _get_actor_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        phase = info["phase"]
        _, anchor_orientation = self._anchor_relative_reference(data, phase)
        root_inverse = _quat_inv(data.qpos[3:7])
        return jp.concatenate(
            (
                self.qpos_reference[phase, 7:],
                self.qvel_reference[phase, 6:],
                _rotation_6d(anchor_orientation),
                _quat_apply(root_inverse, data.qvel[3:6]),
                data.qpos[7:] - self.default_joints,
                data.qvel[6:],
                info["last_act"],
            )
        )

    def _get_critic_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        phase = info["phase"]
        anchor_pos, anchor_orientation = self._anchor_relative_reference(
            data, phase
        )
        body_pos, body_quat, _, _ = self._body_state(data)
        actual_anchor_pos = body_pos[0]
        actual_anchor_quat = body_quat[0]
        inverse_anchor = _quat_inv(actual_anchor_quat)
        inverse_anchor_bodies = jp.broadcast_to(
            inverse_anchor, body_quat.shape
        )
        body_pos_b = _quat_apply(
            inverse_anchor_bodies,
            body_pos - actual_anchor_pos,
        )
        body_quat_b = _quat_mul(inverse_anchor_bodies, body_quat)
        root_inverse = _quat_inv(data.qpos[3:7])
        return jp.concatenate(
            (
                self.qpos_reference[phase, 7:],
                self.qvel_reference[phase, 6:],
                anchor_pos,
                _rotation_6d(anchor_orientation),
                body_pos_b.reshape(-1),
                _rotation_6d(body_quat_b).reshape(-1),
                _quat_apply(root_inverse, data.qvel[:3]),
                _quat_apply(root_inverse, data.qvel[3:6]),
                data.qpos[7:] - self.default_joints,
                data.qvel[6:],
                info["last_act"],
            )
        )

    def _base_info(
        self,
        *,
        rng: jax.Array,
        phase: jax.Array,
        difficulty: jax.Array,
    ) -> dict:
        return {
            "step": jp.array(0, dtype=jp.int32),
            "phase": phase.astype(jp.int32),
            "last_act": jp.zeros(self.action_dim),
            "rng": rng,
            "difficulty": difficulty,
            "foot_bump_ou": jp.zeros((4, 3)),
            "foot_normal_forces": jp.zeros(4),
            "terminal": jp.array(0.0),
        }

    def _init_metrics(self) -> dict:
        zero = jp.float32(0.0)
        return {
            "contact_force": zero,
            "anchor_position_error": zero,
            "anchor_orientation_error": zero,
            "body_position_error": zero,
            "body_orientation_error": zero,
            "body_linear_velocity_error": zero,
            "body_angular_velocity_error": zero,
            "rew_anchor_position": zero,
            "rew_anchor_orientation": zero,
            "rew_body_position": zero,
            "rew_body_orientation": zero,
            "rew_body_linear_velocity": zero,
            "rew_body_angular_velocity": zero,
            "rew_action_rate": zero,
            "rew_joint_limit": zero,
            # Compatibility names for the current SHAC logger.
            "vel_x": zero,
            "vel_y": zero,
            "yaw_rate": zero,
            "cmd_x": zero,
            "cmd_y": zero,
            "cmd_yaw": zero,
            "rew_vel_x": zero,
            "rew_vel_y": zero,
            "rew_yaw": zero,
            "rew_vz": zero,
            "pen_rate": zero,
            "height": zero,
            "tilt": zero,
            "foot_normal_FL": zero,
            "foot_normal_FR": zero,
            "foot_normal_RL": zero,
            "foot_normal_RR": zero,
        }

    def reset_at_phase(
        self,
        rng: jax.Array,
        difficulty: jax.Array,
        phase: jax.Array,
    ) -> EnvState:
        """Creates an exact RSI state at a caller-selected reference frame."""
        phase = jp.asarray(phase, dtype=jp.int32)
        data = mjx.make_data(self.mjx_model)
        data = data.replace(
            qpos=self.qpos_reference[phase],
            qvel=self.qvel_reference[phase],
        )
        data = mjx.forward(self.mjx_model, data)
        info = self._base_info(rng=rng, phase=phase, difficulty=difficulty)
        actor_frame = self._get_actor_obs(data, info)
        actor_history = jp.repeat(
            actor_frame[None, :], self.actor_history_len, axis=0
        )
        critic_obs = self._get_critic_obs(data, info)
        info = {
            **info,
            "actor_obs_history": actor_history,
            "bootstrap_obs": actor_history.reshape(-1),
            "bootstrap_critic_obs": critic_obs,
        }
        return EnvState(
            data=data,
            obs=actor_history.reshape(-1),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            info=info,
            metrics=self._init_metrics(),
        )

    def reset(self, rng: jax.Array, difficulty: jax.Array) -> EnvState:
        rng, phase_key = jax.random.split(rng)
        phase = jax.random.randint(
            phase_key,
            (),
            minval=0,
            maxval=self.reference_length - 2,
            dtype=jp.int32,
        )
        return self.reset_at_phase(rng, difficulty, phase)

    def _termination(
        self,
        data: mjx.Data,
        info: dict,
        body_pos: jax.Array,
        body_quat: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        phase = info["phase"]
        anchor_z_error = jp.abs(
            self.body_pos_reference[phase, 0, 2] - body_pos[0, 2]
        )
        world_down = jp.array([0.0, 0.0, -1.0])
        target_down = _quat_apply(
            _quat_inv(self.body_quat_reference[phase, 0]), world_down
        )
        actual_down = _quat_apply(_quat_inv(body_quat[0]), world_down)
        gravity_error = jp.linalg.norm(target_down - actual_down)
        distal_z_error = jp.max(
            jp.abs(
                self.body_pos_reference[
                    phase, jp.array(self.distal_body_slots), 2
                ]
                - body_pos[jp.array(self.distal_body_slots), 2]
            )
        )
        nan_failure = (
            jp.any(~jp.isfinite(data.qpos))
            | jp.any(~jp.isfinite(data.qvel))
        )
        terminal = (
            (anchor_z_error > 0.25)
            | (gravity_error > 0.8)
            | (distal_z_error > 0.25)
            | nan_failure
        ).astype(jp.float64)
        clip_end = (phase >= self.reference_length - 1).astype(jp.float64)
        return jp.maximum(terminal, clip_end), terminal

    @functools.partial(jax.checkpoint, static_argnums=(0,))
    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        action = jp.clip(action, -1.0, 1.0).astype(jp.float64)
        position_target = self.default_joints + action * self.action_scales

        def physics_step(data, _):
            torque = jp.clip(
                self.kp * (position_target - data.qpos[7:])
                - self.kd * data.qvel[6:],
                -self.effort_limit,
                self.effort_limit,
            )
            applied = jp.zeros(self.mj_model.nv).at[6:].set(torque)
            return (
                mjx.step(
                    self.mjx_model,
                    data.replace(qfrc_applied=applied),
                ),
                None,
            )

        data, _ = jax.lax.scan(
            physics_step, state.data, None, length=self.n_frames
        )
        next_phase = jp.minimum(
            state.info["phase"] + 1, self.reference_length - 1
        )
        pre_reset_info = {
            **state.info,
            "step": state.info["step"] + 1,
            "phase": next_phase,
            "last_act": action,
        }
        body_pos, body_quat, body_lin_vel, body_ang_vel = self._body_state(data)
        reward, components = self._tracking_reward_from_body_state(
            pre_reset_info,
            body_pos,
            body_quat,
            body_lin_vel,
            body_ang_vel,
        )
        regularization_reward, regularization_components = (
            rmr_regularization_reward(
                action=action,
                previous_action=state.info["last_act"],
                joint_pos=data.qpos[7:],
                soft_joint_lower=self.soft_joint_lower,
                soft_joint_upper=self.soft_joint_upper,
            )
        )
        reward = reward + regularization_reward
        components = {**components, **regularization_components}
        done, terminal = self._termination(
            data, pre_reset_info, body_pos, body_quat
        )

        bootstrap_actor_frame = self._get_actor_obs(data, pre_reset_info)
        bootstrap_history = jp.concatenate(
            (
                state.info["actor_obs_history"][1:],
                bootstrap_actor_frame[None, :],
            ),
            axis=0,
        )
        bootstrap_critic_obs = self._get_critic_obs(data, pre_reset_info)

        rng, reset_key = jax.random.split(state.info["rng"])
        reset_state = self.reset(reset_key, state.info["difficulty"])
        next_data = jax.tree_util.tree_map(
            lambda current, reset: jp.where(done, reset, current),
            data,
            reset_state.data,
        )
        continued_info = {
            **pre_reset_info,
            "rng": rng,
            "foot_bump_ou": state.info["foot_bump_ou"],
            "foot_normal_forces": jp.zeros(4),
        }
        next_info = {
            key: jax.tree_util.tree_map(
                lambda current, reset: jp.where(done, reset, current),
                continued_info[key],
                reset_state.info[key],
            )
            for key in continued_info
        }
        # These describe the transition that just ended. Preserve the pre-reset
        # observation for time-limit bootstrap and the true terminal bit even
        # when the returned physical state has already been reset.
        next_info = {
            **next_info,
            "terminal": terminal,
            "bootstrap_obs": bootstrap_history.reshape(-1),
            "bootstrap_critic_obs": bootstrap_critic_obs,
        }
        next_actor_frame = self._get_actor_obs(next_data, next_info)
        continued_history = jp.concatenate(
            (
                state.info["actor_obs_history"][1:],
                next_actor_frame[None, :],
            ),
            axis=0,
        )
        next_history = jp.where(
            done, reset_state.info["actor_obs_history"], continued_history
        )
        next_info = {**next_info, "actor_obs_history": next_history}

        target_body_pos, target_body_quat = (
            self._aligned_reference_body_targets(
                body_pos[0], body_quat[0], next_phase
            )
        )
        anchor_position_error = jp.linalg.norm(
            self.body_pos_reference[next_phase, 0] - body_pos[0]
        )
        anchor_orientation_error = quaternion_error_magnitude(
            self.body_quat_reference[next_phase, 0], body_quat[0]
        )
        body_position_error = jp.mean(
            jp.linalg.norm(target_body_pos - body_pos, axis=-1)
        )
        body_orientation_error = jp.mean(
            quaternion_error_magnitude(target_body_quat, body_quat)
        )
        body_linear_velocity_error = jp.mean(
            jp.linalg.norm(
                self.body_lin_vel_reference[next_phase] - body_lin_vel,
                axis=-1,
            )
        )
        body_angular_velocity_error = jp.mean(
            jp.linalg.norm(
                self.body_ang_vel_reference[next_phase] - body_ang_vel,
                axis=-1,
            )
        )
        contact_force = jp.sum(jp.abs(data.qfrc_constraint[:6]))
        metrics = {
            "contact_force": contact_force,
            "anchor_position_error": anchor_position_error,
            "anchor_orientation_error": anchor_orientation_error,
            "body_position_error": body_position_error,
            "body_orientation_error": body_orientation_error,
            "body_linear_velocity_error": body_linear_velocity_error,
            "body_angular_velocity_error": body_angular_velocity_error,
            "rew_anchor_position": components["anchor_position"],
            "rew_anchor_orientation": components["anchor_orientation"],
            "rew_body_position": components["body_position"],
            "rew_body_orientation": components["body_orientation"],
            "rew_body_linear_velocity": components["body_linear_velocity"],
            "rew_body_angular_velocity": components["body_angular_velocity"],
            "rew_action_rate": components["action_rate"],
            "rew_joint_limit": components["joint_limit"],
            # Current SHAC logger compatibility: these carry errors, not velocity.
            "vel_x": anchor_position_error,
            "vel_y": body_position_error,
            "yaw_rate": body_orientation_error,
            "cmd_x": jp.array(0.0),
            "cmd_y": jp.array(0.0),
            "cmd_yaw": jp.array(0.0),
            "rew_vel_x": components["anchor_position"],
            "rew_vel_y": components["body_position"],
            "rew_yaw": components["body_orientation"],
            "rew_vz": components["body_linear_velocity"],
            "pen_rate": jp.array(0.0),
            "height": body_pos[0, 2],
            "tilt": _quat_apply(
                _quat_inv(body_quat[0]), jp.array([0.0, 0.0, -1.0])
            )[2],
            "foot_normal_FL": jp.array(0.0),
            "foot_normal_FR": jp.array(0.0),
            "foot_normal_RL": jp.array(0.0),
            "foot_normal_RR": jp.array(0.0),
        }
        metrics = jax.tree_util.tree_map(
            lambda value: value.astype(jp.float32), metrics
        )
        return EnvState(
            data=next_data,
            obs=next_history.reshape(-1),
            reward=reward,
            done=done,
            info=next_info,
            metrics=metrics,
        )

    def _apply_obs_noise(
        self, obs: jax.Array, _rng: jax.Array
    ) -> jax.Array:
        """The first registered discriminator intentionally has no obs noise."""
        return obs

    def normalize_actor_obs(
        self, normalizer, norm_state, obs: jax.Array
    ) -> jax.Array:
        frames = obs.reshape(*obs.shape[:-1], self.actor_history_len, -1)
        normalized = normalizer.normalize(norm_state, frames)
        return normalized.reshape(*obs.shape[:-1], self.actor_obs_dim)
