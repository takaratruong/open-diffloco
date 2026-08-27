"""Differentiable MJX implementation of the RMR G1 tracking task."""

import functools
import math
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from src.core.contact import contact_stiffness
from src.core.data_structures import EnvState
from src.envs.g1_tracking.contact_topology import (
    contact_topology_event,
    grouped_body_pair_contacts,
    grouped_foot_support,
)
from src.envs.g1_tracking.controller import load_rmr_controller
from src.envs.g1_tracking.centroidal_momentum import (
    reference_capture_points,
    reference_centroidal_momentum,
    standing_com_height,
)
from src.envs.g1_tracking.reference import (
    RMR_G1_BODY_NAMES,
    load_mujoco_reference,
)
from src.envs.g1_tracking.randomization import (
    G1RandomizationRanges,
    sample_g1_randomization,
)
from src.envs.g1_tracking.reward import (
    quaternion_error_magnitude,
    root_velocity_tracking_reward,
    rmr_regularization_reward,
    rmr_tracking_reward,
    termination_margin_penalty,
    torso_orientation_tracking_reward,
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


def _quat_from_euler_xyz(euler: jax.Array) -> jax.Array:
    """Return a scalar-first quaternion from XYZ Euler angles."""
    roll, pitch, yaw = euler
    cr, sr = jp.cos(0.5 * roll), jp.sin(0.5 * roll)
    cp, sp = jp.cos(0.5 * pitch), jp.sin(0.5 * pitch)
    cy, sy = jp.cos(0.5 * yaw), jp.sin(0.5 * yaw)
    return jp.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
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
    return matrix[..., :, :2].reshape(q.shape[:-1] + (6,))


class G1TrackingEnv:
    """RMR task geometry with first-order MJX dynamics and SHAC interfaces."""

    def __init__(
        self,
        xml_path: str = DEFAULT_MODEL_PATH,
        reference_path: str = DEFAULT_REFERENCE_PATH,
        controller_path: str = DEFAULT_CONTROLLER_PATH,
        actor_history_len: int = 1,
        actor_observation_noise: bool = False,
        actor_observe_motion_anchor_position: bool = False,
        actor_joint_velocity_noise_scale: float = 0.01,
        squash_actor_actions_override: bool | None = None,
        actor_reference_lookahead_steps: tuple[int, ...] = (),
        actor_reference_preview_mode: str = "absolute",
        physics_substeps: int = 5,
        reference_stride: int = 1,
        reward_scale: float = 1.0,
        clip_actions: bool = True,
        actor_joint_order: str = "model",
        physics_timestep: float | None = None,
        solver_iterations: int = 1,
        solver_ls_iterations: int = 5,
        domain_randomization: bool = False,
        friction_range: tuple[float, float] = (1.0, 1.0),
        mass_range: tuple[float, float] = (1.0, 1.0),
        kp_range: tuple[float, float] = (35.0, 35.0),
        kd_range: tuple[float, float] = (0.5, 0.5),
        com_offset_range: tuple[float, float, float] = (0.0, 0.0, 0.0),
        randomization_com_body_name: str = "pelvis",
        randomization_uses_curriculum: bool = True,
        effort_limit_scale: float = 1.0,
        action_magnitude_weight: float = 0.0,
        termination_margin_weight: float = 0.0,
        tracking_velocity_kernel: str = "exponential",
        tracking_anchor_position_kernel: str = "exponential",
        tracking_torso_orientation_weight: float = 0.0,
        tracking_root_velocity_weight: float = 0.0,
        reference_reset_noise_scale: float = 0.0,
        reference_root_reset_noise_multiplier: float = 1.0,
        reference_root_reset_noise_probability: float = 0.0,
        reference_residual_control: bool = False,
        reference_residual_scale: float = 0.5,
        carried_reset_bank_path: str | None = None,
        carried_reset_probability: float = 0.0,
        carried_reset_bank_start: int = 0,
        adaptive_phase_sampling: bool = False,
        adaptive_phase_uniform_ratio: float = 0.5,
        **_unused_go2_options,
    ):
        if actor_history_len < 1:
            raise ValueError("actor_history_len must be at least one")
        if not isinstance(actor_observation_noise, bool):
            raise ValueError("actor_observation_noise must be boolean")
        self.actor_observation_noise = actor_observation_noise
        if not isinstance(actor_observe_motion_anchor_position, bool):
            raise ValueError(
                "actor_observe_motion_anchor_position must be boolean"
            )
        self.actor_observe_motion_anchor_position = (
            actor_observe_motion_anchor_position
        )
        if (
            isinstance(actor_joint_velocity_noise_scale, bool)
            or not np.isfinite(actor_joint_velocity_noise_scale)
            or actor_joint_velocity_noise_scale < 0.0
        ):
            raise ValueError(
                "actor_joint_velocity_noise_scale must be non-negative and finite"
            )
        self.actor_joint_velocity_noise_scale = float(
            actor_joint_velocity_noise_scale
        )
        if squash_actor_actions_override is not None and not isinstance(
            squash_actor_actions_override, bool
        ):
            raise ValueError(
                "squash_actor_actions_override must be boolean or None"
            )
        if not isinstance(actor_reference_lookahead_steps, tuple):
            raise ValueError(
                "actor reference lookahead steps must be a tuple"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in actor_reference_lookahead_steps
        ):
            raise ValueError(
                "actor reference lookahead steps must be positive integers"
            )
        if tuple(sorted(set(actor_reference_lookahead_steps))) != (
            actor_reference_lookahead_steps
        ):
            raise ValueError(
                "actor reference lookahead steps must be strictly increasing"
            )
        self.actor_reference_lookahead_steps = (
            actor_reference_lookahead_steps
        )
        if actor_reference_preview_mode not in {"absolute", "delta"}:
            raise ValueError(
                "actor reference preview mode must be 'absolute' or 'delta'"
            )
        if (
            actor_reference_preview_mode == "delta"
            and not actor_reference_lookahead_steps
        ):
            raise ValueError("delta preview mode requires lookahead steps")
        self.actor_reference_preview_mode = actor_reference_preview_mode
        if physics_substeps < 1:
            raise ValueError("physics_substeps must be at least one")
        if reference_stride < 1:
            raise ValueError("reference_stride must be at least one")
        if reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        if actor_joint_order not in ("model", "source"):
            raise ValueError("actor_joint_order must be 'model' or 'source'")
        if physics_timestep is not None and physics_timestep <= 0.0:
            raise ValueError("physics_timestep must be positive")
        if solver_iterations < 1 or solver_ls_iterations < 1:
            raise ValueError("solver iteration counts must be positive")
        if (
            not np.isfinite(effort_limit_scale)
            or effort_limit_scale <= 0.0
        ):
            raise ValueError(
                "effort_limit_scale must be positive and finite"
            )
        self.effort_limit_scale = float(effort_limit_scale)
        if (
            isinstance(action_magnitude_weight, bool)
            or not np.isfinite(action_magnitude_weight)
            or action_magnitude_weight < 0.0
        ):
            raise ValueError(
                "action_magnitude_weight must be non-negative and finite"
            )
        self.action_magnitude_weight = float(action_magnitude_weight)
        if tracking_velocity_kernel not in {"exponential", "pseudo_huber"}:
            raise ValueError(
                "tracking_velocity_kernel must be 'exponential' or "
                "'pseudo_huber'"
            )
        self.tracking_velocity_kernel = tracking_velocity_kernel
        if tracking_anchor_position_kernel not in {
            "exponential",
            "dual_scale",
            "quadratic",
        }:
            raise ValueError(
                "tracking_anchor_position_kernel must be 'exponential', "
                "'dual_scale', or 'quadratic'"
            )
        self.tracking_anchor_position_kernel = (
            tracking_anchor_position_kernel
        )
        if (
            isinstance(tracking_torso_orientation_weight, bool)
            or not np.isfinite(tracking_torso_orientation_weight)
            or tracking_torso_orientation_weight < 0.0
        ):
            raise ValueError(
                "tracking_torso_orientation_weight must be non-negative and finite"
            )
        self.tracking_torso_orientation_weight = float(
            tracking_torso_orientation_weight
        )
        if (
            isinstance(tracking_root_velocity_weight, bool)
            or not np.isfinite(tracking_root_velocity_weight)
            or tracking_root_velocity_weight < 0.0
        ):
            raise ValueError(
                "tracking_root_velocity_weight must be non-negative and finite"
            )
        self.tracking_root_velocity_weight = float(
            tracking_root_velocity_weight
        )
        if (
            isinstance(termination_margin_weight, bool)
            or not np.isfinite(termination_margin_weight)
            or termination_margin_weight < 0.0
        ):
            raise ValueError(
                "termination_margin_weight must be non-negative and finite"
            )
        self.termination_margin_weight = float(termination_margin_weight)
        if (
            isinstance(reference_reset_noise_scale, bool)
            or not np.isfinite(reference_reset_noise_scale)
            or reference_reset_noise_scale < 0.0
        ):
            raise ValueError(
                "reference_reset_noise_scale must be non-negative and finite"
            )
        self.reference_reset_noise_scale = float(reference_reset_noise_scale)
        if (
            isinstance(reference_root_reset_noise_multiplier, bool)
            or not np.isfinite(reference_root_reset_noise_multiplier)
            or reference_root_reset_noise_multiplier < 1.0
        ):
            raise ValueError(
                "reference root reset noise multiplier must be finite and at least one"
            )
        if (
            isinstance(reference_root_reset_noise_probability, bool)
            or not np.isfinite(reference_root_reset_noise_probability)
            or not 0.0 <= reference_root_reset_noise_probability <= 1.0
        ):
            raise ValueError(
                "reference root reset noise probability must be finite and in [0, 1]"
            )
        self.reference_root_reset_noise_multiplier = float(
            reference_root_reset_noise_multiplier
        )
        self.reference_root_reset_noise_probability = float(
            reference_root_reset_noise_probability
        )
        if not isinstance(reference_residual_control, bool):
            raise ValueError("reference_residual_control must be boolean")
        if (
            isinstance(reference_residual_scale, bool)
            or not np.isfinite(reference_residual_scale)
            or reference_residual_scale <= 0.0
        ):
            raise ValueError(
                "reference_residual_scale must be positive and finite"
            )
        self.reference_residual_control = reference_residual_control
        self.reference_residual_scale = float(reference_residual_scale)
        if not isinstance(adaptive_phase_sampling, bool):
            raise ValueError("adaptive_phase_sampling must be boolean")
        if (
            isinstance(adaptive_phase_uniform_ratio, bool)
            or not np.isfinite(adaptive_phase_uniform_ratio)
            or not 0.0 <= adaptive_phase_uniform_ratio <= 1.0
        ):
            raise ValueError(
                "adaptive_phase_uniform_ratio must be finite and in [0, 1]"
            )
        self.adaptive_phase_sampling = adaptive_phase_sampling
        self.adaptive_phase_uniform_ratio = float(
            adaptive_phase_uniform_ratio
        )
        if (
            isinstance(carried_reset_probability, bool)
            or not np.isfinite(carried_reset_probability)
            or not 0.0 <= carried_reset_probability <= 1.0
        ):
            raise ValueError(
                "carried_reset_probability must be finite and in [0, 1]"
            )
        if (
            isinstance(carried_reset_bank_start, bool)
            or not isinstance(carried_reset_bank_start, int)
            or carried_reset_bank_start < 0
        ):
            raise ValueError(
                "carried_reset_bank_start must be a non-negative integer"
            )
        if carried_reset_probability > 0.0 and carried_reset_bank_path is None:
            raise ValueError(
                "carried_reset_bank_path is required when "
                "carried_reset_probability is positive"
            )
        if carried_reset_bank_path is not None and carried_reset_probability == 0.0:
            raise ValueError(
                "carried_reset_probability must be positive when "
                "carried_reset_bank_path is set"
            )
        self.carried_reset_bank_path = (
            None
            if carried_reset_bank_path is None
            else str(Path(carried_reset_bank_path).resolve())
        )
        self.carried_reset_probability = float(carried_reset_probability)
        self.carried_reset_bank_start = carried_reset_bank_start
        self.carried_reset_bank_size = 0
        self.carried_reset_qpos = None
        self.carried_reset_qvel = None
        self.carried_reset_phase = None
        self.carried_reset_last_act = None
        self.carried_reset_actor_obs_history = None
        self.carried_reset_restores_actor_context = False
        if not isinstance(domain_randomization, bool):
            raise ValueError("domain_randomization must be boolean")
        self.domain_randomization = domain_randomization
        if not isinstance(randomization_uses_curriculum, bool):
            raise ValueError("randomization_uses_curriculum must be boolean")
        self.randomization_uses_curriculum = randomization_uses_curriculum

        def positive_pair(name, values):
            array = np.asarray(values, dtype=np.float64)
            if (
                array.shape != (2,)
                or not np.isfinite(array).all()
                or np.any(array <= 0.0)
                or array[0] > array[1]
            ):
                raise ValueError(
                    f"{name} must be an ordered pair of positive finite values"
                )
            return array

        friction_values = positive_pair("friction_range", friction_range)
        mass_values = positive_pair("mass_range", mass_range)
        kp_values = positive_pair("kp_range", kp_range)
        kd_values = positive_pair("kd_range", kd_range)
        com_values = np.asarray(com_offset_range, dtype=np.float64)
        if (
            com_values.shape != (3,)
            or not np.isfinite(com_values).all()
            or np.any(com_values < 0.0)
        ):
            raise ValueError(
                "com_offset_range must contain three non-negative finite values"
            )
        if not self.domain_randomization and mass_values[0] != mass_values[1]:
            raise ValueError(
                "mass_range must be equal when domain_randomization is disabled"
            )
        self.body_mass_scale = (
            1.0 if self.domain_randomization else float(mass_values[0])
        )
        self.randomization_ranges = G1RandomizationRanges(
            friction=tuple(float(value) for value in friction_values),
            mass=tuple(float(value) for value in mass_values),
            kp_scale=tuple(float(value / 35.0) for value in kp_values),
            kd_scale=tuple(float(value / 0.5) for value in kd_values),
            com_offset=tuple(float(value) for value in com_values),
        )

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
        if physics_timestep is not None:
            self.mj_model.opt.timestep = physics_timestep
        self.mj_model.opt.iterations = solver_iterations
        self.mj_model.opt.ls_iterations = solver_ls_iterations
        self.mj_model.geom_margin[:] = 0.0
        self.mj_model.body_mass[1:] *= self.body_mass_scale
        self.mj_model.body_inertia[1:] *= self.body_mass_scale
        self.mjx_model = mjx.put_model(self.mj_model)
        self.base_friction = self.mjx_model.geom_friction
        self.base_mass = self.mjx_model.body_mass
        self.base_inertia = self.mjx_model.body_inertia
        self.base_ipos = self.mjx_model.body_ipos

        self.controller = load_rmr_controller(
            self.mj_model, self.controller_path
        )
        self.reference = load_mujoco_reference(
            self.mj_model,
            self.reference_path,
            RMR_G1_BODY_NAMES,
            controller=self.controller,
        )
        self.reference_length = self.reference.qpos.shape[0]
        from src.envs.g1_tracking.training_distribution import (
            init_phase_sampler,
        )

        self.phase_sampler_initial_failed_count = init_phase_sampler(
            self.reference_length
        ).failed_count
        self.body_ids = tuple(self.reference.body_ids)
        self.anchor_body_id = self.body_ids[0]
        self.pelvis_body_id = self.anchor_body_id
        self.root_body_id = int(
            self.mj_model.body_rootid[self.anchor_body_id]
        )
        reference_momentum = reference_centroidal_momentum(
            self.mj_model,
            self.reference.qpos,
            self.reference.qvel,
            self.root_body_id,
        )
        self.reference_centroidal_momentum = jp.asarray(reference_momentum)
        self.reference_capture_point = jp.asarray(
            reference_capture_points(
                self.mj_model,
                self.reference.qpos,
                self.reference.qvel,
                self.root_body_id,
            )
        )
        self.standing_com_height = standing_com_height(
            self.mj_model, self.mj_model.qpos0, self.root_body_id
        )
        centroidal_mass = float(
            self.mj_model.body_subtreemass[self.root_body_id]
        )
        centroidal_gravity = float(
            np.linalg.norm(self.mj_model.opt.gravity)
        )
        self.centroidal_gravity = centroidal_gravity
        velocity_scale = math.sqrt(
            centroidal_gravity * self.standing_com_height
        )
        self.centroidal_linear_scale = centroidal_mass * velocity_scale
        self.centroidal_angular_scale = (
            centroidal_mass * self.standing_com_height * velocity_scale
        )
        if (
            not np.isfinite(self.reference_centroidal_momentum).all()
            or not np.isfinite(self.reference_capture_point).all()
            or not math.isfinite(self.centroidal_linear_scale)
            or not math.isfinite(self.centroidal_angular_scale)
            or self.centroidal_linear_scale <= 0.0
            or self.centroidal_angular_scale <= 0.0
        ):
            raise ValueError(
                "centroidal reference and normalization must be finite"
            )
        if not isinstance(randomization_com_body_name, str) or not (
            randomization_com_body_name.strip()
        ):
            raise ValueError("randomization_com_body_name must name a body")
        try:
            self.randomization_com_body_id = int(
                self.mj_model.body(randomization_com_body_name).id
            )
        except KeyError as error:
            raise ValueError(
                "randomization_com_body_name must identify a model body"
            ) from error
        self.randomization_com_body_name = randomization_com_body_name
        self.distal_body_slots = (3, 6, 10, 13)

        self.qpos_reference = jp.asarray(self.reference.qpos)
        self.qvel_reference = jp.asarray(self.reference.qvel)
        carried_context = None
        if self.carried_reset_bank_path is not None:
            bank_path = Path(self.carried_reset_bank_path)
            if not bank_path.is_file():
                raise ValueError(
                    f"carried_reset_bank_path does not exist: {bank_path}"
                )
            try:
                with np.load(bank_path, allow_pickle=False) as archive:
                    carried_qpos = np.asarray(archive["qpos"], dtype=np.float64)
                    carried_qvel = np.asarray(archive["qvel"], dtype=np.float64)
                    carried_phase_raw = np.asarray(archive["phase"])
                    context_presence = (
                        "last_act" in archive,
                        "actor_obs_history" in archive,
                    )
                    if all(context_presence):
                        carried_context = (
                            np.asarray(archive["last_act"], dtype=np.float64),
                            np.asarray(
                                archive["actor_obs_history"],
                                dtype=np.float64,
                            ),
                        )
            except (KeyError, OSError, ValueError) as error:
                raise ValueError(
                    "carried reset bank must be a readable NPZ with "
                    "qpos, qvel, and phase"
                ) from error
            if any(context_presence) and not all(context_presence):
                raise ValueError(
                    "carried reset bank actor context requires both "
                    "last_act and actor_obs_history"
                )
            if carried_phase_raw.ndim != 1:
                raise ValueError("carried reset bank phase must be a vector")
            carried_phase = carried_phase_raw.astype(np.int32)
            if not np.array_equal(carried_phase_raw, carried_phase):
                raise ValueError("carried reset bank phase must be integer-valued")
            if (
                carried_qpos.ndim != 2
                or carried_qpos.shape[1] != self.mj_model.nq
                or carried_qvel.ndim != 2
                or carried_qvel.shape[1] != self.mj_model.nv
                or carried_qpos.shape[0] != carried_qvel.shape[0]
                or carried_qpos.shape[0] != carried_phase.shape[0]
            ):
                raise ValueError(
                    "carried reset bank qpos/qvel/phase shapes do not align"
                )
            if (
                carried_qpos.shape[0] == 0
                or not np.isfinite(carried_qpos).all()
                or not np.isfinite(carried_qvel).all()
            ):
                raise ValueError(
                    "carried reset bank states must be nonempty and finite"
                )
            if np.any(carried_phase < 0) or np.any(
                carried_phase >= self.reference_length - 1
            ):
                raise ValueError(
                    "carried reset bank phase lies outside the training range"
                )
            quaternion_norm = np.linalg.norm(carried_qpos[:, 3:7], axis=1)
            if not np.allclose(quaternion_norm, 1.0, atol=1e-5, rtol=0.0):
                raise ValueError(
                    "carried reset bank root quaternions must be normalized"
                )
            if self.carried_reset_bank_start >= carried_qpos.shape[0]:
                raise ValueError(
                    "carried_reset_bank_start must leave at least one state"
                )
            bank_slice = slice(self.carried_reset_bank_start, None)
            self.carried_reset_qpos = jp.asarray(carried_qpos[bank_slice])
            self.carried_reset_qvel = jp.asarray(carried_qvel[bank_slice])
            self.carried_reset_phase = jp.asarray(carried_phase[bank_slice])
            self.carried_reset_bank_size = int(
                self.carried_reset_qpos.shape[0]
            )
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
        self.effort_limit = jp.asarray(
            self.controller.effort_limit * self.effort_limit_scale
        )
        self.default_joints = jp.asarray(
            self.controller.default_joint_pos
        )
        self.action_scales = jp.asarray(self.controller.action_scale)
        if actor_joint_order == "source":
            actor_to_model = self.controller.actor_to_model_permutation
            model_to_actor = self.controller.model_to_actor_permutation
            self.actor_joint_names = self.controller.actor_joint_names
        else:
            actor_to_model = np.arange(29, dtype=np.int32)
            model_to_actor = actor_to_model
            self.actor_joint_names = self.controller.joint_names
        self.actor_to_model_permutation = jp.asarray(actor_to_model)
        self.model_to_actor_permutation = jp.asarray(model_to_actor)
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

        self.n_frames = physics_substeps
        self.reference_stride = reference_stride
        self.reward_scale = reward_scale
        self.clip_actions = clip_actions
        self.squash_actor_actions = (
            clip_actions or self.reference_residual_control
            if squash_actor_actions_override is None
            else squash_actor_actions_override
        )
        # Keep the learned-mean and sampled-action boundaries independently
        # configurable.  Legacy environments intentionally share one value.
        self.squash_actor_mean = self.squash_actor_actions
        self.clip_sampled_actor_actions = self.squash_actor_actions
        self.dt = float(self.mj_model.opt.timestep * self.n_frames)
        self.control_reference_dt = self.dt
        if self.reference.fps is not None:
            self.control_reference_dt = (
                self.reference_stride / self.reference.fps
            )
            if not math.isclose(
                self.control_reference_dt,
                self.dt,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "reference timebase does not match environment control "
                    f"period: stride/fps={self.control_reference_dt}, "
                    f"dt={self.dt}"
                )
        self.reference_transitions = math.ceil(
            (self.reference_length - 1) / self.reference_stride
        )
        self.max_episode_length = self.reference_transitions
        self.action_dim = 29
        self.actor_history_len = actor_history_len
        self.actor_future_reference_dim = (
            58 * len(self.actor_reference_lookahead_steps)
        )
        self.actor_frame_obs_dim = (
            154
            + 3 * self.actor_observe_motion_anchor_position
            + self.actor_future_reference_dim
        )
        self.actor_obs_dim = self.actor_frame_obs_dim * actor_history_len
        if carried_context is not None:
            carried_last_act, carried_actor_obs_history = carried_context
            expected_rows = int(carried_phase.shape[0])
            if (
                carried_last_act.shape != (expected_rows, self.action_dim)
                or carried_actor_obs_history.shape
                != (
                    expected_rows,
                    self.actor_history_len,
                    self.actor_frame_obs_dim,
                )
                or not np.isfinite(carried_last_act).all()
                or not np.isfinite(carried_actor_obs_history).all()
            ):
                raise ValueError(
                    "carried reset bank actor context has invalid shape or "
                    "non-finite values"
                )
            bank_slice = slice(self.carried_reset_bank_start, None)
            self.carried_reset_last_act = jp.asarray(
                carried_last_act[bank_slice]
            )
            self.carried_reset_actor_obs_history = jp.asarray(
                carried_actor_obs_history[bank_slice]
            )
            self.carried_reset_restores_actor_context = True
        self.actor_noise_mask = jp.concatenate(
            (
                jp.zeros(58),
                jp.zeros(3 * self.actor_observe_motion_anchor_position),
                jp.full(6, 0.05),
                jp.full(3, 0.2),
                jp.full(29, 0.01),
                jp.full(29, self.actor_joint_velocity_noise_scale),
                jp.zeros(29),
                jp.zeros(self.actor_future_reference_dim),
            )
        )
        self.critic_obs_dim = 286

        # Compatibility fields consumed by the unchanged Open-DiffLoco SHAC
        # rollout. Terrain/disturbance amplitudes remain zero in the registered
        # first task discriminator.
        left_foot = self.body_ids[3]
        right_foot = self.body_ids[6]
        self._support_foot_body_ids = jp.asarray(
            [left_foot, right_foot], dtype=jp.int32
        )
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

    def foot_support_signature(self, data: mjx.Data) -> jax.Array:
        """Return grouped active-contact support for left and right feet."""

        contact = data._impl.contact
        return grouped_foot_support(
            contact.geom,
            contact.dist,
            self.mjx_model.geom_bodyid,
            self._support_foot_body_ids,
        )

    def contact_pair_signature(self, data: mjx.Data) -> jax.Array:
        """Return active contacts grouped by unordered model-body pair."""

        contact = data._impl.contact
        return grouped_body_pair_contacts(
            contact.geom,
            contact.dist,
            self.mjx_model.geom_bodyid,
            body_count=self.mj_model.nbody,
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
        reward, components = rmr_tracking_reward(
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
            velocity_kernel=self.tracking_velocity_kernel,
            anchor_position_kernel=self.tracking_anchor_position_kernel,
        )
        if self.tracking_torso_orientation_weight > 0.0:
            torso_orientation = torso_orientation_tracking_reward(
                target_body_quat, body_quat
            )
            reward = (
                reward
                + self.tracking_torso_orientation_weight * torso_orientation
            )
            components = {**components, "torso_orientation": torso_orientation}
        if self.tracking_root_velocity_weight > 0.0:
            root_linear_velocity, root_angular_velocity = (
                root_velocity_tracking_reward(
                    self.body_lin_vel_reference[phase],
                    body_lin_vel,
                    self.body_ang_vel_reference[phase],
                    body_ang_vel,
                )
            )
            reward = reward + self.tracking_root_velocity_weight * 0.5 * (
                root_linear_velocity + root_angular_velocity
            )
            components = {
                **components,
                "root_linear_velocity": root_linear_velocity,
                "root_angular_velocity": root_angular_velocity,
            }
        return reward, components

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
        anchor_position, anchor_orientation = self._anchor_relative_reference(
            data, phase
        )
        root_inverse = _quat_inv(data.qpos[3:7])
        actor_order = self.model_to_actor_permutation
        return jp.concatenate(
            (
                self.qpos_reference[phase, 7:][actor_order],
                self.qvel_reference[phase, 6:][actor_order],
                *(
                    (anchor_position,)
                    if self.actor_observe_motion_anchor_position
                    else ()
                ),
                _rotation_6d(anchor_orientation),
                _quat_apply(root_inverse, data.qvel[3:6]),
                (data.qpos[7:] - self.default_joints)[actor_order],
                data.qvel[6:][actor_order],
                info["last_act"][actor_order],
                self._future_reference_command(phase),
            )
        )

    def _future_reference_command(self, phase: jax.Array) -> jax.Array:
        """Return clamped future joint commands in declared offset order."""
        if not self.actor_reference_lookahead_steps:
            return jp.zeros((0,), dtype=self.qpos_reference.dtype)
        offsets = jp.asarray(
            self.actor_reference_lookahead_steps, dtype=jp.int32
        )
        indices = jp.minimum(
            jp.asarray(phase, dtype=jp.int32)
            + offsets * self.reference_stride,
            self.reference_length - 1,
        )
        actor_order = self.model_to_actor_permutation
        commands = jp.concatenate(
            (
                self.qpos_reference[indices, 7:][:, actor_order],
                self.qvel_reference[indices, 6:][:, actor_order],
            ),
            axis=-1,
        )
        if self.actor_reference_preview_mode == "delta":
            current = jp.concatenate(
                (
                    self.qpos_reference[phase, 7:][actor_order],
                    self.qvel_reference[phase, 6:][actor_order],
                )
            )
            commands = commands - current[None, :]
        return commands.reshape(-1)

    def _get_critic_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        phase = info["phase"]
        anchor_pos, anchor_orientation = self._anchor_relative_reference(
            data, phase
        )
        body_pos, body_quat, _, _ = self._body_state(data)
        actual_anchor_pos = body_pos[0]
        actual_anchor_quat = body_quat[0]
        inverse_anchor = _quat_inv(actual_anchor_quat)
        actor_order = self.model_to_actor_permutation
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
                self.qpos_reference[phase, 7:][actor_order],
                self.qvel_reference[phase, 6:][actor_order],
                anchor_pos,
                _rotation_6d(anchor_orientation),
                body_pos_b.reshape(-1),
                _rotation_6d(body_quat_b).reshape(-1),
                _quat_apply(root_inverse, data.qvel[:3]),
                _quat_apply(root_inverse, data.qvel[3:6]),
                (data.qpos[7:] - self.default_joints)[actor_order],
                data.qvel[6:][actor_order],
                info["last_act"][actor_order],
            )
        )

    def _base_info(
        self,
        *,
        rng: jax.Array,
        phase: jax.Array,
        difficulty: jax.Array,
        randomization: dict[str, jax.Array] | None = None,
        phase_sampler_failed_count: jax.Array | None = None,
    ) -> dict:
        if randomization is None:
            randomization = self._nominal_randomization()
        info = {
            "step": jp.array(0, dtype=jp.int32),
            "phase": phase.astype(jp.int32),
            "last_act": jp.zeros(self.action_dim),
            "rng": rng,
            "difficulty": difficulty,
            "foot_bump_ou": jp.zeros((4, 3)),
            "foot_normal_forces": jp.zeros(4),
            "terminal": jp.array(0.0),
            "transition_contact_stiffness": jp.array(0.0),
            "transition_contact_topology_event": jp.array(False),
            "reset_was_carried": jp.array(False),
            **randomization,
        }
        if self.adaptive_phase_sampling:
            failed_count = (
                self.phase_sampler_initial_failed_count
                if phase_sampler_failed_count is None
                else jp.asarray(phase_sampler_failed_count, dtype=jp.float32)
            )
            if failed_count.shape != self.phase_sampler_initial_failed_count.shape:
                raise ValueError(
                    "phase_sampler_failed_count has the wrong shape"
                )
            info["phase_sampler_failed_count"] = failed_count
        return info

    @staticmethod
    def _nominal_randomization() -> dict[str, jax.Array]:
        return {
            "friction_scale": jp.array(1.0),
            "mass_scale": jp.array(1.0),
            "kp_scale": jp.array(1.0),
            "kd_scale": jp.array(1.0),
            "com_offset": jp.zeros(3),
        }

    def _sample_randomization(
        self,
        key: jax.Array,
        difficulty: jax.Array,
    ) -> dict[str, jax.Array]:
        if not self.domain_randomization:
            return self._nominal_randomization()
        sample_difficulty = (
            difficulty
            if self.randomization_uses_curriculum
            else jp.ones_like(difficulty)
        )
        return sample_g1_randomization(
            key,
            sample_difficulty,
            self.randomization_ranges,
        )

    def _get_randomized_model(self, info: dict):
        """Materialize one environment's MJX model from carried parameters."""
        if not self.domain_randomization:
            return self.mjx_model
        body_ipos = self.base_ipos.at[self.randomization_com_body_id].add(
            info["com_offset"]
        )
        return self.mjx_model.replace(
            geom_friction=self.base_friction * info["friction_scale"],
            body_mass=self.base_mass * info["mass_scale"],
            body_inertia=self.base_inertia * info["mass_scale"],
            body_ipos=body_ipos,
        )

    def _data_from_state(
        self,
        *,
        qpos: jax.Array,
        qvel: jax.Array,
        randomization: dict[str, jax.Array],
    ) -> mjx.Data:
        model = self._get_randomized_model(randomization)
        data = mjx.make_data(model).replace(qpos=qpos, qvel=qvel)
        return mjx.forward(model, data)

    def _init_metrics(self) -> dict:
        zero = jp.float32(0.0)
        return {
            "contact_force": zero,
            "contact_stiffness": zero,
            "anchor_position_error": zero,
            "anchor_orientation_error": zero,
            "body_position_error": zero,
            "body_orientation_error": zero,
            "body_linear_velocity_error": zero,
            "body_angular_velocity_error": zero,
            "termination_anchor_z_error": zero,
            "termination_anchor_xy_error": zero,
            "termination_gravity_z_error": zero,
            "termination_distal_z_error": zero,
            "rew_anchor_position": zero,
            "rew_anchor_orientation": zero,
            "rew_body_position": zero,
            "rew_body_orientation": zero,
            "rew_body_linear_velocity": zero,
            "rew_body_angular_velocity": zero,
            "rew_torso_orientation": zero,
            "rew_root_linear_velocity": zero,
            "rew_root_angular_velocity": zero,
            "rew_action_rate": zero,
            "rew_action_magnitude": zero,
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

    def _initial_state_from_data(
        self,
        *,
        data: mjx.Data,
        rng: jax.Array,
        difficulty: jax.Array,
        phase: jax.Array,
        randomization: dict[str, jax.Array] | None = None,
        phase_sampler_failed_count: jax.Array | None = None,
        last_act: jax.Array | None = None,
        actor_obs_history: jax.Array | None = None,
    ) -> EnvState:
        info = self._base_info(
            rng=rng,
            phase=phase,
            difficulty=difficulty,
            randomization=randomization,
            phase_sampler_failed_count=phase_sampler_failed_count,
        )
        if last_act is not None:
            info = {**info, "last_act": jp.asarray(last_act)}
        actor_frame = self._get_actor_obs(data, info)
        actor_history = (
            jp.repeat(actor_frame[None, :], self.actor_history_len, axis=0)
            if actor_obs_history is None
            else jp.asarray(actor_obs_history)
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
        return self._initial_state_from_data(
            data=data,
            rng=rng,
            difficulty=difficulty,
            phase=phase,
        )

    def _sample_reset_phase(
        self,
        key: jax.Array,
        phase_sampler_failed_count: jax.Array | None,
    ) -> jax.Array:
        if not self.adaptive_phase_sampling:
            return jax.random.randint(
                key,
                (),
                minval=0,
                maxval=self.reference_length - 2,
                dtype=jp.int32,
            )
        from src.envs.g1_tracking.training_distribution import (
            PhaseSamplerState,
            sample_training_phase,
        )

        failed_count = (
            self.phase_sampler_initial_failed_count
            if phase_sampler_failed_count is None
            else jp.asarray(phase_sampler_failed_count, dtype=jp.float32)
        )
        if failed_count.shape != self.phase_sampler_initial_failed_count.shape:
            raise ValueError("phase_sampler_failed_count has the wrong shape")
        return sample_training_phase(
            key,
            PhaseSamplerState(failed_count=failed_count),
            self.reference_length,
            uniform_ratio=self.adaptive_phase_uniform_ratio,
        )

    def _noisy_reference_state(
        self,
        phase: jax.Array,
        pose_key: jax.Array,
        velocity_key: jax.Array,
        joint_key: jax.Array,
        root_multiplier: jax.Array | float = 1.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Perturb one reference state using the registered reset contract."""
        scale = self.reference_reset_noise_scale
        pose_limit = scale * root_multiplier * jp.array(
            [0.02, 0.02, 0.005, 0.1, 0.1, 0.1]
        )
        velocity_limit = scale * root_multiplier * jp.array(
            [0.25, 0.25, 0.1, 0.26, 0.26, 0.39]
        )
        pose_delta = jax.random.uniform(
            pose_key, (6,), minval=-pose_limit, maxval=pose_limit
        )
        velocity_delta = jax.random.uniform(
            velocity_key,
            (6,),
            minval=-velocity_limit,
            maxval=velocity_limit,
        )
        joint_delta = jax.random.uniform(
            joint_key, (29,), minval=-0.05 * scale, maxval=0.05 * scale
        )
        qpos = self.qpos_reference[phase]
        root_quat = _quat_mul(
            _quat_from_euler_xyz(pose_delta[3:]), qpos[3:7]
        )
        root_quat = root_quat / jp.linalg.norm(root_quat)
        qpos = qpos.at[:3].add(pose_delta[:3])
        qpos = qpos.at[3:7].set(root_quat)
        qpos = qpos.at[7:].set(
            jp.clip(
                qpos[7:] + joint_delta,
                self.soft_joint_lower,
                self.soft_joint_upper,
            )
        )
        qvel = self.qvel_reference[phase].at[:6].add(velocity_delta)
        return qpos, qvel

    def reset(
        self,
        rng: jax.Array,
        difficulty: jax.Array,
        phase_sampler_failed_count: jax.Array | None = None,
    ) -> EnvState:
        if self.carried_reset_bank_path is not None:
            noisy_fallback = self.reference_reset_noise_scale > 0.0
            root_treatment = (
                noisy_fallback
                and self.reference_root_reset_noise_probability > 0.0
            )
            if self.domain_randomization and noisy_fallback:
                keys = jax.random.split(rng, 9 if root_treatment else 8)
                (
                    rng,
                    phase_key,
                    bank_key,
                    choice_key,
                    pose_key,
                    velocity_key,
                    joint_key,
                    randomization_key,
                ) = keys[:8]
                root_cohort_key = keys[8] if root_treatment else None
                randomization = self._sample_randomization(
                    randomization_key, difficulty
                )
            elif self.domain_randomization:
                rng, phase_key, bank_key, choice_key, randomization_key = (
                    jax.random.split(rng, 5)
                )
                randomization = self._sample_randomization(
                    randomization_key, difficulty
                )
            elif noisy_fallback:
                keys = jax.random.split(rng, 8 if root_treatment else 7)
                (
                    rng,
                    phase_key,
                    bank_key,
                    choice_key,
                    pose_key,
                    velocity_key,
                    joint_key,
                ) = keys[:7]
                root_cohort_key = keys[7] if root_treatment else None
                randomization = self._nominal_randomization()
            else:
                rng, phase_key, bank_key, choice_key = jax.random.split(
                    rng, 4
                )
                randomization = self._nominal_randomization()
            reference_phase = self._sample_reset_phase(
                phase_key, phase_sampler_failed_count
            )
            bank_index = jax.random.randint(
                bank_key,
                (),
                minval=0,
                maxval=self.carried_reset_bank_size,
                dtype=jp.int32,
            )
            use_carried = jax.random.bernoulli(
                choice_key, self.carried_reset_probability
            )
            if noisy_fallback:
                root_multiplier = (
                    jp.where(
                        jax.random.bernoulli(
                            root_cohort_key,
                            self.reference_root_reset_noise_probability,
                        ),
                        self.reference_root_reset_noise_multiplier,
                        1.0,
                    )
                    if root_treatment
                    else 1.0
                )
                reference_qpos, reference_qvel = self._noisy_reference_state(
                    reference_phase,
                    pose_key,
                    velocity_key,
                    joint_key,
                    root_multiplier=root_multiplier,
                )
            else:
                reference_qpos = self.qpos_reference[reference_phase]
                reference_qvel = self.qvel_reference[reference_phase]
            phase = jp.where(
                use_carried,
                self.carried_reset_phase[bank_index],
                reference_phase,
            )
            qpos = jp.where(
                use_carried,
                self.carried_reset_qpos[bank_index],
                reference_qpos,
            )
            qvel = jp.where(
                use_carried,
                self.carried_reset_qvel[bank_index],
                reference_qvel,
            )
            data = self._data_from_state(
                qpos=qpos,
                qvel=qvel,
                randomization=randomization,
            )
            last_act = (
                jp.where(
                    use_carried,
                    self.carried_reset_last_act[bank_index],
                    jp.zeros(self.action_dim),
                )
                if self.carried_reset_restores_actor_context
                else None
            )
            state = self._initial_state_from_data(
                data=data,
                rng=rng,
                difficulty=difficulty,
                phase=phase,
                randomization=randomization,
                phase_sampler_failed_count=phase_sampler_failed_count,
                last_act=last_act,
            )
            state = state.replace(
                info={**state.info, "reset_was_carried": use_carried}
            )
            if not self.carried_reset_restores_actor_context:
                return state
            actor_history = jp.where(
                use_carried,
                self.carried_reset_actor_obs_history[bank_index],
                state.info["actor_obs_history"],
            )
            info = {
                **state.info,
                "actor_obs_history": actor_history,
                "bootstrap_obs": actor_history.reshape(-1),
            }
            return state.replace(obs=actor_history.reshape(-1), info=info)
        if self.reference_reset_noise_scale == 0.0:
            if self.domain_randomization:
                rng, phase_key, randomization_key = jax.random.split(rng, 3)
                randomization = self._sample_randomization(
                    randomization_key, difficulty
                )
            else:
                rng, phase_key = jax.random.split(rng)
                randomization = self._nominal_randomization()
            phase = self._sample_reset_phase(
                phase_key, phase_sampler_failed_count
            )
            if not self.domain_randomization:
                if not self.adaptive_phase_sampling:
                    return self.reset_at_phase(rng, difficulty, phase)
                data = self._data_from_state(
                    qpos=self.qpos_reference[phase],
                    qvel=self.qvel_reference[phase],
                    randomization=randomization,
                )
                return self._initial_state_from_data(
                    data=data,
                    rng=rng,
                    difficulty=difficulty,
                    phase=phase,
                    randomization=randomization,
                    phase_sampler_failed_count=phase_sampler_failed_count,
                )
            data = self._data_from_state(
                qpos=self.qpos_reference[phase],
                qvel=self.qvel_reference[phase],
                randomization=randomization,
            )
            return self._initial_state_from_data(
                data=data,
                rng=rng,
                difficulty=difficulty,
                phase=phase,
                randomization=randomization,
                phase_sampler_failed_count=phase_sampler_failed_count,
            )

        root_treatment = self.reference_root_reset_noise_probability > 0.0
        if self.domain_randomization:
            keys = jax.random.split(rng, 7 if root_treatment else 6)
            (
                rng,
                phase_key,
                pose_key,
                velocity_key,
                joint_key,
                randomization_key,
            ) = keys[:6]
            root_cohort_key = keys[6] if root_treatment else None
            randomization = self._sample_randomization(
                randomization_key, difficulty
            )
        else:
            keys = jax.random.split(rng, 6 if root_treatment else 5)
            rng, phase_key, pose_key, velocity_key, joint_key = keys[:5]
            root_cohort_key = keys[5] if root_treatment else None
            randomization = self._nominal_randomization()
        phase = self._sample_reset_phase(
            phase_key, phase_sampler_failed_count
        )
        root_multiplier = (
            jp.where(
                jax.random.bernoulli(
                    root_cohort_key,
                    self.reference_root_reset_noise_probability,
                ),
                self.reference_root_reset_noise_multiplier,
                1.0,
            )
            if root_treatment
            else 1.0
        )
        qpos, qvel = self._noisy_reference_state(
            phase,
            pose_key,
            velocity_key,
            joint_key,
            root_multiplier=root_multiplier,
        )

        data = self._data_from_state(
            qpos=qpos,
            qvel=qvel,
            randomization=randomization,
        )
        return self._initial_state_from_data(
            data=data,
            rng=rng,
            difficulty=difficulty,
            phase=phase,
            randomization=randomization,
            phase_sampler_failed_count=phase_sampler_failed_count,
        )

    def termination_errors(
        self,
        *,
        phase: jax.Array,
        body_pos: jax.Array,
        body_quat: jax.Array,
    ) -> dict[str, jax.Array]:
        """Return the four RMR failure errors for rewards or constraints."""
        anchor_z_error = jp.abs(
            self.body_pos_reference[phase, 0, 2] - body_pos[0, 2]
        )
        anchor_xy_error = jp.linalg.norm(
            self.body_pos_reference[phase, 0, :2] - body_pos[0, :2]
        )
        world_down = jp.array([0.0, 0.0, -1.0])
        target_down = _quat_apply(
            _quat_inv(self.body_quat_reference[phase, 0]), world_down
        )
        actual_down = _quat_apply(_quat_inv(body_quat[0]), world_down)
        gravity_z_error = jp.abs(target_down[2] - actual_down[2])
        distal_z_error = jp.max(
            jp.abs(
                self.body_pos_reference[
                    phase, jp.array(self.distal_body_slots), 2
                ]
                - body_pos[jp.array(self.distal_body_slots), 2]
            )
        )
        return {
            "anchor_z_error": anchor_z_error,
            "anchor_xy_error": anchor_xy_error,
            "gravity_z_error": gravity_z_error,
            "distal_z_error": distal_z_error,
        }

    def _termination(
        self,
        data: mjx.Data,
        info: dict,
        body_pos: jax.Array,
        body_quat: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        phase = info["phase"]
        errors = self.termination_errors(
            phase=phase,
            body_pos=body_pos,
            body_quat=body_quat,
        )
        nan_failure = (
            jp.any(~jp.isfinite(data.qpos))
            | jp.any(~jp.isfinite(data.qvel))
        )
        terminal = (
            (errors["anchor_z_error"] > 0.25)
            | (errors["anchor_xy_error"] > 1.3)
            | (errors["gravity_z_error"] > 0.8)
            | (errors["distal_z_error"] > 0.4)
            | nan_failure
        ).astype(jp.float64)
        clip_end = (
            phase >= self.reference_length - self.reference_stride
        ).astype(jp.float64)
        return jp.maximum(terminal, clip_end), terminal

    @functools.partial(jax.checkpoint, static_argnums=(0,))
    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        previous_contact_pairs = self.contact_pair_signature(state.data)
        action = self._prepare_action(action)
        position_target = self.position_target(state, action, prepared=True)
        model = self._get_randomized_model(state.info)
        kp = self.kp * state.info["kp_scale"]
        kd = self.kd * state.info["kd_scale"]

        def physics_step(data, _):
            torque = jp.clip(
                kp * (position_target - data.qpos[7:])
                - kd * data.qvel[6:],
                -self.effort_limit,
                self.effort_limit,
            )
            applied = jp.zeros(self.mj_model.nv).at[6:].set(torque)
            return (
                mjx.step(
                    model,
                    data.replace(qfrc_applied=applied),
                ),
                None,
            )

        data, _ = jax.lax.scan(
            physics_step, state.data, None, length=self.n_frames
        )
        transition_contact_stiffness = contact_stiffness(
            data.qfrc_constraint,
            data.qacc,
        )
        next_phase = jp.minimum(
            state.info["phase"] + self.reference_stride,
            self.reference_length - 1,
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
                action_magnitude_weight=self.action_magnitude_weight,
            )
        )
        reward = reward + regularization_reward
        if self.termination_margin_weight > 0.0:
            phase = pre_reset_info["phase"]
            reward = reward + (
                self.termination_margin_weight
                * termination_margin_penalty(
                    **self.termination_errors(
                        phase=phase,
                        body_pos=body_pos,
                        body_quat=body_quat,
                    ),
                )
            )
        reward = self.reward_scale * reward
        components = {**components, **regularization_components}
        done, terminal = self._termination(
            data, pre_reset_info, body_pos, body_quat
        )
        transition_contact_pairs = self.contact_pair_signature(data)
        transition_contact_topology_event = contact_topology_event(
            previous_contact_pairs,
            transition_contact_pairs,
            done=done,
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
        if self.adaptive_phase_sampling:
            reset_state = self.reset(
                reset_key,
                state.info["difficulty"],
                phase_sampler_failed_count=state.info[
                    "phase_sampler_failed_count"
                ],
            )
        else:
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
            "transition_contact_stiffness": transition_contact_stiffness,
            "transition_contact_topology_event": (
                transition_contact_topology_event
            ),
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
        termination_errors = self.termination_errors(
            phase=next_phase,
            body_pos=body_pos,
            body_quat=body_quat,
        )
        contact_force = jp.sum(jp.abs(data.qfrc_constraint[:6]))
        metrics = {
            "contact_force": contact_force,
            "contact_stiffness": transition_contact_stiffness,
            "anchor_position_error": anchor_position_error,
            "anchor_orientation_error": anchor_orientation_error,
            "body_position_error": body_position_error,
            "body_orientation_error": body_orientation_error,
            "body_linear_velocity_error": body_linear_velocity_error,
            "body_angular_velocity_error": body_angular_velocity_error,
            "termination_anchor_z_error": termination_errors[
                "anchor_z_error"
            ],
            "termination_anchor_xy_error": termination_errors[
                "anchor_xy_error"
            ],
            "termination_gravity_z_error": termination_errors[
                "gravity_z_error"
            ],
            "termination_distal_z_error": termination_errors[
                "distal_z_error"
            ],
            "rew_anchor_position": components["anchor_position"],
            "rew_anchor_orientation": components["anchor_orientation"],
            "rew_body_position": components["body_position"],
            "rew_body_orientation": components["body_orientation"],
            "rew_body_linear_velocity": components["body_linear_velocity"],
            "rew_body_angular_velocity": components["body_angular_velocity"],
            "rew_torso_orientation": components.get(
                "torso_orientation", jp.array(0.0)
            ),
            "rew_root_linear_velocity": components.get(
                "root_linear_velocity", jp.array(0.0)
            ),
            "rew_root_angular_velocity": components.get(
                "root_angular_velocity", jp.array(0.0)
            ),
            "rew_action_rate": components["action_rate"],
            "rew_action_magnitude": components["action_magnitude"],
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

    def position_target(
        self,
        state: EnvState,
        action: jax.Array,
        *,
        prepared: bool = False,
    ) -> jax.Array:
        """Convert an actor action into a model-order joint position target."""
        if not prepared:
            action = self._prepare_action(action)
        if not self.reference_residual_control:
            return self.default_joints + action * self.action_scales
        phase = state.info["phase"]
        return (
            self.qpos_reference[phase, 7:]
            + self.reference_residual_scale * action * self.action_scales
        )

    def _prepare_action(self, action: jax.Array) -> jax.Array:
        action = action.astype(jp.float64)
        if self.clip_actions:
            action = jp.clip(action, -1.0, 1.0)
        return action[self.actor_to_model_permutation]

    def _apply_obs_noise(
        self, obs: jax.Array, rng: jax.Array
    ) -> jax.Array:
        """Apply bounded noise only to the actor's current-state fields."""
        if not self.actor_observation_noise:
            return obs
        noise_mask = jp.tile(self.actor_noise_mask, self.actor_history_len)
        return obs + jax.random.uniform(
            rng,
            obs.shape,
            minval=-noise_mask,
            maxval=noise_mask,
        )

    def normalize_actor_obs(
        self, normalizer, norm_state, obs: jax.Array
    ) -> jax.Array:
        frames = obs.reshape(*obs.shape[:-1], self.actor_history_len, -1)
        normalized = normalizer.normalize(norm_state, frames)
        return normalized.reshape(*obs.shape[:-1], self.actor_obs_dim)


class G1TrackingRMR50HzEnv(G1TrackingEnv):
    """RMR task at its native 50 Hz control and reward-manager timebase."""

    def __init__(self, *args, **kwargs):
        reference_stride = kwargs.pop("reference_stride", 2)
        super().__init__(
            *args,
            physics_substeps=10,
            reference_stride=reference_stride,
            reward_scale=0.02,
            **kwargs,
        )


class G1TrackingRMR50HzUnboundedEnv(G1TrackingRMR50HzEnv):
    """Native RMR timebase with the upstream linear Gaussian action support."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("clip_actions", None)
        kwargs.pop("actor_joint_order", None)
        super().__init__(
            *args,
            clip_actions=False,
            actor_joint_order="source",
            **kwargs,
        )


class G1TrackingRMR50HzSourceStepEnv(G1TrackingEnv):
    """Our RMR G1 model at the source task's 5 ms physics timestep."""

    def __init__(
        self,
        *args,
        solver_iterations: int = 1,
        solver_ls_iterations: int = 5,
        **kwargs,
    ):
        reference_stride = kwargs.pop("reference_stride", 2)
        for option in (
            "physics_substeps",
            "reward_scale",
            "clip_actions",
            "actor_joint_order",
            "physics_timestep",
        ):
            kwargs.pop(option, None)
        super().__init__(
            *args,
            physics_substeps=4,
            reference_stride=reference_stride,
            reward_scale=0.02,
            clip_actions=False,
            actor_joint_order="source",
            physics_timestep=0.005,
            solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations,
            **kwargs,
        )


class G1TrackingRMR50HzSourceStepRobustEnv(
    G1TrackingRMR50HzSourceStepEnv
):
    """Source timestep with the stable solver budget seen in MuJoCo RMR."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("solver_iterations", None)
        kwargs.pop("solver_ls_iterations", None)
        super().__init__(
            *args,
            solver_iterations=10,
            solver_ls_iterations=20,
            **kwargs,
        )


class G1TrackingRMR50HzActionParityEnv(G1TrackingRMR50HzSourceStepEnv):
    """Explicit RMR delta-action and policy-noise compatibility boundary."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("squash_actor_actions_override", None)
        kwargs.pop("actor_joint_velocity_noise_scale", None)
        kwargs.pop("randomization_com_body_name", None)
        kwargs.pop("randomization_uses_curriculum", None)
        super().__init__(
            *args,
            squash_actor_actions_override=False,
            actor_joint_velocity_noise_scale=0.5,
            randomization_com_body_name="torso_link",
            randomization_uses_curriculum=False,
            **kwargs,
        )


class G1TrackingRMR50HzDecoupledExplorationEnv(
    G1TrackingRMR50HzActionParityEnv
):
    """Bound the learned mean while retaining unclipped exploration noise."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.squash_actor_mean = True
        self.clip_sampled_actor_actions = False


class G1TrackingRMR50HzUpstreamBoundaryEnv(
    G1TrackingRMR50HzDecoupledExplorationEnv
):
    """Use Open-DiffLoco's bounded mean and clipped sampled actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_sampled_actor_actions = True


class G1TrackingRMR50HzUpstreamActionPenaltyEnv(
    G1TrackingRMR50HzUpstreamBoundaryEnv
):
    """Add Open-DiffLoco's direct action-magnitude regularizer."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("action_magnitude_weight", None)
        super().__init__(*args, action_magnitude_weight=0.05, **kwargs)


class G1TrackingRMR50HzValidatedEnv(
    G1TrackingRMR50HzSourceStepEnv
):
    """Smallest solver budget passing strict source-policy tracking."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("solver_iterations", None)
        kwargs.pop("solver_ls_iterations", None)
        super().__init__(
            *args,
            solver_iterations=4,
            solver_ls_iterations=5,
            **kwargs,
        )
