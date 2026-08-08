"""Bounded action-only direct shooting for the pinned G1 LAFAN window."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.envs.g1_tracking.reward import rmr_regularization_reward


@dataclass(frozen=True)
class ShootingConfig:
    """Immutable first-gate constants."""

    start_phase: int = 105
    horizon: int = 12
    action_dim: int = 29
    iterations: int = 3
    trust_radius: float = 0.02
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    armijo_factor: float = 1e-4


DEFAULT_SHOOTING_CONFIG = ShootingConfig()


@dataclass(frozen=True)
class ArmijoIteration:
    iteration: int
    objective_before: float
    objective_after: float
    accepted: bool
    alpha: float | None


class ShootingRollout(NamedTuple):
    """Arrays emitted by one uninterrupted physical action rollout."""

    final_data: object
    phases: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    prepared_actions: jax.Array
    raw_torques: jax.Array
    rewards: jax.Array
    terminal: jax.Array
    anchor_z_error: jax.Array
    anchor_xy_error: jax.Array
    gravity_z_error: jax.Array
    distal_z_error: jax.Array


class ActorWindow(NamedTuple):
    """Actor initializer cut from one uninterrupted phase-zero carry."""

    initial_data: object
    initial_previous_action: jax.Array
    initial_history: jax.Array
    final_data: object
    prefix_done: jax.Array
    prefix_terminal: jax.Array
    actions: jax.Array
    phases: jax.Array
    done: jax.Array
    terminal: jax.Array


@dataclass(frozen=True)
class ForwardGradientReport:
    value: float
    gradient: np.ndarray
    scalar_jvps: int
    maximum_primal_error: float


@dataclass(frozen=True)
class DirectionalFDAudit:
    direction: np.ndarray
    autodiff_directional_derivative: float
    finite_difference_directional_derivative: float
    relative_error: float
    positive_support_safe: bool
    negative_support_safe: bool


@dataclass(frozen=True)
class SupportTrace:
    support: np.ndarray
    switch_count: int


def validate_action_sequence(
    actions: np.ndarray, config: ShootingConfig = DEFAULT_SHOOTING_CONFIG
) -> np.ndarray:
    """Return a finite float64 action sequence under the fixed contract."""
    array = np.asarray(actions, dtype=np.float64)
    expected = (config.horizon, config.action_dim)
    if array.shape != expected:
        raise ValueError(f"actions must have shape {expected}")
    if not np.isfinite(array).all():
        raise ValueError("actions must be finite")
    return array


def support_switch_count(support: np.ndarray) -> int:
    """Count transitions between left/right support-foot sets."""
    array = np.asarray(support, dtype=bool)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError("support trace must have shape (T, 2) with T >= 2")
    return int(np.count_nonzero(np.any(array[1:] != array[:-1], axis=1)))


def support_trace_from_states(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    contact_allowance: float = 0.005,
) -> SupportTrace:
    """Classify left/right floor support from carried physical states."""
    qpos_array = np.asarray(qpos, dtype=np.float64)
    qvel_array = np.asarray(qvel, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != model.nq:
        raise ValueError("support qpos must align with the model")
    if qvel_array.shape != (qpos_array.shape[0], model.nv):
        raise ValueError("support qvel must align with qpos and the model")
    if qpos_array.shape[0] < 2:
        raise ValueError("support chronology requires at least two states")
    if not np.isfinite(qpos_array).all() or not np.isfinite(qvel_array).all():
        raise ValueError("support states must be finite")
    if contact_allowance < 0.0 or not np.isfinite(contact_allowance):
        raise ValueError("contact allowance must be finite and non-negative")

    data = mujoco.MjData(model)
    support = np.zeros((qpos_array.shape[0], 2), dtype=bool)
    for row in range(qpos_array.shape[0]):
        data.qpos[:] = qpos_array[row]
        data.qvel[:] = qvel_array[row]
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > contact_allowance:
                continue
            names = tuple(
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(geom_id),
                )
                or ""
                for geom_id in contact.geom
            )
            if "floor" not in names:
                continue
            other = names[1] if names[0] == "floor" else names[0]
            if "left_foot" in other:
                support[row, 0] = True
            if "right_foot" in other:
                support[row, 1] = True
    return SupportTrace(
        support=support,
        switch_count=support_switch_count(support),
    )


def project_trust_box(
    proposed: np.ndarray,
    nominal: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Project around the unbounded validated actor tape, not [-1, 1]."""
    proposed_array = np.asarray(proposed, dtype=np.float64)
    nominal_array = np.asarray(nominal, dtype=np.float64)
    if proposed_array.shape != nominal_array.shape:
        raise ValueError("proposed and nominal actions must align")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust radius must be positive and finite")
    return np.clip(
        proposed_array,
        nominal_array - radius,
        nominal_array + radius,
    )


def run_projected_armijo(
    initial_actions: np.ndarray,
    *,
    objective_and_gate: Callable[[np.ndarray], tuple[float, bool]],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    config: ShootingConfig = DEFAULT_SHOOTING_CONFIG,
) -> tuple[np.ndarray, tuple[ArmijoIteration, ...]]:
    """Run exactly the fixed three projected-gradient iterations."""
    nominal = validate_action_sequence(initial_actions, config)
    current = np.array(nominal, copy=True)
    current_objective, current_feasible = objective_and_gate(current)
    if not current_feasible or not np.isfinite(current_objective):
        raise ValueError("initial action sequence must be finite and feasible")

    trace = []
    for iteration in range(config.iterations):
        gradient = np.asarray(gradient_fn(current), dtype=np.float64)
        if gradient.shape != current.shape or not np.isfinite(gradient).all():
            raise ValueError("gradient must be finite and align with actions")
        maximum = float(np.max(np.abs(gradient)))
        direction = (
            np.zeros_like(gradient)
            if maximum == 0.0
            else -config.trust_radius * gradient / maximum
        )
        accepted = False
        accepted_alpha = None
        next_objective = current_objective
        for alpha in config.line_search_alphas:
            candidate = project_trust_box(
                current + alpha * direction,
                nominal,
                radius=config.trust_radius,
            )
            candidate_objective, feasible = objective_and_gate(candidate)
            if not feasible:
                raise ValueError(
                    "Armijo candidate is infeasible under the exact contact trace"
                )
            step = candidate - current
            armijo_limit = current_objective + config.armijo_factor * float(
                np.sum(gradient * step)
            )
            if np.isfinite(candidate_objective) and candidate_objective <= armijo_limit:
                current = candidate
                next_objective = float(candidate_objective)
                accepted = True
                accepted_alpha = float(alpha)
                break
        trace.append(
            ArmijoIteration(
                iteration=iteration,
                objective_before=float(current_objective),
                objective_after=float(next_objective),
                accepted=accepted,
                alpha=accepted_alpha,
            )
        )
        current_objective = next_objective
    return current, tuple(trace)


def capture_actor_window_without_reset(
    env,
    phase_zero_state,
    policy: Callable[[jax.Array], jax.Array],
    *,
    start_phase: int,
    horizon: int,
) -> ActorWindow:
    """Cut an actor window after carrying full data forward from phase zero."""
    if isinstance(start_phase, bool) or start_phase < 1:
        raise ValueError("actor window start phase must be a positive integer")
    if isinstance(horizon, bool) or horizon < 1:
        raise ValueError("actor window horizon must be a positive integer")

    def actor_step(carry, _):
        data, phase, _last_action, history = carry
        action = policy(history.reshape(-1))
        data, prepared_action, _ = env.advance_physics(data, action)
        next_phase = phase + 1
        info = {
            "phase": next_phase,
            "last_act": prepared_action,
        }
        body_pos, body_quat, _, _ = env._body_state(data)
        done, terminal = env._termination(data, info, body_pos, body_quat)
        actor_frame = env._get_actor_obs(data, info)
        next_history = jnp.concatenate((history[1:], actor_frame[None, :]), axis=0)
        return (
            data,
            next_phase,
            prepared_action,
            next_history,
        ), (action, next_phase, done, terminal)

    prefix_carry, prefix_outputs = jax.lax.scan(
        actor_step,
        (
            phase_zero_state.data,
            phase_zero_state.info["phase"],
            phase_zero_state.info["last_act"],
            phase_zero_state.info["actor_obs_history"],
        ),
        None,
        length=start_phase,
    )
    initial_data, _, initial_previous_action, initial_history = prefix_carry
    final_carry, window_outputs = jax.lax.scan(
        actor_step,
        prefix_carry,
        None,
        length=horizon,
    )
    final_data, _, _, _ = final_carry
    _, _, prefix_done, prefix_terminal = prefix_outputs
    actions, phases, done, terminal = window_outputs
    return ActorWindow(
        initial_data=initial_data,
        initial_previous_action=initial_previous_action,
        initial_history=initial_history,
        final_data=final_data,
        prefix_done=prefix_done,
        prefix_terminal=prefix_terminal,
        actions=actions,
        phases=phases,
        done=done,
        terminal=terminal,
    )


def rollout_actions_without_reset(
    env,
    initial_data,
    *,
    start_phase: int,
    initial_previous_action: jax.Array,
    actions: jax.Array,
) -> ShootingRollout:
    """Carry full physical data through actions without task reset or replay."""
    action_array = jnp.asarray(actions, dtype=jnp.float64)
    if action_array.ndim != 2 or action_array.shape[1] != 29:
        raise ValueError("rollout actions must have shape (T, 29)")
    if action_array.shape[0] < 1:
        raise ValueError("rollout requires at least one action")
    previous_action = jnp.asarray(initial_previous_action, dtype=jnp.float64)
    if previous_action.shape != (29,):
        raise ValueError("initial previous action must have shape (29,)")
    if getattr(env, "termination_margin_weight", 0.0) != 0.0:
        raise ValueError("action shooting requires the unchanged zero-margin task")

    phases = jnp.arange(
        start_phase + 1,
        start_phase + action_array.shape[0] + 1,
        dtype=jnp.int32,
    )

    def step(carry, inputs):
        data, last_action = carry
        action, phase = inputs
        advanced, prepared_action, raw_torque = env.advance_physics(data, action)
        body_pos, body_quat, body_lin_vel, body_ang_vel = env._body_state(advanced)
        tracking_reward, _ = env._tracking_reward_from_body_state(
            {"phase": phase},
            body_pos,
            body_quat,
            body_lin_vel,
            body_ang_vel,
        )
        regularization_reward, _ = rmr_regularization_reward(
            action=prepared_action,
            previous_action=last_action,
            joint_pos=advanced.qpos[7:],
            soft_joint_lower=env.soft_joint_lower,
            soft_joint_upper=env.soft_joint_upper,
        )
        reward = env.reward_scale * (tracking_reward + regularization_reward)
        _, terminal = env._termination(
            advanced,
            {"phase": phase},
            body_pos,
            body_quat,
        )
        errors = env.termination_errors(
            phase=phase,
            body_pos=body_pos,
            body_quat=body_quat,
        )
        outputs = (
            advanced.qpos,
            advanced.qvel,
            prepared_action,
            raw_torque,
            reward,
            terminal,
            errors["anchor_z_error"],
            errors["anchor_xy_error"],
            errors["gravity_z_error"],
            errors["distal_z_error"],
        )
        return (advanced, prepared_action), outputs

    (final_data, _), outputs = jax.lax.scan(
        step,
        (initial_data, previous_action),
        (action_array, phases),
    )
    return ShootingRollout(final_data, phases, *outputs)


def shooting_objective(
    rollout: ShootingRollout,
    actions: jax.Array,
    nominal_actions: jax.Array,
    *,
    action_deviation_weight: float = 1e-3,
) -> jax.Array:
    """Return negative unchanged task reward plus actor-tape trust cost."""
    action_array = jnp.asarray(actions, dtype=jnp.float64)
    nominal_array = jnp.asarray(nominal_actions, dtype=jnp.float64)
    if action_array.shape != nominal_array.shape or action_array.ndim != 2:
        raise ValueError("actions and nominal actions must be aligned matrices")
    if action_deviation_weight < 0.0 or not np.isfinite(action_deviation_weight):
        raise ValueError("action deviation weight must be finite and non-negative")
    return -jnp.mean(rollout.rewards) + action_deviation_weight * jnp.mean(
        jnp.square(action_array - nominal_array)
    )


def canonical_forward_gradient(
    objective_fn: Callable[[jax.Array], jax.Array],
    actions: jax.Array,
    *,
    physical_fn: Callable[[jax.Array], jax.Array] | None = None,
    identity_tolerance: float = 1e-8,
    directional_jvp: Callable | None = None,
) -> ForwardGradientReport:
    """Assemble dJ/dU with one canonical forward JVP per scalar action."""
    if identity_tolerance < 0.0 or not np.isfinite(identity_tolerance):
        raise ValueError("identity tolerance must be finite and non-negative")
    action_array = jnp.asarray(actions, dtype=jnp.float64)
    if action_array.ndim != 2:
        raise ValueError("actions must be a matrix")
    direct_value = float(np.asarray(objective_fn(action_array)))
    if not np.isfinite(direct_value):
        raise ValueError("direct objective must be finite")
    direct_physical = None
    if physical_fn is not None:
        direct_physical = np.asarray(physical_fn(action_array))
        if not np.isfinite(direct_physical).all():
            raise ValueError("direct physical primal must be finite")
    if directional_jvp is None:
        if physical_fn is None:
            directional_jvp = lambda value, direction: jax.jvp(
                objective_fn, (value,), (direction,)
            )
        else:
            combined_fn = lambda value: (
                objective_fn(value),
                physical_fn(value),
            )
            directional_jvp = lambda value, direction: jax.jvp(
                combined_fn, (value,), (direction,)
            )

    derivatives = []
    maximum_primal_error = 0.0
    for scalar_index in range(action_array.size):
        direction = (
            jnp.zeros_like(action_array)
            .reshape(-1)
            .at[scalar_index]
            .set(1.0)
            .reshape(action_array.shape)
        )
        primal, tangent = directional_jvp(action_array, direction)
        if physical_fn is None:
            primal_value = float(np.asarray(primal))
            tangent_value = float(np.asarray(tangent))
        else:
            primal_value = float(np.asarray(primal[0]))
            tangent_value = float(np.asarray(tangent[0]))
            physical_primal = np.asarray(primal[1])
            physical_tangent = np.asarray(tangent[1])
            if (
                not np.isfinite(physical_primal).all()
                or not np.isfinite(physical_tangent).all()
            ):
                raise ValueError(
                    "all forward JVP physical primals and tangents must be finite"
                )
            physical_error = float(np.max(np.abs(physical_primal - direct_physical)))
            maximum_primal_error = max(maximum_primal_error, physical_error)
            if physical_error > identity_tolerance:
                raise ValueError(
                    "forward JVP physical primal identity exceeds tolerance: "
                    f"{physical_error} > {identity_tolerance}"
                )
        if not np.isfinite(primal_value) or not np.isfinite(tangent_value):
            raise ValueError("all forward JVP primals and tangents must be finite")
        primal_error = abs(primal_value - direct_value)
        maximum_primal_error = max(maximum_primal_error, primal_error)
        if primal_error > identity_tolerance:
            raise ValueError(
                "forward JVP primal identity exceeds tolerance: "
                f"{primal_error} > {identity_tolerance}"
            )
        derivatives.append(tangent_value)
    gradient = np.asarray(derivatives, dtype=np.float64).reshape(action_array.shape)
    return ForwardGradientReport(
        value=direct_value,
        gradient=gradient,
        scalar_jvps=action_array.size,
        maximum_primal_error=maximum_primal_error,
    )


def directional_fd_audit(
    objective_fn: Callable[[jax.Array], jax.Array],
    actions: jax.Array,
    gradient: jax.Array,
    *,
    epsilon: float = 1e-3,
    seed: int = 20260808,
    support_gate: Callable[[jax.Array], bool] | None = None,
) -> DirectionalFDAudit:
    """Compare a full action gradient with one deterministic centered FD."""
    action_array = jnp.asarray(actions, dtype=jnp.float64)
    gradient_array = np.asarray(gradient, dtype=np.float64)
    if action_array.shape != gradient_array.shape or action_array.ndim != 2:
        raise ValueError("gradient must align with the action matrix")
    if not np.isfinite(gradient_array).all():
        raise ValueError("gradient must be finite")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("finite-difference epsilon must be positive and finite")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=gradient_array.shape)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("finite-difference direction must be nonzero")
    direction = direction / norm
    direction_jax = jnp.asarray(direction, dtype=jnp.float64)
    positive_actions = action_array + epsilon * direction_jax
    negative_actions = action_array - epsilon * direction_jax
    positive_support_safe = bool(
        True if support_gate is None else support_gate(positive_actions)
    )
    negative_support_safe = bool(
        True if support_gate is None else support_gate(negative_actions)
    )
    positive = float(np.asarray(objective_fn(positive_actions)))
    negative = float(np.asarray(objective_fn(negative_actions)))
    if not np.isfinite(positive) or not np.isfinite(negative):
        raise ValueError("finite-difference objective probes must be finite")
    finite_difference = (positive - negative) / (2.0 * epsilon)
    autodiff = float(np.sum(gradient_array * direction))
    denominator = max(abs(autodiff), abs(finite_difference), 1e-12)
    relative_error = abs(autodiff - finite_difference) / denominator
    return DirectionalFDAudit(
        direction=direction,
        autodiff_directional_derivative=autodiff,
        finite_difference_directional_derivative=finite_difference,
        relative_error=relative_error,
        positive_support_safe=positive_support_safe,
        negative_support_safe=negative_support_safe,
    )
