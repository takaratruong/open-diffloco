"""One-segment physical smoke for failure-centered G1 collocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.envs.g1_tracking.failure_collocation import (
    FailureWindow,
    corrected_episode_mapping,
    failure_objective,
    multiple_shooting_equalities,
    physical_path_slack_components,
    primal_preserving_linearization,
    rollout_segment,
    stateless_physics_step,
    world_body_kinematics,
)
from src.envs.g1_tracking.fixed_solver import fixed_mjx_solver_outer_loop
from tools.evaluate_g1_tracking import make_evaluation_env
from tools.prepare_g1_rmr_reference import sha256_file


PROTOCOL = "g1-lafan-failure-collocation-smoke-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--grail-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def summarize_derivative(derivative) -> dict[str, float | int | bool]:
    """Fail closed and summarize a derivative pytree."""
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(derivative)]
    if not leaves or not all(np.isfinite(leaf).all() for leaf in leaves):
        raise ValueError("derivative leaves must be finite")
    squared_norm = sum(float(np.sum(np.square(leaf))) for leaf in leaves)
    return {
        "finite": True,
        "leaf_count": len(leaves),
        "l2_norm": float(np.sqrt(squared_norm)),
    }


def require_identity_equalities(
    equalities: np.ndarray,
    *,
    atol: float = 1e-8,
) -> None:
    """Fail closed unless a generated segment has negligible defects."""
    values = np.asarray(equalities)
    if not np.isfinite(values).all():
        raise ValueError("identity equalities must be finite")
    if np.max(np.abs(values), initial=0.0) > atol:
        raise ValueError("identity equalities exceed the declared tolerance")


def active_contact_rows(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[int, ...]:
    """Return rows whose forwarded MuJoCo state has active contact."""
    qpos_array = np.asarray(qpos, dtype=np.float64)
    qvel_array = np.asarray(qvel, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != model.nq:
        raise ValueError("contact qpos shape does not match model")
    if qvel_array.shape != (qpos_array.shape[0], model.nv):
        raise ValueError("contact qvel shape does not match model")
    active = []
    data = mujoco.MjData(model)
    for row in range(qpos_array.shape[0]):
        data.qpos[:] = qpos_array[row]
        data.qvel[:] = qvel_array[row]
        mujoco.mj_forward(model, data)
        if data.ncon and np.any(np.asarray(data.contact.dist) <= 0.0):
            active.append(row)
    return tuple(active)


def _model_reference_action(env, phase: int) -> np.ndarray:
    action = (
        np.asarray(env.qpos_reference[phase, 7:])
        - np.asarray(env.default_joints)
    ) / np.asarray(env.action_scales)
    return np.clip(action, -1.0, 1.0)


def _source_order_action(env, model_action: np.ndarray) -> np.ndarray:
    return np.asarray(model_action)[
        np.asarray(env.controller.model_to_actor_permutation)
    ]


def _all_body_state(
    env, qpos: np.ndarray, qvel: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    body_ids = tuple(range(1, env.mj_model.nbody))
    if env.mj_model.nbody - 1 != 30:
        raise ValueError("canonical G1 episode requires 30 non-world bodies")
    body_names = tuple(
        mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in body_ids
    )
    if any(name is None for name in body_names):
        raise ValueError("every canonical G1 body must be named")
    body_pos, body_rot, body_lin_vel, body_ang_vel = world_body_kinematics(
        env.mj_model, qpos, qvel, body_ids
    )
    pelvis_slot = body_names.index("pelvis")
    return (
        body_pos,
        body_rot,
        body_lin_vel,
        body_ang_vel[:, pelvis_slot],
        body_names,
    )


def _write_json_atomically(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _run_smoke(args: argparse.Namespace) -> dict:
    if not args.reference_path.is_file():
        raise ValueError(f"reference does not exist: {args.reference_path}")
    reference_sha256 = sha256_file(args.reference_path)
    if reference_sha256 != args.reference_sha256:
        raise ValueError("reference SHA-256 does not match the pinned value")
    for path, label in (
        (args.checkpoint_path, "checkpoint"),
        (args.config_path, "config"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    for commit, label in (
        (args.code_commit, "code commit"),
        (args.grail_commit, "GRAIL commit"),
    ):
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError(
                f"{label} must be 40 lowercase hex characters"
            )
    checkpoint_sha256 = sha256_file(args.checkpoint_path)
    config_sha256 = sha256_file(args.config_path)

    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_validated",
        reference_path=args.reference_path,
        reference_stride=1,
    )
    window = FailureWindow()
    initial_phase = window.start_phase
    initial_qpos = env.qpos_reference[initial_phase]
    initial_qvel = env.qvel_reference[initial_phase]
    model_actions = np.stack(
        [
            _model_reference_action(env, phase)
            for phase in range(initial_phase, initial_phase + 2)
        ]
    )
    source_actions = jnp.asarray(
        np.stack(
            [_source_order_action(env, action) for action in model_actions]
        ),
        dtype=jnp.float64,
    )

    def direct_step_fn(qpos, qvel, action):
        return stateless_physics_step(env, qpos, qvel, action)

    step_fn = primal_preserving_linearization(direct_step_fn)

    compiled_rollout = jax.jit(
        lambda qpos, qvel, actions: rollout_segment(
            step_fn, qpos, qvel, actions
        )
    )
    next_qpos, next_qvel = compiled_rollout(
        initial_qpos, initial_qvel, source_actions
    )
    knot_phases = jnp.asarray(
        [initial_phase, initial_phase + 2], dtype=jnp.int32
    )
    nominal_actions = source_actions
    decision = (
        next_qpos[None, :],
        next_qvel[None, :],
        source_actions,
    )
    direction = (
        jnp.full_like(decision[0], 1e-5),
        jnp.full_like(decision[1], -1e-5),
        jnp.full_like(decision[2], 1e-5),
    )

    def knots(current_decision):
        free_qpos, free_qvel, _ = current_decision
        return (
            jnp.concatenate((initial_qpos[None, :], free_qpos)),
            jnp.concatenate((initial_qvel[None, :], free_qvel)),
        )

    def objective_fn(current_decision):
        knot_qpos, knot_qvel = knots(current_decision)
        return failure_objective(
            env,
            knot_qpos,
            knot_qvel,
            knot_phases,
            current_decision[2],
            nominal_actions,
        )

    def equality_fn(current_decision):
        knot_qpos, knot_qvel = knots(current_decision)
        return multiple_shooting_equalities(
            step_fn,
            knot_qpos,
            knot_qvel,
            current_decision[2],
            segment_steps=2,
        )

    def slack_component_fn(current_decision):
        knot_qpos, knot_qvel = knots(current_decision)
        return physical_path_slack_components(
            env,
            knot_qpos,
            knot_qvel,
            knot_phases,
            current_decision[2],
            segment_steps=2,
        )

    @jax.jit
    def derivative_probe(current_decision, current_direction):
        objective, objective_gradient = jax.value_and_grad(objective_fn)(
            current_decision
        )
        equalities, equality_jvp = jax.jvp(
            equality_fn,
            (current_decision,),
            (current_direction,),
        )
        slack_components, slack_component_jvp = jax.jvp(
            slack_component_fn,
            (current_decision,),
            (current_direction,),
        )
        return (
            objective,
            objective_gradient,
            equalities,
            equality_jvp,
            slack_components,
            slack_component_jvp,
        )

    (
        objective,
        objective_gradient,
        equalities,
        equality_jvp,
        slack_components,
        slack_component_jvp,
    ) = derivative_probe(decision, direction)
    equality_array = np.asarray(equalities)
    slack_arrays = {
        name: np.asarray(value) for name, value in slack_components.items()
    }
    slack_jvp_arrays = {
        name: np.asarray(value)
        for name, value in slack_component_jvp.items()
    }
    if not np.isfinite(float(objective)):
        raise ValueError("physical objective must be finite")
    if not np.isfinite(equality_array).all():
        raise ValueError("physical equalities must be finite")
    require_identity_equalities(equality_array)
    for name, values in slack_arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} physical slack values must be finite")
    for name in ("terminal", "action", "torque"):
        if not np.isfinite(slack_jvp_arrays[name]).all():
            raise ValueError(f"{name} physical slack JVP must be finite")
    differentiable_slack_array = np.concatenate(
        tuple(slack_arrays[name] for name in ("terminal", "action", "torque"))
    )
    differentiable_slack_jvp_array = np.concatenate(
        tuple(
            slack_jvp_arrays[name]
            for name in ("terminal", "action", "torque")
        )
    )
    contact_slack_array = slack_arrays["contact"]
    contact_slack_jvp_array = slack_jvp_arrays["contact"]

    full_knot_phases = np.asarray(window.knot_phases, dtype=np.int32)
    full_knot_qpos = jnp.asarray(env.qpos_reference[full_knot_phases])
    full_knot_qvel = jnp.asarray(env.qvel_reference[full_knot_phases])
    full_source_actions = jnp.asarray(
        np.stack(
            [
                _source_order_action(
                    env, _model_reference_action(env, phase)
                )
                for phase in range(window.start_phase, window.end_phase)
            ]
        ),
        dtype=jnp.float64,
    )
    @jax.jit
    def segment_equality_probe(segment_inputs, segment_direction):
        def segment_values(current_inputs):
            (
                start_qpos,
                start_qvel,
                end_qpos,
                end_qvel,
                segment_actions,
            ) = current_inputs
            return multiple_shooting_equalities(
                step_fn,
                jnp.stack((start_qpos, end_qpos)),
                jnp.stack((start_qvel, end_qvel)),
                segment_actions,
                segment_steps=window.segment_steps,
            )

        values = segment_values(segment_inputs)
        _, directional_derivative = jax.jvp(
            segment_values,
            (segment_inputs,),
            (segment_direction,),
        )
        return values, directional_derivative

    segment_equality_arrays = []
    segment_equality_jvp_arrays = []
    segment_equality_jvp_norms = []
    for segment in range(window.segments):
        action_start = segment * window.segment_steps
        segment_inputs = (
            full_knot_qpos[segment],
            full_knot_qvel[segment],
            full_knot_qpos[segment + 1],
            full_knot_qvel[segment + 1],
            full_source_actions[
                action_start : action_start + window.segment_steps
            ],
        )
        segment_direction = tuple(
            jnp.zeros_like(value) if segment == 0 and index < 2
            else jnp.full_like(value, 1e-5)
            for index, value in enumerate(segment_inputs)
        )
        segment_equalities, segment_equality_jvp = segment_equality_probe(
            segment_inputs, segment_direction
        )
        segment_equality_array = np.asarray(segment_equalities)
        segment_equality_jvp_array = np.asarray(segment_equality_jvp)
        if not np.isfinite(segment_equality_array).all():
            raise ValueError(
                f"segment {segment} physical equalities must be finite"
            )
        if not np.isfinite(segment_equality_jvp_array).all():
            raise ValueError(
                f"segment {segment} physical equality JVP must be finite"
            )
        segment_equality_arrays.append(segment_equality_array)
        segment_equality_jvp_arrays.append(segment_equality_jvp_array)
        segment_equality_jvp_norms.append(
            float(np.linalg.norm(segment_equality_jvp_array))
        )
    full_equality_array = np.concatenate(segment_equality_arrays)
    full_equality_jvp_array = np.concatenate(segment_equality_jvp_arrays)
    if full_equality_array.shape != (window.equality_size,):
        raise ValueError("full-window equality count does not match design")
    active_contact_segments = active_contact_rows(
        env.mj_model,
        np.asarray(full_knot_qpos[:-1]),
        np.asarray(full_knot_qvel[:-1]),
    )
    if not active_contact_segments:
        raise ValueError("full-window probe must include active contact")

    export_phases = range(initial_phase, initial_phase + 13)
    export_indices = np.fromiter(export_phases, dtype=np.int32)
    export_qpos = np.asarray(env.qpos_reference)[export_indices]
    export_qvel = np.asarray(env.qvel_reference)[export_indices]
    export_actions = np.stack(
        [_model_reference_action(env, phase) for phase in export_phases]
    )
    (
        body_pos,
        body_rot,
        body_lin_vel,
        root_ang_vel,
        body_names,
    ) = _all_body_state(env, export_qpos, export_qvel)
    model_path = Path(env.xml_path).resolve()
    model_sha256 = sha256_file(model_path)
    episode = corrected_episode_mapping(
        qpos=export_qpos,
        qvel=export_qvel,
        root_ang_vel=root_ang_vel,
        body_pos=body_pos,
        body_rot=body_rot,
        body_lin_vel=body_lin_vel,
        actions=export_actions,
        joint_names=tuple(env.controller.joint_names),
        body_names=body_names,
        default_joint_pos=np.asarray(env.default_joints),
        action_scale=np.asarray(env.action_scales),
        clip_name=args.reference_path.stem,
        env_origin=np.zeros(3, dtype=np.float64),
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        checkpoint_path=str(args.checkpoint_path.resolve()),
        config_path=str(args.config_path.resolve()),
        motion_asset_sha256=reference_sha256,
        terrain_asset_sha256=model_sha256,
        motion_asset_path=str(args.reference_path.resolve()),
        terrain_asset_path=str(model_path),
        grail_commit=args.grail_commit,
        correction_method="identity-smoke",
        correction_run_id="g1-lafan-failure-collocation-smoke",
        correction_source_sha256=checkpoint_sha256,
        correction_code_commit=args.code_commit,
        dynamics_model_sha256=model_sha256,
        dynamics_backend="mujoco-mjx-3.9-fixed-scan-solver4-ls5",
        episode_weight=1.0,
    )
    episode_shapes = {
        key: list(value.shape)
        for key, value in episode.items()
        if isinstance(value, np.ndarray)
    }
    return {
        "protocol": PROTOCOL,
        "reference_path": str(args.reference_path.resolve()),
        "reference_sha256": reference_sha256,
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "config_path": str(args.config_path.resolve()),
        "config_sha256": config_sha256,
        "grail_commit": args.grail_commit,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "code_commit": args.code_commit,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "fixed_solver_patch": True,
        "solver_iterations": int(env.mj_model.opt.iterations),
        "solver_ls_iterations": int(env.mj_model.opt.ls_iterations),
        "control_dt": env.dt,
        "window_start_phase": window.start_phase,
        "window_end_phase": window.end_phase,
        "window_segment_steps": window.segment_steps,
        "decision_size": window.decision_size,
        "equality_size": window.equality_size,
        "smoke_segments": 1,
        "smoke_equalities": int(equality_array.size),
        "identity_max_abs_equality": float(np.max(np.abs(equality_array))),
        "full_window_equalities": int(full_equality_array.size),
        "full_window_max_abs_equality": float(
            np.max(np.abs(full_equality_array))
        ),
        "full_window_equality_jvp": summarize_derivative(
            full_equality_jvp_array
        ),
        "segment_equality_jvp_l2_norms": segment_equality_jvp_norms,
        "active_contact_segment_indices": list(active_contact_segments),
        "active_contact_segment_count": len(active_contact_segments),
        "objective": float(objective),
        "minimum_differentiable_slack": float(
            np.min(differentiable_slack_array)
        ),
        "violated_differentiable_slack_count": int(
            np.count_nonzero(differentiable_slack_array < 0.0)
        ),
        "minimum_contact_slack": float(np.min(contact_slack_array)),
        "violated_contact_slack_count": int(
            np.count_nonzero(contact_slack_array < 0.0)
        ),
        "objective_gradient": summarize_derivative(objective_gradient),
        "equality_jvp": summarize_derivative(equality_jvp),
        "constraint_jvp": summarize_derivative(
            differentiable_slack_jvp_array
        ),
        "constraint_component_jvps": {
            name: summarize_derivative(slack_jvp_arrays[name])
            for name in ("terminal", "action", "torque")
        },
        "contact_jvp_finite": bool(
            np.isfinite(contact_slack_jvp_array).all()
        ),
        "contact_jvp_nonfinite_indices": np.flatnonzero(
            ~np.isfinite(contact_slack_jvp_array)
        ).tolist(),
        "contact_derivative_disposition": (
            "value_only_diagnostic_no_smoothing"
        ),
        "episode_schema_version": episode["schema_version"],
        "episode_trajectory_source": episode["trajectory_source"],
        "episode_correction_method": episode["correction_method"],
        "episode_shapes": episode_shapes,
    }


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    args = build_parser().parse_args()
    with fixed_mjx_solver_outer_loop():
        report = _run_smoke(args)
    _write_json_atomically(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
