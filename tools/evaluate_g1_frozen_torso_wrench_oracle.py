"""Paired, strict five-phase evaluation of the frozen E008 torso oracle."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
)
from src.envs.g1_tracking.environment import _quat_apply, _quat_inv, _quat_mul
from src.envs.g1_tracking.environment import _yaw_quaternion
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from src.evaluation.g1_torso_wrench_oracle import (
    TorsoWrenchParameters,
    compute_torso_wrench,
    torso_wrench_parameters_from_environment,
    write_torso_wrench,
)
from tools.compare_g1_tracking_residual import summarize_records
from tools.evaluate_g1_flax_phase_grid import evaluate_actor_action
from tools.evaluate_g1_phase_grid import build_phase_grid_payload
from tools.evaluate_g1_tracking import (
    configure_jax,
    make_evaluation_env,
    remaining_reference_transitions,
    scale_policy_action,
)
from tools.prepare_g1_rmr_reference import sha256_file


PHASES = (0, 100, 200, 300, 400)
EXPECTED_SUFFIX_TRANSITIONS = (499, 399, 299, 199, 99)
EXPECTED_UNASSISTED_SURVIVAL = (70, 63, 95, 70, 44)
EXPECTED_TORSO_BODY_ID = 16
FROZEN_E008_CHECKPOINT_SHA256 = (
    "fbea5e272d1431c08753a3600014623cd5577e34e01aeeba18b16af46d369377"
)
FROZEN_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
FROZEN_SOLVER_PROFILE = "g1-4x5"
FROZEN_SEED = 0
FROZEN_ASSISTANCE_SCALE = 1.0
CAP_COMPLIANCE_ATOL = 1e-5
LOOKAHEAD_STEPS = (4, 8, 12)
ACTOR_HISTORY_LEN = 10
RESIDUAL_HIDDEN = 256
TRACE_COLUMNS = (
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the immutable frozen-E008 oracle evaluator CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(
        seed=FROZEN_SEED,
        phases=PHASES,
        assistance_scale=FROZEN_ASSISTANCE_SCALE,
        solver_profile=FROZEN_SOLVER_PROFILE,
        checkpoint_sha256=FROZEN_E008_CHECKPOINT_SHA256,
        reference_sha256=FROZEN_REFERENCE_SHA256,
    )
    return parser


def paired_reset(env: Any, *, phase: int, seed: int) -> tuple[Any, Any]:
    """Create one exact reset and branch immutable state for both conditions."""
    state = env.reset_at_phase(
        jax.random.PRNGKey(seed), jnp.array(0.0), jnp.array(phase)
    )
    return state, state


def inject_wrench(
    state: Any, *, torso_body_id: int, world_wrench: jax.Array
) -> Any:
    """Overwrite the torso wrench before an otherwise unchanged env step."""
    xfrc_applied = write_torso_wrench(
        state.data.xfrc_applied,
        torso_body_id=torso_body_id,
        world_wrench=world_wrench,
    )
    return state.replace(data=state.data.replace(xfrc_applied=xfrc_applied))


def summarize_wrench_trace(
    trace: np.ndarray,
    *,
    parameters: TorsoWrenchParameters,
    dt: float,
) -> dict[str, float | bool | int]:
    """Summarize force/torque magnitudes and absolute mechanical work."""
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 12:
        raise ValueError("wrench trace must be a nonempty (steps, 12) matrix")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("wrench trace timestep must be positive and finite")
    finite = bool(np.isfinite(values).all())
    force = values[:, :3]
    torque = values[:, 3:6]
    linear_velocity = values[:, 6:9]
    angular_velocity = values[:, 9:12]
    force_norm = np.linalg.norm(force, axis=1)
    torque_norm = np.linalg.norm(torque, axis=1)
    absolute_power = np.abs(
        np.einsum("ij,ij->i", force, linear_velocity)
        + np.einsum("ij,ij->i", torque, angular_velocity)
    )
    tolerance = CAP_COMPLIANCE_ATOL
    return {
        "steps": int(values.shape[0]),
        "max_force": float(np.max(force_norm)),
        "rms_force": float(np.sqrt(np.mean(np.square(force_norm)))),
        "max_torque": float(np.max(torque_norm)),
        "rms_torque": float(np.sqrt(np.mean(np.square(torque_norm)))),
        "absolute_wrench_power": float(np.sum(absolute_power)),
        "max_absolute_wrench_power": float(np.max(absolute_power)),
        "mean_absolute_wrench_power": float(np.mean(absolute_power)),
        "absolute_wrench_work": float(dt * np.sum(absolute_power)),
        "finite": finite,
        "force_cap_compliant": bool(
            finite and np.all(force_norm <= parameters.force_cap + tolerance)
        ),
        "torque_cap_compliant": bool(
            finite and np.all(torque_norm <= parameters.torque_cap + tolerance)
        ),
        "force_cap": float(parameters.force_cap),
        "torque_cap": float(parameters.torque_cap),
    }


def passes_oracle_gate(
    unassisted: dict[int, dict[str, Any]],
    assisted: dict[int, dict[str, Any]],
    telemetry: dict[int, dict[str, Any]],
) -> bool:
    """Require the preregistered all-suffix completion and safe trace gate."""
    if (
        set(unassisted) != set(PHASES)
        or set(assisted) != set(PHASES)
        or set(telemetry) != set(PHASES)
    ):
        return False
    return baseline_is_valid(unassisted) and all(
        int(assisted[phase].get("steps", -1)) == expected_steps
        and not bool(assisted[phase].get("terminal", True))
        and bool(assisted[phase].get("completed_reference_suffix", False))
        and bool(telemetry[phase].get("finite", False))
        and bool(telemetry[phase].get("force_cap_compliant", False))
        and bool(telemetry[phase].get("torque_cap_compliant", False))
        for phase, expected_steps in zip(
            PHASES, EXPECTED_SUFFIX_TRANSITIONS, strict=True
        )
    )


def baseline_is_valid(unassisted: dict[int, dict[str, Any]]) -> bool:
    """Require the registered deterministic E008 baseline survival vector."""
    return set(unassisted) == set(PHASES) and all(
        int(unassisted[phase].get("steps", -1)) == expected_steps
        for phase, expected_steps in zip(
            PHASES, EXPECTED_UNASSISTED_SURVIVAL, strict=True
        )
    )


def frozen_provenance(
    *,
    checkpoint: Path,
    reference: Path,
    torso_body_id: int,
) -> dict[str, Any]:
    """Hash and validate every immutable causal input before evaluation."""
    checkpoint = checkpoint.resolve()
    reference = reference.resolve()
    if not checkpoint.is_file() or not reference.is_file():
        raise ValueError("checkpoint and reference must be readable files")
    checkpoint_sha256 = sha256_file(checkpoint)
    reference_sha256 = sha256_file(reference)
    if checkpoint_sha256 != FROZEN_E008_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match frozen E008")
    if reference_sha256 != FROZEN_REFERENCE_SHA256:
        raise ValueError("reference SHA-256 does not match frozen E008")
    if torso_body_id != EXPECTED_TORSO_BODY_ID:
        raise ValueError("resolved torso body ID does not match frozen model")
    profile = get_solver_profile(FROZEN_SOLVER_PROFILE)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "reference": str(reference),
        "reference_sha256": reference_sha256,
        "solver_profile": FROZEN_SOLVER_PROFILE,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
        "torso_body_id": torso_body_id,
        "phases": list(PHASES),
        "expected_suffix_transitions": list(EXPECTED_SUFFIX_TRANSITIONS),
        "expected_unassisted_survival": list(EXPECTED_UNASSISTED_SURVIVAL),
    }


def frozen_e008_environment_kwargs(
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Return the sole environment layout compatible with frozen E008."""
    profile = get_solver_profile(FROZEN_SOLVER_PROFILE)
    kwargs = {
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
        "reference_stride": 1,
        "actor_history_len": ACTOR_HISTORY_LEN,
        "actor_reference_lookahead_steps": LOOKAHEAD_STEPS,
        "actor_reference_preview_mode": "delta",
        "reference_residual_control": True,
        "reference_residual_scale": 0.5,
    }
    if reference_path is not None:
        kwargs["reference_path"] = reference_path
    return kwargs


def load_frozen_e008_policy(
    env: Any, checkpoint: Path
) -> tuple[Actor, FrozenPreviewResidualParams, PreviewResidualAdapter, Any]:
    """Load the E008 composite actor exactly as its preview training used it."""
    with checkpoint.open("rb") as stream:
        checkpoint_state = pickle.load(stream)
    actor_params = checkpoint_state.actor_params
    if not isinstance(actor_params, FrozenPreviewResidualParams):
        raise ValueError("checkpoint is not a frozen E008 residual preview actor")
    leaves = jax.tree_util.tree_leaves(actor_params)
    if not leaves or not all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise ValueError("frozen E008 composite actor parameters must be finite")
    actor = Actor(
        env.action_dim,
        hidden=(512, 256, 128),
        squash=getattr(env, "squash_actor_actions", True),
        layer_norm=True,
        zero_output=False,
    )
    residual_actor = PreviewResidualAdapter(
        action_dim=env.action_dim, hidden_dim=RESIDUAL_HIDDEN
    )
    return actor, actor_params, residual_actor, checkpoint_state.normalizer


def evaluate_frozen_e008_action(
    actor: Actor,
    actor_params: FrozenPreviewResidualParams,
    normalized_observations: jax.Array,
    *,
    residual_actor: PreviewResidualAdapter,
    treatment_frame_dim: int,
) -> jax.Array:
    """Apply the parent plus E008 residual through the proven evaluator path."""
    return evaluate_actor_action(
        actor,
        actor_params,
        normalized_observations,
        residual_actor=residual_actor,
        history_len=ACTOR_HISTORY_LEN,
        treatment_frame_dim=treatment_frame_dim,
    )


def _aligned_torso_targets(
    env: Any,
    state: Any,
    *,
    torso_slot: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return current and yaw-aligned torso targets in world coordinates."""
    positions, quaternions, linear_velocities, angular_velocities = (
        env._body_state(state.data)
    )
    phase = state.info["phase"]
    reference_positions, reference_quaternions = (
        env._aligned_reference_body_targets(positions[0], quaternions[0], phase)
    )
    reference_anchor_quaternion = env.body_quat_reference[phase, 0]
    yaw_delta = _yaw_quaternion(
        _quat_mul(quaternions[0], _quat_inv(reference_anchor_quaternion))
    )
    reference_linear_velocity = _quat_apply(
        yaw_delta, env.body_lin_vel_reference[phase, torso_slot]
    )
    reference_angular_velocity = _quat_apply(
        yaw_delta, env.body_ang_vel_reference[phase, torso_slot]
    )
    return (
        positions[torso_slot],
        quaternions[torso_slot],
        linear_velocities[torso_slot],
        angular_velocities[torso_slot],
        reference_positions[torso_slot],
        reference_quaternions[torso_slot],
        reference_linear_velocity,
        reference_angular_velocity,
    )


def _wrench_for_state(
    env: Any,
    state: Any,
    *,
    torso_slot: int,
    parameters: TorsoWrenchParameters,
    scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (
        actual_position,
        actual_quaternion,
        actual_linear_velocity,
        actual_angular_velocity,
        reference_position,
        reference_quaternion,
        reference_linear_velocity,
        reference_angular_velocity,
    ) = _aligned_torso_targets(env, state, torso_slot=torso_slot)
    wrench = compute_torso_wrench(
        parameters=parameters,
        actual_position=actual_position,
        actual_quaternion=actual_quaternion,
        actual_linear_velocity=actual_linear_velocity,
        actual_angular_velocity=actual_angular_velocity,
        reference_position=reference_position,
        reference_quaternion=reference_quaternion,
        reference_linear_velocity=reference_linear_velocity,
        reference_angular_velocity=reference_angular_velocity,
        scale=scale,
    )
    return wrench, actual_linear_velocity, actual_angular_velocity


def _summary_record(state: Any) -> tuple[float, ...]:
    return (
        float(state.reward),
        float(state.info["terminal"]),
        float(state.metrics["anchor_position_error"]),
        float(state.metrics["anchor_orientation_error"]),
        float(state.metrics["body_position_error"]),
        float(state.metrics["body_orientation_error"]),
        float(state.metrics["body_linear_velocity_error"]),
        float(state.metrics["body_angular_velocity_error"]),
    )


def rollout_condition(
    env: Any,
    *,
    initial_state: Any,
    action_fn: Callable[[Any], jax.Array],
    phase: int,
    torso_body_id: int,
    torso_slot: int,
    parameters: TorsoWrenchParameters,
    scale: float,
    profile: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one immutable strict suffix while replacing the torso row each step."""
    state = initial_state
    remaining = remaining_reference_transitions(
        env.reference_length, phase, env.reference_stride
    )
    records = []
    trace = []
    for _ in range(remaining):
        wrench, linear_velocity, angular_velocity = _wrench_for_state(
            env,
            state,
            torso_slot=torso_slot,
            parameters=parameters,
            scale=scale,
        )
        # This executes for both conditions: scale zero still clears stale MJX
        # applied wrenches before the unchanged environment step.
        state = inject_wrench(
            state, torso_body_id=torso_body_id, world_wrench=wrench
        )
        trace.append(
            np.concatenate(
                (
                    np.asarray(wrench, dtype=np.float64),
                    np.asarray(linear_velocity, dtype=np.float64),
                    np.asarray(angular_velocity, dtype=np.float64),
                )
            )
        )
        with solver_context(profile):
            state = env.step(state, action_fn(state))
        records.append(_summary_record(state))
        if float(state.done) > 0.5:
            break
    summary = summarize_records(np.asarray(records, dtype=np.float64))
    true_terminal = bool(summary["terminal"])
    summary.update(
        {
            "evaluation_start_phase": phase,
            "remaining_reference_transitions": remaining,
            "completed_reference_suffix": (
                len(records) == remaining and not true_terminal
            ),
            "intermediate_reset_occurred": true_terminal,
        }
    )
    return summary, np.asarray(trace, dtype=np.float64)


def _assert_finite_document(value: Any, path: str = "document") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_document(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_document(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    """Write only fully validated evidence, replacing the final path atomically."""
    _assert_finite_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    configure_jax()
    args = build_parser().parse_args()
    profile = get_solver_profile(FROZEN_SOLVER_PROFILE)
    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_source_step",
        **frozen_e008_environment_kwargs(args.reference_path),
    )
    torso_body_id, parameters = torso_wrench_parameters_from_environment(env)
    provenance = frozen_provenance(
        checkpoint=args.checkpoint,
        reference=args.reference_path,
        torso_body_id=torso_body_id,
    )
    if int(env.mj_model.opt.iterations) != profile.iterations or int(
        env.mj_model.opt.ls_iterations
    ) != profile.ls_iterations:
        raise RuntimeError("environment solver budget differs from frozen profile")
    try:
        torso_slot = env.body_ids.index(torso_body_id)
    except ValueError as error:
        raise RuntimeError("reference body slots do not include torso_link") from error
    if torso_slot != 7:
        raise RuntimeError("torso_link must occupy frozen reference body slot 7")
    actor, actor_params, residual_actor, normalizer_state = load_frozen_e008_policy(
        env, args.checkpoint
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    def action_fn(state: Any) -> jax.Array:
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        return scale_policy_action(
            evaluate_frozen_e008_action(
                actor,
                actor_params,
                normalized,
                residual_actor=residual_actor,
                treatment_frame_dim=env.actor_frame_obs_dim,
            ),
            FROZEN_ASSISTANCE_SCALE,
        ).astype(jnp.float64)

    phase_results: dict[int, dict[str, Any]] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output.with_suffix("")
    for phase in PHASES:
        unassisted_initial, assisted_initial = paired_reset(
            env, phase=phase, seed=args.seed
        )
        unassisted, unassisted_trace = rollout_condition(
            env,
            initial_state=unassisted_initial,
            action_fn=action_fn,
            phase=phase,
            torso_body_id=torso_body_id,
            torso_slot=torso_slot,
            parameters=parameters,
            scale=0.0,
            profile=profile,
        )
        assisted, assisted_trace = rollout_condition(
            env,
            initial_state=assisted_initial,
            action_fn=action_fn,
            phase=phase,
            torso_body_id=torso_body_id,
            torso_slot=torso_slot,
            parameters=parameters,
            scale=FROZEN_ASSISTANCE_SCALE,
            profile=profile,
        )
        unassisted_telemetry = summarize_wrench_trace(
            unassisted_trace, parameters=parameters, dt=env.dt
        )
        assisted_telemetry = summarize_wrench_trace(
            assisted_trace, parameters=parameters, dt=env.dt
        )
        np.savez_compressed(
            output_prefix.with_name(f"{output_prefix.name}.phase_{phase:03d}.npz"),
            columns=np.asarray(TRACE_COLUMNS),
            unassisted=unassisted_trace,
            assisted=assisted_trace,
        )
        phase_results[phase] = {
            "unassisted": unassisted,
            "assisted": assisted,
            "unassisted_wrench": unassisted_telemetry,
            "assisted_wrench": assisted_telemetry,
            "paired_reset": "single-identical-exact-reference-state",
            "unassisted_matches_registered_e008": (
                unassisted["steps"]
                == EXPECTED_UNASSISTED_SURVIVAL[PHASES.index(phase)]
            ),
        }
    assisted_summaries = {
        phase: phase_results[phase]["assisted"] for phase in PHASES
    }
    assisted_telemetry = {
        phase: phase_results[phase]["assisted_wrench"] for phase in PHASES
    }
    unassisted_summaries = {
        phase: phase_results[phase]["unassisted"] for phase in PHASES
    }
    document = {
        "protocol": "frozen-e008-paired-torso-wrench-oracle-v1",
        "provenance": provenance,
        "oracle": {
            "frequency_hz": parameters.frequency_hz,
            "force_cap": parameters.force_cap,
            "torque_cap": parameters.torque_cap,
            "assistance_scale": FROZEN_ASSISTANCE_SCALE,
        },
        "results": {str(phase): phase_results[phase] for phase in PHASES},
        "unassisted_phase_grid": build_phase_grid_payload(
            unassisted_summaries,
            checkpoint_sha256=provenance["checkpoint_sha256"],
            reference_sha256=provenance["reference_sha256"],
            solver_profile=FROZEN_SOLVER_PROFILE,
        ),
        "assisted_phase_grid": build_phase_grid_payload(
            assisted_summaries,
            checkpoint_sha256=provenance["checkpoint_sha256"],
            reference_sha256=provenance["reference_sha256"],
            solver_profile=FROZEN_SOLVER_PROFILE,
        ),
        "baseline_valid": baseline_is_valid(unassisted_summaries),
        "passes_all_suffix_gate": passes_oracle_gate(
            unassisted_summaries, assisted_summaries, assisted_telemetry
        ),
    }
    _write_json_atomically(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
