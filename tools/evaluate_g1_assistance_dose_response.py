"""Evaluate one frozen G1 checkpoint on the fixed assistance dose response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from src.core.data_structures import Normalizer
from src.envs.g1_tracking.solver_profiles import get_solver_profile
from src.evaluation.g1_assistance_dose_response import (
    ASSISTANCE_SCALES,
    CHECKPOINT_LABELS,
    PHASES,
    required_scale,
)
from src.evaluation.g1_torso_wrench_oracle import (
    torso_wrench_parameters_from_environment,
)
from tools.evaluate_g1_frozen_torso_wrench_oracle import (
    EXPECTED_TORSO_BODY_ID,
    FROZEN_REFERENCE_SHA256,
    FROZEN_SOLVER_PROFILE,
    _write_json_atomically,
    evaluate_frozen_e008_action,
    frozen_e008_environment_kwargs,
    load_frozen_e008_policy,
    paired_reset,
    rollout_condition,
    runtime_asset_provenance,
    summarize_wrench_trace,
)
from tools.evaluate_g1_tracking import (
    configure_jax,
    make_evaluation_env,
    scale_policy_action,
)
from tools.prepare_g1_rmr_reference import sha256_file


def build_parser() -> argparse.ArgumentParser:
    """Build the immutable single-checkpoint evaluator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-label", choices=CHECKPOINT_LABELS, required=True
    )
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(
        seed=0,
        phases=PHASES,
        assistance_scales=ASSISTANCE_SCALES,
        solver_profile=FROZEN_SOLVER_PROFILE,
    )
    return parser


def registered_conditions() -> tuple[tuple[int, float], ...]:
    """Return the fixed phase-major dose-response grid."""
    return tuple(
        (phase, scale) for phase in PHASES for scale in ASSISTANCE_SCALES
    )


def condition_is_valid(
    summary: dict[str, Any],
    telemetry: dict[str, Any],
    *,
    scale: float,
) -> bool:
    """Validate one rollout and its bounded wrench telemetry."""
    try:
        steps = int(summary["steps"])
        remaining = int(summary["remaining_reference_transitions"])
        terminal = bool(summary["terminal"])
        completed = bool(summary["completed_reference_suffix"])
        telemetry_steps = int(telemetry["steps"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_completion = steps == remaining and not terminal
    return bool(
        1 <= steps <= remaining
        and telemetry_steps == steps
        and completed == expected_completion
        and telemetry.get("finite") is True
        and telemetry.get("force_cap_compliant") is True
        and telemetry.get("torque_cap_compliant") is True
        and (scale != 0.0 or telemetry.get("exact_zero_wrench") is True)
    )


def build_worker_document(
    *,
    checkpoint_label: str,
    provenance: dict[str, Any],
    conditions: list[dict[str, Any]],
    device: dict[str, Any],
) -> dict[str, Any]:
    """Validate and assemble one manifest-last checkpoint artifact."""
    observed = tuple((item.get("phase"), item.get("scale")) for item in conditions)
    if observed != registered_conditions():
        raise ValueError("conditions must cover the exact phase/scale grid")
    if checkpoint_label not in CHECKPOINT_LABELS:
        raise ValueError("checkpoint label is outside the registered sequence")
    if device.get("platform") != "gpu" or device.get("device_count") != 1:
        raise ValueError("worker must see exactly one GPU")
    required: dict[str, float | None] = {}
    for phase in PHASES:
        phase_records = [item for item in conditions if item["phase"] == phase]
        required[str(phase)] = required_scale(
            phase_records, scales=ASSISTANCE_SCALES
        )
    return {
        "protocol": "g1-assistance-dose-response-worker-v1",
        "checkpoint_label": checkpoint_label,
        "provenance": provenance,
        "device": device,
        "phases": list(PHASES),
        "assistance_scales": list(ASSISTANCE_SCALES),
        "conditions": conditions,
        "required_scales": required,
    }


def _provenance(
    *,
    args: argparse.Namespace,
    env: Any,
    torso_body_id: int,
) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    if not checkpoint.is_file() or not reference.is_file():
        raise ValueError("checkpoint and reference must be readable files")
    checkpoint_sha256 = sha256_file(checkpoint)
    reference_sha256 = sha256_file(reference)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match registration")
    if reference_sha256 != FROZEN_REFERENCE_SHA256:
        raise ValueError("reference SHA-256 does not match registration")
    if torso_body_id != EXPECTED_TORSO_BODY_ID:
        raise ValueError("torso body ID does not match the registered model")
    profile = get_solver_profile(FROZEN_SOLVER_PROFILE)
    return {
        "code_commit": args.code_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "reference": str(reference),
        "reference_sha256": reference_sha256,
        "solver_profile": FROZEN_SOLVER_PROFILE,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
        "runtime_assets": runtime_asset_provenance(env),
        "torso_body_id": torso_body_id,
    }


def main() -> None:
    configure_jax()
    args = build_parser().parse_args()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("dose-response worker must see exactly one JAX GPU")
    device = {
        "platform": devices[0].platform,
        "device_count": len(devices),
        "device_kind": devices[0].device_kind,
        "local_hardware_id": int(devices[0].local_hardware_id),
    }
    profile = get_solver_profile(FROZEN_SOLVER_PROFILE)
    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_source_step",
        **frozen_e008_environment_kwargs(args.reference_path),
    )
    torso_body_id, parameters = torso_wrench_parameters_from_environment(env)
    provenance = _provenance(
        args=args, env=env, torso_body_id=torso_body_id
    )
    if int(env.mj_model.opt.iterations) != profile.iterations or int(
        env.mj_model.opt.ls_iterations
    ) != profile.ls_iterations:
        raise RuntimeError("environment solver differs from registered profile")
    torso_slot = env.body_ids.index(torso_body_id)
    if torso_slot != 7:
        raise RuntimeError("torso_link must occupy reference body slot 7")
    actor, actor_params, residual_actor, normalizer_state = (
        load_frozen_e008_policy(env, args.checkpoint)
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
            1.0,
        ).astype(jnp.float64)

    conditions: list[dict[str, Any]] = []
    for phase, scale in registered_conditions():
        initial_state, _ = paired_reset(env, phase=phase, seed=args.seed)
        summary, trace = rollout_condition(
            env,
            initial_state=initial_state,
            action_fn=action_fn,
            phase=phase,
            torso_body_id=torso_body_id,
            torso_slot=torso_slot,
            parameters=parameters,
            scale=scale,
            profile=profile,
        )
        telemetry = summarize_wrench_trace(
            trace, parameters=parameters, dt=env.dt
        )
        valid = condition_is_valid(summary, telemetry, scale=scale)
        conditions.append(
            {
                "phase": phase,
                "scale": scale,
                **summary,
                "wrench": telemetry,
                "valid": valid,
            }
        )
        if not valid:
            raise RuntimeError(
                f"invalid dose-response condition phase={phase} scale={scale}"
            )
    document = build_worker_document(
        checkpoint_label=args.checkpoint_label,
        provenance=provenance,
        conditions=conditions,
        device=device,
    )
    _write_json_atomically(args.output.resolve(), document)
    print(json.dumps(document["required_scales"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
