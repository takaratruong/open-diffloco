"""Capture a learned G1 rollout as a collocation warm-start diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.core.data_structures import Normalizer
from tools.evaluate_g1_tracking import (
    _load_policy,
    configure_jax,
    make_evaluation_env,
)


RECORD_COLUMNS = (
    "reward",
    "terminal",
    "anchor_position_error",
    "anchor_orientation_error",
    "body_position_error",
    "body_orientation_error",
    "body_linear_velocity_error",
    "body_angular_velocity_error",
)
TERMINATION_ERROR_COLUMNS = (
    "anchor_z_error",
    "anchor_xy_error",
    "gravity_z_error",
    "distal_z_error",
)
TERMINATION_THRESHOLDS = np.array([0.25, 1.3, 0.8, 0.4])


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def summarize_warm_start(
    *,
    records: np.ndarray,
    termination_errors: np.ndarray,
    actions: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    expected_steps: int,
) -> dict:
    """Validate and summarize a carried rollout for collocation reuse."""
    records = _finite_matrix(records, "records")
    termination_errors = _finite_matrix(
        termination_errors, "termination errors"
    )
    actions = _finite_matrix(actions, "actions")
    qpos = _finite_matrix(qpos, "qpos")
    qvel = _finite_matrix(qvel, "qvel")
    steps = records.shape[0]
    if records.shape[1] != len(RECORD_COLUMNS):
        raise ValueError("records have the wrong column count")
    if termination_errors.shape != (steps, len(TERMINATION_ERROR_COLUMNS)):
        raise ValueError("termination errors do not align with records")
    if actions.shape[0] != steps:
        raise ValueError("actions do not align with records")
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError("qpos and qvel knot counts differ")
    if expected_steps < 1:
        raise ValueError("expected steps must be positive")

    terminal_count = int(np.count_nonzero(records[:, 1] > 0.5))
    normalized_clearance = 1.0 - (
        termination_errors / TERMINATION_THRESHOLDS[None, :]
    )
    component_clearance = np.min(normalized_clearance, axis=0)
    complete = (
        steps == expected_steps
        and qpos.shape[0] == expected_steps
        and terminal_count == 0
    )
    return {
        "collocation_warm_start_admitted": bool(complete),
        "steps": int(steps),
        "state_knots": int(qpos.shape[0]),
        "collocation_intervals": max(int(qpos.shape[0]) - 1, 0),
        "terminal_count": terminal_count,
        "mean_reward": float(np.mean(records[:, 0])),
        **{
            f"mean_{name}": float(np.mean(records[:, index]))
            for index, name in enumerate(RECORD_COLUMNS[2:], start=2)
        },
        "minimum_normalized_hard_limit_clearance": float(
            np.min(component_clearance)
        ),
        "minimum_normalized_hard_limit_clearance_by_component": {
            name: float(value)
            for name, value in zip(
                TERMINATION_ERROR_COLUMNS, component_clearance, strict=True
            )
        },
        "fraction_margin_barrier_active": float(
            np.mean(
                termination_errors
                > 0.5 * TERMINATION_THRESHOLDS[None, :]
            )
        ),
        "mean_absolute_action": float(np.mean(np.abs(actions))),
        "maximum_absolute_action": float(np.max(np.abs(actions))),
        "fraction_actions_abs_ge_0p95": float(
            np.mean(np.abs(actions) >= 0.95)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--solver-ls-iterations", type=int, default=5)
    return parser


def _write_json_atomically(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def main() -> None:
    configure_jax()
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_validated",
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
    )
    if args.phase < 0 or args.phase >= env.reference_length:
        raise SystemExit("--phase must index the reference")
    actor, actor_params, normalizer_state = _load_policy(
        env, args.checkpoint, args.seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)
    state = env.reset_at_phase(
        jax.random.PRNGKey(args.seed),
        jnp.array(0.0),
        jnp.array(args.phase),
    )

    phases = []
    qpos = []
    qvel = []
    actions = []
    termination_errors = []
    records = []
    for _ in range(args.max_steps):
        phases.append(int(state.info["phase"]))
        qpos.append(np.asarray(state.data.qpos))
        qvel.append(np.asarray(state.data.qvel))
        body_pos, body_quat, _, _ = env._body_state(state.data)
        errors = env.termination_errors(
            phase=state.info["phase"],
            body_pos=body_pos,
            body_quat=body_quat,
        )
        termination_errors.append(
            [float(errors[name]) for name in TERMINATION_ERROR_COLUMNS]
        )

        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        action = actor.apply(actor_params, normalized).astype(jnp.float64)
        actions.append(np.asarray(action))
        state = env.step(state, action)
        records.append(
            [
                float(state.reward),
                float(state.info["terminal"]),
                float(state.metrics["anchor_position_error"]),
                float(state.metrics["anchor_orientation_error"]),
                float(state.metrics["body_position_error"]),
                float(state.metrics["body_orientation_error"]),
                float(state.metrics["body_linear_velocity_error"]),
                float(state.metrics["body_angular_velocity_error"]),
            ]
        )
        if float(state.done) > 0.5:
            break

    arrays = {
        "phase": np.asarray(phases, dtype=np.int32),
        "qpos": np.asarray(qpos, dtype=np.float64),
        "qvel": np.asarray(qvel, dtype=np.float64),
        "action": np.asarray(actions, dtype=np.float64),
        "collocation_action": np.asarray(actions[:-1], dtype=np.float64),
        "records": np.asarray(records, dtype=np.float64),
        "record_columns": np.asarray(RECORD_COLUMNS),
        "termination_errors": np.asarray(
            termination_errors, dtype=np.float64
        ),
        "termination_error_columns": np.asarray(TERMINATION_ERROR_COLUMNS),
        "termination_thresholds": TERMINATION_THRESHOLDS,
    }
    summary = summarize_warm_start(
        records=arrays["records"],
        termination_errors=arrays["termination_errors"],
        actions=arrays["action"],
        qpos=arrays["qpos"],
        qvel=arrays["qvel"],
        expected_steps=args.max_steps,
    )
    summary.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "phase": args.phase,
            "seed": args.seed,
            "solver_iterations": args.solver_iterations,
            "solver_ls_iterations": args.solver_ls_iterations,
            "protocol": "g1-carried-policy-collocation-warm-start-v1",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "warm_start.npz", **arrays)
    _write_json_atomically(args.output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
