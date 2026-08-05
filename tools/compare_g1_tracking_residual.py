"""Paired replay-free source/residual evaluation across reference phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.core.data_structures import Normalizer
from src.core.rmr_policy import bound_residual_action
from tools.evaluate_g1_tracking import (
    _load_policy,
    configure_jax,
    load_rmr_policy,
    make_evaluation_env,
)


SUMMARY_FIELDS = (
    "mean_reward",
    "mean_anchor_position_error",
    "mean_anchor_orientation_error",
    "mean_body_position_error",
    "mean_body_orientation_error",
    "mean_body_linear_velocity_error",
    "mean_body_angular_velocity_error",
)


def summarize_records(records: np.ndarray) -> dict:
    """Reduce one strict rollout to the registered scalar metrics."""
    return {
        "steps": int(records.shape[0]),
        "terminal": bool(records[-1, 1] > 0.5),
        "mean_reward": float(np.mean(records[:, 0])),
        "mean_anchor_position_error": float(np.mean(records[:, 2])),
        "mean_anchor_orientation_error": float(np.mean(records[:, 3])),
        "mean_body_position_error": float(np.mean(records[:, 4])),
        "mean_body_orientation_error": float(np.mean(records[:, 5])),
        "mean_body_linear_velocity_error": float(np.mean(records[:, 6])),
        "mean_body_angular_velocity_error": float(np.mean(records[:, 7])),
    }


def rollout(
    env,
    action_fn: Callable,
    *,
    phase: int,
    seed: int,
    max_steps: int,
) -> dict:
    """Run one strict closed-loop rollout without rendering or replay."""
    state = env.reset_at_phase(
        jax.random.PRNGKey(seed),
        jnp.array(0.0),
        jnp.array(phase),
    )
    records = []
    for _ in range(max_steps):
        state = env.step(state, action_fn(state))
        records.append(
            (
                float(state.reward),
                float(state.info["terminal"]),
                float(state.metrics["anchor_position_error"]),
                float(state.metrics["anchor_orientation_error"]),
                float(state.metrics["body_position_error"]),
                float(state.metrics["body_orientation_error"]),
                float(state.metrics["body_linear_velocity_error"]),
                float(state.metrics["body_angular_velocity_error"]),
            )
        )
        if float(state.done) > 0.5:
            break
    return summarize_records(np.asarray(records, dtype=np.float64))


def aggregate(results: list[dict], controller: str) -> dict:
    """Average strict summary fields across registered phases."""
    summaries = [item[controller] for item in results]
    return {
        "mean_steps": float(np.mean([item["steps"] for item in summaries])),
        "terminal_count": int(sum(item["terminal"] for item in summaries)),
        **{
            field: float(np.mean([item[field] for item in summaries]))
            for field in SUMMARY_FIELDS
        },
    }


def main() -> None:
    configure_jax()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rmr-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--residual-action-scale", type=float, default=0.1)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--solver-ls-iterations", type=int, default=5)
    args = parser.parse_args()

    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_validated",
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
    )
    if any(phase < 0 or phase >= env.reference.qpos.shape[0] for phase in args.phases):
        parser.error("every phase must index the registered reference")

    source_policy = load_rmr_policy(args.rmr_policy_checkpoint)
    actor, actor_params, normalizer_state = _load_policy(
        env, args.checkpoint, args.seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    def source_action(state):
        return source_policy(state.obs)

    def residual_action(state):
        normalized = env.normalize_actor_obs(
            normalizer,
            normalizer_state,
            state.obs,
        ).astype(jnp.float32)
        residual_logits = actor.apply(actor_params, normalized)
        return source_policy(state.obs) + bound_residual_action(
            residual_logits,
            action_scale=args.residual_action_scale,
        ).astype(jnp.float64)

    results = []
    for phase in args.phases:
        source = rollout(
            env,
            source_action,
            phase=phase,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        residual = rollout(
            env,
            residual_action,
            phase=phase,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        results.append(
            {
                "phase": phase,
                "source": source,
                "residual": residual,
                "delta_residual_minus_source": {
                    "steps": residual["steps"] - source["steps"],
                    **{
                        field: residual[field] - source[field]
                        for field in SUMMARY_FIELDS
                    },
                },
            }
        )

    document = {
        "protocol": "paired-replay-free-four-phase-rmr-residual-v1",
        "phases": args.phases,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "solver_iterations": args.solver_iterations,
        "solver_ls_iterations": args.solver_ls_iterations,
        "source_checkpoint": str(args.rmr_policy_checkpoint.resolve()),
        "residual_checkpoint": str(args.checkpoint.resolve()),
        "residual_action_scale": args.residual_action_scale,
        "results": results,
        "aggregate": {
            "source": aggregate(results, "source"),
            "residual": aggregate(results, "residual"),
        },
    }
    document["aggregate"]["delta_residual_minus_source"] = {
        key: document["aggregate"]["residual"][key]
        - document["aggregate"]["source"][key]
        for key in ("mean_steps", *SUMMARY_FIELDS)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
