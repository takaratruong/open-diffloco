"""Same-process comparison of a standalone differentiable G1 actor and RMR."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from src.core.rmr_policy import apply_trainable_rmr_policy
from tools.compare_g1_tracking_residual import (
    aggregate,
    rollout,
    summary_delta,
)
from tools.evaluate_g1_tracking import (
    configure_jax,
    load_rmr_policy,
    make_evaluation_env,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the full-policy versus source comparison CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rmr-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--solver-ls-iterations", type=int, default=5)
    return parser


def main() -> None:
    configure_jax()
    parser = build_parser()
    args = parser.parse_args()
    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_validated",
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
    )
    if any(
        phase < 0 or phase >= env.reference.qpos.shape[0]
        for phase in args.phases
    ):
        parser.error("every phase must index the registered reference")
    source_policy = load_rmr_policy(args.rmr_policy_checkpoint)
    with args.checkpoint.open("rb") as handle:
        full_actor = pickle.load(handle).actor_params

    def source_action(state):
        return source_policy(state.obs)

    def full_policy_action(state):
        return apply_trainable_rmr_policy(full_actor, state.obs)

    results = []
    for phase in args.phases:
        source = rollout(
            env,
            source_action,
            phase=phase,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        full_policy = rollout(
            env,
            full_policy_action,
            phase=phase,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        results.append(
            {
                "phase": phase,
                "source": source,
                "full_policy": full_policy,
                "delta_full_policy_minus_source": summary_delta(
                    full_policy, source
                ),
            }
        )

    aggregate_document = {
        "source": aggregate(results, "source"),
        "full_policy": aggregate(results, "full_policy"),
    }
    aggregate_document["delta_full_policy_minus_source"] = summary_delta(
        aggregate_document["full_policy"],
        aggregate_document["source"],
    )
    document = {
        "protocol": "paired-replay-free-full-policy-versus-rmr-v1",
        "phases": args.phases,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "solver_iterations": args.solver_iterations,
        "solver_ls_iterations": args.solver_ls_iterations,
        "source_checkpoint": str(args.rmr_policy_checkpoint.resolve()),
        "full_policy_checkpoint": str(args.checkpoint.resolve()),
        "results": results,
        "aggregate": aggregate_document,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
