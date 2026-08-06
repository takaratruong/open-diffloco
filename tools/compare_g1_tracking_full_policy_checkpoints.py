"""Same-process duration selection for standalone differentiable G1 actors."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

from src.core.rmr_policy import apply_trainable_rmr_policy
from tools.compare_g1_tracking_residual import (
    SUMMARY_FIELDS,
    aggregate,
    rollout,
    summary_delta,
)
from tools.evaluate_g1_tracking import (
    configure_jax,
    load_rmr_policy,
    make_evaluation_env,
)


TRACKING_ERROR_FIELDS = tuple(
    field for field in SUMMARY_FIELDS if field != "mean_reward"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ordered full-policy duration-comparison CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--rmr-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--solver-ls-iterations", type=int, default=5)
    return parser


def candidate_passes(candidate: dict) -> bool:
    """Return whether a candidate meets the strict duration-selection gate."""
    aggregate_document = candidate["aggregate"]
    source = aggregate_document["source"]
    full_policy = aggregate_document["full_policy"]
    delta = aggregate_document["delta_full_policy_minus_source"]
    improved_errors = sum(
        delta[field] < 0.0 for field in TRACKING_ERROR_FIELDS
    )
    return (
        full_policy["terminal_count"] <= source["terminal_count"]
        and delta["mean_reward"] > 0.0
        and improved_errors >= 4
    )


def select_earliest_candidate(candidates: list[dict]) -> dict | None:
    """Select the first passing candidate in registered duration order."""
    return next(
        (candidate for candidate in candidates if candidate_passes(candidate)),
        None,
    )


def _assert_finite_document(value, path: str = "document") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_document(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_document(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")


def _write_json_atomically(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary_path, path)


def main() -> None:
    configure_jax()
    parser = build_parser()
    args = parser.parse_args()
    resolved_checkpoints = [path.resolve() for path in args.checkpoints]
    if len(set(resolved_checkpoints)) != len(resolved_checkpoints):
        parser.error("checkpoint paths must be unique")
    missing = [path for path in resolved_checkpoints if not path.is_file()]
    if missing:
        parser.error(f"checkpoint does not exist: {missing[0]}")

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

    def source_action(state):
        return source_policy(state.obs)

    source_results = []
    for phase in args.phases:
        source_results.append(
            {
                "phase": phase,
                "source": rollout(
                    env,
                    source_action,
                    phase=phase,
                    seed=args.seed,
                    max_steps=args.max_steps,
                ),
            }
        )
    source_aggregate = aggregate(source_results, "source")

    candidates = []
    for checkpoint in resolved_checkpoints:
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        actor = state.actor_params

        def full_policy_action(env_state):
            return apply_trainable_rmr_policy(actor, env_state.obs)

        phase_results = []
        aggregate_inputs = []
        for source_result in source_results:
            phase = source_result["phase"]
            full_policy = rollout(
                env,
                full_policy_action,
                phase=phase,
                seed=args.seed,
                max_steps=args.max_steps,
            )
            source = source_result["source"]
            phase_results.append(
                {
                    "phase": phase,
                    "full_policy": full_policy,
                    "delta_full_policy_minus_source": summary_delta(
                        full_policy, source
                    ),
                }
            )
            aggregate_inputs.append(
                {
                    "source": source,
                    "full_policy": full_policy,
                }
            )
        full_policy_aggregate = aggregate(
            aggregate_inputs,
            "full_policy",
        )
        aggregate_document = {
            "source": source_aggregate,
            "full_policy": full_policy_aggregate,
            "delta_full_policy_minus_source": summary_delta(
                full_policy_aggregate,
                source_aggregate,
            ),
        }
        candidate = {
            "checkpoint": str(checkpoint),
            "step": int(state.step),
            "results": phase_results,
            "aggregate": aggregate_document,
        }
        candidate["passes_selection_gate"] = candidate_passes(candidate)
        candidates.append(candidate)

    selected = select_earliest_candidate(candidates)
    document = {
        "protocol": (
            "same-process-source-versus-ordered-full-policy-checkpoints-v1"
        ),
        "phases": args.phases,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "solver_iterations": args.solver_iterations,
        "solver_ls_iterations": args.solver_ls_iterations,
        "source_checkpoint": str(args.rmr_policy_checkpoint.resolve()),
        "source": {
            "results": source_results,
            "aggregate": source_aggregate,
        },
        "candidates": candidates,
        "selected": (
            {
                "checkpoint": selected["checkpoint"],
                "step": selected["step"],
            }
            if selected is not None
            else None
        ),
    }
    _assert_finite_document(document)
    _write_json_atomically(args.output, document)
    print(json.dumps({"selected": document["selected"]}, sort_keys=True))


if __name__ == "__main__":
    main()
