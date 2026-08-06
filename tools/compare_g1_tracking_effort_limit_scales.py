"""Same-process source-policy screen of fixed G1 torque-authority scales."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

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
    """Build the fixed effort-limit-scale comparison CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--effort-limit-scales",
        type=float,
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
    parser.add_argument("--minimum-reward-drop", type=float, default=0.001)
    return parser


def candidate_passes(
    candidate: dict,
    *,
    minimum_reward_drop: float,
) -> bool:
    """Return whether a shift is nonterminal and materially discriminative."""
    aggregate_document = candidate["aggregate"]
    nominal = aggregate_document["nominal"]
    shifted = aggregate_document["shifted"]
    delta = aggregate_document["delta_shifted_minus_nominal"]
    worsened_errors = sum(
        delta[field] > 0.0 for field in TRACKING_ERROR_FIELDS
    )
    reward_drop = -delta["mean_reward"]
    return (
        shifted["terminal_count"] <= nominal["terminal_count"]
        and reward_drop >= minimum_reward_drop
        and worsened_errors >= 4
    )


def select_earliest_scale(
    candidates: list[dict],
    *,
    minimum_reward_drop: float,
) -> dict | None:
    """Select the first passing scale in registered input order."""
    return next(
        (
            candidate
            for candidate in candidates
            if candidate_passes(
                candidate,
                minimum_reward_drop=minimum_reward_drop,
            )
        ),
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


def _evaluate_source(
    *,
    env,
    source_policy,
    phases: list[int],
    seed: int,
    max_steps: int,
    result_key: str,
) -> tuple[list[dict], dict]:
    def source_action(state):
        return source_policy(state.obs)

    results = [
        {
            "phase": phase,
            result_key: rollout(
                env,
                source_action,
                phase=phase,
                seed=seed,
                max_steps=max_steps,
            ),
        }
        for phase in phases
    ]
    return results, aggregate(results, result_key)


def main() -> None:
    configure_jax()
    parser = build_parser()
    args = parser.parse_args()
    scales = args.effort_limit_scales
    if len(scales) < 2:
        parser.error("provide nominal scale 1.0 and at least one reduction")
    if scales[0] != 1.0:
        parser.error("the first effort-limit scale must be nominal 1.0")
    if (
        len(set(scales)) != len(scales)
        or any(not math.isfinite(scale) or scale <= 0.0 for scale in scales)
    ):
        parser.error("effort-limit scales must be unique, positive, and finite")
    if any(scale >= 1.0 for scale in scales[1:]):
        parser.error("shifted effort-limit scales must be below nominal 1.0")
    if (
        not math.isfinite(args.minimum_reward_drop)
        or args.minimum_reward_drop <= 0.0
    ):
        parser.error("minimum reward drop must be positive and finite")
    if args.max_steps < 1:
        parser.error("max steps must be positive")
    source_checkpoint = args.rmr_policy_checkpoint.resolve()
    if not source_checkpoint.is_file():
        parser.error(f"source checkpoint does not exist: {source_checkpoint}")

    nominal_env = make_evaluation_env(
        "g1_tracking_rmr_50hz_validated",
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
        effort_limit_scale=1.0,
    )
    if any(
        phase < 0 or phase >= nominal_env.reference.qpos.shape[0]
        for phase in args.phases
    ):
        parser.error("every phase must index the registered reference")

    source_policy = load_rmr_policy(source_checkpoint)
    nominal_results, nominal_aggregate = _evaluate_source(
        env=nominal_env,
        source_policy=source_policy,
        phases=args.phases,
        seed=args.seed,
        max_steps=args.max_steps,
        result_key="nominal",
    )

    candidates = []
    for scale in scales[1:]:
        shifted_env = make_evaluation_env(
            "g1_tracking_rmr_50hz_validated",
            solver_iterations=args.solver_iterations,
            solver_ls_iterations=args.solver_ls_iterations,
            effort_limit_scale=scale,
        )
        shifted_results, shifted_aggregate = _evaluate_source(
            env=shifted_env,
            source_policy=source_policy,
            phases=args.phases,
            seed=args.seed,
            max_steps=args.max_steps,
            result_key="shifted",
        )
        phase_results = []
        for nominal_result, shifted_result in zip(
            nominal_results,
            shifted_results,
            strict=True,
        ):
            nominal = nominal_result["nominal"]
            shifted = shifted_result["shifted"]
            phase_results.append(
                {
                    "phase": nominal_result["phase"],
                    "shifted": shifted,
                    "delta_shifted_minus_nominal": summary_delta(
                        shifted,
                        nominal,
                    ),
                }
            )
        aggregate_document = {
            "nominal": nominal_aggregate,
            "shifted": shifted_aggregate,
            "delta_shifted_minus_nominal": summary_delta(
                shifted_aggregate,
                nominal_aggregate,
            ),
        }
        candidate = {
            "effort_limit_scale": scale,
            "results": phase_results,
            "aggregate": aggregate_document,
        }
        candidate["passes_selection_gate"] = candidate_passes(
            candidate,
            minimum_reward_drop=args.minimum_reward_drop,
        )
        candidates.append(candidate)

    selected = select_earliest_scale(
        candidates,
        minimum_reward_drop=args.minimum_reward_drop,
    )
    document = {
        "protocol": "same-process-source-effort-limit-scale-screen-v1",
        "effort_limit_scales": scales,
        "phases": args.phases,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "solver_iterations": args.solver_iterations,
        "solver_ls_iterations": args.solver_ls_iterations,
        "minimum_reward_drop": args.minimum_reward_drop,
        "source_checkpoint": str(source_checkpoint),
        "nominal": {
            "effort_limit_scale": 1.0,
            "results": nominal_results,
            "aggregate": nominal_aggregate,
        },
        "candidates": candidates,
        "selected": (
            {"effort_limit_scale": selected["effort_limit_scale"]}
            if selected is not None
            else None
        ),
    }
    _assert_finite_document(document)
    _write_json_atomically(args.output, document)
    print(json.dumps({"selected": document["selected"]}, sort_keys=True))


if __name__ == "__main__":
    main()
