"""Replay-free five-phase evaluation of source and frozen-preview RMR actors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from statistics import median

import jax.numpy as jnp
import numpy as np

from src.core.rmr_policy import RmrPolicy, apply_trainable_rmr_policy
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.compare_g1_tracking_residual import rollout, summary_delta
from tools.evaluate_g1_tracking import (
    build_compiled_step,
    configure_jax,
    make_evaluation_env,
)
from tools.run_g1_tracking_rmr50_shac import load_source_actor_policy


DEFAULT_PHASES = (0, 24, 48, 72, 96)
LOOKAHEAD_STEPS = (4, 8, 12)


def phase_grid_action_contract(
    reference_residual_scale: float,
) -> dict[str, object]:
    """Return the exact source-order action boundary used by evaluation."""
    if reference_residual_scale not in (0.5, 1.0):
        raise ValueError("reference residual scale must be 0.5 or 1.0")
    return {
        "environment_variant": "g1_tracking_rmr_50hz_source_step",
        "reference_residual_control": True,
        "reference_residual_scale": reference_residual_scale,
        "squash_actor_actions": False,
    }


def select_rmr_policy_observation(
    policy: RmrPolicy,
    observation: jnp.ndarray,
) -> jnp.ndarray:
    """Select the evaluator prefix encoded by an RMR checkpoint."""
    input_dim = int(policy.mean.shape[0])
    observation_dim = int(observation.shape[-1])
    if input_dim > observation_dim:
        raise ValueError(
            f"policy input width {input_dim} exceeds evaluator observation "
            f"width {observation_dim}"
        )
    return observation[..., :input_dim]


def interpolate_rmr_policy(
    source: RmrPolicy,
    candidate: RmrPolicy,
    *,
    alpha: float,
) -> RmrPolicy:
    """Interpolate candidate network parameters toward an exact source."""
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("interpolation alpha must be between zero and one")
    if alpha == 1.0:
        return candidate
    if not (
        np.array_equal(np.asarray(source.mean), np.asarray(candidate.mean))
        and np.array_equal(np.asarray(source.std), np.asarray(candidate.std))
    ):
        raise ValueError("source and candidate normalization must match exactly")
    if len(source.weights) != len(candidate.weights) or len(
        source.biases
    ) != len(candidate.biases):
        raise ValueError("source and candidate network structures differ")
    def blend(source_leaf, candidate_leaf):
        if source_leaf.shape != candidate_leaf.shape:
            raise ValueError("source and candidate network structures differ")
        return source_leaf + alpha * (candidate_leaf - source_leaf)

    return RmrPolicy(
        mean=source.mean,
        std=source.std,
        weights=tuple(
            blend(source_leaf, candidate_leaf)
            for source_leaf, candidate_leaf in zip(
                source.weights, candidate.weights, strict=True
            )
        ),
        biases=tuple(
            blend(source_leaf, candidate_leaf)
            for source_leaf, candidate_leaf in zip(
                source.biases, candidate.biases, strict=True
            )
        ),
    )


def build_phase_grid_summary(
    results: list[dict],
    *,
    phases: tuple[int, ...],
    reference_transitions: int,
) -> dict[str, object]:
    """Summarize exact suffix survival without using training metrics."""
    if len(phases) != 5 or len(set(phases)) != 5:
        raise ValueError("phase grid requires five unique phases")
    if len(results) != 5 or tuple(row.get("phase") for row in results) != phases:
        raise ValueError("phase-grid results do not match the requested order")
    if any(
        phase < 0 or phase >= reference_transitions for phase in phases
    ):
        raise ValueError("every phase must leave a reference transition")
    survival = [int(row["steps"]) for row in results]
    completed = [
        not bool(row["terminal"])
        and int(row["steps"]) == reference_transitions - phase
        for phase, row in zip(phases, results, strict=True)
    ]
    return {
        "phases": list(phases),
        "survival": survival,
        "terminal": [bool(row["terminal"]) for row in results],
        "completed_suffix": completed,
        "minimum_survival": min(survival),
        "median_survival": float(median(survival)),
        "mean_survival": float(np.mean(survival)),
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs=5, default=DEFAULT_PHASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference-residual-scale",
        type=float,
        choices=(0.5, 1.0),
        default=0.5,
    )
    parser.add_argument("--interpolation-alpha", type=float, default=1.0)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default="g1-4x5",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    source_path = args.source_policy_checkpoint.resolve()
    reference_path = args.reference_path.resolve()
    for label, path in (
        ("source policy", source_path),
        ("reference", reference_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    checkpoint_path = (
        None if args.checkpoint is None else args.checkpoint.resolve()
    )
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    profile = get_solver_profile(args.solver_profile)
    action_contract = phase_grid_action_contract(
        args.reference_residual_scale
    )
    env = make_evaluation_env(
        str(action_contract["environment_variant"]),
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference_path,
        reference_stride=1,
        actor_history_len=1,
        actor_reference_lookahead_steps=LOOKAHEAD_STEPS,
        reference_residual_control=bool(
            action_contract["reference_residual_control"]
        ),
        reference_residual_scale=float(
            action_contract["reference_residual_scale"]
        ),
    )
    compiled_step = build_compiled_step(env)
    phases = tuple(args.phases)
    reference_transitions = int(env.reference_transitions)
    if len(phases) != 5 or len(set(phases)) != 5 or any(
        phase < 0 or phase >= reference_transitions for phase in phases
    ):
        raise ValueError("phase grid requires five unique valid phases")

    source_actor = load_source_actor_policy(source_path)
    candidate_actor = None
    if checkpoint_path is not None:
        with checkpoint_path.open("rb") as stream:
            candidate_actor = pickle.load(stream).actor_params
        if not isinstance(candidate_actor, RmrPolicy):
            raise ValueError("candidate checkpoint does not contain an RMR actor")
        candidate_actor = interpolate_rmr_policy(
            source_actor,
            candidate_actor,
            alpha=args.interpolation_alpha,
        )

    def source_action(state):
        return apply_trainable_rmr_policy(
            source_actor,
            select_rmr_policy_observation(source_actor, state.obs),
        ).astype(jnp.float64)

    def candidate_action(state):
        return apply_trainable_rmr_policy(
            candidate_actor,
            select_rmr_policy_observation(candidate_actor, state.obs),
        ).astype(jnp.float64)

    source_results = []
    candidate_results = []
    with solver_context(profile):
        for phase in phases:
            remaining = reference_transitions - phase
            source = rollout(
                env,
                source_action,
                phase=phase,
                seed=args.seed,
                max_steps=remaining,
                step_fn=compiled_step,
            )
            source_results.append({"phase": phase, **source})
            if candidate_actor is not None:
                candidate = rollout(
                    env,
                    candidate_action,
                    phase=phase,
                    seed=args.seed,
                    max_steps=remaining,
                    step_fn=compiled_step,
                )
                candidate_results.append({"phase": phase, **candidate})

    payload = {
        "protocol": "g1-rmr-walk-replay-free-five-phase-v1",
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "reference_transitions": reference_transitions,
        "source_policy_path": str(source_path),
        "source_policy_sha256": _sha256(source_path),
        "solver_profile": args.solver_profile,
        "action_contract": action_contract,
        "actor_reference_lookahead_steps": list(LOOKAHEAD_STEPS),
        "source": {
            "results": source_results,
            "summary": build_phase_grid_summary(
                source_results,
                phases=phases,
                reference_transitions=reference_transitions,
            ),
        },
    }
    if candidate_actor is not None:
        payload["checkpoint_path"] = str(checkpoint_path)
        payload["checkpoint_sha256"] = _sha256(checkpoint_path)
        payload["interpolation_alpha"] = args.interpolation_alpha
        payload["candidate"] = {
            "results": candidate_results,
            "summary": build_phase_grid_summary(
                candidate_results,
                phases=phases,
                reference_transitions=reference_transitions,
            ),
            "phase_deltas_candidate_minus_source": [
                {
                    "phase": phase,
                    **summary_delta(candidate, source),
                }
                for phase, candidate, source in zip(
                    phases,
                    candidate_results,
                    source_results,
                    strict=True,
                )
            ],
        }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
