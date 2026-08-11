"""Replay-free five-phase evaluation of one trained Flax G1 actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax.numpy as jnp

from src.core.data_structures import Normalizer
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.compare_g1_tracking_residual import rollout
from tools.evaluate_g1_rmr_phase_grid import build_phase_grid_summary
from tools.evaluate_g1_tracking import (
    _load_policy,
    configure_jax,
    make_evaluation_env,
    scale_policy_action,
)


DEFAULT_PHASES = (0, 100, 200, 300, 400)
LOOKAHEAD_STEPS = (4, 8, 12)
ACTOR_HISTORY_LEN = 10


def build_payload(
    results: list[dict],
    *,
    phases: tuple[int, ...],
    reference_transitions: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
    reference_path: str,
    reference_sha256: str,
    solver_profile: str,
) -> dict[str, object]:
    """Build the immutable no-render phase-grid artifact."""
    return {
        "protocol": "g1-flax-dance-replay-free-five-phase-v1",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "reference_path": reference_path,
        "reference_sha256": reference_sha256,
        "reference_transitions": reference_transitions,
        "solver_profile": solver_profile,
        "actor_history_len": ACTOR_HISTORY_LEN,
        "actor_reference_lookahead_steps": list(LOOKAHEAD_STEPS),
        "results": results,
        "summary": build_phase_grid_summary(
            results,
            phases=phases,
            reference_transitions=reference_transitions,
        ),
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs=5, default=DEFAULT_PHASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default="g1-4x5",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    checkpoint_path = args.checkpoint.resolve()
    reference_path = args.reference_path.resolve()
    for path in (checkpoint_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = get_solver_profile(args.solver_profile)
    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_source_step",
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference_path,
        reference_stride=1,
        actor_history_len=ACTOR_HISTORY_LEN,
        actor_reference_lookahead_steps=LOOKAHEAD_STEPS,
        reference_residual_control=True,
        reference_residual_scale=0.5,
    )
    phases = tuple(args.phases)
    reference_transitions = int(env.reference_transitions)
    if len(phases) != 5 or len(set(phases)) != 5 or any(
        phase < 0 or phase >= reference_transitions for phase in phases
    ):
        raise ValueError("phase grid requires five unique valid phases")
    actor, actor_params, normalizer_state = _load_policy(
        env, checkpoint_path, args.seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    def action(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        return scale_policy_action(
            actor.apply(actor_params, normalized), 1.0
        ).astype(jnp.float64)

    results = []
    with solver_context(profile):
        for phase in phases:
            result = rollout(
                env,
                action,
                phase=phase,
                seed=args.seed,
                max_steps=reference_transitions - phase,
            )
            results.append({"phase": phase, **result})
    payload = build_payload(
        results,
        phases=phases,
        reference_transitions=reference_transitions,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=_sha256(checkpoint_path),
        reference_path=str(reference_path),
        reference_sha256=_sha256(reference_path),
        solver_profile=args.solver_profile,
    )
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
