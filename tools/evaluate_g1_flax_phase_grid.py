"""Replay-free five-phase evaluation of one trained Flax G1 actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
)
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.compare_g1_tracking_residual import rollout
from tools.evaluate_g1_rmr_phase_grid import build_phase_grid_summary
from tools.evaluate_g1_tracking import (
    EVALUATION_ENV_VARIANTS,
    _load_policy,
    build_compiled_step,
    configure_jax,
    make_evaluation_env,
    prepare_evaluation_action,
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
    actor_reference_preview_mode: str,
    actor_residual_preview_adapter: bool = False,
    actor_residual_preview_hidden: int = 256,
    actor_residual_preview_trainable_parameter_count: int = 0,
    post_policy_action_clip: bool = True,
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
        "actor_reference_preview_mode": actor_reference_preview_mode,
        "actor_residual_preview_adapter": actor_residual_preview_adapter,
        "actor_residual_preview_hidden": actor_residual_preview_hidden,
        "actor_residual_preview_trainable_parameter_count": (
            actor_residual_preview_trainable_parameter_count
        ),
        "actor_assistance_conditioning_scale": 0.0,
        "post_policy_action_clip": post_policy_action_clip,
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


def evaluate_actor_action(
    parent_actor,
    actor_params,
    normalized_observations,
    *,
    residual_actor: PreviewResidualAdapter | None = None,
    history_len: int = ACTOR_HISTORY_LEN,
    treatment_frame_dim: int | None = None,
):
    """Apply either a plain Flax actor or the training-identical residual."""
    if residual_actor is None:
        return parent_actor.apply(actor_params, normalized_observations)
    if treatment_frame_dim is None:
        raise ValueError("residual evaluation requires treatment frame width")
    candidate, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        actor_params,
        normalized_observations,
        history_len=history_len,
        treatment_frame_dim=treatment_frame_dim,
    )
    return candidate


def prepare_phase_grid_action(
    action: jax.Array, *, clip_sampled_actor_actions: bool
) -> jax.Array:
    """Apply the exact post-policy boundary used during SHAC training."""
    return prepare_evaluation_action(
        action, squash=clip_sampled_actor_actions
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs=5, default=DEFAULT_PHASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-variant",
        choices=EVALUATION_ENV_VARIANTS,
        default="g1_tracking_rmr_50hz_source_step",
    )
    parser.add_argument(
        "--actor-reference-preview-mode",
        choices=("absolute", "delta"),
        default="absolute",
    )
    parser.add_argument(
        "--actor-residual-preview-adapter", action="store_true"
    )
    parser.add_argument(
        "--actor-residual-preview-hidden", type=int, default=256
    )
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
        args.env_variant,
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference_path,
        reference_stride=1,
        actor_history_len=ACTOR_HISTORY_LEN,
        actor_reference_lookahead_steps=LOOKAHEAD_STEPS,
        actor_reference_preview_mode=args.actor_reference_preview_mode,
        reference_residual_control=True,
        reference_residual_scale=0.5,
    )
    compiled_step = build_compiled_step(env)
    phases = tuple(args.phases)
    reference_transitions = int(env.reference_transitions)
    if len(phases) != 5 or len(set(phases)) != 5 or any(
        phase < 0 or phase >= reference_transitions for phase in phases
    ):
        raise ValueError("phase grid requires five unique valid phases")
    residual_actor = None
    if args.actor_residual_preview_adapter:
        with checkpoint_path.open("rb") as stream:
            checkpoint_state = pickle.load(stream)
        actor_params = checkpoint_state.actor_params
        if not isinstance(actor_params, FrozenPreviewResidualParams):
            raise ValueError(
                "checkpoint is not a frozen residual preview actor"
            )
        actor = Actor(
            env.action_dim,
            hidden=(512, 256, 128),
            squash=getattr(env, "squash_actor_actions", True),
            layer_norm=True,
            zero_output=False,
        )
        residual_actor = PreviewResidualAdapter(
            action_dim=env.action_dim,
            hidden_dim=args.actor_residual_preview_hidden,
        )
        normalizer_state = checkpoint_state.normalizer
    else:
        actor, actor_params, normalizer_state = _load_policy(
            env, checkpoint_path, args.seed
        )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    def action(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        return prepare_phase_grid_action(
            scale_policy_action(
                evaluate_actor_action(
                    actor,
                    actor_params,
                    normalized,
                    residual_actor=residual_actor,
                    history_len=ACTOR_HISTORY_LEN,
                    treatment_frame_dim=env.actor_frame_obs_dim,
                ),
                1.0,
            ),
            clip_sampled_actor_actions=getattr(
                env,
                "clip_sampled_actor_actions",
                getattr(env, "squash_actor_actions", True),
            ),
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
                step_fn=compiled_step,
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
        actor_reference_preview_mode=args.actor_reference_preview_mode,
        actor_residual_preview_adapter=(
            args.actor_residual_preview_adapter
        ),
        actor_residual_preview_hidden=args.actor_residual_preview_hidden,
        actor_residual_preview_trainable_parameter_count=(
            sum(
                int(leaf.size)
                for leaf in jax.tree_util.tree_leaves(actor_params.adapter)
            )
            if args.actor_residual_preview_adapter
            else 0
        ),
        post_policy_action_clip=bool(
            getattr(
                env,
                "clip_sampled_actor_actions",
                getattr(env, "squash_actor_actions", True),
            )
        ),
    )
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
