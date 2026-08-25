"""Replay-free five-phase evaluation of one trained Flax G1 actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.algorithms.shac.algorithm import load_recovery_support_artifact
from src.algorithms.shac.progressive_recovery_expert import (
    RecoverySupport,
    apply_state_gated_recovery,
)
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
from tools.build_g1_e023_carried_reset_bank import validate_code_commit
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
    actor_history_len: int,
    actor_observe_motion_anchor_position: bool,
    tracking_velocity_kernel: str,
    tracking_torso_orientation_weight: float,
    actor_residual_preview_adapter: bool = False,
    actor_residual_preview_hidden: int = 256,
    actor_residual_preview_trainable_parameter_count: int = 0,
    post_policy_action_clip: bool = True,
    recovery_support_sha256: str | None = None,
    recovery_support_bounds: dict[str, int | float] | None = None,
    seed: int = 0,
    code_provenance: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build the immutable no-render phase-grid artifact."""
    if isinstance(actor_history_len, bool) or actor_history_len < 1:
        raise ValueError("actor history length must be positive")
    return {
        "protocol": "g1-flax-dance-replay-free-five-phase-v1",
        "seed": seed,
        "code_provenance": code_provenance,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "reference_path": reference_path,
        "reference_sha256": reference_sha256,
        "reference_transitions": reference_transitions,
        "solver_profile": solver_profile,
        "actor_history_len": actor_history_len,
        "actor_reference_lookahead_steps": list(LOOKAHEAD_STEPS),
        "actor_reference_preview_mode": actor_reference_preview_mode,
        "actor_observe_motion_anchor_position": (
            actor_observe_motion_anchor_position
        ),
        "tracking_velocity_kernel": tracking_velocity_kernel,
        "tracking_torso_orientation_weight": (
            tracking_torso_orientation_weight
        ),
        "actor_residual_preview_adapter": actor_residual_preview_adapter,
        "actor_residual_preview_hidden": actor_residual_preview_hidden,
        "actor_residual_preview_trainable_parameter_count": (
            actor_residual_preview_trainable_parameter_count
        ),
        "actor_assistance_conditioning_scale": 0.0,
        "post_policy_action_clip": post_policy_action_clip,
        "actor_state_gated_recovery": recovery_support_sha256 is not None,
        "actor_state_gated_recovery_support_sha256": recovery_support_sha256,
        "actor_state_gated_recovery_support_bounds": recovery_support_bounds,
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


def load_checkpoint_environment_contract(checkpoint_path: Path) -> dict:
    """Load the training-identical observation and action boundary."""
    hparams_path = checkpoint_path.resolve().with_name("hparams.json")
    if not hparams_path.is_file():
        raise ValueError("phase-grid checkpoint requires sibling hparams.json")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    required = {
        "env_variant",
        "reference_stride",
        "actor_history_len",
        "actor_reference_lookahead_steps",
        "actor_reference_preview_mode",
        "reference_residual_control",
        "reference_residual_scale",
        "solver_profile",
        "squash_actor_mean",
        "clip_sampled_actor_actions",
    }
    if not isinstance(hparams, dict) or not required.issubset(hparams):
        raise ValueError("checkpoint hparams omit the evaluation contract")
    lookahead = tuple(hparams["actor_reference_lookahead_steps"])
    contract = {
        "env_variant": hparams["env_variant"],
        "reference_stride": hparams["reference_stride"],
        "actor_history_len": hparams["actor_history_len"],
        "actor_reference_lookahead_steps": lookahead,
        "actor_reference_preview_mode": hparams[
            "actor_reference_preview_mode"
        ],
        "actor_observe_motion_anchor_position": hparams.get(
            "actor_observe_motion_anchor_position", False
        ),
        "tracking_velocity_kernel": hparams.get(
            "tracking_velocity_kernel", "exponential"
        ),
        "tracking_torso_orientation_weight": hparams.get(
            "tracking_torso_orientation_weight", 0.0
        ),
        "reference_residual_control": hparams[
            "reference_residual_control"
        ],
        "reference_residual_scale": hparams["reference_residual_scale"],
        "solver_profile": hparams["solver_profile"],
        "squash_actor_mean": hparams["squash_actor_mean"],
        "clip_sampled_actor_actions": hparams[
            "clip_sampled_actor_actions"
        ],
    }
    if (
        contract["env_variant"] not in EVALUATION_ENV_VARIANTS
        or isinstance(contract["reference_stride"], bool)
        or not isinstance(contract["reference_stride"], int)
        or contract["reference_stride"] < 1
        or isinstance(contract["actor_history_len"], bool)
        or not isinstance(contract["actor_history_len"], int)
        or contract["actor_history_len"] < 1
        or not lookahead
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in lookahead
        )
        or tuple(sorted(set(lookahead))) != lookahead
        or contract["actor_reference_preview_mode"] not in {"absolute", "delta"}
        or not isinstance(
            contract["actor_observe_motion_anchor_position"], bool
        )
        or contract["tracking_velocity_kernel"]
        not in {"exponential", "pseudo_huber"}
        or isinstance(contract["tracking_torso_orientation_weight"], bool)
        or not isinstance(
            contract["tracking_torso_orientation_weight"], (int, float)
        )
        or not math.isfinite(contract["tracking_torso_orientation_weight"])
        or contract["tracking_torso_orientation_weight"] < 0.0
        or not isinstance(contract["reference_residual_control"], bool)
        or isinstance(contract["reference_residual_scale"], bool)
        or not math.isfinite(float(contract["reference_residual_scale"]))
        or float(contract["reference_residual_scale"]) <= 0.0
        or contract["solver_profile"] not in SOLVER_PROFILES
        or not isinstance(contract["squash_actor_mean"], bool)
        or not isinstance(contract["clip_sampled_actor_actions"], bool)
    ):
        raise ValueError("checkpoint evaluation contract is invalid")
    return contract


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


def evaluate_gated_actor_action(
    parent_actor,
    residual_actor: PreviewResidualAdapter,
    actor_params: FrozenPreviewResidualParams,
    normalized_observations: jax.Array,
    phases: jax.Array,
    support: RecoverySupport,
    *,
    history_len: int,
    treatment_frame_dim: int,
):
    """Apply a recovery expert with the exact pre-step phase."""
    return apply_state_gated_recovery(
        parent_actor,
        residual_actor,
        actor_params,
        normalized_observations,
        phases,
        support,
        history_len=history_len,
        treatment_frame_dim=treatment_frame_dim,
    )


def load_checkpoint_recovery_support(
    checkpoint_path: Path, support_path: Path
) -> tuple[RecoverySupport, dict[str, object]]:
    """Bind evaluator support to the candidate checkpoint hparams."""
    hparams_path = checkpoint_path.resolve().with_name("hparams.json")
    if not hparams_path.is_file():
        raise ValueError("gated recovery requires sibling hparams.json")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    if not isinstance(hparams, dict) or not hparams.get(
        "actor_state_gated_recovery", False
    ):
        raise ValueError("checkpoint is not a state-gated recovery treatment")
    expected_sha256 = hparams.get(
        "actor_state_gated_recovery_support_sha256"
    )
    if not isinstance(expected_sha256, str):
        raise ValueError("checkpoint recovery support SHA-256 is missing")
    support, report = load_recovery_support_artifact(
        support_path.resolve(), expected_sha256=expected_sha256
    )
    return support, report


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
    parser.add_argument("--code-commit", default=None)
    parser.add_argument(
        "--env-variant",
        choices=EVALUATION_ENV_VARIANTS,
        default=None,
    )
    parser.add_argument(
        "--actor-reference-preview-mode",
        choices=("absolute", "delta"),
        default=None,
    )
    parser.add_argument(
        "--actor-residual-preview-adapter", action="store_true"
    )
    parser.add_argument(
        "--actor-residual-preview-hidden", type=int, default=256
    )
    parser.add_argument(
        "--actor-state-gated-recovery-support", type=Path
    )
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    code_provenance = None
    if args.code_commit is not None:
        repository = Path(__file__).resolve().parents[1]
        commit = validate_code_commit(repository, args.code_commit)
        code_provenance = {
            "repository": str(repository),
            "code_commit": commit,
            "dirty_patch_sha256": hashlib.sha256(b"").hexdigest(),
        }
    configure_jax()
    checkpoint_path = args.checkpoint.resolve()
    reference_path = args.reference_path.resolve()
    for path in (checkpoint_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = load_checkpoint_environment_contract(checkpoint_path)
    requested = {
        "env_variant": args.env_variant,
        "actor_reference_preview_mode": args.actor_reference_preview_mode,
        "solver_profile": args.solver_profile,
    }
    for key, value in requested.items():
        if value is not None and value != contract[key]:
            raise ValueError(f"requested {key} conflicts with checkpoint hparams")
    profile = get_solver_profile(contract["solver_profile"])
    env = make_evaluation_env(
        contract["env_variant"],
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference_path,
        reference_stride=contract["reference_stride"],
        actor_history_len=contract["actor_history_len"],
        actor_reference_lookahead_steps=contract[
            "actor_reference_lookahead_steps"
        ],
        actor_reference_preview_mode=contract[
            "actor_reference_preview_mode"
        ],
        actor_observe_motion_anchor_position=contract[
            "actor_observe_motion_anchor_position"
        ],
        tracking_velocity_kernel=contract["tracking_velocity_kernel"],
        tracking_torso_orientation_weight=contract[
            "tracking_torso_orientation_weight"
        ],
        reference_residual_control=contract["reference_residual_control"],
        reference_residual_scale=contract["reference_residual_scale"],
    )
    if (
        getattr(env, "squash_actor_mean", None)
        != contract["squash_actor_mean"]
        or getattr(env, "clip_sampled_actor_actions", None)
        != contract["clip_sampled_actor_actions"]
    ):
        raise ValueError("environment action boundary conflicts with hparams")
    compiled_step = build_compiled_step(env)
    phases = tuple(args.phases)
    reference_transitions = int(env.reference_transitions)
    if len(phases) != 5 or len(set(phases)) != 5 or any(
        phase < 0 or phase >= reference_transitions for phase in phases
    ):
        raise ValueError("phase grid requires five unique valid phases")
    residual_actor = None
    recovery_support = None
    recovery_support_report = None
    if (
        args.actor_state_gated_recovery_support is not None
        and not args.actor_residual_preview_adapter
    ):
        raise ValueError(
            "state-gated recovery evaluation requires the residual adapter"
        )
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
        if args.actor_state_gated_recovery_support is not None:
            recovery_support, recovery_support_report = (
                load_checkpoint_recovery_support(
                    checkpoint_path,
                    args.actor_state_gated_recovery_support,
                )
            )
    else:
        actor, actor_params, normalizer_state = _load_policy(
            env, checkpoint_path, args.seed
        )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    gate_trace: list[float] = []
    gated_residual_trace: list[jax.Array] = []

    def action(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        if recovery_support is not None:
            candidate, _, gated_residual, gate = evaluate_gated_actor_action(
                actor,
                residual_actor,
                actor_params,
                normalized,
                state.info["phase"],
                recovery_support,
                history_len=contract["actor_history_len"],
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            gate_trace.append(float(gate))
            gated_residual_trace.append(jnp.asarray(gated_residual))
        else:
            candidate = evaluate_actor_action(
                actor,
                actor_params,
                normalized,
                residual_actor=residual_actor,
                history_len=contract["actor_history_len"],
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
        return prepare_phase_grid_action(
            scale_policy_action(
                candidate,
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
            gate_trace.clear()
            gated_residual_trace.clear()
            result = rollout(
                env,
                action,
                phase=phase,
                seed=args.seed,
                max_steps=reference_transitions - phase,
                step_fn=compiled_step,
            )
            if recovery_support is not None:
                gate_values = jnp.asarray(gate_trace)
                residual_values = jnp.asarray(gated_residual_trace)
                result.update(
                    {
                        "gate_active_steps": int(jnp.sum(gate_values > 0.0)),
                        "gate_activation_fraction": float(
                            jnp.mean(gate_values > 0.0)
                        ),
                        "gate_max": float(jnp.max(gate_values)),
                        "gated_residual_rms": float(
                            jnp.sqrt(jnp.mean(jnp.square(residual_values)))
                        ),
                    }
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
        solver_profile=contract["solver_profile"],
        actor_reference_preview_mode=contract[
            "actor_reference_preview_mode"
        ],
        actor_history_len=contract["actor_history_len"],
        actor_observe_motion_anchor_position=contract[
            "actor_observe_motion_anchor_position"
        ],
        tracking_velocity_kernel=contract["tracking_velocity_kernel"],
        tracking_torso_orientation_weight=contract[
            "tracking_torso_orientation_weight"
        ],
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
        recovery_support_sha256=(
            recovery_support_report["sha256"]
            if recovery_support_report is not None
            else None
        ),
        recovery_support_bounds=(
            {
                "radius": recovery_support_report["radius"],
                "phase_min": recovery_support_report["phase_min"],
                "phase_max": recovery_support_report["phase_max"],
                "taper": recovery_support_report["taper"],
            }
            if recovery_support_report is not None
            else None
        ),
        seed=args.seed,
        code_provenance=code_provenance,
    )
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
