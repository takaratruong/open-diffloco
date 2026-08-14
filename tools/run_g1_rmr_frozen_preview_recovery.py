"""Train only new preview columns on the corrected full-scale walk."""

import argparse
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_g1_rmr_full_actor_recovery import (
    build_rmr_full_actor_recovery_kwargs,
)
from tools.run_g1_tracking_rmr50_shac import load_source_actor_policy
from tools.run_g1_tracking_shac import configure_jax


def build_rmr_frozen_preview_recovery_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    source_actor,
    actor_policy_anchor_weight: float = 0.0,
    total_updates: int = 128,
    checkpoint_updates: int = 16,
    lr_decay_updates: int | None = None,
) -> dict:
    """Build the frozen-parent, zero-preview-column walking treatment."""
    kwargs = build_rmr_full_actor_recovery_kwargs(
        profile_name,
        reference_path,
        seed,
        source_actor,
        reference_residual_scale=1.0,
        actor_policy_anchor_weight=actor_policy_anchor_weight,
        total_updates=total_updates,
        checkpoint_updates=checkpoint_updates,
    )
    kwargs.update(
        actor_reference_lookahead_steps=(4, 8, 12),
        actor_preview_adapter=True,
        lr_decay_updates=lr_decay_updates,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=tuple(sorted(SOLVER_PROFILES)),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--source-policy-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--actor-policy-anchor-weight", type=float, default=0.0
    )
    parser.add_argument("--total-updates", type=int, default=128)
    parser.add_argument("--checkpoint-updates", type=int, default=16)
    parser.add_argument("--lr-decay-updates", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_rmr_frozen_preview_recovery_runs"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    source_actor = load_source_actor_policy(
        args.source_policy_checkpoint.resolve()
    )
    kwargs = build_rmr_frozen_preview_recovery_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        source_actor,
        actor_policy_anchor_weight=args.actor_policy_anchor_weight,
        total_updates=args.total_updates,
        checkpoint_updates=args.checkpoint_updates,
        lr_decay_updates=args.lr_decay_updates,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profile = get_solver_profile(args.solver_profile)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(profile):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    print((output_root / relative_save_dir).resolve())


if __name__ == "__main__":
    main()
