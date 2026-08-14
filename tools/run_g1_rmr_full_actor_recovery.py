"""Run a short full-RMR-actor differentiable recovery discriminator."""

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
from tools.run_canonical_g1_shac import build_canonical_kwargs
from tools.run_g1_tracking_rmr50_shac import load_source_actor_policy
from tools.run_g1_tracking_shac import configure_jax


def build_rmr_full_actor_recovery_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    source_actor,
) -> dict:
    """Build the immutable 16-update full-policy recovery treatment."""
    kwargs = build_canonical_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from=None,
    )
    kwargs.update(
        total_steps=98_304,
        checkpoint_interval=49_152,
        gradient_accumulation_steps=2,
        actor_lr=1e-4,
        actor_bootstrap_scale=0.0,
        action_noise_std_start=0.05,
        action_noise_std_end=0.05,
        env_variant="g1_tracking_rmr_50hz_action_parity",
        actor_history_len=1,
        actor_reference_lookahead_steps=(),
        initial_full_actor_policy=source_actor,
        domain_randomization=False,
        actor_observation_noise=False,
        reference_reset_noise_scale=0.0,
        friction_range=(1.0, 1.0),
        mass_range=(1.0, 1.0),
        kp_range=(35.0, 35.0),
        kd_range=(0.5, 0.5),
        com_offset_range=(0.0, 0.0, 0.0),
        push_velocity_range=(0.0, 0.0),
        push_interval_s=1e9,
        zero_difficulty_frac=1.0,
        curriculum_grace=98_304,
        curriculum_steps=1,
        actor_cagrad=True,
        actor_cagrad_alpha=0.5,
        actor_cagrad_iterations=32,
        actor_phase_bin_count=5,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_rmr_full_actor_recovery_runs"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    source_actor = load_source_actor_policy(
        args.source_policy_checkpoint.resolve()
    )
    kwargs = build_rmr_full_actor_recovery_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        source_actor,
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
