"""Run the matched-compute H24 full-RMR-actor recovery discriminator."""

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


def build_rmr_full_actor_h24_recovery_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    source_actor,
) -> dict:
    """Double horizon while preserving transitions per optimizer update."""
    kwargs = build_rmr_full_actor_recovery_kwargs(
        profile_name,
        reference_path,
        seed,
        source_actor,
    )
    kwargs.update(num_envs=64, unroll_length=24)
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
        default=Path("g1_rmr_full_actor_h24_recovery_runs"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    source_actor = load_source_actor_policy(
        args.source_policy_checkpoint.resolve()
    )
    kwargs = build_rmr_full_actor_h24_recovery_kwargs(
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
