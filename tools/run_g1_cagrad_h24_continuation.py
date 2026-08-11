"""Continue E008 with an immutable 24-step CAGrad actor horizon."""

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
from tools.run_g1_cagrad_continuation import (
    build_cagrad_continuation_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


def build_cagrad_h24_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Build the fixed horizon-24 exact continuation of E008."""
    kwargs = build_cagrad_continuation_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs.update(
        unroll_length=24,
        total_steps=1_572_864,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Immutable horizon-24 actor-CAGrad continuation of E008."
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_cagrad_h24_continuation_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_cagrad_h24_continuation_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
    )
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
