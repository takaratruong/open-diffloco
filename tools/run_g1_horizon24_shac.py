"""Run the bounded fixed-horizon-24 G1 SHAC experiment."""

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
from tools.run_g1_tracking_shac import configure_jax


def build_horizon24_kwargs(
    profile_name: str, reference_path: str | Path, seed: int
) -> dict:
    """Copy the canonical contract and change only horizon and budget."""
    kwargs = build_canonical_kwargs(profile_name, reference_path, seed)
    kwargs["unroll_length"] = 24
    kwargs["total_steps"] = 393_216
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Immutable G1 SHAC horizon-24 early-gate run."
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
        "--output-root", type=Path, default=Path("g1_horizon24_runs")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_horizon24_kwargs(
        args.solver_profile, args.reference_path.resolve(), args.seed
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
