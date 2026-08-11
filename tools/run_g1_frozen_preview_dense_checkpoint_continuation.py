"""Replay E011 exactly while archiving every eight actor updates."""

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
from tools.run_g1_frozen_preview_adapter_continuation import (
    build_frozen_preview_adapter_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


def build_frozen_preview_dense_checkpoint_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Change only checkpoint cadence from the E011 training contract."""
    kwargs = build_frozen_preview_adapter_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs["checkpoint_interval"] = 49_152
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact frozen-preview replay with eight-update checkpoints."
        )
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
        default=Path("g1_frozen_preview_dense_checkpoint_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_frozen_preview_dense_checkpoint_kwargs(
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
