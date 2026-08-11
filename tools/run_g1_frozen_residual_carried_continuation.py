"""Continue E008 with history-faithful policy-carried resets."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_g1_frozen_residual_preview_continuation import (
    build_frozen_residual_preview_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


def validate_file_sha256(path: Path, expected: str) -> str:
    """Fail closed unless a file matches its registered streaming digest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: {actual} != {expected}"
        )
    return actual


def build_frozen_residual_carried_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
) -> dict:
    """Change only E008's reset distribution and continuation endpoint."""
    kwargs = build_frozen_residual_preview_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs.update(
        total_steps=1_720_320,
        actor_residual_preview_optimizer="adam",
        carried_reset_bank_path=str(Path(carried_bank).resolve()),
        carried_reset_probability=0.5,
        carried_reset_bank_start=0,
        allow_resume_carried_reset_change=True,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_frozen_residual_carried_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument(
        "--carried-reset-bank", type=Path, required=True
    )
    parser.add_argument("--carried-reset-bank-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bank_path = args.carried_reset_bank.resolve()
    validate_file_sha256(bank_path, args.carried_reset_bank_sha256)
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_frozen_residual_carried_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        bank_path,
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
