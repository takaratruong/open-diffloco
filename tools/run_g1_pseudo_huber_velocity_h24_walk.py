"""Train fresh E023 walking SHAC with pseudo-Huber velocity rewards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def build_pseudo_huber_velocity_h24_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Change only E023's linear/angular velocity reward kernel."""
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        tracking_velocity_kernel="pseudo_huber",
        allow_resume_tracking_velocity_kernel_change=False,
    )
    return kwargs


def validate_e023_preflight(**kwargs: Any) -> dict[str, Any]:
    """Load E023's reviewed provenance gate without duplicating it."""
    from tools.run_g1_rmr_noise_h24_walk import validate_preflight

    return validate_preflight(**kwargs)


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Bind E023 provenance and the sole pseudo-Huber treatment delta."""
    base = validate_e023_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    return {
        **base,
        "protocol": "g1-pseudo-huber-velocity-h24-walk-preflight-v1",
        "scientific_delta": ["tracking_velocity_kernel"],
        "tracking_velocity_kernel": "pseudo_huber",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_pseudo_huber_velocity_h24_walk_runs"),
    )
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )

    from src.algorithms.shac.algorithm import train
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.run_g1_fresh_full_action_h24_walk import (
        TOTAL_STEPS,
        expected_checkpoint_steps,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )
    from tools.run_g1_tracking_shac import configure_jax
    from tools.run_g1_zero_assistance_consolidation import (
        _write_json_atomically,
    )

    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_pseudo_huber_velocity_h24_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory,
        expected_kwargs=kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=TOTAL_STEPS,
        protocol="g1-pseudo-huber-velocity-h24-walk-training-v1",
    )
    _write_json_atomically(
        output_root / "training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
