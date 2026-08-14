"""Train fresh full-action walking SHAC with a 24-step credit horizon."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_fresh_ppo_action_contract_walk import (
    build_fresh_ppo_action_contract_kwargs,
    validate_preflight as validate_h12_preflight,
    validate_training_artifacts,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


H24 = 24
EFFECTIVE_POPULATION = 512
TOTAL_UPDATES = 128
CHECKPOINT_UPDATES = 16
TRANSITIONS_PER_UPDATE = EFFECTIVE_POPULATION * H24
TOTAL_STEPS = TOTAL_UPDATES * TRANSITIONS_PER_UPDATE
CHECKPOINT_INTERVAL = CHECKPOINT_UPDATES * TRANSITIONS_PER_UPDATE


def expected_checkpoint_steps() -> tuple[int, ...]:
    return tuple(
        range(CHECKPOINT_INTERVAL, TOTAL_STEPS + 1, CHECKPOINT_INTERVAL)
    )


def build_fresh_full_action_h24_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Change only E021's differentiable horizon and matching cadence."""
    kwargs = build_fresh_ppo_action_contract_kwargs(
        profile_name, reference_path, seed
    )
    kwargs.update(
        unroll_length=H24,
        total_steps=TOTAL_STEPS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    return kwargs


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Bind the clean E021 runtime and the single H24 treatment delta."""
    base = validate_h12_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    return {
        **base,
        "protocol": "g1-fresh-full-action-h24-walk-preflight-v1",
        "unroll_length": H24,
        "effective_population": EFFECTIVE_POPULATION,
        "total_updates": TOTAL_UPDATES,
        "total_steps": TOTAL_STEPS,
        "checkpoint_updates": CHECKPOINT_UPDATES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "scientific_delta": [
            "unroll_length",
            "total_steps",
            "checkpoint_interval",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_fresh_full_action_h24_walk_runs"),
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
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_fresh_full_action_h24_kwargs(
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
        protocol="g1-fresh-full-action-h24-walk-training-v1",
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
