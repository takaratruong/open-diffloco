"""Continue selected E008 with scalar noise matched to the RMR vector RMS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.algorithms.shac.algorithm import train
from src.core.rmr_action_noise import RMR_ACTION_STD
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_rmr_action_noise_continuation import (
    SEED,
    build_action_noise_continuation_kwargs,
    validate_action_noise_training_artifacts,
)
from tools.run_g1_rmr_action_noise_continuation import (
    validate_preflight as validate_rmr_source_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically

MATCHED_RMS_ACTION_NOISE_STD = float(
    np.sqrt(
        np.mean(
            np.square(
                np.asarray(RMR_ACTION_STD, dtype=np.float32).astype(np.float64)
            )
        )
    )
)


def build_matched_rms_action_noise_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Build the exact E010 budget with a scalar equal to the RMR vector RMS."""
    return build_action_noise_continuation_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        action_noise_std=MATCHED_RMS_ACTION_NOISE_STD,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_matched_rms_action_noise_continuation_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def validate_preflight(
    *,
    repository: Path,
    resume_from: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Reuse E010 provenance and bind the scalar matched-RMS treatment."""
    payload = validate_rmr_source_preflight(
        repository=repository,
        resume_from=resume_from,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    payload.update(
        protocol="g1-matched-rms-action-noise-continuation-preflight-v1",
        treatment="fixed-scalar-rmr-rms",
        matched_rms_action_noise_std=MATCHED_RMS_ACTION_NOISE_STD,
        matched_rms_formula="sqrt(mean(float64(rmr_float32_vector ** 2)))",
    )
    return payload


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    """Validate the exact 32-update matched-RMS scalar treatment."""
    return validate_action_noise_training_artifacts(
        run_directory,
        expected_action_noise_std=MATCHED_RMS_ACTION_NOISE_STD,
        protocol="g1-matched-rms-action-noise-continuation-training-v1",
    )


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        resume_from=args.resume_from,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )
    _write_json_atomically(
        output_root / "matched_rms_action_noise_preflight.json", preflight
    )
    configure_jax()
    kwargs = build_matched_rms_action_noise_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        SEED,
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
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(run_directory)
    _write_json_atomically(
        output_root / "matched_rms_action_noise_training_validation.json",
        validation,
    )
    print(run_directory)


if __name__ == "__main__":
    main()
