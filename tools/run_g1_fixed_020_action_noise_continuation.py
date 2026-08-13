"""Continue selected E008 with fixed 0.2 action noise and clean resets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_rmr_action_noise_continuation import (
    SEED,
    build_action_noise_continuation_kwargs,
    validate_action_noise_training_artifacts,
)
from tools.run_g1_rmr_action_noise_continuation import (
    validate_preflight as validate_e008_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically

FIXED_ACTION_NOISE_STD = 0.2
NOMINAL_ENVIRONMENT_HPARAMS: dict[str, object] = {
    "actor_observation_noise": False,
    "reference_reset_noise_scale": 0.0,
    "reference_root_reset_noise_multiplier": 1.0,
    "reference_root_reset_noise_probability": 0.0,
    "domain_randomization": False,
    "kp_range": [35.0, 35.0],
    "kd_range": [0.5, 0.5],
    "friction_range": [1.0, 1.0],
    "mass_range": [1.0, 1.0],
    "com_offset_range": [0.0, 0.0, 0.0],
    "push_velocity_range": [0.0, 0.0],
    "terrain_bump_std": 0.0,
}


def build_fixed_resume_hparams(source_hparams: dict[str, object]) -> dict[str, object]:
    """Return an independent exact-reset, nominal-physics resume contract."""
    staged = json.loads(json.dumps(source_hparams))
    staged.update(NOMINAL_ENVIRONMENT_HPARAMS)
    return staged


def build_fixed_action_noise_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Build the 32-update fixed-0.2 treatment."""
    kwargs = build_action_noise_continuation_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        action_noise_std=FIXED_ACTION_NOISE_STD,
    )
    kwargs.update(
        actor_observation_noise=False,
        reference_reset_noise_scale=0.0,
        domain_randomization=False,
        kp_range=(35.0, 35.0),
        kd_range=(0.5, 0.5),
        friction_range=(1.0, 1.0),
        mass_range=(1.0, 1.0),
        com_offset_range=(0.0, 0.0, 0.0),
        push_velocity_range=(0.0, 0.0),
        terrain_bump_std=0.0,
    )
    return kwargs


def stage_fixed_resume(resume_from: Path, output_root: Path) -> tuple[Path, Path]:
    """Stage the immutable checkpoint beside treatment-specific resume metadata."""
    source_hparams_path = resume_from.with_name("hparams.json")
    source_hparams = json.loads(source_hparams_path.read_text())
    stage_directory = output_root / "fixed_020_resume_input"
    stage_directory.mkdir(parents=True, exist_ok=False)
    staged_checkpoint = stage_directory / resume_from.name
    try:
        os.link(resume_from, staged_checkpoint)
    except OSError:
        shutil.copyfile(resume_from, staged_checkpoint)
    staged_hparams = stage_directory / "hparams.json"
    _write_json_atomically(
        staged_hparams,
        build_fixed_resume_hparams(source_hparams),
    )
    return staged_checkpoint, staged_hparams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_fixed_020_action_noise_continuation_runs"),
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
    """Bind the exact E008 parent and the fixed-0.2 clean-state treatment."""
    payload = validate_e008_preflight(
        repository=repository,
        resume_from=resume_from,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    payload.update(
        protocol="g1-fixed-020-action-noise-continuation-preflight-v1",
        treatment="fixed-scalar-0.2-exact-reference-resets-nominal-physics",
        fixed_action_noise_std=FIXED_ACTION_NOISE_STD,
        environment_hparams=NOMINAL_ENVIRONMENT_HPARAMS,
    )
    return payload


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    """Validate fixed noise and absence of all other perturbations."""
    return validate_action_noise_training_artifacts(
        run_directory,
        expected_action_noise_std=FIXED_ACTION_NOISE_STD,
        protocol="g1-fixed-020-action-noise-continuation-training-v1",
        expected_hparams_overrides=NOMINAL_ENVIRONMENT_HPARAMS,
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
    staged_checkpoint, staged_hparams = stage_fixed_resume(
        args.resume_from.resolve(), output_root
    )
    preflight.update(
        staged_checkpoint=str(staged_checkpoint),
        staged_checkpoint_sha256=sha256_file(staged_checkpoint),
        staged_hparams=str(staged_hparams),
        staged_hparams_sha256=sha256_file(staged_hparams),
    )
    _write_json_atomically(output_root / "fixed_020_preflight.json", preflight)
    configure_jax()
    kwargs = build_fixed_action_noise_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        SEED,
        staged_checkpoint,
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
        output_root / "fixed_020_training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
