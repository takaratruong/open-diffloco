"""Produce exactly one H24 clipped-CAGrad proposal from retained E013."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_clipped_dance_continuation import _git_output, _write_json_atomically
from tools.run_g1_clipped_dance_h24_continuation import (
    build_clipped_dance_h24_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


SOURCE_STEP = 1_867_776
PROPOSAL_END_STEP = 1_880_064
CHECKPOINT_INTERVAL = 12_288
EXPECTED_SOURCE_SHA256 = (
    "993124648547286e06d6ace2ec72859cb304a6b16b0330e77647fb95f909e783"
)
EXPECTED_SOURCE_HPARAMS_SHA256 = (
    "4b93d164cdd14b11a7aa134ce2b9c27886f4968d50d551e3ac479e9609b5c698"
)
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    return (PROPOSAL_END_STEP,)


def build_behavioral_trust_proposal_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Build the E013 recipe with exactly one further optimizer update."""
    kwargs = build_clipped_dance_h24_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs.update(
        total_steps=PROPOSAL_END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def validate_preflight(
    *,
    repository: Path,
    resume_from: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, object]:
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    resume_from = resume_from.resolve()
    reference_path = reference_path.resolve()
    if (
        not resume_from.is_file()
        or sha256_file(resume_from) != EXPECTED_SOURCE_SHA256
    ):
        raise ValueError("retained E013 checkpoint SHA-256 does not match")
    hparams_path = resume_from.with_name("hparams.json")
    if (
        not hparams_path.is_file()
        or sha256_file(hparams_path) != EXPECTED_SOURCE_HPARAMS_SHA256
    ):
        raise ValueError("retained E013 hparams SHA-256 does not match")
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("reference SHA-256 does not match")
    return {
        "protocol": "g1-behavioral-trust-proposal-preflight-v1",
        "code_commit": head,
        "checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "hparams_sha256": EXPECTED_SOURCE_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "solver_profile": "g1-4x5",
        "start_step": SOURCE_STEP,
        "end_step": PROPOSAL_END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "unroll_length": 24,
        "actor_per_env_grad_clip": 1.0,
    }


def validate_training_artifacts(run_directory: Path) -> dict[str, object]:
    hparams = json.loads((run_directory / "hparams.json").read_text())
    expected = {
        "unroll_length": 24,
        "total_steps": PROPOSAL_END_STEP,
        "actor_per_env_grad_clip": 1.0,
        "num_envs": 256,
        "gradient_accumulation_steps": 2,
    }
    for key, value in expected.items():
        if hparams.get(key) != value:
            raise ValueError(f"training hparams {key} does not match")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text()
    )
    if len(rows) != 1 or rows[0].get("step") != PROPOSAL_END_STEP:
        raise ValueError("proposal must persist exactly one checkpoint row")
    row = rows[0]
    checkpoint = run_directory / f"checkpoint_step_{PROPOSAL_END_STEP}.pkl"
    if not checkpoint.is_file():
        raise ValueError("proposal checkpoint is missing")
    if row.get("actor_cagrad_valid") is not True:
        raise ValueError("proposal CAGrad telemetry is invalid")
    for values, label in (
        (row.get("actor_cagrad_bin_counts"), "counts"),
        (row.get("actor_cagrad_bin_gradient_norms"), "norms"),
    ):
        if (
            not isinstance(values, list)
            or len(values) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in values
            )
        ):
            raise ValueError(f"proposal has invalid CAGrad {label}")
    for key in (
        "actor_preview_frozen_parameter_drift_max_abs",
        "actor_preview_frozen_moment_drift_max_abs",
        "actor_preview_normalizer_drift_max_abs",
        "torso_wrench_assistance_scale_current",
        "torso_wrench_assistance_active_fraction",
        "torso_wrench_assistance_max_force",
        "torso_wrench_assistance_max_torque",
    ):
        if row.get(key) != 0.0:
            raise ValueError(f"proposal violates zero field {key}")
    return {
        "protocol": "g1-behavioral-trust-proposal-training-v1",
        "valid": True,
        "run_directory": str(run_directory.resolve()),
        "checkpoint_steps": list(expected_checkpoint_steps()),
    }


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
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_behavioral_trust_proposal_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(run_directory)
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
