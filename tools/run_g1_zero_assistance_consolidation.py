"""Continue E012 for 64 additional exact-zero-assistance updates."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_frozen_residual_assistance_curriculum import (
    ASSISTANCE_END_STEP,
    E008_SELECTED_STEP,
    ZERO_ASSISTANCE_FRACTION,
    build_frozen_residual_assistance_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


E012_FINAL_STEP = 1_720_320
CONSOLIDATION_END_STEP = 2_113_536
CHECKPOINT_INTERVAL = 49_152
EXPECTED_RESUME_SHA256 = (
    "0dccdca442ed15e17e76e4518d6c690e47d06ccd79d1440fb7012b36f78ff22f"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "76a78a6b1176f4d8cff785a8cbc01c0dd18e08de83ae7da61d3be093768f0d5f"
)
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the complete registered dense-checkpoint grid."""
    return tuple(
        range(
            E012_FINAL_STEP + CHECKPOINT_INTERVAL,
            CONSOLIDATION_END_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )


def build_zero_assistance_consolidation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Change only E012's absolute endpoint."""
    kwargs = build_frozen_residual_assistance_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs["total_steps"] = CONSOLIDATION_END_STEP
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=("g1-4x5",),
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
        default=Path("g1_zero_assistance_consolidation_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_preflight(
    *,
    repository: Path,
    resume_from: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Fail closed on executable and immutable E012 input provenance."""
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    resume_from = resume_from.resolve()
    reference_path = reference_path.resolve()
    if not resume_from.is_file() or sha256_file(resume_from) != EXPECTED_RESUME_SHA256:
        raise ValueError("E012 final checkpoint SHA-256 does not match")
    hparams_path = resume_from.with_name("hparams.json")
    if (
        not hparams_path.is_file()
        or sha256_file(hparams_path) != EXPECTED_RESUME_HPARAMS_SHA256
    ):
        raise ValueError("E012 hparams SHA-256 does not match")
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("reference SHA-256 does not match")
    return {
        "protocol": "g1-zero-assistance-consolidation-preflight-v1",
        "code_commit": head,
        "checkpoint": str(resume_from),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams_path),
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "solver_profile": "g1-4x5",
        "start_step": E012_FINAL_STEP,
        "end_step": CONSOLIDATION_END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
    }


def _require_equal(document: dict[str, Any], key: str, expected: Any) -> None:
    if document.get(key) != expected:
        raise ValueError(f"training hparams {key} does not match")


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    """Validate dense checkpoints and exact-zero assistance telemetry."""
    run_directory = run_directory.resolve()
    hparams = json.loads((run_directory / "hparams.json").read_text(encoding="utf-8"))
    expected_hparams = {
        "total_steps": CONSOLIDATION_END_STEP,
        "torso_wrench_assistance": True,
        "torso_wrench_assistance_start_step": E008_SELECTED_STEP,
        "torso_wrench_assistance_end_step": ASSISTANCE_END_STEP,
        "torso_wrench_assistance_zero_fraction": ZERO_ASSISTANCE_FRACTION,
        "reference_reset_noise_scale": 1.0,
        "domain_randomization": True,
        "actor_cagrad": True,
        "actor_residual_preview_adapter": True,
        "gradient_accumulation_steps": 2,
        "unroll_length": 12,
        "num_envs": 256,
        "effective_num_envs": 512,
    }
    for key, expected in expected_hparams.items():
        _require_equal(hparams, key, expected)

    steps = expected_checkpoint_steps()
    missing = [
        step
        for step in steps
        if not (run_directory / f"checkpoint_step_{step}.pkl").is_file()
    ]
    if missing:
        raise ValueError(f"dense checkpoints are missing: {missing}")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    rows_by_step = {row.get("step"): row for row in rows}
    if tuple(step for step in steps if step in rows_by_step) != steps:
        raise ValueError("dense checkpoint telemetry is incomplete")
    for step in steps:
        row = rows_by_step[step]
        exact_zero_fields = (
            "torso_wrench_assistance_scale_current",
            "torso_wrench_assistance_active_fraction",
            "torso_wrench_assistance_max_force",
            "torso_wrench_assistance_max_torque",
        )
        if any(row.get(key) != 0.0 for key in exact_zero_fields):
            raise ValueError(f"checkpoint {step} is not exact-zero assistance")
        zero_drift_fields = (
            "actor_preview_frozen_parameter_drift_max_abs",
            "actor_preview_frozen_moment_drift_max_abs",
            "actor_preview_normalizer_drift_max_abs",
        )
        if any(row.get(key) != 0.0 for key in zero_drift_fields):
            raise ValueError(f"checkpoint {step} changed frozen parent state")
        finite_positive_fields = (
            "actor_preview_gradient_norm",
            "actor_preview_update_norm",
        )
        if any(
            not math.isfinite(float(row.get(key, math.nan)))
            or float(row[key]) <= 0.0
            for key in finite_positive_fields
        ):
            raise ValueError(f"checkpoint {step} has invalid residual update telemetry")
        counts = row.get("actor_cagrad_bin_counts")
        if not isinstance(counts, list) or len(counts) != 5 or any(count <= 0 for count in counts):
            raise ValueError(f"checkpoint {step} has invalid CAGrad phase coverage")
        if row.get("torso_wrench_assistance_valid") is not True or row.get(
            "actor_preview_valid"
        ) is not True:
            raise ValueError(f"checkpoint {step} telemetry is invalid")
    return {
        "protocol": "g1-zero-assistance-consolidation-training-v1",
        "valid": True,
        "run_directory": str(run_directory),
        "checkpoint_steps": list(steps),
        "hparams": expected_hparams,
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
    _write_json_atomically(output_root / "consolidation_preflight.json", preflight)

    configure_jax()
    kwargs = build_zero_assistance_consolidation_kwargs(
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
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(run_directory)
    _write_json_atomically(
        output_root / "consolidation_training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
