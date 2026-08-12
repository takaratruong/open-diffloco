"""Continue selected E008 with immutable per-joint RMR action noise."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.core.rmr_action_noise import (
    RMR_ACTION_STD,
    RMR_ACTION_STD_JOINT_NAMES,
    action_noise_std_hparam,
)
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_root_recovery_continuation import validate_consumed_resume_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _require_equal,
    _write_json_atomically,
)
from tools.run_g1_zero_bootstrap_continuation import (
    ASSISTANCE_END_STEP,
    ZERO_ASSISTANCE_FRACTION,
    build_zero_bootstrap_kwargs,
)
from tools.run_g1_zero_bootstrap_continuation import (
    E008_SELECTED_STEP as _ASSISTANCE_PARENT_STEP,
)

E008_SELECTED_STEP = 1_867_776
RMR_NOISE_END_STEP = 2_064_384
CHECKPOINT_INTERVAL = 49_152
SEED = 0
E008_SELECTED_CHECKPOINT_SHA256 = (
    "2de4af6d78cd5250c87577397c048b06e60c5b8a7b272c0f8966b8bf589b4474"
)
E008_SELECTED_HPARAMS_SHA256 = (
    "e0b78f2185d91e7d2edadff0afb4f470e70d38f1f7716c304cf866380e594dba"
)
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
RMR_SOURCE_CHECKPOINT_SHA256 = (
    "5174a0f1dc8c83ef9ea45769c3b0f19383e5aeeafea2171433f8e7bb88b21746"
)
RMR_SOURCE_STD = RMR_ACTION_STD
RMR_SOURCE_JOINT_NAMES = RMR_ACTION_STD_JOINT_NAMES


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the complete four-checkpoint treatment grid."""
    return tuple(
        range(
            E008_SELECTED_STEP + CHECKPOINT_INTERVAL,
            RMR_NOISE_END_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )


def build_rmr_action_noise_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Change only E008's endpoint and authorized fixed vector noise schedule."""
    kwargs = build_zero_bootstrap_kwargs(
        profile_name, reference_path, seed, resume_from
    )
    kwargs.update(
        total_steps=RMR_NOISE_END_STEP,
        action_noise_std_start=RMR_ACTION_STD,
        action_noise_std_end=RMR_ACTION_STD,
        action_noise_schedule_steps=RMR_NOISE_END_STEP,
        allow_resume_action_noise_change=True,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_rmr_action_noise_continuation_runs"),
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
    """Fail closed on E008 inputs and the pinned RMR noise provenance."""
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    resume_from = resume_from.resolve()
    reference_path = reference_path.resolve()
    if (
        not resume_from.is_file()
        or sha256_file(resume_from) != E008_SELECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("E008 selected checkpoint SHA-256 does not match")
    hparams_path = resume_from.with_name("hparams.json")
    if (
        not hparams_path.is_file()
        or sha256_file(hparams_path) != E008_SELECTED_HPARAMS_SHA256
    ):
        raise ValueError("E008 selected hparams SHA-256 does not match")
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("reference SHA-256 does not match")
    return {
        "protocol": "g1-rmr-action-noise-continuation-preflight-v1",
        "code_commit": head,
        "checkpoint": str(resume_from),
        "checkpoint_sha256": E008_SELECTED_CHECKPOINT_SHA256,
        "hparams": str(hparams_path),
        "hparams_sha256": E008_SELECTED_HPARAMS_SHA256,
        "reference": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "rmr_source_checkpoint_sha256": RMR_SOURCE_CHECKPOINT_SHA256,
        "rmr_action_std": action_noise_std_hparam(RMR_SOURCE_STD),
        "rmr_action_std_dtype": "float32",
        "rmr_action_std_joint_names": list(RMR_SOURCE_JOINT_NAMES),
        "solver_profile": "g1-4x5",
        "start_step": E008_SELECTED_STEP,
        "end_step": RMR_NOISE_END_STEP,
        "update_count": 32,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "seed": SEED,
        "runtime_assets": validate_consumed_resume_assets(hparams_path, reference_path),
    }


def _validate_cagrad_row(row: dict[str, object], step: int) -> None:
    exact_zero_fields = (
        "torso_wrench_assistance_scale_current",
        "torso_wrench_assistance_active_fraction",
        "torso_wrench_assistance_max_force",
        "torso_wrench_assistance_max_torque",
    )
    if row.get("actor_bootstrap_scale_current") != 0.0:
        raise ValueError(f"checkpoint {step} is not zero actor bootstrap")
    if any(row.get(key) != 0.0 for key in exact_zero_fields):
        raise ValueError(f"checkpoint {step} is not exact-zero assistance")
    frozen_fields = (
        "actor_preview_frozen_parameter_drift_max_abs",
        "actor_preview_frozen_moment_drift_max_abs",
        "actor_preview_normalizer_drift_max_abs",
    )
    if any(row.get(key) != 0.0 for key in frozen_fields):
        raise ValueError(f"checkpoint {step} changed frozen parent state")
    for key in ("actor_preview_gradient_norm", "actor_preview_update_norm"):
        value = float(row.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"checkpoint {step} has invalid residual telemetry")
    counts = row.get("actor_cagrad_bin_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != 5
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in counts
        )
    ):
        raise ValueError(f"checkpoint {step} has invalid CAGrad coverage")
    vector_fields = (
        "actor_cagrad_bin_gradient_norms",
        "actor_cagrad_bin_losses",
        "actor_cagrad_weights",
    )
    vectors = {key: row.get(key) for key in vector_fields}
    matrix_fields = ("actor_cagrad_gram_matrix", "actor_cagrad_cosine_matrix")
    matrices = {key: row.get(key) for key in matrix_fields}
    scalar_fields = (
        "actor_cagrad_objective",
        "actor_cagrad_dual_gap",
        "actor_cagrad_uniform_combined_cosine",
        "actor_cagrad_combined_norm",
    )
    if (
        row.get("actor_cagrad_valid") is not True
        or any(
            not isinstance(values, list)
            or len(values) != 5
            or any(not math.isfinite(float(value)) for value in values)
            for values in vectors.values()
        )
        or any(
            not isinstance(matrix, list)
            or len(matrix) != 5
            or any(
                not isinstance(matrix_row, list)
                or len(matrix_row) != 5
                or any(not math.isfinite(float(value)) for value in matrix_row)
                for matrix_row in matrix
            )
            for matrix in matrices.values()
        )
        or any(
            not math.isfinite(float(row.get(key, math.nan))) for key in scalar_fields
        )
        or any(float(value) < 0.0 for value in vectors["actor_cagrad_weights"])
        or not math.isclose(
            sum(float(value) for value in vectors["actor_cagrad_weights"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        or float(row["actor_cagrad_combined_norm"]) <= 0.0
        or row.get("torso_wrench_assistance_valid") is not True
        or row.get("actor_preview_valid") is not True
    ):
        raise ValueError(f"checkpoint {step} has invalid CAGrad telemetry")


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    """Validate the exact 32-update RMR-noise treatment artifacts."""
    run_directory = run_directory.resolve()
    hparams = json.loads((run_directory / "hparams.json").read_text())
    expected_hparams = {
        "total_steps": RMR_NOISE_END_STEP,
        "action_noise_std_start": action_noise_std_hparam(RMR_ACTION_STD),
        "action_noise_std_end": action_noise_std_hparam(RMR_ACTION_STD),
        "action_noise_schedule_steps": RMR_NOISE_END_STEP,
        "allow_resume_action_noise_change": True,
        "actor_bootstrap_scale": 0.0,
        "allow_resume_actor_bootstrap_scale_change": True,
        "torso_wrench_assistance": True,
        "torso_wrench_assistance_start_step": _ASSISTANCE_PARENT_STEP,
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
        "seed": SEED,
    }
    for key, expected in expected_hparams.items():
        _require_equal(hparams, key, expected)
    if (RMR_NOISE_END_STEP - E008_SELECTED_STEP) // (
        hparams["effective_num_envs"] * hparams["unroll_length"]
    ) != 32:
        raise ValueError("training update count does not match 32")
    steps = expected_checkpoint_steps()
    checkpoint_names = {f"checkpoint_step_{step}.pkl" for step in steps}
    archived_checkpoints = tuple(run_directory.glob("checkpoint_step_*.pkl"))
    archived_checkpoint_names = {path.name for path in archived_checkpoints}
    if archived_checkpoint_names != checkpoint_names:
        raise ValueError(
            "dense checkpoint cadence must contain exactly four checkpoints"
        )
    if any(path.is_symlink() or not path.is_file() for path in archived_checkpoints):
        raise ValueError("dense checkpoint artifacts must be regular files")
    rows = json.loads((run_directory / "checkpoint_phase_metrics.json").read_text())
    if (
        not isinstance(rows, list)
        or len(rows) != len(steps)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("dense checkpoint telemetry must contain exactly four rows")
    rows_by_step = {row.get("step"): row for row in rows}
    if len(rows_by_step) != len(steps) or set(rows_by_step) != set(steps):
        raise ValueError(
            "dense checkpoint telemetry must contain exactly four unique steps"
        )
    for step in steps:
        _validate_cagrad_row(rows_by_step[step], step)
    return {
        "protocol": "g1-rmr-action-noise-continuation-training-v1",
        "valid": True,
        "run_directory": str(run_directory),
        "update_count": 32,
        "checkpoint_steps": list(steps),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
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
    _write_json_atomically(output_root / "rmr_action_noise_preflight.json", preflight)
    configure_jax()
    kwargs = build_rmr_action_noise_kwargs(
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
        output_root / "rmr_action_noise_training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
