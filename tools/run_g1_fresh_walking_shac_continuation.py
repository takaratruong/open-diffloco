"""Continue the fresh walking actor with the unchanged native SHAC recipe."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_fresh_fixed_noise_training import (
    build_fresh_fixed_noise_kwargs,
    validate_per_env_gradient_clip_telemetry,
    validate_preflight as validate_fresh_preflight,
)
from tools.run_g1_rmr_action_noise_continuation import _validate_checkpoint
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 786_432
CONTINUATION_END_STEP = 3_145_728
CHECKPOINT_INTERVAL = 393_216
EXPECTED_RESUME_SHA256 = (
    "489d5e989b0554146f8c151b45b4ebe996a1d42ce129584e099ff5f609d4857e"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "25b8199fd8ae3cfc013c42450732a2e45bc5cd6793c11890c15937fb3aad98b6"
)
EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    return tuple(
        range(
            START_STEP + CHECKPOINT_INTERVAL,
            CONTINUATION_END_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )


def build_fresh_walking_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Change only the resume source, absolute endpoint, and archive cadence."""
    kwargs = build_fresh_fixed_noise_kwargs(
        profile_name,
        reference_path,
        seed,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    kwargs.update(
        resume_from=str(Path(resume_from).resolve()),
        total_steps=CONTINUATION_END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_fresh_walking_shac_continuation_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    code_commit: str,
) -> dict[str, Any]:
    base = validate_fresh_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
        expected_reference_sha256=EXPECTED_REFERENCE_SHA256,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    checkpoint = resume_from.resolve()
    hparams = checkpoint.with_name("hparams.json")
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != EXPECTED_RESUME_SHA256
    ):
        raise ValueError("fresh walking resume checkpoint SHA-256 does not match")
    if (
        not hparams.is_file()
        or sha256_file(hparams) != EXPECTED_RESUME_HPARAMS_SHA256
    ):
        raise ValueError("fresh walking resume hparams SHA-256 does not match")
    return {
        **base,
        "protocol": "g1-fresh-walking-shac-continuation-preflight-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "start_step": START_STEP,
        "end_step": CONTINUATION_END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "scientific_delta": ["resume_from", "total_steps", "checkpoint_interval"],
    }


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected = build_fresh_fixed_noise_kwargs(
        "g1-4x5",
        Path(hparams["reference_path"]),
        int(hparams["seed"]),
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    expected["total_steps"] = CONTINUATION_END_STEP
    unpersisted_execution_keys = {"checkpoint_interval", "diagnose", "resume_from"}
    for key in sorted(set(expected) - unpersisted_execution_keys):
        expected_value = expected[key]
        if isinstance(expected_value, tuple):
            expected_value = list(expected_value)
        if hparams.get(key) != expected_value:
            raise ValueError(f"continuation hparams drifted at {key}")

    expected_steps = list(expected_checkpoint_steps())
    expected_names = {
        f"checkpoint_step_{step}.pkl" for step in expected_steps
    }
    checkpoint_paths = tuple(run_directory.glob("checkpoint_step_*.pkl"))
    if (
        {path.name for path in checkpoint_paths} != expected_names
        or any(path.is_symlink() or not path.is_file() for path in checkpoint_paths)
    ):
        raise ValueError("continuation checkpoint archive set is not exact")
    checkpoint_sha256_by_step = {
        str(step): _validate_checkpoint(
            run_directory / f"checkpoint_step_{step}.pkl", step
        )
        for step in expected_steps
    }
    final_sha256 = checkpoint_sha256_by_step[str(CONTINUATION_END_STEP)]
    for name in ("checkpoint_latest.pkl", "policy_final.pkl"):
        path = run_directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or _validate_checkpoint(path, CONTINUATION_END_STEP) != final_sha256
        ):
            raise ValueError(f"{name} does not match the final checkpoint")

    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected_steps)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("continuation telemetry must contain six rows")
    rows_by_step = {int(row["step"]): row for row in rows}
    if len(rows_by_step) != len(expected_steps) or sorted(rows_by_step) != expected_steps:
        raise ValueError("continuation checkpoint telemetry cadence is incomplete")
    for step in expected_steps:
        row = rows_by_step[step]
        vectors = {
            key: np.asarray(row.get(key), dtype=np.float64)
            for key in (
                "actor_cagrad_bin_counts",
                "actor_cagrad_bin_gradient_norms",
                "actor_cagrad_bin_losses",
                "actor_cagrad_weights",
            )
        }
        matrices = {
            key: np.asarray(row.get(key), dtype=np.float64)
            for key in (
                "actor_cagrad_gram_matrix",
                "actor_cagrad_cosine_matrix",
            )
        }
        scalar_keys = (
            "actor_cagrad_objective",
            "actor_cagrad_dual_gap",
            "actor_cagrad_uniform_combined_cosine",
            "actor_cagrad_combined_norm",
        )
        if (
            row.get("actor_cagrad_valid") is not True
            or any(value.shape != (5,) for value in vectors.values())
            or any(value.shape != (5, 5) for value in matrices.values())
            or any(not np.isfinite(value).all() for value in vectors.values())
            or any(not np.isfinite(value).all() for value in matrices.values())
            or np.any(vectors["actor_cagrad_bin_counts"] <= 0.0)
            or np.any(vectors["actor_cagrad_bin_gradient_norms"] < 0.0)
            or np.any(vectors["actor_cagrad_weights"] < 0.0)
            or not math.isclose(
                float(vectors["actor_cagrad_weights"].sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
            or any(
                not math.isfinite(float(row.get(key, math.nan)))
                for key in scalar_keys
            )
            or float(row["actor_cagrad_combined_norm"]) <= 0.0
            or row.get("actor_bootstrap_scale_current") != 0.0
            or row.get("action_noise_current") != 0.2
        ):
            raise ValueError(f"checkpoint {step} CAGrad telemetry is invalid")
        validate_per_env_gradient_clip_telemetry(
            row, actor_per_env_grad_clip=1.0
        )
    return {
        "protocol": "g1-fresh-walking-shac-continuation-training-v1",
        "valid": True,
        "run_directory": str(run_directory),
        "checkpoint_steps": expected_steps,
        "checkpoint_sha256_by_step": checkpoint_sha256_by_step,
        "final_checkpoint_sha256": final_sha256,
    }


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        resume_from=args.resume_from,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_fresh_walking_continuation_kwargs(
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
