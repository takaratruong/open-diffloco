"""Train fresh walking SHAC with the competent PPO action contract."""

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
    CHECKPOINT_INTERVAL,
    TOTAL_STEPS,
    build_fresh_fixed_noise_kwargs,
    validate_per_env_gradient_clip_telemetry,
    validate_preflight as validate_fresh_preflight,
)
from tools.run_g1_rmr_action_noise_continuation import _validate_checkpoint
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)
ENV_VARIANT = "g1_tracking_rmr_50hz_action_parity"


def expected_checkpoint_steps() -> tuple[int, ...]:
    return tuple(range(CHECKPOINT_INTERVAL, TOTAL_STEPS + 1, CHECKPOINT_INTERVAL))


def build_fresh_ppo_action_contract_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Change only E019's coupled physical action contract."""
    kwargs = build_fresh_fixed_noise_kwargs(
        profile_name,
        reference_path,
        seed,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    kwargs.update(
        env_variant=ENV_VARIANT,
        reference_residual_scale=1.0,
    )
    return kwargs


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Bind the clean runtime before allocating a GPU."""
    base = validate_fresh_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
        expected_reference_sha256=EXPECTED_REFERENCE_SHA256,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    return {
        **base,
        "protocol": "g1-fresh-ppo-action-contract-walk-preflight-v1",
        "environment_variant": ENV_VARIANT,
        "reference_residual_scale": 1.0,
        "squash_actor_mean": False,
        "clip_sampled_actor_actions": False,
        "scientific_delta": [
            "environment_variant",
            "reference_residual_scale",
        ],
    }


def _validate_cagrad_row(
    row: dict[str, Any],
    *,
    step: int,
    expected_action_noise: Any = 0.2,
) -> None:
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
    actual_noise = np.asarray(row.get("action_noise_current"), dtype=np.float64)
    expected_noise = np.asarray(expected_action_noise, dtype=np.float64)
    action_noise_valid = (
        actual_noise.shape == expected_noise.shape
        and np.isfinite(actual_noise).all()
        and np.allclose(actual_noise, expected_noise, rtol=0.0, atol=1e-7)
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
        or not action_noise_valid
    ):
        raise ValueError(f"checkpoint {step} CAGrad telemetry is invalid")
    validate_per_env_gradient_clip_telemetry(
        row, actor_per_env_grad_clip=1.0
    )


def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: dict[str, Any] | None = None,
    expected_steps: tuple[int, ...] | None = None,
    total_steps: int = TOTAL_STEPS,
    protocol: str = "g1-fresh-ppo-action-contract-walk-training-v1",
) -> dict[str, Any]:
    """Fail closed on drift, incomplete checkpoints, or nonfinite learning."""
    run_directory = run_directory.resolve()
    hparams_path = run_directory / "hparams.json"
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    contract = {
        "env_variant": ENV_VARIANT,
        "reference_residual_scale": 1.0,
        "squash_actor_mean": False,
        "clip_sampled_actor_actions": False,
    }
    if any(hparams.get(key) != value for key, value in contract.items()):
        raise ValueError("persisted action contract does not match treatment")

    if expected_kwargs is None:
        expected_kwargs = build_fresh_ppo_action_contract_kwargs(
            "g1-4x5", Path(hparams["reference_path"]), int(hparams["seed"])
        )
    unpersisted = {"checkpoint_interval", "diagnose", "resume_from"}
    for key in sorted(set(expected_kwargs) - unpersisted):
        expected_value = expected_kwargs[key]
        if isinstance(expected_value, tuple):
            expected_value = list(expected_value)
        elif hasattr(expected_value, "shape"):
            expected_value = np.asarray(expected_value).tolist()
        if hparams.get(key) != expected_value:
            raise ValueError(f"training hparams drifted at {key}")

    if expected_steps is None:
        expected_steps = expected_checkpoint_steps()
    expected_steps = list(expected_steps)
    if not expected_steps or expected_steps[-1] != total_steps:
        raise ValueError("checkpoint steps must end at total_steps")
    expected_names = {
        f"checkpoint_step_{step:06d}.pkl" for step in expected_steps
    }
    checkpoint_paths = tuple(run_directory.glob("checkpoint_step_*.pkl"))
    if (
        {path.name for path in checkpoint_paths} != expected_names
        or any(path.is_symlink() or not path.is_file() for path in checkpoint_paths)
    ):
        raise ValueError("checkpoint archive set is not exact")
    checkpoint_sha256_by_step = {
        str(step): _validate_checkpoint(
            run_directory / f"checkpoint_step_{step:06d}.pkl", step
        )
        for step in expected_steps
    }
    final_sha256 = checkpoint_sha256_by_step[str(total_steps)]
    for name in ("checkpoint_latest.pkl", "policy_final.pkl"):
        path = run_directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or _validate_checkpoint(path, total_steps) != final_sha256
        ):
            raise ValueError(f"{name} does not match the final checkpoint")

    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("checkpoint telemetry is malformed")
    rows_by_step = {int(row["step"]): row for row in rows}
    if len(rows_by_step) != len(expected_steps) or sorted(rows_by_step) != expected_steps:
        raise ValueError("checkpoint telemetry cadence is incomplete")
    steps_per_update = (
        int(expected_kwargs["num_envs"])
        * int(expected_kwargs["gradient_accumulation_steps"])
        * int(expected_kwargs["unroll_length"])
    )
    noise_start = np.asarray(
        expected_kwargs["action_noise_std_start"], dtype=np.float64
    )
    noise_end = np.asarray(
        expected_kwargs["action_noise_std_end"], dtype=np.float64
    )
    noise_schedule_steps = int(expected_kwargs["action_noise_schedule_steps"])
    for step in expected_steps:
        noise_progress = min(
            max((step - steps_per_update) / noise_schedule_steps, 0.0), 1.0
        )
        expected_action_noise = noise_start + noise_progress * (
            noise_end - noise_start
        )
        _validate_cagrad_row(
            rows_by_step[step],
            step=step,
            expected_action_noise=expected_action_noise,
        )

    diagnostics = json.loads(
        (run_directory / "diag_log.json").read_text(encoding="utf-8")
    )
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("actor diagnostics are missing")
    actor_gradient_norms = np.asarray(
        [row.get("actor_grad", math.nan) for row in diagnostics],
        dtype=np.float64,
    )
    actor_update_norms = np.asarray(
        [row.get("actor_update_norm", math.nan) for row in diagnostics],
        dtype=np.float64,
    )
    if (
        not np.isfinite(actor_gradient_norms).all()
        or not np.isfinite(actor_update_norms).all()
        or not np.any(actor_gradient_norms > 0.0)
        or not np.any(actor_update_norms > 0.0)
    ):
        raise ValueError("actor gradient/update diagnostics are invalid")

    return {
        "protocol": protocol,
        "valid": True,
        "run_directory": str(run_directory),
        "hparams_sha256": sha256_file(hparams_path),
        "checkpoint_steps": expected_steps,
        "checkpoint_sha256_by_step": checkpoint_sha256_by_step,
        "final_checkpoint_sha256": final_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_fresh_ppo_action_contract_walk_runs"),
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
    kwargs = build_fresh_ppo_action_contract_kwargs(
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
    validation = validate_training_artifacts(run_directory)
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
