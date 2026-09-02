"""Capture current E002 actor-gradient cancellation in one diagnostic update."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Mapping

import jax
import numpy as np

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_dual_scale_root_position import (
    build_arm_kwargs,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
EFFECTIVE_NUM_ENVS = 512
UNROLL_LENGTH = 24
TRANSITIONS_PER_UPDATE = EFFECTIVE_NUM_ENVS * UNROLL_LENGTH
END_STEP = START_STEP + TRANSITIONS_PER_UPDATE
MICROBATCH_SIZES = (1, 8, 32, 64, 128, 512)
METRIC_KEYS = (
    "actor_grad_population_mean_norm",
    "actor_grad_population_rms_norm",
    "actor_grad_population_variance_trace",
    "actor_grad_population_cancellation_ratio",
    "actor_grad_population_noise_scale",
    "actor_grad_population_esnr",
)


def build_gradient_audit_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Continue exact E002 for one update without changing its objective."""

    kwargs = build_arm_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        kernel="exponential",
    )
    kwargs.update(
        total_steps=END_STEP,
        checkpoint_steps=(END_STEP,),
        diagnose=True,
    )
    return kwargs


def _finite_number(
    row: Mapping[str, object], name: str, *, minimum: float = 0.0
) -> float:
    value = row.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{name} is not a finite value at least {minimum}")
    return float(value)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=2e-4, abs_tol=2e-7)


def _load_single_metric_row(path: Path) -> dict[str, object]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{path.name} must contain exactly one metric row")
    return rows[0]


def load_gradient_capture_row(
    run_directory: Path, *, expected_step: int
) -> dict[str, object]:
    """Join checkpoint and diagnostic telemetry at one exact update step."""

    checkpoint_row = _load_single_metric_row(
        run_directory / "checkpoint_phase_metrics.json"
    )
    diagnostic_row = _load_single_metric_row(run_directory / "diag_log.json")
    if (
        checkpoint_row.get("step") != expected_step
        or diagnostic_row.get("step") != expected_step
    ):
        raise ValueError("checkpoint and diagnostic telemetry steps do not match")
    for name in ("actor_cagrad_valid", *METRIC_KEYS):
        if checkpoint_row.get(name) != diagnostic_row.get(name):
            raise ValueError(f"checkpoint and diagnostic {name} do not match")
    joined = dict(checkpoint_row)
    joined["actor_grad_finite_fraction"] = diagnostic_row.get(
        "actor_grad_finite_fraction"
    )
    return joined


def summarize_gradient_distribution(
    row: Mapping[str, object], *, expected_step: int
) -> dict[str, object]:
    """Validate the gradient moments and derive sampled-batch ESNR values."""

    if row.get("step") != expected_step:
        raise ValueError("gradient telemetry step does not match")
    if row.get("actor_cagrad_valid") is not True:
        raise ValueError("gradient telemetry requires a valid CAGrad reduction")
    finite_fraction = _finite_number(
        row, "actor_grad_finite_fraction", minimum=0.0
    )
    if finite_fraction != 1.0:
        raise ValueError("all per-environment actor gradients must be finite")

    values = {name: _finite_number(row, name) for name in METRIC_KEYS}
    mean_norm = values["actor_grad_population_mean_norm"]
    rms_norm = values["actor_grad_population_rms_norm"]
    variance = values["actor_grad_population_variance_trace"]
    cancellation = values["actor_grad_population_cancellation_ratio"]
    noise_scale = values["actor_grad_population_noise_scale"]
    population_esnr = values["actor_grad_population_esnr"]
    if mean_norm <= 0.0 or rms_norm <= 0.0 or rms_norm + 1e-7 < mean_norm:
        raise ValueError("gradient population norms are invalid")

    expected_variance = max(rms_norm * rms_norm - mean_norm * mean_norm, 0.0)
    expected_cancellation = mean_norm / rms_norm
    expected_noise_scale = expected_variance / (mean_norm * mean_norm)
    expected_population_esnr = EFFECTIVE_NUM_ENVS / max(
        expected_noise_scale, 1e-12
    )
    if not (
        _close(variance, expected_variance)
        and _close(cancellation, expected_cancellation)
        and _close(noise_scale, expected_noise_scale)
        and _close(population_esnr, expected_population_esnr)
        and 0.0 <= cancellation <= 1.0 + 1e-6
    ):
        raise ValueError("gradient population moment identities do not close")

    estimated_batch_esnr = {
        str(size): size / max(noise_scale, 1e-12)
        for size in MICROBATCH_SIZES
    }
    material_cancellation = cancellation < 0.75
    return {
        "protocol": "g1-gradient-population-audit-v1",
        "valid": True,
        "source_step": START_STEP,
        "captured_update_step": expected_step,
        "effective_num_envs": EFFECTIVE_NUM_ENVS,
        "unroll_length": UNROLL_LENGTH,
        "gradient_semantic": "per-environment-clipped-pathwise-H24",
        "aggregation_semantic": "five-phase-bin-cagrad",
        "population_metrics": values,
        "estimated_batch_esnr": estimated_batch_esnr,
        "sampled_batch_8_esnr_below_one": estimated_batch_esnr["8"] < 1.0,
        "material_cancellation_threshold": 0.75,
        "classification": (
            "population-mean-has-material-cancellation"
            if material_cancellation
            else "population-mean-retains-most-rms-signal"
        ),
        "optimizer_update_retained": False,
        "retained_policy": None,
    }


def _finite_tree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(tree)
    )


def validate_training_artifacts(
    run_directory: Path, *, source_checkpoint: Path
) -> dict[str, object]:
    """Validate the one-update capture and return its distribution summary."""

    hparams_path = run_directory / "hparams.json"
    metrics_path = run_directory / "checkpoint_phase_metrics.json"
    diagnostic_path = run_directory / "diag_log.json"
    checkpoint_path = run_directory / f"checkpoint_step_{END_STEP}.pkl"
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    required_hparams = {
        "total_steps": END_STEP,
        "checkpoint_steps": [END_STEP],
        "effective_num_envs": EFFECTIVE_NUM_ENVS,
        "num_envs": 256,
        "gradient_accumulation_steps": 2,
        "unroll_length": UNROLL_LENGTH,
        "actor_cagrad": True,
        "actor_per_env_grad_clip": 1.0,
        "tracking_anchor_position_kernel": "exponential",
        "tracking_root_velocity_weight": 1.0,
        "actor_bootstrap_scale": 0.0,
        "ahac": False,
    }
    if any(hparams.get(name) != value for name, value in required_hparams.items()):
        raise ValueError("gradient audit hparams do not preserve exact E002")

    row = load_gradient_capture_row(run_directory, expected_step=END_STEP)
    summary = summarize_gradient_distribution(row, expected_step=END_STEP)

    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    with checkpoint_path.open("rb") as stream:
        candidate = pickle.load(stream)
    if (
        int(source.step) != START_STEP
        or int(candidate.step) != END_STEP
        or not _finite_tree(candidate)
    ):
        raise ValueError("gradient audit checkpoint state is invalid")
    return {
        "protocol": "g1-gradient-population-training-validation-v1",
        "valid": True,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "diagnostic_checkpoint_sha256": sha256_file(checkpoint_path),
        "hparams_sha256": sha256_file(hparams_path),
        "checkpoint_metrics_sha256": sha256_file(metrics_path),
        "diagnostic_log_sha256": sha256_file(diagnostic_path),
        "run_directory": str(run_directory.resolve()),
        "distribution": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("gradient population audit seed must equal zero")
    repository = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_checkpoint = args.resume_from.resolve()
    reference_path = args.reference_path.resolve()
    preflight = validate_preflight(
        repository=repository,
        checkpoint=source_checkpoint,
        reference=reference_path,
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-gradient-population-preflight-v1",
        start_step=START_STEP,
        end_step=END_STEP,
        diagnostic_updates=1,
        optimizer_update_retained=False,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    kwargs = build_gradient_audit_kwargs(
        args.solver_profile,
        reference_path,
        args.seed,
        source_checkpoint,
    )
    configure_jax()
    previous = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory, source_checkpoint=source_checkpoint
    )
    validation_path = output_root / "training_validation.json"
    distribution_path = output_root / "gradient_distribution.json"
    _write_json_atomically(validation_path, validation)
    _write_json_atomically(distribution_path, validation["distribution"])
    completion = {
        "protocol": "g1-gradient-population-completion-v1",
        "valid": True,
        "classification": validation["distribution"]["classification"],
        "optimizer_update_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "training_validation.json": sha256_file(validation_path),
            "gradient_distribution.json": sha256_file(distribution_path),
            "checkpoint_phase_metrics.json": validation[
                "checkpoint_metrics_sha256"
            ],
            "diag_log.json": validation["diagnostic_log_sha256"],
            "diagnostic_checkpoint": validation[
                "diagnostic_checkpoint_sha256"
            ],
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(run_directory)


if __name__ == "__main__":
    main()
