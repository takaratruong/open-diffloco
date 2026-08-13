"""Train a fresh bounded G1 actor with fixed 0.2 action noise."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import (
    DEFAULT_CONTROLLER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REFERENCE_PATH,
)
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_canonical_g1_shac import build_canonical_kwargs
from tools.run_g1_root_recovery_continuation import validate_runtime_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _write_json_atomically,
)

TOTAL_STEPS = 786_432
CHECKPOINT_INTERVAL = 98_304
ACTION_NOISE_STD = 0.2
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def build_fresh_fixed_noise_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Build the immutable fresh actor diagnostic contract."""
    kwargs = build_canonical_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        total_steps=TOTAL_STEPS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        env_variant="g1_tracking_rmr_50hz_source_step",
        resume_from=None,
        action_noise_std_start=ACTION_NOISE_STD,
        action_noise_std_end=ACTION_NOISE_STD,
        action_noise_schedule_steps=TOTAL_STEPS,
        actor_observation_noise=False,
        reference_reset_noise_scale=0.0,
        reference_root_reset_noise_multiplier=1.0,
        reference_root_reset_noise_probability=0.0,
        carried_reset_bank_path=None,
        carried_reset_probability=0.0,
        domain_randomization=False,
        friction_range=(1.0, 1.0),
        mass_range=(1.0, 1.0),
        kp_range=(35.0, 35.0),
        kd_range=(0.5, 0.5),
        com_offset_range=(0.0, 0.0, 0.0),
        push_velocity_range=(0.0, 0.0),
        push_interval_s=1e9,
        terrain=False,
        terrain_bump_std=0.0,
        zero_difficulty_frac=1.0,
        curriculum_grace=TOTAL_STEPS,
        curriculum_steps=1,
        torso_wrench_assistance=False,
        actor_torso_wrench_assistance_conditioning=False,
        actor_observe_torso_wrench_assistance=False,
        reference_residual_control=True,
        reference_residual_scale=0.5,
        actor_hidden=(512, 256, 128),
        actor_layer_norm=True,
        actor_zero_output=True,
        gradient_accumulation_steps=2,
        actor_cagrad=True,
        actor_cagrad_alpha=0.5,
        actor_cagrad_iterations=32,
        actor_phase_bin_count=5,
        actor_reference_lookahead_steps=(4, 8, 12),
        actor_reference_preview_mode="delta",
        actor_bootstrap_scale=0.0,
        actor_bootstrap_delay_steps=0,
    )
    return kwargs


def build_gate_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Use the exact contract for one effective-512 H12 update."""
    kwargs = build_fresh_fixed_noise_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        total_steps=6_144,
        checkpoint_interval=6_144,
        curriculum_grace=6_144,
        action_noise_schedule_steps=6_144,
    )
    return kwargs


def validate_preflight(
    *, repository: Path, reference_path: Path, code_commit: str
) -> dict[str, Any]:
    """Bind code and runtime assets before allocating a GPU."""
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or code_commit != head:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    reference_path = reference_path.resolve()
    if not reference_path.is_file():
        raise ValueError("reference file is missing")
    if sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference SHA-256 does not match")
    assets = validate_runtime_assets(
        Path(DEFAULT_MODEL_PATH), Path(DEFAULT_CONTROLLER_PATH)
    )
    return {
        "protocol": "g1-fresh-fixed-020-preflight-v1",
        "code_commit": head,
        "reference_path": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        **assets,
        "fresh_initialization": True,
        "action_noise_std": ACTION_NOISE_STD,
        "observation_noise": False,
        "reset_noise": False,
        "domain_randomization": False,
        "pushes": False,
        "torso_assistance": False,
    }


def validate_training_artifacts(
    run_directory: Path, *, gate_only: bool
) -> dict[str, Any]:
    """Reject incomplete, nonfinite, or contract-drifting training output."""
    run_directory = run_directory.resolve()
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected = build_gate_kwargs(
        "g1-4x5", hparams["reference_path"], int(hparams["seed"])
    ) if gate_only else build_fresh_fixed_noise_kwargs(
        "g1-4x5", hparams["reference_path"], int(hparams["seed"])
    )
    persisted = {
        "total_steps",
        "env_variant",
        "action_noise_std_start",
        "action_noise_std_end",
        "action_noise_schedule_steps",
        "actor_observation_noise",
        "reference_reset_noise_scale",
        "reference_root_reset_noise_probability",
        "carried_reset_probability",
        "domain_randomization",
        "friction_range",
        "mass_range",
        "kp_range",
        "kd_range",
        "com_offset_range",
        "push_velocity_range",
        "terrain_bump_std",
        "torso_wrench_assistance",
        "reference_residual_control",
        "reference_residual_scale",
        "actor_hidden",
        "actor_layer_norm",
        "actor_zero_output",
        "gradient_accumulation_steps",
        "actor_cagrad",
        "actor_phase_bin_count",
        "actor_reference_lookahead_steps",
        "actor_reference_preview_mode",
        "actor_bootstrap_scale",
    }
    for key in persisted:
        expected_value = expected[key]
        if isinstance(expected_value, tuple):
            expected_value = list(expected_value)
        if hparams.get(key) != expected_value:
            raise ValueError(f"training hparams drifted at {key}")
    checkpoint_steps = (
        [6_144]
        if gate_only
        else list(range(CHECKPOINT_INTERVAL, TOTAL_STEPS + 1, CHECKPOINT_INTERVAL))
    )
    for step in checkpoint_steps:
        if not (run_directory / f"checkpoint_step_{step:06d}.pkl").is_file():
            raise ValueError(f"checkpoint {step} is missing")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if [int(row["step"]) for row in rows] != checkpoint_steps:
        raise ValueError("checkpoint telemetry cadence is incomplete")
    for row in rows:
        counts = row.get("actor_cagrad_bin_counts")
        if (
            row.get("actor_cagrad_valid") is not True
            or not isinstance(counts, list)
            or len(counts) != 5
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in counts
            )
        ):
            raise ValueError("CAGrad telemetry is invalid")
    return {
        "protocol": "g1-fresh-fixed-020-training-validation-v1",
        "valid": True,
        "gate_only": gate_only,
        "checkpoint_steps": checkpoint_steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("g1_fresh_fixed_020_runs")
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gate-only", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "fresh_fixed_020_preflight.json", preflight)
    configure_jax()
    builder = build_gate_kwargs if args.gate_only else build_fresh_fixed_noise_kwargs
    kwargs = builder(args.solver_profile, args.reference_path.resolve(), args.seed)
    profile = get_solver_profile(args.solver_profile)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(profile):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory, gate_only=args.gate_only
    )
    _write_json_atomically(
        output_root / "fresh_fixed_020_training_validation.json", validation
    )
    return run_directory


def main() -> None:
    print(execute(build_parser().parse_args()))


if __name__ == "__main__":
    main()
