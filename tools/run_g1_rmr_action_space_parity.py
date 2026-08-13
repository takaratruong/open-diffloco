"""Train G1 SHAC from scratch with the RMR reference-delta action contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from src.algorithms.shac.algorithm import train
from src.core.rmr_action_noise import RMR_ACTION_STD
from src.envs.g1_tracking.environment import (
    DEFAULT_CONTROLLER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REFERENCE_PATH,
)
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_canonical_g1_shac import build_canonical_kwargs
from tools.run_g1_root_recovery_continuation import validate_runtime_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _git_output
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def build_rmr_action_space_parity_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict:
    """Return a fresh SHAC contract with exact RMR delta-action semantics."""
    kwargs = build_canonical_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        env_variant="g1_tracking_rmr_50hz_action_parity",
        action_scale=1.0,
        action_noise_std_start=1.0,
        action_noise_std_end=RMR_ACTION_STD,
        action_noise_schedule_steps=800_000,
        friction_range=(1.0, 1.0),
        mass_range=(1.0, 1.0),
        kp_range=(35.0, 35.0),
        kd_range=(0.5, 0.5),
        com_offset_range=(0.025, 0.05, 0.05),
        push_velocity_range=(0.0, 0.0),
        push_interval_s=2.0,
        reference_residual_scale=1.0,
        gradient_accumulation_steps=2,
        actor_cagrad=True,
        actor_cagrad_alpha=0.5,
        actor_cagrad_iterations=32,
        actor_phase_bin_count=5,
        actor_reference_lookahead_steps=(4, 8, 12),
        checkpoint_interval=196_608,
    )
    return kwargs


def build_parity_gate_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict:
    """Run one exact effective-512 H12 update before a long training launch."""
    kwargs = build_rmr_action_space_parity_kwargs(
        profile_name, reference_path, seed
    )
    kwargs.update(
        total_steps=6_144,
        checkpoint_interval=6_144,
        curriculum_grace=6_144,
        curriculum_steps=1,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=tuple(sorted(SOLVER_PROFILES)),
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
        default=Path("g1_rmr_action_space_parity_runs"),
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gate-only", action="store_true")
    return parser


def validate_preflight(
    *, repository: Path, reference_path: Path, code_commit: str
) -> dict[str, object]:
    """Bind a fresh parity run to clean code and immutable runtime assets."""
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    reference_path = reference_path.resolve()
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("reference SHA-256 does not match")
    assets = validate_runtime_assets(
        Path(DEFAULT_MODEL_PATH), Path(DEFAULT_CONTROLLER_PATH)
    )
    return {
        "protocol": "g1-rmr-action-space-parity-preflight-v1",
        "code_commit": head,
        "reference_path": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        **assets,
        "fresh_initialization": True,
        "environment_variant": "g1_tracking_rmr_50hz_action_parity",
        "normalized_action_clip": False,
        "joint_velocity_observation_noise": 0.5,
        "randomization_com_body_name": "torso_link",
        "randomization_uses_curriculum": False,
        "reference_residual_scale": 1.0,
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "remaining_rmr_randomization_gaps": [
            "friction-and-restitution-material-buckets",
            "joint-default-position-offsets",
            "pushes-disabled",
        ],
    }


def validate_gate_artifacts(run_directory: Path) -> dict[str, object]:
    """Fail closed unless the one-update run executed the parity contract."""
    run_directory = run_directory.resolve()
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected = {
        "total_steps": 6_144,
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "squash_actor_actions": False,
        "actor_observation_noise": True,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "friction_range": [1.0, 1.0],
        "mass_range": [1.0, 1.0],
        "com_offset_range": [0.025, 0.05, 0.05],
        "randomization_com_body_name": "torso_link",
        "randomization_uses_curriculum": False,
        "push_velocity_range": [0.0, 0.0],
        "action_noise_std_start": 1.0,
        "action_noise_std_end": np.asarray(RMR_ACTION_STD).tolist(),
        "actor_cagrad": True,
        "gradient_accumulation_steps": 2,
    }
    for key, value in expected.items():
        if hparams.get(key) != value:
            raise ValueError(f"gate hparams {key} does not match parity contract")
    checkpoint = run_directory / "checkpoint_step_006144.pkl"
    if not checkpoint.is_file():
        raise ValueError("gate checkpoint is missing")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("step") != 6_144:
        raise ValueError("gate checkpoint telemetry is incomplete")
    row = rows[0]
    counts = row.get("actor_cagrad_bin_counts")
    combined_norm = float(row.get("actor_cagrad_combined_norm", math.nan))
    if (
        row.get("actor_cagrad_valid") is not True
        or not isinstance(counts, list)
        or len(counts) != 5
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in counts)
        or not math.isfinite(combined_norm)
        or combined_norm <= 0.0
    ):
        raise ValueError("gate CAGrad telemetry is invalid")
    diagnostics = json.loads(
        (run_directory / "diag_log.json").read_text(encoding="utf-8")
    )
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("gate diagnostics are missing")
    final = diagnostics[-1]
    for key in ("actor_grad", "actor_update_norm"):
        value = float(final.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"gate {key} is not finite and positive")
    return {
        "protocol": "g1-rmr-action-space-parity-gate-validation-v1",
        "valid": True,
        "step": 6_144,
        "actor_cagrad_combined_norm": combined_norm,
        "actor_grad": float(final["actor_grad"]),
        "actor_update_norm": float(final["actor_update_norm"]),
    }


def execute(args: argparse.Namespace) -> Path:
    """Preflight and launch either the one-update gate or full fresh run."""
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )
    _write_json_atomically(
        output_root / "action_space_parity_preflight.json", preflight
    )
    configure_jax()
    builder = (
        build_parity_gate_kwargs
        if args.gate_only
        else build_rmr_action_space_parity_kwargs
    )
    kwargs = builder(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
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
    if args.gate_only:
        gate_validation = validate_gate_artifacts(run_directory)
        _write_json_atomically(
            output_root / "action_space_parity_gate_validation.json",
            gate_validation,
        )
    return run_directory


def main() -> None:
    run_directory = execute(build_parser().parse_args())
    print(run_directory)


if __name__ == "__main__":
    main()
