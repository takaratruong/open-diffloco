"""Train G1 SHAC from scratch with the RMR reference-delta action contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
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
from tools.evaluate_g1_tracking import validate_training_action_mean
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_canonical_g1_shac import build_canonical_kwargs
from tools.run_g1_root_recovery_continuation import validate_runtime_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _write_json_atomically,
)

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
        total_steps=786_432,
        env_variant="g1_tracking_rmr_50hz_action_parity",
        action_scale=1.0,
        action_noise_std_start=1.0,
        action_noise_std_end=RMR_ACTION_STD,
        action_noise_schedule_steps=800_000,
        domain_randomization=False,
        friction_range=(1.0, 1.0),
        mass_range=(1.0, 1.0),
        kp_range=(35.0, 35.0),
        kd_range=(0.5, 0.5),
        com_offset_range=(0.0, 0.0, 0.0),
        push_velocity_range=(0.0, 0.0),
        push_interval_s=2.0,
        actor_observation_noise=False,
        reference_reset_noise_scale=0.0,
        reference_residual_scale=1.0,
        gradient_accumulation_steps=2,
        actor_cagrad=True,
        actor_cagrad_alpha=0.5,
        actor_cagrad_iterations=32,
        actor_phase_bin_count=5,
        actor_reference_lookahead_steps=(4, 8, 12),
        checkpoint_interval=98_304,
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


def build_decoupled_exploration_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict:
    """Use a tanh-bounded mean with unclipped reparameterized exploration."""
    kwargs = build_rmr_action_space_parity_kwargs(
        profile_name, reference_path, seed
    )
    kwargs["env_variant"] = (
        "g1_tracking_rmr_50hz_decoupled_exploration"
    )
    return kwargs


def build_decoupled_gate_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict:
    kwargs = build_decoupled_exploration_kwargs(
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
    parser.add_argument("--decoupled-exploration", action="store_true")
    return parser


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    code_commit: str,
    env_variant: str = "g1_tracking_rmr_50hz_action_parity",
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
        "environment_variant": env_variant,
        "normalized_action_clip": False,
        "joint_velocity_observation_noise": 0.0,
        "exact_reference_resets": True,
        "randomization_com_body_name": "torso_link",
        "randomization_uses_curriculum": False,
        "domain_randomization": False,
        "com_offset_range": [0.0, 0.0, 0.0],
        "push_velocity_range": [0.0, 0.0],
        "reference_residual_scale": 1.0,
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "remaining_rmr_randomization_gaps": [
            "friction-and-restitution-material-buckets",
            "joint-default-position-offsets",
            "pushes-disabled",
            "torso-com-randomization-disabled",
        ],
    }


def validate_gate_artifacts(
    run_directory: Path,
    *,
    env_variant: str = "g1_tracking_rmr_50hz_action_parity",
) -> dict[str, object]:
    """Fail closed unless the one-update run executed the parity contract."""
    run_directory = run_directory.resolve()
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected = {
        "total_steps": 6_144,
        "env_variant": env_variant,
        "squash_actor_actions": False,
        "actor_observation_noise": False,
        "reference_reset_noise_scale": 0.0,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "friction_range": [1.0, 1.0],
        "mass_range": [1.0, 1.0],
        "com_offset_range": [0.0, 0.0, 0.0],
        "domain_randomization": False,
        "randomization_com_body_name": "torso_link",
        "randomization_uses_curriculum": False,
        "push_velocity_range": [0.0, 0.0],
        "action_noise_std_start": 1.0,
        "action_noise_std_end": np.asarray(RMR_ACTION_STD).tolist(),
        "actor_cagrad": True,
        "gradient_accumulation_steps": 2,
    }
    if env_variant == "g1_tracking_rmr_50hz_decoupled_exploration":
        expected.update(
            squash_actor_mean=True,
            clip_sampled_actor_actions=False,
        )
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
    result = {
        "protocol": "g1-rmr-action-space-parity-gate-validation-v1",
        "valid": True,
        "step": 6_144,
        "actor_cagrad_combined_norm": combined_norm,
        "actor_grad": float(final["actor_grad"]),
        "actor_update_norm": float(final["actor_update_norm"]),
    }
    if env_variant == "g1_tracking_rmr_50hz_decoupled_exploration":
        rollout = run_directory / "gate_training_rollout"
        summary = json.loads(
            (rollout / "summary.json").read_text(encoding="utf-8")
        )
        checkpoint_sha = sha256_file(checkpoint)
        if (
            summary.get("training_distribution_rollout") is not True
            or summary.get("training_checkpoint_step") != 6_144
            or summary.get("training_exact_reset_phase") != 0
            or summary.get("checkpoint_sha256") != checkpoint_sha
            or summary.get("steps") != 12
        ):
            raise ValueError("bounded-mean gate rollout provenance is invalid")
        with np.load(rollout / "training_action_noise.npz") as archive:
            action_mean = np.asarray(archive["action_mean"])
            if action_mean.shape != (12, 29):
                raise ValueError("bounded-mean gate action tape must be H12x29")
            validate_training_action_mean(action_mean)
        result.update(
            bounded_mean_rollout_valid=True,
            bounded_mean_max_abs=float(np.max(np.abs(action_mean))),
            bounded_mean_checkpoint_sha256=checkpoint_sha,
        )
    return result


def render_decoupled_gate_rollout(
    *, repository: Path, run_directory: Path
) -> None:
    """Replay one checkpoint-loaded H12 slice before gate publication."""
    checkpoint = run_directory / "checkpoint_step_006144.pkl"
    command = [
        sys.executable,
        str(repository / "tools/evaluate_g1_tracking.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(run_directory / "gate_training_rollout"),
        "--training-distribution-rollout",
        "--disable-training-observation-noise",
        "--exact-training-reset-phase",
        "0",
        "--max-steps",
        "12",
        "--seed",
        "0",
    ]
    subprocess.run(command, cwd=repository, check=True)


def execute(args: argparse.Namespace) -> Path:
    """Preflight and launch either the one-update gate or full fresh run."""
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
        env_variant=(
            "g1_tracking_rmr_50hz_decoupled_exploration"
            if getattr(args, "decoupled_exploration", False)
            else "g1_tracking_rmr_50hz_action_parity"
        ),
    )
    _write_json_atomically(
        output_root / "action_space_parity_preflight.json", preflight
    )
    configure_jax()
    if getattr(args, "decoupled_exploration", False):
        builder = (
            build_decoupled_gate_kwargs
            if args.gate_only
            else build_decoupled_exploration_kwargs
        )
    else:
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
        if getattr(args, "decoupled_exploration", False):
            render_decoupled_gate_rollout(
                repository=repository,
                run_directory=run_directory,
            )
            gate_validation = validate_gate_artifacts(
                run_directory,
                env_variant=kwargs["env_variant"],
            )
        else:
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
