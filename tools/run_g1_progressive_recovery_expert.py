"""Run the guarded E029 compact-support recovery discriminator."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.build_g1_e023_carried_reset_bank import validate_code_commit
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_lafan_anchored_carried_recovery import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
    build_lafan_recovery_kwargs,
)
from tools.run_g1_rmr_noise_h24_continuation import (
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
    START_STEP,
)
from tools.run_g1_root_recovery_continuation import validate_runtime_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


TRANSITIONS_PER_UPDATE = 512 * 24
END_STEP = START_STEP + 32 * TRANSITIONS_PER_UPDATE
EXPECTED_CHECKPOINT_STEPS = (
    START_STEP + 8 * TRANSITIONS_PER_UPDATE,
    START_STEP + 16 * TRANSITIONS_PER_UPDATE,
    END_STEP,
)
EXPECTED_SOURCE_BANK_SHA256 = (
    "d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5"
)


def build_progressive_recovery_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    targeted_bank: str | Path,
    support_path: str | Path,
    support_sha256: str,
) -> dict[str, Any]:
    """Build the single preregistered compact-support treatment."""
    kwargs = build_lafan_recovery_kwargs(
        profile_name, reference_path, seed, resume_from, targeted_bank
    )
    kwargs.update(
        total_steps=END_STEP,
        checkpoint_steps=EXPECTED_CHECKPOINT_STEPS,
        checkpoint_interval=END_STEP - START_STEP,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
        actor_policy_anchor_weight=1.0,
        actor_residual_preview_adapter=True,
        allow_resume_actor_residual_preview_adapter_start=True,
        actor_state_gated_recovery_support_path=str(
            Path(support_path).resolve()
        ),
        actor_state_gated_recovery_support_sha256=support_sha256,
        allow_resume_actor_state_gated_recovery_start=True,
        actor_cagrad=False,
        allow_resume_actor_cagrad_change=True,
        carried_reset_bank_path=str(Path(targeted_bank).resolve()),
        carried_reset_probability=0.25,
        carried_reset_bank_start=0,
        allow_resume_carried_reset_change=True,
        actor_bootstrap_scale=0.0,
        actor_observation_noise=False,
        reference_reset_noise_scale=0.0,
        domain_randomization=False,
        push_velocity_range=(0.0, 0.0),
        friction_range=(1.0, 1.0),
        torso_wrench_assistance=False,
    )
    return kwargs


def validate_recovery_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Fail closed on the exact three-checkpoint recovery telemetry."""
    if [row.get("step") for row in rows] != list(EXPECTED_CHECKPOINT_STEPS):
        raise ValueError("recovery checkpoint grid is incomplete")
    finite_keys = (
        "actor_preview_gradient_norm",
        "actor_preview_update_norm",
        "actor_recovery_gate_activation_fraction",
        "actor_recovery_gate_max",
        "actor_recovery_carried_activation_fraction",
        "actor_recovery_reference_activation_fraction",
        "actor_recovery_gated_residual_rms",
        "actor_recovery_gated_residual_max_abs",
    )
    for row in rows:
        if "actor_cagrad_valid" in row:
            raise ValueError("recovery treatment must not persist CAGrad telemetry")
        values = [row.get(key) for key in finite_keys]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise ValueError("recovery telemetry is nonfinite or negative")
        if (
            float(row["actor_preview_gradient_norm"]) <= 0.0
            or float(row["actor_preview_update_norm"]) <= 0.0
            or not 0.0 <= float(row["actor_recovery_gate_max"]) <= 1.0
            or not 0.0
            <= float(row["actor_recovery_gate_activation_fraction"])
            <= 1.0
            or row.get("actor_preview_valid") is not True
            or row.get("actor_recovery_valid") is not True
            or any(
                row.get(key) != 0.0
                for key in (
                    "actor_preview_frozen_parameter_drift_max_abs",
                    "actor_preview_frozen_moment_drift_max_abs",
                    "actor_preview_normalizer_drift_max_abs",
                )
            )
        ):
            raise ValueError("recovery telemetry violates the frozen boundary")
    return {
        "valid": True,
        "protocol": "g1-progressive-recovery-training-telemetry-v1",
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
    }


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    source_bank: Path,
    code_commit: str,
) -> dict[str, object]:
    """Bind clean code and the immutable E023/LAFAN/source evidence."""
    resolved_commit = validate_code_commit(repository, code_commit)
    hparams = resume_from.resolve().with_name("hparams.json")
    for path, expected, label in (
        (resume_from.resolve(), EXPECTED_RESUME_SHA256, "E023 checkpoint"),
        (hparams, EXPECTED_RESUME_HPARAMS_SHA256, "E023 hparams"),
        (
            reference_path.resolve(),
            EXPECTED_LAFAN_REFERENCE_SHA256,
            "LAFAN reference",
        ),
        (
            source_bank.resolve(),
            EXPECTED_SOURCE_BANK_SHA256,
            "E027 source bank",
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    parent_hparams = json.loads(hparams.read_text(encoding="utf-8"))
    assets = validate_runtime_assets(
        Path(str(parent_hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    return {
        **assets,
        "valid": True,
        "protocol": "g1-progressive-recovery-preflight-v1",
        "code_commit": resolved_commit,
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "source_bank_sha256": EXPECTED_SOURCE_BANK_SHA256,
        "start_step": START_STEP,
        "end_step": END_STEP,
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
    }


def _validate_checkpoint(path: Path, expected_step: int) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint {expected_step} is missing")
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if int(state.step) != expected_step:
        raise ValueError(f"checkpoint {expected_step} has the wrong step")
    return sha256_file(path)


def validate_training_artifacts(
    run_directory: Path, *, expected_kwargs: Mapping[str, Any]
) -> dict[str, object]:
    """Validate exact archives, hparams, migration, and recovery telemetry."""
    run_directory = run_directory.resolve()
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected_contract = {
        "total_steps": END_STEP,
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "actor_state_gated_recovery": True,
        "actor_state_gated_recovery_support_sha256": expected_kwargs[
            "actor_state_gated_recovery_support_sha256"
        ],
        "actor_cagrad": False,
        "actor_policy_anchor_weight": 1.0,
        "carried_reset_probability": 0.25,
        "actor_bootstrap_scale": 0.0,
        "reference_reset_noise_scale": 0.0,
        "actor_observation_noise": False,
        "domain_randomization": False,
    }
    if any(hparams.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("recovery training hparams drifted")
    paths = {
        step: run_directory / f"checkpoint_step_{step:06d}.pkl"
        for step in EXPECTED_CHECKPOINT_STEPS
    }
    actual_names = {path.name for path in run_directory.glob("checkpoint_step_*.pkl")}
    if actual_names != {path.name for path in paths.values()}:
        raise ValueError("recovery checkpoint archive set is not exact")
    hashes = {
        str(step): _validate_checkpoint(path, step)
        for step, path in paths.items()
    }
    final_hash = hashes[str(END_STEP)]
    for name in ("checkpoint_latest.pkl", "policy_final.pkl"):
        if _validate_checkpoint(run_directory / name, END_STEP) != final_hash:
            raise ValueError(f"{name} does not match the final checkpoint")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("recovery checkpoint telemetry is malformed")
    telemetry = validate_recovery_training_rows(rows)
    return {
        **telemetry,
        "protocol": "g1-progressive-recovery-training-v1",
        "run_directory": str(run_directory),
        "checkpoint_sha256_by_step": hashes,
        "final_checkpoint_sha256": final_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--targeted-bank", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--support-sha256", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
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
        resume_from=args.resume_from,
        source_bank=args.source_bank,
        code_commit=args.code_commit,
    )
    if not args.targeted_bank.is_file() or not args.support.is_file():
        raise ValueError("recovery support artifacts must be built before training")
    if sha256_file(args.support) != args.support_sha256:
        raise ValueError("recovery support SHA-256 does not match")
    preflight.update(
        {
            "targeted_bank": str(args.targeted_bank.resolve()),
            "targeted_bank_sha256": sha256_file(args.targeted_bank),
            "support": str(args.support.resolve()),
            "support_sha256": args.support_sha256,
        }
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_progressive_recovery_kwargs(
        args.solver_profile,
        args.reference_path,
        args.seed,
        args.resume_from,
        args.targeted_bank,
        args.support,
        args.support_sha256,
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory, expected_kwargs=kwargs
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
