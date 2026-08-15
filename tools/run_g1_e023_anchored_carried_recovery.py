"""Continue E023 with an anchored frozen recovery adapter and carried states."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_fresh_ppo_action_contract_walk import (
    validate_training_artifacts as validate_base_training_artifacts,
)
from tools.run_g1_rmr_noise_h24_continuation import (
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
    START_STEP,
)
from tools.run_g1_rmr_noise_h24_walk import (
    build_rmr_noise_h24_kwargs,
    validate_preflight as validate_e023_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


TRANSITIONS_PER_UPDATE = 512 * 24
CONTINUATION_UPDATES = 64
CHECKPOINT_UPDATES = 8
CHECKPOINT_INTERVAL = CHECKPOINT_UPDATES * TRANSITIONS_PER_UPDATE
CONTINUATION_END_STEP = START_STEP + CONTINUATION_UPDATES * TRANSITIONS_PER_UPDATE
CARRIED_RESET_PROBABILITY = 0.25
ACTOR_POLICY_ANCHOR_WEIGHT = 1.0
E023_FLOORS = (116, 99, 67, 49, 24)
COMPLETE_SUFFIXES = (124, 99, 74, 49, 24)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the fixed eight-checkpoint recovery grid."""
    return tuple(
        range(
            START_STEP + CHECKPOINT_INTERVAL,
            CONTINUATION_END_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )


def build_anchored_carried_recovery_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
) -> dict[str, Any]:
    """Apply only the registered frozen-adapter carried-recovery treatment."""
    kwargs = build_rmr_noise_h24_kwargs(
        profile_name, reference_path, seed
    )
    kwargs.update(
        resume_from=str(Path(resume_from).resolve()),
        total_steps=CONTINUATION_END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        actor_residual_preview_adapter=True,
        allow_resume_actor_residual_preview_adapter_start=True,
        actor_policy_anchor_weight=ACTOR_POLICY_ANCHOR_WEIGHT,
        carried_reset_bank_path=str(Path(carried_bank).resolve()),
        carried_reset_probability=CARRIED_RESET_PROBABILITY,
        allow_resume_carried_reset_change=True,
    )
    return kwargs


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    carried_bank: Path,
    carried_bank_summary: Path,
    carried_bank_sha256: str,
    carried_bank_summary_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    """Bind clean source, E023 parent, and the immutable carried bank."""
    base = validate_e023_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    checkpoint = resume_from.resolve()
    hparams = checkpoint.with_name("hparams.json")
    bank = carried_bank.resolve()
    summary_path = carried_bank_summary.resolve()
    expected_files = (
        (checkpoint, EXPECTED_RESUME_SHA256, "E023 checkpoint"),
        (hparams, EXPECTED_RESUME_HPARAMS_SHA256, "E023 hparams"),
        (bank, carried_bank_sha256, "carried bank"),
        (summary_path, carried_bank_summary_sha256, "carried summary"),
    )
    for path, expected, label in expected_files:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("carried bank summary must be an object")
    expected_summary = {
        "valid": True,
        "protocol": "g1-e023-history-carried-reset-bank-v1",
        "rows": 48,
        "rows_per_source": [24, 24],
        "source_phases": [0, 50],
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "bank_sha256": carried_bank_sha256,
        "code_commit": code_commit,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("carried bank summary does not match registration")
    return {
        **base,
        "protocol": "g1-e023-anchored-carried-recovery-preflight-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams),
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "carried_bank": str(bank),
        "carried_bank_sha256": carried_bank_sha256,
        "carried_bank_summary": str(summary_path),
        "carried_bank_summary_sha256": carried_bank_summary_sha256,
        "bank_rows": 48,
        "start_step": START_STEP,
        "end_step": CONTINUATION_END_STEP,
        "additional_updates": CONTINUATION_UPDATES,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "scientific_delta": [
            "resume_from",
            "total_steps",
            "checkpoint_interval",
            "actor_residual_preview_adapter",
            "actor_policy_anchor_weight",
            "carried_reset_bank_path",
            "carried_reset_probability",
        ],
        "fresh_initialization": False,
    }


def validate_recovery_telemetry(
    rows: Sequence[Mapping[str, Any]], migration: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact migration and immutable-parent adapter updates."""
    migration_true = (
        "valid",
        "parent_parameters_exact",
        "parent_mu_exact",
        "parent_nu_exact",
        "optimizer_count_exact",
        "optimizer_outer_state_exact",
        "adapter_parameters_finite",
        "adapter_mu_zero",
        "adapter_nu_zero",
        "residual_action_zero",
        "reconstructed_parent_exact",
    )
    if (
        any(migration.get(key) is not True for key in migration_true)
        or migration.get("max_action_absolute_error") != 0.0
        or migration.get("max_action_relative_error") != 0.0
    ):
        raise ValueError("residual adapter migration is invalid")
    if [row.get("step") for row in rows] != list(expected_checkpoint_steps()):
        raise ValueError("recovery telemetry must match the exact checkpoint grid")
    rows_by_step = {row.get("step"): row for row in rows}
    for step in expected_checkpoint_steps():
        row = rows_by_step.get(step)
        if row is None:
            raise ValueError(f"checkpoint {step} recovery telemetry is missing")
        for key in (
            "actor_preview_frozen_parameter_drift_max_abs",
            "actor_preview_frozen_moment_drift_max_abs",
            "actor_preview_normalizer_drift_max_abs",
        ):
            if row.get(key) != 0.0:
                raise ValueError(f"checkpoint {step} has frozen drift")
        positive = (
            row.get("actor_preview_gradient_norm"),
            row.get("actor_preview_update_norm"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in positive
        ):
            raise ValueError(f"checkpoint {step} adapter update is invalid")
        anchor_error = row.get("actor_policy_anchor_squared_error")
        if (
            row.get("actor_preview_valid") is not True
            or row.get("actor_policy_anchor_valid") is not True
            or row.get("actor_policy_anchor_weight")
            != ACTOR_POLICY_ANCHOR_WEIGHT
            or isinstance(anchor_error, bool)
            or not isinstance(anchor_error, (int, float))
            or not math.isfinite(float(anchor_error))
            or float(anchor_error) < 0.0
        ):
            raise ValueError(f"checkpoint {step} anchored adapter is invalid")
    return {
        "valid": True,
        "protocol": "g1-e023-anchored-carried-recovery-telemetry-v1",
        "checkpoint_count": len(expected_checkpoint_steps()),
        "migration_valid": True,
    }


def select_checkpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select only componentwise E023-preserving replay-free candidates."""
    normalized = []
    for record in records:
        update = record.get("update")
        survival = record.get("survival")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update < 1
            or not isinstance(survival, (list, tuple))
            or len(survival) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in survival
            )
        ):
            raise ValueError("phase-grid selection record is invalid")
        vector = tuple(survival)
        normalized.append((update, vector))
    eligible = [
        (update, vector)
        for update, vector in normalized
        if all(value >= floor for value, floor in zip(vector, E023_FLOORS))
    ]
    if not eligible:
        return {
            "valid": True,
            "outcome": "anchored-carried-insufficient",
            "floors": list(E023_FLOORS),
            "eligible_updates": [],
            "selected_update": None,
            "selected_survival": None,
        }
    selected_update, selected_survival = max(
        eligible,
        key=lambda item: (
            min(item[1]),
            median(item[1]),
            sum(item[1]) / len(item[1]),
            -item[0],
        ),
    )
    solved = all(
        value >= target
        for value, target in zip(selected_survival, COMPLETE_SUFFIXES)
    )
    return {
        "valid": True,
        "outcome": (
            "anchored-carried-solves-walk"
            if solved
            else "anchored-carried-advances"
        ),
        "floors": list(E023_FLOORS),
        "eligible_updates": [update for update, _ in eligible],
        "selected_update": selected_update,
        "selected_survival": list(selected_survival),
    }


def validate_training_artifacts(
    run_directory: Path, *, expected_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Validate base SHAC evidence plus the frozen recovery boundary."""
    run_directory = run_directory.resolve()
    base = validate_base_training_artifacts(
        run_directory,
        expected_kwargs=expected_kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=CONTINUATION_END_STEP,
        protocol="g1-e023-anchored-carried-recovery-base-v1",
    )
    hparams = _load_json(run_directory / "hparams.json")
    rows = _load_json(run_directory / "checkpoint_phase_metrics.json")
    migration = _load_json(run_directory / "residual_adapter_migration.json")
    if not isinstance(hparams, dict) or not isinstance(migration, dict):
        raise ValueError("recovery training metadata must be objects")
    if not isinstance(rows, list):
        raise ValueError("recovery checkpoint telemetry must be a list")
    treatment = {
        "actor_residual_preview_adapter": True,
        "actor_residual_preview_hidden": 256,
        "actor_residual_preview_optimizer": "adam",
        "allow_resume_actor_residual_preview_adapter_start": True,
        "resume_residual_adapter_upgrade": True,
        "actor_policy_anchor_weight": ACTOR_POLICY_ANCHOR_WEIGHT,
        "carried_reset_probability": CARRIED_RESET_PROBABILITY,
        "carried_reset_bank_start": 0,
        "allow_resume_carried_reset_change": True,
        "actor_normalizer_frozen": True,
        "reference_reset_noise_scale": 0.0,
        "domain_randomization": False,
    }
    if any(hparams.get(key) != value for key, value in treatment.items()):
        raise ValueError("recovery training hparams do not match treatment")
    recovery = validate_recovery_telemetry(rows, migration)
    return {
        **base,
        **recovery,
        "protocol": "g1-e023-anchored-carried-recovery-training-v1",
        "treatment": treatment,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--carried-reset-bank", type=Path, required=True)
    parser.add_argument("--carried-reset-bank-sha256", required=True)
    parser.add_argument("--carried-reset-bank-summary", type=Path, required=True)
    parser.add_argument("--carried-reset-bank-summary-sha256", required=True)
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
        reference_path=args.reference_path.resolve(),
        resume_from=args.resume_from.resolve(),
        carried_bank=args.carried_reset_bank.resolve(),
        carried_bank_summary=args.carried_reset_bank_summary.resolve(),
        carried_bank_sha256=args.carried_reset_bank_sha256,
        carried_bank_summary_sha256=(
            args.carried_reset_bank_summary_sha256
        ),
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_anchored_carried_recovery_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        args.carried_reset_bank.resolve(),
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
    _write_json_atomically(
        output_root / "training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
