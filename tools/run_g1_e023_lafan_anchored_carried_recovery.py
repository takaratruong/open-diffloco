"""Continue E023 on LAFAN with frozen-parent anchored carried recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.build_g1_e023_carried_reset_bank import (
    validate_code_commit,
    validate_e023_hparams,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_anchored_carried_recovery import (
    CONTINUATION_END_STEP,
    CONTINUATION_UPDATES,
    START_STEP,
    build_anchored_carried_recovery_kwargs,
    expected_checkpoint_steps,
    validate_training_artifacts as validate_short_training_artifacts,
)
from tools.run_g1_rmr_noise_h24_continuation import (
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
)
from tools.run_g1_root_recovery_continuation import validate_runtime_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


EXPECTED_PARENT_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)
EXPECTED_LAFAN_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
E023_LAFAN_FLOORS = (118, 63, 49, 39, 46)
COMPLETE_SUFFIXES = (499, 399, 299, 199, 99)


def build_lafan_recovery_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
) -> dict[str, Any]:
    """Apply the E026 architecture with one explicit LAFAN reference change."""
    kwargs = build_anchored_carried_recovery_kwargs(
        profile_name, reference_path, seed, resume_from, carried_bank
    )
    kwargs["allow_resume_reference_path_change"] = True
    return kwargs


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_reference_migration(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact authorized short-walk to LAFAN state reset."""
    expected = {
        "protocol": "g1-reference-path-migration-v1",
        "valid": True,
        "previous_reference_sha256": EXPECTED_PARENT_REFERENCE_SHA256,
        "requested_reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "environment_state_reinitialized": True,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("reference migration does not match treatment")
    return {"valid": True, "protocol": "g1-e023-lafan-reference-migration-v1"}


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
    """Bind the E023 parent, LAFAN reference, bank, assets, and clean code."""
    code_commit = validate_code_commit(repository, code_commit)
    checkpoint = resume_from.resolve()
    hparams = checkpoint.with_name("hparams.json")
    reference = reference_path.resolve()
    bank = carried_bank.resolve()
    summary_path = carried_bank_summary.resolve()
    for path, expected, label in (
        (checkpoint, EXPECTED_RESUME_SHA256, "E023 checkpoint"),
        (hparams, EXPECTED_RESUME_HPARAMS_SHA256, "E023 hparams"),
        (reference, EXPECTED_LAFAN_REFERENCE_SHA256, "LAFAN reference"),
        (bank, carried_bank_sha256, "carried bank"),
        (summary_path, carried_bank_summary_sha256, "carried summary"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    hparams_payload = _load_json(hparams)
    if not isinstance(hparams_payload, dict):
        raise ValueError("E023 hparams must be an object")
    validate_e023_hparams(hparams_payload)
    runtime_assets = validate_runtime_assets(
        Path(str(hparams_payload["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    summary = _load_json(summary_path)
    expected_summary = {
        "valid": True,
        "protocol": "g1-e023-lafan-history-carried-reset-bank-v1",
        "rows": 120,
        "rows_per_source": [24] * 5,
        "source_phases": [0, 100, 200, 300, 400],
        "source_survival": list(E023_LAFAN_FLOORS),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "bank_sha256": carried_bank_sha256,
    }
    if not isinstance(summary, dict) or any(
        summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise ValueError("LAFAN carried bank summary does not match registration")
    return {
        **runtime_assets,
        "valid": True,
        "protocol": "g1-e023-lafan-anchored-carried-preflight-v1",
        "code_commit": code_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams),
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference_path": str(reference),
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "carried_bank": str(bank),
        "carried_bank_sha256": carried_bank_sha256,
        "carried_bank_summary": str(summary_path),
        "carried_bank_summary_sha256": carried_bank_summary_sha256,
        "bank_rows": 120,
        "start_step": START_STEP,
        "end_step": CONTINUATION_END_STEP,
        "additional_updates": CONTINUATION_UPDATES,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "scientific_delta": [
            "reference_path",
            "actor_residual_preview_adapter",
            "actor_policy_anchor_weight",
            "carried_reset_bank_path",
            "carried_reset_probability",
        ],
    }


def select_checkpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select a LAFAN candidate only when every E023 suffix is preserved."""
    normalized: list[tuple[int, tuple[int, ...], tuple[bool, ...]]] = []
    for record in records:
        update = record.get("update")
        survival = record.get("survival")
        completed = record.get("completed_suffix")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update < 1
            or not isinstance(survival, (list, tuple))
            or len(survival) != 5
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in survival)
            or not isinstance(completed, (list, tuple))
            or len(completed) != 5
            or any(not isinstance(x, bool) for x in completed)
        ):
            raise ValueError("LAFAN phase-grid selection record is invalid")
        normalized.append((update, tuple(survival), tuple(completed)))
    eligible = [
        row
        for row in normalized
        if all(value >= floor for value, floor in zip(row[1], E023_LAFAN_FLOORS))
    ]
    if not eligible:
        return {
            "valid": True,
            "outcome": "lafan-carried-recovery-insufficient",
            "floors": list(E023_LAFAN_FLOORS),
            "eligible_updates": [],
            "selected_update": None,
            "selected_survival": None,
        }
    selected = max(
        eligible,
        key=lambda row: (min(row[1]), median(row[1]), sum(row[1]) / 5, -row[0]),
    )
    update, survival, completed = selected
    solved = all(completed) and all(
        value >= target for value, target in zip(survival, COMPLETE_SUFFIXES)
    )
    return {
        "valid": True,
        "outcome": (
            "lafan-carried-recovery-solves"
            if solved
            else "lafan-carried-recovery-advances"
        ),
        "floors": list(E023_LAFAN_FLOORS),
        "eligible_updates": [row[0] for row in eligible],
        "selected_update": update,
        "selected_survival": list(survival),
        "selected_completed_suffix": list(completed),
    }


def validate_training_artifacts(
    run_directory: Path, *, expected_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Validate the short-recipe invariants plus the reference migration."""
    base = validate_short_training_artifacts(
        run_directory, expected_kwargs=expected_kwargs
    )
    hparams = _load_json(run_directory / "hparams.json")
    migration = _load_json(run_directory / "reference_path_migration.json")
    if not isinstance(hparams, dict) or not isinstance(migration, dict):
        raise ValueError("LAFAN migration metadata must be objects")
    if (
        hparams.get("allow_resume_reference_path_change") is not True
        or hparams.get("reference_sha256") != EXPECTED_LAFAN_REFERENCE_SHA256
        or hparams.get("reference_states") != 500
        or hparams.get("reference_transitions") != 499
    ):
        raise ValueError("LAFAN training reference contract drifted")
    reference_migration = validate_reference_migration(migration)
    return {
        **base,
        **reference_migration,
        "protocol": "g1-e023-lafan-anchored-carried-training-v1",
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
        carried_bank_summary_sha256=args.carried_reset_bank_summary_sha256,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_lafan_recovery_kwargs(
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
    validation = validate_training_artifacts(run_directory, expected_kwargs=kwargs)
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
