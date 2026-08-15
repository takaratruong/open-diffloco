"""Train E041 with one bounded conflict-projected E036 teacher objective."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_anchored_carried_recovery import (
    expected_checkpoint_steps,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically
from tools.run_g1_zero_head_feature_transfer import (
    build_zero_head_feature_transfer_kwargs,
    validate_preflight as validate_zero_head_preflight,
    validate_training_artifacts as validate_zero_head_training_artifacts,
)


RECOVERY_TEACHER_SHA256 = (
    "203effe85e34794a76ebd344018e928f224d9cb8c9cedca9e2c4108f62343ad2"
)
RECOVERY_TEACHER_GRADIENT_RATIO = 0.5


def _zero_seed(value: str) -> int:
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("E042 requires seed exactly zero")
    return seed


def build_conflict_projected_recovery_teacher_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
    *,
    expert_path: str | Path,
    expert_sha256: str,
    teacher_path: str | Path,
    teacher_sha256: str,
) -> dict[str, Any]:
    """Apply only E042's registered teacher-gradient treatment to E041."""
    if teacher_sha256 != RECOVERY_TEACHER_SHA256:
        raise ValueError("registered recovery teacher SHA-256 does not match")
    kwargs = build_zero_head_feature_transfer_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        carried_bank,
        expert_path=expert_path,
        expert_sha256=expert_sha256,
    )
    kwargs.update(
        actor_recovery_teacher_dataset_path=str(Path(teacher_path).resolve()),
        actor_recovery_teacher_dataset_sha256=teacher_sha256,
        actor_recovery_teacher_gradient_ratio=(
            RECOVERY_TEACHER_GRADIENT_RATIO
        ),
        allow_resume_actor_recovery_teacher_change=True,
    )
    return kwargs


def validate_recovery_teacher_telemetry(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require finite bounded teacher evidence at every registered checkpoint."""
    if [row.get("step") for row in rows] != list(expected_checkpoint_steps()):
        raise ValueError("recovery teacher telemetry must match the checkpoint grid")
    scalar_names = (
        "loss",
        "raw_gradient_norm",
        "projected_gradient_norm",
        "applied_gradient_norm",
        "physics_gradient_norm",
        "combined_gradient_norm",
        "physics_dot",
        "physics_cosine",
        "applied_scale",
        "parent_gradient_max_abs",
    )
    nonnegative = {
        "loss",
        "raw_gradient_norm",
        "projected_gradient_norm",
        "applied_gradient_norm",
        "physics_gradient_norm",
        "combined_gradient_norm",
        "applied_scale",
        "parent_gradient_max_abs",
    }
    for row in rows:
        values: dict[str, float] = {}
        for name in scalar_names:
            value = row.get(f"actor_recovery_teacher_{name}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("recovery teacher telemetry must be finite")
            values[name] = float(value)
        if any(values[name] < 0.0 for name in nonnegative):
            raise ValueError("recovery teacher norm telemetry is invalid")
        if not -1.000001 <= values["physics_cosine"] <= 1.000001:
            raise ValueError("recovery teacher cosine telemetry is invalid")
        if values["applied_scale"] > 1.0:
            raise ValueError("recovery teacher applied scale is invalid")
        if values["parent_gradient_max_abs"] != 0.0:
            raise ValueError("recovery teacher parent gradient is not frozen")
        if values["applied_gradient_norm"] > (
            RECOVERY_TEACHER_GRADIENT_RATIO
            * values["physics_gradient_norm"]
            + 1e-7
        ):
            raise ValueError("recovery teacher gradient violates the norm cap")
        if row.get("actor_recovery_teacher_valid") is not True:
            raise ValueError("recovery teacher telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-conflict-projected-recovery-teacher-telemetry-v1",
        "checkpoint_count": len(rows),
    }


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{path.name} must contain JSON object rows")
    return payload


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    carried_bank: Path,
    carried_bank_summary: Path,
    carried_bank_sha256: str,
    carried_bank_summary_sha256: str,
    expert_checkpoint: Path,
    expert_sha256: str,
    teacher_dataset: Path,
    teacher_sha256: str,
    code_commit: str,
) -> dict[str, object]:
    """Bind E041 preflight plus the immutable E036 teacher dataset."""
    base = validate_zero_head_preflight(
        repository=repository,
        reference_path=reference_path,
        resume_from=resume_from,
        carried_bank=carried_bank,
        carried_bank_summary=carried_bank_summary,
        carried_bank_sha256=carried_bank_sha256,
        carried_bank_summary_sha256=carried_bank_summary_sha256,
        expert_checkpoint=expert_checkpoint,
        expert_sha256=expert_sha256,
        code_commit=code_commit,
    )
    teacher = teacher_dataset.resolve()
    if (
        teacher_sha256 != RECOVERY_TEACHER_SHA256
        or not teacher.is_file()
        or sha256_file(teacher) != RECOVERY_TEACHER_SHA256
    ):
        raise ValueError("E036 recovery teacher SHA-256 does not match")
    return {
        **base,
        "protocol": "g1-conflict-projected-recovery-teacher-preflight-v1",
        "teacher_dataset": str(teacher),
        "teacher_sha256": RECOVERY_TEACHER_SHA256,
        "teacher_rows": 416,
        "teacher_gradient_ratio": RECOVERY_TEACHER_GRADIENT_RATIO,
        "scientific_delta": [
            "actor_recovery_teacher_dataset_path",
            "actor_recovery_teacher_dataset_sha256",
            "actor_recovery_teacher_gradient_ratio",
            "allow_resume_actor_recovery_teacher_change",
        ],
    }


def validate_training_artifacts(
    run_directory: Path, *, expected_kwargs: dict[str, Any]
) -> dict[str, object]:
    """Require E041 validity plus exact teacher settings and telemetry."""
    base = validate_zero_head_training_artifacts(
        run_directory, expected_kwargs=expected_kwargs
    )
    hparams = _load_object(run_directory / "hparams.json")
    expected = {
        "actor_recovery_teacher_enabled": True,
        "actor_recovery_teacher_dataset_path": expected_kwargs[
            "actor_recovery_teacher_dataset_path"
        ],
        "actor_recovery_teacher_dataset_sha256": RECOVERY_TEACHER_SHA256,
        "actor_recovery_teacher_gradient_ratio": (
            RECOVERY_TEACHER_GRADIENT_RATIO
        ),
        "allow_resume_actor_recovery_teacher_change": True,
    }
    if any(hparams.get(key) != value for key, value in expected.items()):
        raise ValueError("recovery teacher hparams drifted")
    telemetry = validate_recovery_teacher_telemetry(
        _load_rows(run_directory / "checkpoint_phase_metrics.json")
    )
    return {
        **base,
        **telemetry,
        "protocol": "g1-conflict-projected-recovery-teacher-training-v1",
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
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-sha256", required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--seed", type=_zero_seed, default=0)
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
        expert_checkpoint=args.expert_checkpoint.resolve(),
        expert_sha256=args.expert_sha256,
        teacher_dataset=args.teacher_dataset.resolve(),
        teacher_sha256=args.teacher_sha256,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_conflict_projected_recovery_teacher_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        args.carried_reset_bank.resolve(),
        expert_path=args.expert_checkpoint.resolve(),
        expert_sha256=args.expert_sha256,
        teacher_path=args.teacher_dataset.resolve(),
        teacher_sha256=args.teacher_sha256,
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
