"""Continue the exact E026 short-walk residual with direct torso tracking."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_anchored_carried_recovery import (
    build_anchored_carried_recovery_kwargs,
)
from tools.run_g1_fresh_ppo_action_contract_walk import (
    validate_training_artifacts as validate_base_training_artifacts,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _write_json_atomically,
)

START_STEP = 1_769_472
TRANSITIONS_PER_UPDATE = 512 * 24
CONTINUATION_UPDATES = 32
CHECKPOINT_UPDATES = 8
CHECKPOINT_INTERVAL = CHECKPOINT_UPDATES * TRANSITIONS_PER_UPDATE
END_STEP = START_STEP + CONTINUATION_UPDATES * TRANSITIONS_PER_UPDATE
TORSO_ORIENTATION_WEIGHT = 1.0
SOURCE_SURVIVAL = (124, 99, 74, 49, 24)
SOURCE_TAIL_MEAN_ABS_PITCH_DEGREES = 12.76
SOURCE_TAIL_MAX_ABS_PITCH_DEGREES = 17.85
EXPECTED_RESUME_SHA256 = (
    "4f9a2b49c7368f5323ab81c4c3de4aae208413987ab4858c44bf76872d0f86dd"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "6b60d0b8ea96fa27d633c6f80f8df82a6c09c848d58b9636fd75759bbda486f7"
)
EXPECTED_CARRIED_BANK_SHA256 = (
    "a303e04c9fdd8c52c7fe1a2091c74d555f9b498bc380137d58f8b63fe98792ea"
)
EXPECTED_CARRIED_SUMMARY_SHA256 = (
    "cf5546c3c166df52a9a6c90e651a51003c66a61419b458fdc523c13801f6ba7e"
)
EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the exact four-checkpoint continuation grid."""
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def build_torso_orientation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
) -> dict[str, Any]:
    """Apply only the direct torso-orientation objective continuation."""
    kwargs = build_anchored_carried_recovery_kwargs(
        profile_name, reference_path, seed, resume_from, carried_bank
    )
    kwargs.update(
        total_steps=END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        tracking_torso_orientation_weight=TORSO_ORIENTATION_WEIGHT,
        allow_resume_tracking_torso_orientation_change=True,
    )
    return kwargs


def select_torso_orientation_checkpoint(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select only survival-preserving checkpoints with a stable torso tail."""
    normalized: list[dict[str, object]] = []
    for record in records:
        update = record.get("update")
        survival = record.get("survival")
        position = record.get("body_position_error_ratio")
        orientation = record.get("body_orientation_error_ratio")
        mean_pitch = record.get("phase0_tail_mean_abs_pitch_degrees")
        max_pitch = record.get("phase0_tail_max_abs_pitch_degrees")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update < 1
            or not isinstance(survival, (list, tuple))
            or len(survival) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in survival
            )
            or not isinstance(position, (list, tuple))
            or len(position) != 5
            or not isinstance(orientation, (list, tuple))
            or len(orientation) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                for value in (*position, *orientation, mean_pitch, max_pitch)
            )
        ):
            raise ValueError("torso checkpoint record is invalid")
        normalized.append(
            {
                "update": update,
                "survival": tuple(survival),
                "position": tuple(float(value) for value in position),
                "orientation": tuple(float(value) for value in orientation),
                "mean_pitch": float(mean_pitch),
                "max_pitch": float(max_pitch),
            }
        )

    pitch_target = 0.75 * SOURCE_TAIL_MEAN_ABS_PITCH_DEGREES
    tail_improvers = [row for row in normalized if row["mean_pitch"] <= pitch_target]
    eligible = [
        row
        for row in tail_improvers
        if all(value >= floor for value, floor in zip(row["survival"], SOURCE_SURVIVAL))
        and all(value <= 1.05 for value in row["position"])
        and all(value <= 1.05 for value in row["orientation"])
        and row["max_pitch"] <= SOURCE_TAIL_MAX_ABS_PITCH_DEGREES
    ]
    if not eligible:
        return {
            "valid": True,
            "outcome": (
                "torso-objective-redistributes"
                if tail_improvers
                else "torso-objective-insufficient"
            ),
            "eligible_updates": [],
            "selected_update": None,
            "selected_survival": None,
        }
    selected = min(
        eligible,
        key=lambda row: (row["mean_pitch"], row["max_pitch"], row["update"]),
    )
    return {
        "valid": True,
        "outcome": "torso-objective-stabilizes-short-walk",
        "eligible_updates": [row["update"] for row in eligible],
        "selected_update": selected["update"],
        "selected_survival": list(selected["survival"]),
        "selected_tail_mean_abs_pitch_degrees": selected["mean_pitch"],
        "selected_tail_max_abs_pitch_degrees": selected["max_pitch"],
    }


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    carried_bank: Path,
    code_commit: str,
) -> dict[str, object]:
    """Bind clean code and every E026 resume asset before GPU work."""
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    checkpoint = resume_from.resolve()
    hparams = checkpoint.with_name("hparams.json")
    bank = carried_bank.resolve()
    bank_summary = bank.with_suffix(".json")
    for path, expected, label in (
        (checkpoint, EXPECTED_RESUME_SHA256, "E026 checkpoint"),
        (hparams, EXPECTED_RESUME_HPARAMS_SHA256, "E026 hparams"),
        (bank, EXPECTED_CARRIED_BANK_SHA256, "carried bank"),
        (bank_summary, EXPECTED_CARRIED_SUMMARY_SHA256, "carried summary"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    resume_hparams = json.loads(hparams.read_text(encoding="utf-8"))
    consumed_reference = Path(str(resume_hparams.get("reference_path", ""))).resolve()
    model = Path(str(resume_hparams.get("xml_path", ""))).resolve()
    controller = Path(DEFAULT_CONTROLLER_PATH).resolve()
    if consumed_reference != reference_path.resolve():
        raise ValueError("resume-consumed reference path does not match registration")
    for path, expected, label in (
        (consumed_reference, EXPECTED_REFERENCE_SHA256, "reference"),
        (model, EXPECTED_MODEL_SHA256, "model"),
        (controller, EXPECTED_CONTROLLER_SHA256, "controller"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"runtime {label} SHA-256 does not match")
    assets = {
        "reference_path": str(consumed_reference),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "model_path": str(model),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "controller_path": str(controller),
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
    }
    return {
        "valid": True,
        "protocol": "g1-e026-torso-orientation-preflight-v1",
        "code_commit": head,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams),
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "carried_bank": str(bank),
        "carried_bank_sha256": EXPECTED_CARRIED_BANK_SHA256,
        "carried_summary": str(bank_summary),
        "carried_summary_sha256": EXPECTED_CARRIED_SUMMARY_SHA256,
        **assets,
        "start_step": START_STEP,
        "end_step": END_STEP,
        "additional_updates": CONTINUATION_UPDATES,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "scientific_delta": [
            "resume_from",
            "total_steps",
            "checkpoint_interval",
            "tracking_torso_orientation_weight",
            "allow_resume_tracking_torso_orientation_change",
        ],
    }


def validate_training_artifacts(
    run_directory: Path, *, expected_kwargs: dict[str, Any]
) -> dict[str, object]:
    """Require exact treatment hparams and immutable-parent finite updates."""
    base = validate_base_training_artifacts(
        run_directory,
        expected_kwargs=expected_kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=END_STEP,
        protocol="g1-e026-torso-orientation-training-v1",
        expected_actor_bootstrap_scale=0.0,
    )
    root = run_directory.resolve()
    hparams = json.loads((root / "hparams.json").read_text(encoding="utf-8"))
    rows = json.loads(
        (root / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    if (
        hparams.get("tracking_torso_orientation_weight") != TORSO_ORIENTATION_WEIGHT
        or hparams.get("allow_resume_tracking_torso_orientation_change") is not True
    ):
        raise ValueError("torso orientation treatment hparams drifted")
    if not isinstance(rows, list) or [row.get("step") for row in rows] != list(
        expected_checkpoint_steps()
    ):
        raise ValueError("torso continuation telemetry cadence is invalid")
    for row in rows:
        for key in (
            "actor_preview_frozen_parameter_drift_max_abs",
            "actor_preview_frozen_moment_drift_max_abs",
            "actor_preview_normalizer_drift_max_abs",
        ):
            if row.get(key) != 0.0:
                raise ValueError("frozen parent drifted")
        for key in ("actor_preview_gradient_norm", "actor_preview_update_norm"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError("residual update telemetry is invalid")
    return {**base, "valid": True, "torso_orientation_weight": 1.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--carried-reset-bank", type=Path, required=True)
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
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_torso_orientation_kwargs(
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
