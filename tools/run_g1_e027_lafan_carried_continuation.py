"""Continue the exact E027 frozen-parent LAFAN adapter to update 128."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.build_g1_e023_carried_reset_bank import validate_code_commit
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_anchored_carried_recovery import CHECKPOINT_INTERVAL
from tools.run_g1_e023_lafan_anchored_carried_recovery import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
    build_lafan_recovery_kwargs,
)
from tools.run_g1_fresh_ppo_action_contract_walk import (
    validate_training_artifacts as validate_base_training_artifacts,
)
from tools.run_g1_root_recovery_continuation import validate_runtime_assets
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 2_359_296
CONTINUATION_UPDATES = 64
TRANSITIONS_PER_UPDATE = 512 * 24
CONTINUATION_END_STEP = START_STEP + CONTINUATION_UPDATES * TRANSITIONS_PER_UPDATE
EXPECTED_RESUME_SHA256 = (
    "e941d1189a82047ca28e98bccc4543c0f1c3de11fa0f2b0c642d65b915736645"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "fd112a300faba9023589ffbee6f09776cf35fe659fe5387b2e4fc6a91431e476"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    return tuple(
        range(
            START_STEP + CHECKPOINT_INTERVAL,
            CONTINUATION_END_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )


def build_lafan_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
) -> dict[str, Any]:
    """Change only the exact resume parent and endpoint from E027."""
    kwargs = build_lafan_recovery_kwargs(
        profile_name, reference_path, seed, resume_from, carried_bank
    )
    kwargs["total_steps"] = CONTINUATION_END_STEP
    return kwargs


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_exact_continuation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require eight finite adapter-only checkpoints with zero frozen drift."""
    if [row.get("step") for row in rows] != list(expected_checkpoint_steps()):
        raise ValueError("continuation telemetry does not match checkpoint grid")
    for row in rows:
        step = row["step"]
        for key in (
            "actor_preview_frozen_parameter_drift_max_abs",
            "actor_preview_frozen_moment_drift_max_abs",
            "actor_preview_normalizer_drift_max_abs",
        ):
            if row.get(key) != 0.0:
                raise ValueError(f"checkpoint {step} has frozen drift")
        for key in ("actor_preview_gradient_norm", "actor_preview_update_norm"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"checkpoint {step} adapter update is invalid")
        if row.get("actor_preview_valid") is not True:
            raise ValueError(f"checkpoint {step} adapter telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-e027-lafan-carried-continuation-telemetry-v1",
        "checkpoint_count": len(rows),
    }


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    carried_bank: Path,
    carried_bank_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    """Bind the exact valid E027 endpoint and unchanged treatment assets."""
    code_commit = validate_code_commit(repository, code_commit)
    checkpoint = resume_from.resolve()
    hparams = checkpoint.with_name("hparams.json")
    reference = reference_path.resolve()
    bank = carried_bank.resolve()
    for path, expected, label in (
        (checkpoint, EXPECTED_RESUME_SHA256, "E027 checkpoint"),
        (hparams, EXPECTED_RESUME_HPARAMS_SHA256, "E027 hparams"),
        (reference, EXPECTED_LAFAN_REFERENCE_SHA256, "LAFAN reference"),
        (bank, carried_bank_sha256, "carried bank"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    payload = _load_json(hparams)
    if not isinstance(payload, dict):
        raise ValueError("E027 hparams must be an object")
    expected_hparams = {
        "total_steps": START_STEP,
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "actor_residual_preview_adapter": True,
        "actor_residual_preview_hidden": 256,
        "actor_residual_preview_optimizer": "adam",
        "actor_normalizer_frozen": True,
        "actor_policy_anchor_weight": 1.0,
        "carried_reset_probability": 0.25,
        "carried_reset_bank_sha256": carried_bank_sha256,
        "actor_bootstrap_scale": 0.0,
        "actor_cagrad": True,
    }
    if any(payload.get(key) != value for key, value in expected_hparams.items()):
        raise ValueError("E027 hparams do not match exact continuation")
    runtime_assets = validate_runtime_assets(
        Path(str(payload["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    return {
        **runtime_assets,
        "valid": True,
        "protocol": "g1-e027-lafan-carried-continuation-preflight-v1",
        "code_commit": code_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams),
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference_path": str(reference),
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "carried_bank": str(bank),
        "carried_bank_sha256": carried_bank_sha256,
        "start_step": START_STEP,
        "end_step": CONTINUATION_END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "scientific_delta": ["resume_from", "total_steps"],
    }


def validate_training_artifacts(
    run_directory: Path, *, expected_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Validate base SHAC evidence and exact composite-state continuation."""
    base = validate_base_training_artifacts(
        run_directory,
        expected_kwargs=expected_kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=CONTINUATION_END_STEP,
        protocol="g1-e027-lafan-carried-continuation-base-v1",
    )
    hparams = _load_json(run_directory / "hparams.json")
    rows = _load_json(run_directory / "checkpoint_phase_metrics.json")
    if not isinstance(hparams, dict) or not isinstance(rows, list):
        raise ValueError("continuation metadata is invalid")
    if (
        hparams.get("resume_residual_adapter_upgrade") is not False
        or hparams.get("reference_path_migration_artifact") is not None
        or (run_directory / "residual_adapter_migration.json").exists()
        or (run_directory / "reference_path_migration.json").exists()
    ):
        raise ValueError("continuation unexpectedly remigrated treatment state")
    exact = validate_exact_continuation_rows(rows)
    return {
        **base,
        **exact,
        "protocol": "g1-e027-lafan-carried-continuation-training-v1",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--carried-reset-bank", type=Path, required=True)
    parser.add_argument("--carried-reset-bank-sha256", required=True)
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
        carried_bank_sha256=args.carried_reset_bank_sha256,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_lafan_continuation_kwargs(
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
