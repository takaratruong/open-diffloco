"""Train one E023 recovery adapter from E038 hidden features and a zero head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_lafan_anchored_carried_recovery import (
    build_lafan_recovery_kwargs,
    validate_preflight as validate_lafan_preflight,
    validate_training_artifacts as validate_lafan_training_artifacts,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


EXPERT_SHA256 = "373fd6528d135dac65b38c35728800da693780558a03bb0cca6a412e314f7bd2"
E027_BANK_SHA256 = "d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5"
E027_BANK_SUMMARY_SHA256 = (
    "c7b9f2e3d35a1d01d9154913d6b20eb6d7a413341226e2806f7c2904b23b9feb"
)


def _zero_seed(value: str) -> int:
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("E041 requires seed exactly zero")
    return seed


def validate_registered_hash_arguments(
    *,
    bank_sha256: str,
    bank_summary_sha256: str,
    expert_sha256: str,
) -> None:
    """Reject caller substitutions for E041's three immutable artifacts."""
    if (
        bank_sha256 != E027_BANK_SHA256
        or bank_summary_sha256 != E027_BANK_SUMMARY_SHA256
        or expert_sha256 != EXPERT_SHA256
    ):
        raise ValueError("registered E027/E038 artifact hashes do not match")


def build_zero_head_feature_transfer_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
    *,
    expert_path: str | Path,
    expert_sha256: str,
) -> dict[str, Any]:
    """Apply only the registered E038-hidden/zero-head initialization delta."""
    kwargs = build_lafan_recovery_kwargs(
        profile_name, reference_path, seed, resume_from, carried_bank
    )
    kwargs.update(
        actor_residual_preview_initial_adapter_path=str(
            Path(expert_path).resolve()
        ),
        actor_residual_preview_initial_adapter_sha256=expert_sha256,
    )
    return kwargs


def validate_feature_transfer_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Require exact E038 hidden transfer with a zero-effect output head."""
    expected = {
        "protocol": "g1-zero-head-recovery-feature-transfer-v1",
        "source_sha256": EXPERT_SHA256,
        "input_dim": 328,
        "hidden_dim": 256,
        "action_dim": 29,
        "hidden_kernel_exact": True,
        "hidden_bias_exact": True,
        "output_head_zero": True,
        "parent_parameters_exact": True,
        "initial_action_exact": True,
        "adapter_optimizer_moments_zero": True,
        "valid": True,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("zero-head feature transfer report is invalid")
    return {
        "valid": True,
        "protocol": "g1-zero-head-feature-transfer-validation-v1",
    }


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
    code_commit: str,
) -> dict[str, object]:
    """Bind E027's exact inputs plus the immutable E038 expert source."""
    validate_registered_hash_arguments(
        bank_sha256=carried_bank_sha256,
        bank_summary_sha256=carried_bank_summary_sha256,
        expert_sha256=expert_sha256,
    )
    base = validate_lafan_preflight(
        repository=repository,
        reference_path=reference_path,
        resume_from=resume_from,
        carried_bank=carried_bank,
        carried_bank_summary=carried_bank_summary,
        carried_bank_sha256=carried_bank_sha256,
        carried_bank_summary_sha256=carried_bank_summary_sha256,
        code_commit=code_commit,
    )
    expert = expert_checkpoint.resolve()
    if (
        expert_sha256 != EXPERT_SHA256
        or not expert.is_file()
        or sha256_file(expert) != EXPERT_SHA256
    ):
        raise ValueError("E038 recovery expert SHA-256 does not match")
    return {
        **base,
        "protocol": "g1-zero-head-feature-transfer-preflight-v1",
        "expert_checkpoint": str(expert),
        "expert_sha256": EXPERT_SHA256,
        "scientific_delta": [
            "actor_residual_preview_initial_adapter_path",
            "actor_residual_preview_initial_adapter_sha256",
        ],
    }


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: dict[str, Any],
) -> dict[str, object]:
    """Require E027 training validity plus the exact zero-head migration."""
    base = validate_lafan_training_artifacts(
        run_directory, expected_kwargs=expected_kwargs
    )
    report = _load_json(run_directory / "zero_head_feature_transfer.json")
    hparams = _load_json(run_directory / "hparams.json")
    path = expected_kwargs["actor_residual_preview_initial_adapter_path"]
    if (
        hparams.get("actor_residual_preview_initial_adapter_path") != path
        or hparams.get("actor_residual_preview_initial_adapter_sha256")
        != EXPERT_SHA256
        or report.get("source_path") != path
    ):
        raise ValueError("zero-head feature transfer hparams drifted")
    feature = validate_feature_transfer_report(report)
    return {
        **base,
        **feature,
        "protocol": "g1-zero-head-feature-transfer-training-v1",
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
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_zero_head_feature_transfer_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        args.carried_reset_bank.resolve(),
        expert_path=args.expert_checkpoint.resolve(),
        expert_sha256=args.expert_sha256,
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
