"""Continue E012 with a bounded root-focused reset mixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import (
    DEFAULT_CONTROLLER_PATH,
    DEFAULT_REFERENCE_PATH,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_frozen_residual_assistance_curriculum import (
    build_frozen_residual_assistance_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_zero_assistance_consolidation import (
    CONSOLIDATION_END_STEP,
    EXPECTED_REFERENCE_SHA256,
    _write_json_atomically,
    expected_checkpoint_steps as _expected_checkpoint_steps,
    validate_preflight,
    validate_training_artifacts as validate_zero_assistance_training_artifacts,
)


ROOT_RECOVERY_MULTIPLIER = 2.0
ROOT_RECOVERY_PROBABILITY = 0.5
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the E017-matched dense checkpoint grid."""
    return _expected_checkpoint_steps()


def validate_runtime_assets(model_path: Path, controller_path: Path) -> dict[str, str]:
    """Fail closed on the runtime robot and controller assets."""
    model_path = model_path.resolve()
    controller_path = controller_path.resolve()
    if not model_path.is_file() or sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("runtime model SHA-256 does not match")
    if (
        not controller_path.is_file()
        or sha256_file(controller_path) != EXPECTED_CONTROLLER_SHA256
    ):
        raise ValueError("runtime controller SHA-256 does not match")
    return {
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "controller_path": str(controller_path),
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
    }


def validate_consumed_resume_assets(
    hparams_path: Path, registered_reference_path: Path
) -> dict[str, str]:
    """Bind preflight to the model/reference paths restored by train()."""
    hparams = json.loads(hparams_path.resolve().read_text(encoding="utf-8"))
    consumed_reference = Path(str(hparams.get("reference_path", ""))).resolve()
    registered_reference = registered_reference_path.resolve()
    if consumed_reference != registered_reference:
        raise ValueError("resume-consumed reference path does not match registration")
    if (
        not consumed_reference.is_file()
        or sha256_file(consumed_reference) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("resume-consumed reference SHA-256 does not match")
    consumed_model = Path(str(hparams.get("xml_path", ""))).resolve()
    runtime_assets = validate_runtime_assets(
        consumed_model, Path(DEFAULT_CONTROLLER_PATH)
    )
    return {
        **runtime_assets,
        "reference_path": str(consumed_reference),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
    }


def build_root_recovery_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    """Change only the endpoint and registered root-reset distribution."""
    kwargs = build_frozen_residual_assistance_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs.update(
        total_steps=CONSOLIDATION_END_STEP,
        reference_root_reset_noise_multiplier=ROOT_RECOVERY_MULTIPLIER,
        reference_root_reset_noise_probability=ROOT_RECOVERY_PROBABILITY,
        allow_resume_reference_root_reset_noise_change=True,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_root_recovery_continuation_runs"),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def _require_equal(document: dict[str, Any], key: str, expected: Any) -> None:
    if document.get(key) != expected:
        raise ValueError(f"training hparams {key} does not match")


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    """Validate E017's boundary plus the exact root-recovery treatment."""
    validation = validate_zero_assistance_training_artifacts(run_directory)
    hparams = json.loads(
        (Path(run_directory).resolve() / "hparams.json").read_text(encoding="utf-8")
    )
    treatment = {
        "reference_root_reset_noise_multiplier": ROOT_RECOVERY_MULTIPLIER,
        "reference_root_reset_noise_probability": ROOT_RECOVERY_PROBABILITY,
        "allow_resume_reference_root_reset_noise_change": True,
    }
    for key, expected in treatment.items():
        _require_equal(hparams, key, expected)
    return {
        **validation,
        "protocol": "g1-root-recovery-continuation-training-v1",
        "root_recovery_treatment": treatment,
    }


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        resume_from=args.resume_from,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )
    preflight.update(
        {
            "protocol": "g1-root-recovery-continuation-preflight-v1",
            "root_recovery_multiplier": ROOT_RECOVERY_MULTIPLIER,
            "root_recovery_probability": ROOT_RECOVERY_PROBABILITY,
            "runtime_assets": validate_consumed_resume_assets(
                Path(str(preflight["hparams"])), args.reference_path
            ),
        }
    )
    _write_json_atomically(output_root / "root_recovery_preflight.json", preflight)

    configure_jax()
    kwargs = build_root_recovery_continuation_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
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
    validation = validate_training_artifacts(run_directory)
    _write_json_atomically(
        output_root / "root_recovery_training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
