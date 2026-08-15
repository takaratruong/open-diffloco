"""Continue E023 exactly because its final learning curve has not plateaued."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_fresh_ppo_action_contract_walk import validate_training_artifacts
from tools.run_g1_rmr_noise_h24_walk import (
    build_rmr_noise_h24_kwargs,
    validate_preflight as validate_e023_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_572_864
CONTINUATION_UPDATES = 128
TRANSITIONS_PER_UPDATE = 512 * 24
CONTINUATION_END_STEP = START_STEP + CONTINUATION_UPDATES * TRANSITIONS_PER_UPDATE
CHECKPOINT_INTERVAL = 16 * TRANSITIONS_PER_UPDATE
EXPECTED_RESUME_SHA256 = (
    "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the eight immutable continuation checkpoints."""
    return tuple(
        range(
            START_STEP + CHECKPOINT_INTERVAL,
            CONTINUATION_END_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )


def build_rmr_noise_h24_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Resume E023 while changing only its absolute endpoint."""
    kwargs = build_rmr_noise_h24_kwargs(
        profile_name, reference_path, seed
    )
    kwargs.update(
        resume_from=str(Path(resume_from).resolve()),
        total_steps=CONTINUATION_END_STEP,
    )
    return kwargs


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Pin the clean runtime and exact E023 final TrainState."""
    base = validate_e023_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    checkpoint = resume_from.resolve()
    hparams = checkpoint.with_name("hparams.json")
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != EXPECTED_RESUME_SHA256
    ):
        raise ValueError("E023 resume checkpoint SHA-256 does not match")
    if (
        not hparams.is_file()
        or sha256_file(hparams) != EXPECTED_RESUME_HPARAMS_SHA256
    ):
        raise ValueError("E023 resume hparams SHA-256 does not match")
    return {
        **base,
        "protocol": "g1-rmr-noise-h24-continuation-preflight-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams),
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "start_step": START_STEP,
        "end_step": CONTINUATION_END_STEP,
        "additional_updates": CONTINUATION_UPDATES,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "scientific_delta": ["resume_from", "total_steps"],
        "fresh_initialization": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
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
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_rmr_noise_h24_continuation_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
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
        run_directory,
        expected_kwargs=kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=CONTINUATION_END_STEP,
        protocol="g1-rmr-noise-h24-continuation-training-v1",
    )
    _write_json_atomically(
        output_root / "training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
