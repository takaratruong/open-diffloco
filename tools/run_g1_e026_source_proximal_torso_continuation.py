"""Continue E026 with torso tracking anchored to the exact E026 policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e026_torso_orientation_continuation import (
    EXPECTED_RESUME_SHA256,
    build_torso_orientation_kwargs,
    validate_preflight,
    validate_training_artifacts,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


def build_source_proximal_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    carried_bank: str | Path,
) -> dict[str, Any]:
    """Use the exact E026 source as the weight-one proximal anchor target."""
    kwargs = build_torso_orientation_kwargs(
        profile_name, reference_path, seed, resume_from, carried_bank
    )
    kwargs.update(
        actor_policy_anchor_source_path=str(Path(resume_from).resolve()),
        actor_policy_anchor_source_sha256=EXPECTED_RESUME_SHA256,
        allow_resume_actor_policy_anchor_source_change=True,
    )
    return kwargs


def extend_source_proximal_preflight(
    base: dict[str, object],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, object]:
    """Bind the proximal target and replace the inherited delta declaration."""
    source = source_path.resolve()
    if not source.is_file() or sha256_file(source) != source_sha256:
        raise ValueError("policy anchor source SHA-256 does not match")
    return {
        **base,
        "protocol": "g1-e026-source-proximal-torso-preflight-v1",
        "policy_anchor_source_path": str(source),
        "policy_anchor_source_sha256": source_sha256,
        "scientific_delta": [
            "actor_policy_anchor_source_path",
            "actor_policy_anchor_source_sha256",
            "allow_resume_actor_policy_anchor_source_change",
        ],
    }


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
    inherited = validate_preflight(
        repository=repository,
        reference_path=args.reference_path.resolve(),
        resume_from=args.resume_from.resolve(),
        carried_bank=args.carried_reset_bank.resolve(),
        code_commit=args.code_commit,
    )
    preflight = extend_source_proximal_preflight(
        inherited,
        source_path=args.resume_from,
        source_sha256=EXPECTED_RESUME_SHA256,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_source_proximal_kwargs(
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
    validation.update(
        protocol="g1-e026-source-proximal-torso-training-v1",
        policy_anchor_source_sha256=EXPECTED_RESUME_SHA256,
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
