"""Continue the exact E008 demonstration-replay treatment to update 128."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from tools.prepare_g1_rmr_reference import sha256_file


START_STEP = 393_216
END_STEP = 1_572_864
TRANSITIONS_PER_UPDATE = 512 * 24
CHECKPOINT_INTERVAL = 16 * TRANSITIONS_PER_UPDATE
EXPECTED_RESUME_SHA256 = (
    "2ed61b2f1fd9e8a5858a2a7aae133bb7af725a464fa080e4f0055e75acda4573"
)
EXPECTED_HPARAMS_SHA256 = (
    "271384cf97d90796f45f87b79bf56ecc5a5f04c3877e94735aa2283dfb0714fe"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def build_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    from tools.run_g1_demonstration_replay_h24_walk import (
        build_demonstration_replay_kwargs,
    )

    kwargs = build_demonstration_replay_kwargs(
        profile_name, reference_path, seed
    )
    kwargs.update(
        resume_from=str(Path(resume_from).resolve()),
        total_steps=END_STEP,
    )
    return kwargs


def validate_continuation_replay_telemetry(path: Path) -> dict[int, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = set(expected_checkpoint_steps())
    if not isinstance(rows, list) or {row.get("step") for row in rows} != expected:
        raise ValueError("continuation replay telemetry rows are incomplete")
    fractions = {}
    for row in rows:
        count = row.get("demonstration_replay_count")
        fraction = row.get("demonstration_replay_fraction")
        if (
            row.get("demonstration_replay_valid") is not True
            or row.get("demonstration_replay_threshold") != 0.2
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError("continuation replay telemetry is invalid")
        fractions[int(row["step"])] = float(fraction)
    return fractions


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    resume_from: Path,
    code_commit: str,
) -> dict[str, Any]:
    from tools.run_g1_demonstration_replay_h24_walk import (
        validate_preflight as validate_parent_preflight,
    )

    parent = validate_parent_preflight(
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
        raise ValueError("E008 resume checkpoint SHA-256 does not match")
    if (
        not hparams.is_file()
        or sha256_file(hparams) != EXPECTED_HPARAMS_SHA256
    ):
        raise ValueError("E008 resume hparams SHA-256 does not match")
    return {
        **parent,
        "protocol": "g1-demonstration-replay-h24-continuation-preflight-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams": str(hparams),
        "hparams_sha256": EXPECTED_HPARAMS_SHA256,
        "start_step": START_STEP,
        "end_step": END_STEP,
        "additional_updates": (END_STEP - START_STEP) // TRANSITIONS_PER_UPDATE,
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
    from src.algorithms.shac.algorithm import train
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )
    from tools.run_g1_tracking_shac import configure_jax
    from tools.run_g1_zero_assistance_consolidation import (
        _write_json_atomically,
    )

    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path.resolve(),
        resume_from=args.resume_from.resolve(),
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_continuation_kwargs(
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
        total_steps=END_STEP,
        protocol="g1-demonstration-replay-h24-continuation-training-v1",
    )
    fractions = validate_continuation_replay_telemetry(
        run_directory / "checkpoint_phase_metrics.json"
    )
    _write_json_atomically(
        output_root / "training_validation.json",
        {**validation, "demonstration_replay_fractions_by_step": fractions},
    )
    print(run_directory)


if __name__ == "__main__":
    main()
