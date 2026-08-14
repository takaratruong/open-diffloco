"""Continue selected clipped dance policy with a 24-step actor horizon."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_clipped_dance_continuation import (
    _git_output,
    _write_json_atomically,
    build_clipped_dance_continuation_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


RESUME_STEP = 1_769_472
CONTINUATION_END_STEP = 1_966_080
CHECKPOINT_INTERVAL = 98_304
EXPECTED_RESUME_SHA256 = (
    "fabf35d1a330f8768d06bb746d11bf6a74f3f25b9460a62961f30c962d585de5"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "574597a32d73733521b957400bb624f2a9f4c0f5ed9f35c61d661d2aede594a4"
)
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    return (1_867_776, 1_966_080)


def build_clipped_dance_h24_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    kwargs = build_clipped_dance_continuation_kwargs(
        profile_name, reference_path, seed, resume_from
    )
    kwargs.update(
        unroll_length=24,
        total_steps=CONTINUATION_END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("g1_clipped_dance_h24_runs")
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def validate_preflight(
    *, repository: Path, resume_from: Path, reference_path: Path, code_commit: str
) -> dict[str, object]:
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    resume_from = resume_from.resolve()
    reference_path = reference_path.resolve()
    if not resume_from.is_file() or sha256_file(resume_from) != EXPECTED_RESUME_SHA256:
        raise ValueError("selected E012 checkpoint SHA-256 does not match")
    hparams_path = resume_from.with_name("hparams.json")
    if (
        not hparams_path.is_file()
        or sha256_file(hparams_path) != EXPECTED_RESUME_HPARAMS_SHA256
    ):
        raise ValueError("selected E012 hparams SHA-256 does not match")
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("reference SHA-256 does not match")
    return {
        "protocol": "g1-clipped-dance-h24-preflight-v1",
        "code_commit": head,
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "solver_profile": "g1-4x5",
        "start_step": RESUME_STEP,
        "end_step": CONTINUATION_END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "unroll_length": 24,
    }


def validate_training_artifacts(run_directory: Path) -> dict[str, object]:
    hparams = json.loads((run_directory / "hparams.json").read_text())
    expected = {
        "unroll_length": 24,
        "total_steps": CONTINUATION_END_STEP,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "actor_per_env_grad_clip": 1.0,
        "num_envs": 256,
        "gradient_accumulation_steps": 2,
    }
    for key, value in expected.items():
        if hparams.get(key) != value:
            raise ValueError(f"training hparams {key} does not match")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text()
    )
    rows_by_step = {row.get("step"): row for row in rows}
    for step in expected_checkpoint_steps():
        if not (run_directory / f"checkpoint_step_{step}.pkl").is_file():
            raise ValueError(f"checkpoint {step} is missing")
        row = rows_by_step.get(step)
        if row is None or row.get("actor_cagrad_valid") is not True:
            raise ValueError(f"checkpoint {step} CAGrad telemetry is invalid")
        counts = row.get("actor_cagrad_bin_counts")
        norms = row.get("actor_cagrad_bin_gradient_norms")
        for values, label in ((counts, "counts"), (norms, "norms")):
            if (
                not isinstance(values, list)
                or len(values) != 5
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError(f"checkpoint {step} has invalid CAGrad {label}")
        for key in (
            "actor_preview_frozen_parameter_drift_max_abs",
            "actor_preview_frozen_moment_drift_max_abs",
            "actor_preview_normalizer_drift_max_abs",
            "torso_wrench_assistance_scale_current",
            "torso_wrench_assistance_active_fraction",
            "torso_wrench_assistance_max_force",
            "torso_wrench_assistance_max_torque",
        ):
            if row.get(key) != 0.0:
                raise ValueError(f"checkpoint {step} violates zero field {key}")
    return {
        "protocol": "g1-clipped-dance-h24-training-v1",
        "valid": True,
        "run_directory": str(run_directory.resolve()),
        "checkpoint_steps": list(expected_checkpoint_steps()),
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
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_clipped_dance_h24_kwargs(
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
    validation = validate_training_artifacts(run_directory)
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
