"""Continue selected E012 with only pre-CAGrad per-environment clipping."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_frozen_residual_assistance_curriculum import (
    build_frozen_residual_assistance_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax


E012_SELECTED_STEP = 1_671_168
CONTINUATION_END_STEP = 1_769_472
EXPECTED_RESUME_SHA256 = (
    "f375cadc9bf8b5cef26fc7414133071910fed393344c99bbacffea963aa9f4f7"
)
EXPECTED_RESUME_HPARAMS_SHA256 = (
    "76a78a6b1176f4d8cff785a8cbc01c0dd18e08de83ae7da61d3be093768f0d5f"
)
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    return (1_720_320, 1_769_472)


def build_clipped_dance_continuation_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict:
    kwargs = build_frozen_residual_assistance_kwargs(
        profile_name, reference_path, seed, resume_from
    )
    kwargs.update(
        total_steps=CONTINUATION_END_STEP,
        actor_per_env_grad_clip=1.0,
        allow_resume_actor_per_env_grad_clip_change=True,
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
        "--output-root", type=Path, default=Path("g1_clipped_dance_runs")
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_preflight(
    *, repository: Path, resume_from: Path, reference_path: Path, code_commit: str
) -> dict[str, Any]:
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    resume_from = resume_from.resolve()
    reference_path = reference_path.resolve()
    if not resume_from.is_file() or sha256_file(resume_from) != EXPECTED_RESUME_SHA256:
        raise ValueError("E012 selected checkpoint SHA-256 does not match")
    hparams_path = resume_from.with_name("hparams.json")
    if (
        not hparams_path.is_file()
        or sha256_file(hparams_path) != EXPECTED_RESUME_HPARAMS_SHA256
    ):
        raise ValueError("E012 hparams SHA-256 does not match")
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("reference SHA-256 does not match")
    return {
        "protocol": "g1-clipped-dance-continuation-preflight-v1",
        "code_commit": head,
        "checkpoint": str(resume_from),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "solver_profile": "g1-4x5",
        "start_step": E012_SELECTED_STEP,
        "end_step": CONTINUATION_END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "actor_per_env_grad_clip": 1.0,
    }


def validate_training_artifacts(run_directory: Path) -> dict[str, Any]:
    hparams = json.loads((run_directory / "hparams.json").read_text())
    if hparams.get("actor_per_env_grad_clip") != 1.0:
        raise ValueError("actor per-environment clip is not exact")
    if hparams.get("allow_resume_actor_per_env_grad_clip_change") is not True:
        raise ValueError("clip resume authority was not persisted")
    if hparams.get("total_steps") != CONTINUATION_END_STEP:
        raise ValueError("training endpoint is not exact")
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
        if (
            not isinstance(counts, list)
            or len(counts) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in counts
            )
        ):
            raise ValueError(f"checkpoint {step} phase coverage is invalid")
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
        "protocol": "g1-clipped-dance-continuation-training-v1",
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
    kwargs = build_clipped_dance_continuation_kwargs(
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
