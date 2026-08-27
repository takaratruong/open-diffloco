"""Run one arm of the matched E002 dual-scale root-position experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_root_velocity_continuation import build_root_velocity_kwargs
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
UPDATES = 16
TRANSITIONS_PER_UPDATE = 512 * 24
CHECKPOINT_EVERY_UPDATES = 4
CHECKPOINT_INTERVAL = CHECKPOINT_EVERY_UPDATES * TRANSITIONS_PER_UPDATE
END_STEP = START_STEP + UPDATES * TRANSITIONS_PER_UPDATE
E002_SURVIVAL = (136, 144, 84, 90, 79)
SOURCE_CHECKPOINT_SHA256 = (
    "52aa142dabf382671a5fe7e6b1f26954b77e4fde492bb413a25b85358a1c4325"
)
SOURCE_HPARAMS_SHA256 = (
    "79927f89ef75cf0a6fbfd5c92746a59db587c00319db780dcad702f0c3bbd5eb"
)
REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_checkpoint_steps() -> tuple[int, ...]:
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def build_arm_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    kernel: str,
) -> dict[str, Any]:
    """Continue exact E002 while changing only the registered reward kernel."""
    if kernel not in {"exponential", "dual_scale"}:
        raise ValueError("arm kernel must be exponential or dual_scale")
    kwargs = build_root_velocity_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        enabled=True,
    )
    kwargs.update(
        tracking_root_velocity_weight=1.0,
        allow_resume_tracking_root_velocity_change=False,
        tracking_anchor_position_kernel=kernel,
        allow_resume_tracking_anchor_position_kernel_change=(
            kernel == "dual_scale"
        ),
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    return kwargs


def _rank(record: dict[str, object]) -> tuple[float, float, float, int]:
    survival = record["survival"]
    assert isinstance(survival, list)
    return (
        float(min(survival)),
        float(statistics.median(survival)),
        float(statistics.fmean(survival)),
        -int(record["checkpoint_step"]),
    )


def _validated_records(
    candidates: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    if set(candidates) != set(expected_checkpoint_steps()):
        raise ValueError("pair selection requires the exact checkpoint grid")
    records = []
    for step in expected_checkpoint_steps():
        candidate = candidates[step]
        checkpoint_sha256 = candidate.get("checkpoint_sha256")
        survival = candidate.get("survival")
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or not isinstance(survival, list)
            or len(survival) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in survival
            )
        ):
            raise ValueError("pair candidate is invalid")
        preserves = all(
            value >= baseline
            for value, baseline in zip(survival, E002_SURVIVAL, strict=True)
        )
        improves = any(
            value > baseline
            for value, baseline in zip(survival, E002_SURVIVAL, strict=True)
        )
        records.append(
            {
                "checkpoint_step": step,
                "checkpoint_sha256": checkpoint_sha256,
                "survival": survival,
                "eligible": preserves and improves,
            }
        )
    return records


def classify_pair(
    *,
    control: dict[int, dict[str, object]],
    treatment: dict[int, dict[str, object]],
    source_survival: list[int],
) -> dict[str, object]:
    """Select a safe treatment only when it also exceeds matched continuation."""
    if tuple(source_survival) != E002_SURVIVAL:
        raise ValueError("source E002 survival does not match the registered baseline")
    control_records = _validated_records(control)
    treatment_records = _validated_records(treatment)
    eligible_control = [record for record in control_records if record["eligible"]]
    eligible_treatment = [
        record for record in treatment_records if record["eligible"]
    ]
    best_control = max(eligible_control, key=_rank) if eligible_control else None
    best_treatment = (
        max(eligible_treatment, key=_rank) if eligible_treatment else None
    )
    advances = bool(
        best_treatment is not None
        and (best_control is None or _rank(best_treatment) > _rank(best_control))
    )
    return {
        "protocol": "g1-dual-scale-root-position-pair-v1",
        "source_survival": source_survival,
        "control": control_records,
        "treatment": treatment_records,
        "outcome": (
            "dual-scale-advances"
            if advances
            else "dual-scale-matched-control"
            if best_treatment is not None
            else "dual-scale-insufficient"
        ),
        "selected_treatment_step": (
            best_treatment["checkpoint_step"] if advances else None
        ),
        "selected_treatment_checkpoint_sha256": (
            best_treatment["checkpoint_sha256"] if advances else None
        ),
        "selected_treatment_survival": (
            best_treatment["survival"] if advances else None
        ),
        "policy_retained": advances,
    }


def validate_preflight(
    *, repository: Path, checkpoint: Path, reference: Path, code_commit: str
) -> dict[str, object]:
    """Bind the exact retained E002 source and clean code revision."""
    if sha256_file(checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("source checkpoint SHA-256 mismatch")
    hparams = checkpoint.with_name("hparams.json")
    if not hparams.is_file() or sha256_file(hparams) != SOURCE_HPARAMS_SHA256:
        raise ValueError("source hparams SHA-256 mismatch")
    if sha256_file(reference) != REFERENCE_SHA256:
        raise ValueError("reference SHA-256 mismatch")
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if actual_commit != code_commit or dirty:
        raise ValueError("experiment requires the exact clean code commit")
    return {
        "protocol": "g1-dual-scale-root-position-preflight-v1",
        "code_commit": actual_commit,
        "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "hparams_sha256": SOURCE_HPARAMS_SHA256,
        "reference_sha256": REFERENCE_SHA256,
    }


def _phase_grid_command(
    *, checkpoint: Path, reference: Path, output: Path, code_commit: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.evaluate_g1_flax_phase_grid",
        "--checkpoint",
        str(checkpoint),
        "--reference-path",
        str(reference),
        "--output",
        str(output),
        "--phases",
        "0",
        "25",
        "50",
        "75",
        "100",
        "--seed",
        "0",
        "--code-commit",
        code_commit,
    ]


def evaluate_arm(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    reference: Path,
    output_root: Path,
    code_commit: str,
    kernel: str,
) -> dict[str, object]:
    """Evaluate source and all arm checkpoints using CPU replay-free rollouts."""
    evaluation_root = output_root / "phase_grid"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    source_output = evaluation_root / "source_e002.json"
    subprocess.run(
        _phase_grid_command(
            checkpoint=source_checkpoint,
            reference=reference,
            output=source_output,
            code_commit=code_commit,
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    source_payload = json.loads(source_output.read_text(encoding="utf-8"))
    source_survival = source_payload.get("summary", {}).get("survival")
    if source_survival != list(E002_SURVIVAL):
        raise ValueError("source E002 CPU corroboration failed")
    candidates: dict[int, dict[str, object]] = {}
    for step in expected_checkpoint_steps():
        checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
        output = evaluation_root / f"checkpoint_step_{step}.json"
        subprocess.run(
            _phase_grid_command(
                checkpoint=checkpoint,
                reference=reference,
                output=output,
                code_commit=code_commit,
            ),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        survival = payload.get("summary", {}).get("survival")
        if (
            payload.get("checkpoint_sha256") != sha256_file(checkpoint)
            or payload.get("reference_sha256") != REFERENCE_SHA256
            or payload.get("tracking_anchor_position_kernel") != kernel
            or payload.get("tracking_root_velocity_weight") != 1.0
            or payload.get("summary", {}).get("phases") != [0, 25, 50, 75, 100]
            or not isinstance(survival, list)
        ):
            raise ValueError("arm phase-grid evidence is invalid")
        candidates[step] = {
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "survival": survival,
        }
    result = {
        "protocol": "g1-dual-scale-root-position-arm-v1",
        "kernel": kernel,
        "source_survival": source_survival,
        "source_phase_grid_sha256": sha256_file(source_output),
        "candidates": {str(step): value for step, value in candidates.items()},
    }
    _write_json_atomically(output_root / "arm_results.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--kernel", choices=("exponential", "dual_scale"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("paired treatment seed must equal zero")
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        kernel=args.kernel,
        start_step=START_STEP,
        end_step=END_STEP,
        updates=UPDATES,
        checkpoint_steps=list(expected_checkpoint_steps()),
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_arm_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        kernel=args.kernel,
    )
    configure_jax()
    previous = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (output_root / relative_save_dir).resolve()
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    if (
        hparams.get("tracking_anchor_position_kernel") != args.kernel
        or hparams.get("tracking_root_velocity_weight") != 1.0
        or hparams.get("total_steps") != END_STEP
        or any(
            not (run_directory / f"checkpoint_step_{step}.pkl").is_file()
            for step in expected_checkpoint_steps()
        )
    ):
        raise ValueError("arm training artifacts are invalid")
    _write_json_atomically(
        output_root / "training_validation.json",
        {
            "protocol": "g1-dual-scale-root-position-training-v1",
            "valid": True,
            "kernel": args.kernel,
            "run_directory": str(run_directory),
            "checkpoint_steps": list(expected_checkpoint_steps()),
        },
    )
    if not args.skip_evaluation:
        evaluate_arm(
            run_directory,
            source_checkpoint=args.resume_from.resolve(),
            reference=args.reference_path.resolve(),
            output_root=output_root,
            code_commit=args.code_commit,
            kernel=args.kernel,
        )
    print(run_directory)


if __name__ == "__main__":
    main()
