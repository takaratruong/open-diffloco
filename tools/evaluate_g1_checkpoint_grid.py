"""Evaluate four exact continuation checkpoints concurrently from phase zero."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from tools.evaluate_g1_phase_grid import (
    build_evaluator_command as build_phase_command,
    make_contact_sheet,
    validate_phase_summary,
)


CHECKPOINT_STEPS = (442368, 491520, 540672, 589824)
REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs=4, type=Path, required=True)
    parser.add_argument("--checkpoint-steps", nargs=4, type=int, required=True)
    parser.add_argument("--checkpoint-sha256", nargs=4, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", default=REFERENCE_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs=4, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name("evaluate_g1_tracking.py"),
    )
    return parser


def validate_grid(
    *,
    checkpoints: tuple[Path, ...],
    checkpoint_steps: tuple[int, ...],
    checkpoint_sha256: tuple[str, ...],
    gpu_ids: tuple[str, ...],
) -> None:
    """Require the fixed continuation grid and one unique GPU per actor."""
    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise ValueError("exactly four checkpoints are required")
    if checkpoint_steps != CHECKPOINT_STEPS:
        raise ValueError(f"checkpoint steps must be exactly {CHECKPOINT_STEPS}")
    if len(checkpoint_sha256) != len(CHECKPOINT_STEPS):
        raise ValueError("exactly four checkpoint SHA-256 values are required")
    if len(gpu_ids) != len(CHECKPOINT_STEPS):
        raise ValueError("exactly four GPU IDs are required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint paths must be unique")


def classify_checkpoint_grid(
    survival: dict[int, int], completed: dict[int, bool]
) -> str:
    """Classify the preregistered 50-percent survival-gain boundary."""
    if set(survival) != set(CHECKPOINT_STEPS) or set(completed) != set(
        CHECKPOINT_STEPS
    ):
        raise ValueError("results must contain the fixed checkpoint grid")
    if any(completed.values()):
        return "complete-long-reference-tracking"
    if max(survival.values()) >= 120:
        return "material-continuation-gain"
    return "no-material-continuation-gain"


def select_checkpoint(summaries: dict[int, dict[str, Any]]) -> int:
    """Select by survival, then reward, then earliest global step."""
    if set(summaries) != set(CHECKPOINT_STEPS):
        raise ValueError("summaries must contain the fixed checkpoint grid")
    return max(
        CHECKPOINT_STEPS,
        key=lambda step: (
            int(summaries[step]["steps"]),
            float(summaries[step]["mean_reward"]),
            -step,
        ),
    )


def build_evaluator_command(
    *,
    python: Path,
    evaluator: Path,
    checkpoint: Path,
    reference: Path,
    output_dir: Path,
) -> list[str]:
    """Build the existing strict evaluator command at exact phase zero."""
    return build_phase_command(
        python=python,
        evaluator=evaluator,
        checkpoint=checkpoint,
        reference=reference,
        output_dir=output_dir,
        phase=0,
    )


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = tuple(args.checkpoints)
    checkpoint_steps = tuple(args.checkpoint_steps)
    checkpoint_sha256 = tuple(args.checkpoint_sha256)
    gpu_ids = tuple(args.gpu_ids)
    validate_grid(
        checkpoints=checkpoints,
        checkpoint_steps=checkpoint_steps,
        checkpoint_sha256=checkpoint_sha256,
        gpu_ids=gpu_ids,
    )
    for name, path in (
        ("reference", args.reference_path),
        ("evaluator", args.evaluator),
        ("python", args.python),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for checkpoint, expected_sha256 in zip(
        checkpoints, checkpoint_sha256
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        with checkpoint.open("rb") as stream:
            actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint SHA-256 does not match: {checkpoint}")
    with args.reference_path.open("rb") as stream:
        reference_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if reference_sha256 != args.reference_sha256:
        raise ValueError("reference SHA-256 does not match")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    children = []
    for step, checkpoint, gpu_id in zip(
        CHECKPOINT_STEPS, checkpoints, gpu_ids
    ):
        output = args.output_dir / f"checkpoint_{step:06d}_phase0"
        command = build_evaluator_command(
            python=args.python,
            evaluator=args.evaluator,
            checkpoint=checkpoint,
            reference=args.reference_path,
            output_dir=output,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "JAX_ENABLE_X64": "true",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
                "MUJOCO_GL": "egl",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        stdout = (args.output_dir / f"checkpoint_{step:06d}.stdout.log").open(
            "wb"
        )
        stderr = (args.output_dir / f"checkpoint_{step:06d}.stderr.log").open(
            "wb"
        )
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        children.append((step, process, stdout, stderr))

    failures = []
    for step, process, stdout, stderr in children:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        if return_code != 0:
            failures.append((step, return_code))
    if failures:
        raise RuntimeError(f"checkpoint evaluators failed: {failures}")

    summaries = {}
    for step in CHECKPOINT_STEPS:
        output = args.output_dir / f"checkpoint_{step:06d}_phase0"
        summary = json.loads((output / "summary.json").read_text())
        validate_phase_summary(
            summary, phase=0, reference_sha256=reference_sha256
        )
        summaries[step] = summary
        make_contact_sheet(
            imageio.mimread(output / "evaluation.mp4"),
            output / "contact_sheet.png",
        )

    survival = {step: int(summaries[step]["steps"]) for step in CHECKPOINT_STEPS}
    completed = {
        step: bool(summaries[step]["completed_reference_suffix"])
        for step in CHECKPOINT_STEPS
    }
    selected_step = select_checkpoint(summaries)
    payload = {
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_sha256": {
            str(step): digest
            for step, digest in zip(CHECKPOINT_STEPS, checkpoint_sha256)
        },
        "steps": {str(step): survival[step] for step in CHECKPOINT_STEPS},
        "completed_suffix": {
            str(step): completed[step] for step in CHECKPOINT_STEPS
        },
        "decision": classify_checkpoint_grid(survival, completed),
        "selected_checkpoint_step": selected_step,
        "reference_sha256": reference_sha256,
        "summaries": {
            str(step): summaries[step] for step in CHECKPOINT_STEPS
        },
    }
    output_json = args.output_dir / "checkpoint_grid_summary.json"
    temporary_json = output_json.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary_json.replace(output_json)

    values = [survival[step] for step in CHECKPOINT_STEPS]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot(CHECKPOINT_STEPS, values, marker="o", color="#7c3aed")
    axis.axhline(80, color="#dc2626", linestyle="--", label="E000 final")
    axis.axhline(120, color="#16a34a", linestyle="--", label="material gate")
    for step, value in zip(CHECKPOINT_STEPS, values):
        axis.text(step, value + 2, str(value), ha="center")
    axis.set(
        xlabel="Global training transitions",
        ylabel="Strict phase-zero transitions survived",
        title="G1 LAFAN exact-continuation checkpoint grid",
    )
    axis.legend(frameon=False)
    figure.savefig(args.output_dir / "checkpoint_grid_survival.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
