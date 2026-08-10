"""Evaluate one trained G1 actor on a fixed exact-reference phase grid."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
)


PHASES = (0, 100, 200, 300, 400)
REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
CHECKPOINT_SHA256 = (
    "0c42d6c8e7c135d7adb836506bc5eedc5ea32f0a646245fed17226ec4c2c6407"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-grid evaluation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--phase-zero-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs=4, required=True)
    parser.add_argument("--phases", nargs=5, type=int, default=PHASES)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--reference-sha256", default=REFERENCE_SHA256)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default="g1-4x5",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name("evaluate_g1_tracking.py"),
    )
    return parser


def validate_grid(
    phases: tuple[int, ...], gpu_ids: tuple[str, ...]
) -> None:
    """Require the preregistered phase grid and one unique GPU per new phase."""
    if phases != PHASES:
        raise ValueError(f"phases must be exactly {PHASES}")
    if len(gpu_ids) != len(PHASES) - 1:
        raise ValueError("exactly four GPU IDs are required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique")


def classify_phase_grid(
    survival: dict[int, int], completed: dict[int, bool]
) -> str:
    """Classify exact-state suffix behavior by the registered decision gate."""
    if set(survival) != set(PHASES) or set(completed) != set(PHASES):
        raise ValueError(f"results must contain exactly phases {PHASES}")
    nonzero = PHASES[1:]
    carried = [survival[phase] for phase in nonzero]
    if any(
        survival[phase] <= 12 and not completed[phase]
        for phase in nonzero
    ) or median(carried) < 40:
        return "phase-local-difficulty"
    if all(
        survival[phase] >= 40 or completed[phase]
        for phase in nonzero
    ):
        return "broad-phase-local-competence"
    return "mixed-evidence"


def build_evaluator_command(
    *,
    python: Path,
    evaluator: Path,
    checkpoint: Path,
    reference: Path,
    output_dir: Path,
    phase: int,
    solver_profile: str = "g1-4x5",
) -> list[str]:
    """Build one exact existing-evaluator command without changing behavior."""
    profile = get_solver_profile(solver_profile)
    return [
        str(python),
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seed",
        "0",
        "--phase",
        str(phase),
        "--render-every",
        "2",
        "--env-variant",
        "g1_tracking_rmr_50hz_source_step",
        "--solver-iterations",
        str(profile.iterations),
        "--solver-ls-iterations",
        str(profile.ls_iterations),
        "--actor-history-len",
        "10",
        "--reference-residual-control",
        "--reference-residual-scale",
        "0.5",
        "--reference-path",
        str(reference),
        "--reference-stride",
        "1",
    ]


def validate_phase_summary(
    summary: dict[str, Any], *, phase: int, reference_sha256: str
) -> None:
    """Validate the scientific identity and bounds of one evaluator summary."""
    if summary.get("evaluation_start_phase") != phase:
        raise ValueError(f"summary phase does not match {phase}")
    if summary.get("reference_sha256") != reference_sha256:
        raise ValueError("summary reference SHA-256 does not match")
    if summary.get("reference_transitions") != 499:
        raise ValueError("summary reference must contain 499 transitions")
    remaining = 499 - phase
    if summary.get("remaining_reference_transitions") != remaining:
        raise ValueError("summary remaining transition count does not match")
    steps = summary.get("steps")
    if not isinstance(steps, int) or not 1 <= steps <= remaining:
        raise ValueError("summary steps are outside the exact suffix")


def build_phase_grid_payload(
    summaries: dict[int, dict[str, Any]],
    *,
    checkpoint_sha256: str,
    reference_sha256: str,
    solver_profile: str = "g1-4x5",
) -> dict[str, Any]:
    """Build the canonical finite phase-grid comparison payload."""
    if set(summaries) != set(PHASES):
        raise ValueError(f"summaries must contain exactly phases {PHASES}")
    survival = {phase: int(summaries[phase]["steps"]) for phase in PHASES}
    completed = {
        phase: bool(summaries[phase]["completed_reference_suffix"])
        for phase in PHASES
    }
    terminal = {
        phase: bool(summaries[phase]["terminal"]) for phase in PHASES
    }
    return {
        "phases": list(PHASES),
        "steps": {str(phase): survival[phase] for phase in PHASES},
        "completed_suffix": {
            str(phase): completed[phase] for phase in PHASES
        },
        "terminal": {str(phase): terminal[phase] for phase in PHASES},
        "decision": classify_phase_grid(survival, completed),
        "checkpoint_sha256": checkpoint_sha256,
        "reference_sha256": reference_sha256,
        "solver_profile": solver_profile,
    }


def enrich_phase_summary(
    summary: dict[str, Any],
    *,
    solver_profile: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Attach the nominal replay-free protocol beside every phase video."""
    return {
        **summary,
        "solver_profile": solver_profile,
        "checkpoint_sha256": checkpoint_sha256,
        "randomization": "disabled-nominal",
        "actor_observation_noise": False,
        "reset_mode": "exact-reference-phase",
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_contact_sheet(frames: list[np.ndarray], output: Path) -> None:
    """Write twelve evenly spaced paired-video frames as a labeled PNG."""
    if not frames:
        raise ValueError("contact sheet requires at least one frame")
    if any(
        frame.ndim != 3
        or frame.shape[-1] != 3
        or not np.issubdtype(frame.dtype, np.integer)
        for frame in frames
    ):
        raise ValueError("contact sheet frames must be integer RGB arrays")
    indices = np.linspace(
        0, len(frames) - 1, min(12, len(frames)), dtype=int
    )
    cell_width, cell_height = 640, 240
    columns = 3
    rows = int(np.ceil(len(indices) / columns))
    canvas = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), "black"
    )
    font = ImageFont.load_default()
    for slot, index in enumerate(indices):
        image = Image.fromarray(np.asarray(frames[index], dtype=np.uint8))
        image = image.resize(
            (cell_width, cell_height), Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 110, 22), fill=(0, 0, 0))
        draw.text((6, 5), f"frame={index}", fill="white", font=font)
        canvas.paste(
            image,
            ((slot % columns) * cell_width, (slot // columns) * cell_height),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    args = build_parser().parse_args()
    phases = tuple(args.phases)
    gpu_ids = tuple(args.gpu_ids)
    validate_grid(phases, gpu_ids)
    for name, path in (
        ("checkpoint", args.checkpoint),
        ("reference", args.reference_path),
        ("phase-zero directory", args.phase_zero_dir),
        ("evaluator", args.evaluator),
        ("python", args.python),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    with args.checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    with args.reference_path.open("rb") as handle:
        reference_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match")
    if reference_sha256 != args.reference_sha256:
        raise ValueError("reference SHA-256 does not match")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    phase_zero_output = args.output_dir / "phase_000"
    shutil.copytree(args.phase_zero_dir, phase_zero_output)
    summaries: dict[int, dict[str, Any]] = {}
    summaries[0] = json.loads(
        (phase_zero_output / "summary.json").read_text(encoding="utf-8")
    )
    validate_phase_summary(
        summaries[0], phase=0, reference_sha256=reference_sha256
    )
    summaries[0] = enrich_phase_summary(
        summaries[0],
        solver_profile=args.solver_profile,
        checkpoint_sha256=checkpoint_sha256,
    )
    _write_summary(phase_zero_output / "summary.json", summaries[0])

    children = []
    for phase, gpu_id in zip(PHASES[1:], gpu_ids):
        phase_output = args.output_dir / f"phase_{phase:03d}"
        command = build_evaluator_command(
            python=args.python,
            evaluator=args.evaluator,
            checkpoint=args.checkpoint,
            reference=args.reference_path,
            output_dir=phase_output,
            phase=phase,
            solver_profile=args.solver_profile,
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
        stdout = (args.output_dir / f"phase_{phase:03d}.stdout.log").open(
            "wb"
        )
        stderr = (args.output_dir / f"phase_{phase:03d}.stderr.log").open(
            "wb"
        )
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        children.append((phase, process, stdout, stderr))

    failures = []
    for phase, process, stdout, stderr in children:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        if return_code != 0:
            failures.append((phase, return_code))
    if failures:
        raise RuntimeError(f"phase evaluators failed: {failures}")

    for phase in PHASES[1:]:
        phase_output = args.output_dir / f"phase_{phase:03d}"
        summary = json.loads(
            (phase_output / "summary.json").read_text(encoding="utf-8")
        )
        validate_phase_summary(
            summary, phase=phase, reference_sha256=reference_sha256
        )
        summary = enrich_phase_summary(
            summary,
            solver_profile=args.solver_profile,
            checkpoint_sha256=checkpoint_sha256,
        )
        _write_summary(phase_output / "summary.json", summary)
        summaries[phase] = summary
        frames = imageio.mimread(phase_output / "evaluation.mp4")
        make_contact_sheet(frames, phase_output / "contact_sheet.png")

    payload = build_phase_grid_payload(
        summaries,
        checkpoint_sha256=checkpoint_sha256,
        reference_sha256=reference_sha256,
        solver_profile=args.solver_profile,
    )
    payload["summaries"] = {
        str(phase): summaries[phase] for phase in PHASES
    }
    output_json = args.output_dir / "phase_grid_summary.json"
    temporary_json = output_json.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(output_json)

    survival = [payload["steps"][str(phase)] for phase in PHASES]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.bar([str(phase) for phase in PHASES], survival, color="#7c3aed")
    axis.axhline(
        12, color="#dc2626", linestyle="--", label="training horizon"
    )
    axis.axhline(
        40,
        color="#16a34a",
        linestyle="--",
        label="local competence gate",
    )
    for slot, value in enumerate(survival):
        axis.text(slot, value + 1, str(value), ha="center")
    axis.set(
        xlabel="Exact reference start phase",
        ylabel="Replay-free transitions survived",
        title="G1 LAFAN final actor — exact-state phase grid",
    )
    axis.legend(frameon=False)
    figure.savefig(args.output_dir / "phase_grid_survival.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
