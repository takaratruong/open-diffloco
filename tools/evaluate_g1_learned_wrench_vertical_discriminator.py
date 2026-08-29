"""Test vertical learned-wrench necessity and sufficiency with reused controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from src.envs.g1_tracking.solver_profiles import SOLVER_PROFILES
from tools.evaluate_g1_learned_wrench_component_ablation import (
    _validate_condition,
    _write_json,
    build_evaluator_command,
    npz_content_sha256,
)
from tools.evaluate_g1_tracking import LEARNED_WRENCH_COMPONENT_MASKS
from tools.prepare_g1_rmr_reference import sha256_file


TREATMENT_CONDITIONS = (
    ("vertical-force-only-a", "vertical-force-only"),
    ("vertical-force-only-b", "vertical-force-only"),
    (
        "horizontal-force-and-torque-a",
        "horizontal-force-and-torque",
    ),
    (
        "horizontal-force-and-torque-b",
        "horizontal-force-and-torque",
    ),
)

CONTROL_CONDITIONS = ("current-default-a", "current-default-b")


def default_xla_environment(ambient: Mapping[str, str]) -> dict[str, str]:
    """Create the historical/default XLA environment without ambient flags."""
    environment = dict(ambient)
    environment.pop("XLA_FLAGS", None)
    return environment


def classify_vertical_discriminator(
    *, vertical: list[bool], no_vertical: list[bool]
) -> dict[str, Any]:
    """Classify duplicated vertical-only and no-vertical completion."""
    if len(vertical) != 2 or len(no_vertical) != 2:
        raise ValueError("each treatment must have exactly two repetitions")
    if len(set(vertical)) != 1 or len(set(no_vertical)) != 1:
        return {
            "outcome": "replicate-divergence",
            "treatments_unanimous": False,
        }
    vertical_completes = vertical[0]
    no_vertical_completes = no_vertical[0]
    if vertical_completes and not no_vertical_completes:
        outcome = "vertical-alone-sufficient-no-vertical-insufficient"
    elif no_vertical_completes:
        outcome = "vertical-not-required"
    else:
        outcome = "vertical-required-but-not-alone-sufficient"
    return {"outcome": outcome, "treatments_unanimous": True}


def validate_control_matrix(
    payload: dict[str, Any],
    *,
    checkpoint_sha256: str,
    reference_sha256: str,
    phase: int,
    solver_profile: str,
    expected_steps: int,
) -> list[dict[str, Any]]:
    """Select two behaviorally successful current/default E012 controls."""
    if payload.get("protocol") != "g1-learned-wrench-replay-matrix-v1":
        raise ValueError("control matrix protocol does not match")
    exact_fields = {
        "checkpoint_sha256": checkpoint_sha256,
        "reference_sha256": reference_sha256,
        "phase": phase,
        "solver_profile": solver_profile,
    }
    for name, expected in exact_fields.items():
        if payload.get(name) != expected:
            raise ValueError(f"control matrix field does not match: {name}")
    selected = [
        row
        for row in payload.get("rows", [])
        if row.get("condition") in CONTROL_CONDITIONS
    ]
    selected.sort(key=lambda row: CONTROL_CONDITIONS.index(row["condition"]))
    if [row.get("condition") for row in selected] != list(CONTROL_CONDITIONS):
        raise ValueError("control matrix lacks the two current/default controls")
    for row in selected:
        if row.get("source") != "current" or row.get("execution") != "default":
            raise ValueError("control row is not current/default XLA")
        if row.get("steps") != expected_steps or row.get("completed") is not True:
            raise ValueError("control row does not complete the reference suffix")
    return selected


def _validate_control_artifacts(
    *,
    rows: list[dict[str, Any]],
    control_root: Path,
    checkpoint_sha256: str,
    reference_sha256: str,
    phase: int,
    solver_profile: str,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for row in rows:
        summary_path = control_root / row["summary"]
        archive_path = control_root / row["evaluation"]
        video_path = control_root / row["video"]
        expected_files = (
            (summary_path, row["summary_sha256"], "summary"),
            (archive_path, row["evaluation_sha256"], "evaluation"),
            (video_path, row["video_sha256"], "video"),
        )
        for path, expected_sha256, label in expected_files:
            if sha256_file(path) != expected_sha256:
                raise ValueError(
                    f'control {row["condition"]} {label} SHA-256 does not match'
                )
        if npz_content_sha256(archive_path) != row["evaluation_content_sha256"]:
            raise ValueError(
                f'control {row["condition"]} NPZ content SHA-256 does not match'
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_condition(
            summary=summary,
            archive_path=archive_path,
            components="full",
            checkpoint_sha256=checkpoint_sha256,
            reference_sha256=reference_sha256,
            phase=phase,
            solver_profile=solver_profile,
        )
        controls.append(
            {
                "condition": row["condition"],
                "components": "full",
                "mask": list(LEARNED_WRENCH_COMPONENT_MASKS["full"]),
                "steps": int(row["steps"]),
                "completed": True,
                "summary": str(summary_path),
                "summary_sha256": row["summary_sha256"],
                "evaluation": str(archive_path),
                "evaluation_sha256": row["evaluation_sha256"],
                "evaluation_content_sha256": row[
                    "evaluation_content_sha256"
                ],
                "video": str(video_path),
                "video_sha256": row["video_sha256"],
                "reused": True,
            }
        )
    return controls


def _write_plot(
    controls: list[dict[str, Any]],
    treatments: list[dict[str, Any]],
    path: Path,
    *,
    expected_steps: int,
) -> None:
    rows = controls + treatments
    labels = [row["condition"] for row in rows]
    steps = [row["steps"] for row in rows]
    colors = []
    for row in rows:
        if row["components"] == "full":
            colors.append("#2b6cb0")
        elif row["components"] == "vertical-force-only":
            colors.append("#38a169")
        else:
            colors.append("#dd6b20")
    figure, axis = plt.subplots(figsize=(11.5, 5.7))
    positions = np.arange(len(rows))
    axis.bar(positions, steps, color=colors)
    axis.axhline(expected_steps, color="#2d3748", linestyle="--", linewidth=1)
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("survived reference transitions")
    axis.set_title("Learned-wrench vertical-force discriminator")
    axis.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, steps, strict=True):
        axis.text(position, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-matrix", type=Path, required=True)
    parser.add_argument("--control-matrix-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--expected-completion-steps", type=int, default=271)
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


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    control_matrix = args.control_matrix.resolve()
    output_root = args.output_root.resolve()
    evaluator = args.evaluator.resolve()
    python = args.python.resolve()

    if args.source_commit is not None:
        actual_commit = _git_output(repository, "rev-parse", "HEAD")
        if actual_commit != args.source_commit:
            raise ValueError("source commit does not match")
        dirty = _git_output(
            repository, "status", "--porcelain", "--untracked-files=all"
        )
        if dirty:
            raise ValueError(f"source repository is dirty:\n{dirty}")
    exact_files = (
        (checkpoint, args.checkpoint_sha256, "checkpoint"),
        (reference, args.reference_sha256, "reference"),
        (control_matrix, args.control_matrix_sha256, "control matrix"),
        (evaluator, args.evaluator_sha256, "evaluator"),
    )
    for path, expected_sha256, label in exact_files:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} SHA-256 does not match")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)

    control_payload = json.loads(control_matrix.read_text(encoding="utf-8"))
    selected_control_rows = validate_control_matrix(
        control_payload,
        checkpoint_sha256=args.checkpoint_sha256,
        reference_sha256=args.reference_sha256,
        phase=args.phase,
        solver_profile=args.solver_profile,
        expected_steps=args.expected_completion_steps,
    )
    controls = _validate_control_artifacts(
        rows=selected_control_rows,
        control_root=control_matrix.parent,
        checkpoint_sha256=args.checkpoint_sha256,
        reference_sha256=args.reference_sha256,
        phase=args.phase,
        solver_profile=args.solver_profile,
    )

    treatments: list[dict[str, Any]] = []
    for index, (condition, components) in enumerate(
        TREATMENT_CONDITIONS, start=1
    ):
        print(
            f"[{index}/{len(TREATMENT_CONDITIONS)}] starting {condition}",
            flush=True,
        )
        condition_dir = output_root / condition
        command = build_evaluator_command(
            python=python,
            evaluator=evaluator,
            checkpoint=checkpoint,
            reference=reference,
            output_dir=condition_dir,
            components=components,
            phase=args.phase,
            solver_profile=args.solver_profile,
        )
        environment = default_xla_environment(os.environ)
        environment["PYTHONPATH"] = str(repository)
        with (
            (output_root / f"{condition}.stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout,
            (output_root / f"{condition}.stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr,
        ):
            subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        summary_path = condition_dir / "summary.json"
        archive_path = condition_dir / "evaluation.npz"
        video_path = condition_dir / "evaluation.mp4"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_condition(
            summary=summary,
            archive_path=archive_path,
            components=components,
            checkpoint_sha256=args.checkpoint_sha256,
            reference_sha256=args.reference_sha256,
            phase=args.phase,
            solver_profile=args.solver_profile,
        )
        treatment = {
            "condition": condition,
            "components": components,
            "mask": list(LEARNED_WRENCH_COMPONENT_MASKS[components]),
            "steps": int(summary["steps"]),
            "completed": bool(summary["completed_reference_suffix"]),
            "rms_force_newtons": summary["learned_torso_wrench_rms_force"],
            "rms_torque_newton_metres": summary[
                "learned_torso_wrench_rms_torque"
            ],
            "summary": str(summary_path.relative_to(output_root)),
            "summary_sha256": sha256_file(summary_path),
            "evaluation": str(archive_path.relative_to(output_root)),
            "evaluation_sha256": sha256_file(archive_path),
            "evaluation_content_sha256": npz_content_sha256(archive_path),
            "video": str(video_path.relative_to(output_root)),
            "video_sha256": sha256_file(video_path),
            "command": command,
            "reused": False,
        }
        treatments.append(treatment)
        _write_json(
            output_root / "vertical_discriminator.partial.json",
            {
                "protocol": "g1-learned-wrench-vertical-discriminator-v1-partial",
                "controls": controls,
                "treatments": treatments,
            },
        )
        print(
            f"[{index}/{len(TREATMENT_CONDITIONS)}] finished {condition}: "
            f'{treatment["steps"]} steps',
            flush=True,
        )

    vertical = [
        row["completed"]
        for row in treatments
        if row["components"] == "vertical-force-only"
    ]
    no_vertical = [
        row["completed"]
        for row in treatments
        if row["components"] == "horizontal-force-and-torque"
    ]
    classification = classify_vertical_discriminator(
        vertical=vertical, no_vertical=no_vertical
    )
    payload = {
        "protocol": "g1-learned-wrench-vertical-discriminator-v1",
        "source_commit": _git_output(repository, "rev-parse", "HEAD"),
        "evaluator": str(evaluator),
        "evaluator_sha256": args.evaluator_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "reference": str(reference),
        "reference_sha256": args.reference_sha256,
        "control_matrix": str(control_matrix),
        "control_matrix_sha256": args.control_matrix_sha256,
        "phase": args.phase,
        "solver_profile": args.solver_profile,
        "expected_completion_steps": args.expected_completion_steps,
        "xla_flags": None,
        "behavioral_controls_complete": True,
        **classification,
        "valid": bool(classification["treatments_unanimous"]),
        "controls": controls,
        "treatments": treatments,
    }
    _write_json(output_root / "vertical_discriminator.json", payload)
    _write_plot(
        controls,
        treatments,
        output_root / "vertical_discriminator.png",
        expected_steps=args.expected_completion_steps,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
