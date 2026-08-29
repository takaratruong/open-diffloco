"""Ablate one learned torso wrench over the complete force/torque factorial."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
)
from tools.evaluate_g1_tracking import LEARNED_WRENCH_COMPONENT_MASKS
from tools.prepare_g1_rmr_reference import sha256_file

COMPONENT_CONDITIONS = (
    ("full-control-a", "full"),
    ("full-control-b", "full"),
    ("force-only", "force-only"),
    ("vertical-force-and-torque", "vertical-force-and-torque"),
    ("horizontal-force-and-torque", "horizontal-force-and-torque"),
    ("vertical-force-only", "vertical-force-only"),
    ("horizontal-force-only", "horizontal-force-only"),
    ("torque-only", "torque-only"),
    ("zero", "zero"),
)


def npz_content_sha256(path: Path) -> str:
    """Hash sorted array contents rather than timestamped ZIP metadata."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            values = np.asarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(values.dtype.str.encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def classify_component_ablation(
    completed: dict[str, bool], *, controls_exact: bool
) -> str:
    """Classify phase completion over horizontal, vertical, and torque groups."""
    expected = {name for name, _components in COMPONENT_CONDITIONS}
    if set(completed) != expected:
        raise ValueError("component results do not match the registered factorial")
    if not controls_exact:
        return "invalid-control-repeat"
    if not completed["full-control-a"] or not completed["full-control-b"]:
        return "invalid-full-control"
    vertical_only = completed["vertical-force-only"]
    no_vertical = completed["horizontal-force-and-torque"]
    if vertical_only and not no_vertical:
        return "vertical-force-alone-sufficient-no-vertical-insufficient"
    if no_vertical:
        return "vertical-force-not-required"
    if completed["force-only"] or completed["vertical-force-and-torque"]:
        return "vertical-force-required-in-combination"
    if any(
        completed[name]
        for name in (
            "horizontal-force-only",
            "torque-only",
            "zero",
        )
    ):
        return "mixed-component-dependence"
    return "full-wrench-combination-required"


def build_evaluator_command(
    *,
    python: Path,
    evaluator: Path,
    checkpoint: Path,
    reference: Path,
    output_dir: Path,
    components: str,
    phase: int,
    solver_profile: str,
) -> list[str]:
    """Build the canonical evaluator command with one component-mask delta."""
    profile = get_solver_profile(solver_profile)
    return [
        str(python),
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--reference-path",
        str(reference),
        "--output-dir",
        str(output_dir),
        "--seed",
        "0",
        "--phase",
        str(phase),
        "--render-every",
        "2",
        "--env-variant",
        "g1_tracking_rmr_50hz_action_parity",
        "--reference-stride",
        "1",
        "--actor-history-len",
        "10",
        "--actor-reference-preview-mode",
        "delta",
        "--reference-residual-control",
        "--reference-residual-scale",
        "1.0",
        "--solver-iterations",
        str(profile.iterations),
        "--solver-ls-iterations",
        str(profile.ls_iterations),
        "--solver-profile",
        solver_profile,
        "--learned-wrench-components",
        components,
        "--actor-reference-lookahead-steps",
        "4",
        "8",
        "12",
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_condition(
    *,
    summary: dict[str, Any],
    archive_path: Path,
    components: str,
    checkpoint_sha256: str,
    reference_sha256: str,
    phase: int,
    solver_profile: str,
) -> None:
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("condition checkpoint SHA-256 does not match")
    if summary.get("reference_sha256") != reference_sha256:
        raise ValueError("condition reference SHA-256 does not match")
    if summary.get("evaluation_start_phase") != phase:
        raise ValueError("condition phase does not match")
    if summary.get("solver_profile") != solver_profile:
        raise ValueError("condition solver profile does not match")
    if summary.get("environment_variant") != "g1_tracking_rmr_50hz_action_parity":
        raise ValueError("condition environment variant does not match")
    if summary.get("reference_stride") != 1:
        raise ValueError("condition reference stride does not match")
    if summary.get("actor_history_len") != 10:
        raise ValueError("condition actor history does not match")
    if summary.get("actor_reference_preview_mode") != "delta":
        raise ValueError("condition reference preview mode does not match")
    if summary.get("reference_residual_control") is not True:
        raise ValueError("condition reference residual control does not match")
    if summary.get("reference_residual_scale") != 1.0:
        raise ValueError("condition reference residual scale does not match")
    if summary.get("actor_learned_torso_wrench") is not True:
        raise ValueError("condition did not load a learned-wrench checkpoint")
    if summary.get("learned_torso_wrench_components") != components:
        raise ValueError("condition component label does not match")
    expected_mask = np.asarray(
        LEARNED_WRENCH_COMPONENT_MASKS[components], dtype=np.float64
    )
    if not np.array_equal(
        np.asarray(summary.get("learned_torso_wrench_component_mask")),
        expected_mask,
    ):
        raise ValueError("condition summary component mask does not match")
    steps = summary.get("steps")
    remaining = summary.get("remaining_reference_transitions")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or isinstance(remaining, bool)
        or not isinstance(remaining, int)
        or not 1 <= steps <= remaining
    ):
        raise ValueError("condition survival is outside the reference suffix")
    with np.load(archive_path, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["learned_torso_wrench_component_mask"], expected_mask
        ):
            raise ValueError("condition NPZ component mask does not match")
        if str(archive["learned_torso_wrench_components"]) != components:
            raise ValueError("condition NPZ component label does not match")
        applied = np.asarray(archive["learned_torso_wrench"])
        if applied.ndim != 2 or applied.shape[1] != 6:
            raise ValueError("condition applied wrench shape is invalid")
        if not np.isfinite(applied).all():
            raise ValueError("condition applied wrench is non-finite")
        if np.any(applied[:, expected_mask == 0.0] != 0.0):
            raise ValueError("condition disabled wrench components are nonzero")


def _write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["condition"] for row in rows]
    steps = [row["steps"] for row in rows]
    colors = ["#2f855a" if row["completed"] else "#c53030" for row in rows]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(rows))
    axis.bar(positions, steps, color=colors)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("survived reference transitions")
    axis.set_title("Learned torso-wrench component ablation")
    axis.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, steps, strict=True):
        axis.text(position, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", type=int, default=0)
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


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    output_root = args.output_root.resolve()
    evaluator = args.evaluator.resolve()
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match")
    if sha256_file(reference) != args.reference_sha256:
        raise ValueError("reference SHA-256 does not match")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)

    repository = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    for condition, components in COMPONENT_CONDITIONS:
        condition_dir = output_root / condition
        command = build_evaluator_command(
            python=args.python.resolve(),
            evaluator=evaluator,
            checkpoint=checkpoint,
            reference=reference,
            output_dir=condition_dir,
            components=components,
            phase=args.phase,
            solver_profile=args.solver_profile,
        )
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
        rows.append(
            {
                "condition": condition,
                "components": components,
                "mask": list(LEARNED_WRENCH_COMPONENT_MASKS[components]),
                "steps": int(summary["steps"]),
                "completed": bool(summary["completed_reference_suffix"]),
                "rms_force_newtons": summary["learned_torso_wrench_rms_force"],
                "rms_torque_newton_metres": summary["learned_torso_wrench_rms_torque"],
                "summary": str(summary_path.relative_to(output_root)),
                "summary_sha256": sha256_file(summary_path),
                "evaluation": str(archive_path.relative_to(output_root)),
                "evaluation_sha256": sha256_file(archive_path),
                "evaluation_content_sha256": npz_content_sha256(archive_path),
                "video": str(video_path.relative_to(output_root)),
                "video_sha256": sha256_file(video_path),
                "command": command,
            }
        )

    row_by_name = {row["condition"]: row for row in rows}
    controls_exact = (
        row_by_name["full-control-a"]["evaluation_content_sha256"]
        == row_by_name["full-control-b"]["evaluation_content_sha256"]
    )
    completed = {row["condition"]: row["completed"] for row in rows}
    outcome = classify_component_ablation(completed, controls_exact=controls_exact)
    payload = {
        "protocol": "g1-learned-wrench-world-component-factorial-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "reference": str(reference),
        "reference_sha256": args.reference_sha256,
        "phase": args.phase,
        "solver_profile": args.solver_profile,
        "controls_exact": controls_exact,
        "outcome": outcome,
        "valid": not outcome.startswith("invalid-"),
        "rows": rows,
    }
    _write_json(output_root / "component_ablation.json", payload)
    _write_plot(rows, output_root / "component_ablation.png")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
