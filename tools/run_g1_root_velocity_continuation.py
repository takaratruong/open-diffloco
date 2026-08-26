"""Train one frozen-E026 residual with explicit pelvis-velocity tracking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_capture_point_continuation import (
    build_capture_point_kwargs,
    validate_training_artifacts as validate_parent_training_artifacts,
)
from tools.run_g1_learned_torso_wrench import (
    validate_preflight as validate_e026_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_769_472
UPDATES = 32
TRANSITIONS_PER_UPDATE = 512 * 24
CHECKPOINT_EVERY_UPDATES = 8
CHECKPOINT_INTERVAL = CHECKPOINT_EVERY_UPDATES * TRANSITIONS_PER_UPDATE
END_STEP = START_STEP + UPDATES * TRANSITIONS_PER_UPDATE
ROOT_VELOCITY_WEIGHT = 1.0
E026_SURVIVAL = (131, 114, 74, 71, 74)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the immutable four-checkpoint treatment grid."""
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def build_root_velocity_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Apply only the explicit root-velocity treatment over frozen E026."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")
    kwargs = build_capture_point_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        capture_enabled=False,
    )
    kwargs.update(
        actor_centroidal_propulsion=False,
        actor_capture_point_tracking=False,
        actor_counterfactual_wrench_distillation=False,
        tracking_root_velocity_weight=(ROOT_VELOCITY_WEIGHT if enabled else 0.0),
        allow_resume_tracking_root_velocity_change=enabled,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    return kwargs


def classify_selection(
    candidates: dict[int, dict[str, object]],
    *,
    source_survival: list[int],
) -> dict[str, object]:
    """Select only a componentwise-safe strict improvement over E026."""
    if tuple(source_survival) != E026_SURVIVAL:
        raise ValueError("source E026 survival does not match the registered baseline")
    if set(candidates) != set(expected_checkpoint_steps()):
        raise ValueError("root-velocity selection requires the exact checkpoint grid")
    records: list[dict[str, object]] = []
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
            raise ValueError("root-velocity candidate is invalid")
        preserves = all(
            value >= baseline
            for value, baseline in zip(survival, E026_SURVIVAL, strict=True)
        )
        improves = any(
            value > baseline
            for value, baseline in zip(survival, E026_SURVIVAL, strict=True)
        )
        records.append(
            {
                "checkpoint_step": step,
                "checkpoint_sha256": checkpoint_sha256,
                "survival": survival,
                "minimum": min(survival),
                "median": float(statistics.median(survival)),
                "mean": float(statistics.fmean(survival)),
                "eligible": preserves and improves,
                "improves_any_phase": improves,
            }
        )
    eligible = [record for record in records if record["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda record: (
                record["minimum"],
                record["median"],
                record["mean"],
                -record["checkpoint_step"],
            ),
        )
        if eligible
        else None
    )
    any_gain = any(record["improves_any_phase"] for record in records)
    return {
        "protocol": "g1-root-velocity-selection-v1",
        "phases": [0, 25, 50, 75, 100],
        "source_survival": source_survival,
        "checkpoints": records,
        "outcome": (
            "root-velocity-advances"
            if selected is not None
            else "root-velocity-redistributes"
            if any_gain
            else "root-velocity-insufficient"
        ),
        "selected_step": (
            selected["checkpoint_step"] if selected is not None else None
        ),
        "selected_checkpoint_sha256": (
            selected["checkpoint_sha256"] if selected is not None else None
        ),
        "selected_survival": (
            selected["survival"] if selected is not None else None
        ),
        "policy_retained": selected is not None,
    }


def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: dict[str, Any],
    source_checkpoint: Path,
) -> dict[str, object]:
    """Require the frozen-parent continuation and exact root treatment."""
    parent = validate_parent_training_artifacts(
        run_directory,
        expected_kwargs=expected_kwargs,
        source_checkpoint=source_checkpoint,
    )
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    if (
        hparams.get("tracking_root_velocity_weight") != ROOT_VELOCITY_WEIGHT
        or hparams.get("allow_resume_tracking_root_velocity_change") is not True
    ):
        raise ValueError("root-velocity hparams do not match the treatment")
    return {
        **parent,
        "protocol": "g1-root-velocity-continuation-training-v1",
        "root_velocity_weight": ROOT_VELOCITY_WEIGHT,
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


def _render_command(
    *, checkpoint: Path, reference: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.evaluate_g1_tracking",
        "--checkpoint",
        str(checkpoint),
        "--reference-path",
        str(reference),
        "--output-dir",
        str(output),
        "--phase",
        "0",
        "--seed",
        "0",
        "--env-variant",
        "g1_tracking_rmr_50hz_action_parity",
        "--reference-stride",
        "1",
        "--actor-history-len",
        "10",
        "--actor-reference-lookahead-steps",
        "4",
        "8",
        "12",
        "--actor-reference-preview-mode",
        "delta",
        "--reference-residual-control",
        "--reference-residual-scale",
        "1.0",
        "--tracking-root-velocity-weight",
        "1.0",
        "--solver-profile",
        "g1-4x5",
        "--render-every",
        "2",
    ]


def _plot_survival(selection: dict[str, object], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = selection["checkpoints"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for phase_index, phase in enumerate(selection["phases"]):
        axis.plot(
            [record["checkpoint_step"] for record in records],
            [record["survival"][phase_index] for record in records],
            marker="o",
            label=f"phase {phase}",
        )
        axis.axhline(E026_SURVIVAL[phase_index], linestyle="--", linewidth=0.7)
    axis.set_title("Explicit root-velocity continuation")
    axis.set_xlabel("transition step")
    axis.set_ylabel("steps survived")
    axis.legend(fontsize=8)
    figure.tight_layout()
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.replace(temporary, output)


def evaluate_and_select(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    reference: Path,
    output_root: Path,
    code_commit: str,
) -> dict[str, object]:
    """Evaluate every checkpoint on CPU and render one diagnostic policy."""
    evaluation_root = output_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "MUJOCO_GL": "egl",
    }
    source_phase_grid = evaluation_root / "source_e026.json"
    subprocess.run(
        _phase_grid_command(
            checkpoint=source_checkpoint,
            reference=reference,
            output=source_phase_grid,
            code_commit=code_commit,
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    source_payload = json.loads(source_phase_grid.read_text(encoding="utf-8"))
    source_summary = source_payload.get("summary")
    source_survival = (
        source_summary.get("survival")
        if isinstance(source_summary, dict)
        else None
    )
    if (
        source_payload.get("checkpoint_path")
        != str(source_checkpoint.resolve())
        or source_payload.get("checkpoint_sha256")
        != sha256_file(source_checkpoint)
        or source_payload.get("reference_sha256") != sha256_file(reference)
        or source_payload.get("tracking_root_velocity_weight") != 0.0
        or not isinstance(source_summary, dict)
        or source_summary.get("phases") != [0, 25, 50, 75, 100]
        or not isinstance(source_survival, list)
        or tuple(source_survival) != E026_SURVIVAL
    ):
        raise ValueError("source E026 phase-grid evidence is invalid")
    candidates: dict[int, dict[str, object]] = {}
    for step in expected_checkpoint_steps():
        checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
        phase_grid = evaluation_root / f"checkpoint_step_{step}.json"
        subprocess.run(
            _phase_grid_command(
                checkpoint=checkpoint,
                reference=reference,
                output=phase_grid,
                code_commit=code_commit,
            ),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
        )
        payload = json.loads(phase_grid.read_text(encoding="utf-8"))
        summary = payload.get("summary")
        survival = summary.get("survival") if isinstance(summary, dict) else None
        if (
            payload.get("checkpoint_path") != str(checkpoint.resolve())
            or payload.get("checkpoint_sha256") != sha256_file(checkpoint)
            or payload.get("reference_sha256") != sha256_file(reference)
            or payload.get("tracking_root_velocity_weight") != ROOT_VELOCITY_WEIGHT
            or not isinstance(summary, dict)
            or summary.get("phases") != [0, 25, 50, 75, 100]
            or not isinstance(survival, list)
        ):
            raise ValueError("root-velocity phase-grid evidence is invalid")
        candidates[step] = {
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "survival": survival,
        }
    selection = classify_selection(
        candidates,
        source_survival=source_survival,
    )
    selection["source_checkpoint_sha256"] = source_payload[
        "checkpoint_sha256"
    ]
    selection["source_phase_grid_sha256"] = sha256_file(source_phase_grid)
    records = selection["checkpoints"]
    rendered = (
        next(
            record
            for record in records
            if record["checkpoint_step"] == selection["selected_step"]
        )
        if selection["policy_retained"]
        else max(
            records,
            key=lambda record: (
                record["minimum"],
                record["median"],
                record["mean"],
                -record["checkpoint_step"],
            ),
        )
    )
    selection["rendered_step"] = rendered["checkpoint_step"]
    selection["render_purpose"] = (
        "retained-policy" if selection["policy_retained"] else "diagnostic-only"
    )
    _plot_survival(selection, output_root / "learning_curves.png")
    render_directory = output_root / "selected_preview"
    subprocess.run(
        _render_command(
            checkpoint=(
                run_directory / f"checkpoint_step_{rendered['checkpoint_step']}.pkl"
            ),
            reference=reference,
            output=render_directory,
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    render_summary_path = render_directory / "summary.json"
    render_video_path = render_directory / "evaluation.mp4"
    render_contact_sheet_path = render_directory / "contact_sheet.png"
    render_summary = json.loads(render_summary_path.read_text(encoding="utf-8"))
    rendered_checkpoint = (
        run_directory / f"checkpoint_step_{rendered['checkpoint_step']}.pkl"
    )
    if (
        render_summary.get("evaluation_start_phase") != 0
        or render_summary.get("tracking_root_velocity_weight")
        != ROOT_VELOCITY_WEIGHT
        or render_summary.get("checkpoint_path")
        != str(rendered_checkpoint.resolve())
        or render_summary.get("checkpoint_sha256")
        != rendered["checkpoint_sha256"]
        or render_summary.get("reference_sha256") != sha256_file(reference)
        or not render_video_path.is_file()
        or not render_contact_sheet_path.is_file()
    ):
        raise ValueError("root-velocity preview is invalid")
    selection.update(
        render_checkpoint_sha256=render_summary["checkpoint_sha256"],
        render_summary_sha256=sha256_file(render_summary_path),
        render_mp4_sha256=sha256_file(render_video_path),
        render_contact_sheet_sha256=sha256_file(render_contact_sheet_path),
        learning_curves_sha256=sha256_file(
            output_root / "learning_curves.png"
        ),
    )
    _write_json_atomically(output_root / "selection.json", selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("root-velocity treatment seed must equal zero")
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_e026_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-root-velocity-continuation-preflight-v1",
        root_velocity_weight=ROOT_VELOCITY_WEIGHT,
        start_step=START_STEP,
        end_step=END_STEP,
        updates=UPDATES,
        checkpoint_steps=list(expected_checkpoint_steps()),
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_root_velocity_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
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
    validation = validate_training_artifacts(
        run_directory,
        expected_kwargs=kwargs,
        source_checkpoint=args.resume_from.resolve(),
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    evaluate_and_select(
        run_directory,
        source_checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        output_root=output_root,
        code_commit=args.code_commit,
    )
    print(run_directory)


if __name__ == "__main__":
    main()
