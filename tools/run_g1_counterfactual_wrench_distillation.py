"""Train one zero-wrench leg residual from the frozen E004 teacher."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.algorithm import (
    build_counterfactual_wrench_telemetry,
    train,
)
from src.algorithms.shac.counterfactual_wrench_distillation import (
    load_counterfactual_feasibility,
    resolve_leg_action_indices,
)
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.algorithms.shac.residual_preview_adapter import (
    split_residual_adapter_params,
)
from src.core.rmr_action_noise import RMR_ACTION_STD_JOINT_NAMES
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.evaluate_g1_counterfactual_wrench_feasibility import (
    EXPECTED_E026_TREE_SHA256,
    EXPECTED_HPARAMS_SHA256,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TEACHER_TREE_SHA256,
    EXPECTED_WRENCH_TREE_SHA256,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_capture_point_continuation import (
    build_capture_point_kwargs,
    validate_training_artifacts as validate_parent_training_artifacts,
)
from tools.run_g1_learned_torso_wrench import (
    validate_preflight as validate_e026_preflight,
)
from tools.run_g1_fresh_ppo_action_contract_walk import _validate_cagrad_row
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_769_472
UPDATES = 32
TRANSITIONS_PER_UPDATE = 512 * 24
CHECKPOINT_EVERY_UPDATES = 8
CHECKPOINT_INTERVAL = CHECKPOINT_EVERY_UPDATES * TRANSITIONS_PER_UPDATE
END_STEP = START_STEP + UPDATES * TRANSITIONS_PER_UPDATE
E026_SURVIVAL = (131, 114, 74, 71, 74)
EXPECTED_FEASIBILITY_SHA256 = (
    "0a1e5bceddc70b440e397891f1d9808b596bb0a5939f7c512af9cfe09499cc68"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the immutable four-checkpoint treatment grid."""
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def classify_counterfactual_selection(
    candidates: dict[int, dict[str, object]],
) -> dict[str, object]:
    """Select only a checkpoint that improves without phase regression."""
    if set(candidates) != set(expected_checkpoint_steps()):
        raise ValueError("counterfactual selection requires the exact checkpoint grid")
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
            raise ValueError("counterfactual candidate is invalid")
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
    return {
        "protocol": "g1-counterfactual-wrench-distillation-selection-v1",
        "phases": [0, 25, 50, 75, 100],
        "e026_survival": list(E026_SURVIVAL),
        "checkpoints": records,
        "outcome": (
            "leg-counterfactual-advances"
            if selected is not None
            else "leg-counterfactual-feasible-but-insufficient"
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


def build_counterfactual_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    teacher_path: str | Path,
    teacher_sha256: str,
    feasibility_path: str | Path,
    feasibility_sha256: str,
) -> dict[str, Any]:
    """Apply only the fixed leg-counterfactual treatment over exact E026."""
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
        actor_counterfactual_wrench_distillation=True,
        actor_counterfactual_wrench_teacher_path=str(Path(teacher_path).resolve()),
        actor_counterfactual_wrench_teacher_sha256=teacher_sha256,
        actor_counterfactual_wrench_feasibility_path=str(
            Path(feasibility_path).resolve()
        ),
        actor_counterfactual_wrench_feasibility_sha256=feasibility_sha256,
        allow_resume_actor_counterfactual_wrench_distillation_start=True,
        domain_randomization=False,
        actor_observation_noise=False,
        push_velocity_range=(0.0, 0.0),
        push_interval_s=1_000_000_000.0,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    return kwargs


def validate_preflight(
    *,
    repository: Path,
    checkpoint: Path,
    reference: Path,
    teacher: Path,
    teacher_sha256: str,
    feasibility: Path,
    feasibility_sha256: str,
    code_commit: str,
    seed: int,
) -> dict[str, object]:
    """Bind the exact parent, teacher, feasibility evidence, and clean code."""
    if seed != 0:
        raise ValueError("counterfactual treatment seed must equal zero")
    parent = validate_e026_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=code_commit,
    )
    if teacher_sha256 != EXPECTED_TEACHER_SHA256:
        raise ValueError("counterfactual teacher is not the registered E004 checkpoint")
    if feasibility_sha256 != EXPECTED_FEASIBILITY_SHA256:
        raise ValueError("counterfactual feasibility is not the registered artifact")
    if not teacher.is_file() or sha256_file(teacher) != teacher_sha256:
        raise ValueError("counterfactual teacher SHA-256 does not match")
    hparams_path = teacher.with_name("hparams.json")
    if (
        not hparams_path.is_file()
        or sha256_file(hparams_path) != EXPECTED_HPARAMS_SHA256
    ):
        raise ValueError("counterfactual teacher hparams do not match E004")
    with teacher.open("rb") as stream:
        teacher_state = pickle.load(stream)
    if parameter_tree_sha256(teacher_state.actor_params) != EXPECTED_TEACHER_TREE_SHA256:
        raise ValueError("counterfactual teacher parameter tree does not match E004")
    report = load_counterfactual_feasibility(
        feasibility, expected_sha256=feasibility_sha256
    )
    if report.teacher_checkpoint_sha256 != teacher_sha256:
        raise ValueError("feasibility evidence binds a different teacher")
    if (
        report.teacher_tree_sha256 != EXPECTED_TEACHER_TREE_SHA256
        or report.e026_tree_sha256 != EXPECTED_E026_TREE_SHA256
        or report.wrench_tree_sha256 != EXPECTED_WRENCH_TREE_SHA256
    ):
        raise ValueError("feasibility evidence parameter provenance does not match")
    return {
        **parent,
        "protocol": "g1-counterfactual-wrench-distillation-preflight-v1",
        "teacher_checkpoint": str(teacher.resolve()),
        "teacher_checkpoint_sha256": teacher_sha256,
        "feasibility_artifact": str(feasibility.resolve()),
        "feasibility_artifact_sha256": feasibility_sha256,
        "feasibility_outcome": "leg-counterfactual-feasible",
        "start_step": START_STEP,
        "end_step": END_STEP,
        "updates": UPDATES,
        "checkpoint_steps": list(expected_checkpoint_steps()),
    }


def validate_training_artifacts(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    teacher_sha256: str,
    feasibility_sha256: str,
) -> dict[str, object]:
    """Require exact frozen state, 12-output adapter, and zero-wrench evidence."""
    root = run_directory.resolve()
    parent_report = validate_parent_training_artifacts(
        root,
        expected_kwargs={
            "actor_capture_point_tracking": False,
            "actor_capture_point_weight": 1.0,
        },
        source_checkpoint=source_checkpoint,
    )
    hparams = json.loads((root / "hparams.json").read_text(encoding="utf-8"))
    expected_indices = list(
        resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)
    )
    required = {
        "actor_counterfactual_wrench_distillation": True,
        "actor_counterfactual_wrench_teacher_sha256": teacher_sha256,
        "actor_counterfactual_wrench_feasibility_sha256": feasibility_sha256,
        "actor_counterfactual_wrench_leg_indices": expected_indices,
        "actor_counterfactual_wrench_loss_weight": 1.0,
        "actor_counterfactual_wrench_residual_magnitude_weight": 0.01,
        "actor_counterfactual_wrench_residual_temporal_weight": 0.001,
    }
    if any(hparams.get(name) != value for name, value in required.items()):
        raise ValueError("counterfactual hparams do not match the treatment")
    archives = sorted(root.glob("checkpoint_step_*.pkl"))
    if [path.name for path in archives] != [
        f"checkpoint_step_{step}.pkl" for step in expected_checkpoint_steps()
    ]:
        raise ValueError("counterfactual checkpoint archive set is invalid")
    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    source_actor_sha = parameter_tree_sha256(source.actor_params)
    for path, step in zip(archives, expected_checkpoint_steps(), strict=True):
        with path.open("rb") as stream:
            state = pickle.load(stream)
        if (
            int(state.step) != step
            or not isinstance(state.actor_params, FrozenControllerResidualParams)
            or parameter_tree_sha256(state.actor_params.parent) != source_actor_sha
        ):
            raise ValueError("counterfactual checkpoint structure is invalid")
        _, adapter_aux = split_residual_adapter_params(
            state.actor_params.adapter
        )
        if adapter_aux.dense1_bias.shape != (12,) or not all(
            np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(state.actor_params.adapter)
        ):
            raise ValueError("counterfactual adapter is invalid")
    rows = json.loads(
        (root / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    if [row.get("step") for row in rows] != list(expected_checkpoint_steps()):
        raise ValueError("counterfactual telemetry grid is invalid")
    for row in rows:
        _validate_cagrad_row(
            row,
            step=int(row["step"]),
            expected_action_noise=hparams["action_noise_std_end"],
            expected_actor_bootstrap_scale=0.0,
        )
        build_counterfactual_wrench_telemetry(row)
    return {
        **parent_report,
        "protocol": "g1-counterfactual-wrench-distillation-training-v1",
        "counterfactual_valid": True,
        "teacher_checkpoint_sha256": teacher_sha256,
        "feasibility_artifact_sha256": feasibility_sha256,
    }


def _phase_grid_command(
    *,
    checkpoint: Path,
    reference: Path,
    output: Path,
    code_commit: str,
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
        "--solver-profile",
        "g1-4x5",
        "--render-every",
        "2",
    ]


def _plot_evaluation_curves(
    *,
    training_rows: list[dict[str, object]],
    selection: dict[str, object],
    output: Path,
) -> None:
    """Publish the compact treatment-loss and survival diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in training_rows]
    losses = [float(row["actor_counterfactual_loss"]) for row in training_rows]
    records = selection["checkpoints"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, losses, marker="o")
    axes[0].set_title("Counterfactual objective")
    axes[0].set_xlabel("transition step")
    axes[0].set_ylabel("loss")
    for phase_index, phase in enumerate(selection["phases"]):
        axes[1].plot(
            [record["checkpoint_step"] for record in records],
            [record["survival"][phase_index] for record in records],
            marker="o",
            label=f"phase {phase}",
        )
        axes[1].axhline(
            E026_SURVIVAL[phase_index], linestyle="--", linewidth=0.7
        )
    axes[1].set_title("Replay-free phase survival")
    axes[1].set_xlabel("transition step")
    axes[1].set_ylabel("steps survived")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.replace(temporary, output)


def evaluate_and_select(
    run_directory: Path,
    *,
    reference: Path,
    output_root: Path,
    code_commit: str,
) -> dict[str, object]:
    """Evaluate every archive, select fail-closed, and render one diagnostic."""
    evaluation_root = output_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "MUJOCO_GL": "egl",
    }
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
            or not isinstance(summary, dict)
            or summary.get("phases") != [0, 25, 50, 75, 100]
            or not isinstance(survival, list)
        ):
            raise ValueError("counterfactual phase-grid evidence is invalid")
        candidates[step] = {
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "survival": survival,
        }
    selection = classify_counterfactual_selection(candidates)
    records = selection["checkpoints"]
    diagnostic = (
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
    selection["rendered_step"] = diagnostic["checkpoint_step"]
    selection["render_purpose"] = (
        "retained-policy"
        if selection["policy_retained"]
        else "diagnostic-only"
    )
    _write_json_atomically(output_root / "selection.json", selection)
    training_rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    _plot_evaluation_curves(
        training_rows=training_rows,
        selection=selection,
        output=output_root / "learning_curves.png",
    )
    render_directory = output_root / "selected_preview"
    subprocess.run(
        _render_command(
            checkpoint=(
                run_directory
                / f"checkpoint_step_{diagnostic['checkpoint_step']}.pkl"
            ),
            reference=reference,
            output=render_directory,
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    render_summary = json.loads(
        (render_directory / "summary.json").read_text(encoding="utf-8")
    )
    if (
        render_summary.get("evaluation_start_phase") != 0
        or render_summary.get("counterfactual_leg_residual_evidence") is not True
        or render_summary.get("counterfactual_student_torso_wrench_max_abs") != 0.0
        or not (render_directory / "evaluation.mp4").is_file()
        or not (render_directory / "contact_sheet.png").is_file()
    ):
        raise ValueError("counterfactual selected preview is invalid")
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--feasibility-artifact", type=Path, required=True)
    parser.add_argument("--feasibility-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        teacher=args.teacher_checkpoint.resolve(),
        teacher_sha256=args.teacher_sha256,
        feasibility=args.feasibility_artifact.resolve(),
        feasibility_sha256=args.feasibility_sha256,
        code_commit=args.code_commit,
        seed=args.seed,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_counterfactual_kwargs(
        args.solver_profile,
        args.reference_path,
        args.seed,
        args.resume_from,
        args.teacher_checkpoint,
        args.teacher_sha256,
        args.feasibility_artifact,
        args.feasibility_sha256,
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
        source_checkpoint=args.resume_from.resolve(),
        teacher_sha256=args.teacher_sha256,
        feasibility_sha256=args.feasibility_sha256,
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    evaluate_and_select(
        run_directory,
        reference=args.reference_path.resolve(),
        output_root=output_root,
        code_commit=args.code_commit,
    )
    print(run_directory)


if __name__ == "__main__":
    main()
