"""Train one zero-wrench leg residual from the frozen E004 teacher."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
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


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the immutable four-checkpoint treatment grid."""
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


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
    if not teacher.is_file() or sha256_file(teacher) != teacher_sha256:
        raise ValueError("counterfactual teacher SHA-256 does not match")
    report = load_counterfactual_feasibility(
        feasibility, expected_sha256=feasibility_sha256
    )
    if report.teacher_checkpoint_sha256 != teacher_sha256:
        raise ValueError("feasibility evidence binds a different teacher")
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
    print(run_directory)


if __name__ == "__main__":
    main()
