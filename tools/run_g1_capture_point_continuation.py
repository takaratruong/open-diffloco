"""Train one frozen-E026 residual with dense capture-point tracking."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.run_g1_learned_torso_wrench import (
    EXPECTED_CHECKPOINT_SHA256,
    build_learned_wrench_kwargs,
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


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the immutable four-checkpoint treatment grid."""
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def build_capture_point_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    capture_weight: float,
) -> dict[str, Any]:
    """Apply only a new joint residual and capture-point auxiliary loss."""
    if not math.isfinite(capture_weight) or capture_weight <= 0.0:
        raise ValueError("capture weight must be positive and finite")
    kwargs = build_learned_wrench_kwargs(
        profile_name, reference_path, seed, resume_from
    )
    kwargs.update(
        actor_learned_torso_wrench=False,
        torso_wrench_assistance=False,
        actor_cagrad=True,
        allow_resume_actor_cagrad_change=False,
        actor_frozen_controller_residual=True,
        actor_frozen_controller_residual_hidden=256,
        allow_resume_actor_frozen_controller_residual_start=True,
        actor_capture_point_tracking=True,
        actor_capture_point_delta=0.1,
        actor_capture_point_weight=float(capture_weight),
        allow_resume_actor_capture_point_tracking_start=True,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    for name in (
        "actor_learned_torso_wrench_hidden",
        "actor_learned_torso_wrench_scale",
        "actor_learned_torso_wrench_penalty",
        "allow_resume_actor_learned_torso_wrench_start",
    ):
        kwargs.pop(name, None)
    return kwargs


def _finite_tree(tree: object) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            if not bool(np.all(np.isfinite(np.asarray(leaf)))):
                return False
        except TypeError:
            if not hasattr(leaf, "__dict__") or not _finite_tree(vars(leaf)):
                return False
    return True


def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: dict[str, Any],
    source_checkpoint: Path,
) -> dict[str, object]:
    """Require an immutable parent, finite adapter, and capture telemetry."""
    root = run_directory.resolve()
    hparams = json.loads((root / "hparams.json").read_text(encoding="utf-8"))
    required = {
        "actor_frozen_controller_residual": True,
        "actor_capture_point_tracking": True,
        "actor_capture_point_delta": 0.1,
        "actor_capture_point_weight": expected_kwargs[
            "actor_capture_point_weight"
        ],
        "actor_learned_torso_wrench": False,
        "torso_wrench_assistance": False,
        "total_steps": END_STEP,
    }
    if any(hparams.get(name) != value for name, value in required.items()):
        raise ValueError("capture-point hparams do not match the treatment")
    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    source_actor_hash = parameter_tree_sha256(source.actor_params)
    source_normalizer_hash = parameter_tree_sha256(source.normalizer)
    for step in expected_checkpoint_steps():
        with (root / f"checkpoint_step_{step}.pkl").open("rb") as stream:
            state = pickle.load(stream)
        if int(state.step) != step or not isinstance(
            state.actor_params, FrozenControllerResidualParams
        ):
            raise ValueError("capture-point checkpoint structure is invalid")
        if not _finite_tree(state):
            raise ValueError("capture-point checkpoint contains nonfinite state")
        if (
            parameter_tree_sha256(state.actor_params.parent) != source_actor_hash
            or parameter_tree_sha256(state.normalizer) != source_normalizer_hash
        ):
            raise ValueError("frozen E026 state changed")

    rows = json.loads(
        (root / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    if [row.get("step") for row in rows] != list(expected_checkpoint_steps()):
        raise ValueError("capture-point checkpoint telemetry grid is invalid")
    scalar_keys = (
        "actor_preview_gradient_norm",
        "actor_preview_update_norm",
        "actor_capture_point_loss",
        "actor_capture_point_p99_norm",
    )
    for row in rows:
        components = np.asarray(
            row.get("actor_capture_point_component_rms"), dtype=np.float64
        )
        if (
            row.get("actor_preview_valid") is not True
            or any(
                not isinstance(row.get(key), (int, float))
                or isinstance(row.get(key), bool)
                or not math.isfinite(float(row[key]))
                for key in scalar_keys
            )
            or float(row["actor_preview_gradient_norm"]) <= 0.0
            or float(row["actor_preview_update_norm"]) <= 0.0
            or not isinstance(row.get("actor_capture_point_valid_count"), int)
            or int(row["actor_capture_point_valid_count"]) <= 0
            or components.shape != (2,)
            or not np.isfinite(components).all()
            or row.get("actor_preview_frozen_parameter_drift_max_abs") != 0.0
            or row.get("actor_preview_frozen_moment_drift_max_abs") != 0.0
            or row.get("actor_preview_normalizer_drift_max_abs") != 0.0
        ):
            raise ValueError("capture-point checkpoint telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-capture-point-continuation-training-v1",
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "capture_weight": expected_kwargs["actor_capture_point_weight"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--capture-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        protocol="g1-capture-point-continuation-preflight-v1",
        start_step=START_STEP,
        end_step=END_STEP,
        updates=UPDATES,
        checkpoint_steps=list(expected_checkpoint_steps()),
        capture_weight=args.capture_weight,
        scientific_delta=[
            "actor_frozen_controller_residual",
            "actor_capture_point_tracking",
            "actor_capture_point_weight",
            "reference_path",
            "total_steps",
        ],
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_capture_point_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        capture_weight=args.capture_weight,
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
    print(run_directory)


if __name__ == "__main__":
    main()
