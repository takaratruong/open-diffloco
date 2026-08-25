"""Continuously anneal a learned torso wrench while adapting joint residuals."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import jax

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.learned_torso_wrench import (
    FrozenControllerWrenchParams,
    learned_wrench_scale_at_step,
)
from src.algorithms.shac.residual_preview_adapter import FrozenPreviewResidualParams
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_learned_torso_wrench import build_learned_wrench_kwargs
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _write_json_atomically,
)


TRANSITIONS_PER_UPDATE = 12_288
START_STEP = 1_966_080
ANNEAL_UPDATES = 128
ZERO_TAIL_UPDATES = 16
CHECKPOINT_EVERY_UPDATES = 8
ANNEAL_END_STEP = START_STEP + ANNEAL_UPDATES * TRANSITIONS_PER_UPDATE
END_STEP = ANNEAL_END_STEP + ZERO_TAIL_UPDATES * TRANSITIONS_PER_UPDATE
EXPECTED_CHECKPOINT_SHA256 = (
    "b7fd54a82380e032f91da6e12b7252f2bd42f4a1f5fe6be0f206849282811870"
)
EXPECTED_HPARAMS_SHA256 = (
    "6838465c6b6190a9ab165c82d61b35effd93ceab613d1d18993ea8b3154bffb6"
)
EXPECTED_REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    interval = CHECKPOINT_EVERY_UPDATES * TRANSITIONS_PER_UPDATE
    return tuple(range(START_STEP + interval, END_STEP + 1, interval))


def build_anneal_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    kwargs = build_learned_wrench_kwargs(
        profile_name, reference_path, seed, resume_from
    )
    kwargs.update(
        allow_resume_actor_learned_torso_wrench_start=False,
        allow_resume_actor_learned_torso_wrench_change=True,
        actor_learned_torso_wrench_scale=1.0,
        actor_learned_torso_wrench_scale_end=0.0,
        actor_learned_torso_wrench_scale_start_step=START_STEP,
        actor_learned_torso_wrench_scale_end_step=ANNEAL_END_STEP,
        actor_learned_torso_wrench_condition_on_scale=True,
        actor_learned_torso_wrench_train_controller=True,
        actor_learned_torso_wrench_penalty=0.01,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    return kwargs


def validate_preflight(
    *, repository: Path, checkpoint: Path, reference: Path, code_commit: str
) -> dict[str, object]:
    if _git_output(repository, "rev-parse", "HEAD") != code_commit:
        raise ValueError("anneal execution commit does not match registered code")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("anneal execution requires a clean worktree")
    hparams = checkpoint.with_name("hparams.json")
    expected = (
        (checkpoint, EXPECTED_CHECKPOINT_SHA256, "parent checkpoint"),
        (hparams, EXPECTED_HPARAMS_SHA256, "parent hparams"),
        (reference, EXPECTED_REFERENCE_SHA256, "reference"),
    )
    for path, digest, label in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{label} provenance mismatch")
    return {
        "valid": True,
        "code_commit": code_commit,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "hparams_sha256": EXPECTED_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
    }


def validate_training_artifacts(
    run_directory: Path, *, parent_checkpoint: Path
) -> dict[str, object]:
    expected_steps = expected_checkpoint_steps()
    archives = sorted(run_directory.glob("checkpoint_step_*.pkl"))
    observed_steps = tuple(int(path.stem.rsplit("_", 1)[1]) for path in archives)
    if observed_steps != expected_steps:
        raise ValueError("anneal checkpoint grid is incomplete or contains extras")
    with parent_checkpoint.open("rb") as handle:
        parent = pickle.load(handle)
    if not isinstance(parent.actor_params, FrozenControllerWrenchParams):
        raise ValueError("anneal parent is not a learned-wrench checkpoint")
    parent_root_sha = parameter_tree_sha256(parent.actor_params.controller.parent)
    parent_normalizer_sha = parameter_tree_sha256(parent.normalizer)
    for archive in archives:
        with archive.open("rb") as handle:
            state = pickle.load(handle)
        if (
            not isinstance(state.actor_params, FrozenControllerWrenchParams)
            or not isinstance(
                state.actor_params.controller, FrozenPreviewResidualParams
            )
            or parameter_tree_sha256(state.actor_params.controller.parent)
            != parent_root_sha
            or parameter_tree_sha256(state.normalizer) != parent_normalizer_sha
        ):
            raise ValueError("anneal changed a frozen parameter boundary")
        if not all(
            bool(jax.numpy.all(jax.numpy.isfinite(leaf)))
            for leaf in jax.tree_util.tree_leaves(state)
            if hasattr(leaf, "dtype")
        ):
            raise ValueError("anneal checkpoint contains nonfinite state")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if tuple(int(row["step"]) for row in rows) != expected_steps:
        raise ValueError("anneal telemetry grid does not match checkpoints")
    for row in rows:
        expected_scale = float(
            learned_wrench_scale_at_step(
                int(row["step"]) - TRANSITIONS_PER_UPDATE,
                start_step=START_STEP,
                end_step=ANNEAL_END_STEP,
                start_scale=1.0,
                end_scale=0.0,
            )
        )
        scalars = (
            row.get("learned_torso_wrench_scale"),
            row.get("actor_preview_gradient_norm"),
            row.get("actor_preview_update_norm"),
        )
        if (
            not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scalars)
            or not math.isclose(scalars[0], expected_scale, abs_tol=1e-7)
            or scalars[1] <= 0.0
            or scalars[2] <= 0.0
            or row.get("actor_preview_valid") is not True
            or row.get("learned_torso_wrench_valid") is not True
        ):
            raise ValueError("anneal checkpoint telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-learned-torso-wrench-continuous-anneal-v1",
        "checkpoint_steps": list(expected_steps),
        "anneal_end_step": ANNEAL_END_STEP,
        "zero_tail_updates": ZERO_TAIL_UPDATES,
        "parent_frozen": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-profile", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
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
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_anneal_kwargs(
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
        run_directory, parent_checkpoint=args.resume_from.resolve()
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
