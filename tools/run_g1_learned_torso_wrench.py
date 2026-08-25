"""Train a zero-start learned torso-wrench head on frozen E026 control."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.learned_torso_wrench import (
    FrozenControllerWrenchParams,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_anchored_carried_recovery import (
    build_anchored_carried_recovery_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _write_json_atomically,
)


START_STEP = 1_769_472
UPDATES = 64
CHECKPOINT_EVERY_UPDATES = 8
TRANSITIONS_PER_UPDATE = 512 * 24
END_STEP = START_STEP + UPDATES * TRANSITIONS_PER_UPDATE
EXPECTED_CHECKPOINT_SHA256 = (
    "4f9a2b49c7368f5323ab81c4c3de4aae208413987ab4858c44bf76872d0f86dd"
)
EXPECTED_HPARAMS_SHA256 = (
    "6b60d0b8ea96fa27d633c6f80f8df82a6c09c848d58b9636fd75759bbda486f7"
)
EXPECTED_REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the fixed eight-checkpoint learned-wrench grid."""
    interval = CHECKPOINT_EVERY_UPDATES * TRANSITIONS_PER_UPDATE
    return tuple(range(START_STEP + interval, END_STEP + 1, interval))


def build_learned_wrench_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Change only the reference reset distribution and learned wrench head."""
    # The carried path argument is removed below; using the established E026
    # builder keeps every other controller and SHAC setting exact.
    kwargs = build_anchored_carried_recovery_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        "/unused/carried-reset-bank.npz",
    )
    kwargs.update(
        resume_from=str(resume_from),
        reference_path=str(reference_path),
        allow_resume_reference_path_change=True,
        carried_reset_bank_path=None,
        carried_reset_probability=0.0,
        carried_reset_bank_start=0,
        allow_resume_carried_reset_change=True,
        actor_policy_anchor_weight=0.0,
        actor_policy_anchor_source_path=None,
        actor_policy_anchor_source_sha256=None,
        allow_resume_actor_policy_anchor_source_change=True,
        actor_cagrad=False,
        allow_resume_actor_cagrad_change=True,
        actor_learned_torso_wrench=True,
        actor_learned_torso_wrench_hidden=256,
        actor_learned_torso_wrench_scale=1.0,
        actor_learned_torso_wrench_penalty=0.0,
        allow_resume_actor_learned_torso_wrench_start=True,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    kwargs.pop("checkpoint_interval", None)
    return kwargs


def validate_preflight(
    *,
    repository: Path,
    checkpoint: Path,
    reference: Path,
    code_commit: str,
) -> dict[str, object]:
    """Bind clean code and the immutable E026/reference inputs."""
    head = _git_output(repository, "rev-parse", "HEAD")
    dirty = _git_output(repository, "status", "--porcelain")
    hparams = checkpoint.with_name("hparams.json")
    if head != code_commit or dirty:
        raise ValueError("learned-wrench execution requires the clean registered commit")
    expected = (
        (checkpoint, EXPECTED_CHECKPOINT_SHA256, "E026 checkpoint"),
        (hparams, EXPECTED_HPARAMS_SHA256, "E026 hparams"),
        (reference, EXPECTED_REFERENCE_SHA256, "continuous reference"),
    )
    for path, digest, label in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{label} SHA-256 does not match")
    return {
        "valid": True,
        "protocol": "g1-learned-torso-wrench-preflight-v1",
        "code_commit": code_commit,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "hparams_sha256": EXPECTED_HPARAMS_SHA256,
        "reference": str(reference.resolve()),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "start_step": START_STEP,
        "end_step": END_STEP,
        "updates": UPDATES,
        "checkpoint_steps": list(expected_checkpoint_steps()),
    }


def _finite_tree(tree: object) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            array = np.asarray(leaf)
            if not bool(np.all(np.isfinite(array))):
                return False
        except TypeError:
            if not hasattr(leaf, "__dict__") or not _finite_tree(vars(leaf)):
                return False
    return True


def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_steps: Sequence[int],
    source_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Fail closed on malformed, nonfinite, or controller-changing output."""
    steps = tuple(int(step) for step in expected_steps)
    hparams = json.loads((run_directory / "hparams.json").read_text())
    expected_hparams = {
        "actor_learned_torso_wrench": True,
        "actor_learned_torso_wrench_scale": 1.0,
        "actor_learned_torso_wrench_penalty": 0.0,
    }
    if any(hparams.get(key) != value for key, value in expected_hparams.items()):
        raise ValueError("learned torso wrench hparams are invalid")

    checkpoint_states = []
    for step in steps:
        path = run_directory / f"checkpoint_step_{step}.pkl"
        with path.open("rb") as stream:
            state = pickle.load(stream)
        if int(state.step) != step:
            raise ValueError("learned wrench checkpoint step is invalid")
        if not isinstance(state.actor_params, FrozenControllerWrenchParams):
            raise ValueError("checkpoint is missing the learned wrench wrapper")
        if not _finite_tree(state):
            raise ValueError("learned wrench checkpoint contains nonfinite state")
        checkpoint_states.append(state)

    if source_checkpoint is not None:
        with source_checkpoint.open("rb") as stream:
            source = pickle.load(stream)
        source_hash = parameter_tree_sha256(source.actor_params)
        source_normalizer_hash = parameter_tree_sha256(source.normalizer)
        for state in checkpoint_states:
            if parameter_tree_sha256(state.actor_params.controller) != source_hash:
                raise ValueError("frozen E026 controller changed")
            if parameter_tree_sha256(state.normalizer) != source_normalizer_hash:
                raise ValueError("frozen E026 normalizer changed")

    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text()
    )
    if [row.get("step") for row in rows] != list(steps):
        raise ValueError("learned wrench checkpoint telemetry grid is invalid")
    finite_keys = (
        "actor_preview_gradient_norm",
        "actor_preview_update_norm",
        "learned_torso_wrench_rms_force",
        "learned_torso_wrench_rms_torque",
        "learned_torso_wrench_max_force",
        "learned_torso_wrench_max_torque",
        "learned_torso_wrench_saturation_fraction",
    )
    for row in rows:
        if row.get("actor_preview_valid") is not True or row.get(
            "learned_torso_wrench_valid"
        ) is not True:
            raise ValueError("learned wrench checkpoint telemetry is invalid")
        if any(
            not isinstance(row.get(key), (int, float))
            or isinstance(row.get(key), bool)
            or not math.isfinite(float(row[key]))
            for key in finite_keys
        ):
            raise ValueError("learned wrench checkpoint telemetry is nonfinite")
        if (
            row.get("actor_preview_frozen_parameter_drift_max_abs") != 0.0
            or row.get("actor_preview_frozen_moment_drift_max_abs") != 0.0
            or row.get("actor_preview_normalizer_drift_max_abs") != 0.0
            or float(row["actor_preview_gradient_norm"]) <= 0.0
            or float(row["actor_preview_update_norm"]) <= 0.0
        ):
            raise ValueError("learned wrench update did not preserve frozen state")
    return {
        "valid": True,
        "protocol": "g1-learned-torso-wrench-training-v1",
        "checkpoint_steps": list(steps),
        "controller_frozen": source_checkpoint is not None,
    }


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
    kwargs = build_learned_wrench_kwargs(
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
        expected_steps=expected_checkpoint_steps(),
        source_checkpoint=args.resume_from.resolve(),
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
