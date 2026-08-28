"""Run a bounded JAVE-on-SHAC continuation from retained G1 E002."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.counterfactual_wrench_distillation import (
    parameter_tree_sha256,
)
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.run_g1_root_velocity_continuation import (
    build_root_velocity_kwargs,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _write_json_atomically,
)


SOURCE_STEP = 1_867_776
TRANSITIONS_PER_UPDATE = 512 * 24
JAVE_VG_WEIGHT = 0.1
EXPECTED_SOURCE_SHA256 = (
    "52aa142dabf382671a5fe7e6b1f26954b77e4fde492bb413a25b85358a1c4325"
)
EXPECTED_REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_checkpoint_steps(updates: int) -> tuple[int, ...]:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    return tuple(
        SOURCE_STEP + TRANSITIONS_PER_UPDATE * index
        for index in range(1, updates + 1)
    )


def build_jave_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    enabled: bool,
    updates: int,
    warmup_updates: int = 1,
) -> dict[str, Any]:
    """Preserve E002 and change only bootstrap authority plus JAVE by arm."""

    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")
    if (
        isinstance(warmup_updates, bool)
        or not isinstance(warmup_updates, int)
        or warmup_updates < 0
        or warmup_updates >= updates
    ):
        raise ValueError("warmup_updates must be in [0, updates)")
    kwargs = build_root_velocity_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        enabled=True,
    )
    kwargs.update(
        actor_bootstrap_scale=1.0,
        allow_resume_actor_bootstrap_scale_change=True,
        jave_vg_weight=(JAVE_VG_WEIGHT if enabled else 0.0),
        jave_vg_warmup_steps=warmup_updates * TRANSITIONS_PER_UPDATE,
        jave_ldm_hidden=(256, 256),
        jave_ldm_lr=3e-4,
        jave_ldm_iterations=4,
        jave_ldm_batch_size=256,
        jave_vg_batch_size=256,
        jave_ldm_buffer_capacity=100_000,
        jave_reward_feature_scale=8.0,
        allow_resume_jave_start=enabled,
        total_steps=SOURCE_STEP + updates * TRANSITIONS_PER_UPDATE,
        checkpoint_steps=expected_checkpoint_steps(updates),
    )
    return kwargs


def _finite_tree(tree) -> bool:
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            return False
    return True


def validate_preflight(
    *,
    repository: Path,
    checkpoint: Path,
    reference: Path,
    code_commit: str,
) -> dict[str, object]:
    """Bind the executable, retained state, model, and reference before GPU use."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    model = Path(
        "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
        "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
    )
    checkpoint_hparams = json.loads(
        (checkpoint.parent / "hparams.json").read_text(encoding="utf-8")
    )
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    required_hparams = {
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "tracking_root_velocity_weight": 1.0,
        "actor_bootstrap_scale": 0.0,
        "actor_frozen_controller_residual": True,
        "actor_residual_preview_adapter": True,
        "actor_cagrad": True,
        "gradient_accumulation_steps": 2,
        "num_envs": 256,
        "unroll_length": 24,
    }
    errors = []
    if head != code_commit:
        errors.append("code commit mismatch")
    if status:
        errors.append("code worktree is dirty")
    if sha256_file(checkpoint) != EXPECTED_SOURCE_SHA256:
        errors.append("source checkpoint hash mismatch")
    if sha256_file(reference) != EXPECTED_REFERENCE_SHA256:
        errors.append("reference hash mismatch")
    if sha256_file(model) != EXPECTED_MODEL_SHA256:
        errors.append("model hash mismatch")
    if int(state.step) != SOURCE_STEP or not _finite_tree(state):
        errors.append("source checkpoint state is invalid")
    if any(
        checkpoint_hparams.get(name) != value
        for name, value in required_hparams.items()
    ):
        errors.append("source checkpoint hparams mismatch")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "valid": True,
        "protocol": "g1-jave-continuation-preflight-v1",
        "authoritative_entrypoint": (
            "python -m tools.run_g1_jave_continuation"
        ),
        "code_commit": head,
        "source_step": SOURCE_STEP,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "reference": str(reference.resolve()),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
    }


def validate_training_artifacts(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    enabled: bool,
    updates: int,
    warmup_updates: int,
) -> dict[str, object]:
    """Require finite checkpoints and complete JAVE state/telemetry."""

    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected_steps = expected_checkpoint_steps(updates)
    expected_hparams = {
        "actor_bootstrap_scale": 1.0,
        "jave_vg_weight": JAVE_VG_WEIGHT if enabled else 0.0,
        "jave_vg_warmup_steps": warmup_updates * TRANSITIONS_PER_UPDATE,
        "jave_start_step": SOURCE_STEP if enabled else 0,
        "allow_resume_jave_start": enabled,
        "total_steps": expected_steps[-1],
    }
    if any(
        hparams.get(name) != value for name, value in expected_hparams.items()
    ):
        raise ValueError("JAVE continuation hparams are invalid")
    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    if not isinstance(source.actor_params, FrozenControllerResidualParams):
        raise ValueError("source actor structure is invalid")
    source_parent_hash = parameter_tree_sha256(source.actor_params.parent)
    source_normalizer_hash = parameter_tree_sha256(source.normalizer)
    checkpoint_hashes = []
    for step in expected_steps:
        checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        if (
            int(state.step) != step
            or not _finite_tree(state)
            or not isinstance(state.actor_params, FrozenControllerResidualParams)
            or parameter_tree_sha256(state.actor_params.parent)
            != source_parent_hash
            or parameter_tree_sha256(state.normalizer)
            != source_normalizer_hash
        ):
            raise ValueError("JAVE continuation checkpoint is invalid")
        has_jave_state = all(
            getattr(state, name, None) is not None
            for name in ("ldm_params", "ldm_opt", "replay_buffer")
        )
        if has_jave_state != enabled:
            raise ValueError("JAVE learned-dynamics state presence is invalid")
        checkpoint_hashes.append(sha256_file(checkpoint))
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if [row.get("step") for row in rows] != list(expected_steps):
        raise ValueError("JAVE checkpoint telemetry grid is invalid")
    if enabled:
        for index, row in enumerate(rows):
            scalars = (
                row.get("jave_ldm_loss"),
                row.get("jave_vg_loss"),
                row.get("jave_vg_target_norm"),
            )
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in scalars
                )
                or not isinstance(row.get("jave_replay_size"), int)
                or row["jave_replay_size"] < TRANSITIONS_PER_UPDATE
                or bool(row.get("jave_vg_active"))
                != (index >= warmup_updates)
            ):
                raise ValueError("JAVE checkpoint telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-jave-continuation-training-v1",
        "arm": "jave" if enabled else "control",
        "checkpoint_steps": list(expected_steps),
        "checkpoint_sha256": checkpoint_hashes,
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "actor_bootstrap_scale": 1.0,
        "jave_vg_weight": JAVE_VG_WEIGHT if enabled else 0.0,
        "jave_warmup_updates": warmup_updates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--arm", choices=("control", "jave"), required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("JAVE continuation seed must equal zero")
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    enabled = args.arm == "jave"
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        arm=args.arm,
        updates=args.updates,
        warmup_updates=args.warmup_updates,
        end_step=SOURCE_STEP + args.updates * TRANSITIONS_PER_UPDATE,
        checkpoint_steps=list(expected_checkpoint_steps(args.updates)),
        actor_bootstrap_scale=1.0,
        jave_vg_weight=JAVE_VG_WEIGHT if enabled else 0.0,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_jave_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        enabled=enabled,
        updates=args.updates,
        warmup_updates=args.warmup_updates,
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
        enabled=enabled,
        updates=args.updates,
        warmup_updates=args.warmup_updates,
    )
    _write_json_atomically(
        output_root / "training_validation.json", validation
    )
    print(run_directory)


if __name__ == "__main__":
    main()
