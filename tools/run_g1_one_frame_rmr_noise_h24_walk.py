"""Run the bounded one-frame-history ablation on E023 walking SHAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

TOTAL_UPDATES = 32
TRANSITIONS_PER_UPDATE = 512 * 24
TOTAL_STEPS = TOTAL_UPDATES * TRANSITIONS_PER_UPDATE
CHECKPOINT_INTERVAL = 16 * TRANSITIONS_PER_UPDATE
ACTOR_HISTORY_LEN = 1
ACTOR_FRAME_OBS_DIM = 328
E023_TOTAL_STEPS = 1_572_864
EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)
EXPECTED_MODEL_PATH = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_PATH = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
RMR_WALK_MODEL_999_SHA256 = (
    "5db9d8371754a635d162c416e192b49ec2064d3133d20eea0df63463d1c8ae03"
)
PHASE_CAPS = (124, 99, 74, 49, 24)
CONTROL_SURVIVAL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}


def expected_checkpoint_steps() -> tuple[int, ...]:
    return (CHECKPOINT_INTERVAL, TOTAL_STEPS)


def build_one_frame_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Change only actor history and the bounded execution budget from E023."""
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        actor_history_len=ACTOR_HISTORY_LEN,
        expected_actor_obs_dim=ACTOR_FRAME_OBS_DIM,
        total_steps=TOTAL_STEPS,
    )
    return kwargs


def _validated_survival(
    treatment: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    if set(treatment) != set(CONTROL_SURVIVAL):
        raise ValueError("survival evidence must contain exactly updates 16 and 32")
    validated: dict[int, tuple[int, ...]] = {}
    for update in sorted(CONTROL_SURVIVAL):
        values = treatment[update]
        if len(values) != len(PHASE_CAPS):
            raise ValueError("each survival vector must contain five phases")
        row: list[int] = []
        for value, cap in zip(values, PHASE_CAPS, strict=True):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError("survival values must be integers")
            integer = int(value)
            if integer < 0 or integer > cap:
                raise ValueError("survival value is outside its phase suffix")
            row.append(integer)
        validated[update] = tuple(row)
    return validated


def classify_history_ablation(
    treatment: Mapping[int, Sequence[int]],
) -> str:
    """Classify the bounded treatment against matched E023 checkpoints."""
    rows = _validated_survival(treatment)
    deltas = {
        update: tuple(
            candidate - control
            for candidate, control in zip(
                rows[update], CONTROL_SURVIVAL[update], strict=True
            )
        )
        for update in rows
    }
    if any(
        all(delta >= 0 for delta in row)
        and any(delta > 0 for delta in row[:4])
        for row in deltas.values()
    ):
        return "one-frame-early-advances"
    if any(all(abs(delta) <= 2 for delta in row) for row in deltas.values()):
        return "one-frame-early-parity"
    if any(
        any(delta > 2 for delta in row) and any(delta < -2 for delta in row)
        for row in deltas.values()
    ):
        return "one-frame-early-mixed"
    return "one-frame-early-underperforms"


def select_history_checkpoint(
    treatment: Mapping[int, Sequence[int]],
) -> int:
    """Select by replay-free first-four-phase min/median/mean, earliest tie."""
    rows = _validated_survival(treatment)

    def key(update: int) -> tuple[float, float, float, int]:
        values = rows[update][:4]
        return min(values), median(values), sum(values) / len(values), -update

    return max(rows, key=key)


def validate_preflight(
    *, repository: Path, reference_path: Path, code_commit: str
) -> dict[str, Any]:
    """Bind E023 provenance and the single one-frame scientific delta."""
    repository = repository.resolve()
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(code_commit) != 40 or code_commit != head:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    reference_path = reference_path.resolve()
    assets = (
        (reference_path, EXPECTED_REFERENCE_SHA256, "reference"),
        (EXPECTED_MODEL_PATH, EXPECTED_MODEL_SHA256, "model"),
        (EXPECTED_CONTROLLER_PATH, EXPECTED_CONTROLLER_SHA256, "controller"),
    )
    for path, expected_sha256, label in assets:
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"runtime {label} SHA-256 does not match")
    return {
        "protocol": "g1-one-frame-rmr-noise-h24-walk-preflight-v1",
        "valid": True,
        "code_commit": head,
        "reference_path": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "model_path": str(EXPECTED_MODEL_PATH.resolve()),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "controller_path": str(EXPECTED_CONTROLLER_PATH.resolve()),
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
        "solver_profile": "g1-4x5",
        "rmr_walk_model_999_sha256": RMR_WALK_MODEL_999_SHA256,
        "scientific_delta": ["actor_history_len"],
        "actor_history_len": ACTOR_HISTORY_LEN,
        "actor_frame_obs_dim": ACTOR_FRAME_OBS_DIM,
        "actor_input_dim": ACTOR_HISTORY_LEN * ACTOR_FRAME_OBS_DIM,
        "total_updates": TOTAL_UPDATES,
        "total_steps": TOTAL_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "action_noise_schedule_steps": E023_TOTAL_STEPS,
    }


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_one_frame_rmr_noise_h24_walk_runs"),
    )
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    from src.algorithms.shac.algorithm import train
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )
    from tools.run_g1_tracking_shac import configure_jax

    configure_jax()
    kwargs = build_one_frame_kwargs(
        args.solver_profile, args.reference_path.resolve(), args.seed
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory,
        expected_kwargs=kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=TOTAL_STEPS,
        protocol="g1-one-frame-rmr-noise-h24-walk-training-v1",
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
