"""Fail-closed CLI contract for the frozen G1 SHAC gradient-quality audit.

This module deliberately owns only immutable-input validation and durable
evidence writes.  The expensive simulator and gradient implementation remains
an explicit dependency of :func:`main` until the audit engine is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FIXED_SHARD_SEEDS = (0, 1, 2, 3)
FIXED_HELD_OUT_SEEDS = (4, 5, 6, 7)
FIXED_PHASES = (0, 100, 200, 300, 400)
E064_CHECKPOINT_SHA256 = (
    "6b5c6bb208f9acd9f5988fee201915f8aa67cba42c15231d361a4d2ae530a094"
)
E064_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)

FROZEN_ARGUMENTS = {
    "horizon": 48,
    "population": 64,
    "sigma": 0.1,
    "gamma": 0.99,
    "per_env_clip": 1.0,
    "functional_rms": 0.01,
    "solver_iterations": 4,
    "solver_ls_iterations": 5,
}

# Exact E064 final-checkpoint hparams.  reference_path is replaced at
# validation time by the resolved selected path, while its key and hash remain
# mandatory parts of this exact 62-key document.
FROZEN_E064_HPARAMS = {
    "algorithm": "shac",
    "total_steps": 393216,
    "unroll_length": 48,
    "num_envs": 64,
    "gradient_accumulation_steps": 1,
    "effective_num_envs": 64,
    "steps_per_actor_update": 3072,
    "actor_lr": 0.001,
    "critic_lr": 0.0005,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "target_update_rate": 0.01,
    "critic_iterations": 16,
    "xml_path": (
        "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
        "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
    ),
    "action_scale": 1.0,
    "cmd_vel_x_range": [-2.0, 2.0],
    "cmd_vel_y_range": [-1.0, 1.0],
    "cmd_yaw_rate_range": [-1.5, 1.5],
    "cmd_zero_prob": [0.1, 0.7, 0.5],
    "cmd_ctrl_interval_range": [60, 140],
    "action_noise_std_start": 1.0,
    "action_noise_std_end": 0.1,
    "friction_range": [1.0, 1.0],
    "mass_range": [1.0, 1.0],
    "effort_limit_scale": 1.0,
    "termination_margin_weight": 0.0,
    "kp_range": [1.0, 1.0],
    "kd_range": [1.0, 1.0],
    "com_offset_range": [0.0, 0.0, 0.0],
    "push_velocity_range": [0.0, 0.0],
    "push_interval_s": 1000000000.0,
    "terrain_flat_prob": 0.2,
    "terrain_slope_max": 5.0,
    "terrain_bump_std": 0.4,
    "terrain_bump_decay": 0.4,
    "terrain": False,
    "zero_difficulty_frac": 1.0,
    "curriculum_grace": 393216,
    "curriculum_steps": 1,
    "seed": 1,
    "best_reward": 0.07537891473167549,
    "max_episode_length": 499,
    "actor_history_len": 1,
    "actor_per_env_grad_clip": 1.0,
    "critic_per_env_grad_clip": 1.0,
    "actor_bootstrap_scale": 0.0,
    "actor_bootstrap_delay_steps": 0,
    "actor_hidden": [512, 512],
    "actor_layer_norm": False,
    "actor_zero_output": False,
    "source_actor_policy": False,
    "actor_kind": "flax",
    "residual_action_scale": 0.0,
    "differentiate_source_feedback": True,
    "env_variant": "g1_tracking_rmr_50hz_validated",
    "squash_actor_actions": False,
    "reference_path": (
        "/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/"
        "artifacts/E-20260808-000/reference/"
        "dance1_subject2_f122_422_50hz.npz"
    ),
    "reference_sha256": E064_REFERENCE_SHA256,
    "reference_fps": 50.0,
    "reference_stride": 1,
    "reference_states": 500,
    "reference_transitions": 499,
}


@dataclass(frozen=True)
class AuditContract:
    """Validated immutable inputs consumed by the future audit engine."""

    checkpoint: Path
    checkpoint_sha256: str
    reference: Path
    reference_sha256: str
    hparams_path: Path
    output_dir: Path
    shard_seeds: tuple[int, int, int, int]
    held_out_seeds: tuple[int, int, int, int]
    phases: tuple[int, int, int, int, int]
    horizon: int
    population: int
    sigma: float
    gamma: float
    per_env_clip: float
    functional_rms: float
    solver_iterations: int
    solver_ls_iterations: int


def build_parser() -> argparse.ArgumentParser:
    """Build the audit CLI with the preregistered frozen defaults."""
    parser = argparse.ArgumentParser(
        description="Run the frozen G1 SHAC gradient-quality audit."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-seeds", type=int, nargs=4, default=FIXED_SHARD_SEEDS
    )
    parser.add_argument(
        "--held-out-seeds", type=int, nargs=4, default=FIXED_HELD_OUT_SEEDS
    )
    parser.add_argument("--phases", type=int, nargs=5, default=FIXED_PHASES)
    parser.add_argument("--horizon", type=int, default=FROZEN_ARGUMENTS["horizon"])
    parser.add_argument(
        "--population", type=int, default=FROZEN_ARGUMENTS["population"]
    )
    parser.add_argument("--sigma", type=float, default=FROZEN_ARGUMENTS["sigma"])
    parser.add_argument("--gamma", type=float, default=FROZEN_ARGUMENTS["gamma"])
    parser.add_argument(
        "--per-env-clip", type=float, default=FROZEN_ARGUMENTS["per_env_clip"]
    )
    parser.add_argument(
        "--functional-rms",
        type=float,
        default=FROZEN_ARGUMENTS["functional_rms"],
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=FROZEN_ARGUMENTS["solver_iterations"],
    )
    parser.add_argument(
        "--solver-ls-iterations",
        type=int,
        default=FROZEN_ARGUMENTS["solver_ls_iterations"],
    )
    return parser


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest without loading the whole artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _require_exact(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be frozen at {expected!r}, got {actual!r}")


def _read_frozen_hparams(
    checkpoint: Path,
    reference: Path,
) -> Path:
    hparams_path = checkpoint.parent / "hparams.json"
    if not hparams_path.is_file():
        raise ValueError(f"frozen hparams file does not exist: {hparams_path}")
    try:
        hparams = json.loads(hparams_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"frozen hparams are not valid JSON: {hparams_path}") from error
    if not isinstance(hparams, Mapping):
        raise ValueError("frozen hparams must be a JSON object")

    expected_hparams = dict(FROZEN_E064_HPARAMS)
    expected_hparams["reference_path"] = str(reference)
    missing = sorted(set(expected_hparams) - set(hparams))
    extra = sorted(set(hparams) - set(expected_hparams))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"extra keys: {', '.join(extra)}")
        raise ValueError(f"frozen hparams key set mismatch ({'; '.join(details)})")

    for name, expected in expected_hparams.items():
        _require_exact(f"frozen hparams {name}", hparams[name], expected)
    return hparams_path.resolve()


def validate_audit_contract(args: argparse.Namespace) -> AuditContract:
    """Fail closed unless every input matches the preregistered protocol."""
    checkpoint = args.checkpoint.resolve()
    reference = args.reference.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    if not reference.is_file():
        raise ValueError(f"reference does not exist: {reference}")

    checkpoint_sha256 = _require_sha256(
        args.checkpoint_sha256, "checkpoint SHA-256"
    )
    reference_sha256 = _require_sha256(
        args.reference_sha256, "reference SHA-256"
    )
    _require_exact(
        "checkpoint SHA-256", checkpoint_sha256, E064_CHECKPOINT_SHA256
    )
    _require_exact("reference SHA-256", reference_sha256, E064_REFERENCE_SHA256)
    _require_exact(
        "checkpoint file SHA-256", sha256_file(checkpoint), E064_CHECKPOINT_SHA256
    )
    _require_exact(
        "reference file SHA-256", sha256_file(reference), E064_REFERENCE_SHA256
    )

    for name, expected in FROZEN_ARGUMENTS.items():
        _require_exact(name.replace("_", " "), getattr(args, name), expected)
    _require_exact("shard seeds", tuple(args.shard_seeds), FIXED_SHARD_SEEDS)
    _require_exact(
        "held-out seeds", tuple(args.held_out_seeds), FIXED_HELD_OUT_SEEDS
    )
    _require_exact("phases", tuple(args.phases), FIXED_PHASES)

    hparams_path = _read_frozen_hparams(checkpoint, reference)
    return AuditContract(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        reference=reference,
        reference_sha256=reference_sha256,
        hparams_path=hparams_path,
        output_dir=args.output_dir.resolve(),
        shard_seeds=tuple(args.shard_seeds),
        held_out_seeds=tuple(args.held_out_seeds),
        phases=tuple(args.phases),
        horizon=args.horizon,
        population=args.population,
        sigma=args.sigma,
        gamma=args.gamma,
        per_env_clip=args.per_env_clip,
        functional_rms=args.functional_rms,
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
    )


def assert_finite_json(value: Any, path: str = "document") -> None:
    """Reject a JSON-shaped evidence document containing NaN or infinity."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")


def _replace_atomically(path: Path, write: Callable[[Any], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomically(path: Path, document: Mapping[str, Any]) -> None:
    """Validate finite JSON before atomically replacing an evidence document."""
    assert_finite_json(document)
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _replace_atomically(path, lambda stream: stream.write(encoded))


def write_pickle_atomically(path: Path, value: Any) -> None:
    """Atomically materialize a candidate checkpoint or audit artifact."""
    _replace_atomically(
        path, lambda stream: pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    )


def _load_future_run_audit() -> Callable[..., Any]:
    """Import the engine lazily so parser/contract checks stay lightweight."""
    try:
        from src.algorithms.shac.gradient_audit import run_audit
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "the gradient-quality execution engine is not implemented yet"
        ) from error
    return run_audit


def main(
    argv: Sequence[str] | None = None,
    *,
    run_audit_impl: Callable[..., Any] | None = None,
) -> Any:
    """Validate the immutable contract and delegate to the future audit engine."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        contract = validate_audit_contract(args)
    except ValueError as error:
        parser.error(str(error))
    implementation = run_audit_impl or _load_future_run_audit()
    return implementation(contract=contract)


if __name__ == "__main__":
    main()
