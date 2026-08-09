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

# These are the run hparams that define the audit's unmodified stochastic
# objective.  The checkpoint and reference identifiers are checked separately
# because they are supplied as immutable CLI inputs.
FROZEN_HPARAMS = {
    "algorithm": "shac",
    "env_variant": "g1_tracking_rmr_50hz_validated",
    "unroll_length": 48,
    "num_envs": 64,
    "gamma": 0.99,
    "action_noise_std_start": 0.1,
    "action_noise_std_end": 0.1,
    "actor_per_env_grad_clip": 1.0,
    "actor_bootstrap_scale": 0.0,
    "squash_actor_actions": False,
    "friction_range": [1.0, 1.0],
    "mass_range": [1.0, 1.0],
    "com_offset_range": [0.0, 0.0, 0.0],
    "push_velocity_range": [0.0, 0.0],
    "terrain": False,
    "reference_reset_noise_scale": 0.0,
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
    reference_sha256: str,
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

    for name, expected in FROZEN_HPARAMS.items():
        if name not in hparams:
            raise ValueError(f"frozen hparams are missing {name}")
        _require_exact(f"frozen hparams {name}", hparams[name], expected)

    if hparams.get("reference_sha256") != reference_sha256:
        raise ValueError("frozen hparams reference_sha256 does not match --reference")
    recorded_reference = hparams.get("reference_path")
    if not isinstance(recorded_reference, str):
        raise ValueError("frozen hparams are missing reference_path")
    if Path(recorded_reference).resolve() != reference:
        raise ValueError("frozen hparams reference_path does not match --reference")
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
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match --checkpoint")
    if sha256_file(reference) != reference_sha256:
        raise ValueError("reference SHA-256 does not match --reference")

    for name, expected in FROZEN_ARGUMENTS.items():
        _require_exact(name.replace("_", " "), getattr(args, name), expected)
    _require_exact("shard seeds", tuple(args.shard_seeds), FIXED_SHARD_SEEDS)
    _require_exact(
        "held-out seeds", tuple(args.held_out_seeds), FIXED_HELD_OUT_SEEDS
    )
    _require_exact("phases", tuple(args.phases), FIXED_PHASES)

    hparams_path = _read_frozen_hparams(
        checkpoint, reference, reference_sha256
    )
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
