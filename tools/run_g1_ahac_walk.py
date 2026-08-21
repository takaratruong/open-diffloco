"""Train raw-reference G1 walking with the matched AHAC optimizer family."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.calibrate_g1_ahac_contact_threshold import (
    validate_calibration_payload,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_fresh_full_action_h24_walk import (
    CHECKPOINT_INTERVAL,
    TOTAL_STEPS,
    expected_checkpoint_steps,
)
from tools.run_g1_fresh_ppo_action_contract_walk import (
    validate_training_artifacts as validate_parent_training_artifacts,
)
from tools.run_g1_rmr_noise_h24_walk import (
    build_rmr_noise_h24_kwargs,
    validate_preflight as validate_rmr_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


E023_CHECKPOINT_SHA256 = (
    "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f"
)
AHAC_CHANGED_FIELDS = (
    "ahac",
    "ahac_horizon_min",
    "ahac_horizon_max",
    "ahac_contact_threshold",
    "ahac_dual_lr",
    "ahac_critic_max_iterations",
    "ahac_critic_tolerance",
    "actor_bootstrap_scale",
)


def build_ahac_walk_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    *,
    contact_threshold: float,
) -> dict[str, Any]:
    """Change only the registered, coupled AHAC optimizer-family fields."""
    if not math.isfinite(contact_threshold) or contact_threshold <= 0.0:
        raise ValueError("AHAC contact threshold must be positive and finite")
    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        ahac=True,
        ahac_horizon_min=8,
        ahac_horizon_max=24,
        ahac_contact_threshold=float(contact_threshold),
        ahac_dual_lr=5e-4,
        ahac_critic_max_iterations=64,
        ahac_critic_tolerance=0.2,
        actor_bootstrap_scale=1.0,
    )
    return kwargs


def validate_ahac_telemetry_row(
    row: dict[str, object], *, threshold: float
) -> None:
    """Require complete finite bounded AHAC telemetry at one checkpoint."""
    scalar_names = (
        "ahac_horizon",
        "ahac_horizon_before_update",
        "ahac_dual_mean",
        "ahac_dual_max",
        "ahac_contact_stiffness_mean",
        "ahac_contact_stiffness_max",
        "ahac_contact_threshold",
        "ahac_critic_head_disagreement",
    )
    try:
        scalars = {name: float(row[name]) for name in scalar_names}
        active = int(row["ahac_active_transitions"])
        iterations = int(row["ahac_critic_iterations"])
        losses = np.asarray(row["ahac_critic_loss_history"], dtype=np.float64)
        head_losses = np.asarray(row["ahac_critic_head_losses"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("AHAC telemetry is incomplete") from error
    if (
        row.get("ahac_valid") is not True
        or row.get("ahac_horizon_valid") is not True
        or not all(math.isfinite(value) for value in scalars.values())
        or not 8.0 <= scalars["ahac_horizon"] <= 24.0
        or not 8.0 <= scalars["ahac_horizon_before_update"] <= 24.0
        or not 8 <= active <= 24
        or scalars["ahac_dual_mean"] < 0.0
        or scalars["ahac_dual_max"] < scalars["ahac_dual_mean"]
        or scalars["ahac_contact_stiffness_mean"] < 0.0
        or scalars["ahac_contact_stiffness_max"]
        < scalars["ahac_contact_stiffness_mean"]
        or scalars["ahac_contact_threshold"] != threshold
        or not 1 <= iterations <= 64
        or losses.shape != (5,)
        or head_losses.shape != (2,)
        or not np.isfinite(losses).all()
        or not np.isfinite(head_losses).all()
        or scalars["ahac_critic_head_disagreement"] < 0.0
        or not isinstance(row.get("ahac_critic_converged"), bool)
    ):
        raise ValueError("AHAC telemetry is invalid")


def _validate_double_critic_checkpoint(
    path: Path, *, threshold: float
) -> None:
    with path.open("rb") as stream:
        state = pickle.load(stream)
    horizon = float(np.asarray(state.ahac_horizon))
    dual = np.asarray(state.ahac_dual, dtype=np.float64)
    params = state.critic_params["params"]
    if (
        not 8.0 <= horizon <= 24.0
        or dual.shape != (24,)
        or not np.isfinite(dual).all()
        or np.any(dual < 0.0)
        or set(params) != {"critic_0", "critic_1"}
        or not math.isfinite(threshold)
    ):
        raise ValueError(f"checkpoint {path.name} AHAC state is invalid")


def validate_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: dict[str, Any],
    threshold: float,
) -> dict[str, object]:
    """Validate the parent training contract plus every AHAC-specific gate."""
    base = validate_parent_training_artifacts(
        run_directory,
        expected_kwargs=expected_kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=TOTAL_STEPS,
        protocol="g1-ahac-walk-training-v1",
        expected_actor_bootstrap_scale=1.0,
    )
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    if hparams.get("algorithm") != "ahac":
        raise ValueError("training did not persist the AHAC algorithm identity")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    for row in rows:
        validate_ahac_telemetry_row(row, threshold=threshold)
    for step in expected_checkpoint_steps():
        _validate_double_critic_checkpoint(
            run_directory / f"checkpoint_step_{step:06d}.pkl",
            threshold=threshold,
        )
    return {**base, "contact_threshold": threshold, "ahac_valid": True}


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    calibration_path: Path,
    code_commit: str,
) -> dict[str, object]:
    """Bind source assets, current code, and the immutable calibration."""
    base = validate_rmr_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    calibration_path = calibration_path.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold = validate_calibration_payload(calibration)
    provenance = calibration["provenance"]
    expected = {
        "checkpoint_sha256": E023_CHECKPOINT_SHA256,
        "reference_sha256": base["reference_sha256"],
        "model_sha256": base["model_sha256"],
        "controller_sha256": base["controller_sha256"],
        "code_commit": code_commit,
    }
    if provenance != expected:
        raise ValueError("calibration provenance does not match this run")
    return {
        **base,
        "protocol": "g1-ahac-walk-preflight-v1",
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "contact_threshold": threshold,
        "source_checkpoint_sha256": E023_CHECKPOINT_SHA256,
        "scientific_delta": list(AHAC_CHANGED_FIELDS),
        "total_updates": 128,
        "total_steps": TOTAL_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("g1_ahac_walk_runs")
    )
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("the registered AHAC treatment uses seed zero only")
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path.resolve(),
        calibration_path=args.calibration.resolve(),
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_ahac_walk_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        contact_threshold=float(preflight["contact_threshold"]),
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
        threshold=float(preflight["contact_threshold"]),
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
