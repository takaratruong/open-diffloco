"""Localize the retained G1 adjoint failure at one materialized action boundary.

This diagnostic evaluates the ordinary H1 policy reverse gradient, a direct
reverse gradient of the exact one-step objective with respect to its 29-D
action, all 29 coordinate forward derivatives, and one central finite
difference direction.  It computes no retained optimizer update or policy.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import jax

from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
    _validate_population_report,
    build_probe_kwargs as build_source_probe_kwargs,
)
from experiments.g1_ahac_first_proposal_audit.run import (
    START_STEP,
    validate_preflight,
)
from experiments.g1_ahac_reverse_forward_discriminator.run import (
    validate_source_e017,
)
from src.algorithms.shac.algorithm import (
    H1_ACTION_DIRECTION_SEED,
    H1_ACTION_FINITE_DIFFERENCE_EPSILON,
    train,
)
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


POPULATION_SIZE = 512
ACTION_DIMENSION = 29
H1_END_STEP = START_STEP + POPULATION_SIZE


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant {value} in {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def build_probe_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    probe_output: str | Path,
) -> dict[str, Any]:
    """Build one H1 action-localization probe from the E017 scale-zero arm."""

    kwargs = build_source_probe_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        arm="scale_zero",
        probe_output=probe_output,
    )
    kwargs.update(
        total_steps=H1_END_STEP,
        checkpoint_steps=(H1_END_STEP,),
        unroll_length=1,
        ahac_horizon_min=1,
        ahac_horizon_max=1,
        actor_bootstrap_graph_mode="excised",
        actor_forward_jvp_probe=False,
        actor_h1_action_derivative_probe=True,
        determinism_probe_output=str(Path(probe_output).resolve()),
    )
    return kwargs


def classify_action_derivatives(report: Mapping[str, object]) -> dict[str, object]:
    """Fail closed on primal drift, then classify the direct action slopes."""

    population = report.get("population_size")
    action_dimension = report.get("action_dimension")
    count_names = (
        "policy_reverse_finite_count",
        "action_reverse_finite_count",
        "action_forward_finite_count",
        "forward_recovers_action_reverse_count",
        "forward_recovers_policy_reverse_count",
        "jointly_finite_count",
        "jointly_finite_gradient_agreement_count",
        "finite_difference_finite_count",
        "finite_difference_agreement_count",
    )
    counts = {name: report.get(name) for name in count_names}
    if (
        report.get("protocol") != "shac-h1-action-derivative-pair-v1"
        or population != POPULATION_SIZE
        or action_dimension != ACTION_DIMENSION
        or any(
            type(value) is not int or not 0 <= value <= POPULATION_SIZE
            for value in counts.values()
        )
    ):
        raise ValueError("H1 action derivative report header is invalid")

    primal_valid = bool(
        report.get("valid") is True
        and report.get("source_primal_matches_action_reverse") is True
        and report.get("source_primal_matches_action_forward") is True
        and report.get("repeat_exact") is True
        and report.get("metadata_exact") is True
    )
    if not primal_valid:
        outcome = "invalid-source-primal"
        interpretable = False
    elif (
        counts["jointly_finite_gradient_agreement_count"]
        != counts["jointly_finite_count"]
    ):
        outcome = "reverse-forward-action-gradient-disagreement"
        interpretable = False
    elif (
        counts["finite_difference_finite_count"] > 0
        and counts["finite_difference_agreement_count"]
        / counts["finite_difference_finite_count"]
        < 0.9
    ):
        outcome = "forward-action-fails-finite-difference-gate"
        interpretable = False
    elif report.get("action_reverse_mask_matches_policy_reverse") is not True:
        outcome = "direct-action-does-not-localize-policy-failure"
        interpretable = True
    elif (
        counts["action_reverse_finite_count"] < POPULATION_SIZE
        and counts["action_forward_finite_count"] == POPULATION_SIZE
        and counts["forward_recovers_action_reverse_count"]
        == POPULATION_SIZE - counts["action_reverse_finite_count"]
    ):
        outcome = "forward-action-bypasses-reverse-transpose"
        interpretable = True
    elif (
        counts["action_forward_finite_count"] == counts["action_reverse_finite_count"]
        and counts["forward_recovers_action_reverse_count"] == 0
    ):
        outcome = "forward-action-shares-reverse-failure"
        interpretable = True
    elif counts["action_forward_finite_count"] > counts["action_reverse_finite_count"]:
        outcome = "forward-action-partially-bypasses-reverse-transpose"
        interpretable = True
    else:
        outcome = "forward-action-does-not-improve-finiteness"
        interpretable = True

    return {
        "protocol": "g1-e002-ahac-h1-action-derivative-classification-v1",
        "valid": primal_valid,
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "population_size": population,
        "action_dimension": action_dimension,
        **counts,
        "action_reverse_mask_matches_policy_reverse": report.get(
            "action_reverse_mask_matches_policy_reverse"
        ),
        "source_primal_matches_action_reverse": report.get(
            "source_primal_matches_action_reverse"
        ),
        "source_primal_matches_action_forward": report.get(
            "source_primal_matches_action_forward"
        ),
        "finite_difference_agreement_fraction": (
            counts["finite_difference_agreement_count"]
            / counts["finite_difference_finite_count"]
            if counts["finite_difference_finite_count"]
            else None
        ),
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": None,
    }


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "run_json_path": args.source_e017_run_json.resolve(),
        "audit_path": args.source_e017_audit.resolve(),
        "probe_path": args.source_e017_probe.resolve(),
        "arm_path": args.source_e017_arm.resolve(),
        "comparison_path": args.source_e017_comparison.resolve(),
        "completion_path": args.source_e017_completion.resolve(),
        "receipt_path": args.source_e017_receipt.resolve(),
    }


def _validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        resume_audit=args.resume_audit.resolve(),
        ahac_fidelity_audit=args.ahac_fidelity_audit.resolve(),
        critic_calibration_audit=args.critic_calibration_audit.resolve(),
        code_commit=args.code_commit,
    )
    source, _, _ = validate_source_e017(**_source_paths(args))
    preflight.update(
        protocol="g1-e002-ahac-h1-action-derivative-preflight-v1",
        source_e017=source,
        source_step=START_STEP,
        computed_output_step=H1_END_STEP,
        horizon=1,
        population_size=POPULATION_SIZE,
        action_dimension=ACTION_DIMENSION,
        derivative_modes=["reverse", "coordinate-forward", "central-fd"],
        exact_source_primal_required=True,
        deterministic_invocations=2,
        ordinary_training_loop_entered=False,
        optimizer_update_retained=False,
        policy_retained=False,
    )
    return preflight


def _validate_run_artifacts(
    run_directory: Path, *, probe_output: Path
) -> dict[str, object]:
    hparams_path = run_directory / "hparams.json"
    migration_path = run_directory / "ahac_resume_migration.json"
    hparams = _read_json(hparams_path)
    migration = _read_json(migration_path)
    expected = {
        "algorithm": "ahac",
        "total_steps": H1_END_STEP,
        "steps_per_actor_update": POPULATION_SIZE,
        "unroll_length": 1,
        "effective_num_envs": POPULATION_SIZE,
        "gradient_accumulation_steps": 2,
        "ahac": True,
        "ahac_horizon_min": 1,
        "ahac_horizon_max": 1,
        "ahac_semantics": "paper_equation_10_no_target",
        "actor_bootstrap_scale": 0.0,
        "actor_bootstrap_delay_steps": 0,
        "actor_bootstrap_graph_mode": "excised",
        "actor_forward_jvp_probe": False,
        "actor_h1_action_derivative_probe": True,
        "actor_h1_action_direction_seed": H1_ACTION_DIRECTION_SEED,
        "actor_h1_action_finite_difference_epsilon": (
            H1_ACTION_FINITE_DIFFERENCE_EPSILON
        ),
        "actor_inactive_horizon_gradient_mode": "connected",
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "actor_per_env_grad_clip": 1.0,
        "determinism_probe": True,
        "determinism_probe_output": str(probe_output.resolve()),
    }
    if any(hparams.get(name) != value for name, value in expected.items()):
        raise ValueError("H1 action derivative hparams are invalid")
    if (
        migration.get("valid") is not True
        or migration.get("source_step") != START_STEP
        or migration.get("actor_and_actor_optimizer_preserved") is not True
        or migration.get("critic_optimizer_reset") is not True
        or migration.get("horizon") != 1.0
        or migration.get("dual_all_zero") is not True
    ):
        raise ValueError("H1 action derivative AHAC migration is invalid")
    report = _read_json(probe_output)
    population = report.get("cagrad_population")
    action_report = report.get("actor_h1_action_derivatives")
    if not isinstance(population, Mapping) or not isinstance(action_report, Mapping):
        raise TypeError("H1 action derivative population report is missing")
    _validate_population_report(
        population,
        expected_scale=0.0,
        expected_input_step=START_STEP,
        expected_output_step=H1_END_STEP,
    )
    if list(run_directory.rglob("*.pkl")):
        raise ValueError("H1 action derivative probe wrote a checkpoint")
    return {
        "run_directory": str(run_directory.resolve()),
        "hparams_sha256": sha256_file(hparams_path),
        "migration_sha256": sha256_file(migration_path),
        "probe_sha256": sha256_file(probe_output),
        "action_report": action_report,
        "policy_retained": False,
    }


def _run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        raise FileExistsError("H1 action derivative preflight already exists")
    preflight = _validate_preflight(args)
    _write_json_atomically(preflight_path, preflight)

    probe_output = output_root / "h1_action_derivative_probe.json"
    kwargs = build_probe_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        probe_output=probe_output,
    )
    training_root = output_root / "h1-action-derivative"
    training_root.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(training_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_run_directory = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (training_root / relative_run_directory).resolve()
    manifest = _validate_run_artifacts(run_directory, probe_output=probe_output)
    action_report = manifest.pop("action_report")
    manifest.update(
        protocol="g1-e002-ahac-h1-action-derivative-run-v1",
        valid=True,
        optimizer_update_retained=False,
        persisted_optimizer_updates=0,
        policy_retained=False,
    )
    manifest_path = output_root / "run_manifest.json"
    _write_json_atomically(manifest_path, manifest)

    classification = classify_action_derivatives(action_report)
    classification.update(
        code_commit=args.code_commit,
        source_e017=preflight["source_e017"],
        run_manifest=manifest,
    )
    classification_path = output_root / "classification.json"
    _write_json_atomically(classification_path, classification)
    completion = {
        "protocol": "g1-e002-ahac-h1-action-derivative-completion-v1",
        "valid": classification["scientifically_interpretable"],
        "outcome": classification["outcome"],
        "computed_train_step_invocations": 2,
        "persisted_optimizer_updates": 0,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "h1_action_derivative_probe.json": sha256_file(probe_output),
            "run_manifest.json": sha256_file(manifest_path),
            "classification.json": sha256_file(classification_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(classification, indent=2, sort_keys=True))
    jax.clear_caches()
    return 0 if completion["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--resume-audit", type=Path, required=True)
    parser.add_argument("--ahac-fidelity-audit", type=Path, required=True)
    parser.add_argument("--critic-calibration-audit", type=Path, required=True)
    parser.add_argument("--source-e017-run-json", type=Path, required=True)
    parser.add_argument("--source-e017-audit", type=Path, required=True)
    parser.add_argument("--source-e017-probe", type=Path, required=True)
    parser.add_argument("--source-e017-arm", type=Path, required=True)
    parser.add_argument("--source-e017-comparison", type=Path, required=True)
    parser.add_argument("--source-e017-completion", type=Path, required=True)
    parser.add_argument("--source-e017-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
