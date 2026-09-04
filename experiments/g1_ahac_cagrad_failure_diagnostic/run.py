"""Diagnose the retained-E002 first-AHAC-proposal CAGrad rejection.

The run intentionally computes the same single proposal that failed E014,
requires the trainer to reject it, validates the create-only failure report,
and retains no computed training state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.g1_ahac_first_proposal_audit.run import (
    END_STEP,
    START_STEP,
    build_proposal_kwargs,
    validate_preflight,
)
from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


SOURCE_RUN_JSON_SHA256 = (
    "6c1307d073bb83acb8b04c20b53013b22a6868e558ef2f89bd1dc493cabb6835"
)
SOURCE_STDERR_SHA256 = (
    "4a9c623df2d990356c935f8b9d705f470437bcb18a7ffdb64f7e04c5add4500b"
)
EXPECTED_ERROR = "actor CAGrad aggregation is invalid"
FAILURE_CLASSES = {
    "empty-loss-phase-bin",
    "nonfinite-actor-losses",
    "empty-gradient-phase-bin",
    "nonfinite-per-environment-gradients",
    "nonfinite-bin-gradient-norms",
    "cagrad-solver-invalid",
    "cagrad-reduction-inconsistent",
    "cagrad-validity-inconsistent",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _nonnegative_integer_vector(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty integer vector")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ValueError(f"{name} must contain nonnegative integers")
    return value


def _contains_nonfinite(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_nonfinite(item) for item in value)
    return isinstance(value, float) and not math.isfinite(value)


def _infer_failure_class(
    subgates: Mapping[str, object], gradient_finite_fraction: float
) -> str:
    if not bool(subgates.get("loss_bins_occupied")):
        return "empty-loss-phase-bin"
    if not bool(subgates.get("losses_finite")) or not bool(
        subgates.get("loss_valid")
    ):
        return "nonfinite-actor-losses"
    if not bool(subgates.get("gradient_bins_occupied")):
        return "empty-gradient-phase-bin"
    if not bool(subgates.get("counts_match")) or gradient_finite_fraction < 1.0:
        return "nonfinite-per-environment-gradients"
    if not bool(subgates.get("gradient_norms_finite")):
        return "nonfinite-bin-gradient-norms"
    if not bool(subgates.get("solver_valid")):
        return "cagrad-solver-invalid"
    if not bool(subgates.get("reduction_valid")):
        return "cagrad-reduction-inconsistent"
    return "cagrad-validity-inconsistent"


def validate_failure_artifact(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and classify one report without accepting candidate state."""

    if (
        report.get("protocol") != "shac-cagrad-failure-v1"
        or report.get("report_valid") is not True
        or report.get("actor_cagrad_valid") is not False
    ):
        raise ValueError("CAGrad failure report header is invalid")
    if report.get("computed_candidate_state_persisted") is not False:
        raise ValueError("CAGrad diagnostic violates the no-retention gate")
    if (
        report.get("input_step") != START_STEP
        or report.get("computed_output_step") != END_STEP
    ):
        raise ValueError("CAGrad diagnostic step boundary is invalid")
    if _contains_nonfinite(report):
        raise ValueError("CAGrad diagnostic is not strict finite-or-null JSON")

    phase_bins = report.get("phase_bins")
    subgates = report.get("subgates")
    actor_gradient = report.get("actor_gradient")
    population = report.get("population")
    ahac = report.get("ahac")
    if not all(
        isinstance(item, Mapping)
        for item in (phase_bins, subgates, actor_gradient, population, ahac)
    ):
        raise ValueError("CAGrad diagnostic sections are incomplete")
    subgate_names = (
        "loss_bins_occupied",
        "losses_finite",
        "loss_valid",
        "gradient_bins_occupied",
        "gradient_norms_finite",
        "counts_match",
        "solver_valid",
        "reduction_valid",
    )
    if any(type(subgates.get(name)) is not bool for name in subgate_names):
        raise ValueError("CAGrad diagnostic validity sub-gates are invalid")
    gradient_counts = _nonnegative_integer_vector(
        phase_bins.get("gradient_counts"), name="gradient counts"
    )
    loss_counts = _nonnegative_integer_vector(
        phase_bins.get("loss_counts"), name="loss counts"
    )
    missing = _nonnegative_integer_vector(
        phase_bins.get("missing_gradient_contributors"),
        name="missing gradient contributors",
    )
    if (
        len(gradient_counts) != 5
        or len(loss_counts) != 5
        or len(missing) != 5
        or sum(loss_counts) != 512
        or missing
        != [loss - gradient for loss, gradient in zip(loss_counts, gradient_counts)]
        or bool(subgates.get("counts_match")) != (gradient_counts == loss_counts)
    ):
        raise ValueError("CAGrad diagnostic contributor accounting is invalid")

    nonfinite_gradient_envs = population.get(
        "nonfinite_gradient_environments"
    )
    nonfinite_loss_envs = population.get("nonfinite_loss_environments")
    if not isinstance(nonfinite_gradient_envs, list) or not isinstance(
        nonfinite_loss_envs, list
    ):
        raise ValueError("CAGrad per-environment localization is invalid")
    gradient_finite_by_env = population.get("gradient_finite_by_env")
    loss_finite_by_env = population.get("loss_finite_by_env")
    start_phases = population.get("start_phases")
    start_support_modes = population.get("start_support_modes")
    terminal_modes = population.get("terminal_modes")
    losses_by_env = population.get("losses_by_env")
    full_vectors = (
        gradient_finite_by_env,
        loss_finite_by_env,
        start_phases,
        start_support_modes,
        terminal_modes,
        losses_by_env,
    )
    if any(not isinstance(vector, list) or len(vector) != 512 for vector in full_vectors):
        raise ValueError("CAGrad full population vectors are missing or incomplete")
    if (
        any(type(value) is not bool for value in gradient_finite_by_env)
        or any(type(value) is not bool for value in loss_finite_by_env)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 271
            for value in start_phases
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 3
            for value in (*start_support_modes, *terminal_modes)
        )
        or any(
            (finite and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))))
            or (not finite and value is not None)
            for finite, value in zip(loss_finite_by_env, losses_by_env, strict=True)
        )
    ):
        raise ValueError("CAGrad full population vectors are invalid")

    phase_bin_by_env = [min(phase * 5 // 271, 4) for phase in start_phases]
    reconstructed_loss_counts = [
        sum(bin_index == target for bin_index in phase_bin_by_env)
        for target in range(5)
    ]
    reconstructed_gradient_counts = [
        sum(
            bin_index == target and finite
            for bin_index, finite in zip(
                phase_bin_by_env, gradient_finite_by_env, strict=True
            )
        )
        for target in range(5)
    ]

    def population_row(index: int) -> dict[str, object]:
        return {
            "index": index,
            "phase": start_phases[index],
            "start_support_mode": start_support_modes[index],
            "terminal_mode": terminal_modes[index],
            "loss": losses_by_env[index],
            "loss_finite": loss_finite_by_env[index],
        }

    expected_nonfinite_gradient_envs = [
        population_row(index)
        for index, finite in enumerate(gradient_finite_by_env)
        if not finite
    ]
    expected_nonfinite_loss_envs = [
        population_row(index)
        for index, finite in enumerate(loss_finite_by_env)
        if not finite
    ]
    if (
        population.get("size") != 512
        or population.get("finite_gradient_count") != sum(gradient_counts)
        or population.get("finite_loss_count")
        != 512 - len(nonfinite_loss_envs)
        or sum(gradient_finite_by_env) != sum(gradient_counts)
        or sum(loss_finite_by_env) != population.get("finite_loss_count")
        or reconstructed_loss_counts != loss_counts
        or reconstructed_gradient_counts != gradient_counts
        or len(nonfinite_gradient_envs) != sum(missing)
        or nonfinite_gradient_envs != expected_nonfinite_gradient_envs
        or nonfinite_loss_envs != expected_nonfinite_loss_envs
        or any(
            not isinstance(row, Mapping)
            or isinstance(row.get("index"), bool)
            or not isinstance(row.get("index"), int)
            or not 0 <= row["index"] < 512
            for row in nonfinite_gradient_envs
        )
    ):
        raise ValueError("CAGrad per-environment localization is invalid")

    finite_fraction = actor_gradient.get("finite_fraction")
    if (
        isinstance(finite_fraction, bool)
        or not isinstance(finite_fraction, (int, float))
        or not math.isfinite(float(finite_fraction))
        or not 0.0 <= float(finite_fraction) <= 1.0
    ):
        raise ValueError("CAGrad finite-gradient fraction is invalid")
    inferred = _infer_failure_class(subgates, float(finite_fraction))
    outcome = report.get("failure_class")
    if outcome not in FAILURE_CLASSES or outcome != inferred:
        raise ValueError("CAGrad failure classification is inconsistent")
    if (
        ahac.get("actor_bootstrap_scale_current") != 1.0
        or ahac.get("ahac_horizon_before_update") != 8.0
        or ahac.get("ahac_actor_constraint_penalty") != 0.0
        or ahac.get("ahac_critic_head_disagreement") != 0.0
        or ahac.get("ahac_horizon_valid") is not True
    ):
        raise ValueError("CAGrad diagnostic does not match proposal-one AHAC state")
    return {
        "protocol": "g1-e002-ahac-cagrad-failure-validation-v1",
        "valid": True,
        "outcome": outcome,
        "gradient_bin_counts": gradient_counts,
        "loss_bin_counts": loss_counts,
        "missing_gradient_contributors": missing,
        "missing_gradient_contributor_count": sum(missing),
        "population_size": 512,
        "full_population_vectors_valid": True,
        "nonfinite_gradient_environments": nonfinite_gradient_envs,
        "nonfinite_loss_environments": nonfinite_loss_envs,
        "actor_gradient_finite_fraction": float(finite_fraction),
        "computed_optimizer_updates": 1,
        "persisted_optimizer_updates": 0,
        "policy_retained": False,
        "retained_policy": None,
    }


def validate_source_failure(run_json_path: Path, stderr_path: Path) -> dict[str, Any]:
    """Bind the exact E014 failure that authorized this diagnostic."""

    if (
        sha256_file(run_json_path) != SOURCE_RUN_JSON_SHA256
        or sha256_file(stderr_path) != SOURCE_STDERR_SHA256
    ):
        raise ValueError("source E014 failure artifact SHA-256 mismatch")
    run = _read_json(run_json_path)
    attempts = run.get("attempts")
    stderr = stderr_path.read_text(encoding="utf-8")
    if (
        run.get("experiment") != "E-20260903-014"
        or run.get("return_code") != 1
        or run.get("timed_out") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or attempts[0].get("return_code") != 1
        or EXPECTED_ERROR not in stderr
    ):
        raise ValueError("source E014 failure contract is invalid")
    return {
        "experiment": "E-20260903-014",
        "run_json": str(run_json_path.resolve()),
        "run_json_sha256": SOURCE_RUN_JSON_SHA256,
        "stderr": str(stderr_path.resolve()),
        "stderr_sha256": SOURCE_STDERR_SHA256,
        "return_code": 1,
        "timed_out": False,
        "failure": EXPECTED_ERROR,
    }


def run_diagnostic(args: argparse.Namespace) -> None:
    configure_jax()
    repository = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        raise ValueError("diagnostic output already contains preflight.json")

    source_failure = validate_source_failure(
        args.source_run_json.resolve(), args.source_stderr.resolve()
    )
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.checkpoint.resolve(),
        reference=args.reference_path.resolve(),
        resume_audit=args.resume_audit.resolve(),
        ahac_fidelity_audit=args.ahac_fidelity_audit.resolve(),
        critic_calibration_audit=args.critic_calibration_audit.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-e002-ahac-cagrad-failure-preflight-v1",
        diagnostic_only=True,
        expected_failure=EXPECTED_ERROR,
        source_failure=source_failure,
    )
    _write_json_atomically(preflight_path, preflight)

    kwargs = build_proposal_kwargs(
        "g1-4x5",
        args.reference_path.resolve(),
        args.seed,
        args.checkpoint.resolve(),
    )
    existing = set(output_root.glob("training_runs/ahac_*"))
    previous_directory = Path.cwd()
    observed_error: RuntimeError | None = None
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile("g1-4x5")):
            train(**kwargs)
    except RuntimeError as error:
        if str(error) != EXPECTED_ERROR:
            raise
        observed_error = error
    finally:
        os.chdir(previous_directory)
    if observed_error is None:
        raise RuntimeError("registered CAGrad failure did not reproduce")

    created = set(output_root.glob("training_runs/ahac_*")) - existing
    if len(created) != 1:
        raise ValueError("diagnostic must create exactly one AHAC run directory")
    run_directory = created.pop().resolve()
    report_path = run_directory / "cagrad_failure.json"
    report = _read_json(report_path)
    validation = validate_failure_artifact(report)
    state_artifacts = sorted(run_directory.glob("*.pkl"))
    if state_artifacts:
        raise ValueError("CAGrad diagnostic unexpectedly persisted training state")

    validation.update(
        run_directory=str(run_directory),
        failure_report=str(report_path),
        failure_report_sha256=sha256_file(report_path),
        hparams=str(run_directory / "hparams.json"),
        hparams_sha256=sha256_file(run_directory / "hparams.json"),
        migration=str(run_directory / "ahac_resume_migration.json"),
        migration_sha256=sha256_file(
            run_directory / "ahac_resume_migration.json"
        ),
        source_failure=source_failure,
    )
    diagnostic_path = output_root / "diagnostic.json"
    _write_json_atomically(diagnostic_path, validation)
    completion = {
        "protocol": "g1-e002-ahac-cagrad-failure-completion-v1",
        "valid": True,
        "outcome": validation["outcome"],
        "computed_optimizer_updates": 1,
        "persisted_optimizer_updates": 0,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "diagnostic.json": sha256_file(diagnostic_path),
            "cagrad_failure.json": validation["failure_report_sha256"],
            "hparams.json": validation["hparams_sha256"],
            "ahac_resume_migration.json": validation["migration_sha256"],
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-audit", type=Path, required=True)
    parser.add_argument("--ahac-fidelity-audit", type=Path, required=True)
    parser.add_argument("--critic-calibration-audit", type=Path, required=True)
    parser.add_argument("--source-run-json", type=Path, required=True)
    parser.add_argument("--source-stderr", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("CAGrad diagnostic seed must equal zero")
    run_diagnostic(args)


if __name__ == "__main__":
    main()
