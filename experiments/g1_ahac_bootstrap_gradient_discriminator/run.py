"""Compare scale-one and scale-zero AHAC actor adjoints on exact paired rollouts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import jax

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


ARMS = ("scale_one", "scale_zero")
ARM_SCALES = {"scale_one": 1.0, "scale_zero": 0.0}
PHYSICAL_BOUNDARIES = (
    "random_inputs",
    "first_actor_action",
    "first_mjx_substep",
    "first_mjx_control_step",
    "first_env_step",
    "rollout",
    "learned_dynamics",
    "critic",
)
SOURCE_RUN_JSON_SHA256 = (
    "cf3a6436847f11b711f5e1da20ba7a52417edc2b3707657c8f4205f9739aa11d"
)
SOURCE_AUDIT_SHA256 = "040e9c95182f44bb11d7105dbcf6f2218abcdf946824b386f056abed0ee079e0"
SOURCE_FAILURE_SHA256 = (
    "d20efb53fa51949ecf338ccd2ca8049fe26db26a7ee2552d9ef90c5d5c43c1eb"
)
POPULATION_FIELDS = (
    "gradient_finite_by_env",
    "loss_finite_by_env",
    "start_phases",
    "start_support_modes",
    "terminal_modes",
    "losses_by_env",
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant {value} in {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def build_probe_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    arm: str,
    probe_output: str | Path,
) -> dict[str, Any]:
    """Build one exact no-retention proposal with only bootstrap scale varied."""

    if arm not in ARMS:
        raise ValueError("bootstrap discriminator arm is invalid")
    kwargs = build_proposal_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
    )
    kwargs.update(
        actor_bootstrap_scale=ARM_SCALES[arm],
        allow_resume_actor_bootstrap_scale_change=True,
        allow_ahac_actor_bootstrap_ablation=True,
        determinism_probe_output=str(Path(probe_output).resolve()),
    )
    return kwargs


def validate_source_e016(
    *,
    run_json_path: Path,
    audit_path: Path,
    failure_path: Path,
) -> dict[str, object]:
    """Bind the exact valid E016 population that motivates this discriminator."""

    expected = (
        (run_json_path, SOURCE_RUN_JSON_SHA256, "run"),
        (audit_path, SOURCE_AUDIT_SHA256, "audit"),
        (failure_path, SOURCE_FAILURE_SHA256, "failure"),
    )
    for path, digest, label in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"source E016 {label} SHA-256 mismatch")
    run = _read_json(run_json_path)
    audit = _read_json(audit_path)
    failure = _read_json(failure_path)
    attempts = run.get("attempts")
    population = failure.get("population")
    if (
        run.get("experiment") != "E-20260903-016"
        or run.get("return_code") != 0
        or run.get("timed_out") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or audit.get("audit_valid") is not True
        or audit.get("experiment_valid") is not True
        or audit.get("classification") != "nonfinite-per-environment-gradients"
        or audit.get("checks_passed") != audit.get("checks_total")
        or audit.get("checks_total") == 0
        or failure.get("protocol") != "shac-cagrad-failure-v1"
        or failure.get("failure_class") != "nonfinite-per-environment-gradients"
        or not isinstance(population, Mapping)
        or population.get("size") != 512
        or population.get("finite_gradient_count") != 326
        or population.get("finite_loss_count") != 512
    ):
        raise ValueError("source E016 evidence does not support this discriminator")
    return {
        "experiment": "E-20260903-016",
        "run_id": audit.get("run_id"),
        "run_json": str(run_json_path.resolve()),
        "run_json_sha256": SOURCE_RUN_JSON_SHA256,
        "audit": str(audit_path.resolve()),
        "audit_sha256": SOURCE_AUDIT_SHA256,
        "failure": str(failure_path.resolve()),
        "failure_sha256": SOURCE_FAILURE_SHA256,
        "classification": "nonfinite-per-environment-gradients",
        "finite_gradient_count": 326,
        "population_size": 512,
    }


def _probe_boundary(report: Mapping[str, object], name: str) -> object:
    boundaries = report.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("determinism probe boundaries are missing")
    boundary = boundaries.get(name)
    if not isinstance(boundary, Mapping) or boundary.get("exact") is not True:
        raise ValueError(f"determinism boundary {name} is invalid")
    return boundary.get("first")


def _population_row(
    index: int,
    *,
    phases: list[int],
    supports: list[int],
    terminals: list[int],
    losses: list[float],
    loss_mask: list[bool],
) -> dict[str, object]:
    return {
        "index": index,
        "phase": phases[index],
        "start_support_mode": supports[index],
        "terminal_mode": terminals[index],
        "loss": losses[index],
        "loss_finite": loss_mask[index],
    }


def _validate_population_report(
    report: Mapping[str, object],
    *,
    expected_scale: float,
    expected_input_step: int = START_STEP,
    expected_output_step: int = END_STEP,
) -> dict[str, object]:
    if (
        report.get("protocol") != "shac-cagrad-population-v1"
        or report.get("report_valid") is not True
        or report.get("input_step") != expected_input_step
        or report.get("computed_output_step") != expected_output_step
        or report.get("computed_candidate_state_persisted") is not False
    ):
        raise ValueError("CAGrad population report header is invalid")
    phase_bins = report.get("phase_bins")
    population = report.get("population")
    subgates = report.get("subgates")
    ahac = report.get("ahac")
    if not all(
        isinstance(value, Mapping) for value in (phase_bins, population, subgates, ahac)
    ):
        raise ValueError("CAGrad population report is incomplete")
    vectors = {name: population.get(name) for name in POPULATION_FIELDS}
    if any(
        not isinstance(value, list) or len(value) != 512 for value in vectors.values()
    ):
        raise ValueError("CAGrad population vectors must each have 512 rows")
    gradient_mask = vectors["gradient_finite_by_env"]
    loss_mask = vectors["loss_finite_by_env"]
    phases = vectors["start_phases"]
    supports = vectors["start_support_modes"]
    terminals = vectors["terminal_modes"]
    losses = vectors["losses_by_env"]
    if (
        any(type(value) is not bool for value in gradient_mask)
        or any(type(value) is not bool for value in loss_mask)
        or any(type(value) is not int or not 0 <= value < 271 for value in phases)
        or any(type(value) is not int or not 0 <= value <= 3 for value in supports)
        or any(type(value) is not int or not 0 <= value <= 3 for value in terminals)
        or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in losses
        )
        or not all(loss_mask)
    ):
        raise ValueError("CAGrad population vectors are invalid")

    phase_indices = [min(phase * 5 // 271, 4) for phase in phases]
    loss_counts = [
        sum(index == target for index in phase_indices) for target in range(5)
    ]
    gradient_counts = [
        sum(
            index == target and finite
            for index, finite in zip(phase_indices, gradient_mask, strict=True)
        )
        for target in range(5)
    ]
    missing = [
        loss - gradient
        for loss, gradient in zip(loss_counts, gradient_counts, strict=True)
    ]
    bad_gradient_rows = [
        _population_row(
            index,
            phases=phases,
            supports=supports,
            terminals=terminals,
            losses=losses,
            loss_mask=loss_mask,
        )
        for index, finite in enumerate(gradient_mask)
        if not finite
    ]
    finite_count = sum(gradient_mask)
    actor_valid = finite_count == 512
    classification = (
        "cagrad-valid" if actor_valid else "nonfinite-per-environment-gradients"
    )
    if (
        phase_bins.get("loss_counts") != loss_counts
        or phase_bins.get("gradient_counts") != gradient_counts
        or phase_bins.get("missing_gradient_contributors") != missing
        or population.get("size") != 512
        or population.get("finite_gradient_count") != finite_count
        or population.get("finite_loss_count") != 512
        or population.get("nonfinite_gradient_environments") != bad_gradient_rows
        or population.get("nonfinite_loss_environments") != []
        or report.get("actor_cagrad_valid") is not actor_valid
        or report.get("classification") != classification
        or ahac.get("actor_bootstrap_scale_current") != expected_scale
        or subgates.get("loss_bins_occupied") is not True
        or subgates.get("losses_finite") is not True
        or subgates.get("loss_valid") is not True
        or subgates.get("gradient_bins_occupied") is not True
        or subgates.get("gradient_norms_finite") is not True
        or subgates.get("counts_match") is not actor_valid
        or subgates.get("solver_valid") is not True
        or subgates.get("reduction_valid") is not True
    ):
        raise ValueError("CAGrad population reconstruction is inconsistent")
    return {
        "gradient_mask": gradient_mask,
        "loss_mask": loss_mask,
        "phases": phases,
        "supports": supports,
        "terminals": terminals,
        "losses": losses,
        "loss_counts": loss_counts,
        "gradient_counts": gradient_counts,
        "missing": missing,
        "finite_count": finite_count,
        "bad_gradient_rows": bad_gradient_rows,
    }


def _control_matches_source(
    control_report: Mapping[str, object], source_failure: Mapping[str, object]
) -> bool:
    if control_report == source_failure:
        return True
    normalized = deepcopy(dict(control_report))
    normalized["protocol"] = "shac-cagrad-failure-v1"
    normalized["failure_class"] = normalized.pop("classification", None)
    return normalized == source_failure


def compare_probe_reports(
    control: Mapping[str, object],
    treatment: Mapping[str, object],
    *,
    source_failure: Mapping[str, object],
) -> dict[str, object]:
    """Validate exact rollout pairing and classify the zero-bootstrap adjoints."""

    reports = {"scale_one": control, "scale_zero": treatment}
    summaries: dict[str, dict[str, object]] = {}
    for arm, report in reports.items():
        if (
            report.get("valid") is not True
            or report.get("input_step") != START_STEP
            or report.get("compiled_callable_reused") is not True
            or report.get("ordinary_training_loop_entered") is not False
            or report.get("full_state_exact") is not True
            or report.get("metrics_exact") is not True
        ):
            raise ValueError(f"{arm} determinism probe is invalid")
        population = report.get("cagrad_population")
        if not isinstance(population, Mapping):
            raise ValueError(f"{arm} CAGrad population is missing")
        summaries[arm] = _validate_population_report(
            population, expected_scale=ARM_SCALES[arm]
        )
    if control.get("input_state_sha256") != treatment.get("input_state_sha256"):
        raise ValueError("paired bootstrap probes did not start from one exact state")

    physical_matches = {
        name: _probe_boundary(control, name) == _probe_boundary(treatment, name)
        for name in PHYSICAL_BOUNDARIES
    }
    if not all(physical_matches.values()):
        raise ValueError("paired bootstrap probes do not share one physical rollout")
    for name in ("phases", "supports", "terminals", "loss_mask", "loss_counts"):
        if summaries["scale_one"][name] != summaries["scale_zero"][name]:
            raise ValueError(f"paired bootstrap population field {name} differs")

    control_population = control.get("cagrad_population")
    if not isinstance(control_population, Mapping) or not _control_matches_source(
        control_population, source_failure
    ):
        raise ValueError("scale-one control does not reproduce E016 exactly")
    control_mask = summaries["scale_one"]["gradient_mask"]
    treatment_mask = summaries["scale_zero"]["gradient_mask"]
    if not isinstance(control_mask, list) or not isinstance(treatment_mask, list):
        raise ValueError("paired bootstrap masks are invalid")
    control_count = int(summaries["scale_one"]["finite_count"])
    treatment_count = int(summaries["scale_zero"]["finite_count"])
    if control_count != 326:
        raise ValueError("scale-one control does not reproduce 326 finite gradients")
    recovered = [
        index
        for index, (before, after) in enumerate(
            zip(control_mask, treatment_mask, strict=True)
        )
        if not before and after
    ]
    introduced = [
        index
        for index, (before, after) in enumerate(
            zip(control_mask, treatment_mask, strict=True)
        )
        if before and not after
    ]
    if treatment_count == 512:
        outcome = "zero-bootstrap-restores-all-adjoints"
    elif treatment_count > control_count:
        outcome = "zero-bootstrap-partially-restores-adjoints"
    elif treatment_mask == control_mask:
        outcome = "zero-bootstrap-preserves-identical-invalid-mask"
    else:
        outcome = "zero-bootstrap-redistributes-or-worsens-invalid-adjoints"

    control_losses = summaries["scale_one"]["losses"]
    treatment_losses = summaries["scale_zero"]["losses"]
    if not isinstance(control_losses, list) or not isinstance(treatment_losses, list):
        raise ValueError("paired bootstrap losses are invalid")
    loss_delta = [
        float(one) - float(zero)
        for one, zero in zip(control_losses, treatment_losses, strict=True)
    ]
    actor_update_changed = _probe_boundary(control, "actor_cagrad") != _probe_boundary(
        treatment, "actor_cagrad"
    )
    return {
        "protocol": "g1-e002-ahac-bootstrap-gradient-pair-v1",
        "valid": True,
        "outcome": outcome,
        "source_step": START_STEP,
        "computed_output_step": END_STEP,
        "population_size": 512,
        "physical_boundaries_match": physical_matches,
        "input_state_exact": True,
        "scale_one_control_reproduces_e016": True,
        "actor_update_changed": actor_update_changed,
        "scale_one_finite_gradient_count": control_count,
        "scale_zero_finite_gradient_count": treatment_count,
        "scale_one_nonfinite_gradient_count": 512 - control_count,
        "scale_zero_nonfinite_gradient_count": 512 - treatment_count,
        "recovered_gradient_count": len(recovered),
        "introduced_nonfinite_gradient_count": len(introduced),
        "recovered_gradient_indices": recovered,
        "introduced_nonfinite_gradient_indices": introduced,
        "gradient_masks_equal": control_mask == treatment_mask,
        "scale_one_gradient_bin_counts": summaries["scale_one"]["gradient_counts"],
        "scale_zero_gradient_bin_counts": summaries["scale_zero"]["gradient_counts"],
        "loss_delta_scale_one_minus_zero": {
            "mean": sum(loss_delta) / len(loss_delta),
            "min": min(loss_delta),
            "max": max(loss_delta),
            "l2": math.sqrt(sum(value * value for value in loss_delta)),
        },
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": None,
    }


def _validate_arm_hparams(
    run_directory: Path,
    *,
    arm: str,
    probe_output: Path,
) -> dict[str, object]:
    hparams_path = run_directory / "hparams.json"
    migration_path = run_directory / "ahac_resume_migration.json"
    hparams = _read_json(hparams_path)
    migration = _read_json(migration_path)
    expected = {
        "algorithm": "ahac",
        "total_steps": END_STEP,
        "ahac": True,
        "ahac_horizon_min": 8,
        "ahac_horizon_max": 24,
        "ahac_semantics": "paper_equation_10_no_target",
        "actor_bootstrap_scale": ARM_SCALES[arm],
        "actor_bootstrap_delay_steps": 0,
        "allow_resume_actor_bootstrap_scale_change": True,
        "allow_ahac_actor_bootstrap_ablation": True,
        "effective_num_envs": 512,
        "gradient_accumulation_steps": 2,
        "unroll_length": 24,
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "actor_per_env_grad_clip": 1.0,
        "determinism_probe": True,
        "determinism_probe_output": str(probe_output.resolve()),
    }
    if any(hparams.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{arm} hparams are invalid")
    if (
        migration.get("valid") is not True
        or migration.get("source_step") != START_STEP
        or migration.get("actor_and_actor_optimizer_preserved") is not True
        or migration.get("critic_optimizer_reset") is not True
        or migration.get("horizon") != 8.0
        or migration.get("dual_all_zero") is not True
    ):
        raise ValueError(f"{arm} migration is invalid")
    report = _read_json(probe_output)
    population = report.get("cagrad_population")
    if not isinstance(population, Mapping):
        raise ValueError(f"{arm} population report is missing")
    _validate_population_report(population, expected_scale=ARM_SCALES[arm])
    if list(run_directory.glob("*.pkl")):
        raise ValueError(f"{arm} unexpectedly persisted a checkpoint")
    return {
        "run_directory": str(run_directory.resolve()),
        "hparams_sha256": sha256_file(hparams_path),
        "migration_sha256": sha256_file(migration_path),
        "probe_sha256": sha256_file(probe_output),
        "policy_retained": False,
    }


def _validate_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = Path(__file__).resolve().parents[2]
    source = validate_source_e016(
        run_json_path=args.source_run_json.resolve(),
        audit_path=args.source_audit.resolve(),
        failure_path=args.source_failure.resolve(),
    )
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        resume_audit=args.resume_audit.resolve(),
        ahac_fidelity_audit=args.ahac_fidelity_audit.resolve(),
        critic_calibration_audit=args.critic_calibration_audit.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-e002-ahac-bootstrap-gradient-preflight-v1",
        source_e016=source,
        arms=list(ARMS),
        actor_bootstrap_scales=ARM_SCALES,
        deterministic_invocations_per_arm=2,
        ordinary_training_loop_entered=False,
        optimizer_update_retained=False,
        policy_retained=False,
    )
    return preflight, _read_json(args.source_failure.resolve())


def _run_worker(args: argparse.Namespace) -> int:
    if args.worker_arm not in ARMS or args.probe_output is None:
        raise ValueError("worker arm and probe output are required")
    _validate_preflight(args)
    probe_output = args.probe_output.resolve()
    kwargs = build_probe_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        arm=args.worker_arm,
        probe_output=probe_output,
    )
    worker_root = args.output_root.resolve() / args.worker_arm
    worker_root.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(worker_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_run_directory = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (worker_root / relative_run_directory).resolve()
    arm = _validate_arm_hparams(
        run_directory,
        arm=args.worker_arm,
        probe_output=probe_output,
    )
    arm.update(
        protocol="g1-e002-ahac-bootstrap-gradient-arm-v1",
        valid=True,
        arm=args.worker_arm,
        actor_bootstrap_scale=ARM_SCALES[args.worker_arm],
    )
    _write_json_atomically(
        args.output_root.resolve() / f"{args.worker_arm}_arm.json",
        arm,
    )
    del kwargs
    gc.collect()
    jax.clear_caches()
    return 0


def _worker_command(
    args: argparse.Namespace, *, arm: str, probe_output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.g1_ahac_bootstrap_gradient_discriminator.run",
        "--worker-arm",
        arm,
        "--solver-profile",
        args.solver_profile,
        "--reference-path",
        str(args.reference_path.resolve()),
        "--resume-from",
        str(args.resume_from.resolve()),
        "--resume-audit",
        str(args.resume_audit.resolve()),
        "--ahac-fidelity-audit",
        str(args.ahac_fidelity_audit.resolve()),
        "--critic-calibration-audit",
        str(args.critic_calibration_audit.resolve()),
        "--source-run-json",
        str(args.source_run_json.resolve()),
        "--source-audit",
        str(args.source_audit.resolve()),
        "--source-failure",
        str(args.source_failure.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--probe-output",
        str(probe_output.resolve()),
        "--code-commit",
        args.code_commit,
        "--seed",
        str(args.seed),
    ]


def _run_parent(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        raise FileExistsError("bootstrap discriminator preflight already exists")
    preflight, source_failure = _validate_preflight(args)
    _write_json_atomically(preflight_path, preflight)

    probe_paths = {arm: output_root / f"{arm}_probe.json" for arm in ARMS}
    processes = []
    for arm in ARMS:
        command = _worker_command(args, arm=arm, probe_output=probe_paths[arm])
        completed = subprocess.run(command, check=False)
        processes.append(
            {"arm": arm, "command": command, "returncode": completed.returncode}
        )
        if completed.returncode != 0:
            _write_json_atomically(
                output_root / "completion.json",
                {
                    "protocol": "g1-e002-ahac-bootstrap-gradient-completion-v1",
                    "valid": False,
                    "outcome": "invalid-execution",
                    "processes": processes,
                    "optimizer_update_retained": False,
                    "policy_retained": False,
                    "retained_policy": None,
                },
            )
            return completed.returncode
    processes_path = output_root / "worker_processes.json"
    _write_json_atomically(processes_path, processes)

    reports = {arm: _read_json(path) for arm, path in probe_paths.items()}
    comparison = compare_probe_reports(
        reports["scale_one"],
        reports["scale_zero"],
        source_failure=source_failure,
    )
    arm_manifests = {arm: _read_json(output_root / f"{arm}_arm.json") for arm in ARMS}
    comparison.update(
        code_commit=args.code_commit,
        source_e016=preflight["source_e016"],
        probe_sha256={arm: sha256_file(path) for arm, path in probe_paths.items()},
        arm_manifests=arm_manifests,
    )
    comparison_path = output_root / "comparison.json"
    _write_json_atomically(comparison_path, comparison)
    completion = {
        "protocol": "g1-e002-ahac-bootstrap-gradient-completion-v1",
        "valid": True,
        "outcome": comparison["outcome"],
        "computed_optimizer_updates_per_arm": 2,
        "persisted_optimizer_updates": 0,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "worker_processes.json": sha256_file(processes_path),
            "comparison.json": sha256_file(comparison_path),
            **{
                f"{arm}_probe.json": sha256_file(path)
                for arm, path in probe_paths.items()
            },
            **{
                f"{arm}_arm.json": sha256_file(output_root / f"{arm}_arm.json")
                for arm in ARMS
            },
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--resume-audit", type=Path, required=True)
    parser.add_argument("--ahac-fidelity-audit", type=Path, required=True)
    parser.add_argument("--critic-calibration-audit", type=Path, required=True)
    parser.add_argument("--source-run-json", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-failure", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    if args.worker_arm is not None and args.probe_output is None:
        raise ValueError("worker probe output is required")
    raise SystemExit(
        _run_worker(args) if args.worker_arm is not None else _run_parent(args)
    )


if __name__ == "__main__":
    main()
