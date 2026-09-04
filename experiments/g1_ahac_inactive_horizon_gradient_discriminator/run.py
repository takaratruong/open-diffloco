"""Test whether inactive AHAC scan slots poison otherwise unused actor adjoints.

Both arms replay the exact retained E002 state and random tape with the terminal
critic graph structurally excised.  The treatment changes only whether MJX
steps beyond the dynamic H=8 horizon remain connected to the actor pullback.
Each arm is compiled in a separate subprocess and computes no retained update.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax

from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
    PHYSICAL_BOUNDARIES,
    _probe_boundary,
    _validate_population_report,
)
from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
    build_probe_kwargs as build_source_probe_kwargs,
)
from experiments.g1_ahac_first_proposal_audit.run import (
    END_STEP,
    START_STEP,
    validate_preflight,
)
from experiments.g1_ahac_reverse_forward_discriminator.run import (
    SOURCE_E017_MIGRATION_SHA256,
    validate_source_e017,
)
from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


ARMS = ("connected", "excised")
POPULATION_SIZE = 512


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
    arm: str,
    probe_output: str | Path,
) -> dict[str, Any]:
    """Build exact E017 scale-zero arms differing only at inactive pullbacks."""

    if arm not in ARMS:
        raise ValueError("inactive-horizon discriminator arm is invalid")
    kwargs = build_source_probe_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        arm="scale_zero",
        probe_output=probe_output,
    )
    kwargs.update(
        actor_bootstrap_graph_mode="excised",
        actor_forward_jvp_probe=False,
        actor_inactive_horizon_gradient_mode=arm,
        determinism_probe_output=str(Path(probe_output).resolve()),
    )
    return kwargs


def classify_gradient_masks(
    connected_mask: Sequence[bool], excised_mask: Sequence[bool]
) -> dict[str, object]:
    """Classify exact row-level recovery after inactive-slot graph excision."""

    masks = (connected_mask, excised_mask)
    if any(
        len(mask) != POPULATION_SIZE
        or any(type(value) is not bool for value in mask)
        for mask in masks
    ):
        raise ValueError("inactive-horizon masks must each contain 512 booleans")
    connected = list(connected_mask)
    excised = list(excised_mask)
    connected_count = sum(connected)
    excised_count = sum(excised)
    if connected_count != 326:
        raise ValueError("connected control must contain 326 finite adjoints")

    recovered = [
        index
        for index, (before, after) in enumerate(
            zip(connected, excised, strict=True)
        )
        if not before and after
    ]
    introduced = [
        index
        for index, (before, after) in enumerate(
            zip(connected, excised, strict=True)
        )
        if before and not after
    ]
    if excised_count == POPULATION_SIZE:
        outcome = "inactive-excision-restores-all-adjoints"
    elif excised_count > connected_count:
        outcome = "inactive-excision-partially-restores-adjoints"
    elif excised == connected:
        outcome = "inactive-excision-preserves-invalid-mask"
    else:
        outcome = "inactive-excision-redistributes-or-worsens-adjoints"
    return {
        "outcome": outcome,
        "connected_finite_count": connected_count,
        "excised_finite_count": excised_count,
        "connected_nonfinite_count": POPULATION_SIZE - connected_count,
        "excised_nonfinite_count": POPULATION_SIZE - excised_count,
        "gradient_masks_equal": connected == excised,
        "recovered_count": len(recovered),
        "introduced_nonfinite_count": len(introduced),
        "recovered_indices": recovered,
        "introduced_nonfinite_indices": introduced,
    }


def compare_probe_reports(
    *,
    source: Mapping[str, object],
    connected: Mapping[str, object],
    excised: Mapping[str, object],
) -> dict[str, object]:
    """Enforce exact forward pairing, reproduce E017, and classify gradients."""

    reports = {
        "source_e017": source,
        "connected": connected,
        "excised": excised,
    }
    summaries: dict[str, dict[str, object]] = {}
    for label, report in reports.items():
        if (
            report.get("valid") is not True
            or report.get("input_step") != START_STEP
            or report.get("compiled_callable_reused") is not True
            or report.get("ordinary_training_loop_entered") is not False
            or report.get("full_state_exact") is not True
            or report.get("metrics_exact") is not True
        ):
            raise ValueError(f"{label} determinism probe is invalid")
        population = report.get("cagrad_population")
        if not isinstance(population, Mapping):
            raise TypeError(f"{label} CAGrad population is missing")
        summaries[label] = _validate_population_report(
            population, expected_scale=0.0
        )

    input_hashes = {report.get("input_state_sha256") for report in reports.values()}
    if len(input_hashes) != 1:
        raise ValueError("inactive-horizon probes did not share one exact state")

    physical_matches = {
        name: (
            _probe_boundary(source, name)
            == _probe_boundary(connected, name)
            == _probe_boundary(excised, name)
        )
        for name in PHYSICAL_BOUNDARIES
    }
    if not all(physical_matches.values()):
        raise ValueError("inactive-horizon treatment changed a physical boundary")

    paired_fields = (
        "phases",
        "supports",
        "terminals",
        "loss_mask",
        "loss_counts",
        "losses",
    )
    for name in paired_fields:
        values = {json.dumps(summary[name]) for summary in summaries.values()}
        if len(values) != 1:
            raise ValueError(f"inactive-horizon treatment changed paired {name}")

    source_mask = summaries["source_e017"]["gradient_mask"]
    connected_mask = summaries["connected"]["gradient_mask"]
    excised_mask = summaries["excised"]["gradient_mask"]
    if not all(
        isinstance(mask, list)
        for mask in (source_mask, connected_mask, excised_mask)
    ):
        raise TypeError("inactive-horizon gradient masks are missing")
    if connected_mask != source_mask:
        raise ValueError("connected control does not reproduce E017 exactly")
    classified = classify_gradient_masks(connected_mask, excised_mask)
    return {
        "protocol": "g1-e002-ahac-inactive-horizon-gradient-pair-v1",
        "valid": True,
        **classified,
        "source_step": START_STEP,
        "computed_output_step": END_STEP,
        "population_size": POPULATION_SIZE,
        "input_state_exact": True,
        "connected_control_reproduces_e017": True,
        "physical_boundaries_match": physical_matches,
        "losses_exact": True,
        "actor_update_changed": (
            _probe_boundary(connected, "actor_cagrad")
            != _probe_boundary(excised, "actor_cagrad")
        ),
        "connected_gradient_bin_counts": summaries["connected"][
            "gradient_counts"
        ],
        "excised_gradient_bin_counts": summaries["excised"]["gradient_counts"],
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


def _validate_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    source, source_probe, source_arm = validate_source_e017(**_source_paths(args))
    preflight.update(
        protocol="g1-e002-ahac-inactive-horizon-gradient-preflight-v1",
        source_e017=source,
        arms=list(ARMS),
        actor_bootstrap_scale=0.0,
        actor_bootstrap_graph_mode="excised",
        actor_forward_jvp_probe=False,
        inactive_horizon_gradient_modes=list(ARMS),
        deterministic_invocations_per_arm=2,
        ordinary_training_loop_entered=False,
        optimizer_update_retained=False,
        policy_retained=False,
    )
    return preflight, source_probe, source_arm


def _validate_arm_hparams(
    run_directory: Path,
    *,
    arm: str,
    probe_output: Path,
    source_arm: Mapping[str, object],
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
        "actor_bootstrap_scale": 0.0,
        "actor_bootstrap_delay_steps": 0,
        "allow_resume_actor_bootstrap_scale_change": True,
        "allow_ahac_actor_bootstrap_ablation": True,
        "actor_bootstrap_graph_mode": "excised",
        "actor_forward_jvp_probe": False,
        "actor_forward_jvp_seed": None,
        "actor_inactive_horizon_gradient_mode": arm,
        "effective_num_envs": POPULATION_SIZE,
        "gradient_accumulation_steps": 2,
        "unroll_length": 24,
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "actor_per_env_grad_clip": 1.0,
        "determinism_probe": True,
        "determinism_probe_output": str(probe_output.resolve()),
    }
    if any(hparams.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{arm} inactive-horizon hparams are invalid")
    if (
        migration.get("valid") is not True
        or migration.get("source_step") != START_STEP
        or migration.get("actor_and_actor_optimizer_preserved") is not True
        or migration.get("critic_optimizer_reset") is not True
        or migration.get("horizon") != 8.0
        or migration.get("dual_all_zero") is not True
        or sha256_file(migration_path) != SOURCE_E017_MIGRATION_SHA256
    ):
        raise ValueError(f"{arm} inactive-horizon migration is invalid")

    source_hparams_path = Path(str(source_arm["run_directory"])) / "hparams.json"
    source_hparams = _read_json(source_hparams_path)
    allowed = {
        "actor_bootstrap_graph_mode",
        "actor_forward_jvp_probe",
        "actor_forward_jvp_seed",
        "actor_inactive_horizon_gradient_mode",
        "determinism_probe_output",
    }
    common_source = {
        name: value for name, value in source_hparams.items() if name not in allowed
    }
    common_treatment = {
        name: value for name, value in hparams.items() if name not in allowed
    }
    if common_source != common_treatment:
        raise ValueError(f"{arm} changed an unregistered hparam")
    if list(run_directory.rglob("*.pkl")):
        raise ValueError(f"{arm} unexpectedly wrote a checkpoint")
    report = _read_json(probe_output)
    population = report.get("cagrad_population")
    if not isinstance(population, Mapping):
        raise TypeError(f"{arm} reverse-gradient population is missing")
    _validate_population_report(population, expected_scale=0.0)
    if report.get("actor_forward_jvp") is not None:
        raise ValueError(f"{arm} unexpectedly computed a forward JVP")
    return {
        "run_directory": str(run_directory.resolve()),
        "hparams_sha256": sha256_file(hparams_path),
        "migration_sha256": sha256_file(migration_path),
        "probe_sha256": sha256_file(probe_output),
        "policy_retained": False,
    }


def _run_worker(args: argparse.Namespace) -> int:
    if args.worker_arm not in ARMS or args.probe_output is None:
        raise ValueError("worker arm and probe output are required")
    _, _, source_arm = _validate_preflight(args)
    arm = args.worker_arm
    probe_output = args.probe_output.resolve()
    kwargs = build_probe_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        arm=arm,
        probe_output=probe_output,
    )
    worker_root = args.output_root.resolve() / arm
    worker_root.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(worker_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_run_directory = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (worker_root / relative_run_directory).resolve()
    manifest = _validate_arm_hparams(
        run_directory,
        arm=arm,
        probe_output=probe_output,
        source_arm=source_arm,
    )
    manifest.update(
        protocol="g1-e002-ahac-inactive-horizon-gradient-arm-v1",
        valid=True,
        arm=arm,
        actor_bootstrap_scale=0.0,
        actor_bootstrap_graph_mode="excised",
        actor_inactive_horizon_gradient_mode=arm,
    )
    _write_json_atomically(
        args.output_root.resolve() / f"{arm}_arm.json", manifest
    )
    del kwargs
    gc.collect()
    jax.clear_caches()
    return 0


def _worker_command(
    args: argparse.Namespace, *, arm: str, probe_output: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.g1_ahac_inactive_horizon_gradient_discriminator.run",
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
    ]
    for option, path in (
        ("--source-e017-run-json", args.source_e017_run_json),
        ("--source-e017-audit", args.source_e017_audit),
        ("--source-e017-probe", args.source_e017_probe),
        ("--source-e017-arm", args.source_e017_arm),
        ("--source-e017-comparison", args.source_e017_comparison),
        ("--source-e017-completion", args.source_e017_completion),
        ("--source-e017-receipt", args.source_e017_receipt),
    ):
        command.extend((option, str(path.resolve())))
    command.extend(
        (
            "--output-root",
            str(args.output_root.resolve()),
            "--probe-output",
            str(probe_output.resolve()),
            "--code-commit",
            args.code_commit,
            "--seed",
            str(args.seed),
        )
    )
    return command


def _run_parent(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        raise FileExistsError("inactive-horizon preflight already exists")
    preflight, source_probe, _ = _validate_preflight(args)
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
            _write_json_atomically(output_root / "worker_processes.json", processes)
            _write_json_atomically(
                output_root / "completion.json",
                {
                    "protocol": (
                        "g1-e002-ahac-inactive-horizon-gradient-completion-v1"
                    ),
                    "valid": False,
                    "outcome": "invalid-execution",
                    "processes": processes,
                    "computed_optimizer_updates_per_arm": 2,
                    "persisted_optimizer_updates": 0,
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
        source=source_probe,
        connected=reports["connected"],
        excised=reports["excised"],
    )
    arm_manifests = {
        arm: _read_json(output_root / f"{arm}_arm.json") for arm in ARMS
    }
    comparison.update(
        code_commit=args.code_commit,
        source_e017=preflight["source_e017"],
        probe_sha256={arm: sha256_file(path) for arm, path in probe_paths.items()},
        arm_manifests=arm_manifests,
    )
    comparison_path = output_root / "comparison.json"
    _write_json_atomically(comparison_path, comparison)
    completion = {
        "protocol": "g1-e002-ahac-inactive-horizon-gradient-completion-v1",
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
    parser.add_argument("--source-e017-run-json", type=Path, required=True)
    parser.add_argument("--source-e017-audit", type=Path, required=True)
    parser.add_argument("--source-e017-probe", type=Path, required=True)
    parser.add_argument("--source-e017-arm", type=Path, required=True)
    parser.add_argument("--source-e017-comparison", type=Path, required=True)
    parser.add_argument("--source-e017-completion", type=Path, required=True)
    parser.add_argument("--source-e017-receipt", type=Path, required=True)
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
