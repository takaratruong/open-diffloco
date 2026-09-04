"""Compare exact H8 AHAC reverse adjoints with one dense forward derivative.

This diagnostic removes the complete terminal-critic graph from E017's
scale-zero actor objective, repeats the ordinary per-environment reverse
gradient, and evaluates one deterministic dense forward-mode direction over
the retained policy's trainable residual adapter.  It computes no retained
training update or policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
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
from src.algorithms.shac.algorithm import ACTOR_FORWARD_JVP_SEED, train
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically

SOURCE_E017_RUN_SHA256 = (
    "2391b5bd98a09eb74090cfab9aa9cdbb6ae46267b92e4a9663a5babd19a18469"
)
SOURCE_E017_AUDIT_SHA256 = (
    "703cdce9c0a6378ab61b0ac4c09eb22b0945d76f1eae0c0acccccff8d423422b"
)
SOURCE_E017_PROBE_SHA256 = (
    "a08133a130624e65b6732fc755464e3d984faa63e06937c58a60077f62c15fcc"
)
SOURCE_E017_ARM_SHA256 = (
    "b7bb5932a0e94fb0ecb84b08b1208477bed1d37072d61b1cc19a1aaf26f26fd4"
)
SOURCE_E017_COMPARISON_SHA256 = (
    "4686578f8236bc91829a446e1e7a90bbf7b34e010871e20773e8d61a8cd525de"
)
SOURCE_E017_COMPLETION_SHA256 = (
    "36c7c5f37f672d4ffd61a0ec32d0b42281fdac80920a33e17c02e2ca573acaed"
)
SOURCE_E017_RECEIPT_SHA256 = (
    "ac66419837449b098bea410392425b36327b73e4a29ec7bc87cfc1fe921dc2a1"
)
SOURCE_E017_HPARAMS_SHA256 = (
    "624bd0ffd0441cec793a2042d13235dc4eaa94496afb8469dd517b1572df578e"
)
SOURCE_E017_MIGRATION_SHA256 = (
    "c2acb291c4964654e9e1cbb175ea4bb2a0f6abc93ed42997684c647cfca5e74e"
)
TRAINABLE_SCALAR_COUNT = 91_677
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
    probe_output: str | Path,
) -> dict[str, Any]:
    """Build E017 scale zero plus only graph excision and a dense JVP."""

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
        actor_forward_jvp_probe=True,
    )
    return kwargs


def _validate_bool_vector(value: object, *, name: str) -> list[bool]:
    if (
        not isinstance(value, list)
        or len(value) != POPULATION_SIZE
        or any(type(item) is not bool for item in value)
    ):
        raise ValueError(f"{name} must contain {POPULATION_SIZE} booleans")
    return value


def classify_reverse_forward_masks(
    *,
    source_mask: Sequence[bool],
    structural_mask: Sequence[bool],
    forward_finite_mask: Sequence[bool],
) -> dict[str, object]:
    """Classify graph excision before interpreting one forward direction."""

    masks = (source_mask, structural_mask, forward_finite_mask)
    if any(
        len(mask) != POPULATION_SIZE or any(type(value) is not bool for value in mask)
        for mask in masks
    ):
        raise ValueError("reverse/forward masks must each contain 512 booleans")
    source = list(source_mask)
    structural = list(structural_mask)
    forward = list(forward_finite_mask)
    source_count = sum(source)
    structural_count = sum(structural)
    if source_count != 326:
        raise ValueError("source E017 mask must contain 326 finite adjoints")

    recovered = [
        index
        for index, (before, after) in enumerate(zip(source, structural, strict=True))
        if not before and after
    ]
    introduced = [
        index
        for index, (before, after) in enumerate(zip(source, structural, strict=True))
        if before and not after
    ]
    forward_nonfinite = [index for index, finite in enumerate(forward) if not finite]
    overlap = [index for index in forward_nonfinite if not structural[index]]

    if structural_count == POPULATION_SIZE:
        outcome = "structural-excision-restores-all-reverse-adjoints"
    elif structural_count > source_count:
        outcome = "structural-excision-partially-restores-reverse-adjoints"
    elif structural != source:
        outcome = "structural-excision-redistributes-or-worsens-reverse-adjoints"
    elif not forward_nonfinite:
        outcome = "identical-reverse-mask-dense-forward-direction-finite"
    elif overlap:
        outcome = "dense-forward-direction-shares-reverse-invalidity"
    else:
        outcome = "dense-forward-direction-diverges-outside-reverse-invalidity"

    return {
        "outcome": outcome,
        "source_finite_reverse_count": source_count,
        "structural_finite_reverse_count": structural_count,
        "forward_finite_count": sum(forward),
        "reverse_masks_equal": source == structural,
        "recovered_reverse_indices": recovered,
        "introduced_nonfinite_reverse_indices": introduced,
        "forward_nonfinite_indices": forward_nonfinite,
        "forward_nonfinite_reverse_overlap_indices": overlap,
    }


def validate_forward_jvp_report(
    report: Mapping[str, object],
    *,
    expected_losses: Sequence[float],
    expected_reverse_mask: Sequence[bool],
) -> dict[str, object]:
    """Independently reconstruct the complete dense-direction JVP report."""

    if len(expected_losses) != POPULATION_SIZE:
        raise ValueError("expected forward-JVP losses must contain 512 values")
    reverse_mask = list(expected_reverse_mask)
    if len(reverse_mask) != POPULATION_SIZE or any(
        type(value) is not bool for value in reverse_mask
    ):
        raise ValueError("expected reverse mask must contain 512 booleans")

    finite_mask = _validate_bool_vector(
        report.get("finite_by_env"), name="forward finite mask"
    )
    nonzero_mask = _validate_bool_vector(
        report.get("nonzero_by_env"), name="forward nonzero mask"
    )
    reported_reverse = _validate_bool_vector(
        report.get("reverse_gradient_finite_by_env"),
        name="forward report reverse mask",
    )
    primals = report.get("primals_by_env")
    tangents = report.get("directional_derivatives_by_env")
    fingerprint = report.get("direction_fingerprint")
    direction_norm = report.get("direction_norm")
    if (
        report.get("protocol") != "shac-actor-forward-jvp-population-v1"
        or report.get("valid") is not True
        or report.get("population_size") != POPULATION_SIZE
        or report.get("primal_matches_reverse_losses") is not True
        or report.get("repeat_primal_exact") is not True
        or report.get("repeat_tangent_exact") is not True
        or not isinstance(fingerprint, list)
        or len(fingerprint) != 4
        or any(
            type(value) is not int or not 0 <= value <= 0xFFFFFFFF
            for value in fingerprint
        )
        or type(direction_norm) not in (int, float)
        or not math.isfinite(float(direction_norm))
        or abs(float(direction_norm) - 1.0) > 5e-5
        or report.get("trainable_scalar_count") != TRAINABLE_SCALAR_COUNT
        or not isinstance(primals, list)
        or len(primals) != POPULATION_SIZE
        or not isinstance(tangents, list)
        or len(tangents) != POPULATION_SIZE
    ):
        raise ValueError("forward-JVP report header or vectors are invalid")
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in primals
    ):
        raise ValueError("forward-JVP primals must all be finite numbers")
    if list(primals) != list(expected_losses):
        raise ValueError("forward-JVP primals do not equal reverse losses")
    if reported_reverse != reverse_mask:
        raise ValueError("forward-JVP report carries the wrong reverse mask")
    if any(
        value is not None
        and (type(value) not in (int, float) or not math.isfinite(float(value)))
        for value in tangents
    ):
        raise ValueError("forward-JVP tangents must be finite numbers or null")

    reconstructed_finite = [value is not None for value in tangents]
    reconstructed_nonzero = [
        value is not None and float(value) != 0.0 for value in tangents
    ]
    reverse_invalid = [not value for value in reverse_mask]
    finite_on_reverse_invalid = sum(
        finite and invalid
        for finite, invalid in zip(reconstructed_finite, reverse_invalid, strict=True)
    )
    if (
        finite_mask != reconstructed_finite
        or nonzero_mask != reconstructed_nonzero
        or report.get("finite_count") != sum(reconstructed_finite)
        or report.get("nonfinite_count") != POPULATION_SIZE - sum(reconstructed_finite)
        or report.get("nonzero_count") != sum(reconstructed_nonzero)
        or report.get("reverse_invalid_count") != sum(reverse_invalid)
        or report.get("finite_on_reverse_invalid_count") != finite_on_reverse_invalid
    ):
        raise ValueError("forward-JVP population counts do not reconstruct")
    return {
        "finite_mask": finite_mask,
        "nonzero_mask": nonzero_mask,
        "nonfinite_indices": [
            index for index, finite in enumerate(finite_mask) if not finite
        ],
        "nonzero_on_reverse_invalid_count": sum(
            nonzero and invalid
            for nonzero, invalid in zip(nonzero_mask, reverse_invalid, strict=True)
        ),
        "direction_norm": float(direction_norm),
        "direction_fingerprint": fingerprint,
    }


def validate_source_e017(
    *,
    run_json_path: Path,
    audit_path: Path,
    probe_path: Path,
    arm_path: Path,
    comparison_path: Path,
    completion_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, object], dict[str, Any], dict[str, Any]]:
    """Bind the exact independently audited E017 scale-zero population."""

    evidence = (
        (run_json_path, SOURCE_E017_RUN_SHA256, "run"),
        (audit_path, SOURCE_E017_AUDIT_SHA256, "audit"),
        (probe_path, SOURCE_E017_PROBE_SHA256, "scale-zero probe"),
        (arm_path, SOURCE_E017_ARM_SHA256, "scale-zero arm"),
        (comparison_path, SOURCE_E017_COMPARISON_SHA256, "comparison"),
        (completion_path, SOURCE_E017_COMPLETION_SHA256, "completion"),
        (receipt_path, SOURCE_E017_RECEIPT_SHA256, "receipt"),
    )
    for path, expected, label in evidence:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source E017 {label} SHA-256 mismatch")

    run = _read_json(run_json_path)
    audit = _read_json(audit_path)
    probe = _read_json(probe_path)
    arm = _read_json(arm_path)
    comparison = _read_json(comparison_path)
    completion = _read_json(completion_path)
    receipt = _read_json(receipt_path)
    attempts = run.get("attempts")
    population = probe.get("cagrad_population")
    if (
        run.get("experiment") != "E-20260903-017"
        or run.get("return_code") != 0
        or run.get("timed_out") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or audit.get("audit_valid") is not True
        or audit.get("experiment_valid") is not True
        or audit.get("classification")
        != "zero-bootstrap-preserves-identical-invalid-mask"
        or audit.get("checks_passed") != audit.get("checks_total")
        or comparison.get("valid") is not True
        or comparison.get("outcome")
        != "zero-bootstrap-preserves-identical-invalid-mask"
        or comparison.get("scale_zero_finite_gradient_count") != 326
        or comparison.get("gradient_masks_equal") is not True
        or completion.get("valid") is not True
        or completion.get("outcome")
        != "zero-bootstrap-preserves-identical-invalid-mask"
        or completion.get("persisted_optimizer_updates") != 0
        or completion.get("policy_retained") is not False
        or receipt.get("schema") != "agentic-evidence/v1"
        or receipt.get("run_id") != "E-20260903-017/20260904T012426Z"
        or receipt.get("inputs_sha256") != SOURCE_E017_AUDIT_SHA256
        or receipt.get("audit_valid") is not True
        or arm.get("valid") is not True
        or arm.get("arm") != "scale_zero"
        or arm.get("actor_bootstrap_scale") != 0.0
        or arm.get("probe_sha256") != SOURCE_E017_PROBE_SHA256
        or arm.get("hparams_sha256") != SOURCE_E017_HPARAMS_SHA256
        or arm.get("migration_sha256") != SOURCE_E017_MIGRATION_SHA256
        or not isinstance(population, Mapping)
    ):
        raise ValueError("source E017 evidence does not support this discriminator")
    summary = _validate_population_report(population, expected_scale=0.0)
    if int(summary["finite_count"]) != 326:
        raise ValueError("source E017 scale-zero population is not exact")
    source_hparams = Path(str(arm["run_directory"])) / "hparams.json"
    source_migration = Path(str(arm["run_directory"])) / "ahac_resume_migration.json"
    if (
        not source_hparams.is_file()
        or sha256_file(source_hparams) != SOURCE_E017_HPARAMS_SHA256
        or not source_migration.is_file()
        or sha256_file(source_migration) != SOURCE_E017_MIGRATION_SHA256
    ):
        raise ValueError("source E017 hparams or migration are missing")
    return (
        {
            "experiment": "E-20260903-017",
            "run_id": "E-20260903-017/20260904T012426Z",
            "outcome": "zero-bootstrap-preserves-identical-invalid-mask",
            "population_size": POPULATION_SIZE,
            "finite_reverse_gradient_count": 326,
            "run_json": str(run_json_path.resolve()),
            "run_json_sha256": SOURCE_E017_RUN_SHA256,
            "audit": str(audit_path.resolve()),
            "audit_sha256": SOURCE_E017_AUDIT_SHA256,
            "probe": str(probe_path.resolve()),
            "probe_sha256": SOURCE_E017_PROBE_SHA256,
            "arm": str(arm_path.resolve()),
            "arm_sha256": SOURCE_E017_ARM_SHA256,
            "comparison_sha256": SOURCE_E017_COMPARISON_SHA256,
            "completion_sha256": SOURCE_E017_COMPLETION_SHA256,
            "receipt_sha256": SOURCE_E017_RECEIPT_SHA256,
        },
        probe,
        arm,
    )


def compare_probe_reports(
    source: Mapping[str, object], structural: Mapping[str, object]
) -> dict[str, object]:
    """Verify exact pairing and classify reverse versus dense forward mode."""

    for label, report in (("source", source), ("structural", structural)):
        if (
            report.get("valid") is not True
            or report.get("input_step") != START_STEP
            or report.get("compiled_callable_reused") is not True
            or report.get("ordinary_training_loop_entered") is not False
            or report.get("full_state_exact") is not True
            or report.get("metrics_exact") is not True
        ):
            raise ValueError(f"{label} determinism probe is invalid")
    if source.get("input_state_sha256") != structural.get("input_state_sha256"):
        raise ValueError("source and structural probes did not share exact state")
    source_population = source.get("cagrad_population")
    structural_population = structural.get("cagrad_population")
    if not isinstance(source_population, Mapping) or not isinstance(
        structural_population, Mapping
    ):
        raise TypeError("reverse-gradient population report is missing")
    summaries = {
        "source": _validate_population_report(source_population, expected_scale=0.0),
        "structural": _validate_population_report(
            structural_population, expected_scale=0.0
        ),
    }
    for name in (
        "phases",
        "supports",
        "terminals",
        "loss_mask",
        "loss_counts",
        "losses",
    ):
        if summaries["source"][name] != summaries["structural"][name]:
            raise ValueError(f"structural probe changed paired population {name}")
    physical_matches = {
        name: _probe_boundary(source, name) == _probe_boundary(structural, name)
        for name in PHYSICAL_BOUNDARIES
    }
    if not all(physical_matches.values()):
        raise ValueError("structural probe changed a physical/critic boundary")

    source_mask = summaries["source"]["gradient_mask"]
    structural_mask = summaries["structural"]["gradient_mask"]
    losses = summaries["structural"]["losses"]
    forward_report = structural.get("actor_forward_jvp")
    if (
        not isinstance(source_mask, list)
        or not isinstance(structural_mask, list)
        or not isinstance(losses, list)
        or not isinstance(forward_report, Mapping)
    ):
        raise TypeError("structural reverse/forward vectors are missing")
    forward = validate_forward_jvp_report(
        forward_report,
        expected_losses=losses,
        expected_reverse_mask=structural_mask,
    )
    classified = classify_reverse_forward_masks(
        source_mask=source_mask,
        structural_mask=structural_mask,
        forward_finite_mask=forward["finite_mask"],
    )
    source_actor_update = _probe_boundary(source, "actor_cagrad")
    structural_actor_update = _probe_boundary(structural, "actor_cagrad")
    return {
        "protocol": "g1-e002-ahac-reverse-forward-discriminator-v1",
        "valid": True,
        **classified,
        "source_step": START_STEP,
        "computed_output_step": END_STEP,
        "population_size": POPULATION_SIZE,
        "structural_reverse_losses_match_source": True,
        "physical_boundaries_match": physical_matches,
        "input_state_exact": True,
        "actor_update_matches_source": (source_actor_update == structural_actor_update),
        "forward_direction_norm": forward["direction_norm"],
        "forward_direction_fingerprint": forward["direction_fingerprint"],
        "forward_nonzero_count": int(forward_report["nonzero_count"]),
        "forward_nonzero_on_reverse_invalid_count": forward[
            "nonzero_on_reverse_invalid_count"
        ],
        "forward_direction_count": 1,
        "forward_full_jacobian_claimed": False,
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
        protocol="g1-e002-ahac-reverse-forward-preflight-v1",
        source_e017=source,
        actor_bootstrap_scale=0.0,
        actor_bootstrap_graph_mode="excised",
        forward_mode="one-dense-trainable-adapter-direction",
        forward_jvp_seed=ACTOR_FORWARD_JVP_SEED,
        trainable_scalar_count=TRAINABLE_SCALAR_COUNT,
        deterministic_invocations=2,
        ordinary_training_loop_entered=False,
        optimizer_update_retained=False,
        policy_retained=False,
    )
    return preflight, source_probe, source_arm


def _validate_arm_hparams(
    run_directory: Path,
    *,
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
        "actor_forward_jvp_probe": True,
        "actor_forward_jvp_seed": ACTOR_FORWARD_JVP_SEED,
        "actor_residual_preview_trainable_parameter_count": (TRAINABLE_SCALAR_COUNT),
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
        raise ValueError("structural forward-probe hparams are invalid")
    if (
        migration.get("valid") is not True
        or migration.get("source_step") != START_STEP
        or migration.get("actor_and_actor_optimizer_preserved") is not True
        or migration.get("critic_optimizer_reset") is not True
        or migration.get("horizon") != 8.0
        or migration.get("dual_all_zero") is not True
        or sha256_file(migration_path) != SOURCE_E017_MIGRATION_SHA256
    ):
        raise ValueError("structural forward-probe migration is invalid")

    source_hparams_path = Path(str(source_arm["run_directory"])) / "hparams.json"
    source_hparams = _read_json(source_hparams_path)
    allowed = {
        "actor_bootstrap_graph_mode",
        "actor_forward_jvp_probe",
        "actor_forward_jvp_seed",
        "determinism_probe_output",
    }
    common_source = {
        name: value for name, value in source_hparams.items() if name not in allowed
    }
    common_treatment = {
        name: value for name, value in hparams.items() if name not in allowed
    }
    if common_source != common_treatment:
        raise ValueError("structural forward probe changed an unregistered hparam")
    if list(run_directory.rglob("*.pkl")):
        raise ValueError("structural forward probe unexpectedly wrote a checkpoint")
    report = _read_json(probe_output)
    population = report.get("cagrad_population")
    if not isinstance(population, Mapping):
        raise TypeError("structural forward reverse population is missing")
    _validate_population_report(population, expected_scale=0.0)
    if not isinstance(report.get("actor_forward_jvp"), Mapping):
        raise TypeError("structural forward JVP population is missing")
    return {
        "run_directory": str(run_directory.resolve()),
        "hparams_sha256": sha256_file(hparams_path),
        "migration_sha256": sha256_file(migration_path),
        "probe_sha256": sha256_file(probe_output),
        "policy_retained": False,
    }


def _run_worker(args: argparse.Namespace) -> int:
    if args.probe_output is None:
        raise ValueError("worker probe output is required")
    _, _, source_arm = _validate_preflight(args)
    probe_output = args.probe_output.resolve()
    kwargs = build_probe_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        probe_output=probe_output,
    )
    worker_root = args.output_root.resolve() / "structural_forward"
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
        probe_output=probe_output,
        source_arm=source_arm,
    )
    arm.update(
        protocol="g1-e002-ahac-reverse-forward-arm-v1",
        valid=True,
        actor_bootstrap_scale=0.0,
        actor_bootstrap_graph_mode="excised",
        forward_jvp_seed=ACTOR_FORWARD_JVP_SEED,
        forward_direction_count=1,
    )
    _write_json_atomically(
        args.output_root.resolve() / "structural_forward_arm.json", arm
    )
    del kwargs
    gc.collect()
    jax.clear_caches()
    return 0


def _worker_command(args: argparse.Namespace, *, probe_output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.g1_ahac_reverse_forward_discriminator.run",
        "--worker",
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
        raise FileExistsError("reverse-forward preflight already exists")
    preflight, source_probe, _ = _validate_preflight(args)
    _write_json_atomically(preflight_path, preflight)

    probe_path = output_root / "structural_forward_probe.json"
    command = _worker_command(args, probe_output=probe_path)
    completed = subprocess.run(command, check=False)
    process = {"command": command, "returncode": completed.returncode}
    process_path = output_root / "worker_process.json"
    _write_json_atomically(process_path, process)
    if completed.returncode != 0:
        _write_json_atomically(
            output_root / "completion.json",
            {
                "protocol": "g1-e002-ahac-reverse-forward-completion-v1",
                "valid": False,
                "outcome": "invalid-execution",
                "worker": process,
                "persisted_optimizer_updates": 0,
                "optimizer_update_retained": False,
                "policy_retained": False,
                "retained_policy": None,
            },
        )
        return completed.returncode

    structural_probe = _read_json(probe_path)
    comparison = compare_probe_reports(source_probe, structural_probe)
    arm_path = output_root / "structural_forward_arm.json"
    comparison.update(
        code_commit=args.code_commit,
        source_e017=preflight["source_e017"],
        structural_probe_sha256=sha256_file(probe_path),
        structural_arm=_read_json(arm_path),
    )
    comparison_path = output_root / "comparison.json"
    _write_json_atomically(comparison_path, comparison)
    completion = {
        "protocol": "g1-e002-ahac-reverse-forward-completion-v1",
        "valid": True,
        "outcome": comparison["outcome"],
        "computed_optimizer_updates": 2,
        "persisted_optimizer_updates": 0,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "worker_process.json": sha256_file(process_path),
            "structural_forward_probe.json": sha256_file(probe_path),
            "structural_forward_arm.json": sha256_file(arm_path),
            "comparison.json": sha256_file(comparison_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
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
    if args.worker and args.probe_output is None:
        raise ValueError("worker probe output is required")
    raise SystemExit(_run_worker(args) if args.worker else _run_parent(args))


if __name__ == "__main__":
    main()
