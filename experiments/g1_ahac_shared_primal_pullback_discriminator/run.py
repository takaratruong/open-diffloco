"""Test inactive AHAC pullbacks with one shared compiled primal graph.

The connected and excised arms are runtime boolean calls to the same compiled
SHAC update.  The environment step is evaluated once by a common custom-VJP
forward rule; only the backward rule may skip inactive-horizon transposes.  No
computed optimizer update or policy is retained.
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
)
from experiments.g1_ahac_first_proposal_audit.run import END_STEP, START_STEP
from experiments.g1_ahac_inactive_horizon_gradient_discriminator.run import (
    _read_json,
    _validate_preflight as validate_source_preflight,
    build_probe_kwargs as build_source_probe_kwargs,
    compare_probe_reports,
)
from experiments.g1_ahac_reverse_forward_discriminator.run import (
    SOURCE_E017_MIGRATION_SHA256,
)
from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


POPULATION_SIZE = 512
RUNTIME_MODE = "runtime-paired"


def build_probe_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    probe_output: str | Path,
) -> dict[str, Any]:
    """Build the E020 control with one dynamic paired pullback mode."""

    kwargs = build_source_probe_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        arm="connected",
        probe_output=probe_output,
    )
    kwargs.update(
        actor_inactive_horizon_gradient_mode=RUNTIME_MODE,
        determinism_probe_output=str(Path(probe_output).resolve()),
    )
    return kwargs


def extract_runtime_pair_arms(
    report: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Validate the common-input runtime pair and return its two arm reports."""

    if (
        report.get("protocol") != "shac-runtime-pullback-pair-v1"
        or report.get("valid") is not True
        or report.get("compiled_callable_reused") is not True
        or report.get("ordinary_training_loop_entered") is not False
    ):
        raise ValueError("runtime pullback pair report is invalid")
    arms = report.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"connected", "excised"}:
        raise ValueError("runtime pullback pair arms are invalid")
    connected = arms["connected"]
    excised = arms["excised"]
    if not isinstance(connected, Mapping) or not isinstance(excised, Mapping):
        raise TypeError("runtime pullback arms must be mappings")
    expected = ((connected, False), (excised, True))
    for arm, flag in expected:
        if (
            arm.get("valid") is not True
            or arm.get("runtime_excise_inactive") is not flag
            or arm.get("compiled_callable_reused") is not True
            or arm.get("ordinary_training_loop_entered") is not False
            or arm.get("input_step") != report.get("input_step")
        ):
            raise ValueError("runtime pullback arm contract is invalid")
    input_hashes = {
        report.get("input_state_sha256"),
        connected.get("input_state_sha256"),
        excised.get("input_state_sha256"),
    }
    if None in input_hashes or len(input_hashes) != 1:
        raise ValueError("runtime pullback arms lack a shared input state")
    return connected, excised


def _validate_hparams(
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
        "actor_forward_jvp_probe": False,
        "actor_forward_jvp_seed": None,
        "actor_inactive_horizon_gradient_mode": RUNTIME_MODE,
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
        raise ValueError("runtime-paired inactive-horizon hparams are invalid")
    if (
        migration.get("valid") is not True
        or migration.get("source_step") != START_STEP
        or migration.get("actor_and_actor_optimizer_preserved") is not True
        or migration.get("critic_optimizer_reset") is not True
        or migration.get("horizon") != 8.0
        or migration.get("dual_all_zero") is not True
        or sha256_file(migration_path) != SOURCE_E017_MIGRATION_SHA256
    ):
        raise ValueError("runtime-paired AHAC migration is invalid")

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
        raise ValueError("runtime-paired probe changed an unregistered hparam")
    if list(run_directory.rglob("*.pkl")):
        raise ValueError("runtime-paired probe unexpectedly wrote a checkpoint")
    return {
        "run_directory": str(run_directory.resolve()),
        "hparams_sha256": sha256_file(hparams_path),
        "migration_sha256": sha256_file(migration_path),
        "probe_sha256": sha256_file(probe_output),
        "policy_retained": False,
    }


def _run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        raise FileExistsError("shared-primal preflight already exists")
    preflight, source_probe, source_arm = validate_source_preflight(args)
    preflight.update(
        protocol="g1-e002-ahac-shared-primal-pullback-preflight-v1",
        actor_inactive_horizon_gradient_mode=RUNTIME_MODE,
        arms=[RUNTIME_MODE],
        inactive_horizon_gradient_modes=[RUNTIME_MODE],
        compiled_train_callable_count=1,
        runtime_arms=["connected", "excised"],
        deterministic_invocations_per_arm=2,
        compile_warmup_invocations=1,
        ordinary_training_loop_entered=False,
        optimizer_update_retained=False,
        policy_retained=False,
    )
    _write_json_atomically(preflight_path, preflight)

    probe_output = output_root / "runtime_paired_probe.json"
    kwargs = build_probe_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        probe_output=probe_output,
    )
    training_root = output_root / "runtime-paired"
    training_root.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(training_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_run_directory = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (training_root / relative_run_directory).resolve()

    manifest = _validate_hparams(
        run_directory,
        probe_output=probe_output,
        source_arm=source_arm,
    )
    manifest.update(
        protocol="g1-e002-ahac-shared-primal-pullback-run-v1",
        valid=True,
        actor_inactive_horizon_gradient_mode=RUNTIME_MODE,
        compiled_train_callable_count=1,
        optimizer_update_retained=False,
        policy_retained=False,
    )
    manifest_path = output_root / "runtime_paired_run.json"
    _write_json_atomically(manifest_path, manifest)

    pair_report = _read_json(probe_output)
    connected, excised = extract_runtime_pair_arms(pair_report)
    try:
        comparison = compare_probe_reports(
            source=source_probe,
            connected=connected,
            excised=excised,
        )
    except Exception as error:
        _write_json_atomically(
            output_root / "completion.json",
            {
                "protocol": "g1-e002-ahac-shared-primal-pullback-completion-v1",
                "valid": False,
                "outcome": "invalid-execution",
                "error_type": type(error).__name__,
                "error": str(error),
                "computed_train_step_invocations": 5,
                "persisted_optimizer_updates": 0,
                "optimizer_update_retained": False,
                "policy_retained": False,
                "retained_policy": None,
            },
        )
        raise

    for arm in (connected, excised):
        population = arm.get("cagrad_population")
        if not isinstance(population, Mapping):
            raise TypeError("runtime arm CAGrad population is missing")
        _validate_population_report(population, expected_scale=0.0)
    comparison.update(
        protocol="g1-e002-ahac-shared-primal-pullback-pair-v1",
        code_commit=args.code_commit,
        shared_compiled_callable=True,
        shared_custom_vjp_primal=True,
        runtime_pair_probe_sha256=sha256_file(probe_output),
        source_e017=preflight["source_e017"],
        run_manifest=manifest,
    )
    comparison_path = output_root / "comparison.json"
    _write_json_atomically(comparison_path, comparison)
    completion = {
        "protocol": "g1-e002-ahac-shared-primal-pullback-completion-v1",
        "valid": True,
        "outcome": comparison["outcome"],
        "computed_train_step_invocations": 5,
        "scientific_invocations_per_arm": 2,
        "compile_warmup_invocations": 1,
        "persisted_optimizer_updates": 0,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "runtime_paired_probe.json": sha256_file(probe_output),
            "runtime_paired_run.json": sha256_file(manifest_path),
            "comparison.json": sha256_file(comparison_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    jax.clear_caches()
    return 0


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
