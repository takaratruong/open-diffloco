"""Compare E002 multi-episode and first-terminal gradients on one rollout."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import jax

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_dual_scale_root_position import (
    build_arm_kwargs,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
EFFECTIVE_NUM_ENVS = 512
UNROLL_LENGTH = 24
TRANSITIONS_PER_UPDATE = EFFECTIVE_NUM_ENVS * UNROLL_LENGTH
END_STEP = START_STEP + TRANSITIONS_PER_UPDATE
SEMANTICS = ("multi_episode", "first_terminal")
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
COUNT_KEYS = (
    "actor_return_done_env_count",
    "actor_return_done_event_count",
    "actor_return_post_first_done_transition_count",
    "actor_return_post_first_done_env_count",
)
FINITE_SUMMARY_KEYS = (
    "actor_loss",
    "actor_return_mean",
    "actor_return_post_first_done_reward_sum",
    "actor_return_post_first_done_reward_mean",
    "actor_grad",
    "actor_update_norm",
    "actor_grad_population_mean_norm",
    "actor_grad_population_rms_norm",
    "actor_grad_population_variance_trace",
    "actor_grad_population_cancellation_ratio",
    "actor_grad_population_noise_scale",
    "actor_grad_population_esnr",
)


def build_probe_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    semantics: str,
    probe_output: str | Path,
) -> dict[str, Any]:
    """Build one no-retention E002 update probe with one semantic delta."""

    if semantics not in SEMANTICS:
        raise ValueError("actor return semantics are invalid")
    kwargs = build_arm_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        kernel="exponential",
    )
    kwargs.update(
        total_steps=END_STEP,
        checkpoint_steps=(END_STEP,),
        diagnose=True,
        actor_return_semantics=semantics,
        allow_resume_actor_return_semantics_change=(semantics == "first_terminal"),
        determinism_probe_output=str(Path(probe_output).resolve()),
    )
    return kwargs


def _probe_boundary(report: Mapping[str, object], name: str) -> object:
    boundaries = report.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("determinism probe boundaries are missing")
    boundary = boundaries.get(name)
    if not isinstance(boundary, Mapping) or boundary.get("exact") is not True:
        raise ValueError(f"determinism boundary {name} is invalid")
    return boundary.get("first")


def _validated_summary(report: Mapping[str, object]) -> dict[str, object]:
    summary = report.get("actor_update_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("actor update summary is missing")
    result = dict(summary)
    for name in (*COUNT_KEYS, "actor_return_included_transition_count"):
        value = result.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or int(value) != float(value)
            or int(value) < 0
        ):
            raise ValueError(f"actor update count {name} is invalid")
        result[name] = int(value)
    for name in FINITE_SUMMARY_KEYS:
        value = result.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"actor update metric {name} is invalid")
        result[name] = float(value)
    return result


def compare_probe_reports(
    control: Mapping[str, object], treatment: Mapping[str, object]
) -> dict[str, object]:
    """Validate common physics and classify the semantic gradient change."""

    for name, report in (("control", control), ("treatment", treatment)):
        if (
            report.get("valid") is not True
            or report.get("input_step") != START_STEP
            or report.get("ordinary_training_loop_entered") is not False
            or report.get("compiled_callable_reused") is not True
            or report.get("full_state_exact") is not True
            or report.get("metrics_exact") is not True
        ):
            raise ValueError(f"{name} determinism probe is invalid")
    if control.get("input_state_sha256") != treatment.get("input_state_sha256"):
        raise ValueError("paired probes did not start from one exact state")

    physical_matches = {
        name: _probe_boundary(control, name) == _probe_boundary(treatment, name)
        for name in PHYSICAL_BOUNDARIES
    }
    if not all(physical_matches.values()):
        raise ValueError("paired probes do not share one physical rollout")

    control_summary = _validated_summary(control)
    treatment_summary = _validated_summary(treatment)
    for name in COUNT_KEYS:
        if control_summary[name] != treatment_summary[name]:
            raise ValueError(f"paired physical count {name} differs")
    for name in (
        "actor_return_post_first_done_reward_sum",
        "actor_return_post_first_done_reward_mean",
    ):
        if control_summary[name] != treatment_summary[name]:
            raise ValueError(f"paired physical reward metric {name} differs")

    post_reset_steps = control_summary["actor_return_post_first_done_transition_count"]
    control_included = control_summary["actor_return_included_transition_count"]
    treatment_included = treatment_summary["actor_return_included_transition_count"]
    if control_included != TRANSITIONS_PER_UPDATE:
        raise ValueError("multi-episode control does not include the full rollout")
    if treatment_included + post_reset_steps != TRANSITIONS_PER_UPDATE:
        raise ValueError("first-terminal prefix accounting does not close")

    control_actor_update = _probe_boundary(control, "actor_cagrad")
    treatment_actor_update = _probe_boundary(treatment, "actor_cagrad")
    actor_update_changed = control_actor_update != treatment_actor_update
    has_reset_contamination = post_reset_steps > 0
    outcome = (
        "first-terminal-changes-gradient"
        if has_reset_contamination and actor_update_changed
        else "first-terminal-no-gradient-change"
        if has_reset_contamination
        else "no-within-horizon-reset-contamination"
    )
    control_return = control_summary["actor_return_mean"]
    treatment_return = treatment_summary["actor_return_mean"]
    return_delta = control_return - treatment_return
    return_fraction = return_delta / control_return if control_return != 0.0 else None
    return {
        "protocol": "g1-e002-first-terminal-gradient-pair-v1",
        "valid": True,
        "outcome": outcome,
        "source_step": START_STEP,
        "effective_num_envs": EFFECTIVE_NUM_ENVS,
        "unroll_length": UNROLL_LENGTH,
        "physical_boundaries_match": physical_matches,
        "actor_update_changed": actor_update_changed,
        "post_first_done_transition_count": post_reset_steps,
        "post_first_done_env_count": control_summary[
            "actor_return_post_first_done_env_count"
        ],
        "multi_episode_return_mean": control_return,
        "first_terminal_return_mean": treatment_return,
        "post_reset_return_contribution_mean": return_delta,
        "post_reset_return_fraction": return_fraction,
        "control": control_summary,
        "treatment": treatment_summary,
        "gradient_ratios": {
            name: (
                treatment_summary[name] / control_summary[name]
                if control_summary[name] != 0.0
                else None
            )
            for name in (
                "actor_grad",
                "actor_update_norm",
                "actor_grad_population_mean_norm",
                "actor_grad_population_rms_norm",
                "actor_grad_population_variance_trace",
                "actor_grad_population_noise_scale",
                "actor_grad_population_esnr",
            )
        },
        "optimizer_update_retained": False,
        "retained_policy": None,
    }


def _validate_arm_hparams(
    run_directory: Path, *, semantics: str, probe_output: Path
) -> dict[str, object]:
    hparams_path = run_directory / "hparams.json"
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    expected = {
        "total_steps": END_STEP,
        "checkpoint_steps": [END_STEP],
        "effective_num_envs": EFFECTIVE_NUM_ENVS,
        "num_envs": 256,
        "gradient_accumulation_steps": 2,
        "unroll_length": UNROLL_LENGTH,
        "actor_cagrad": True,
        "actor_per_env_grad_clip": 1.0,
        "tracking_anchor_position_kernel": "exponential",
        "tracking_root_velocity_weight": 1.0,
        "actor_bootstrap_scale": 0.0,
        "ahac": False,
        "actor_return_semantics": semantics,
        "allow_resume_actor_return_semantics_change": (semantics == "first_terminal"),
        "determinism_probe": True,
        "determinism_probe_output": str(probe_output.resolve()),
    }
    if any(hparams.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{semantics} probe hparams are invalid")
    return {
        "run_directory": str(run_directory.resolve()),
        "hparams_sha256": sha256_file(hparams_path),
        "probe_sha256": sha256_file(probe_output),
    }


def _run_worker(args: argparse.Namespace) -> int:
    if args.worker_semantics not in SEMANTICS:
        raise ValueError("worker semantics are required")
    repository = Path(__file__).resolve().parents[2]
    source_checkpoint = args.resume_from.resolve()
    reference_path = args.reference_path.resolve()
    probe_output = args.probe_output.resolve()
    validate_preflight(
        repository=repository,
        checkpoint=source_checkpoint,
        reference=reference_path,
        code_commit=args.code_commit,
    )
    kwargs = build_probe_kwargs(
        args.solver_profile,
        reference_path,
        args.seed,
        source_checkpoint,
        semantics=args.worker_semantics,
        probe_output=probe_output,
    )
    configure_jax()
    worker_root = args.output_root.resolve() / args.worker_semantics
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
        semantics=args.worker_semantics,
        probe_output=probe_output,
    )
    arm.update(
        protocol="g1-e002-first-terminal-arm-v1",
        valid=True,
        semantics=args.worker_semantics,
    )
    _write_json_atomically(
        args.output_root.resolve() / f"{args.worker_semantics}_arm.json",
        arm,
    )
    del kwargs
    gc.collect()
    jax.clear_caches()
    return 0


def _worker_command(
    args: argparse.Namespace, *, semantics: str, probe_output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.g1_e002_first_terminal_audit.run",
        "--worker-semantics",
        semantics,
        "--solver-profile",
        args.solver_profile,
        "--reference-path",
        str(args.reference_path.resolve()),
        "--resume-from",
        str(args.resume_from.resolve()),
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
    repository = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-e002-first-terminal-preflight-v1",
        source_step=START_STEP,
        effective_num_envs=EFFECTIVE_NUM_ENVS,
        unroll_length=UNROLL_LENGTH,
        semantics=list(SEMANTICS),
        deterministic_invocations_per_arm=2,
        optimizer_update_retained=False,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    probe_paths = {
        semantics: output_root / f"{semantics}_probe.json" for semantics in SEMANTICS
    }
    processes = []
    for semantics in SEMANTICS:
        command = _worker_command(
            args,
            semantics=semantics,
            probe_output=probe_paths[semantics],
        )
        completed = subprocess.run(command, check=False)
        processes.append(
            {
                "semantics": semantics,
                "command": command,
                "returncode": completed.returncode,
            }
        )
        if completed.returncode != 0:
            _write_json_atomically(
                output_root / "completion.json",
                {
                    "protocol": "g1-e002-first-terminal-completion-v1",
                    "valid": False,
                    "outcome": "invalid-execution",
                    "processes": processes,
                    "optimizer_update_retained": False,
                    "retained_policy": None,
                },
            )
            return completed.returncode
    _write_json_atomically(output_root / "worker_processes.json", processes)

    reports = {
        semantics: json.loads(path.read_text(encoding="utf-8"))
        for semantics, path in probe_paths.items()
    }
    comparison = compare_probe_reports(
        reports["multi_episode"], reports["first_terminal"]
    )
    comparison.update(
        code_commit=args.code_commit,
        source_checkpoint_sha256=sha256_file(args.resume_from.resolve()),
        probe_sha256={
            semantics: sha256_file(path) for semantics, path in probe_paths.items()
        },
        arm_manifests={
            semantics: json.loads(
                (output_root / f"{semantics}_arm.json").read_text(encoding="utf-8")
            )
            for semantics in SEMANTICS
        },
    )
    comparison_path = output_root / "comparison.json"
    _write_json_atomically(comparison_path, comparison)
    completion = {
        "protocol": "g1-e002-first-terminal-completion-v1",
        "valid": True,
        "outcome": comparison["outcome"],
        "optimizer_update_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "worker_processes.json": sha256_file(output_root / "worker_processes.json"),
            "comparison.json": sha256_file(comparison_path),
            **{
                f"{semantics}_probe.json": sha256_file(path)
                for semantics, path in probe_paths.items()
            },
            **{
                f"{semantics}_arm.json": sha256_file(
                    output_root / f"{semantics}_arm.json"
                )
                for semantics in SEMANTICS
            },
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-semantics", choices=SEMANTICS, help=argparse.SUPPRESS)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("first-terminal audit seed must equal zero")
    if args.worker_semantics is not None and args.probe_output is None:
        raise ValueError("worker probe output is required")
    raise SystemExit(
        _run_worker(args) if args.worker_semantics is not None else _run_parent(args)
    )


if __name__ == "__main__":
    main()
