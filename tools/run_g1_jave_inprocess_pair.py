"""Run a same-process, cloned-branch JAVE discriminator from retained G1 E002."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


SOURCE_STEP = 1_867_776
TRANSITIONS_PER_UPDATE = 512 * 24
WARMUP_STEP = SOURCE_STEP + TRANSITIONS_PER_UPDATE
JAVE_VG_WEIGHT = 0.1
PHASES = (0, 25, 50, 75, 100)
RETAINED_E002_SURVIVAL = (136, 144, 84, 90, 79)


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_branch_checkpoint_steps(active_updates: int) -> tuple[int, ...]:
    if (
        isinstance(active_updates, bool)
        or not isinstance(active_updates, int)
        or active_updates < 1
    ):
        raise ValueError("active_updates must be a positive integer")
    return tuple(
        WARMUP_STEP + index * TRANSITIONS_PER_UPDATE
        for index in range(1, active_updates + 1)
    )


def build_branch_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    *,
    enabled: bool,
    active_updates: int,
) -> dict[str, Any]:
    """Resume the common JAVE warm-up and change only its dynamic weight."""

    from tools.run_g1_jave_continuation import build_jave_kwargs

    steps = expected_branch_checkpoint_steps(active_updates)
    kwargs = build_jave_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        enabled=enabled,
        updates=active_updates + 1,
        warmup_updates=0,
    )
    kwargs.update(
        checkpoint_steps=steps,
        total_steps=steps[-1],
        allow_resume_jave_weight_change=True,
    )
    return kwargs


def compare_control_repeats(
    control_a: dict[str, object], control_b: dict[str, object]
) -> dict[str, object]:
    """Require exact numeric states and telemetry from two control branches."""

    steps_a = control_a.get("checkpoint_steps")
    steps_b = control_b.get("checkpoint_steps")
    hashes_a = control_a.get("state_sha256")
    hashes_b = control_b.get("state_sha256")
    telemetry_a = control_a.get("telemetry")
    telemetry_b = control_b.get("telemetry")
    structurally_valid = (
        isinstance(steps_a, list)
        and steps_a == steps_b
        and isinstance(hashes_a, list)
        and isinstance(hashes_b, list)
        and len(hashes_a) == len(steps_a)
        and len(hashes_b) == len(steps_a)
        and isinstance(telemetry_a, list)
        and isinstance(telemetry_b, list)
        and len(telemetry_a) == len(steps_a)
        and len(telemetry_b) == len(steps_a)
    )
    first_mismatch_step = None
    if structurally_valid:
        for index, step in enumerate(steps_a):
            if (
                hashes_a[index] != hashes_b[index]
                or telemetry_a[index] != telemetry_b[index]
            ):
                first_mismatch_step = step
                break
    valid = structurally_valid and first_mismatch_step is None
    return {
        "protocol": "g1-jave-inprocess-control-repeatability-v1",
        "valid": valid,
        "checkpoint_steps": steps_a if isinstance(steps_a, list) else None,
        "state_hashes_match": bool(
            structurally_valid and hashes_a == hashes_b
        ),
        "telemetry_matches": bool(
            structurally_valid and telemetry_a == telemetry_b
        ),
        "first_mismatch_step": first_mismatch_step,
    }


def _validate_survival(vector: object) -> list[int]:
    if (
        not isinstance(vector, (list, tuple))
        or len(vector) != len(PHASES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in vector
        )
    ):
        raise ValueError("phase survival vector is invalid")
    return list(vector)


def classify_pair(
    control: dict[int, list[int]],
    treatment: dict[int, list[int]],
    *,
    checkpoint_steps: tuple[int, ...],
    retained_survival: list[int] | tuple[int, ...],
) -> dict[str, object]:
    """Select JAVE only when it preserves its control and retained E002."""

    if (
        not checkpoint_steps
        or set(control) != set(checkpoint_steps)
        or set(treatment) != set(checkpoint_steps)
    ):
        raise ValueError("paired checkpoint grid is invalid")
    retained = _validate_survival(retained_survival)
    records: list[dict[str, object]] = []
    for step in checkpoint_steps:
        control_vector = _validate_survival(control[step])
        treatment_vector = _validate_survival(treatment[step])
        delta = [
            candidate - baseline
            for candidate, baseline in zip(
                treatment_vector, control_vector, strict=True
            )
        ]
        preserves_control = all(value >= 0 for value in delta)
        improves_control = any(value > 0 for value in delta)
        preserves_retained = all(
            candidate >= baseline
            for candidate, baseline in zip(
                treatment_vector, retained, strict=True
            )
        )
        records.append(
            {
                "checkpoint_step": step,
                "control_survival": control_vector,
                "treatment_survival": treatment_vector,
                "treatment_minus_control": delta,
                "preserves_control": preserves_control,
                "improves_control": improves_control,
                "preserves_retained_e002": preserves_retained,
                "eligible": (
                    preserves_control
                    and improves_control
                    and preserves_retained
                ),
                "minimum": min(treatment_vector),
                "median": float(statistics.median(treatment_vector)),
                "mean": float(statistics.fmean(treatment_vector)),
            }
        )
    eligible = [record for record in records if record["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda record: (
                record["minimum"],
                record["median"],
                record["mean"],
                -record["checkpoint_step"],
            ),
        )
        if eligible
        else None
    )
    any_gain = any(record["improves_control"] for record in records)
    diagnostic = max(
        records,
        key=lambda record: (
            record["minimum"],
            record["median"],
            record["mean"],
            sum(record["treatment_minus_control"]),
            -record["checkpoint_step"],
        ),
    )
    return {
        "protocol": "g1-jave-inprocess-pair-selection-v1",
        "phases": list(PHASES),
        "retained_e002_survival": retained,
        "checkpoints": records,
        "outcome": (
            "jave-late-advances"
            if selected is not None
            else "jave-late-mixed"
            if any_gain
            else "jave-late-no-gain"
        ),
        "selected_step": (
            selected["checkpoint_step"] if selected is not None else None
        ),
        "selected_survival": (
            selected["treatment_survival"] if selected is not None else None
        ),
        "policy_retained": selected is not None,
        "diagnostic_step": diagnostic["checkpoint_step"],
    }


def _finite_tree(tree: object) -> bool:
    import jax
    import numpy as np

    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            return False
    return True


def _load_state(path: Path):
    import pickle

    with path.open("rb") as stream:
        return pickle.load(stream)


def _run_train_stage(
    output_root: Path,
    kwargs: dict[str, Any],
    *,
    solver_profile: str,
) -> Path:
    import gc

    from src.algorithms.shac.algorithm import train
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(solver_profile)):
            result = train(**kwargs)
    finally:
        os.chdir(previous)
    relative_run_directory = result[1]
    del result
    gc.collect()
    return (output_root / relative_run_directory).resolve()


def _validate_branch_training(
    run_directory: Path,
    *,
    branch_checkpoint: Path,
    branch_checkpoint_sha256: str,
    branch_state_sha256: str,
    enabled: bool,
    active_updates: int,
) -> dict[str, object]:
    from src.algorithms.shac.counterfactual_wrench_distillation import (
        parameter_tree_sha256,
    )
    from src.algorithms.shac.frozen_controller_residual import (
        FrozenControllerResidualParams,
    )

    steps = expected_branch_checkpoint_steps(active_updates)
    if sha256_file(branch_checkpoint) != branch_checkpoint_sha256:
        raise ValueError("common JAVE branch checkpoint changed")
    branch_state = _load_state(branch_checkpoint)
    if parameter_tree_sha256(branch_state) != branch_state_sha256:
        raise ValueError("common JAVE branch state changed")
    parent_sha256 = parameter_tree_sha256(branch_state.actor_params.parent)
    normalizer_sha256 = parameter_tree_sha256(branch_state.normalizer)
    del branch_state

    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    expected_hparams = {
        "actor_bootstrap_scale": 1.0,
        "jave_vg_weight": JAVE_VG_WEIGHT if enabled else 0.0,
        "jave_vg_warmup_steps": 0,
        "jave_start_step": SOURCE_STEP,
        "jave_collect_transitions": True,
        "allow_resume_jave_weight_change": True,
        "total_steps": steps[-1],
        "checkpoint_steps": list(steps),
    }
    if any(hparams.get(key) != value for key, value in expected_hparams.items()):
        raise ValueError("paired JAVE branch hparams are invalid")

    state_hashes: list[str] = []
    checkpoint_hashes: list[str] = []
    for step in steps:
        checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
        state = _load_state(checkpoint)
        if (
            int(state.step) != step
            or not _finite_tree(state)
            or not isinstance(state.actor_params, FrozenControllerResidualParams)
            or parameter_tree_sha256(state.actor_params.parent) != parent_sha256
            or parameter_tree_sha256(state.normalizer) != normalizer_sha256
            or float(state.jave_vg_weight)
            != (JAVE_VG_WEIGHT if enabled else 0.0)
            or any(
                getattr(state, name, None) is None
                for name in ("ldm_params", "ldm_opt", "replay_buffer")
            )
        ):
            raise ValueError("paired JAVE checkpoint is invalid")
        state_hashes.append(parameter_tree_sha256(state))
        checkpoint_hashes.append(sha256_file(checkpoint))
        del state

    telemetry = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if [row.get("step") for row in telemetry] != list(steps):
        raise ValueError("paired JAVE telemetry grid is invalid")
    for row in telemetry:
        scalars = (
            row.get("jave_ldm_loss"),
            row.get("jave_vg_loss"),
            row.get("jave_vg_target_norm"),
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in scalars
            )
            or bool(row.get("jave_vg_active")) is not enabled
            or not isinstance(row.get("jave_replay_size"), int)
            or row["jave_replay_size"] < 2 * TRANSITIONS_PER_UPDATE
        ):
            raise ValueError("paired JAVE telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-jave-inprocess-branch-training-v1",
        "arm": "jave" if enabled else "control",
        "run_directory": str(run_directory),
        "branch_checkpoint_sha256": branch_checkpoint_sha256,
        "branch_state_sha256": branch_state_sha256,
        "checkpoint_steps": list(steps),
        "checkpoint_sha256": checkpoint_hashes,
        "state_sha256": state_hashes,
        "telemetry": telemetry,
    }


def _run_worker(args: argparse.Namespace) -> int:
    from src.algorithms.shac.counterfactual_wrench_distillation import (
        parameter_tree_sha256,
    )
    from tools.run_g1_jave_continuation import (
        build_jave_kwargs,
        validate_preflight,
        validate_training_artifacts,
    )
    from tools.run_g1_tracking_shac import configure_jax

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-jave-inprocess-pair-preflight-v1",
        authoritative_entrypoint=(
            "python -m tools.run_g1_jave_inprocess_pair"
        ),
        active_updates=args.active_updates,
        warmup_step=WARMUP_STEP,
        branch_checkpoint_steps=list(
            expected_branch_checkpoint_steps(args.active_updates)
        ),
        execution_shape="one-worker-one-gpu-common-warmup-cloned-branches",
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()

    warmup_kwargs = build_jave_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        enabled=False,
        updates=1,
        warmup_updates=0,
    )
    warmup_run = _run_train_stage(
        output_root / "warmup",
        warmup_kwargs,
        solver_profile=args.solver_profile,
    )
    warmup_validation = validate_training_artifacts(
        warmup_run,
        source_checkpoint=args.resume_from.resolve(),
        enabled=False,
        updates=1,
        warmup_updates=0,
    )
    _write_json_atomically(
        output_root / "warmup" / "training_validation.json",
        warmup_validation,
    )
    branch_checkpoint = warmup_run / f"checkpoint_step_{WARMUP_STEP}.pkl"
    branch_checkpoint_sha256 = sha256_file(branch_checkpoint)
    branch_state = _load_state(branch_checkpoint)
    branch_state_sha256 = parameter_tree_sha256(branch_state)
    del branch_state
    branch_report = {
        "protocol": "g1-jave-common-branch-point-v1",
        "valid": True,
        "checkpoint": str(branch_checkpoint),
        "checkpoint_step": WARMUP_STEP,
        "checkpoint_sha256": branch_checkpoint_sha256,
        "state_sha256": branch_state_sha256,
        "jave_vg_weight": 0.0,
        "jave_active": False,
    }
    _write_json_atomically(output_root / "branch_point.json", branch_report)

    validations: dict[str, dict[str, object]] = {}
    for arm in ("control_a", "control_b"):
        kwargs = build_branch_kwargs(
            args.solver_profile,
            args.reference_path.resolve(),
            args.seed,
            branch_checkpoint,
            enabled=False,
            active_updates=args.active_updates,
        )
        run_directory = _run_train_stage(
            output_root / arm,
            kwargs,
            solver_profile=args.solver_profile,
        )
        validation = _validate_branch_training(
            run_directory,
            branch_checkpoint=branch_checkpoint,
            branch_checkpoint_sha256=branch_checkpoint_sha256,
            branch_state_sha256=branch_state_sha256,
            enabled=False,
            active_updates=args.active_updates,
        )
        validations[arm] = validation
        _write_json_atomically(
            output_root / arm / "training_validation.json", validation
        )
    repeatability = compare_control_repeats(
        validations["control_a"], validations["control_b"]
    )
    _write_json_atomically(output_root / "repeatability.json", repeatability)

    if repeatability["valid"]:
        kwargs = build_branch_kwargs(
            args.solver_profile,
            args.reference_path.resolve(),
            args.seed,
            branch_checkpoint,
            enabled=True,
            active_updates=args.active_updates,
        )
        run_directory = _run_train_stage(
            output_root / "jave",
            kwargs,
            solver_profile=args.solver_profile,
        )
        validation = _validate_branch_training(
            run_directory,
            branch_checkpoint=branch_checkpoint,
            branch_checkpoint_sha256=branch_checkpoint_sha256,
            branch_state_sha256=branch_state_sha256,
            enabled=True,
            active_updates=args.active_updates,
        )
        validations["jave"] = validation
        _write_json_atomically(
            output_root / "jave" / "training_validation.json", validation
        )

    worker_result = {
        "protocol": "g1-jave-inprocess-training-pair-v1",
        "valid": bool(repeatability["valid"] and "jave" in validations),
        "branch_point": branch_report,
        "repeatability": repeatability,
        "arms": {
            name: {
                "run_directory": validation["run_directory"],
                "checkpoint_steps": validation["checkpoint_steps"],
                "state_sha256": validation["state_sha256"],
            }
            for name, validation in validations.items()
        },
    }
    _write_json_atomically(output_root / "training_pair.json", worker_result)
    return 0


def _phase_grid_command(
    *, checkpoint: Path, reference: Path, output: Path, code_commit: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.evaluate_g1_flax_phase_grid",
        "--checkpoint",
        str(checkpoint),
        "--reference-path",
        str(reference),
        "--output",
        str(output),
        "--phases",
        *(str(phase) for phase in PHASES),
        "--seed",
        "0",
        "--code-commit",
        code_commit,
    ]


def _render_command(
    *, checkpoint: Path, reference: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.evaluate_g1_tracking",
        "--checkpoint",
        str(checkpoint),
        "--reference-path",
        str(reference),
        "--output-dir",
        str(output),
        "--phase",
        "0",
        "--seed",
        "0",
        "--env-variant",
        "g1_tracking_rmr_50hz_action_parity",
        "--reference-stride",
        "1",
        "--actor-history-len",
        "10",
        "--actor-reference-lookahead-steps",
        "4",
        "8",
        "12",
        "--actor-reference-preview-mode",
        "delta",
        "--reference-residual-control",
        "--reference-residual-scale",
        "1.0",
        "--tracking-root-velocity-weight",
        "1.0",
        "--solver-profile",
        "g1-4x5",
        "--render-every",
        "2",
    ]


def _run_parent(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    worker_command = [
        sys.executable,
        "-m",
        "tools.run_g1_jave_inprocess_pair",
        "--worker",
        "--solver-profile",
        args.solver_profile,
        "--reference-path",
        str(args.reference_path.resolve()),
        "--resume-from",
        str(args.resume_from.resolve()),
        "--output-root",
        str(output_root),
        "--code-commit",
        args.code_commit,
        "--active-updates",
        str(args.active_updates),
        "--seed",
        str(args.seed),
    ]
    worker = subprocess.run(worker_command, check=False)
    _write_json_atomically(
        output_root / "worker_process.json",
        {
            "command": worker_command,
            "returncode": worker.returncode,
        },
    )
    training_pair_path = output_root / "training_pair.json"
    if worker.returncode != 0 or not training_pair_path.is_file():
        _write_json_atomically(
            output_root / "selection.json",
            {
                "protocol": "g1-jave-inprocess-pair-selection-v1",
                "outcome": "invalid-execution",
                "reason": "training worker failed",
                "policy_retained": False,
            },
        )
        return 1
    training_pair = json.loads(training_pair_path.read_text(encoding="utf-8"))
    if training_pair.get("valid") is not True:
        _write_json_atomically(
            output_root / "selection.json",
            {
                "protocol": "g1-jave-inprocess-pair-selection-v1",
                "outcome": "invalid-execution",
                "reason": "same-process control repeatability failed",
                "repeatability": training_pair.get("repeatability"),
                "policy_retained": False,
            },
        )
        return 0

    steps = expected_branch_checkpoint_steps(args.active_updates)
    survival: dict[str, dict[int, list[int]]] = {
        "control_a": {},
        "jave": {},
    }
    evaluation_artifacts: list[dict[str, object]] = []
    for arm in ("control_a", "jave"):
        run_directory = Path(training_pair["arms"][arm]["run_directory"])
        for step in steps:
            checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
            output = (
                output_root
                / "evaluation"
                / arm
                / f"checkpoint_step_{step}.json"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            command = _phase_grid_command(
                checkpoint=checkpoint,
                reference=args.reference_path.resolve(),
                output=output,
                code_commit=args.code_commit,
            )
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0 or not output.is_file():
                _write_json_atomically(
                    output_root / "selection.json",
                    {
                        "protocol": "g1-jave-inprocess-pair-selection-v1",
                        "outcome": "invalid-execution",
                        "reason": f"phase-grid evaluation failed for {arm} {step}",
                        "policy_retained": False,
                    },
                )
                return 1
            payload = json.loads(output.read_text(encoding="utf-8"))
            vector = _validate_survival(payload["summary"]["survival"])
            survival[arm][step] = vector
            evaluation_artifacts.append(
                {
                    "arm": arm,
                    "checkpoint_step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "phase_grid": str(output),
                    "phase_grid_sha256": sha256_file(output),
                    "survival": vector,
                }
            )

    selection = classify_pair(
        survival["control_a"],
        survival["jave"],
        checkpoint_steps=steps,
        retained_survival=RETAINED_E002_SURVIVAL,
    )
    selection.update(
        code_commit=args.code_commit,
        training_pair_sha256=sha256_file(training_pair_path),
        evaluation_artifacts=evaluation_artifacts,
    )
    diagnostic_step = int(selection["diagnostic_step"])
    render_artifacts = []
    for arm in ("control_a", "jave"):
        run_directory = Path(training_pair["arms"][arm]["run_directory"])
        checkpoint = run_directory / f"checkpoint_step_{diagnostic_step}.pkl"
        output = output_root / "evaluation" / arm / "diagnostic_render"
        command = _render_command(
            checkpoint=checkpoint,
            reference=args.reference_path.resolve(),
            output=output,
        )
        completed = subprocess.run(command, check=False)
        required = (
            output / "summary.json",
            output / "evaluation.npz",
            output / "evaluation.mp4",
            output / "contact_sheet.png",
        )
        if completed.returncode != 0 or any(not path.is_file() for path in required):
            selection.update(
                outcome="invalid-execution",
                reason=f"diagnostic render failed for {arm}",
                policy_retained=False,
                selected_step=None,
                selected_survival=None,
            )
            _write_json_atomically(output_root / "selection.json", selection)
            return 1
        render_artifacts.append(
            {
                "arm": arm,
                "checkpoint_step": diagnostic_step,
                "files": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in required
                ],
            }
        )
    selection["render_artifacts"] = render_artifacts
    _write_json_atomically(output_root / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--active-updates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("JAVE in-process pair seed must equal zero")
    expected_branch_checkpoint_steps(args.active_updates)
    raise SystemExit(_run_worker(args) if args.worker else _run_parent(args))


if __name__ == "__main__":
    main()
