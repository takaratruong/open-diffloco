"""Run paired true-scalar and blind torso-assistance continuations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import traceback


E012_SELECTED_STEP = 1_671_168
STEPS_PER_UPDATE = 6_144
ASSISTED_UPDATES = 96
ZERO_TAIL_UPDATES = 32
ASSISTANCE_END_STEP = E012_SELECTED_STEP + ASSISTED_UPDATES * STEPS_PER_UPDATE
CONTINUATION_END_STEP = ASSISTANCE_END_STEP + ZERO_TAIL_UPDATES * STEPS_PER_UPDATE
CHECKPOINT_INTERVAL = 8 * STEPS_PER_UPDATE
ZERO_ASSISTANCE_FRACTION = 0.25
RESUME_RANDOM_SEED = 2
E012_SELECTED_CHECKPOINT_SHA256 = (
    "f375cadc9bf8b5cef26fc7414133071910fed393344c99bbacffea963aa9f4f7"
)
FROZEN_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
FROZEN_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
FROZEN_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
E012_SELECTED_HPARAMS_SHA256 = (
    "76a78a6b1176f4d8cff785a8cbc01c0dd18e08de83ae7da61d3be093768f0d5f"
)
FROZEN_MODEL_PATH = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
FROZEN_CONTROLLER_PATH = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)
ZERO_TAIL_CHECKPOINT_STEPS = tuple(
    range(
        ASSISTANCE_END_STEP + CHECKPOINT_INTERVAL,
        CONTINUATION_END_STEP + 1,
        CHECKPOINT_INTERVAL,
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_code_provenance(expected_commit: str) -> dict[str, str]:
    """Require the exact registered clean code checkout."""
    repository = Path(__file__).resolve().parents[1]
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        raise ValueError("registered code commit must be a full Git SHA-1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError(
            f"registered code commit mismatch: expected {expected_commit}, got {head}"
        )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "src", "tools"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "tools",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if diff or untracked:
        raise ValueError("registered code checkout has executable dirty changes")
    return {
        "repository": str(repository),
        "code_commit": head,
        "dirty_patch_sha256": hashlib.sha256(diff).hexdigest(),
    }


def validate_parent_checkpoint(path: str | Path) -> Path:
    """Require the exact selected E012 checkpoint before launching compute."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"E012 selected checkpoint is missing: {checkpoint}")
    actual = _sha256(checkpoint)
    if actual != E012_SELECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "E012 selected checkpoint SHA-256 mismatch: "
            f"expected {E012_SELECTED_CHECKPOINT_SHA256}, got {actual}"
        )
    hparams_path = checkpoint.parent / "hparams.json"
    if not hparams_path.is_file():
        raise ValueError("E012 selected checkpoint sibling hparams.json is missing")
    hparams_sha256 = _sha256(hparams_path)
    if hparams_sha256 != E012_SELECTED_HPARAMS_SHA256:
        raise ValueError(
            "E012 selected hparams SHA-256 mismatch: "
            f"expected {E012_SELECTED_HPARAMS_SHA256}, got {hparams_sha256}"
        )
    return checkpoint


def validate_parent_hparams(
    checkpoint: str | Path,
    *,
    reference_path: str | Path,
    solver_profile: str,
) -> dict[str, object]:
    """Require resumed metadata to select the registered task and solver."""
    path = Path(checkpoint).resolve().parent / "hparams.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "reference_path": str(Path(reference_path).resolve()),
        "reference_sha256": FROZEN_REFERENCE_SHA256,
        "xml_path": str(FROZEN_MODEL_PATH.resolve()),
        "solver_profile": solver_profile,
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
        "env_variant": "g1_tracking_rmr_50hz_source_step",
        "actor_history_len": 10,
        "actor_reference_lookahead_steps": [4, 8, 12],
        "actor_reference_preview_mode": "delta",
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"E012 selected hparams contract mismatch: {mismatches}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "validated": expected,
    }


def run_assistance_conditioning_migration_preflight(
    checkpoint: str | Path,
) -> dict[str, object]:
    """Run the real checkpoint action-equivalence audit on CPU in isolation."""
    repository = Path(__file__).resolve().parents[1]
    program = r'''
import json
import pickle
import sys
import jax.numpy as jp
import numpy as np
from src.algorithms.shac.residual_preview_adapter import (
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
    migrate_residual_adapter_assistance_conditioning,
)
from src.core.networks import Actor

checkpoint = sys.argv[1]
with open(checkpoint, "rb") as stream:
    state = pickle.load(stream)
candidate_params, candidate_opt, report = (
    migrate_residual_adapter_assistance_conditioning(
        params=state.actor_params,
        optimizer_state=state.actor_opt,
        expected_input_dim=328,
    )
)
frames = state.env_state.obs[:1].reshape(1, 10, 328)
normalized = (
    (frames - state.normalizer.mean)
    / jp.sqrt(state.normalizer.var + 1e-4)
).reshape(1, 3280).astype(jp.float32)
actor = Actor(
    29,
    hidden=(512, 256, 128),
    squash=True,
    layer_norm=True,
    zero_output=False,
)
adapter = PreviewResidualAdapter(action_dim=29, hidden_dim=256)
legacy_action, _, _ = apply_frozen_preview_residual(
    actor,
    adapter,
    state.actor_params,
    normalized,
    history_len=10,
    treatment_frame_dim=328,
)
candidate_action, _, _ = apply_frozen_preview_residual(
    actor,
    adapter,
    candidate_params,
    normalized,
    history_len=10,
    treatment_frame_dim=328,
    assistance_scale=jp.asarray(0.0, dtype=jp.float32),
)
report["zero_scale_action_exact"] = bool(
    np.array_equal(np.asarray(legacy_action), np.asarray(candidate_action))
)
report["valid"] = bool(report["valid"] and report["zero_scale_action_exact"])
print(json.dumps(report, sort_keys=True))
'''
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
            "PYTHONPATH": str(repository),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(Path(checkpoint).resolve())],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise ValueError("migration preflight did not emit exactly one audit")
    report = json.loads(lines[0])
    if report.get("valid") is not True:
        raise ValueError("migration preflight action equivalence failed")
    return report


def validate_runtime_assets(reference_path: str | Path) -> dict[str, str]:
    """Bind the exact reference, robot XML, and controller archive."""
    bindings = (
        ("reference", Path(reference_path).resolve(), FROZEN_REFERENCE_SHA256),
        ("model", FROZEN_MODEL_PATH.resolve(), FROZEN_MODEL_SHA256),
        ("controller", FROZEN_CONTROLLER_PATH.resolve(), FROZEN_CONTROLLER_SHA256),
    )
    report: dict[str, str] = {}
    for name, path, expected in bindings:
        if not path.is_file():
            raise ValueError(f"runtime {name} asset is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"runtime {name} SHA-256 mismatch: expected {expected}, got {actual}"
            )
        report[f"{name}_path"] = str(path)
        report[f"{name}_sha256"] = actual
    return report


def available_nvidia_device_ids() -> tuple[str, ...]:
    """Enumerate physical NVIDIA device indices without initializing JAX."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = tuple(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    if not devices:
        raise ValueError("no NVIDIA GPU devices are available")
    return devices


def nvidia_hardware_provenance(devices: tuple[str, str]) -> list[dict[str, str]]:
    """Record immutable-enough physical GPU identity before worker isolation."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = {}
    for line in result.stdout.splitlines():
        fields = tuple(part.strip() for part in line.split(",", 3))
        if len(fields) == 4:
            rows[fields[0]] = {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": fields[3],
            }
    if any(device not in rows for device in devices):
        raise ValueError("GPU hardware provenance is incomplete")
    selected = [rows[device] for device in devices]
    if selected[0]["uuid"] == selected[1]["uuid"]:
        raise ValueError("paired arms resolved to the same physical GPU")
    return selected


def validate_device_ids(
    aware_device: str,
    blind_device: str,
    *,
    available: tuple[str, ...],
) -> tuple[str, str]:
    """Require two distinct enumerated physical GPUs."""
    requested = (str(aware_device), str(blind_device))
    if requested[0] == requested[1]:
        raise ValueError("paired arms require distinct GPU devices")
    missing = tuple(device for device in requested if device not in available)
    if missing:
        raise ValueError(f"paired arms requested unavailable GPU devices: {missing}")
    return requested


def validate_zero_tail_checkpoints(
    paths: list[Path] | tuple[Path, ...],
    *,
    require_files: bool = True,
) -> tuple[Path, ...]:
    """Require exactly the registered final zero-assistance checkpoint grid."""
    candidates = tuple(Path(path) for path in paths)
    expected_names = tuple(
        f"checkpoint_step_{step}.pkl" for step in ZERO_TAIL_CHECKPOINT_STEPS
    )
    if tuple(path.name for path in candidates) != expected_names:
        raise ValueError("zero-tail checkpoint grid does not match registration")
    if require_files and any(not path.is_file() for path in candidates):
        raise ValueError("zero-tail checkpoint grid contains missing files")
    return candidates


def select_zero_tail_checkpoint(
    payloads: dict[int, dict],
) -> dict[str, object]:
    """Select exact-zero evaluation by minimum, median, mean, then earliest."""
    if not payloads:
        raise ValueError("zero-tail selection window is empty")
    records = []
    for step, payload in sorted(payloads.items()):
        if payload.get("actor_assistance_conditioning_scale") != 0.0:
            raise ValueError("selection requires exact-zero assistance conditioning")
        if payload.get("reference_sha256") != FROZEN_REFERENCE_SHA256:
            raise ValueError("selection reference provenance mismatch")
        if payload.get("solver_profile") != "g1-4x5":
            raise ValueError("selection solver provenance mismatch")
        checkpoint_sha256 = payload.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError("selection checkpoint SHA-256 is invalid")
        summary = payload.get("summary", {})
        if summary.get("phases") != [0, 100, 200, 300, 400]:
            raise ValueError("selection phase grid is invalid")
        survival = summary.get("survival")
        if (
            not isinstance(survival, list)
            or len(survival) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in survival
            )
        ):
            raise ValueError("selection survival vector is invalid")
        records.append(
            {
                "step": int(step),
                "checkpoint_sha256": checkpoint_sha256,
                "survival": survival,
                "minimum": min(survival),
                "median": float(statistics.median(survival)),
                "mean": float(statistics.fmean(survival)),
            }
        )
    selected = max(
        records,
        key=lambda row: (
            row["minimum"],
            row["median"],
            row["mean"],
            -row["step"],
        ),
    )
    return {
        "protocol": "g1-zero-tail-minimum-median-mean-earliest-v1",
        "assistance_scale": 0.0,
        "records": records,
        "selected_step": selected["step"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_survival": selected["survival"],
        "selected_key": [
            selected["minimum"],
            selected["median"],
            selected["mean"],
        ],
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_zero_tail_training_artifacts(
    training_directory: str | Path,
    *,
    observed: bool,
) -> tuple[Path, ...]:
    """Require final checkpoints and exact-zero training telemetry."""
    directory = Path(training_directory).resolve()
    hparams = json.loads((directory / "hparams.json").read_text(encoding="utf-8"))
    expected = {
        "total_steps": CONTINUATION_END_STEP,
        "torso_wrench_assistance_start_step": E012_SELECTED_STEP,
        "torso_wrench_assistance_end_step": ASSISTANCE_END_STEP,
        "torso_wrench_assistance_zero_fraction": ZERO_ASSISTANCE_FRACTION,
        "torso_wrench_assistance_continuous": True,
        "actor_torso_wrench_assistance_conditioning": True,
        "actor_observe_torso_wrench_assistance": observed,
    }
    mismatches = {
        key: {"expected": value, "actual": hparams.get(key)}
        for key, value in expected.items()
        if hparams.get(key) != value
    }
    if mismatches:
        raise ValueError(f"zero-tail training hparams mismatch: {mismatches}")
    migration = json.loads(
        (directory / "assistance_conditioning_migration.json").read_text(
            encoding="utf-8"
        )
    )
    if migration.get("valid") is not True or migration.get(
        "zero_scale_action_exact"
    ) is not True:
        raise ValueError("zero-tail conditioning migration is invalid")
    metrics = json.loads(
        (directory / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    rows = {int(row["step"]): row for row in metrics}
    for step in ZERO_TAIL_CHECKPOINT_STEPS:
        row = rows.get(step)
        if row is None:
            raise ValueError(f"zero-tail checkpoint telemetry missing step {step}")
        if (
            row.get("torso_wrench_assistance_scale_current") != 0.0
            or row.get("torso_wrench_assistance_active_fraction") != 0.0
            or row.get("torso_wrench_assistance_valid") is not True
        ):
            raise ValueError(f"checkpoint {step} is not exact-zero assistance")
    checkpoints = tuple(
        directory / f"checkpoint_step_{step}.pkl"
        for step in ZERO_TAIL_CHECKPOINT_STEPS
    )
    return validate_zero_tail_checkpoints(checkpoints)


def evaluate_zero_tail_arm(
    *,
    arm: str,
    training_directory: str | Path,
    observed: bool,
    device: str,
    reference_path: str | Path,
) -> dict[str, object]:
    """Evaluate and select every eligible exact-zero checkpoint for one arm."""
    repository = Path(__file__).resolve().parents[1]
    directory = Path(training_directory).resolve()
    checkpoints = validate_zero_tail_training_artifacts(
        directory, observed=observed
    )
    evaluation_root = directory.parents[1] / "evaluations"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device),
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONPATH": str(repository),
        }
    )
    payloads: dict[int, dict] = {}
    for step, checkpoint in zip(
        ZERO_TAIL_CHECKPOINT_STEPS, checkpoints, strict=True
    ):
        output_directory = evaluation_root / f"checkpoint_step_{step}"
        output_directory.mkdir(parents=True, exist_ok=False)
        output = output_directory / "phase_grid_summary.json"
        command = [
            sys.executable,
            str(repository / "tools/evaluate_g1_flax_phase_grid.py"),
            "--checkpoint",
            str(checkpoint),
            "--reference-path",
            str(Path(reference_path).resolve()),
            "--output",
            str(output),
            "--actor-reference-preview-mode",
            "delta",
            "--actor-residual-preview-adapter",
            "--actor-residual-preview-hidden",
            "256",
            "--solver-profile",
            "g1-4x5",
        ]
        with (
            (output_directory / "stdout.log").open("wb") as stdout,
            (output_directory / "stderr.log").open("wb") as stderr,
        ):
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{arm} zero-tail evaluator failed at step {step}: "
                f"return code {completed.returncode}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("checkpoint_path") != str(checkpoint):
            raise ValueError("zero-tail evaluator checkpoint path mismatch")
        if payload.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("zero-tail evaluator checkpoint hash mismatch")
        payloads[step] = payload
    selection = select_zero_tail_checkpoint(payloads)
    selection.update(
        {
            "arm": arm,
            "observed": observed,
            "training_directory": str(directory),
            "evaluation_root": str(evaluation_root),
        }
    )
    _write_json_atomic(directory / "dense_checkpoint_selection.json", selection)
    return selection


def render_selected_arm(
    *,
    selection: dict[str, object],
    device: str,
    reference_path: str | Path,
) -> str:
    """Produce trajectory, metrics, video, and montage for all selected phases."""
    repository = Path(__file__).resolve().parents[1]
    directory = Path(str(selection["training_directory"])).resolve()
    step = int(selection["selected_step"])
    checkpoint = directory / f"checkpoint_step_{step}.pkl"
    if _sha256(checkpoint) != selection["selected_checkpoint_sha256"]:
        raise ValueError("selected rendering checkpoint SHA-256 mismatch")
    artifact_root = directory.parents[1] / "selected_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device),
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "MUJOCO_GL": "egl",
            "PYTHONPATH": str(repository),
        }
    )
    for phase in (0, 100, 200, 300, 400):
        phase_root = artifact_root / f"phase_{phase:03d}"
        command = [
            sys.executable,
            str(repository / "tools/evaluate_g1_tracking.py"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(phase_root),
            "--seed",
            "0",
            "--phase",
            str(phase),
            "--render-every",
            "2",
            "--env-variant",
            "g1_tracking_rmr_50hz_source_step",
            "--solver-iterations",
            "4",
            "--solver-ls-iterations",
            "5",
            "--solver-profile",
            "g1-4x5",
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
            "0.5",
            "--reference-path",
            str(Path(reference_path).resolve()),
            "--reference-stride",
            "1",
        ]
        with (
            (artifact_root / f"phase_{phase:03d}.stdout.log").open("wb") as stdout,
            (artifact_root / f"phase_{phase:03d}.stderr.log").open("wb") as stderr,
        ):
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"selected artifact rendering failed at phase {phase}: "
                f"return code {completed.returncode}"
            )
        required = (
            phase_root / "evaluation.npz",
            phase_root / "summary.json",
            phase_root / "evaluation.mp4",
            phase_root / "contact_sheet.png",
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise ValueError(f"selected phase {phase} artifacts are incomplete")
        summary = json.loads(required[1].read_text(encoding="utf-8"))
        phase_index = (0, 100, 200, 300, 400).index(phase)
        if (
            summary.get("evaluation_start_phase") != phase
            or summary.get("reference_sha256") != FROZEN_REFERENCE_SHA256
            or summary.get("checkpoint_sha256")
            != selection["selected_checkpoint_sha256"]
            or summary.get("solver_profile") != "g1-4x5"
            or summary.get("actor_assistance_conditioning_scale") != 0.0
            or summary.get("steps")
            != selection["selected_survival"][phase_index]
        ):
            raise ValueError("selected artifact summary provenance mismatch")
    return str(artifact_root)


def compare_pair_selections(
    aware: dict[str, object], blind: dict[str, object]
) -> dict[str, object]:
    """Compare paired zero-wrench selectors against the E012 robust key."""
    aware_key = tuple(aware["selected_key"])
    blind_key = tuple(blind["selected_key"])
    baseline_key = (51, 66.0, 68.6)
    if aware_key > blind_key and aware_key > baseline_key:
        outcome = "scalar-observability-advances"
    elif max(aware_key, blind_key) > baseline_key:
        outcome = "curriculum-advances-without-observability-edge"
    else:
        outcome = "finite-no-gain"
    return {
        "protocol": "g1-assistance-observability-pair-comparison-v1",
        "aware": aware,
        "blind": blind,
        "e012_key": list(baseline_key),
        "outcome": outcome,
    }


def complete_worker_results(
    results: list[tuple[str, bool, str]],
    workers,
) -> list[tuple[str, bool, str]]:
    """Convert a worker crash before queue publication into an explicit failure."""
    completed = list(results)
    reported = {result[0] for result in completed}
    for arm, process in workers:
        if arm not in reported:
            completed.append(
                (
                    arm,
                    False,
                    f"worker exited with code {process.exitcode} without a report",
                )
            )
    return completed


def build_assistance_observability_pair_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> tuple[dict, dict]:
    """Return paired contracts differing only in scalar observability."""
    from tools.run_g1_frozen_residual_preview_continuation import (
        build_frozen_residual_preview_kwargs,
    )

    common = build_frozen_residual_preview_kwargs(
        profile_name,
        Path(reference_path),
        seed,
        Path(resume_from),
    )
    common.update(
        total_steps=CONTINUATION_END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        resume_random_seed=RESUME_RANDOM_SEED,
        torso_wrench_assistance=True,
        torso_wrench_assistance_start_step=E012_SELECTED_STEP,
        torso_wrench_assistance_end_step=ASSISTANCE_END_STEP,
        torso_wrench_assistance_zero_fraction=ZERO_ASSISTANCE_FRACTION,
        torso_wrench_assistance_continuous=True,
        actor_torso_wrench_assistance_conditioning=True,
        allow_resume_torso_wrench_assistance_change=True,
        allow_resume_assistance_conditioning_change=True,
    )
    aware = {**common, "actor_observe_torso_wrench_assistance": True}
    blind = {**common, "actor_observe_torso_wrench_assistance": False}
    return aware, blind


def validate_pair_delta(aware: dict, blind: dict) -> None:
    """Fail closed unless observability is the sole paired difference."""
    if set(aware) != set(blind):
        raise ValueError("paired assistance contracts have different keys")
    changed = {key for key in aware if aware[key] != blind[key]}
    if changed != {"actor_observe_torso_wrench_assistance"}:
        raise ValueError(f"invalid paired assistance delta: {sorted(changed)}")
    if aware["actor_observe_torso_wrench_assistance"] is not True:
        raise ValueError("aware arm must observe the assistance scalar")
    if blind["actor_observe_torso_wrench_assistance"] is not False:
        raise ValueError("blind arm must receive exact-zero conditioning")


def _run_arm(
    *,
    arm: str,
    observed: bool,
    device: str,
    profile_name: str,
    reference_path: str,
    seed: int,
    resume_from: str,
    output_root: str,
    result_queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        from src.algorithms.shac.algorithm import train
        from src.envs.g1_tracking.solver_profiles import (
            get_solver_profile,
            solver_context,
        )
        from tools.run_g1_tracking_shac import configure_jax

        configure_jax()
        import jax

        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError(
                "paired worker must see exactly one JAX GPU device"
            )
        aware, blind = build_assistance_observability_pair_kwargs(
            profile_name, reference_path, seed, resume_from
        )
        kwargs = aware if observed else blind
        arm_root = (Path(output_root) / arm).resolve()
        arm_root.mkdir(parents=True, exist_ok=False)
        profile = get_solver_profile(profile_name)
        previous_directory = Path.cwd()
        try:
            os.chdir(arm_root)
            with solver_context(profile):
                _, relative_save_dir = train(**kwargs)
        finally:
            os.chdir(previous_directory)
        result_queue.put(
            (arm, True, str((arm_root / relative_save_dir).resolve()))
        )
    except BaseException:
        result_queue.put((arm, False, traceback.format_exc()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_assistance_observability_pair_runs"),
    )
    parser.add_argument("--aware-device", default="0")
    parser.add_argument("--blind-device", default="1")
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    code_provenance = validate_code_provenance(args.code_commit)
    devices = validate_device_ids(
        args.aware_device,
        args.blind_device,
        available=available_nvidia_device_ids(),
    )
    hardware = nvidia_hardware_provenance(devices)
    checkpoint = validate_parent_checkpoint(args.resume_from)
    if args.reference_path is None:
        from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH

        reference_path = Path(DEFAULT_REFERENCE_PATH).resolve()
    else:
        reference_path = args.reference_path.resolve()
    assets = validate_runtime_assets(reference_path)
    parent_hparams = validate_parent_hparams(
        checkpoint,
        reference_path=reference_path,
        solver_profile=args.solver_profile,
    )
    migration = run_assistance_conditioning_migration_preflight(checkpoint)
    aware, blind = build_assistance_observability_pair_kwargs(
        args.solver_profile,
        reference_path,
        args.seed,
        checkpoint,
    )
    validate_pair_delta(aware, blind)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    preflight = {
        "protocol": "g1-assistance-observability-pair-preflight-v1",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "parent_hparams": parent_hparams,
        "code_provenance": code_provenance,
        "runtime_assets": assets,
        "migration": migration,
        "gpu_hardware": hardware,
        "aware_device": devices[0],
        "blind_device": devices[1],
        "paired_delta": ["actor_observe_torso_wrench_assistance"],
        "valid": True,
    }
    preflight_path = output_root / "preflight.json"
    preflight_temp = output_root / ".preflight.json.tmp"
    preflight_temp.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(preflight_temp, preflight_path)

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    common = {
        "profile_name": args.solver_profile,
        "reference_path": str(reference_path),
        "seed": args.seed,
        "resume_from": str(checkpoint),
        "output_root": str(output_root),
        "result_queue": result_queue,
    }
    processes = [
        context.Process(
            target=_run_arm,
            kwargs={
                **common,
                "arm": "aware",
                "observed": True,
                "device": devices[0],
            },
        ),
        context.Process(
            target=_run_arm,
            kwargs={
                **common,
                "arm": "blind",
                "observed": False,
                "device": devices[1],
            },
        ),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break
    results = complete_worker_results(
        results,
        tuple(zip(("aware", "blind"), processes, strict=True)),
    )
    failures = [result for result in results if not result[1]]
    if failures:
        details = "\n".join(f"[{arm}]\n{detail}" for arm, _, detail in failures)
        raise RuntimeError(f"paired assistance run failed:\n{details}")
    training_paths = {arm: path for arm, _, path in results}
    with ThreadPoolExecutor(max_workers=2) as executor:
        aware_future = executor.submit(
            evaluate_zero_tail_arm,
            arm="aware",
            training_directory=training_paths["aware"],
            observed=True,
            device=devices[0],
            reference_path=reference_path,
        )
        blind_future = executor.submit(
            evaluate_zero_tail_arm,
            arm="blind",
            training_directory=training_paths["blind"],
            observed=False,
            device=devices[1],
            reference_path=reference_path,
        )
        aware_selection = aware_future.result()
        blind_selection = blind_future.result()
    with ThreadPoolExecutor(max_workers=2) as executor:
        aware_artifacts = executor.submit(
            render_selected_arm,
            selection=aware_selection,
            device=devices[0],
            reference_path=reference_path,
        )
        blind_artifacts = executor.submit(
            render_selected_arm,
            selection=blind_selection,
            device=devices[1],
            reference_path=reference_path,
        )
        aware_selection["selected_artifact_root"] = aware_artifacts.result()
        blind_selection["selected_artifact_root"] = blind_artifacts.result()
    _write_json_atomic(
        Path(str(aware_selection["training_directory"]))
        / "dense_checkpoint_selection.json",
        aware_selection,
    )
    _write_json_atomic(
        Path(str(blind_selection["training_directory"]))
        / "dense_checkpoint_selection.json",
        blind_selection,
    )
    comparison = compare_pair_selections(aware_selection, blind_selection)
    _write_json_atomic(
        output_root / "paired_observability_comparison.json", comparison
    )
    for arm, path in sorted(training_paths.items()):
        print(f"{arm}: {path}")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
