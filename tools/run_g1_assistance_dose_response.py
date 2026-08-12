"""Run the registered four-checkpoint G1 assistance dose response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.g1_assistance_dose_response import (
    ASSISTANCE_SCALES,
    CHECKPOINT_LABELS,
    PHASES,
    classify_threshold_trajectory,
    condition_is_valid,
    required_scale,
)
from tools.prepare_g1_rmr_reference import sha256_file


EXPECTED_CHECKPOINT_SHA256 = {
    "parent": "fbea5e272d1431c08753a3600014623cd5577e34e01aeeba18b16af46d369377",
    "midpoint": "736aecbb5efc7b16c5764379d9d19cf331de637513bc380610fdfd060f138cff",
    "assistance_end": "959aafaf30a423baf19d23149df94e00f8b2abf8199f793e1b9606c626ffa612",
    "final": "0dccdca442ed15e17e76e4518d6c690e47d06ccd79d1440fb7012b36f78ff22f",
}
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def parse_checkpoint_specs(values: Sequence[str]) -> dict[str, Path]:
    """Parse the exact four label/path arguments without reordering them."""
    items: list[tuple[str, Path]] = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("checkpoint specs must use LABEL=PATH")
        items.append((label, Path(path)))
    if tuple(label for label, _ in items) != CHECKPOINT_LABELS:
        raise ValueError(
            f"checkpoint specs must follow registered checkpoint order {CHECKPOINT_LABELS}"
        )
    return dict(items)


def gpu_inventory() -> dict[str, str]:
    """Return the physical NVIDIA index-to-UUID map without importing JAX."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        index, uuid = (field.strip() for field in line.split(",", 1))
        inventory[index] = uuid
    return inventory


def validate_devices(
    devices: Sequence[str], *, inventory: Mapping[str, str]
) -> dict[str, str]:
    """Require one existing, distinct physical GPU for each checkpoint."""
    if len(devices) != len(CHECKPOINT_LABELS) or len(set(devices)) != len(devices):
        raise ValueError("runner requires one distinct physical GPU per checkpoint")
    missing = [device for device in devices if device not in inventory]
    if missing:
        raise ValueError(f"GPU devices are not present in inventory: {missing}")
    uuids = [inventory[device] for device in devices]
    if len(set(uuids)) != len(uuids):
        raise ValueError("runner requires one distinct physical GPU UUID per checkpoint")
    return {device: inventory[device] for device in devices}


def monitor_workers(
    workers: Mapping[str, Any], *, poll_interval_seconds: float = 1.0
) -> None:
    """Wait for all workers and terminate every live peer after any failure."""
    while True:
        return_codes = {label: process.poll() for label, process in workers.items()}
        failures = {
            label: code
            for label, code in return_codes.items()
            if code is not None and code != 0
        }
        if failures:
            for label, process in workers.items():
                if return_codes[label] is None:
                    process.terminate()
            for process in workers.values():
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
            detail = ", ".join(
                f"{label}={code}" for label, code in failures.items()
            )
            raise RuntimeError(f"dose-response worker failure: {detail}")
        if all(code == 0 for code in return_codes.values()):
            return
        time.sleep(poll_interval_seconds)


def stop_live_workers(workers: Mapping[str, Any]) -> None:
    """Terminate any worker that is still live after launch exits or aborts."""
    live = [process for process in workers.values() if process.poll() is None]
    for process in live:
        process.terminate()
    for process in live:
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)


def build_aggregate(
    workers: Sequence[dict[str, Any]],
    *,
    expected_checkpoint_sha256: Mapping[str, str],
    expected_device_uuid: Mapping[str, str],
    reference_sha256: str,
    code_commit: str,
    model_sha256: str,
    controller_sha256: str,
) -> dict[str, Any]:
    """Validate four worker manifests and classify the threshold trajectory."""
    if tuple(worker.get("checkpoint_label") for worker in workers) != CHECKPOINT_LABELS:
        raise ValueError("worker artifacts are outside the registered order")
    checkpoints = []
    for worker in workers:
        label = worker["checkpoint_label"]
        provenance = worker.get("provenance", {})
        if worker.get("protocol") != "g1-assistance-dose-response-worker-v1":
            raise ValueError(f"{label} worker protocol does not match")
        if provenance.get("checkpoint_sha256") != expected_checkpoint_sha256[label]:
            raise ValueError(f"{label} checkpoint SHA-256 does not match")
        if provenance.get("reference_sha256") != reference_sha256:
            raise ValueError(f"{label} reference SHA-256 does not match")
        if provenance.get("code_commit") != code_commit:
            raise ValueError(f"{label} code commit does not match")
        if provenance.get("solver_profile") != "g1-4x5":
            raise ValueError(f"{label} solver profile does not match")
        if provenance.get("solver_iterations") != 4 or provenance.get(
            "solver_ls_iterations"
        ) != 5:
            raise ValueError(f"{label} solver iteration budget does not match")
        runtime_assets = provenance.get("runtime_assets", {})
        if runtime_assets.get("model_sha256") != model_sha256:
            raise ValueError(f"{label} model SHA-256 does not match")
        if runtime_assets.get("controller_sha256") != controller_sha256:
            raise ValueError(f"{label} controller SHA-256 does not match")
        device = worker.get("device", {})
        if (
            device.get("platform") != "gpu"
            or device.get("device_count") != 1
            or device.get("physical_uuid") != expected_device_uuid[label]
        ):
            raise ValueError(f"{label} device identity does not match")
        if worker.get("phases") != list(PHASES) or worker.get(
            "assistance_scales"
        ) != list(ASSISTANCE_SCALES):
            raise ValueError(f"{label} worker grid does not match")
        conditions = worker.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != len(PHASES) * len(
            ASSISTANCE_SCALES
        ):
            raise ValueError(f"{label} worker condition grid is incomplete")
        expected_conditions = tuple(
            (phase, scale)
            for phase in PHASES
            for scale in ASSISTANCE_SCALES
        )
        observed_conditions = tuple(
            (item.get("phase"), item.get("scale")) for item in conditions
        )
        if observed_conditions != expected_conditions:
            raise ValueError(f"{label} worker condition grid does not match")
        for item in conditions:
            try:
                scale = float(item["scale"])
                wrench = item["wrench"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{label} raw condition is incomplete") from error
            recomputed_valid = condition_is_valid(item, wrench, scale=scale)
            if item.get("valid") is not True or not recomputed_valid:
                raise ValueError(f"{label} has an invalid raw condition")
        derived_thresholds = {
            str(phase): required_scale(
                [item for item in conditions if item["phase"] == phase],
                scales=ASSISTANCE_SCALES,
            )
            for phase in PHASES
        }
        if worker.get("required_scales") != derived_thresholds:
            raise ValueError(f"{label} derived thresholds do not match conditions")
        checkpoints.append(
            {
                "label": label,
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "provenance": provenance,
                "device": device,
                "required_scales": derived_thresholds,
                "conditions": conditions,
            }
        )
    return {
        "protocol": "g1-assistance-dose-response-v1",
        "verdict": classify_threshold_trajectory(checkpoints),
        "phases": list(PHASES),
        "assistance_scales": list(ASSISTANCE_SCALES),
        "checkpoints": checkpoints,
        "reference_sha256": reference_sha256,
        "code_commit": code_commit,
    }


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _preflight(
    *,
    repository: Path,
    checkpoints: Mapping[str, Path],
    devices: Sequence[str],
    reference: Path,
    code_commit: str,
) -> dict[str, Any]:
    head = _git_output(repository, "rev-parse", "HEAD")
    if head != code_commit:
        raise ValueError("runtime code commit does not match registration")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("runtime code worktree must be clean")
    if len(code_commit) != 40:
        raise ValueError("code commit must be a full SHA-1")
    checkpoint_records = {}
    for label, path in checkpoints.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        digest = sha256_file(resolved)
        if digest != EXPECTED_CHECKPOINT_SHA256[label]:
            raise ValueError(f"{label} checkpoint SHA-256 does not match")
        checkpoint_records[label] = {"path": str(resolved), "sha256": digest}
    reference = reference.resolve()
    if not reference.is_file() or sha256_file(reference) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference SHA-256 does not match")
    physical_devices = validate_devices(devices, inventory=gpu_inventory())
    return {
        "protocol": "g1-assistance-dose-response-preflight-v1",
        "code_commit": head,
        "dirty_patch_sha256": EMPTY_PATCH_SHA256,
        "checkpoints": checkpoint_records,
        "reference": str(reference),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "solver_profile": "g1-4x5",
        "phases": list(PHASES),
        "assistance_scales": list(ASSISTANCE_SCALES),
        "physical_devices": physical_devices,
    }


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    checkpoints = parse_checkpoint_specs(args.checkpoint)
    devices = tuple(args.device)
    output_root = args.output_root.resolve()
    aggregate_path = output_root / "assistance_dose_response.json"
    if aggregate_path.exists():
        raise FileExistsError(aggregate_path)
    preflight = _preflight(
        repository=repository,
        checkpoints=checkpoints,
        devices=devices,
        reference=args.reference_path,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    worker_root = output_root / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen[bytes]] = {}
    log_handles = []
    try:
        for (label, checkpoint), device in zip(
            checkpoints.items(), devices, strict=True
        ):
            output = worker_root / f"{label}.json"
            stdout = (worker_root / f"{label}.stdout.log").open("wb")
            stderr = (worker_root / f"{label}.stderr.log").open("wb")
            log_handles.extend((stdout, stderr))
            command = [
                sys.executable,
                "tools/evaluate_g1_assistance_dose_response.py",
                "--checkpoint",
                str(checkpoint.resolve()),
                "--checkpoint-label",
                label,
                "--checkpoint-sha256",
                EXPECTED_CHECKPOINT_SHA256[label],
                "--reference-path",
                str(args.reference_path.resolve()),
                "--code-commit",
                args.code_commit,
                "--physical-gpu-uuid",
                preflight["physical_devices"][device],
                "--output",
                str(output),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": device,
                    "JAX_ENABLE_X64": "true",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "PYTHONPATH": str(repository),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            processes[label] = subprocess.Popen(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
        monitor_workers(processes)
    finally:
        stop_live_workers(processes)
        for handle in log_handles:
            handle.close()
    worker_documents = [
        json.loads((worker_root / f"{label}.json").read_text(encoding="utf-8"))
        for label in CHECKPOINT_LABELS
    ]
    aggregate = build_aggregate(
        worker_documents,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        expected_device_uuid={
            label: preflight["physical_devices"][device]
            for label, device in zip(CHECKPOINT_LABELS, devices, strict=True)
        },
        reference_sha256=EXPECTED_REFERENCE_SHA256,
        code_commit=args.code_commit,
        model_sha256=EXPECTED_MODEL_SHA256,
        controller_sha256=EXPECTED_CONTROLLER_SHA256,
    )
    aggregate["preflight"] = preflight
    _write_json_atomically(aggregate_path, aggregate)
    print(json.dumps({"verdict": aggregate["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
