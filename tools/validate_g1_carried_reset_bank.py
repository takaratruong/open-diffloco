"""Validate an immutable G1 carried-state reset-bank artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bank_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_rows: int,
    phase_start: int,
    phase_stride: int,
    bank_start: int,
) -> dict[str, object]:
    """Validate the fixed G1 state/action/termination bank contract."""
    if expected_rows < 1 or phase_stride < 1:
        raise ValueError("expected_rows and phase_stride must be positive")
    if not 0 <= bank_start < expected_rows:
        raise ValueError("bank_start must index the expected rows")
    required_shapes = {
        "qpos": (expected_rows, 36),
        "qvel": (expected_rows, 35),
        "phase": (expected_rows,),
        "action": (expected_rows, 29),
        "records": (expected_rows, 8),
        "termination_errors": (expected_rows, 4),
        "termination_thresholds": (4,),
    }
    normalized = {}
    for name, shape in required_shapes.items():
        if name not in arrays:
            raise ValueError(f"bank is missing {name}")
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} does not match {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        normalized[name] = value

    expected_phase = phase_start + phase_stride * np.arange(
        expected_rows, dtype=np.int32
    )
    phase = normalized["phase"]
    if not np.array_equal(phase, expected_phase):
        raise ValueError("phase rows do not match the fixed start and stride")
    if np.any(normalized["records"][:, 1] > 0.5):
        raise ValueError("bank source contains a terminal transition")
    thresholds = normalized["termination_thresholds"]
    if np.any(thresholds <= 0.0):
        raise ValueError("termination thresholds must be positive")
    clearance = 1.0 - (
        normalized["termination_errors"][bank_start:]
        / thresholds[None, :]
    )
    if np.any(clearance <= 0.0):
        raise ValueError("retained bank states must remain inside hard limits")
    quaternion_norm = np.linalg.norm(
        normalized["qpos"][:, 3:7], axis=1
    )
    if not np.allclose(quaternion_norm, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("bank root quaternions must be normalized")
    return {
        "rows": expected_rows,
        "bank_start": bank_start,
        "bank_rows": expected_rows - bank_start,
        "phase_first": int(phase[0]),
        "phase_last": int(phase[-1]),
        "bank_phase_first": int(phase[bank_start]),
        "terminal_count": 0,
        "minimum_bank_clearance": float(np.min(clearance)),
        "minimum_bank_clearance_by_component": np.min(
            clearance, axis=0
        ).tolist(),
    }


def _write_json_atomically(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--phase-start", type=int, required=True)
    parser.add_argument("--phase-stride", type=int, required=True)
    parser.add_argument("--bank-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    actual_sha256 = _sha256_file(args.bank)
    if actual_sha256 != args.bank_sha256:
        raise SystemExit(
            f"bank SHA-256 mismatch: {actual_sha256} != {args.bank_sha256}"
        )
    with np.load(args.bank, allow_pickle=False) as archive:
        summary = validate_bank_arrays(
            {name: np.asarray(archive[name]) for name in archive.files},
            expected_rows=args.expected_rows,
            phase_start=args.phase_start,
            phase_stride=args.phase_stride,
            bank_start=args.bank_start,
        )
    summary.update(
        {
            "bank_path": str(args.bank.resolve()),
            "bank_sha256": actual_sha256,
            "verdict": "carried-bank-admitted",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(args.output, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
