"""Independently validate a G1 named-reference contract audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_PROTOCOL = "g1-named-reference-contract-audit-v1"
VALIDATION_PROTOCOL = "g1-named-reference-contract-validation-v1"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("audit report must be a JSON object")
    return value


def _persistent_suffix_start(mask: np.ndarray) -> int | None:
    false_indices = np.flatnonzero(~mask)
    start = 0 if false_indices.size == 0 else int(false_indices[-1] + 1)
    return start if start < mask.size else None


def _reconstruct_placeholder(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        root = int(np.asarray(archive["root_body_index"]).item())
        pos = np.asarray(archive["body_pos_w"])
        quat = np.asarray(archive["body_quat_w"])
        linear = np.asarray(archive["body_lin_vel_w"])
        angular = np.asarray(archive["body_ang_vel_w"])
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("body_pos_w has an incompatible shape")
    frames, bodies = pos.shape[:2]
    expected_shapes = (
        (quat, (frames, bodies, 4), "body_quat_w"),
        (linear, (frames, bodies, 3), "body_lin_vel_w"),
        (angular, (frames, bodies, 3), "body_ang_vel_w"),
    )
    if frames <= 0 or bodies <= 1 or not 0 <= root < bodies:
        raise ValueError("invalid frame, body, or root count")
    if not np.isfinite(pos).all():
        raise ValueError("body_pos_w contains non-finite values")
    for value, shape, name in expected_shapes:
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} is shape-incompatible or non-finite")
    nonroot = np.arange(bodies) != root
    masks = {
        "nonroot_positions_repeat_root": np.all(
            pos[:, nonroot] == pos[:, root : root + 1], axis=(1, 2)
        ),
        "nonroot_quaternions_repeat_root": np.all(
            quat[:, nonroot] == quat[:, root : root + 1], axis=(1, 2)
        ),
        "nonroot_linear_velocities_zero": np.all(
            linear[:, nonroot] == 0.0, axis=(1, 2)
        ),
        "nonroot_angular_velocities_zero": np.all(
            angular[:, nonroot] == 0.0, axis=(1, 2)
        ),
    }
    combined = np.logical_and.reduce(tuple(masks.values()))
    return {
        "frames": frames,
        "bodies": bodies,
        "root_index": root,
        "criteria_frame_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in masks.items()
        },
        "combined_placeholder_frame_count": int(np.count_nonzero(combined)),
        "combined_placeholder_frame_fraction": float(np.mean(combined)),
        "persistent_combined_suffix_start": _persistent_suffix_start(combined),
        "combined_placeholder_frame_indices": np.flatnonzero(combined).tolist(),
    }


def validate_audit(
    *,
    reference_path: Path,
    reference_sha256: str,
    report_path: Path,
    report_sha256: str,
    expected_prefix_frames: int,
) -> dict[str, object]:
    reference_path = reference_path.resolve()
    report_path = report_path.resolve()
    if not reference_path.is_file() or not report_path.is_file():
        raise ValueError("reference and audit report must exist")
    actual_reference_sha = sha256_file(reference_path)
    actual_report_sha = sha256_file(report_path)
    report = _strict_json(report_path)
    direct = _reconstruct_placeholder(reference_path)
    reported = report.get("placeholder_diagnostics")
    classification = report.get("classification")
    if not isinstance(reported, dict) or not isinstance(classification, dict):
        raise ValueError("audit report is missing diagnostics or classification")
    detected = direct["persistent_combined_suffix_start"] is not None
    report_input = report.get("inputs", {}).get("reference", {})
    checks = {
        "reference_hash": actual_reference_sha == reference_sha256,
        "report_hash": actual_report_sha == report_sha256,
        "audit_protocol": report.get("protocol") == AUDIT_PROTOCOL,
        "report_reference_path": report_input.get("path")
        == str(reference_path),
        "report_reference_hash": report_input.get("sha256")
        == reference_sha256,
        "prefix_frames": report.get("reference", {}).get("prefix_frames")
        == expected_prefix_frames,
        "frame_count": report.get("reference", {}).get("frames")
        == direct["frames"],
        "reported_placeholder_diagnostics": reported == direct,
        "detected_classification": classification.get(
            "persistent_placeholder_suffix_detected"
        )
        is detected,
        "native_classification": classification.get(
            "valid_native_rmr_positive_control_reference"
        )
        is (not detected),
        "diffsim_load_classification": classification.get(
            "valid_current_diffsimp_load_contract"
        )
        is True,
        "cross_backend_classification": classification.get(
            "same_target_contract_across_backends"
        )
        is (not detected),
    }
    return {
        "protocol": VALIDATION_PROTOCOL,
        "valid": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "inputs": {
            "reference": {
                "path": str(reference_path),
                "sha256": actual_reference_sha,
            },
            "audit_report": {
                "path": str(report_path),
                "sha256": actual_report_sha,
            },
        },
        "reconstructed_placeholder_suffix_start": direct[
            "persistent_combined_suffix_start"
        ],
        "reconstructed_placeholder_frame_count": direct[
            "combined_placeholder_frame_count"
        ],
        "reconstructed_placeholder_frame_indices": direct[
            "combined_placeholder_frame_indices"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--expected-prefix-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate_audit(
        reference_path=args.reference_path,
        reference_sha256=args.reference_sha256,
        report_path=args.report_path,
        report_sha256=args.report_sha256,
        expected_prefix_frames=args.expected_prefix_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
