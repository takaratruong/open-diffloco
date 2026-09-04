"""Independently validate a source-consistent G1 reference-build run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "g1-source-consistent-reference-validation-v1"
BUILD_PROTOCOL = "g1-source-consistent-reference-build-v1"
RAW_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
BODY_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


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
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    result = json.loads(
        path.read_text(),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {token}")
        ),
    )
    if not isinstance(result, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def analyze_repair_arrays(
    source: Mapping[str, np.ndarray],
    replay: Mapping[str, np.ndarray],
    corrected: Mapping[str, np.ndarray],
    *,
    prefix_frames: int,
    body_names: tuple[str, ...],
) -> dict[str, object]:
    """Recompute the exact set of values a valid suffix repair may change."""
    missing = {
        label: sorted(set(RAW_KEYS).difference(arrays))
        for label, arrays in (
            ("source", source),
            ("replay", replay),
            ("corrected", corrected),
        )
    }
    if any(missing.values()):
        raise ValueError(f"reference arrays missing: {missing}")
    frames = int(np.asarray(source["joint_pos"]).shape[0])
    bodies = int(np.asarray(source["body_pos_w"]).shape[1])
    if not 0 < prefix_frames < frames or bodies != len(body_names):
        raise ValueError("invalid frame, prefix, or body count")
    for name in RAW_KEYS:
        shape = np.asarray(source[name]).shape
        if np.asarray(replay[name]).shape != shape or np.asarray(corrected[name]).shape != shape:
            raise ValueError(f"shape mismatch for {name}")

    changed_counts: dict[str, int] = {}
    forbidden_count = 0
    suffix_matches_replay = True
    for name in RAW_KEYS:
        source_value = np.asarray(source[name])
        corrected_value = np.asarray(corrected[name])
        changed = source_value != corrected_value
        changed_counts[name] = int(np.count_nonzero(changed))
        allowed = np.zeros(shape=source_value.shape, dtype=bool)
        if name in BODY_KEYS:
            allowed[prefix_frames:, 1:] = True
            suffix_matches_replay = suffix_matches_replay and bool(
                np.array_equal(
                    corrected_value[prefix_frames:, 1:],
                    np.asarray(replay[name])[prefix_frames:, 1:],
                )
            )
        forbidden_count += int(np.count_nonzero(changed & ~allowed))

    exact_prefix = all(
        np.array_equal(
            np.asarray(corrected[name])[:prefix_frames],
            np.asarray(source[name])[:prefix_frames],
        )
        for name in RAW_KEYS
        if name != "fps"
    )
    load_bearing = all(
        np.array_equal(np.asarray(corrected[name]), np.asarray(source[name]))
        for name in ("fps", "joint_pos", "joint_vel")
    ) and all(
        np.array_equal(
            np.asarray(corrected[name])[:, 0], np.asarray(source[name])[:, 0]
        )
        for name in BODY_KEYS
    )
    metadata_body_names = tuple(map(str, corrected.get("body_names", ())))
    finite = all(
        np.issubdtype(np.asarray(corrected[name]).dtype, np.number)
        and np.isfinite(np.asarray(corrected[name])).all()
        for name in RAW_KEYS
    )
    checks = {
        "exact_prefix_preserved": exact_prefix,
        "load_bearing_state_preserved": load_bearing,
        "no_forbidden_changes": forbidden_count == 0,
        "suffix_matches_isaac_replay": suffix_matches_replay,
        "body_names_match_sidecar": metadata_body_names == body_names,
        "all_numeric_arrays_finite": finite,
    }
    return {
        "checks": checks,
        "frames": frames,
        "bodies": bodies,
        "prefix_frames": prefix_frames,
        "changed_value_counts": changed_counts,
        "forbidden_changed_value_count": forbidden_count,
    }


def _placeholder_count(arrays: Mapping[str, np.ndarray]) -> int:
    position = np.asarray(arrays["body_pos_w"])
    quaternion = np.asarray(arrays["body_quat_w"])
    linear = np.asarray(arrays["body_lin_vel_w"])
    angular = np.asarray(arrays["body_ang_vel_w"])
    masks = (
        np.all(position[:, 1:] == position[:, :1], axis=(1, 2)),
        np.all(quaternion[:, 1:] == quaternion[:, :1], axis=(1, 2)),
        np.all(linear[:, 1:] == 0.0, axis=(1, 2)),
        np.all(angular[:, 1:] == 0.0, axis=(1, 2)),
    )
    return int(np.count_nonzero(np.logical_and.reduce(masks)))


def _exact_csv_prefix(candidate: Path, baseline: Path) -> bool:
    baseline_bytes = baseline.read_bytes()
    baseline_lines = baseline_bytes.splitlines(keepends=True)
    candidate_lines = candidate.read_bytes().splitlines(keepends=True)
    return b"".join(candidate_lines[: len(baseline_lines)]) == baseline_bytes


def validate_build(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    build_dir = run_dir / "seed-0" / "reference_build"
    paths = {
        "run": run_dir / "run.json",
        "completion": build_dir / "completion.json",
        "csv": build_dir / "lafan_walk_win137_300.csv",
        "csv_manifest": build_dir / "lafan_walk_win137_300.csv.manifest.json",
        "replay": build_dir / "lafan_walk_win137_300_exact_state_isaac_fk.npz",
        "joint_names": build_dir
        / "lafan_walk_win137_300_exact_state_isaac_fk.npz.joint_names.npy",
        "body_names": build_dir
        / "lafan_walk_win137_300_exact_state_isaac_fk.npz.body_names.npy",
        "corrected": build_dir
        / "lafan_walk_win137_300_source_consistent_named.npz",
        "source_metadata": build_dir / "source_metadata.json",
        "reference_manifest": build_dir / "reference_manifest.json",
        "reference_audit": build_dir / "corrected_reference_audit.json",
        "contract_validation": build_dir / "reference_contract_validation.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"required artifacts missing: {missing}")

    external = {
        "old_reference": (args.old_reference_path.resolve(), args.old_reference_sha256),
        "short_reference": (
            args.short_reference_path.resolve(),
            args.short_reference_sha256,
        ),
        "baseline_csv": (args.baseline_csv_path.resolve(), args.baseline_csv_sha256),
    }
    external_hash_checks = {
        name: path.is_file() and sha256_file(path) == expected
        for name, (path, expected) in external.items()
    }
    artifact_hashes = {name: sha256_file(path) for name, path in paths.items()}
    run = _strict_json(paths["run"])
    completion = _strict_json(paths["completion"])
    csv_manifest = _strict_json(paths["csv_manifest"])
    source_metadata = _strict_json(paths["source_metadata"])
    reference_manifest = _strict_json(paths["reference_manifest"])
    reference_audit = _strict_json(paths["reference_audit"])
    contract_validation = _strict_json(paths["contract_validation"])

    old = _load_npz(external["old_reference"][0])
    short = _load_npz(external["short_reference"][0])
    replay = _load_npz(paths["replay"])
    corrected = _load_npz(paths["corrected"])
    joint_names = tuple(map(str, np.load(paths["joint_names"], allow_pickle=False)))
    body_names = tuple(map(str, np.load(paths["body_names"], allow_pickle=False)))
    repair = analyze_repair_arrays(
        old,
        replay,
        corrected,
        prefix_frames=args.prefix_frames,
        body_names=body_names,
    )

    short_prefix = all(
        np.array_equal(
            np.asarray(corrected[name])
            if name == "fps"
            else np.asarray(corrected[name])[: args.prefix_frames],
            np.asarray(short[name]),
        )
        for name in RAW_KEYS
    )
    replay_errors = {
        "joint_pos_max_abs": float(
            np.max(np.abs(old["joint_pos"] - replay["joint_pos"]))
        ),
        "joint_vel_max_abs": float(
            np.max(np.abs(old["joint_vel"] - replay["joint_vel"]))
        ),
    }
    for name in BODY_KEYS:
        replay_errors[f"root_{name}_max_abs"] = float(
            np.max(np.abs(old[name][:, 0] - replay[name][:, 0]))
        )

    attempts = run.get("attempts", [])
    sole_attempt = attempts[0] if len(attempts) == 1 else {}
    completion_outputs = completion.get("outputs", {})
    completion_hashes_match = all(
        isinstance(entry, dict)
        and Path(str(entry.get("path", ""))).is_file()
        and sha256_file(Path(str(entry["path"]))) == entry.get("sha256")
        for entry in completion_outputs.values()
    )
    audit_classification = reference_audit.get("classification", {})
    load_comparison = reference_audit.get("load_bearing_state_vs_baseline", {})
    checks: dict[str, bool] = {
        **{f"external_{name}_hash": value for name, value in external_hash_checks.items()},
        "corrected_reference_hash": artifact_hashes["corrected"]
        == args.corrected_reference_sha256,
        "reference_audit_hash": artifact_hashes["reference_audit"]
        == args.reference_audit_sha256,
        "run_identity": run.get("experiment") == "E-20260904-004"
        and run.get("gpu_count") == 1
        and run.get("seeds") == [0],
        "sole_successful_attempt": len(attempts) == 1
        and sole_attempt.get("return_code") == 0
        and sole_attempt.get("timed_out") is False,
        "pushed_build_commit": completion.get("repository_identity", {})
        .get("rmr", {})
        .get("head")
        == args.expected_code_commit
        and completion.get("repository_identity", {}).get("rmr", {}).get("upstream")
        == args.expected_code_commit,
        "completion_protocol": completion.get("protocol") == BUILD_PROTOCOL,
        "construction_only": completion.get("no_dynamics_steps") is True
        and completion.get("policy_evaluated") is False
        and completion.get("optimizer_updates") == 0,
        "completion_output_hashes": completion_hashes_match,
        "csv_manifest_hash": csv_manifest.get("output_sha256")
        == artifact_hashes["csv"],
        "source_csv_prefix": _exact_csv_prefix(
            paths["csv"], external["baseline_csv"][0]
        ),
        "short_reference_prefix": short_prefix,
        "joint_order": len(joint_names) == 29
        and len(set(joint_names)) == 29
        and tuple(map(str, corrected["joint_names"])) == joint_names,
        "body_order": len(body_names) == 30
        and len(set(body_names)) == 30
        and body_names[0] == "pelvis",
        "repair_array_contract": all(repair["checks"].values()),
        "replay_state_error_bound": max(replay_errors.values()) <= 1e-4,
        "completion_replay_errors": completion.get("isaac_replay_state_errors")
        == replay_errors,
        "reference_manifest_output": reference_manifest.get("output_sha256")
        == artifact_hashes["corrected"],
        "reference_manifest_source_metadata": reference_manifest.get(
            "source_metadata_sha256"
        )
        == artifact_hashes["source_metadata"],
        "source_metadata_old_reference": source_metadata.get(
            "load_bearing_long_reference", {}
        ).get("sha256")
        == args.old_reference_sha256,
        "zero_placeholder_frames": _placeholder_count(corrected) == 0,
        "reference_audit_protocol": reference_audit.get("protocol")
        == "g1-named-reference-contract-audit-v1",
        "reference_audit_classification": audit_classification.get(
            "persistent_placeholder_suffix_detected"
        )
        is False
        and audit_classification.get("valid_native_rmr_positive_control_reference")
        is True
        and audit_classification.get("valid_current_diffsimp_load_contract") is True
        and audit_classification.get("same_target_contract_across_backends") is True,
        "diffsim_load_bearing_exact": load_comparison.get("qpos_array_equal") is True
        and load_comparison.get("qvel_array_equal") is True
        and load_comparison.get("qpos_max_abs") == 0.0
        and load_comparison.get("qvel_max_abs") == 0.0,
        "contract_validation": contract_validation.get("valid") is True
        and contract_validation.get("checks_passed")
        == contract_validation.get("checks_total"),
        "no_checkpoints": not any(run_dir.rglob("*.pkl"))
        and not any(run_dir.rglob("*.pt")),
    }
    return {
        "protocol": PROTOCOL,
        "valid": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "artifact_hashes": artifact_hashes,
        "repair_analysis": repair,
        "isaac_replay_state_errors": replay_errors,
        "placeholder_frame_count": _placeholder_count(corrected),
        "corrected_reference": {
            "path": str(paths["corrected"]),
            "sha256": artifact_hashes["corrected"],
            "frames": int(corrected["joint_pos"].shape[0]),
            "bodies": int(corrected["body_pos_w"].shape[1]),
            "joints": int(corrected["joint_pos"].shape[1]),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old-reference-path", type=Path, required=True)
    parser.add_argument("--old-reference-sha256", required=True)
    parser.add_argument("--short-reference-path", type=Path, required=True)
    parser.add_argument("--short-reference-sha256", required=True)
    parser.add_argument("--baseline-csv-path", type=Path, required=True)
    parser.add_argument("--baseline-csv-sha256", required=True)
    parser.add_argument("--corrected-reference-sha256", required=True)
    parser.add_argument("--reference-audit-sha256", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--prefix-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate_build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
