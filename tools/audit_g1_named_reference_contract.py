"""Audit whether a named G1 reference is valid for each tracking backend.

The native RMR/Isaac environment consumes every stored rigid-body field.  The
DiffSim G1 loader consumes joint state and only the pelvis rigid-body state,
then reconstructs all other rigid bodies with MuJoCo.  This tool makes that
otherwise easy-to-miss contract difference explicit and detects the placeholder
encoding used by the original MJX-only LAFAN conversion helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from src.envs.g1_tracking.controller import load_rmr_controller
from src.envs.g1_tracking.reference import load_mujoco_reference


PROTOCOL = "g1-named-reference-contract-audit-v1"
BODY_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

# Isaac/PhysX reports the links in articulation (breadth-first) order.  RMR's
# MotionLoader indexes its raw arrays with indices from this exact ordering.
RMR_G1_ALL_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str, label: str) -> tuple[Path, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return resolved, actual


def _validate_body_arrays(
    arrays: Mapping[str, np.ndarray], *, root_index: int
) -> tuple[int, int]:
    missing = sorted(set(BODY_KEYS).difference(arrays))
    if missing:
        raise ValueError(f"reference missing rigid-body arrays: {missing}")
    positions = np.asarray(arrays["body_pos_w"])
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("body_pos_w must have shape (T, B, 3)")
    frames, bodies = positions.shape[:2]
    if frames <= 0 or bodies <= 1 or not 0 <= root_index < bodies:
        raise ValueError("reference must have T > 0, B > 1, and a valid root index")
    expected = {
        "body_pos_w": (frames, bodies, 3),
        "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3),
        "body_ang_vel_w": (frames, bodies, 3),
    }
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    return frames, bodies


def _persistent_true_suffix_start(mask: np.ndarray) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("placeholder mask must be a nonempty vector")
    false_indices = np.flatnonzero(~mask)
    start = 0 if false_indices.size == 0 else int(false_indices[-1] + 1)
    return start if start < mask.size else None


def placeholder_diagnostics(
    arrays: Mapping[str, np.ndarray], *, root_index: int
) -> dict[str, object]:
    """Detect frames whose non-root fields use the MJX placeholder encoding."""
    frames, bodies = _validate_body_arrays(arrays, root_index=root_index)
    nonroot = np.arange(bodies) != root_index
    positions = np.asarray(arrays["body_pos_w"])
    quaternions = np.asarray(arrays["body_quat_w"])
    linear = np.asarray(arrays["body_lin_vel_w"])
    angular = np.asarray(arrays["body_ang_vel_w"])

    criteria = {
        "nonroot_positions_repeat_root": np.all(
            positions[:, nonroot] == positions[:, root_index : root_index + 1],
            axis=(1, 2),
        ),
        "nonroot_quaternions_repeat_root": np.all(
            quaternions[:, nonroot]
            == quaternions[:, root_index : root_index + 1],
            axis=(1, 2),
        ),
        "nonroot_linear_velocities_zero": np.all(
            linear[:, nonroot] == 0.0, axis=(1, 2)
        ),
        "nonroot_angular_velocities_zero": np.all(
            angular[:, nonroot] == 0.0, axis=(1, 2)
        ),
    }
    combined = np.logical_and.reduce(tuple(criteria.values()))
    return {
        "frames": frames,
        "bodies": bodies,
        "root_index": root_index,
        "criteria_frame_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in criteria.items()
        },
        "combined_placeholder_frame_count": int(np.count_nonzero(combined)),
        "combined_placeholder_frame_fraction": float(np.mean(combined)),
        "persistent_combined_suffix_start": _persistent_true_suffix_start(combined),
        "combined_placeholder_frame_indices": np.flatnonzero(combined).tolist(),
    }


def _error_summary(error: np.ndarray) -> tuple[float, float]:
    value = np.asarray(error, dtype=np.float64)
    if value.size == 0:
        raise ValueError("cannot summarize an empty reference segment")
    return float(np.sqrt(np.mean(np.square(value)))), float(np.max(np.abs(value)))


def segment_error_metrics(
    raw: Mapping[str, np.ndarray],
    reconstructed: Mapping[str, np.ndarray],
    *,
    prefix_frames: int,
) -> dict[str, dict[str, float | int]]:
    """Compare raw and reconstructed arrays before and after a known boundary."""
    frames, bodies = _validate_body_arrays(raw, root_index=0)
    other_frames, other_bodies = _validate_body_arrays(reconstructed, root_index=0)
    if (other_frames, other_bodies) != (frames, bodies):
        raise ValueError("raw and reconstructed rigid-body shapes differ")
    if not 0 < prefix_frames < frames:
        raise ValueError("prefix_frames must split the reference into two segments")

    raw_quat = np.asarray(raw["body_quat_w"], dtype=np.float64)
    reconstructed_quat = np.asarray(
        reconstructed["body_quat_w"], dtype=np.float64
    )
    quat_error = np.minimum(
        np.linalg.norm(raw_quat - reconstructed_quat, axis=-1),
        np.linalg.norm(raw_quat + reconstructed_quat, axis=-1),
    )
    errors = {
        "body_position": np.asarray(raw["body_pos_w"])
        - np.asarray(reconstructed["body_pos_w"]),
        "body_quaternion_chordal": quat_error,
        "body_linear_velocity": np.asarray(raw["body_lin_vel_w"])
        - np.asarray(reconstructed["body_lin_vel_w"]),
        "body_angular_velocity": np.asarray(raw["body_ang_vel_w"])
        - np.asarray(reconstructed["body_ang_vel_w"]),
    }
    result: dict[str, dict[str, float | int]] = {}
    for label, selection in (
        ("prefix", slice(0, prefix_frames)),
        ("suffix", slice(prefix_frames, frames)),
        ("all", slice(0, frames)),
    ):
        metrics: dict[str, float | int] = {
            "start_frame": int(selection.start or 0),
            "end_frame_exclusive": int(selection.stop or frames),
        }
        for name, error in errors.items():
            rms, maximum = _error_summary(error[selection])
            unit = "_m" if name == "body_position" else ""
            if "velocity" in name:
                unit = "_m_per_s" if "linear" in name else "_rad_per_s"
            metrics[f"{name}_rms{unit}"] = rms
            metrics[f"{name}_max{unit}"] = maximum
        result[label] = metrics
    return result


def compare_load_bearing_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    baseline_qpos: np.ndarray,
    baseline_qvel: np.ndarray,
) -> dict[str, object]:
    same_shape = qpos.shape == baseline_qpos.shape and qvel.shape == baseline_qvel.shape
    if not same_shape:
        return {
            "same_shape": False,
            "qpos_array_equal": False,
            "qvel_array_equal": False,
            "qpos_max_abs": None,
            "qvel_max_abs": None,
        }
    qpos_error = np.asarray(qpos) - np.asarray(baseline_qpos)
    qvel_error = np.asarray(qvel) - np.asarray(baseline_qvel)
    return {
        "same_shape": True,
        "qpos_array_equal": bool(np.array_equal(qpos, baseline_qpos)),
        "qvel_array_equal": bool(np.array_equal(qvel, baseline_qvel)),
        "qpos_max_abs": float(np.max(np.abs(qpos_error))),
        "qvel_max_abs": float(np.max(np.abs(qvel_error))),
    }


def _load_raw_body_arrays(path: Path) -> tuple[dict[str, np.ndarray], int]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in BODY_KEYS}
        root_index = int(np.asarray(archive["root_body_index"]).item())
    _validate_body_arrays(arrays, root_index=root_index)
    return arrays, root_index


def audit_reference(
    *,
    reference_path: Path,
    reference_sha256: str,
    model_path: Path,
    model_sha256: str,
    controller_path: Path,
    controller_sha256: str,
    prefix_frames: int,
    baseline_reference_path: Path | None = None,
    baseline_reference_sha256: str | None = None,
) -> dict[str, object]:
    reference_path, _ = _verify_file(
        reference_path, reference_sha256, "reference"
    )
    model_path, _ = _verify_file(model_path, model_sha256, "model")
    controller_path, _ = _verify_file(
        controller_path, controller_sha256, "controller"
    )
    if (baseline_reference_path is None) != (baseline_reference_sha256 is None):
        raise ValueError("baseline path and SHA-256 must be supplied together")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    controller = load_rmr_controller(model, controller_path)
    reference = load_mujoco_reference(
        model,
        reference_path,
        body_names=RMR_G1_ALL_BODY_NAMES,
        controller=controller,
    )
    raw, root_index = _load_raw_body_arrays(reference_path)
    reconstructed = {
        "body_pos_w": reference.body_pos,
        "body_quat_w": reference.body_quat,
        "body_lin_vel_w": reference.body_lin_vel,
        "body_ang_vel_w": reference.body_ang_vel,
    }
    placeholder = placeholder_diagnostics(raw, root_index=root_index)
    report: dict[str, object] = {
        "protocol": PROTOCOL,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "reference": {"path": str(reference_path), "sha256": reference_sha256},
            "model": {"path": str(model_path), "sha256": model_sha256},
            "controller": {
                "path": str(controller_path),
                "sha256": controller_sha256,
            },
        },
        "reference": {
            "frames": int(reference.qpos.shape[0]),
            "fps": reference.fps,
            "prefix_frames": prefix_frames,
            "rigid_body_order": list(RMR_G1_ALL_BODY_NAMES),
        },
        "placeholder_diagnostics": placeholder,
        "raw_vs_current_diffsimp_mujoco_reconstruction": segment_error_metrics(
            raw, reconstructed, prefix_frames=prefix_frames
        ),
        "backend_field_contracts": {
            "native_rmr_isaac": {
                "consumes_raw": [
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w[all bodies]",
                    "body_quat_w[all bodies]",
                    "body_lin_vel_w[all bodies]",
                    "body_ang_vel_w[all bodies]",
                ],
                "placeholder_suffix_is_valid": False,
            },
            "current_diffsimp_mujoco": {
                "consumes_raw": [
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w[root only]",
                    "body_quat_w[root only]",
                    "body_lin_vel_w[root only]",
                    "body_ang_vel_w[root only]",
                ],
                "recomputes": [
                    "body_pos_w[non-root]",
                    "body_quat_w[non-root]",
                    "body_lin_vel_w[non-root]",
                    "body_ang_vel_w[non-root]",
                ],
            },
        },
        "classification": {
            "persistent_placeholder_suffix_detected": placeholder[
                "persistent_combined_suffix_start"
            ]
            is not None,
            "valid_native_rmr_positive_control_reference": placeholder[
                "persistent_combined_suffix_start"
            ]
            is None,
            "valid_current_diffsimp_load_contract": True,
            "same_target_contract_across_backends": placeholder[
                "persistent_combined_suffix_start"
            ]
            is None,
        },
    }

    if baseline_reference_path is not None and baseline_reference_sha256 is not None:
        baseline_reference_path, _ = _verify_file(
            baseline_reference_path,
            baseline_reference_sha256,
            "baseline reference",
        )
        baseline = load_mujoco_reference(
            model,
            baseline_reference_path,
            body_names=RMR_G1_ALL_BODY_NAMES,
            controller=controller,
        )
        report["inputs"]["baseline_reference"] = {
            "path": str(baseline_reference_path),
            "sha256": baseline_reference_sha256,
        }
        report["load_bearing_state_vs_baseline"] = compare_load_bearing_state(
            reference.qpos,
            reference.qvel,
            baseline.qpos,
            baseline.qvel,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--prefix-frames", type=int, required=True)
    parser.add_argument("--baseline-reference-path", type=Path)
    parser.add_argument("--baseline-reference-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_reference(
        reference_path=args.reference_path,
        reference_sha256=args.reference_sha256,
        model_path=args.model_path,
        model_sha256=args.model_sha256,
        controller_path=args.controller_path,
        controller_sha256=args.controller_sha256,
        prefix_frames=args.prefix_frames,
        baseline_reference_path=args.baseline_reference_path,
        baseline_reference_sha256=args.baseline_reference_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
