from __future__ import annotations

import hashlib
import json

import numpy as np


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validator_independently_reconstructs_placeholder_suffix(tmp_path) -> None:
    from tools.validate_g1_named_reference_contract import validate_audit

    frames, bodies = 5, 3
    pos = np.arange(frames * bodies * 3, dtype=np.float32).reshape(
        frames, bodies, 3
    )
    quat = np.zeros((frames, bodies, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    quat[:, 1:, 1] = np.asarray([0.01, 0.02], dtype=np.float32)
    lin = pos / 10.0
    ang = pos / 20.0
    pos[2:, 1:] = pos[2:, :1]
    quat[2:, 1:] = quat[2:, :1]
    lin[2:, 1:] = 0.0
    ang[2:, 1:] = 0.0
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=lin,
        body_ang_vel_w=ang,
        root_body_index=np.asarray(0, dtype=np.int32),
    )
    reference_sha = _sha256(reference)
    report = {
        "protocol": "g1-named-reference-contract-audit-v1",
        "inputs": {"reference": {"path": str(reference), "sha256": reference_sha}},
        "reference": {"frames": frames, "prefix_frames": 2},
        "placeholder_diagnostics": {
            "frames": frames,
            "bodies": bodies,
            "root_index": 0,
            "criteria_frame_counts": {
                "nonroot_positions_repeat_root": 3,
                "nonroot_quaternions_repeat_root": 3,
                "nonroot_linear_velocities_zero": 3,
                "nonroot_angular_velocities_zero": 3,
            },
            "combined_placeholder_frame_count": 3,
            "combined_placeholder_frame_fraction": 0.6,
            "persistent_combined_suffix_start": 2,
            "combined_placeholder_frame_indices": [2, 3, 4],
        },
        "classification": {
            "persistent_placeholder_suffix_detected": True,
            "valid_native_rmr_positive_control_reference": False,
            "valid_current_diffsimp_load_contract": True,
            "same_target_contract_across_backends": False,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    result = validate_audit(
        reference_path=reference,
        reference_sha256=reference_sha,
        report_path=report_path,
        report_sha256=_sha256(report_path),
        expected_prefix_frames=2,
    )

    assert result["valid"] is True
    assert result["checks_passed"] == result["checks_total"]
    assert result["reconstructed_placeholder_suffix_start"] == 2
    assert result["reconstructed_placeholder_frame_count"] == 3
