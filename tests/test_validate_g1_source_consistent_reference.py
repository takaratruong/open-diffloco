from __future__ import annotations

import numpy as np


def _fixture(frames: int = 5, bodies: int = 3) -> dict[str, np.ndarray]:
    pos = np.arange(frames * bodies * 3, dtype=np.float32).reshape(
        frames, bodies, 3
    )
    quat = np.arange(frames * bodies * 4, dtype=np.float32).reshape(
        frames, bodies, 4
    )
    return {
        "fps": np.asarray([50], dtype=np.int64),
        "joint_pos": np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
        "joint_vel": np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
        "body_pos_w": pos,
        "body_quat_w": quat,
        "body_lin_vel_w": pos / 10.0,
        "body_ang_vel_w": pos / 20.0,
    }


def test_repair_analysis_accepts_only_nonroot_suffix_replacement() -> None:
    from tools.validate_g1_source_consistent_reference import analyze_repair_arrays

    source = _fixture()
    replay = {name: value.copy() for name, value in source.items()}
    corrected = {name: value.copy() for name, value in source.items()}
    for name in (
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        replay[name][2:, 1:] += 100.0
        corrected[name][2:, 1:] = replay[name][2:, 1:]
    corrected["body_names"] = np.asarray(["pelvis", "a", "b"])

    result = analyze_repair_arrays(
        source,
        replay,
        corrected,
        prefix_frames=2,
        body_names=("pelvis", "a", "b"),
    )

    assert all(result["checks"].values())
    assert result["forbidden_changed_value_count"] == 0
    assert result["changed_value_counts"]["body_pos_w"] == 18


def test_repair_analysis_rejects_a_joint_change() -> None:
    from tools.validate_g1_source_consistent_reference import analyze_repair_arrays

    source = _fixture()
    replay = {name: value.copy() for name, value in source.items()}
    corrected = {name: value.copy() for name, value in source.items()}
    corrected["joint_pos"][3, 0] += 1.0
    corrected["body_names"] = np.asarray(["pelvis", "a", "b"])

    result = analyze_repair_arrays(
        source,
        replay,
        corrected,
        prefix_frames=2,
        body_names=("pelvis", "a", "b"),
    )

    assert result["checks"]["load_bearing_state_preserved"] is False
    assert result["forbidden_changed_value_count"] == 1
