from __future__ import annotations

import numpy as np


def _body_arrays(frames: int = 6, bodies: int = 4) -> dict[str, np.ndarray]:
    positions = np.arange(frames * bodies * 3, dtype=np.float64).reshape(
        frames, bodies, 3
    )
    quaternions = np.zeros((frames, bodies, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    quaternions[..., 1] = np.arange(bodies, dtype=np.float64) / 100.0
    linear = positions / 10.0
    angular = positions / 20.0
    return {
        "body_pos_w": positions,
        "body_quat_w": quaternions,
        "body_lin_vel_w": linear,
        "body_ang_vel_w": angular,
    }


def test_placeholder_suffix_is_detected_at_exact_onset() -> None:
    from tools.audit_g1_named_reference_contract import placeholder_diagnostics

    arrays = _body_arrays()
    for name in ("body_pos_w", "body_quat_w"):
        arrays[name][2:, 1:] = arrays[name][2:, :1]
    arrays["body_lin_vel_w"][2:, 1:] = 0.0
    arrays["body_ang_vel_w"][2:, 1:] = 0.0

    result = placeholder_diagnostics(arrays, root_index=0)

    assert result["persistent_combined_suffix_start"] == 2
    assert result["combined_placeholder_frame_count"] == 4
    assert result["criteria_frame_counts"] == {
        "nonroot_positions_repeat_root": 4,
        "nonroot_quaternions_repeat_root": 4,
        "nonroot_linear_velocities_zero": 4,
        "nonroot_angular_velocities_zero": 4,
    }


def test_isolated_placeholder_frame_is_not_called_a_suffix() -> None:
    from tools.audit_g1_named_reference_contract import placeholder_diagnostics

    arrays = _body_arrays()
    for name in ("body_pos_w", "body_quat_w"):
        arrays[name][2, 1:] = arrays[name][2, :1]
    arrays["body_lin_vel_w"][2, 1:] = 0.0
    arrays["body_ang_vel_w"][2, 1:] = 0.0

    result = placeholder_diagnostics(arrays, root_index=0)

    assert result["persistent_combined_suffix_start"] is None
    assert result["combined_placeholder_frame_count"] == 1


def test_segment_error_metrics_are_sign_invariant_for_quaternions() -> None:
    from tools.audit_g1_named_reference_contract import segment_error_metrics

    raw = _body_arrays(frames=4, bodies=2)
    reconstructed = {name: value.copy() for name, value in raw.items()}
    reconstructed["body_quat_w"][2:] *= -1.0
    reconstructed["body_pos_w"][2:, 1, 0] += 3.0

    result = segment_error_metrics(raw, reconstructed, prefix_frames=2)

    assert result["prefix"]["body_position_rms_m"] == 0.0
    assert result["suffix"]["body_position_max_m"] == 3.0
    assert result["suffix"]["body_position_rms_m"] == np.sqrt(1.5)
    assert result["suffix"]["body_quaternion_chordal_rms"] == 0.0


def test_load_bearing_comparison_reports_exact_generalized_state() -> None:
    from tools.audit_g1_named_reference_contract import compare_load_bearing_state

    qpos = np.arange(20, dtype=np.float64).reshape(4, 5)
    qvel = np.arange(16, dtype=np.float64).reshape(4, 4)

    result = compare_load_bearing_state(qpos, qvel, qpos.copy(), qvel.copy())

    assert result == {
        "same_shape": True,
        "qpos_array_equal": True,
        "qvel_array_equal": True,
        "qpos_max_abs": 0.0,
        "qvel_max_abs": 0.0,
    }
