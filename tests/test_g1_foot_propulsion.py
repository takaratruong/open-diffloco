import math

import jax
import numpy as np
import pytest

from src.evaluation.g1_foot_propulsion import (
    constraint_propulsion_sample,
    reference_required_force,
    summarize_propulsion,
    yaw_frame_vector,
)


def _yaw_quaternion(angle: float) -> np.ndarray:
    return np.asarray(
        [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]
    )


def test_yaw_frame_vector_rotates_world_force_into_heading_frame():
    actual = yaw_frame_vector(
        np.asarray([0.0, 10.0, 3.0]), _yaw_quaternion(math.pi / 2.0)
    )

    np.testing.assert_allclose(actual, [10.0, 0.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(actual), np.linalg.norm([0.0, 10.0, 3.0]), atol=1e-6
    )


def test_constraint_sample_returns_force_and_interval_impulse():
    force, impulse = constraint_propulsion_sample(
        qfrc_constraint=np.asarray([4.0, -2.0, 10.0, 99.0]),
        root_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
        dt=0.04,
    )

    np.testing.assert_allclose(force, [4.0, -2.0, 10.0], rtol=0, atol=0)
    np.testing.assert_allclose(
        impulse, np.asarray([4.0, -2.0, 10.0]) * 0.04, rtol=0, atol=1e-7
    )


def test_zero_constraint_force_has_zero_gradient_safe_outputs():
    def scalar(force):
        yaw_force, impulse = constraint_propulsion_sample(
            qfrc_constraint=force,
            root_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
            dt=0.04,
        )
        return np.asarray(0.0) + yaw_force[0] + impulse[0]

    force, impulse = constraint_propulsion_sample(
        qfrc_constraint=np.zeros(6),
        root_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
        dt=0.04,
    )

    np.testing.assert_array_equal(force, np.zeros(3))
    np.testing.assert_array_equal(impulse, np.zeros(3))
    assert np.isfinite(jax.grad(scalar)(np.zeros(6))).all()


def test_reference_required_force_uses_stride_matched_velocity_difference():
    required = reference_required_force(
        reference_root_velocity=np.asarray(
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.8, 0.1, 0.0]]
        ),
        phase=0,
        stride=2,
        dt=0.08,
        total_mass=32.0,
        root_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(required, [320.0, 40.0, 0.0], atol=1e-5)


def test_reference_endpoint_uses_zero_one_sided_acceleration():
    required = reference_required_force(
        reference_root_velocity=np.asarray([[1.0, 0.0, 0.0]]),
        phase=0,
        stride=1,
        dt=0.04,
        total_mass=33.0,
        root_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(required, np.zeros(3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"qfrc_constraint": np.zeros(2)}, "constraint force"),
        ({"root_quaternion": np.zeros(4)}, "quaternion"),
        ({"dt": 0.0}, "dt"),
    ],
)
def test_constraint_sample_rejects_invalid_inputs(kwargs, message):
    values = {
        "qfrc_constraint": np.zeros(6),
        "root_quaternion": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "dt": 0.04,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        constraint_propulsion_sample(**values)


def test_summarize_propulsion_reports_error_and_peak():
    summary = summarize_propulsion(
        actual_forward=np.asarray([1.0, 3.0, -2.0]),
        required_forward=np.asarray([2.0, 1.0, -2.0]),
    )

    assert summary == {
        "propulsion_forward_error_rms": pytest.approx(math.sqrt(5.0 / 3.0)),
        "propulsion_forward_force_peak_abs": 3.0,
    }


def test_summarize_propulsion_rejects_misaligned_or_nonfinite_rows():
    with pytest.raises(ValueError, match="aligned finite"):
        summarize_propulsion(
            actual_forward=np.asarray([1.0, 2.0]),
            required_forward=np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="aligned finite"):
        summarize_propulsion(
            actual_forward=np.asarray([np.nan]),
            required_forward=np.asarray([1.0]),
        )
