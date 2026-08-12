import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.rmr_action_noise import (
    RMR_ACTION_STD,
    RMR_ACTION_STD_JOINT_NAMES,
    action_noise_std_hparam,
    validate_action_noise_std,
)

RMR_STD = np.array(
    [
        0.240585923,
        0.241455898,
        0.170482919,
        0.199771196,
        0.210099757,
        0.265766233,
        0.178219095,
        0.179201528,
        0.261124432,
        0.218309745,
        0.227685556,
        0.277082771,
        0.279196233,
        0.233728826,
        0.249337018,
        0.255797625,
        0.247496039,
        0.209423468,
        0.209507123,
        0.185532168,
        0.189358085,
        0.237778917,
        0.234698176,
        0.223405078,
        0.221006036,
        0.358317554,
        0.35941571,
        0.373313069,
        0.34844774,
    ],
    dtype=np.float32,
)
RMR_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def test_rmr_noise_constants_preserve_pinned_float32_values_and_actor_order():
    np.testing.assert_array_equal(np.asarray(RMR_ACTION_STD), RMR_STD)
    assert tuple(RMR_ACTION_STD_JOINT_NAMES) == RMR_JOINT_NAMES


def test_scalar_noise_remains_a_scalar_jax_array():
    actual = validate_action_noise_std(
        jnp.array(0.32, dtype=jnp.float32),
        action_dim=29,
        actor_joint_names=RMR_JOINT_NAMES,
    )

    assert actual.shape == ()
    assert actual.dtype == jnp.float32
    assert float(actual) == pytest.approx(0.32)


def test_vector_noise_multiplies_every_rollout_epsilon_joint():
    action_std = validate_action_noise_std(
        RMR_ACTION_STD,
        action_dim=29,
        actor_joint_names=RMR_JOINT_NAMES,
    )
    epsilon = jnp.ones((12, 29), dtype=jnp.float32)

    actual = epsilon * action_std

    assert actual.shape == (12, 29)
    np.testing.assert_array_equal(actual, np.broadcast_to(RMR_STD, (12, 29)))


def test_vector_noise_hparam_is_a_json_list_while_scalar_stays_a_scalar():
    assert action_noise_std_hparam(jnp.array(0.32, dtype=jnp.float32)) == pytest.approx(
        0.32
    )
    assert action_noise_std_hparam(RMR_ACTION_STD) == RMR_STD.tolist()


def test_scalar_noise_preserves_python_float_schedule_and_hparam_under_x64():
    from src.algorithms.shac.algorithm import resolve_action_noise_resume_settings

    with jax.experimental.enable_x64():
        start, end = resolve_action_noise_resume_settings(
            None,
            is_resume=False,
            requested_start=0.5,
            requested_end=0.32,
            allow_change=False,
            action_dim=12,
            actor_joint_names=(),
        )
        progress = jnp.array(0.625, dtype=jnp.float64)
        epsilon = jnp.array([0.25, -0.5], dtype=jnp.float64)
        expected_std = 0.5 + progress * (0.32 - 0.5)
        expected_noisy = epsilon * expected_std
        actual_std = start + progress * (end - start)
        actual_noisy = epsilon * actual_std

        assert isinstance(start, float)
        assert isinstance(end, float)
        assert actual_std.dtype == jnp.float64
        np.testing.assert_array_equal(actual_noisy, expected_noisy)
        assert json.dumps(action_noise_std_hparam(0.32)) == "0.32"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (jnp.ones((1, 29)), "scalar or a rank-one"),
        (jnp.ones((28,)), "shape"),
        (jnp.array(float("nan")), "finite"),
        (jnp.array([0.0, float("inf")]), "finite"),
        (jnp.array(-0.1), "non-negative"),
        (jnp.array([0.0, -0.1]), "non-negative"),
    ],
)
def test_noise_validation_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        validate_action_noise_std(
            value, action_dim=29, actor_joint_names=RMR_JOINT_NAMES
        )


def test_vector_noise_rejects_a_joint_order_mismatch():
    with pytest.raises(ValueError, match="actor joint order"):
        validate_action_noise_std(
            RMR_ACTION_STD,
            action_dim=29,
            actor_joint_names=tuple(reversed(RMR_JOINT_NAMES)),
        )


def test_vector_noise_requires_the_fixed_29_action_contract():
    with pytest.raises(ValueError, match="action dimension must be 29"):
        validate_action_noise_std(
            jnp.ones((30,), dtype=jnp.float32),
            action_dim=30,
            actor_joint_names=RMR_JOINT_NAMES + ("extra_joint",),
        )


def test_resume_noise_change_requires_explicit_authority():
    from src.algorithms.shac.algorithm import resolve_action_noise_resume_settings

    previous = {
        "action_noise_std_start": 0.5,
        "action_noise_std_end": 0.32,
    }
    with pytest.raises(ValueError, match="allow_resume_action_noise_change"):
        resolve_action_noise_resume_settings(
            previous,
            is_resume=True,
            requested_start=RMR_ACTION_STD,
            requested_end=RMR_ACTION_STD,
            allow_change=False,
            action_dim=29,
            actor_joint_names=RMR_JOINT_NAMES,
        )

    start, end = resolve_action_noise_resume_settings(
        previous,
        is_resume=True,
        requested_start=RMR_ACTION_STD,
        requested_end=RMR_ACTION_STD,
        allow_change=True,
        action_dim=29,
        actor_joint_names=RMR_JOINT_NAMES,
    )
    np.testing.assert_array_equal(start, RMR_STD)
    np.testing.assert_array_equal(end, RMR_STD)


def test_resume_noise_without_a_change_restores_scalar_checkpoint_values():
    from src.algorithms.shac.algorithm import resolve_action_noise_resume_settings

    start, end = resolve_action_noise_resume_settings(
        {"action_noise_std_start": 0.5, "action_noise_std_end": 0.32},
        is_resume=True,
        requested_start=0.5,
        requested_end=0.32,
        allow_change=False,
        action_dim=12,
        actor_joint_names=(),
    )

    assert isinstance(start, float)
    assert isinstance(end, float)
    assert start == 0.5
    assert end == 0.32


@pytest.mark.parametrize(
    "resumed_hparams",
    ({}, {"action_noise_std_start": 0.5}),
)
def test_incomplete_resume_noise_metadata_requires_explicit_authority(
    resumed_hparams,
):
    from src.algorithms.shac.algorithm import resolve_action_noise_resume_settings

    with pytest.raises(ValueError, match="complete action noise metadata"):
        resolve_action_noise_resume_settings(
            resumed_hparams,
            is_resume=True,
            requested_start=0.5,
            requested_end=0.32,
            allow_change=False,
            action_dim=12,
            actor_joint_names=(),
        )


def test_resumed_checkpoint_without_hparams_requires_explicit_authority():
    from src.algorithms.shac.algorithm import resolve_action_noise_resume_settings

    kwargs = {
        "requested_start": 0.5,
        "requested_end": 0.32,
        "action_dim": 12,
        "actor_joint_names": (),
        "is_resume": True,
    }
    with pytest.raises(ValueError, match="complete action noise metadata"):
        resolve_action_noise_resume_settings(
            None,
            allow_change=False,
            **kwargs,
        )

    start, end = resolve_action_noise_resume_settings(
        None,
        allow_change=True,
        **kwargs,
    )
    assert start == 0.5
    assert end == 0.32


def test_train_passes_explicit_resume_context_to_noise_resolver():
    from src.algorithms.shac.algorithm import train

    assert "is_resume=resume_from is not None" in inspect.getsource(train)
