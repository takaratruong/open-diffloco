from __future__ import annotations

import numpy as np


def test_first_terminal_returns_include_done_reward_and_mask_post_reset() -> None:
    from experiments.g1_e002_critic_calibration_audit.run import (
        first_terminal_returns,
    )

    rewards = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [100.0, 30.0], [100.0, 40.0]]
    )
    dones = np.asarray(
        [[False, False], [True, False], [False, False], [False, True]]
    )

    realized, alive = first_terminal_returns(rewards, dones, gamma=0.5)

    np.testing.assert_array_equal(
        alive,
        [[True, True], [True, True], [False, True], [False, True]],
    )
    np.testing.assert_allclose(realized[:, 0], [2.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(realized[:, 1], [32.5, 45.0, 50.0, 40.0])


def test_calibration_mode_distinguishes_scale_error_from_bad_ranking() -> None:
    from experiments.g1_e002_critic_calibration_audit.run import (
        summarize_calibration_mode,
    )

    returns = np.linspace(0.5, 5.0, 40).reshape(5, 8)
    alive = np.ones_like(returns, dtype=bool)
    calibrated = summarize_calibration_mode(
        returns,
        returns,
        returns,
        alive,
        boundary_index=2,
    )
    scaled = summarize_calibration_mode(
        0.5 * returns,
        0.5 * returns,
        returns,
        alive,
        boundary_index=2,
    )
    reversed_values = summarize_calibration_mode(
        -returns,
        -returns,
        returns,
        alive,
        boundary_index=2,
    )

    assert calibrated["online_status"] == "calibrated"
    assert scaled["online_status"] == "rank-predictive-miscalibrated"
    assert reversed_values["online_status"] == "not-predictive"


def test_two_rollout_modes_require_online_calibration_in_both() -> None:
    from experiments.g1_e002_critic_calibration_audit.run import (
        classify_calibration_modes,
    )

    assert (
        classify_calibration_modes(
            {"deterministic": {"online_status": "calibrated"},
             "training_noise": {"online_status": "calibrated"}}
        )
        == "online-critic-calibrated-for-ahac-bootstrap"
    )
    assert (
        classify_calibration_modes(
            {
                "deterministic": {
                    "online_status": "rank-predictive-miscalibrated"
                },
                "training_noise": {
                    "online_status": "rank-predictive-miscalibrated"
                },
            }
        )
        == "online-critic-rank-predictive-but-miscalibrated"
    )
    assert (
        classify_calibration_modes(
            {"deterministic": {"online_status": "calibrated"},
             "training_noise": {"online_status": "not-predictive"}}
        )
        == "online-critic-not-predictive"
    )


def test_calibration_rollout_modes_compile_in_separate_graphs() -> None:
    from experiments.g1_e002_critic_calibration_audit.run import (
        execute_separately_compiled_rollouts,
    )

    compiled = []

    def fake_jit(function):
        compiled.append(function)
        return function

    def deterministic(state, noise):
        return state + noise

    def training_noise(state, noise):
        return state * noise

    first, second = execute_separately_compiled_rollouts(
        deterministic,
        training_noise,
        3,
        2,
        compile_fn=fake_jit,
    )

    assert compiled == [deterministic, training_noise]
    assert first == 5
    assert second == 6
