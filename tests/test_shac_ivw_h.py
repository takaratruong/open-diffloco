import numpy as np
import pytest

import jax.numpy as jnp

from src.algorithms.shac.ivw_h import (
    discounted_reward_to_go,
    fuse_action_gradients,
    gaussian_mean_score_gradients,
    leave_one_out_phase_advantages,
    phase_step_action_ivw,
)


def test_discounted_reward_to_go_stops_at_done():
    reward = jnp.asarray([[1.0, 2.0, 4.0, 8.0]])
    done = jnp.asarray([[False, True, False, False]])

    actual = discounted_reward_to_go(reward, done, gamma=0.5)

    np.testing.assert_allclose(actual, [[2.0, 2.0, 8.0, 8.0]])


def test_discounted_reward_to_go_rejects_bad_contract():
    with pytest.raises(ValueError, match="shape"):
        discounted_reward_to_go(jnp.ones((2, 3)), jnp.ones((3, 2)), gamma=0.99)
    with pytest.raises(ValueError, match="gamma"):
        discounted_reward_to_go(jnp.ones((2, 3)), jnp.zeros((2, 3)), gamma=1.1)


def test_leave_one_out_baseline_excludes_own_return():
    returns = jnp.asarray([[1.0], [3.0], [9.0], [11.0]])
    phases = jnp.asarray([0, 0, 1, 1])

    actual = leave_one_out_phase_advantages(returns, phases)

    np.testing.assert_allclose(actual[:, 0], [-2.0, 2.0, -2.0, 2.0])


def test_leave_one_out_rejects_singleton_phase():
    with pytest.raises(ValueError, match="at least two"):
        leave_one_out_phase_advantages(
            jnp.asarray([[1.0], [2.0], [3.0]]),
            jnp.asarray([0, 0, 1]),
        )


def test_gaussian_score_gradient_has_policy_loss_sign():
    means = jnp.asarray([[[0.0, 1.0]]])
    actions = jnp.asarray([[[2.0, 0.0]]])
    advantage = jnp.asarray([[4.0]])

    actual = gaussian_mean_score_gradients(
        means,
        actions,
        advantage,
        jnp.asarray([2.0, 1.0]),
        horizon=2,
    )

    np.testing.assert_allclose(actual, [[[-1.0, 2.0]]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_score_gradient_rejects_nonfinite_inputs(bad):
    means = np.zeros((2, 1, 1))
    means[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        gaussian_mean_score_gradients(
            means,
            np.zeros_like(means),
            np.ones((2, 1)),
            np.ones(1),
            horizon=1,
        )


def test_score_gradient_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="positive"):
        gaussian_mean_score_gradients(
            np.zeros((2, 1, 1)),
            np.zeros((2, 1, 1)),
            np.ones((2, 1)),
            np.zeros(1),
            horizon=1,
        )


def test_ivw_uses_phase_local_sample_variance():
    score = jnp.asarray([[[0.0]], [[2.0]], [[10.0]], [[10.0]]])
    pathwise = jnp.asarray([[[0.0]], [[0.0]], [[8.0]], [[12.0]]])

    alpha = phase_step_action_ivw(
        score,
        pathwise,
        jnp.asarray([0, 0, 1, 1]),
    )

    np.testing.assert_allclose(alpha[:2], 1.0)
    np.testing.assert_allclose(alpha[2:], 0.0)


def test_ivw_rejects_singleton_phase_group():
    gradient = np.zeros((3, 1, 1))
    with pytest.raises(ValueError, match="at least two"):
        phase_step_action_ivw(gradient, gradient, np.asarray([0, 0, 1]))


def test_ivw_exact_zero_variance_selects_score_weight():
    gradient = np.zeros((2, 1, 1))
    np.testing.assert_array_equal(
        phase_step_action_ivw(gradient, gradient, np.asarray([0, 0])),
        gradient,
    )


def test_fusion_selects_the_registered_weight_per_sample():
    actual = fuse_action_gradients(
        jnp.asarray([[[2.0]]]),
        jnp.asarray([[[10.0]]]),
        jnp.asarray([[[0.25]]]),
    )

    np.testing.assert_allclose(actual, [[[8.0]]])


def test_fusion_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="shape"):
        fuse_action_gradients(
            jnp.zeros((2, 1, 1)),
            jnp.zeros((1, 1, 1)),
            jnp.zeros((2, 1, 1)),
        )
