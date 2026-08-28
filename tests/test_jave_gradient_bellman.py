import jax.numpy as jnp
import numpy as np

from src.algorithms.jave.gradient_bellman import (
    denormalize_jave_observation,
    gradient_bellman_loss,
    gradient_bellman_targets,
    learned_dynamics_loss,
    normalize_jave_observation,
)
from src.core.data_structures import NormState


class _LinearDynamics:
    @staticmethod
    def apply(params, obs, action):
        del action
        return obs @ params["matrix"].T


class _LinearCritic:
    @staticmethod
    def apply(params, obs):
        return jnp.sum(obs * params["weights"], axis=-1, keepdims=True)


def test_jave_observation_normalization_round_trips_reward_feature():
    norm = NormState(
        mean=jnp.array([1.0, -2.0]),
        var=jnp.array([4.0, 9.0]),
        count=jnp.array(10.0),
    )
    observation = jnp.array([[5.0, 1.0, 6.0]])

    normalized = normalize_jave_observation(
        observation,
        norm,
        critic_dim=2,
        reward_feature_scale=3.0,
        eps=0.0,
    )
    reconstructed = denormalize_jave_observation(
        normalized,
        norm,
        critic_dim=2,
        reward_feature_scale=3.0,
        eps=0.0,
    )

    np.testing.assert_allclose(normalized, [[2.0, 1.0, 2.0]])
    np.testing.assert_allclose(reconstructed, observation)


def test_learned_dynamics_loss_masks_terminal_reset_transition():
    params = {"matrix": jnp.zeros((3, 3))}
    batch = (
        jnp.zeros((2, 3)),
        jnp.zeros((2, 1)),
        jnp.array([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]),
        jnp.array([0.0, 1.0]),
    )

    loss = learned_dynamics_loss(_LinearDynamics(), params, batch)

    np.testing.assert_allclose(loss, 14.0)


def test_gradient_bellman_target_and_critic_matching_are_analytic():
    matrix = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.3],
        ]
    )
    reward_weights = jnp.array([0.7, -0.5, 0.2])
    critic_weights = jnp.array([0.4, -0.1])
    gamma = 0.9
    observations = jnp.array([[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2]])
    actions = jnp.zeros((2, 1))

    def reward_fn(_obs, next_obs, _action):
        return jnp.dot(next_obs, reward_weights)

    targets = gradient_bellman_targets(
        dynamics_model=_LinearDynamics(),
        dynamics_params={"matrix": matrix},
        critic=_LinearCritic(),
        target_critic_params={"weights": critic_weights},
        normalized_observations=observations,
        actions=actions,
        critic_dim=2,
        gamma=gamma,
        analytical_reward=reward_fn,
    )
    padded_value_weights = jnp.array(
        [critic_weights[0], critic_weights[1], 0.0]
    )
    expected_vector = (jnp.eye(3) + matrix).T @ (
        reward_weights + gamma * padded_value_weights
    )
    expected = jnp.broadcast_to(expected_vector, targets.shape)

    np.testing.assert_allclose(targets, expected, rtol=1e-6, atol=1e-6)
    matching_loss = gradient_bellman_loss(
        critic=_LinearCritic(),
        critic_params={"weights": expected_vector[:2]},
        normalized_critic_observations=observations[:, :2],
        targets=targets,
        critic_dim=2,
    )
    np.testing.assert_allclose(matching_loss, 0.0, atol=1e-12)
