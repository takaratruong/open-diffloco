"""Reusable gradient-Bellman pieces from the upstream JAVE trainer."""

import jax
import jax.numpy as jp


def normalize_jave_observation(
    observation,
    critic_normalizer_state,
    *,
    critic_dim: int,
    reward_feature_scale: float,
    eps: float,
):
    """Normalize critic state and the auxiliary reward feature separately."""

    critic_observation = observation[..., :critic_dim]
    normalized_critic = (
        (critic_observation - critic_normalizer_state.mean)
        / jp.sqrt(critic_normalizer_state.var + eps)
    )
    normalized_reward = (
        observation[..., critic_dim:] / reward_feature_scale
    )
    return jp.concatenate((normalized_critic, normalized_reward), axis=-1)


def denormalize_jave_observation(
    normalized_observation,
    critic_normalizer_state,
    *,
    critic_dim: int,
    reward_feature_scale: float,
    eps: float,
):
    """Invert :func:`normalize_jave_observation`."""

    normalized_critic = normalized_observation[..., :critic_dim]
    critic_observation = (
        normalized_critic
        * jp.sqrt(critic_normalizer_state.var + eps)
        + critic_normalizer_state.mean
    )
    reward_feature = (
        normalized_observation[..., critic_dim:] * reward_feature_scale
    )
    return jp.concatenate((critic_observation, reward_feature), axis=-1)


def learned_dynamics_loss(dynamics_model, dynamics_params, batch):
    """Fit normalized observation residuals, excluding reset transitions."""

    observation, action, next_observation, done = batch
    predicted_delta = dynamics_model.apply(
        dynamics_params, observation, action
    )
    target_delta = next_observation - observation
    valid = 1.0 - done
    valid_count = jp.maximum(jp.sum(valid), 1.0)
    squared_error = jp.sum(
        jp.square(predicted_delta - target_delta), axis=-1
    )
    return jp.sum(squared_error * valid) / valid_count


def gradient_bellman_targets(
    *,
    dynamics_model,
    dynamics_params,
    critic,
    target_critic_params,
    normalized_observations,
    actions,
    critic_dim: int,
    gamma: float,
    analytical_reward,
):
    """Compute action-frozen one-step JAVE targets for each state input."""

    actions = jax.lax.stop_gradient(actions)
    dynamics_params = jax.tree.map(jax.lax.stop_gradient, dynamics_params)
    target_critic_params = jax.tree.map(
        jax.lax.stop_gradient, target_critic_params
    )

    def one_step_return(observation, action):
        predicted_delta = dynamics_model.apply(
            dynamics_params, observation, action
        )
        next_observation = observation + predicted_delta
        reward = analytical_reward(
            observation, next_observation, action
        )
        next_value = critic.apply(
            target_critic_params,
            next_observation[:critic_dim],
        ).squeeze()
        return reward + gamma * next_value

    return jax.vmap(jax.grad(one_step_return, argnums=0))(
        normalized_observations, actions
    )


def gradient_bellman_loss(
    *,
    critic,
    critic_params,
    normalized_critic_observations,
    targets,
    critic_dim: int,
):
    """Match critic input gradients to the JAVE Bellman targets."""

    def critic_input_gradient(observation):
        return jax.grad(
            lambda value: critic.apply(critic_params, value).squeeze()
        )(observation)

    predicted = jax.vmap(critic_input_gradient)(
        normalized_critic_observations
    )
    targets = jax.lax.stop_gradient(targets[..., :critic_dim])
    return jp.mean(jp.sum(jp.square(predicted - targets), axis=-1))
