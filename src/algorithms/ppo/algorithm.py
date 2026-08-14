"""PPO primitives for the exact G1 MJX tracking positive control."""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.data_structures import NormState, Normalizer
from src.core.rmr_training_policy import (
    GaussianRmrActorParams,
    RmrMlpParams,
    apply_rmr_mlp,
    gaussian_entropy as _gaussian_entropy,
    init_gaussian_rmr_actor,
    init_rmr_critic,
)


@flax.struct.dataclass
class PPORollout:
    """Stopped on-policy transition batch in time-major layout."""

    observations: jax.Array
    critic_observations: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    old_values: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminals: jax.Array
    bootstrap_values: jax.Array
    means: jax.Array


@flax.struct.dataclass
class PPOTrainState:
    """Optimizer and normalization state for score-function PPO."""

    actor_params: GaussianRmrActorParams
    critic_params: RmrMlpParams
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    actor_norm: NormState
    critic_norm: NormState
    key: jax.Array
    step: jax.Array

    @property
    def normalizer(self) -> NormState:
        """Compatibility alias used by checkpoint evaluators."""

        return self.actor_norm


def init_ppo_actor(
    key: jax.Array,
    *,
    input_dim: int,
    action_dim: int,
    initial_std: float,
) -> GaussianRmrActorParams:
    """Initializes the RMR actor with an explicit learned Gaussian scale."""

    if not math.isfinite(initial_std) or initial_std <= 0.0:
        raise ValueError("initial_std must be positive and finite")
    params = init_gaussian_rmr_actor(
        key,
        input_dim=input_dim,
        action_dim=action_dim,
        dtype=jnp.float32,
    )
    return params._replace(
        log_std=jnp.full(
            (action_dim,), jnp.log(jnp.float32(initial_std)), dtype=jnp.float32
        )
    )


def gaussian_log_prob(
    action: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Returns the joint log probability of a diagonal Gaussian action."""

    action = jnp.asarray(action)
    mean = jnp.asarray(mean, dtype=action.dtype)
    log_std = jnp.asarray(log_std, dtype=action.dtype)
    inverse_variance_error = (action - mean) * jnp.exp(-log_std)
    per_coordinate = -0.5 * (
        jnp.square(inverse_variance_error)
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    return jnp.sum(per_coordinate, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Returns summed entropy for a diagonal Gaussian."""

    return _gaussian_entropy(log_std)


def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    dones: jax.Array,
    terminals: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes GAE while distinguishing true terminals from truncations."""

    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values, dtype=rewards.dtype)
    bootstrap_values = jnp.asarray(bootstrap_values, dtype=rewards.dtype)
    dones = jnp.asarray(dones, dtype=rewards.dtype)
    terminals = jnp.asarray(terminals, dtype=rewards.dtype)
    if not (
        rewards.shape
        == values.shape
        == bootstrap_values.shape
        == dones.shape
        == terminals.shape
    ):
        raise ValueError("GAE inputs must have identical shapes")

    deltas = (
        rewards
        + gamma * (1.0 - terminals) * bootstrap_values
        - values
    )

    def backward(carry, inputs):
        delta, done = inputs
        advantage = delta + gamma * gae_lambda * (1.0 - done) * carry
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(values[0]),
        (deltas[::-1], dones[::-1]),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def ppo_loss(
    *,
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    entropy: jax.Array,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Returns the standard clipped PPO actor-critic objective."""

    ratio = jnp.exp(new_log_prob - old_log_prob)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -jnp.mean(
        jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    )

    clipped_values = old_values + jnp.clip(
        values - old_values, -clip_epsilon, clip_epsilon
    )
    value_errors = jnp.square(values - returns)
    clipped_value_errors = jnp.square(clipped_values - returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_errors, clipped_value_errors))
    mean_entropy = jnp.mean(entropy)
    total = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )
    approximate_kl = jnp.mean(old_log_prob - new_log_prob)
    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
    )
    return total, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": mean_entropy,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
    }


def _critic_value(params: RmrMlpParams, observations: jax.Array) -> jax.Array:
    return jnp.squeeze(apply_rmr_mlp(params, observations), axis=-1)


def collect_rollout(
    *,
    env: Any,
    env_state: Any,
    actor_params: GaussianRmrActorParams,
    critic_params: RmrMlpParams,
    actor_normalizer: Normalizer,
    actor_norm_state: NormState,
    critic_normalizer: Normalizer,
    critic_norm_state: NormState,
    key: jax.Array,
    horizon: int,
) -> tuple[PPORollout, Any, jax.Array]:
    """Collects a vmapped rollout and cuts every simulator derivative."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    vector_step = jax.vmap(env.step)
    vector_critic_obs = jax.vmap(env._get_critic_obs)
    initial_state = jax.tree_util.tree_map(jax.lax.stop_gradient, env_state)

    def step(carry, _):
        state, current_key = carry
        current_key, noise_key = jax.random.split(current_key)
        observations = state.obs
        critic_observations = vector_critic_obs(state.data, state.info)
        normalized_observations = actor_normalizer.normalize(
            actor_norm_state, observations
        ).astype(jnp.float32)
        normalized_critic_observations = critic_normalizer.normalize(
            critic_norm_state, critic_observations
        ).astype(jnp.float32)
        means = apply_rmr_mlp(actor_params.mlp, normalized_observations)
        noise = jax.random.normal(noise_key, means.shape, dtype=means.dtype)
        actions = means + jnp.exp(actor_params.log_std) * noise
        log_probs = gaussian_log_prob(actions, means, actor_params.log_std)
        values = _critic_value(critic_params, normalized_critic_observations)

        next_state = vector_step(state, actions)
        bootstrap_observations = next_state.info["bootstrap_critic_obs"]
        normalized_bootstrap = critic_normalizer.normalize(
            critic_norm_state, bootstrap_observations
        ).astype(jnp.float32)
        bootstrap_values = _critic_value(critic_params, normalized_bootstrap)
        stopped_next_state = jax.tree_util.tree_map(
            jax.lax.stop_gradient, next_state
        )
        transition = PPORollout(
            observations=observations,
            critic_observations=critic_observations,
            actions=actions,
            old_log_probs=log_probs,
            old_values=values,
            rewards=next_state.reward,
            dones=next_state.done,
            terminals=next_state.info["terminal"],
            bootstrap_values=bootstrap_values,
            means=means,
        )
        transition = jax.tree_util.tree_map(jax.lax.stop_gradient, transition)
        return (stopped_next_state, current_key), transition

    (final_state, final_key), rollout = jax.lax.scan(
        step,
        (initial_state, key),
        None,
        length=horizon,
    )
    return rollout, final_state, final_key


def update_ppo(
    state: PPOTrainState,
    rollout: PPORollout,
    *,
    actor_normalizer: Normalizer,
    critic_normalizer: Normalizer,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    num_epochs: int,
    num_minibatches: int,
) -> tuple[PPOTrainState, dict[str, jax.Array]]:
    """Applies shuffled clipped-PPO epochs to one stopped rollout."""

    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.old_values,
        bootstrap_values=rollout.bootstrap_values,
        dones=rollout.dones,
        terminals=rollout.terminals,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    observations = rollout.observations.reshape(
        -1, rollout.observations.shape[-1]
    )
    critic_observations = rollout.critic_observations.reshape(
        -1, rollout.critic_observations.shape[-1]
    )
    actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])
    old_log_probs = rollout.old_log_probs.reshape(-1)
    old_values = rollout.old_values.reshape(-1)
    returns = returns.reshape(-1)
    advantages = advantages.reshape(-1)
    advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1e-8
    )
    total_samples = observations.shape[0]
    if total_samples % num_minibatches:
        raise ValueError("rollout size must be divisible by num_minibatches")
    if num_epochs < 1 or num_minibatches < 1:
        raise ValueError("PPO epochs and minibatches must be positive")
    minibatch_size = total_samples // num_minibatches

    key, permutation_key = jax.random.split(state.key)
    permutation_keys = jax.random.split(permutation_key, num_epochs)
    permutations = jax.vmap(
        lambda permutation_rng: jax.random.permutation(
            permutation_rng, total_samples
        )
    )(permutation_keys)
    minibatches = permutations.reshape(
        num_epochs * num_minibatches, minibatch_size
    )

    def minibatch_update(carry, indices):
        actor_params, critic_params, actor_opt, critic_opt = carry

        def objective(candidate_actor, candidate_critic):
            actor_obs = actor_normalizer.normalize(
                state.actor_norm, observations[indices]
            ).astype(jnp.float32)
            critic_obs = critic_normalizer.normalize(
                state.critic_norm, critic_observations[indices]
            ).astype(jnp.float32)
            means = apply_rmr_mlp(candidate_actor.mlp, actor_obs)
            new_log_probs = gaussian_log_prob(
                actions[indices], means, candidate_actor.log_std
            )
            values = _critic_value(candidate_critic, critic_obs)
            entropy = jnp.broadcast_to(
                gaussian_entropy(candidate_actor.log_std), new_log_probs.shape
            )
            return ppo_loss(
                new_log_prob=new_log_probs,
                old_log_prob=old_log_probs[indices],
                advantages=advantages[indices],
                values=values,
                old_values=old_values[indices],
                returns=returns[indices],
                entropy=entropy,
                clip_epsilon=clip_epsilon,
                value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,
            )

        (loss, metrics), (actor_grad, critic_grad) = jax.value_and_grad(
            objective, argnums=(0, 1), has_aux=True
        )(actor_params, critic_params)
        actor_updates, actor_opt = actor_optimizer.update(
            actor_grad, actor_opt, actor_params
        )
        critic_updates, critic_opt = critic_optimizer.update(
            critic_grad, critic_opt, critic_params
        )
        actor_params = optax.apply_updates(actor_params, actor_updates)
        critic_params = optax.apply_updates(critic_params, critic_updates)
        metrics = {
            **metrics,
            "loss": loss,
            "actor_gradient_norm": optax.tree.norm(actor_grad),
            "critic_gradient_norm": optax.tree.norm(critic_grad),
        }
        return (actor_params, critic_params, actor_opt, critic_opt), metrics

    final_carry, metric_history = jax.lax.scan(
        minibatch_update,
        (
            state.actor_params,
            state.critic_params,
            state.actor_opt,
            state.critic_opt,
        ),
        minibatches,
    )
    actor_params, critic_params, actor_opt, critic_opt = final_carry
    actor_norm = actor_normalizer.update(state.actor_norm, observations)
    critic_norm = critic_normalizer.update(
        state.critic_norm, critic_observations
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metric_history)
    metrics = {
        **metrics,
        "return_mean": jnp.mean(jnp.sum(rollout.rewards, axis=0)),
        "action_mean_rms": jnp.sqrt(jnp.mean(jnp.square(rollout.means))),
        "action_std_rms": jnp.sqrt(
            jnp.mean(jnp.square(jnp.exp(actor_params.log_std)))
        ),
    }
    return (
        state.replace(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            actor_norm=actor_norm,
            critic_norm=critic_norm,
            key=key,
            step=state.step + 1,
        ),
        metrics,
    )


def write_pickle_atomically(path: Path, payload: Any) -> None:
    """Writes a pickle through a same-directory atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomically(path: Path, payload: Any) -> None:
    """Writes strict JSON through a same-directory atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_is_finite(tree: Any) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def train(
    *,
    env: Any,
    output_dir: Path,
    total_iterations: int,
    num_envs: int,
    horizon: int,
    seed: int,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    initial_action_std: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.0,
    num_epochs: int = 4,
    num_minibatches: int = 8,
    max_grad_norm: float = 1.0,
    checkpoint_interval_iterations: int = 8,
    hparams: dict[str, Any] | None = None,
) -> tuple[PPOTrainState, Path]:
    """Trains PPO on the supplied exact MJX environment."""

    if total_iterations < 1 or num_envs < 1 or horizon < 1:
        raise ValueError("iterations, environments, and horizon must be positive")
    if checkpoint_interval_iterations < 1:
        raise ValueError("checkpoint interval must be positive")
    if (num_envs * horizon) % num_minibatches:
        raise ValueError("rollout population must divide evenly into minibatches")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    key, actor_key, critic_key, reset_key, train_key = jax.random.split(key, 5)
    actor_params = init_ppo_actor(
        actor_key,
        input_dim=env.actor_obs_dim,
        action_dim=env.action_dim,
        initial_std=initial_action_std,
    )
    critic_params = init_rmr_critic(
        critic_key,
        input_dim=env.critic_obs_dim,
        dtype=jnp.float32,
    )
    actor_normalizer = Normalizer(env.actor_obs_dim)
    critic_normalizer = Normalizer(env.critic_obs_dim)
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(actor_learning_rate),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(critic_learning_rate),
    )
    state = PPOTrainState(
        actor_params=actor_params,
        critic_params=critic_params,
        actor_opt=actor_optimizer.init(actor_params),
        critic_opt=critic_optimizer.init(critic_params),
        actor_norm=actor_normalizer.init(),
        critic_norm=critic_normalizer.init(),
        key=train_key,
        step=jnp.array(0, dtype=jnp.int32),
    )
    reset_keys = jax.random.split(reset_key, num_envs)
    difficulties = jnp.zeros(num_envs, dtype=jnp.float64)
    env_state = jax.jit(jax.vmap(env.reset))(reset_keys, difficulties)

    collect = jax.jit(
        functools.partial(
            collect_rollout,
            env=env,
            actor_normalizer=actor_normalizer,
            critic_normalizer=critic_normalizer,
            horizon=horizon,
        )
    )
    update = jax.jit(
        functools.partial(
            update_ppo,
            actor_normalizer=actor_normalizer,
            critic_normalizer=critic_normalizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            num_epochs=num_epochs,
            num_minibatches=num_minibatches,
        )
    )
    contract = {
        "algorithm": "ppo",
        "total_iterations": total_iterations,
        "num_envs": num_envs,
        "horizon": horizon,
        "seed": seed,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "initial_action_std": initial_action_std,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_epsilon,
        "value_coefficient": value_coefficient,
        "entropy_coefficient": entropy_coefficient,
        "num_epochs": num_epochs,
        "num_minibatches": num_minibatches,
        "max_grad_norm": max_grad_norm,
        "checkpoint_interval_iterations": checkpoint_interval_iterations,
        **(hparams or {}),
    }
    write_json_atomically(output_dir / "hparams.json", contract)
    telemetry: list[dict[str, Any]] = []

    for iteration in range(1, total_iterations + 1):
        started = time.monotonic()
        rollout, env_state, rollout_key = collect(
            env_state=env_state,
            actor_params=state.actor_params,
            critic_params=state.critic_params,
            actor_norm_state=state.actor_norm,
            critic_norm_state=state.critic_norm,
            key=state.key,
        )
        state = state.replace(key=rollout_key)
        state, metrics = update(state, rollout)
        jax.block_until_ready(metrics)
        finite = _tree_is_finite(state) and _tree_is_finite(metrics)
        if not finite:
            raise FloatingPointError(f"nonfinite PPO state at iteration {iteration}")
        dones = np.asarray(rollout.dones)
        terminals = np.asarray(rollout.terminals)
        first_done = np.argmax(dones > 0.0, axis=0) + 1
        first_done = np.where(np.any(dones > 0.0, axis=0), first_done, horizon)
        row = {
            "iteration": iteration,
            "transitions": iteration * num_envs * horizon,
            "walltime_seconds": time.monotonic() - started,
            "finite": finite,
            "rollout_mean_survival": float(np.mean(first_done)),
            "rollout_terminal_fraction": float(np.mean(terminals > 0.0)),
            **{name: float(value) for name, value in metrics.items()},
        }
        telemetry.append(row)
        write_json_atomically(output_dir / "training_metrics.json", telemetry)
        print(json.dumps(row, sort_keys=True), flush=True)
        if (
            iteration % checkpoint_interval_iterations == 0
            or iteration == total_iterations
        ):
            write_pickle_atomically(
                output_dir / f"checkpoint_iter_{iteration:05d}.pkl", state
            )

    write_pickle_atomically(output_dir / "policy_final.pkl", state)
    return state, output_dir
