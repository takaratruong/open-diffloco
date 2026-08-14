import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algorithms.ppo.algorithm import (
    PPORollout,
    PPOTrainState,
    collect_rollout,
    compute_gae,
    gaussian_entropy,
    gaussian_log_prob,
    init_ppo_actor,
    ppo_loss,
    update_ppo,
    write_json_atomically,
    write_pickle_atomically,
)
from src.core.data_structures import EnvState, Normalizer
from src.core.rmr_training_policy import (
    GaussianRmrActorParams,
    init_rmr_mlp,
)


def test_gaussian_log_prob_matches_closed_form() -> None:
    mean = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    action = jnp.array([[1.0, -1.0]], dtype=jnp.float32)
    log_std = jnp.log(jnp.array([0.5, 2.0], dtype=jnp.float32))

    observed = gaussian_log_prob(action, mean, log_std)
    expected = -0.5 * (
        ((1.0 - 0.0) / 0.5) ** 2
        + ((-1.0 - 1.0) / 2.0) ** 2
        + 2.0 * math.log(0.5)
        + 2.0 * math.log(2.0)
        + 2.0 * math.log(2.0 * math.pi)
    )

    np.testing.assert_allclose(observed, [expected], rtol=1e-6, atol=1e-6)


def test_compute_gae_bootstraps_clip_end_but_not_true_terminal() -> None:
    rewards = jnp.array([[1.0], [1.0]], dtype=jnp.float32)
    values = jnp.array([[0.25], [0.25]], dtype=jnp.float32)
    bootstrap_values = jnp.array([[2.0], [2.0]], dtype=jnp.float32)
    dones = jnp.ones((2, 1), dtype=jnp.float32)
    terminals = jnp.array([[1.0], [0.0]], dtype=jnp.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        bootstrap_values=bootstrap_values,
        dones=dones,
        terminals=terminals,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_returns = np.array([[1.0], [1.0 + 0.99 * 2.0]])
    np.testing.assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        advantages,
        expected_returns - np.asarray(values),
        rtol=1e-6,
        atol=1e-6,
    )


def test_ppo_loss_uses_clipped_surrogate_and_value_objective() -> None:
    old_log_prob = jnp.zeros(2, dtype=jnp.float32)
    new_log_prob = jnp.log(jnp.array([1.5, 0.5], dtype=jnp.float32))
    advantages = jnp.array([1.0, -1.0], dtype=jnp.float32)
    values = jnp.array([2.0, -2.0], dtype=jnp.float32)
    old_values = jnp.zeros(2, dtype=jnp.float32)
    returns = jnp.array([1.0, -1.0], dtype=jnp.float32)

    loss, metrics = ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        values=values,
        old_values=old_values,
        returns=returns,
        entropy=jnp.array([2.0, 2.0], dtype=jnp.float32),
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    # Ratios 1.5 and 0.5 both hit their adverse clipped bound, yielding
    # surrogate terms 1.2 and -0.8.  The clipped value prediction is +/-0.2.
    expected_policy_loss = -np.mean([1.2, -0.8])
    # PPO takes the larger of unclipped and clipped value errors.
    expected_value_loss = 0.5 * np.mean([1.0**2, (-1.0) ** 2])
    expected = expected_policy_loss + 0.5 * expected_value_loss - 0.01 * 2.0
    np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metrics["clip_fraction"], 1.0)


def test_initial_actor_standard_deviation_is_exactly_point_two() -> None:
    params = init_ppo_actor(
        jax.random.PRNGKey(4),
        input_dim=154,
        action_dim=29,
        initial_std=0.2,
    )

    np.testing.assert_allclose(
        np.exp(np.asarray(params.log_std)),
        np.full(29, 0.2, dtype=np.float32),
        rtol=1e-7,
        atol=1e-7,
    )


def test_ppo_objective_has_finite_gradients() -> None:
    def objective(new_log_prob, values, log_std):
        entropy = jnp.broadcast_to(
            gaussian_entropy(log_std), new_log_prob.shape
        )
        loss, _ = ppo_loss(
            new_log_prob=new_log_prob,
            old_log_prob=jnp.zeros(3),
            advantages=jnp.array([-1.0, 0.5, 2.0]),
            values=values,
            old_values=jnp.zeros(3),
            returns=jnp.array([-0.5, 1.0, 1.5]),
            entropy=entropy,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )
        return loss

    gradients = jax.grad(objective, argnums=(0, 1, 2))(
        jnp.array([-0.1, 0.0, 0.1]),
        jnp.array([-0.25, 0.5, 1.25]),
        jnp.log(jnp.full(29, 0.2)),
    )

    assert all(np.all(np.isfinite(np.asarray(gradient))) for gradient in gradients)


class _ToyRolloutEnv:
    actor_obs_dim = 1
    critic_obs_dim = 1

    @staticmethod
    def _get_critic_obs(data, _info):
        return data[None]

    @staticmethod
    def step(state, action):
        next_data = state.data + action[0]
        step = state.info["step"] + 1
        done = (step >= 2).astype(jnp.float32)
        terminal = (next_data < -100.0).astype(jnp.float32)
        bootstrap = next_data[None]
        return state.replace(
            data=next_data,
            obs=bootstrap,
            reward=next_data,
            done=done,
            info={
                "step": jnp.where(done, 0, step),
                "terminal": terminal,
                "bootstrap_critic_obs": bootstrap,
            },
        )


def _toy_actor_and_critic():
    actor_mlp = init_rmr_mlp(
        jax.random.PRNGKey(11),
        input_dim=1,
        hidden_dims=(2,),
        output_dim=1,
    )
    actor = GaussianRmrActorParams(
        mlp=actor_mlp,
        log_std=jnp.array([math.log(0.2)], dtype=jnp.float32),
    )
    critic = init_rmr_mlp(
        jax.random.PRNGKey(12),
        input_dim=1,
        hidden_dims=(2,),
        output_dim=1,
    )
    return actor, critic


def _toy_batched_state() -> EnvState:
    return EnvState(
        data=jnp.array([0.0, 1.0], dtype=jnp.float32),
        obs=jnp.array([[0.0], [1.0]], dtype=jnp.float32),
        reward=jnp.zeros(2, dtype=jnp.float32),
        done=jnp.zeros(2, dtype=jnp.float32),
        info={
            "step": jnp.zeros(2, dtype=jnp.int32),
            "terminal": jnp.zeros(2, dtype=jnp.float32),
            "bootstrap_critic_obs": jnp.zeros((2, 1), dtype=jnp.float32),
        },
        metrics={},
    )


def test_collector_stops_simulator_gradients_and_preserves_transition_bits() -> None:
    actor, critic = _toy_actor_and_critic()
    normalizer = Normalizer(1)
    norm_state = normalizer.init()

    def run(actor_params):
        return collect_rollout(
            env=_ToyRolloutEnv(),
            env_state=_toy_batched_state(),
            actor_params=actor_params,
            critic_params=critic,
            actor_normalizer=normalizer,
            actor_norm_state=norm_state,
            critic_normalizer=normalizer,
            critic_norm_state=norm_state,
            key=jax.random.PRNGKey(13),
            horizon=2,
        )

    rollout, final_state, _ = run(actor)
    assert rollout.actions.shape == (2, 2, 1)
    assert rollout.dones.shape == (2, 2)
    assert rollout.terminals.shape == (2, 2)
    assert np.all(np.asarray(rollout.dones[-1]) == 1.0)
    assert np.all(np.asarray(rollout.terminals) == 0.0)
    np.testing.assert_allclose(
        final_state.data,
        _toy_batched_state().data + jnp.sum(rollout.actions[..., 0], axis=0),
    )

    gradients = jax.grad(lambda params: jnp.sum(run(params)[0].rewards))(actor)
    assert all(
        np.count_nonzero(np.asarray(leaf)) == 0
        for leaf in jax.tree_util.tree_leaves(gradients)
    )


def test_one_ppo_update_is_finite_and_changes_policy_parameters() -> None:
    actor, critic = _toy_actor_and_critic()
    actor_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    critic_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    normalizer = Normalizer(1)
    train_state = PPOTrainState(
        actor_params=actor,
        critic_params=critic,
        actor_opt=actor_optimizer.init(actor),
        critic_opt=critic_optimizer.init(critic),
        actor_norm=normalizer.init(),
        critic_norm=normalizer.init(),
        key=jax.random.PRNGKey(15),
        step=jnp.array(0, dtype=jnp.int32),
    )
    observations = jnp.array([[[0.0], [0.5]], [[1.0], [-0.5]]])
    critic_observations = observations
    actions = jnp.array([[[0.1], [-0.2]], [[0.3], [-0.1]]])
    old_means = jnp.zeros_like(actions)
    old_log_probs = gaussian_log_prob(actions, old_means, actor.log_std)
    rollout = PPORollout(
        observations=observations,
        critic_observations=critic_observations,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=jnp.zeros((2, 2)),
        rewards=jnp.array([[1.0, 0.5], [0.25, 1.5]]),
        dones=jnp.zeros((2, 2)),
        terminals=jnp.zeros((2, 2)),
        bootstrap_values=jnp.zeros((2, 2)),
        means=old_means,
    )

    updated, metrics = update_ppo(
        train_state,
        rollout,
        actor_normalizer=normalizer,
        critic_normalizer=normalizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        num_epochs=2,
        num_minibatches=2,
    )

    before = np.concatenate(
        [np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(actor)]
    )
    after = np.concatenate(
        [np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(updated.actor_params)]
    )
    assert np.linalg.norm(after - before) > 0.0
    assert int(updated.step) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())


def test_atomic_artifact_writers_round_trip_without_temporary_files(tmp_path) -> None:
    pickle_path = tmp_path / "checkpoint.pkl"
    json_path = tmp_path / "metrics.json"
    payload = {"step": 4, "finite": True}

    write_pickle_atomically(pickle_path, payload)
    write_json_atomically(json_path, payload)

    with pickle_path.open("rb") as stream:
        assert pickle.load(stream) == payload
    assert json_path.read_text(encoding="utf-8") == (
        '{\n  "finite": true,\n  "step": 4\n}\n'
    )
    assert list(tmp_path.glob(".*.tmp")) == []
