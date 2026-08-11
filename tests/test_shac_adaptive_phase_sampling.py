import inspect
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac import algorithm
from src.algorithms.shac.algorithm import (
    adaptive_phase_diagnostics,
    broadcast_adaptive_phase_state,
    migrate_adaptive_phase_env_state,
    train,
    transition_phase_before_reset,
    update_adaptive_phase_state,
)
from src.core.data_structures import EnvState, TrainState
from src.envs.g1_tracking.training_distribution import init_phase_sampler


def _leaf_bytes(tree):
    return [
        (np.asarray(leaf).dtype.str, np.asarray(leaf).shape, np.asarray(leaf).tobytes())
        for leaf in jax.tree_util.tree_leaves(tree)
    ]


def test_adaptive_phase_settings_are_default_off_with_fixed_treatment_values():
    parameters = inspect.signature(train).parameters

    assert parameters["adaptive_phase_sampling"].default is False
    assert parameters["adaptive_phase_uniform_ratio"].default == 0.5
    assert parameters["adaptive_phase_alpha"].default == 0.001


def test_adaptive_phase_update_uses_terminal_phases_from_both_shards():
    failed_count = init_phase_sampler(reference_length=500).failed_count
    transition_phases = jnp.array([[10, 11], [149, 150]])
    terminals = jnp.array([[0.0, 1.0], [0.0, 1.0]])

    updated = update_adaptive_phase_state(
        failed_count=failed_count,
        transition_phases=transition_phases,
        terminals=terminals,
        reference_length=500,
        alpha=0.001,
    )
    flattened = update_adaptive_phase_state(
        failed_count=failed_count,
        transition_phases=transition_phases.reshape(-1),
        terminals=terminals.reshape(-1),
        reference_length=500,
        alpha=0.001,
    )

    expected = np.zeros(failed_count.shape, dtype=np.float32)
    expected[0] = 0.001
    expected[3] = 0.001
    np.testing.assert_allclose(updated, expected, rtol=0.0, atol=1e-8)
    np.testing.assert_array_equal(updated, flattened)


def test_adaptive_phase_diagnostics_are_finite_and_keep_literal_uniform_floor():
    failed_count = init_phase_sampler(reference_length=500).failed_count
    failed_count = failed_count.at[3].set(4.0)
    transition_phases = jnp.array([[11, 150], [151, 499]])
    terminals = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    diagnostics = adaptive_phase_diagnostics(
        failed_count=failed_count,
        transition_phases=transition_phases,
        terminals=terminals,
        reference_length=500,
        uniform_ratio=0.5,
    )

    bin_count = failed_count.shape[0]
    np.testing.assert_array_equal(
        diagnostics["terminal_bin_counts"],
        np.array([1.0, 0.0, 0.0, 2.0] + [0.0] * (bin_count - 4)),
    )
    assert np.isfinite(np.asarray(diagnostics["failure_ema"])).all()
    assert np.isfinite(np.asarray(diagnostics["probabilities"])).all()
    np.testing.assert_allclose(
        np.sum(np.asarray(diagnostics["probabilities"])), 1.0, atol=1e-7
    )
    assert float(diagnostics["minimum_probability"]) >= (
        0.5 / bin_count - 1e-7
    )
    assert bool(diagnostics["valid"])

    below_treatment_floor = adaptive_phase_diagnostics(
        failed_count=failed_count,
        transition_phases=transition_phases,
        terminals=terminals,
        reference_length=500,
        uniform_ratio=0.2,
    )
    assert not bool(below_treatment_floor["valid"])


def test_e006_checkpoint_migration_adds_only_broadcast_zero_ema():
    legacy_env_state = EnvState(
        data={
            "qpos": jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
            "qvel": jnp.arange(9, dtype=jnp.float32).reshape(3, 3),
        },
        obs=jnp.arange(15, dtype=jnp.float32).reshape(3, 5),
        reward=jnp.array([1.0, 2.0, 3.0]),
        done=jnp.array([0.0, 1.0, 0.0]),
        info={
            "phase": jnp.array([10, 11, 150], dtype=jnp.int32),
            "rng": jnp.arange(6, dtype=jnp.uint32).reshape(3, 2),
            "difficulty": jnp.array([0.0, 0.5, 1.0]),
        },
        metrics={"reward": jnp.array([5.0, 6.0, 7.0])},
    )
    legacy_state = TrainState(
        key=jax.random.PRNGKey(0),
        env_state=legacy_env_state,
        actor_params={"w": jnp.array([1.0, 2.0])},
        critic_params={"w": jnp.array([3.0])},
        target_critic_params={"w": jnp.array([4.0])},
        normalizer=None,
        actor_opt=(),
        critic_opt=(),
        step=jnp.array(786_432, dtype=jnp.int32),
    )
    restored = pickle.loads(pickle.dumps(legacy_state))
    old_data = _leaf_bytes(restored.env_state.data)
    old_obs = _leaf_bytes(restored.env_state.obs)
    old_reward = _leaf_bytes(restored.env_state.reward)
    old_done = _leaf_bytes(restored.env_state.done)
    old_info = {
        name: _leaf_bytes(value) for name, value in restored.env_state.info.items()
    }
    old_metrics = _leaf_bytes(restored.env_state.metrics)

    migrated = migrate_adaptive_phase_env_state(
        restored.env_state, reference_length=500
    )

    assert _leaf_bytes(migrated.data) == old_data
    assert _leaf_bytes(migrated.obs) == old_obs
    assert _leaf_bytes(migrated.reward) == old_reward
    assert _leaf_bytes(migrated.done) == old_done
    assert {
        name: _leaf_bytes(migrated.info[name]) for name in old_info
    } == old_info
    assert _leaf_bytes(migrated.metrics) == old_metrics
    np.testing.assert_array_equal(
        migrated.info["phase_sampler_failed_count"],
        np.zeros((3, init_phase_sampler(500).failed_count.shape[0]), np.float32),
    )


def test_transition_phase_is_recorded_before_automatic_reset_changes_info():
    phases = jnp.array([10, 149, 499], dtype=jnp.int32)

    transition_phases = transition_phase_before_reset(
        phases,
        reference_stride=1,
        reference_length=500,
    )

    np.testing.assert_array_equal(transition_phases, np.array([11, 150, 499]))


def test_completed_ema_is_broadcast_only_after_the_actor_update():
    legacy = EnvState(
        data={"qpos": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)},
        obs=jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        reward=jnp.array([1.0, 2.0]),
        done=jnp.array([0.0, 1.0]),
        info={"phase": jnp.array([10, 150], dtype=jnp.int32)},
        metrics={"reward": jnp.array([3.0, 4.0])},
    )
    state = migrate_adaptive_phase_env_state(legacy, reference_length=500)
    prior = state.info["phase_sampler_failed_count"]
    completed = jnp.linspace(0.0, 1.0, prior.shape[-1], dtype=jnp.float32)
    old_leaves = {
        name: _leaf_bytes(value)
        for name, value in state.info.items()
        if name != "phase_sampler_failed_count"
    }

    updated = broadcast_adaptive_phase_state(state, completed)

    np.testing.assert_array_equal(prior, np.zeros_like(prior))
    np.testing.assert_array_equal(
        updated.info["phase_sampler_failed_count"],
        np.broadcast_to(completed, prior.shape),
    )
    assert {
        name: _leaf_bytes(updated.info[name]) for name in old_leaves
    } == old_leaves


def _checkpoint_state(*, adaptive: bool):
    info = {"phase": jnp.array([10, 150], dtype=jnp.int32)}
    if adaptive:
        info["phase_sampler_failed_count"] = jnp.zeros((2, 11))
    return SimpleNamespace(
        step=12_288,
        env_state=SimpleNamespace(info=info),
    )


def test_due_adaptive_checkpoint_persists_complete_hparams_before_state():
    state = _checkpoint_state(adaptive=True)
    hparams = {
        "algorithm": "shac",
        "adaptive_phase_sampling": True,
        "adaptive_phase_uniform_ratio": 0.5,
        "adaptive_phase_alpha": 0.001,
        "best_reward": 1.25,
    }

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        observed_hparams = None

        def assert_hparams_then_save(_state, save_dir, step):
            nonlocal observed_hparams
            with (Path(save_dir) / "hparams.json").open() as stream:
                observed_hparams = json.load(stream)
            return Path(save_dir) / f"checkpoint_step_{step:06d}.pkl"

        with mock.patch.object(
            algorithm,
            "save_periodic_checkpoint",
            side_effect=assert_hparams_then_save,
        ):
            last_step, checkpoint_path = (
                algorithm.archive_periodic_checkpoint_if_due(
                    state,
                    output,
                    last_checkpoint_step=0,
                    checkpoint_interval=12_288,
                    hparams=hparams,
                )
            )

    assert last_step == 12_288
    assert checkpoint_path.name == "checkpoint_step_012288.pkl"
    assert observed_hparams == hparams


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"adaptive_phase_sampling": False},
        {"adaptive_phase_sampling": True},
    ],
)
def test_adaptive_checkpoint_rejects_missing_or_false_resume_metadata(metadata):
    with pytest.raises(ValueError, match="adaptive checkpoint.*metadata"):
        algorithm.validate_adaptive_phase_resume_metadata(
            _checkpoint_state(adaptive=True),
            metadata,
            requested_adaptive_phase_sampling=False,
        )


def test_legacy_e006_state_allows_cli_adaptive_migration():
    resolved = algorithm.resolve_adaptive_phase_resume_settings(
        _checkpoint_state(adaptive=False),
        {"algorithm": "shac", "adaptive_phase_sampling": False},
        requested_adaptive_phase_sampling=True,
        requested_uniform_ratio=0.5,
        requested_alpha=0.001,
    )

    assert resolved == (True, 0.5, 0.001)
