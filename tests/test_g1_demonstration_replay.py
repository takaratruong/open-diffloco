from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np
import pytest


MODEL = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
REFERENCE = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/"
    "rmr_motion_walk_win137_212_named.npz"
)


@pytest.fixture(scope="module")
def env():
    from src.envs.g1_tracking.environment import (
        G1TrackingRMR50HzActionParityEnv,
    )

    return G1TrackingRMR50HzActionParityEnv(
        xml_path=str(MODEL),
        reference_path=str(REFERENCE),
        actor_history_len=10,
        actor_reference_lookahead_steps=(4, 8, 12),
        actor_reference_preview_mode="delta",
        reference_stride=1,
    )


def _candidate_state(env, *, phase: int, error: float, done: float = 0.0):
    state = env.reset_at_phase(
        jax.random.PRNGKey(11), jnp.array(0.0), jnp.array(phase)
    )
    qpos = state.data.qpos.at[0].add(0.37)
    model = env._get_randomized_model(state.info)
    data = mjx.forward(model, state.data.replace(qpos=qpos))
    metrics = {**state.metrics, "body_position_error": jnp.array(error)}
    return state.replace(data=data, metrics=metrics, done=jnp.array(done))


def test_threshold_replay_restores_exact_reference_and_coherent_history(env):
    from src.envs.g1_tracking.demonstration_replay import (
        apply_demonstration_replay,
    )

    candidate = _candidate_state(env, phase=25, error=0.21)
    candidate = candidate.replace(
        info={
            **candidate.info,
            "step": jnp.array(7, dtype=jnp.int32),
            "last_act": jnp.ones(env.action_dim),
        }
    )

    replayed, event = apply_demonstration_replay(
        env, candidate, threshold=0.20
    )

    assert bool(event)
    np.testing.assert_array_equal(replayed.data.qpos, env.qpos_reference[25])
    np.testing.assert_array_equal(replayed.data.qvel, env.qvel_reference[25])
    np.testing.assert_array_equal(replayed.info["last_act"], 0.0)
    assert int(replayed.info["step"]) == 7
    np.testing.assert_array_equal(replayed.info["rng"], candidate.info["rng"])
    history = np.asarray(replayed.info["actor_obs_history"])
    assert history.shape == (10, env.actor_frame_obs_dim)
    np.testing.assert_array_equal(history, np.repeat(history[:1], 10, axis=0))
    np.testing.assert_array_equal(replayed.obs, history.reshape(-1))
    np.testing.assert_array_equal(
        replayed.info["bootstrap_obs"], replayed.obs
    )
    assert np.isfinite(np.asarray(replayed.info["bootstrap_critic_obs"])).all()


@pytest.mark.parametrize("error,done", [(0.20, 0.0), (0.21, 1.0)])
def test_replay_bypasses_threshold_boundary_and_completed_transitions(
    env, error, done
):
    from src.envs.g1_tracking.demonstration_replay import (
        apply_demonstration_replay,
    )

    candidate = _candidate_state(env, phase=25, error=error, done=done)
    replayed, event = apply_demonstration_replay(
        env, candidate, threshold=0.20
    )

    assert not bool(event)
    np.testing.assert_array_equal(replayed.data.qpos, candidate.data.qpos)
    np.testing.assert_array_equal(replayed.obs, candidate.obs)


def test_replay_boundary_stops_state_gradient(env):
    from src.envs.g1_tracking.demonstration_replay import (
        apply_demonstration_replay,
    )

    candidate = _candidate_state(env, phase=25, error=0.21)

    def replayed_root_x(delta):
        changed = candidate.replace(
            data=candidate.data.replace(
                qpos=candidate.data.qpos.at[0].add(delta)
            )
        )
        replayed, _ = apply_demonstration_replay(
            env, changed, threshold=0.20
        )
        return replayed.data.qpos[0]

    assert float(jax.grad(replayed_root_x)(jnp.array(0.0))) == 0.0

    def replayed_reward(delta):
        changed = candidate.replace(reward=delta)
        replayed, _ = apply_demonstration_replay(
            env, changed, threshold=0.20
        )
        return replayed.reward

    assert float(jax.grad(replayed_reward)(jnp.array(0.0))) == 1.0


@pytest.mark.parametrize("threshold", [None, 0.0, -0.1, np.nan, np.inf, True])
def test_enabled_threshold_must_be_a_positive_finite_scalar(threshold):
    from src.envs.g1_tracking.demonstration_replay import (
        validate_demonstration_replay_threshold,
    )

    if threshold is None:
        assert validate_demonstration_replay_threshold(threshold) is None
    else:
        with pytest.raises(ValueError):
            validate_demonstration_replay_threshold(threshold)


def test_resume_requires_explicit_authority_for_replay_change():
    from src.envs.g1_tracking.demonstration_replay import (
        resolve_demonstration_replay_resume_setting,
    )

    assert (
        resolve_demonstration_replay_resume_setting(
            None,
            is_resume=False,
            requested=0.2,
            allow_change=False,
        )
        == 0.2
    )
    with pytest.raises(ValueError):
        resolve_demonstration_replay_resume_setting(
            {"demonstration_replay_threshold": None},
            is_resume=True,
            requested=0.2,
            allow_change=False,
        )


def test_checkpoint_replay_telemetry_is_finite_and_bounded():
    from src.algorithms.shac.algorithm import (
        build_checkpoint_demonstration_replay_telemetry,
    )

    row = build_checkpoint_demonstration_replay_telemetry(
        {
            "demonstration_replay_count": jnp.array(123),
            "demonstration_replay_fraction": jnp.array(0.25),
        },
        threshold=0.2,
    )
    assert row == {
        "demonstration_replay_threshold": 0.2,
        "demonstration_replay_count": 123,
        "demonstration_replay_fraction": 0.25,
        "demonstration_replay_valid": True,
    }
    for fraction in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            build_checkpoint_demonstration_replay_telemetry(
                {
                    "demonstration_replay_count": jnp.array(1),
                    "demonstration_replay_fraction": jnp.array(fraction),
                },
                threshold=0.2,
            )


def test_train_source_applies_replay_after_real_step_and_persists_contract():
    import inspect

    from src.algorithms.shac import algorithm

    source = inspect.getsource(algorithm.train)
    assert "candidate_unreplayed_state = env.step(state, noisy_action)" in source
    assert "apply_demonstration_replay(" in source
    assert '"demonstration_replay": jp.where(' in source
    assert '"demonstration_replay_threshold":' in source
