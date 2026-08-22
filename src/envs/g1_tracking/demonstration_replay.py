"""DiffMimic-style intra-rollout state replay for G1 tracking."""

from __future__ import annotations

import math
from typing import Any, Mapping

import jax
import jax.numpy as jp


def validate_demonstration_replay_threshold(
    threshold: float | None,
) -> float | None:
    """Validate the optional mean-link-position error threshold in metres."""
    if threshold is None:
        return None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0.0
    ):
        raise ValueError(
            "demonstration_replay_threshold must be positive and finite or None"
        )
    return float(threshold)


def resolve_demonstration_replay_resume_setting(
    resumed_hparams: Mapping[str, Any] | None,
    *,
    is_resume: bool,
    requested: float | None,
    allow_change: bool,
) -> float | None:
    """Fail closed when a resume changes the demonstration-replay treatment."""
    requested = validate_demonstration_replay_threshold(requested)
    if not is_resume:
        return requested
    if resumed_hparams is None:
        raise ValueError(
            "resumed demonstration replay requires complete hparams provenance"
        )
    saved = validate_demonstration_replay_threshold(
        resumed_hparams.get("demonstration_replay_threshold")
    )
    if saved != requested and not allow_change:
        raise ValueError(
            "changing demonstration replay on resume requires explicit authority"
        )
    return requested if allow_change else saved


def _select_tree(condition, treatment, control):
    return jax.tree_util.tree_map(
        lambda selected, original: jp.where(condition, selected, original),
        treatment,
        control,
    )


def apply_demonstration_replay(env, state, *, threshold: float):
    """Restore a coherent exact reference state after an excessive transition."""
    threshold = validate_demonstration_replay_threshold(threshold)
    assert threshold is not None
    error = state.metrics["body_position_error"]
    event = (
        jp.isfinite(error)
        & (error > threshold)
        & (state.done == 0.0)
    )
    phase = state.info["phase"]
    randomization = {
        name: state.info[name]
        for name in (
            "friction_scale",
            "mass_scale",
            "kp_scale",
            "kd_scale",
            "com_offset",
        )
    }
    replay_data = env._data_from_state(
        qpos=env.qpos_reference[phase],
        qvel=env.qvel_reference[phase],
        randomization=randomization,
    )
    replay_info = {
        **state.info,
        "last_act": jp.zeros(env.action_dim),
        "terminal": jp.array(0.0),
        "transition_contact_stiffness": jp.array(0.0),
        "transition_contact_topology_event": jp.array(False),
        "foot_bump_ou": jp.zeros((4, 3)),
        "foot_normal_forces": jp.zeros(4),
    }
    replay_frame = env._get_actor_obs(replay_data, replay_info)
    replay_history = jp.repeat(
        replay_frame[None, :], env.actor_history_len, axis=0
    )
    replay_info = {
        **replay_info,
        "actor_obs_history": replay_history,
        "bootstrap_obs": replay_history.reshape(-1),
        "bootstrap_critic_obs": env._get_critic_obs(replay_data, replay_info),
    }
    replay_state = state.replace(
        data=jax.lax.stop_gradient(replay_data),
        obs=jax.lax.stop_gradient(replay_history.reshape(-1)),
        info=jax.lax.stop_gradient(replay_info),
    )
    return _select_tree(event, replay_state, state), event
