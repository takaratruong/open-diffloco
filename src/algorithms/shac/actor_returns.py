"""Actor-return boundaries for fixed-horizon differentiable rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jp


ACTOR_RETURN_SEMANTICS = ("multi_episode", "first_terminal")


class ActorReturn(NamedTuple):
    """One scalar return and the rollout steps that contribute to it."""

    total: jax.Array
    reward_mask: jax.Array
    post_first_done_mask: jax.Array


def actor_return_rollout_metrics(
    *,
    dones: jax.Array,
    active: jax.Array,
    discounted_returns: jax.Array,
    reward_masks: jax.Array,
    post_first_done_masks: jax.Array,
    rewards: jax.Array,
) -> dict[str, jax.Array]:
    """Summarize a batched rollout while accepting numeric done flags."""

    done_flags = jp.asarray(dones, dtype=jp.bool_)
    active_flags = jp.asarray(active, dtype=jp.bool_)
    reward_mask = jp.asarray(reward_masks, dtype=jp.bool_)
    post_mask = jp.asarray(post_first_done_masks, dtype=jp.bool_)
    return_values = jp.asarray(discounted_returns)
    reward_values = jp.asarray(rewards)
    if done_flags.ndim != 2 or done_flags.shape[0] < 1 or done_flags.shape[1] < 1:
        raise ValueError("actor return rollout flags must be a nonempty matrix")
    if any(
        value.shape != done_flags.shape
        for value in (active_flags, reward_mask, post_mask, reward_values)
    ) or return_values.shape != (done_flags.shape[0],):
        raise ValueError("actor return rollout metric shapes do not match")
    post_count = jp.sum(post_mask)
    post_reward_sum = jp.sum(jp.where(post_mask, reward_values, 0.0))
    return {
        "actor_return_mean": jp.mean(return_values),
        "actor_return_done_env_count": jp.sum(
            jp.any(done_flags & active_flags, axis=1)
        ),
        "actor_return_done_event_count": jp.sum(done_flags & active_flags),
        "actor_return_included_transition_count": jp.sum(reward_mask),
        "actor_return_post_first_done_transition_count": post_count,
        "actor_return_post_first_done_env_count": jp.sum(jp.any(post_mask, axis=1)),
        "actor_return_post_first_done_reward_sum": post_reward_sum,
        "actor_return_post_first_done_reward_mean": post_reward_sum
        / jp.maximum(post_count, 1),
    }


def _validate_semantics(semantics: object) -> str:
    if not isinstance(semantics, str) or semantics not in ACTOR_RETURN_SEMANTICS:
        raise ValueError(
            f"actor_return_semantics must be one of {ACTOR_RETURN_SEMANTICS}"
        )
    return semantics


def _first_episode_reward_mask(dones: jax.Array, active: jax.Array) -> jax.Array:
    """Include active steps through, and including, the first done."""

    def scan_step(already_done, values):
        done, is_active = values
        include = is_active & ~already_done
        next_already_done = already_done | (include & done)
        return next_already_done, include

    _, mask = jax.lax.scan(
        scan_step,
        jp.asarray(False),
        (dones, active),
    )
    return mask


def discounted_actor_return(
    *,
    rewards: jax.Array,
    dones: jax.Array,
    terminals: jax.Array,
    bootstrap_values: jax.Array,
    active: jax.Array,
    final_value: jax.Array,
    gamma: float | jax.Array,
    bootstrap_scale: float | jax.Array,
    semantics: str,
) -> ActorReturn:
    """Accumulate SHAC rewards under an explicit episode-boundary contract.

    ``multi_episode`` preserves the historical behavior: every reset episode
    encountered inside the fixed rollout contributes another discounted return.
    ``first_terminal`` keeps only the prefix through the first done, matching an
    evaluation that stops at that boundary. A time-limit done may bootstrap from
    its pre-reset state; a true terminal never does.
    """

    semantics = _validate_semantics(semantics)
    rewards = jp.asarray(rewards)
    dones = jp.asarray(dones, dtype=jp.bool_)
    terminals = jp.asarray(terminals, dtype=jp.bool_)
    bootstrap_values = jp.asarray(bootstrap_values)
    active = jp.asarray(active, dtype=jp.bool_)
    if rewards.ndim != 1 or rewards.shape[0] < 1:
        raise ValueError("actor rewards must be a nonempty vector")
    expected_shape = rewards.shape
    if any(
        value.shape != expected_shape
        for value in (dones, terminals, bootstrap_values, active)
    ):
        raise ValueError("actor return inputs must have the same vector shape")

    first_episode_mask = _first_episode_reward_mask(dones, active)
    post_first_done_mask = active & ~first_episode_mask
    reward_mask = active if semantics == "multi_episode" else first_episode_mask
    effective_rewards = jp.where(reward_mask, rewards, jp.zeros_like(rewards))
    effective_dones = dones & reward_mask

    # Preserve JAX's weak-scalar promotion here. The historical accumulator
    # starts with Python scalars and multiplies a float32 scheduled bootstrap
    # scale before combining it with the float64 reward carry. Strengthening
    # those scalars changes exact resumes.
    zero = 0.0
    one = 1.0
    gamma_value = gamma
    bootstrap_scale_value = bootstrap_scale

    def accum_return(carry, values):
        total, running, discount = carry
        reward, done, terminal, value, is_active = values
        next_discount = jp.where(is_active, discount * gamma_value, discount)
        running = running + discount * reward
        truncation_bootstrap = (
            bootstrap_scale_value * (1.0 - terminal) * next_discount * value
        )
        total = total + jp.where(done, running + truncation_bootstrap, zero)
        running = jp.where(done, zero, running)
        discount = jp.where(done, one, next_discount)
        return (total, running, discount), None

    (total, running, final_discount), _ = jax.lax.scan(
        accum_return,
        (zero, zero, one),
        (
            effective_rewards,
            effective_dones,
            terminals,
            bootstrap_values,
            reward_mask,
        ),
    )
    first_done_seen = jp.any(dones & first_episode_mask)
    allow_final_bootstrap = (
        ~dones[-1] if semantics == "multi_episode" else ~first_done_seen
    )
    final_bootstrap = jp.where(
        allow_final_bootstrap,
        bootstrap_scale_value * final_discount * final_value,
        zero,
    )
    return ActorReturn(
        total=total + running + final_bootstrap,
        reward_mask=reward_mask,
        post_first_done_mask=post_first_done_mask,
    )


def resolve_actor_return_semantics(
    resumed_hparams: Mapping[str, object] | None,
    *,
    requested: object,
    is_resume: bool,
    allow_change: bool,
) -> str:
    """Preserve a checkpoint's objective unless a change is authorized."""

    requested = _validate_semantics(requested)
    if not isinstance(is_resume, bool) or not isinstance(allow_change, bool):
        raise ValueError("actor return resume settings must be boolean")
    if not is_resume:
        return requested

    saved = (
        "multi_episode"
        if not resumed_hparams
        else resumed_hparams.get("actor_return_semantics", "multi_episode")
    )
    saved = _validate_semantics(saved)
    if saved != requested and not allow_change:
        raise ValueError(
            "changing actor return semantics requires explicit resume authority"
        )
    return requested if allow_change else saved
