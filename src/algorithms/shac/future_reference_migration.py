"""Append-only SHAC checkpoint migration for G1 future references."""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jp
import numpy as np


def expand_history_input_rows(
    array: jax.Array,
    *,
    history_len: int,
    old_frame_dim: int,
    new_frame_dim: int,
) -> jax.Array:
    """Scatter legacy input rows into wider chronological frame blocks."""
    value = jp.asarray(array)
    if history_len < 1 or old_frame_dim < 1 or new_frame_dim <= old_frame_dim:
        raise ValueError("history and frame dimensions are invalid")
    if value.shape[0] != history_len * old_frame_dim:
        raise ValueError(
            "legacy input-row shape does not match declared history"
        )
    blocks = value.reshape(
        (history_len, old_frame_dim) + value.shape[1:]
    )
    expanded = jp.zeros(
        (history_len, new_frame_dim) + value.shape[1:],
        dtype=value.dtype,
    )
    expanded = expanded.at[:, :old_frame_dim].set(blocks)
    return expanded.reshape(
        (history_len * new_frame_dim,) + value.shape[1:]
    )


def _replace_first_kernel(tree: Any, kernel: jax.Array) -> Any:
    copied = copy.deepcopy(tree)
    try:
        copied["params"]["Dense_0"]["kernel"] = kernel
    except (KeyError, TypeError) as error:
        raise ValueError(
            "actor parameters must expose params/Dense_0/kernel"
        ) from error
    return copied


def _expand_optimizer_input_rows(
    old_state: Any,
    new_state: Any,
    *,
    history_len: int,
    old_frame_dim: int,
    new_frame_dim: int,
    hidden_dim: int,
) -> Any:
    old_kernel_shape = (history_len * old_frame_dim, hidden_dim)
    new_kernel_shape = (history_len * new_frame_dim, hidden_dim)
    expanded_count = 0

    def migrate(old, new):
        nonlocal expanded_count
        old_value = jp.asarray(old)
        new_value = jp.asarray(new)
        if old_value.shape == new_value.shape:
            return old
        if (
            old_value.shape == old_kernel_shape
            and new_value.shape == new_kernel_shape
        ):
            expanded_count += 1
            return expand_history_input_rows(
                old_value,
                history_len=history_len,
                old_frame_dim=old_frame_dim,
                new_frame_dim=new_frame_dim,
            )
        raise ValueError(
            "optimizer template has an unsupported shape mismatch: "
            f"{old_value.shape} -> {new_value.shape}"
        )

    try:
        migrated = jax.tree_util.tree_map(migrate, old_state, new_state)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("optimizer template structure does not match") from error
    if expanded_count != 2:
        raise ValueError(
            "optimizer migration must expand exactly Adam mu and nu rows"
        )
    return migrated


def _future_commands(env, phases: jax.Array) -> jax.Array:
    phase_array = jp.asarray(phases, dtype=jp.int32)
    flat = phase_array.reshape(-1)
    values = jax.vmap(env._future_reference_command)(flat)
    return values.reshape(phase_array.shape + (values.shape[-1],))


def _historical_phases(env, phase: jax.Array, step: jax.Array) -> jax.Array:
    ages = jp.arange(
        env.actor_history_len - 1, -1, -1, dtype=jp.int32
    )
    effective_ages = jp.minimum(
        ages, jp.asarray(step, dtype=jp.int32)[..., None]
    )
    return jp.maximum(
        jp.asarray(phase, dtype=jp.int32)[..., None]
        - effective_ages * env.reference_stride,
        0,
    )


def _expand_history(
    legacy_history: jax.Array,
    future_commands: jax.Array,
    *,
    history_len: int,
    old_frame_dim: int,
) -> jax.Array:
    history = jp.asarray(legacy_history)
    if history.shape[-2:] != (history_len, old_frame_dim):
        raise ValueError("saved actor history has an unsupported shape")
    if future_commands.shape[:-1] != history.shape[:-1]:
        raise ValueError("future command history leading shape does not match")
    return jp.concatenate(
        (history, future_commands.astype(history.dtype)), axis=-1
    )


def migrate_future_reference_train_state(
    resumed_state,
    initialized_state,
    env,
    *,
    expected_history_len: int = 10,
):
    """Expand one legacy SHAC TrainState without changing its initial actor."""
    if env.actor_history_len != expected_history_len:
        raise ValueError(
            "future reference migration requires the expected history length"
        )
    if not env.actor_reference_lookahead_steps:
        raise ValueError("future reference migration requires lookahead steps")
    old_frame_dim = int(resumed_state.normalizer.mean.shape[0])
    new_frame_dim = int(env.actor_frame_obs_dim)
    future_dim = int(env.actor_future_reference_dim)
    if old_frame_dim != 154 or new_frame_dim - old_frame_dim != future_dim:
        raise ValueError("checkpoint frame dimensions are not append-only")

    old_kernel = resumed_state.actor_params["params"]["Dense_0"][
        "kernel"
    ]
    template_kernel = initialized_state.actor_params["params"]["Dense_0"][
        "kernel"
    ]
    if old_kernel.ndim != 2 or template_kernel.ndim != 2:
        raise ValueError("actor input kernel must be a matrix")
    if old_kernel.shape[1] != template_kernel.shape[1]:
        raise ValueError("actor hidden width must not change during migration")
    expanded_kernel = expand_history_input_rows(
        old_kernel,
        history_len=expected_history_len,
        old_frame_dim=old_frame_dim,
        new_frame_dim=new_frame_dim,
    )
    if expanded_kernel.shape != template_kernel.shape:
        raise ValueError("expanded actor kernel does not match template")
    actor_params = _replace_first_kernel(
        resumed_state.actor_params, expanded_kernel
    )
    actor_opt = _expand_optimizer_input_rows(
        resumed_state.actor_opt,
        initialized_state.actor_opt,
        history_len=expected_history_len,
        old_frame_dim=old_frame_dim,
        new_frame_dim=new_frame_dim,
        hidden_dim=old_kernel.shape[1],
    )

    reference_phases = jp.arange(env.reference_length, dtype=jp.int32)
    reference_table = _future_commands(env, reference_phases)
    suffix_mean = jp.mean(reference_table, axis=0).astype(
        resumed_state.normalizer.mean.dtype
    )
    suffix_var = jp.var(reference_table, axis=0).astype(
        resumed_state.normalizer.var.dtype
    )
    normalizer = resumed_state.normalizer.replace(
        mean=jp.concatenate((resumed_state.normalizer.mean, suffix_mean)),
        var=jp.concatenate((resumed_state.normalizer.var, suffix_var)),
    )

    legacy_history = resumed_state.env_state.info["actor_obs_history"]
    phases = _historical_phases(
        env,
        resumed_state.env_state.info["phase"],
        resumed_state.env_state.info["step"],
    )
    future_history = _future_commands(env, phases)
    actor_history = _expand_history(
        legacy_history,
        future_history,
        history_len=expected_history_len,
        old_frame_dim=old_frame_dim,
    )
    legacy_bootstrap = resumed_state.env_state.info["bootstrap_obs"].reshape(
        legacy_history.shape
    )
    bootstrap_history = _expand_history(
        legacy_bootstrap,
        future_history,
        history_len=expected_history_len,
        old_frame_dim=old_frame_dim,
    )
    env_state = resumed_state.env_state.replace(
        obs=actor_history.reshape(actor_history.shape[:-2] + (-1,)),
        info={
            **resumed_state.env_state.info,
            "actor_obs_history": actor_history,
            "bootstrap_obs": bootstrap_history.reshape(
                bootstrap_history.shape[:-2] + (-1,)
            ),
        },
    )
    return resumed_state.replace(
        actor_params=actor_params,
        actor_opt=actor_opt,
        normalizer=normalizer,
        env_state=env_state,
    )


def _tree_equal(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return left_structure == right_structure and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _without_first_kernel(tree: Any) -> Any:
    copied = copy.deepcopy(tree)
    del copied["params"]["Dense_0"]["kernel"]
    return copied


def _optimizer_migration_evidence(
    original_state,
    migrated_state,
    *,
    history_len: int,
    legacy_frame_dim: int,
    treatment_frame_dim: int,
    hidden_dim: int,
) -> tuple[bool, bool, bool]:
    old_leaves, old_structure = jax.tree_util.tree_flatten(
        original_state.actor_opt
    )
    new_leaves, new_structure = jax.tree_util.tree_flatten(
        migrated_state.actor_opt
    )
    if old_structure != new_structure:
        return False, False, False
    old_shape = (history_len * legacy_frame_dim, hidden_dim)
    new_shape = (history_len * treatment_frame_dim, hidden_dim)
    expanded = 0
    legacy_exact = True
    new_zero = True
    other_exact = True
    for old, new in zip(old_leaves, new_leaves, strict=True):
        old_array = np.asarray(old)
        new_array = np.asarray(new)
        if old_array.shape == old_shape and new_array.shape == new_shape:
            expanded += 1
            old_blocks = old_array.reshape(
                history_len, legacy_frame_dim, hidden_dim
            )
            new_blocks = new_array.reshape(
                history_len, treatment_frame_dim, hidden_dim
            )
            legacy_exact &= np.array_equal(
                old_blocks, new_blocks[:, :legacy_frame_dim]
            )
            new_zero &= bool(
                np.all(new_blocks[:, legacy_frame_dim:] == 0.0)
            )
        else:
            other_exact &= np.array_equal(old_array, new_array)
    return (
        bool(expanded == 2 and legacy_exact),
        bool(expanded == 2 and new_zero),
        bool(other_exact),
    )


def _normalized_actor_input(
    state, *, history_len: int, frame_dim: int
) -> jax.Array:
    frames = state.env_state.obs.reshape(
        state.env_state.obs.shape[:-1] + (history_len, frame_dim)
    )
    normalized = (frames - state.normalizer.mean) / jp.sqrt(
        state.normalizer.var + 1e-4
    )
    return normalized.reshape(state.env_state.obs.shape)


def future_reference_migration_report(
    original_state,
    migrated_state,
    actor,
    *,
    legacy_frame_dim: int = 154,
    treatment_frame_dim: int = 328,
    history_len: int = 10,
) -> dict[str, object]:
    """Return JSON-safe evidence that an append-only migration is equivalent."""
    old_kernel = original_state.actor_params["params"]["Dense_0"][
        "kernel"
    ].reshape(history_len, legacy_frame_dim, -1)
    new_kernel = migrated_state.actor_params["params"]["Dense_0"][
        "kernel"
    ].reshape(history_len, treatment_frame_dim, -1)
    legacy_rows_exact = np.array_equal(
        np.asarray(old_kernel), np.asarray(new_kernel[:, :legacy_frame_dim])
    )
    new_rows_zero = bool(
        np.all(np.asarray(new_kernel[:, legacy_frame_dim:]) == 0.0)
    )
    non_input_actor_params_exact = _tree_equal(
        _without_first_kernel(original_state.actor_params),
        _without_first_kernel(migrated_state.actor_params),
    )
    (
        legacy_optimizer_rows_exact,
        new_optimizer_rows_zero,
        other_optimizer_leaves_exact,
    ) = _optimizer_migration_evidence(
        original_state,
        migrated_state,
        history_len=history_len,
        legacy_frame_dim=legacy_frame_dim,
        treatment_frame_dim=treatment_frame_dim,
        hidden_dim=old_kernel.shape[-1],
    )
    old_input = _normalized_actor_input(
        original_state,
        history_len=history_len,
        frame_dim=legacy_frame_dim,
    )
    new_input = _normalized_actor_input(
        migrated_state,
        history_len=history_len,
        frame_dim=treatment_frame_dim,
    )
    old_action = actor.apply(original_state.actor_params, old_input)
    new_action = actor.apply(migrated_state.actor_params, new_input)
    difference = jp.abs(new_action - old_action)
    absolute_error = float(jp.max(difference))
    relative_error = float(
        jp.max(difference / jp.maximum(jp.abs(old_action), 1e-12))
    )
    legacy_normalizer_exact = bool(
        np.array_equal(
            np.asarray(original_state.normalizer.mean),
            np.asarray(migrated_state.normalizer.mean[:legacy_frame_dim]),
        )
        and np.array_equal(
            np.asarray(original_state.normalizer.var),
            np.asarray(migrated_state.normalizer.var[:legacy_frame_dim]),
        )
        and np.array_equal(
            np.asarray(original_state.normalizer.count),
            np.asarray(migrated_state.normalizer.count),
        )
    )
    new_normalizer_finite = bool(
        np.isfinite(
            np.asarray(migrated_state.normalizer.mean[legacy_frame_dim:])
        ).all()
        and np.isfinite(
            np.asarray(migrated_state.normalizer.var[legacy_frame_dim:])
        ).all()
    )
    non_actor_state_exact = bool(
        _tree_equal(original_state.key, migrated_state.key)
        and _tree_equal(
            original_state.critic_params, migrated_state.critic_params
        )
        and _tree_equal(
            original_state.target_critic_params,
            migrated_state.target_critic_params,
        )
        and _tree_equal(original_state.critic_opt, migrated_state.critic_opt)
        and _tree_equal(
            original_state.critic_normalizer,
            migrated_state.critic_normalizer,
        )
        and int(original_state.step) == int(migrated_state.step)
    )
    valid = bool(
        legacy_rows_exact
        and new_rows_zero
        and non_input_actor_params_exact
        and legacy_optimizer_rows_exact
        and new_optimizer_rows_zero
        and other_optimizer_leaves_exact
        and legacy_normalizer_exact
        and new_normalizer_finite
        and non_actor_state_exact
        and absolute_error <= 1e-7
        and relative_error <= 1e-7
    )
    return {
        "legacy_actor_input_shape": list(original_state.env_state.obs.shape),
        "treatment_actor_input_shape": list(
            migrated_state.env_state.obs.shape
        ),
        "legacy_parameter_rows_exact": legacy_rows_exact,
        "new_parameter_rows_zero": new_rows_zero,
        "non_input_actor_params_exact": non_input_actor_params_exact,
        "legacy_optimizer_rows_exact": legacy_optimizer_rows_exact,
        "new_optimizer_rows_zero": new_optimizer_rows_zero,
        "other_optimizer_leaves_exact": other_optimizer_leaves_exact,
        "legacy_normalizer_exact": legacy_normalizer_exact,
        "new_normalizer_finite": new_normalizer_finite,
        "non_actor_state_exact": non_actor_state_exact,
        "max_action_absolute_error": absolute_error,
        "max_action_relative_error": relative_error,
        "valid": valid,
    }


def validate_future_reference_migration_report(
    report: dict[str, object],
) -> None:
    """Reject migration evidence that does not prove action equivalence."""
    if (
        report.get("valid") is not True
        or float(report.get("max_action_absolute_error", np.inf)) > 1e-7
        or float(report.get("max_action_relative_error", np.inf)) > 1e-7
    ):
        raise ValueError("future reference migration equivalence failed")
