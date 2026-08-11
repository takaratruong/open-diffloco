from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.networks import Actor


def _tree_arrays_equal(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return left_structure == right_structure and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _adam_state(state):
    return state[1][0]


def _toy_policy():
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
    )

    parent_actor = Actor(
        action_dim=2,
        hidden=(4,),
        squash=True,
        layer_norm=False,
        zero_output=False,
    )
    residual_actor = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    parent_params = parent_actor.init(
        jax.random.PRNGKey(1), jnp.zeros((1, 15), dtype=jnp.float32)
    )
    adapter_params = residual_actor.init(
        jax.random.PRNGKey(2), jnp.zeros((1, 5), dtype=jnp.float32)
    )
    params = FrozenPreviewResidualParams(
        parent=parent_params,
        adapter=adapter_params,
    )
    return parent_actor, residual_actor, params


def test_zero_head_preserves_parent_action_exactly():
    from src.algorithms.shac.residual_preview_adapter import (
        apply_frozen_preview_residual,
    )

    parent_actor, residual_actor, params = _toy_policy()
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 10.0

    candidate, parent, residual = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        params,
        observations,
        history_len=3,
        treatment_frame_dim=5,
    )

    np.testing.assert_array_equal(candidate, parent)
    np.testing.assert_array_equal(residual, jnp.zeros((1, 2)))
    np.testing.assert_array_equal(
        parent, parent_actor.apply(params.parent, observations)
    )


def test_adapter_reads_only_newest_treatment_frame():
    from src.algorithms.shac.residual_preview_adapter import (
        current_treatment_frame,
    )

    observations = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 15)

    actual = current_treatment_frame(
        observations, history_len=3, treatment_frame_dim=5
    )

    np.testing.assert_array_equal(actual, observations.reshape(2, 3, 5)[:, -1])


def test_composite_has_observation_feedback_but_zero_parent_parameter_gradient():
    from src.algorithms.shac.residual_preview_adapter import (
        apply_frozen_preview_residual,
    )

    parent_actor, residual_actor, params = _toy_policy()
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 20.0

    def parameter_loss(candidate_params):
        action, _, _ = apply_frozen_preview_residual(
            parent_actor,
            residual_actor,
            candidate_params,
            observations,
            history_len=3,
            treatment_frame_dim=5,
        )
        return jnp.sum(action)

    parameter_gradients = jax.grad(parameter_loss)(params)
    parent_norm = sum(
        float(jnp.sum(jnp.square(leaf)))
        for leaf in jax.tree_util.tree_leaves(parameter_gradients.parent)
    )
    adapter_norm = sum(
        float(jnp.sum(jnp.square(leaf)))
        for leaf in jax.tree_util.tree_leaves(parameter_gradients.adapter)
    )

    def observation_loss(candidate_observations):
        action, _, _ = apply_frozen_preview_residual(
            parent_actor,
            residual_actor,
            params,
            candidate_observations,
            history_len=3,
            treatment_frame_dim=5,
        )
        return jnp.sum(action)

    observation_gradient = jax.grad(observation_loss)(observations)

    assert parent_norm == 0.0
    assert adapter_norm > 0.0
    assert float(jnp.linalg.norm(observation_gradient)) > 0.0


def test_mask_selects_every_adapter_scalar_and_no_parent_scalar():
    from src.algorithms.shac.residual_preview_adapter import (
        build_residual_adapter_mask,
    )

    _, _, params = _toy_policy()
    mask = build_residual_adapter_mask(params)

    assert not any(bool(jnp.any(leaf)) for leaf in jax.tree_util.tree_leaves(mask.parent))
    assert all(bool(jnp.all(leaf)) for leaf in jax.tree_util.tree_leaves(mask.adapter))
    assert sum(
        int(np.count_nonzero(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(mask.adapter)
    ) == sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(params.adapter)
    )


def test_optimizer_migration_preserves_parent_state_and_zeros_adapter_moments():
    from src.algorithms.shac.residual_preview_adapter import (
        initialize_residual_adapter_optimizer,
    )

    _, _, params = _toy_policy()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.linear_schedule(1e-3, 5e-4, 20)),
    )
    parent_optimizer = optimizer.init(params.parent)
    parent_gradients = jax.tree_util.tree_map(jnp.ones_like, params.parent)
    _, parent_optimizer = optimizer.update(
        parent_gradients, parent_optimizer, params.parent
    )

    candidate_optimizer = initialize_residual_adapter_optimizer(
        optimizer,
        parent_optimizer_state=parent_optimizer,
        composite_params=params,
    )
    parent_adam = _adam_state(parent_optimizer)
    candidate_adam = _adam_state(candidate_optimizer)

    assert np.array_equal(
        np.asarray(candidate_adam.count), np.asarray(parent_adam.count)
    )
    assert _tree_arrays_equal(candidate_adam.mu.parent, parent_adam.mu)
    assert _tree_arrays_equal(candidate_adam.nu.parent, parent_adam.nu)
    assert all(
        bool(jnp.all(leaf == 0.0))
        for leaf in jax.tree_util.tree_leaves(candidate_adam.mu.adapter)
    )
    assert all(
        bool(jnp.all(leaf == 0.0))
        for leaf in jax.tree_util.tree_leaves(candidate_adam.nu.adapter)
    )
    assert _tree_arrays_equal(candidate_optimizer[0], parent_optimizer[0])
    assert _tree_arrays_equal(candidate_optimizer[1][1], parent_optimizer[1][1])


def test_migration_report_detects_parent_or_adapter_moment_drift():
    from src.algorithms.shac.residual_preview_adapter import (
        initialize_residual_adapter_optimizer,
        residual_adapter_migration_report,
    )

    parent_actor, residual_actor, params = _toy_policy()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-3)
    )
    parent_optimizer = optimizer.init(params.parent)
    candidate_optimizer = initialize_residual_adapter_optimizer(
        optimizer,
        parent_optimizer_state=parent_optimizer,
        composite_params=params,
    )
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 20.0

    valid = residual_adapter_migration_report(
        parent_actor=parent_actor,
        residual_actor=residual_actor,
        parent_params=params.parent,
        parent_optimizer_state=parent_optimizer,
        candidate_params=params,
        candidate_optimizer_state=candidate_optimizer,
        normalized_observations=observations,
        history_len=3,
        treatment_frame_dim=5,
    )
    assert valid["valid"] is True
    assert valid["max_action_absolute_error"] == 0.0

    changed_parent = jax.tree_util.tree_map(jnp.array, params.parent)
    changed_parent["params"]["Dense_0"]["bias"] = (
        changed_parent["params"]["Dense_0"]["bias"].at[0].add(1.0)
    )
    invalid_params = params._replace(parent=changed_parent)
    invalid = residual_adapter_migration_report(
        parent_actor=parent_actor,
        residual_actor=residual_actor,
        parent_params=params.parent,
        parent_optimizer_state=parent_optimizer,
        candidate_params=invalid_params,
        candidate_optimizer_state=candidate_optimizer,
        normalized_observations=observations,
        history_len=3,
        treatment_frame_dim=5,
    )
    assert invalid["valid"] is False
    assert invalid["parent_parameters_exact"] is False
