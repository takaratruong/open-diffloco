from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualOptState,
    FrozenControllerResidualParams,
    apply_frozen_controller_residual,
    migrate_frozen_controller_residual,
    update_frozen_controller_residual,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
)


def test_nested_residual_resume_skips_plain_preview_validator() -> None:
    from src.algorithms.shac.algorithm import (
        requires_plain_residual_preview_resume_validation,
    )

    assert requires_plain_residual_preview_resume_validation(
        actor_residual_preview_adapter=True,
        actor_frozen_controller_residual=False,
        actor_learned_torso_wrench=False,
        learned_wrench_state=False,
    )
    assert not requires_plain_residual_preview_resume_validation(
        actor_residual_preview_adapter=True,
        actor_frozen_controller_residual=True,
        actor_learned_torso_wrench=False,
        learned_wrench_state=False,
    )


def _parent() -> FrozenPreviewResidualParams:
    return FrozenPreviewResidualParams(
        parent={"weights": jnp.arange(8 * 29, dtype=jnp.float32).reshape(8, 29) / 1000},
        adapter={"bias": jnp.linspace(-0.1, 0.1, 29, dtype=jnp.float32)},
    )


def _parent_apply(params, observation):
    return observation @ params.parent["weights"] + params.adapter["bias"]


def _migrated():
    parent = _parent()
    parent_optimizer_state = {"count": jnp.asarray(17, dtype=jnp.int32)}
    adapter = PreviewResidualAdapter(action_dim=29, hidden_dim=16)
    optimizer = optax.adam(1e-3)
    observation = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 20
    params, state, report = migrate_frozen_controller_residual(
        parent_params=parent,
        parent_optimizer_state=parent_optimizer_state,
        parent_apply=_parent_apply,
        adapter_actor=adapter,
        adapter_optimizer=optimizer,
        rng=jax.random.PRNGKey(4),
        normalized_observations=observation,
        history_len=2,
        frame_dim=4,
    )
    return parent, parent_optimizer_state, adapter, optimizer, observation, params, state, report


def test_migration_preserves_complete_parent_action_bit_exactly() -> None:
    parent, parent_opt, adapter, _, observation, params, state, report = _migrated()

    actual, parent_action, residual = apply_frozen_controller_residual(
        _parent_apply,
        adapter,
        params,
        observation,
        history_len=2,
        frame_dim=4,
    )
    expected = _parent_apply(parent, observation)

    np.testing.assert_array_equal(parent_action, expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(residual, np.zeros((2, 29), np.float32))
    assert report == {
        "parent_action_exact": True,
        "residual_action_zero": True,
        "parent_optimizer_preserved": True,
        "valid": True,
    }
    assert state.parent_optimizer_state is parent_opt


def test_second_migration_preserves_complete_nested_parent_bit_exactly() -> None:
    (
        _parent_params,
        _parent_opt,
        adapter,
        optimizer,
        observation,
        first_params,
        first_state,
        _report,
    ) = _migrated()

    def apply_controller(params, observations):
        if isinstance(params, FrozenPreviewResidualParams):
            return _parent_apply(params, observations)
        return apply_frozen_controller_residual(
            apply_controller,
            adapter,
            params,
            observations,
            history_len=2,
            frame_dim=4,
        )[0]

    second_params, second_state, report = migrate_frozen_controller_residual(
        parent_params=first_params,
        parent_optimizer_state=first_state,
        parent_apply=apply_controller,
        adapter_actor=adapter,
        adapter_optimizer=optimizer,
        rng=jax.random.PRNGKey(8),
        normalized_observations=observation,
        history_len=2,
        frame_dim=4,
    )

    expected = apply_controller(first_params, observation)
    actual = apply_controller(second_params, observation)
    np.testing.assert_array_equal(actual, expected)
    assert second_params.parent is first_params
    assert second_state.parent_optimizer_state is first_state
    assert report["valid"] is True


def test_only_new_adapter_changes_after_update() -> None:
    _, parent_opt, _, optimizer, _, params, state, _ = _migrated()
    gradients = FrozenControllerResidualParams(
        parent=jax.tree.map(jnp.ones_like, params.parent),
        adapter=jax.tree.map(jnp.ones_like, params.adapter),
    )

    updates, next_state = update_frozen_controller_residual(
        gradients=gradients,
        optimizer_state=state,
        params=params,
        adapter_optimizer=optimizer,
    )
    candidate = optax.apply_updates(params, updates)

    for before, after in zip(
        jax.tree.leaves(params.parent), jax.tree.leaves(candidate.parent)
    ):
        np.testing.assert_array_equal(before, after)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(params.adapter), jax.tree.leaves(candidate.adapter)
        )
    )
    assert next_state.parent_optimizer_state is parent_opt


def test_application_rejects_invalid_parent_and_wrong_action_width() -> None:
    _, _, adapter, _, observation, params, _, _ = _migrated()
    invalid = FrozenControllerResidualParams(
        parent={"not": "e026"}, adapter=params.adapter
    )
    with pytest.raises(ValueError, match="frozen-preview base"):
        apply_frozen_controller_residual(
            _parent_apply,
            adapter,
            invalid,
            observation,
            history_len=2,
            frame_dim=4,
        )

    with pytest.raises(ValueError, match="29 actions"):
        apply_frozen_controller_residual(
            lambda _params, obs: jnp.zeros(obs.shape[:-1] + (28,)),
            adapter,
            params,
            observation,
            history_len=2,
            frame_dim=4,
        )


def test_update_rejects_wrong_optimizer_state() -> None:
    _, _, _, optimizer, _, params, _, _ = _migrated()
    gradients = jax.tree.map(jnp.ones_like, params)
    with pytest.raises(ValueError, match="optimizer state"):
        update_frozen_controller_residual(
            gradients=gradients,
            optimizer_state=object(),
            params=params,
            adapter_optimizer=optimizer,
        )


def test_optimizer_state_is_registered_pytree() -> None:
    _, _, _, _, _, params, state, _ = _migrated()
    assert isinstance(params, FrozenControllerResidualParams)
    assert isinstance(state, FrozenControllerResidualOptState)
    leaves = jax.tree.leaves(state)
    assert leaves


def test_nested_resume_mask_builds_production_shaped_forward_direction() -> None:
    from src.algorithms.shac.algorithm import build_masked_rademacher_direction
    from src.algorithms.shac.frozen_controller_residual import (
        build_frozen_controller_residual_mask,
    )

    params = FrozenControllerResidualParams(
        parent=_parent(),
        adapter={
            "params": {
                "Dense_0": {
                    "kernel": jnp.zeros((328, 256), dtype=jnp.float32),
                    "bias": jnp.zeros((256,), dtype=jnp.float32),
                },
                "Dense_1": {
                    "kernel": jnp.zeros((256, 29), dtype=jnp.float32),
                    "bias": jnp.zeros((29,), dtype=jnp.float32),
                },
            }
        },
    )

    mask = build_frozen_controller_residual_mask(params)
    direction = build_masked_rademacher_direction(params, mask, seed=20260904)

    assert jax.tree.structure(mask) == jax.tree.structure(params)
    assert jax.tree.structure(direction) == jax.tree.structure(params)
    assert not any(bool(jnp.any(leaf)) for leaf in jax.tree.leaves(mask.parent))
    assert all(bool(jnp.all(leaf)) for leaf in jax.tree.leaves(mask.adapter))
    assert not any(
        bool(jnp.any(leaf)) for leaf in jax.tree.leaves(direction.parent)
    )
    assert all(
        bool(jnp.all(leaf != 0.0))
        for leaf in jax.tree.leaves(direction.adapter)
    )
    assert sum(
        int(np.count_nonzero(np.asarray(leaf)))
        for leaf in jax.tree.leaves(mask)
    ) == 91_677
    assert float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(leaf))
                for leaf in jax.tree.leaves(direction)
            )
        )
    ) == pytest.approx(1.0, abs=5e-5)
