from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze
import numpy as np
import optax
import pytest

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


def _optimizer_counts(state):
    count_types = (
        optax.ScaleByAdamState,
        optax.ScaleByScheduleState,
        optax.contrib.MuonState,
    )
    counts = []

    def visit(value):
        if isinstance(value, count_types):
            counts.append(int(np.asarray(value.count)))
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(state)
    return sorted(counts)


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


def test_conditioned_adapter_reads_scalar_without_changing_parent_input():
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
    )

    parent_actor, _, legacy_params = _toy_policy()
    residual_actor = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    adapter = residual_actor.init(
        jax.random.PRNGKey(31), jnp.zeros((1, 6), dtype=jnp.float32)
    )
    adapter = jax.tree_util.tree_map(jnp.array, adapter)
    adapter["params"]["Dense_0"]["kernel"] = (
        jnp.zeros_like(adapter["params"]["Dense_0"]["kernel"])
        .at[-1]
        .set(1.0)
    )
    adapter["params"]["Dense_1"]["kernel"] = jnp.ones((4, 2))
    params = FrozenPreviewResidualParams(
        parent=legacy_params.parent,
        adapter=adapter,
    )
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 10.0

    zero_action, zero_parent, zero_residual = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        params,
        observations,
        history_len=3,
        treatment_frame_dim=5,
        assistance_scale=jnp.array(0.0),
    )
    assisted_action, assisted_parent, assisted_residual = (
        apply_frozen_preview_residual(
            parent_actor,
            residual_actor,
            params,
            observations,
            history_len=3,
            treatment_frame_dim=5,
            assistance_scale=jnp.array(1.0),
        )
    )
    omitted_action, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        params,
        observations,
        history_len=3,
        treatment_frame_dim=5,
    )

    np.testing.assert_array_equal(zero_parent, assisted_parent)
    np.testing.assert_array_equal(zero_action, omitted_action)
    np.testing.assert_array_equal(zero_residual, jnp.zeros((1, 2)))
    assert bool(jnp.all(assisted_residual > 0.0))
    assert bool(jnp.all(assisted_action > zero_action))


def test_conditioned_adapter_rejects_scalar_shape_mismatch():
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
    )

    parent_actor, _, legacy_params = _toy_policy()
    residual_actor = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    adapter = residual_actor.init(
        jax.random.PRNGKey(37), jnp.zeros((2, 6), dtype=jnp.float32)
    )
    params = FrozenPreviewResidualParams(legacy_params.parent, adapter)
    observations = jnp.zeros((2, 15), dtype=jnp.float32)

    with pytest.raises(ValueError, match="assistance scale shape"):
        apply_frozen_preview_residual(
            parent_actor,
            residual_actor,
            params,
            observations,
            history_len=3,
            treatment_frame_dim=5,
            assistance_scale=jnp.zeros((3,), dtype=jnp.float32),
        )


@pytest.mark.parametrize("scale", [float("nan"), -0.1, 1.1])
def test_conditioned_adapter_fails_closed_on_invalid_scalar(scale: float):
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
    )

    parent_actor, _, legacy_params = _toy_policy()
    residual_actor = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    adapter = residual_actor.init(
        jax.random.PRNGKey(41), jnp.zeros((1, 6), dtype=jnp.float32)
    )
    params = FrozenPreviewResidualParams(legacy_params.parent, adapter)

    action, _, _ = jax.jit(
        lambda assistance: apply_frozen_preview_residual(
            parent_actor,
            residual_actor,
            params,
            jnp.zeros((1, 15), dtype=jnp.float32),
            history_len=3,
            treatment_frame_dim=5,
            assistance_scale=assistance,
        )
    )(jnp.asarray(scale, dtype=jnp.float32))

    assert not bool(jnp.all(jnp.isfinite(action)))


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


def test_assistance_conditioning_migration_appends_exact_zero_parameter_and_moments():
    from src.algorithms.shac.residual_preview_adapter import (
        apply_frozen_preview_residual,
        migrate_residual_adapter_assistance_conditioning,
        split_residual_adapter_params,
    )

    parent_actor, residual_actor, params = _toy_policy()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-3)
    )
    optimizer_state = optimizer.init(params)
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)
    _, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 20.0
    original_action, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        params,
        observations,
        history_len=3,
        treatment_frame_dim=5,
    )

    migrated_params, migrated_optimizer, report = (
        migrate_residual_adapter_assistance_conditioning(
            params=params,
            optimizer_state=optimizer_state,
        )
    )
    original_kernel, _ = split_residual_adapter_params(params.adapter)
    migrated_kernel, _ = split_residual_adapter_params(migrated_params.adapter)
    original_adam = _adam_state(optimizer_state)
    migrated_adam = _adam_state(migrated_optimizer)
    original_mu, _ = split_residual_adapter_params(
        original_adam.mu.adapter
    )
    migrated_mu, _ = split_residual_adapter_params(
        migrated_adam.mu.adapter
    )
    original_nu, _ = split_residual_adapter_params(
        original_adam.nu.adapter
    )
    migrated_nu, _ = split_residual_adapter_params(
        migrated_adam.nu.adapter
    )
    migrated_action, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        migrated_params,
        observations,
        history_len=3,
        treatment_frame_dim=5,
        assistance_scale=jnp.array(0.0),
    )

    assert migrated_kernel.shape == (6, 4)
    np.testing.assert_array_equal(migrated_kernel[:-1], original_kernel)
    np.testing.assert_array_equal(migrated_kernel[-1], jnp.zeros(4))
    np.testing.assert_array_equal(migrated_mu[:-1], original_mu)
    np.testing.assert_array_equal(migrated_mu[-1], jnp.zeros(4))
    np.testing.assert_array_equal(migrated_nu[:-1], original_nu)
    np.testing.assert_array_equal(migrated_nu[-1], jnp.zeros(4))
    np.testing.assert_array_equal(original_action, migrated_action)
    assert _tree_arrays_equal(params.parent, migrated_params.parent)
    assert _tree_arrays_equal(
        original_adam.mu.parent, migrated_adam.mu.parent
    )
    assert _tree_arrays_equal(
        original_adam.nu.parent, migrated_adam.nu.parent
    )
    assert int(original_adam.count) == int(migrated_adam.count)
    assert report["valid"] is True
    assert report["conditioning_rows"] == 1


def test_assistance_conditioning_migration_rejects_second_expansion():
    from src.algorithms.shac.residual_preview_adapter import (
        migrate_residual_adapter_assistance_conditioning,
    )

    _, _, params = _toy_policy()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-3)
    )
    optimizer_state = optimizer.init(params)
    migrated_params, migrated_optimizer, _ = (
        migrate_residual_adapter_assistance_conditioning(
            params=params,
            optimizer_state=optimizer_state,
        )
    )

    with pytest.raises(ValueError, match="expected adapter input width"):
        migrate_residual_adapter_assistance_conditioning(
            params=migrated_params,
            optimizer_state=migrated_optimizer,
            expected_input_dim=5,
        )


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


def test_residual_muon_partition_round_trips_registered_adapter_tree():
    from src.algorithms.shac.residual_preview_adapter import (
        merge_residual_adapter_params,
        split_residual_adapter_params,
    )

    _, _, params = _toy_policy()
    kernel, auxiliary = split_residual_adapter_params(params.adapter)

    assert kernel.shape == (5, 4)
    assert auxiliary.dense0_bias.shape == (4,)
    assert auxiliary.dense1_kernel.shape == (4, 2)
    assert auxiliary.dense1_bias.shape == (2,)
    rebuilt = merge_residual_adapter_params(
        params.adapter, kernel, auxiliary
    )
    assert _tree_arrays_equal(rebuilt, params.adapter)
    assert type(rebuilt) is type(params.adapter)

    frozen = freeze(params.adapter)
    frozen_kernel, frozen_auxiliary = split_residual_adapter_params(frozen)
    frozen_rebuilt = merge_residual_adapter_params(
        frozen, frozen_kernel, frozen_auxiliary
    )
    assert isinstance(frozen_rebuilt, FrozenDict)
    assert _tree_arrays_equal(frozen_rebuilt, frozen)


def test_residual_muon_partition_rejects_unregistered_or_nonmatrix_kernel():
    from src.algorithms.shac.residual_preview_adapter import (
        split_residual_adapter_params,
    )

    _, _, params = _toy_policy()
    layers = params.adapter["params"]
    missing_kernel = {
        "params": {
            "Dense_0": {"bias": layers["Dense_0"]["bias"]},
            "Dense_1": layers["Dense_1"],
        }
    }
    wrong_rank = {
        "params": {
            "Dense_0": {
                "kernel": layers["Dense_0"]["kernel"].reshape(-1),
                "bias": layers["Dense_0"]["bias"],
            },
            "Dense_1": layers["Dense_1"],
        }
    }

    with pytest.raises(ValueError, match="only kernel and bias"):
        split_residual_adapter_params(missing_kernel)
    with pytest.raises(ValueError, match="must be a matrix"):
        split_residual_adapter_params(wrong_rank)


def test_residual_muon_initialization_inherits_counts_and_zeroes_momenta():
    from src.algorithms.shac.residual_preview_adapter import (
        build_residual_muon_optimizers,
        initialize_residual_muon_optimizer,
        residual_muon_migration_report,
    )

    _, _, params = _toy_policy()
    schedule = optax.linear_schedule(1e-3, 5e-4, 20)
    parent_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(schedule)
    )
    parent_state = parent_optimizer.init(params.parent)
    parent_gradients = jax.tree_util.tree_map(jnp.ones_like, params.parent)
    _, parent_state = parent_optimizer.update(
        parent_gradients, parent_state, params.parent
    )
    muon_optimizer, adam_optimizer = build_residual_muon_optimizers(schedule)

    candidate_state = initialize_residual_muon_optimizer(
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        parent_optimizer_state=parent_state,
        adapter_params=params.adapter,
    )
    report = residual_muon_migration_report(
        parent_optimizer_state=parent_state,
        candidate_optimizer_state=candidate_state,
    )

    assert _tree_arrays_equal(
        candidate_state.parent_optimizer_state, parent_state
    )
    assert len(_optimizer_counts(candidate_state.muon_state)) >= 2
    assert set(_optimizer_counts(candidate_state.muon_state)) == {1}
    assert _optimizer_counts(candidate_state.adam_state) == [1, 1]
    assert report["parent_optimizer_snapshot_exact"] is True
    assert report["muon_momentum_zero"] is True
    assert report["adam_mu_zero"] is True
    assert report["adam_nu_zero"] is True
    assert report["optimizer_counts_exact"] is True
    assert report["valid"] is True

    changed_parent_adam = _adam_state(parent_state)._replace(
        count=_adam_state(parent_state).count + 1
    )
    changed_parent_snapshot = (
        parent_state[0],
        (changed_parent_adam, parent_state[1][1]),
    )
    snapshot_drift = residual_muon_migration_report(
        parent_optimizer_state=parent_state,
        candidate_optimizer_state=candidate_state._replace(
            parent_optimizer_state=changed_parent_snapshot
        ),
    )
    reset_muon_counts = jax.tree_util.tree_map(
        lambda value: value._replace(count=jnp.asarray(0, value.count.dtype))
        if isinstance(
            value,
            (
                optax.ScaleByAdamState,
                optax.ScaleByScheduleState,
                optax.contrib.MuonState,
            ),
        )
        else value,
        candidate_state.muon_state,
        is_leaf=lambda value: isinstance(
            value,
            (
                optax.ScaleByAdamState,
                optax.ScaleByScheduleState,
                optax.contrib.MuonState,
            ),
        ),
    )
    count_reset = residual_muon_migration_report(
        parent_optimizer_state=parent_state,
        candidate_optimizer_state=candidate_state._replace(
            muon_state=reset_muon_counts
        ),
    )
    assert snapshot_drift["parent_optimizer_snapshot_exact"] is False
    assert snapshot_drift["valid"] is False
    assert count_reset["optimizer_counts_exact"] is False
    assert count_reset["valid"] is False


def test_residual_muon_update_clips_once_and_preserves_parent_snapshot():
    from src.algorithms.shac.residual_preview_adapter import (
        apply_residual_muon_update,
        build_residual_muon_optimizers,
        initialize_residual_muon_optimizer,
        split_residual_adapter_params,
    )

    _, _, params = _toy_policy()
    schedule = optax.linear_schedule(1e-3, 5e-4, 20)
    parent_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(schedule)
    )
    parent_state = parent_optimizer.init(params.parent)
    parent_gradients = jax.tree_util.tree_map(jnp.ones_like, params.parent)
    _, parent_state = parent_optimizer.update(
        parent_gradients, parent_state, params.parent
    )
    muon_optimizer, adam_optimizer = build_residual_muon_optimizers(schedule)
    state = initialize_residual_muon_optimizer(
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        parent_optimizer_state=parent_state,
        adapter_params=params.adapter,
    )
    gradients = params._replace(
        parent=jax.tree_util.tree_map(
            lambda value: jnp.full_like(value, 7.0), params.parent
        ),
        adapter=jax.tree_util.tree_map(
            lambda value: jnp.full_like(value, 3.0), params.adapter
        ),
    )
    clipped_adapter, _ = optax.clip_by_global_norm(1.0).update(
        gradients.adapter, optax.EmptyState()
    )
    _, clipped_auxiliary = split_residual_adapter_params(clipped_adapter)
    _, adapter_auxiliary = split_residual_adapter_params(params.adapter)
    expected_aux_updates, _ = adam_optimizer.update(
        clipped_auxiliary,
        state.adam_state,
        adapter_auxiliary,
    )

    updates, new_state, diagnostics = apply_residual_muon_update(
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        gradients=gradients,
        optimizer_state=state,
        params=params,
    )
    kernel_update, auxiliary_updates = split_residual_adapter_params(
        updates.adapter
    )

    assert all(
        bool(jnp.all(leaf == 0.0))
        for leaf in jax.tree_util.tree_leaves(updates.parent)
    )
    assert _tree_arrays_equal(auxiliary_updates, expected_aux_updates)
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(kernel_update)
    )
    assert float(jnp.linalg.norm(kernel_update)) > 0.0
    assert _tree_arrays_equal(
        new_state.parent_optimizer_state, parent_state
    )
    assert float(diagnostics["muon_kernel_update_norm"]) > 0.0
    assert float(diagnostics["aux_adam_update_norm"]) > 0.0
    assert float(diagnostics["frozen_update_max_abs"]) == 0.0
    assert float(diagnostics["frozen_moment_drift_max_abs"]) == 0.0
