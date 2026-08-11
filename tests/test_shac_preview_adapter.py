import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.data_structures import Normalizer


def _params(rows=6, hidden=4):
    return {
        "params": {
            "Dense_0": {
                "kernel": jnp.arange(rows * hidden, dtype=jnp.float32).reshape(
                    rows, hidden
                ),
                "bias": jnp.arange(hidden, dtype=jnp.float32),
            },
            "Dense_1": {
                "kernel": jnp.ones((hidden, 2), dtype=jnp.float32),
                "bias": jnp.zeros(2, dtype=jnp.float32),
            },
        }
    }


def _assert_trees_equal(first, second):
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for actual, expected in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_mask_selects_only_newest_preview_rows():
    from src.algorithms.shac.preview_adapter import (
        build_current_preview_mask,
    )

    params = _params(rows=2 * 3)
    mask = build_current_preview_mask(
        params,
        history_len=2,
        legacy_frame_dim=2,
        treatment_frame_dim=3,
    )

    kernel = mask["params"]["Dense_0"]["kernel"].reshape(2, 3, 4)
    assert int(jnp.sum(kernel)) == 4
    np.testing.assert_array_equal(kernel[0], False)
    np.testing.assert_array_equal(kernel[1, :2], False)
    np.testing.assert_array_equal(kernel[1, 2:], True)
    assert not bool(jnp.any(mask["params"]["Dense_0"]["bias"]))
    assert not bool(jnp.any(mask["params"]["Dense_1"]["kernel"]))


def test_mask_rejects_a_kernel_that_does_not_match_history_layout():
    from src.algorithms.shac.preview_adapter import (
        build_current_preview_mask,
    )

    with np.testing.assert_raises_regex(ValueError, "history layout"):
        build_current_preview_mask(
            _params(rows=5),
            history_len=2,
            legacy_frame_dim=2,
            treatment_frame_dim=3,
        )


def test_masked_adam_preserves_frozen_values_and_inherited_moments():
    from src.algorithms.shac.preview_adapter import (
        apply_preview_adapter_update,
        build_current_preview_mask,
    )

    params = _params()
    mask = build_current_preview_mask(
        params,
        history_len=2,
        legacy_frame_dim=2,
        treatment_frame_dim=3,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-2)
    )
    state = optimizer.init(params)
    _, state = optimizer.update(
        jax.tree_util.tree_map(jnp.ones_like, params), state, params
    )
    original_params = jax.tree_util.tree_map(jnp.array, params)
    original_mu = jax.tree_util.tree_map(jnp.array, state[1][0].mu)
    original_nu = jax.tree_util.tree_map(jnp.array, state[1][0].nu)
    original_count = int(state[1][0].count)

    for _ in range(3):
        gradients = jax.tree_util.tree_map(
            lambda value: jnp.full_like(value, 2.0), params
        )
        updates, state, diagnostics = apply_preview_adapter_update(
            optimizer, gradients, state, params, mask
        )
        params = optax.apply_updates(params, updates)
        assert float(diagnostics["frozen_update_max_abs"]) == 0.0
        assert float(diagnostics["frozen_moment_drift_max_abs"]) == 0.0
        assert float(diagnostics["preview_gradient_norm"]) > 0.0
        assert float(diagnostics["preview_update_norm"]) > 0.0

    frozen = jax.tree_util.tree_map(
        lambda value, selected: jnp.where(selected, 0.0, value),
        params,
        mask,
    )
    expected = jax.tree_util.tree_map(
        lambda value, selected: jnp.where(selected, 0.0, value),
        original_params,
        mask,
    )
    _assert_trees_equal(frozen, expected)
    frozen_mu = jax.tree_util.tree_map(
        lambda value, selected: jnp.where(selected, 0.0, value),
        state[1][0].mu,
        mask,
    )
    expected_mu = jax.tree_util.tree_map(
        lambda value, selected: jnp.where(selected, 0.0, value),
        original_mu,
        mask,
    )
    _assert_trees_equal(frozen_mu, expected_mu)
    frozen_nu = jax.tree_util.tree_map(
        lambda value, selected: jnp.where(selected, 0.0, value),
        state[1][0].nu,
        mask,
    )
    expected_nu = jax.tree_util.tree_map(
        lambda value, selected: jnp.where(selected, 0.0, value),
        original_nu,
        mask,
    )
    _assert_trees_equal(frozen_nu, expected_nu)
    assert int(state[1][0].count) == original_count + 3


def test_frozen_state_audit_ignores_authorized_rows():
    from src.algorithms.shac.preview_adapter import (
        build_current_preview_mask,
        frozen_preview_state_drift,
    )

    parent_params = _params()
    mask = build_current_preview_mask(
        parent_params,
        history_len=2,
        legacy_frame_dim=2,
        treatment_frame_dim=3,
    )
    candidate_params = jax.tree_util.tree_map(
        lambda value, selected: value + selected.astype(value.dtype),
        parent_params,
        mask,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-2)
    )
    parent_opt = optimizer.init(parent_params)
    candidate_opt = optimizer.init(candidate_params)
    parent_normalizer = Normalizer(3).init()

    report = frozen_preview_state_drift(
        parent_params,
        candidate_params,
        parent_opt,
        candidate_opt,
        parent_normalizer,
        parent_normalizer,
        mask,
    )

    assert report == {
        "frozen_parameter_max_abs": 0.0,
        "frozen_mu_max_abs": 0.0,
        "frozen_nu_max_abs": 0.0,
        "actor_normalizer_max_abs": 0.0,
        "valid": True,
    }


def test_frozen_state_audit_detects_a_legacy_parameter_change():
    from src.algorithms.shac.preview_adapter import (
        build_current_preview_mask,
        frozen_preview_state_drift,
    )

    parent_params = _params()
    mask = build_current_preview_mask(
        parent_params,
        history_len=2,
        legacy_frame_dim=2,
        treatment_frame_dim=3,
    )
    candidate_params = {
        **parent_params,
        "params": {
            **parent_params["params"],
            "Dense_0": {
                **parent_params["params"]["Dense_0"],
                "kernel": parent_params["params"]["Dense_0"]["kernel"].at[
                    0, 0
                ].add(0.5),
            },
        },
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-2)
    )
    parent_opt = optimizer.init(parent_params)
    candidate_opt = optimizer.init(candidate_params)
    normalizer = Normalizer(3).init()

    report = frozen_preview_state_drift(
        parent_params,
        candidate_params,
        parent_opt,
        candidate_opt,
        normalizer,
        normalizer,
        mask,
    )

    assert report["frozen_parameter_max_abs"] == 0.5
    assert not report["valid"]


def test_zero_current_preview_removes_only_newest_suffix():
    from src.algorithms.shac.preview_adapter import zero_current_preview

    observations = jnp.arange(2 * 2 * 3, dtype=jnp.float32).reshape(2, -1)

    result = zero_current_preview(
        observations,
        history_len=2,
        legacy_frame_dim=2,
        treatment_frame_dim=3,
    )

    frames = result.reshape(2, 2, 3)
    original = observations.reshape(2, 2, 3)
    np.testing.assert_array_equal(frames[:, 0], original[:, 0])
    np.testing.assert_array_equal(frames[:, 1, :2], original[:, 1, :2])
    np.testing.assert_array_equal(frames[:, 1, 2:], 0.0)


def test_phase_binned_action_deviation_reports_mean_max_counts_and_validity():
    from src.algorithms.shac.preview_adapter import (
        phase_binned_action_deviation,
    )

    candidate = jnp.array([[[1.0]], [[4.0]], [[8.0]], [[16.0]]])
    parent = jnp.zeros_like(candidate)

    result = phase_binned_action_deviation(
        candidate,
        parent,
        jnp.array([[0], [2], [4], [7]]),
        phase_count=8,
        bin_count=2,
    )

    np.testing.assert_array_equal(result["bin_counts"], [2, 2])
    np.testing.assert_allclose(result["mean_abs"], [2.5, 12.0])
    np.testing.assert_allclose(result["max_abs"], [4.0, 16.0])
    assert bool(result["valid"])


def test_phase_binned_action_deviation_rejects_nonfinite_or_empty_bins():
    from src.algorithms.shac.preview_adapter import (
        phase_binned_action_deviation,
    )

    result = phase_binned_action_deviation(
        jnp.array([[1.0], [jnp.nan]]),
        jnp.zeros((2, 1)),
        jnp.array([0, 1]),
        phase_count=8,
        bin_count=2,
    )

    assert not bool(result["valid"])
