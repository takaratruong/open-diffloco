from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.core.data_structures import NormState
from src.core.networks import Critic
from tools.compare_g1_future_preview_critic import (
    CONFIRMATION_PHASES,
    build_paired_dataset,
    concatenate_paired,
    fit_critic_arm,
    future_preview_advances,
    future_preview_rows,
    migrate_critic_input,
    row_phases,
    validate_initial_equivalence,
)


class FakePreviewEnv:
    reference_length = 8
    reference_stride = 1
    actor_future_reference_dim = 2

    def _future_reference_command(self, phase):
        value = jnp.asarray(phase, dtype=jnp.float32)
        return jnp.stack((value, value + 10.0))


def test_row_phases_match_clamped_source_step_progression():
    np.testing.assert_array_equal(
        row_phases(5, 5, reference_length=8, reference_stride=1),
        np.array([5, 6, 7, 7, 7]),
    )


def test_future_preview_rows_use_actor_suffix_normalization_in_row_order():
    state = NormState(
        mean=jnp.array([99.0, 98.0, 1.0, 11.0]),
        var=jnp.array([4.0, 4.0, 4.0, 9.0]),
        count=jnp.array(12.0),
    )
    rows = future_preview_rows(
        FakePreviewEnv(),
        start_phase=5,
        count=3,
        actor_normalizer_state=state,
    )
    expected_raw = np.array([[5.0, 15.0], [6.0, 16.0], [7.0, 17.0]])
    expected = (expected_raw - np.array([1.0, 11.0])) / np.sqrt(
        np.array([4.0, 9.0]) + 1e-4
    )
    np.testing.assert_allclose(rows, expected, rtol=1e-6, atol=1e-6)


def test_migration_appends_zero_kernel_and_adam_rows_with_equal_predictions():
    critic = Critic(hidden=(4, 3))
    old_inputs = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10.0
    old_params = critic.init(jax.random.PRNGKey(0), old_inputs)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    old_opt = optimizer.init(old_params)

    new_params, new_opt = migrate_critic_input(
        old_params,
        old_opt,
        extra_dim=2,
        optimizer=optimizer,
    )
    old_kernel = old_params["params"]["Dense_0"]["kernel"]
    new_kernel = new_params["params"]["Dense_0"]["kernel"]
    assert new_kernel.shape == (5, 4)
    np.testing.assert_array_equal(new_kernel[:3], old_kernel)
    np.testing.assert_array_equal(new_kernel[3:], np.zeros((2, 4)))

    augmented = jnp.concatenate(
        (old_inputs, jax.random.normal(jax.random.PRNGKey(1), (5, 2))), axis=-1
    )
    np.testing.assert_allclose(
        critic.apply(old_params, old_inputs),
        critic.apply(new_params, augmented),
        rtol=1e-6,
        atol=1e-6,
    )
    old_leaves = jax.tree_util.tree_leaves(old_opt)
    new_leaves = jax.tree_util.tree_leaves(new_opt)
    shape_changes = [
        (old.shape, new.shape)
        for old, new in zip(old_leaves, new_leaves)
        if old.shape != new.shape
    ]
    assert shape_changes == [((3, 4), (5, 4)), ((3, 4), (5, 4))]
    assert int(new_opt[1][0].count) == int(old_opt[1][0].count)


def test_future_preview_gate_requires_absolute_pass_and_no_paired_regression():
    assert CONFIRMATION_PHASES == (15, 115, 215, 315, 415)
    original = {"rank_correlation": 0.5, "nrmse": 1.5}
    baseline = {"rank_correlation": 0.91, "nrmse": 0.31}
    preview = {"rank_correlation": 0.95, "nrmse": 0.20}
    baseline_h12 = [
        {"phase": phase, "relative_error": value}
        for phase, value in zip(CONFIRMATION_PHASES, (0.1, 0.31, 0.2, 0.1, 0.2))
    ]
    preview_h12 = [
        {"phase": phase, "relative_error": value}
        for phase, value in zip(CONFIRMATION_PHASES, (0.05, 0.2, 0.15, 0.08, 0.1))
    ]
    assert future_preview_advances(
        original,
        baseline,
        preview,
        baseline_h12,
        preview_h12,
    )
    regressed = [dict(row) for row in preview_h12]
    regressed[0]["relative_error"] = 0.11
    assert not future_preview_advances(
        original,
        baseline,
        preview,
        baseline_h12,
        regressed,
    )
    misses_absolute = dict(preview, nrmse=0.26)
    assert not future_preview_advances(
        original,
        baseline,
        misses_absolute,
        baseline_h12,
        preview_h12,
    )


def test_build_paired_dataset_appends_only_normalized_future_features():
    env = FakePreviewEnv()
    env.critic_obs_dim = 3
    state = SimpleNamespace(
        critic_normalizer=NormState(
            mean=jnp.array([1.0, 2.0, 3.0]),
            var=jnp.array([1.0, 4.0, 9.0]),
            count=jnp.array(20.0),
        ),
        normalizer=NormState(
            mean=jnp.array([99.0, 98.0, 1.0, 11.0]),
            var=jnp.array([4.0, 4.0, 4.0, 9.0]),
            count=jnp.array(20.0),
        ),
    )
    raw = {
        5: {
            "critic_observations": np.array(
                [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float32
            ),
            "rewards": np.array([1.0, 0.5]),
            "returns": np.array([1.5, 0.5]),
        }
    }
    paired = build_paired_dataset(env, state, raw)
    assert paired[5]["control_observations"].shape == (2, 3)
    assert paired[5]["preview_observations"].shape == (2, 5)
    np.testing.assert_array_equal(
        paired[5]["preview_observations"][:, :3],
        paired[5]["control_observations"],
    )
    np.testing.assert_array_equal(paired[5]["returns"], raw[5]["returns"])


def test_fit_critic_arm_executes_exact_requested_update_count():
    critic = Critic(hidden=(4, 3))
    observations = np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0
    returns = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    params = critic.init(jax.random.PRNGKey(4), jnp.asarray(observations))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    opt_state = optimizer.init(params)
    fitted_params, fitted_opt, loss = fit_critic_arm(
        critic,
        params,
        opt_state,
        observations,
        returns,
        steps=3,
        optimizer=optimizer,
    )
    assert int(fitted_opt[1][0].count) == int(opt_state[1][0].count) + 3
    assert np.isfinite(loss)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(fitted_params),
        )
    )


def test_concatenate_paired_preserves_declared_phase_order():
    paired = {
        2: {
            "control_observations": np.full((2, 1), 2.0),
            "preview_observations": np.full((2, 2), 20.0),
            "returns": np.array([2.5, 2.0]),
        },
        1: {
            "control_observations": np.full((1, 1), 1.0),
            "preview_observations": np.full((1, 2), 10.0),
            "returns": np.array([1.0]),
        },
    }
    observations, returns = concatenate_paired(
        paired, (1, 2), observation_key="control_observations"
    )
    np.testing.assert_array_equal(observations[:, 0], [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(returns, [1.0, 2.5, 2.0])


def test_initial_equivalence_fails_closed_above_tolerance():
    assert validate_initial_equivalence(
        np.array([1.0, 2.0]), np.array([1.0, 2.0 + 5e-7]), tolerance=1e-6
    ) == pytest.approx(5e-7)
    with pytest.raises(ValueError, match="initial predictions"):
        validate_initial_equivalence(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0 + 2e-6]),
            tolerance=1e-6,
        )
