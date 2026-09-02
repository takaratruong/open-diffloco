from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class _FakeReferenceEnv:
    def reset(self, key, difficulty, phase_sampler_failed_count=None):
        marker = jax.random.randint(key, (), 0, 1_000_000)
        state = {
            "marker": marker,
            "difficulty": difficulty,
            "history": jnp.full((2,), marker),
        }
        if phase_sampler_failed_count is not None:
            state["failed_count"] = phase_sampler_failed_count
        return state


def test_fresh_reference_fraction_validation_and_exact_count() -> None:
    from src.algorithms.shac.fresh_reference import (
        fresh_reference_count,
        validate_fresh_reference_fraction,
    )

    for fraction in (0.0, 0.25, 1.0):
        assert validate_fresh_reference_fraction(fraction) == fraction
    for invalid in (True, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_fresh_reference_fraction(invalid)

    assert fresh_reference_count(0.0, population_size=512) == 0
    assert fresh_reference_count(0.25, population_size=512) == 128
    assert fresh_reference_count(1.0, population_size=512) == 512
    with pytest.raises(ValueError):
        fresh_reference_count(0.25, population_size=0)


def test_fixed_count_mask_samples_without_replacement() -> None:
    from src.algorithms.shac.fresh_reference import sample_fixed_count_mask

    first = sample_fixed_count_mask(jax.random.PRNGKey(7), 16, count=4)
    repeat = sample_fixed_count_mask(jax.random.PRNGKey(7), 16, count=4)
    other = sample_fixed_count_mask(jax.random.PRNGKey(8), 16, count=4)

    np.testing.assert_array_equal(first, repeat)
    assert int(first.sum()) == 4
    assert int(other.sum()) == 4
    assert not np.array_equal(first, other)


def test_refresh_population_replaces_only_exact_sampled_cohort() -> None:
    from src.algorithms.shac.fresh_reference import refresh_reference_population

    population_size = 8
    carried = {
        "marker": -jnp.ones((population_size,), dtype=jnp.int32),
        "difficulty": -jnp.ones((population_size,), dtype=jnp.float32),
        "history": -jnp.ones((population_size, 2), dtype=jnp.int32),
    }
    difficulties = jnp.linspace(0.0, 1.0, population_size)
    refreshed, mask = jax.jit(
        lambda state: refresh_reference_population(
            _FakeReferenceEnv(),
            state,
            mask_key=jax.random.PRNGKey(11),
            reset_key=jax.random.PRNGKey(12),
            difficulties=difficulties,
            fraction=0.25,
        )
    )(carried)

    mask = np.asarray(mask)
    assert int(mask.sum()) == 2
    marker = np.asarray(refreshed["marker"])
    np.testing.assert_array_equal(marker[~mask], -np.ones(6, dtype=np.int32))
    assert np.all(marker[mask] >= 0)
    np.testing.assert_allclose(
        np.asarray(refreshed["difficulty"])[mask],
        np.asarray(difficulties)[mask],
    )
    np.testing.assert_array_equal(
        np.asarray(refreshed["history"])[~mask],
        -np.ones((6, 2), dtype=np.int32),
    )


def test_refresh_population_transports_adaptive_sampler_state() -> None:
    from src.algorithms.shac.fresh_reference import refresh_reference_population

    population_size = 4
    carried = {
        "marker": -jnp.ones((population_size,), dtype=jnp.int32),
        "difficulty": -jnp.ones((population_size,), dtype=jnp.float32),
        "history": -jnp.ones((population_size, 2), dtype=jnp.int32),
        "failed_count": -jnp.ones((population_size, 3), dtype=jnp.float32),
    }
    failed_count = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    refreshed, mask = refresh_reference_population(
        _FakeReferenceEnv(),
        carried,
        mask_key=jax.random.PRNGKey(21),
        reset_key=jax.random.PRNGKey(22),
        difficulties=jnp.arange(4, dtype=jnp.float32),
        fraction=0.5,
        phase_sampler_failed_count=failed_count,
    )

    mask = np.asarray(mask)
    np.testing.assert_array_equal(
        np.asarray(refreshed["failed_count"])[mask],
        np.asarray(failed_count)[mask],
    )
    np.testing.assert_array_equal(
        np.asarray(refreshed["failed_count"])[~mask],
        -np.ones((2, 3), dtype=np.float32),
    )


def test_fresh_reference_resume_change_fails_closed() -> None:
    from src.algorithms.shac.fresh_reference import (
        resolve_fresh_reference_resume_fraction,
    )

    assert (
        resolve_fresh_reference_resume_fraction(
            None,
            is_resume=False,
            requested=0.25,
            allow_change=False,
        )
        == 0.25
    )
    assert (
        resolve_fresh_reference_resume_fraction(
            {},
            is_resume=True,
            requested=0.0,
            allow_change=False,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="explicit authority"):
        resolve_fresh_reference_resume_fraction(
            {"actor_update_fresh_reference_fraction": 0.0},
            is_resume=True,
            requested=0.25,
            allow_change=False,
        )
    assert (
        resolve_fresh_reference_resume_fraction(
            {"actor_update_fresh_reference_fraction": 0.0},
            is_resume=True,
            requested=0.25,
            allow_change=True,
        )
        == 0.25
    )


def test_train_exposes_and_persists_fresh_reference_treatment() -> None:
    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters
    assert parameters["actor_update_fresh_reference_fraction"].default == 0.0
    assert (
        parameters["allow_resume_actor_update_fresh_reference_change"].default
        is False
    )
    source = inspect.getsource(train)
    assert "refresh_reference_population(" in source
    assert '"actor_update_fresh_reference_fraction"' in source
    assert '"actor_update_fresh_reference_count"' in source
    assert '"actor_update_fresh_reference_actual_fraction"' in source
    assert 'shard_reduction["fresh_reference"]' in source
    assert "actor_gradient_fresh_reference_distribution" in source
    assert 'build_checkpoint_gradient_group_telemetry(\n' in source


def test_fresh_reference_gradient_group_serialization() -> None:
    from src.algorithms.shac.algorithm import (
        build_checkpoint_gradient_group_telemetry,
    )

    prefix = "actor_grad_fresh_reference"
    metrics = {
        f"{prefix}_bin_counts": np.asarray([384, 128]),
        f"{prefix}_bin_mean_norms": np.asarray([0.1, 0.2]),
        f"{prefix}_bin_rms_norms": np.asarray([0.3, 0.4]),
        f"{prefix}_bin_variance_traces": np.asarray([0.5, 0.6]),
        f"{prefix}_bin_cancellation_ratios": np.asarray([0.7, 0.8]),
        f"{prefix}_bin_noise_scales": np.asarray([1.0, 2.0]),
        f"{prefix}_bin_esnr": np.asarray([3.0, 4.0]),
        f"{prefix}_bin_cosine_matrix": np.eye(2),
        f"{prefix}_within_variance_trace": 5.0,
        f"{prefix}_between_variance_trace": 1.0,
        f"{prefix}_total_variance_trace": 6.0,
        f"{prefix}_within_variance_fraction": 5.0 / 6.0,
        f"{prefix}_between_variance_fraction": 1.0 / 6.0,
    }
    row = build_checkpoint_gradient_group_telemetry(
        metrics, group="fresh_reference"
    )

    assert row[f"{prefix}_bin_counts"] == [384, 128]
    assert row[f"{prefix}_bin_cosine_matrix"] == np.eye(2).tolist()
    assert row[f"{prefix}_between_variance_fraction"] == 1.0 / 6.0
