import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.cagrad import accumulate_phase_gradients


def test_cagrad_settings_are_default_off_with_fixed_treatment_values():
    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters

    assert parameters["actor_cagrad"].default is False
    assert parameters["actor_cagrad_alpha"].default == 0.5
    assert parameters["actor_cagrad_iterations"].default == 32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"actor_cagrad": "yes"}, "actor_cagrad must be boolean"),
        (
            {"actor_cagrad_alpha": float("nan")},
            "actor_cagrad_alpha must be non-negative and finite",
        ),
        (
            {"actor_cagrad_iterations": True},
            "actor_cagrad_iterations must be a positive integer",
        ),
        (
            {
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
                "env_variant": "g1_tracking_rmr",
                "adaptive_phase_sampling": True,
            },
            "cannot combine with adaptive phase sampling",
        ),
        (
            {
                "actor_cagrad": True,
                "gradient_accumulation_steps": 1,
                "env_variant": "g1_tracking_rmr",
                "actor_phase_robust_weighting": True,
            },
            "cannot combine with phase-robust weighting",
        ),
        (
            {
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
                "env_variant": "blind_nolinvel_nokinref",
            },
            "requires G1 reference phases",
        ),
        (
            {
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
                "env_variant": "g1_tracking_rmr",
                "actor_per_env_grad_clip": 1.0,
            },
            "cannot combine with per-env clipping",
        ),
        (
            {
                "actor_cagrad": True,
                "gradient_accumulation_steps": 1,
                "env_variant": "g1_tracking_rmr",
            },
            "requires exactly two population shards",
        ),
        (
            {
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
                "env_variant": "g1_tracking_rmr",
                "actor_phase_bin_count": 4,
            },
            "requires exactly five phase bins",
        ),
    ],
)
def test_train_rejects_invalid_cagrad_contracts(kwargs, message):
    from src.algorithms.shac.algorithm import train

    with pytest.raises(ValueError, match=message):
        train(**kwargs)


def test_resume_restores_cagrad_but_legacy_hparams_allow_treatment_start():
    from src.algorithms.shac.algorithm import resolve_cagrad_resume_settings

    restored = resolve_cagrad_resume_settings(
        {
            "actor_cagrad": True,
            "actor_cagrad_alpha": 0.5,
            "actor_cagrad_iterations": 32,
        },
        requested_actor_cagrad=False,
        requested_alpha=0.1,
        requested_iterations=4,
    )
    assert restored == (True, 0.5, 32)

    treatment_start = resolve_cagrad_resume_settings(
        {"algorithm": "shac"},
        requested_actor_cagrad=True,
        requested_alpha=0.5,
        requested_iterations=32,
    )
    assert treatment_start == (True, 0.5, 32)


def test_cagrad_resume_metadata_must_be_complete():
    from src.algorithms.shac.algorithm import resolve_cagrad_resume_settings

    with pytest.raises(ValueError, match="CAGrad checkpoint.*metadata"):
        resolve_cagrad_resume_settings(
            {"actor_cagrad": True},
            requested_actor_cagrad=False,
            requested_alpha=0.5,
            requested_iterations=32,
        )


def test_two_shard_reducer_matches_concatenated_population():
    from src.algorithms.shac.algorithm import reduce_cagrad_shard_accumulators

    first_gradients = {
        "dense": {
            "kernel": jnp.arange(24, dtype=jnp.float32).reshape(6, 2, 2) / 10.0,
            "bias": jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 7.0,
        }
    }
    second_gradients = {
        "dense": {
            "kernel": -jnp.arange(24, 48, dtype=jnp.float32).reshape(6, 2, 2) / 13.0,
            "bias": jnp.arange(12, 24, dtype=jnp.float32).reshape(6, 2) / 11.0,
        }
    }
    first_phases = jnp.array([0, 40, 100, 200, 300, 400], dtype=jnp.int32)
    second_phases = jnp.array([75, 175, 275, 375, 425, 498], dtype=jnp.int32)

    shard_accumulators = [
        accumulate_phase_gradients(
            gradients,
            phases,
            phase_count=499,
            bin_count=5,
        )
        for gradients, phases in (
            (first_gradients, first_phases),
            (second_gradients, second_phases),
        )
    ]
    stacked_accumulators = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *shard_accumulators
    )
    sharded = reduce_cagrad_shard_accumulators(
        stacked_accumulators,
        alpha=0.5,
        iterations=32,
    )

    concatenated_gradients = jax.tree_util.tree_map(
        lambda first, second: jnp.concatenate((first, second)),
        first_gradients,
        second_gradients,
    )
    concatenated = accumulate_phase_gradients(
        concatenated_gradients,
        jnp.concatenate((first_phases, second_phases)),
        phase_count=499,
        bin_count=5,
    )
    expected = reduce_cagrad_shard_accumulators(
        jax.tree_util.tree_map(lambda leaf: leaf[None], concatenated),
        alpha=0.5,
        iterations=32,
    )

    np.testing.assert_array_equal(
        sharded["accumulator"].env_counts,
        np.array([3, 2, 2, 2, 3]),
    )
    np.testing.assert_array_equal(
        sharded["accumulator"].env_counts,
        expected["accumulator"].env_counts,
    )
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(sharded["accumulator"].sums),
        jax.tree_util.tree_leaves(expected["accumulator"].sums),
        strict=True,
    ):
        np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-6)
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(sharded["accumulator"].finite_counts),
        jax.tree_util.tree_leaves(expected["accumulator"].finite_counts),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, wanted)
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(sharded["task_gradients"]),
        jax.tree_util.tree_leaves(expected["task_gradients"]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        sharded["result"].weights,
        expected["result"].weights,
        rtol=1e-6,
        atol=1e-6,
    )
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(sharded["result"].combined_gradient),
        jax.tree_util.tree_leaves(expected["result"].combined_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-6)
    assert bool(sharded["valid"])
