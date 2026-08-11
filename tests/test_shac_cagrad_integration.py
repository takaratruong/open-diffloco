import inspect
import json

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


def test_preview_adapter_configuration_requires_future_reference_cagrad():
    from src.algorithms.shac.algorithm import (
        validate_preview_adapter_configuration,
    )

    with pytest.raises(ValueError, match="requires future-reference CAGrad"):
        validate_preview_adapter_configuration(
            enabled=True,
            actor_reference_lookahead_steps=(),
            actor_cagrad=True,
            history_len=10,
            source_actor_policy=None,
            initial_full_actor_policy=None,
            env_variant="g1_tracking_rmr",
        )


def test_preview_adapter_configuration_accepts_one_frame_full_rmr_parent():
    from src.algorithms.shac.algorithm import (
        validate_preview_adapter_configuration,
    )

    validate_preview_adapter_configuration(
        enabled=True,
        actor_reference_lookahead_steps=(4, 8, 12),
        actor_cagrad=True,
        history_len=1,
        source_actor_policy=None,
        initial_full_actor_policy=object(),
        env_variant="g1_tracking_rmr_50hz_source_step",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"enabled": "yes"}, "must be boolean"),
        ({"actor_cagrad": False}, "requires future-reference CAGrad"),
        ({"history_len": 9}, "ten-frame history"),
        ({"source_actor_policy": object()}, "plain Flax actor"),
        ({"initial_full_actor_policy": object()}, "one-frame history"),
        ({"env_variant": "go2"}, "G1 tracking"),
    ],
)
def test_preview_adapter_configuration_rejects_changed_contract(
    changes, message
):
    from src.algorithms.shac.algorithm import (
        validate_preview_adapter_configuration,
    )

    kwargs = {
        "enabled": True,
        "actor_reference_lookahead_steps": (4, 8, 12),
        "actor_cagrad": True,
        "history_len": 10,
        "source_actor_policy": None,
        "initial_full_actor_policy": None,
        "env_variant": "g1_tracking_rmr",
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        validate_preview_adapter_configuration(**kwargs)


def test_checkpoint_phase_metrics_are_atomic_and_step_addressed(tmp_path):
    from src.algorithms.shac.algorithm import (
        persist_checkpoint_phase_metric,
    )

    persist_checkpoint_phase_metric(
        tmp_path,
        {"step": 1_376_256, "actor_cagrad_bin_losses": [1, 2, 3, 4, 5]},
    )
    persist_checkpoint_phase_metric(
        tmp_path,
        {"step": 1_572_864, "actor_cagrad_bin_losses": [5, 4, 3, 2, 1]},
    )
    persist_checkpoint_phase_metric(
        tmp_path,
        {"step": 1_376_256, "actor_cagrad_bin_losses": [0, 0, 0, 0, 0]},
    )

    rows = json.loads(
        (tmp_path / "checkpoint_phase_metrics.json").read_text()
    )
    assert [row["step"] for row in rows] == [1_376_256, 1_572_864]
    assert rows[0]["actor_cagrad_bin_losses"] == [0, 0, 0, 0, 0]


def test_train_wires_preview_adapter_without_changing_disabled_path():
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)

    assert "apply_preview_adapter_update(" in source
    assert "zero_current_preview(" in source
    assert "phase_binned_action_deviation(" in source
    assert "new_actor_norm = state.normalizer" in source
    assert '"actor_preview_adapter": actor_preview_adapter' in source
    assert "if checkpoint_path is not None" in source
    assert "persist_checkpoint_phase_metric(" in source


def test_residual_preview_configuration_requires_isolated_delta_cagrad():
    from src.algorithms.shac.algorithm import (
        validate_residual_preview_adapter_configuration,
    )

    valid = {
        "enabled": True,
        "hidden_dim": 256,
        "linear_preview_enabled": False,
        "actor_reference_lookahead_steps": (4, 8, 12),
        "actor_reference_preview_mode": "delta",
        "actor_cagrad": True,
        "history_len": 10,
        "source_actor_policy": None,
        "initial_full_actor_policy": None,
        "env_variant": "g1_tracking_rmr",
    }
    validate_residual_preview_adapter_configuration(**valid)
    invalid_cases = (
        ({"enabled": "yes"}, "must be boolean"),
        ({"hidden_dim": 0}, "positive integer"),
        ({"linear_preview_enabled": True}, "mutually exclusive"),
        ({"actor_reference_lookahead_steps": ()}, "future-reference CAGrad"),
        ({"actor_reference_preview_mode": "absolute"}, "delta preview"),
        ({"actor_cagrad": False}, "future-reference CAGrad"),
        ({"history_len": 1}, "ten-frame history"),
        ({"source_actor_policy": object()}, "plain Flax actor"),
        ({"initial_full_actor_policy": object()}, "plain Flax actor"),
        ({"env_variant": "go2"}, "G1 tracking"),
    )
    for changes, message in invalid_cases:
        candidate = {**valid, **changes}
        with pytest.raises(ValueError, match=message):
            validate_residual_preview_adapter_configuration(**candidate)


def test_train_wires_residual_preview_through_existing_frozen_boundary():
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)

    assert "PreviewResidualAdapter(" in source
    assert "apply_frozen_preview_residual(" in source
    assert "initialize_residual_adapter_optimizer(" in source
    assert "build_residual_adapter_mask(" in source
    assert "residual_adapter_migration_report(" in source
    assert '"actor_residual_preview_adapter": (' in source
    assert '"actor_residual_preview_hidden": (' in source
    assert '"residual_adapter_migration.json"' in source
    assert '"flax_residual_preview"' in source


def test_train_wires_native_rmr_preview_migration_and_parent_action():
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)

    assert "migrate_rmr_preview_policy(" in source
    assert "build_rmr_preview_mask(" in source
    assert "rmr_preview_migration_report(" in source
    assert "apply_trainable_rmr_policy(\n                        initial_full_actor_policy" in source


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
            "actor_phase_bin_count": 5,
        },
        requested_actor_cagrad=False,
        requested_alpha=0.1,
        requested_iterations=4,
        requested_bin_count=3,
    )
    assert restored == (True, 0.5, 32, 5)

    treatment_start = resolve_cagrad_resume_settings(
        {"algorithm": "shac"},
        requested_actor_cagrad=True,
        requested_alpha=0.5,
        requested_iterations=32,
        requested_bin_count=5,
    )
    assert treatment_start == (True, 0.5, 32, 5)


def test_cagrad_resume_metadata_must_be_complete():
    from src.algorithms.shac.algorithm import resolve_cagrad_resume_settings

    with pytest.raises(ValueError, match="CAGrad checkpoint.*metadata"):
        resolve_cagrad_resume_settings(
            {
                "actor_cagrad": True,
                "actor_cagrad_alpha": 0.5,
                "actor_cagrad_iterations": 32,
            },
            requested_actor_cagrad=False,
            requested_alpha=0.5,
            requested_iterations=32,
            requested_bin_count=5,
        )


def test_cagrad_resume_rejects_non_five_bin_metadata():
    from src.algorithms.shac.algorithm import resolve_cagrad_resume_settings

    with pytest.raises(ValueError, match="CAGrad checkpoint.*phase bins"):
        resolve_cagrad_resume_settings(
            {
                "actor_cagrad": True,
                "actor_cagrad_alpha": 0.5,
                "actor_cagrad_iterations": 32,
                "actor_phase_bin_count": 4,
            },
            requested_actor_cagrad=False,
            requested_alpha=0.5,
            requested_iterations=32,
            requested_bin_count=5,
        )


def test_cagrad_phase_loss_diagnostics_use_full_population_bins():
    from src.algorithms.shac.algorithm import cagrad_phase_loss_diagnostics

    diagnostics = cagrad_phase_loss_diagnostics(
        losses=jnp.array([1.0, 3.0, 2.0, 4.0, 8.0, 10.0]),
        phases=jnp.array([0, 40, 100, 200, 300, 498]),
        phase_count=499,
        bin_count=5,
    )

    np.testing.assert_array_equal(
        diagnostics["bin_counts"], np.array([2, 1, 1, 1, 1])
    )
    np.testing.assert_allclose(
        diagnostics["bin_losses"], np.array([2.0, 2.0, 4.0, 8.0, 10.0])
    )
    assert bool(diagnostics["valid"])

    invalid = cagrad_phase_loss_diagnostics(
        losses=jnp.array([1.0, jnp.nan, 2.0, 4.0, 8.0, 10.0]),
        phases=jnp.array([0, 40, 100, 200, 300, 498]),
        phase_count=499,
        bin_count=5,
    )
    assert not bool(invalid["valid"])


def test_logging_cadence_starts_at_first_iteration_of_each_invocation():
    from src.algorithms.shac.algorithm import should_log_training_iteration

    assert should_log_training_iteration(0, start_iteration=0)
    assert should_log_training_iteration(10, start_iteration=0)
    assert not should_log_training_iteration(1, start_iteration=0)

    assert should_log_training_iteration(128, start_iteration=128)
    assert should_log_training_iteration(138, start_iteration=128)
    assert not should_log_training_iteration(130, start_iteration=128)


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
