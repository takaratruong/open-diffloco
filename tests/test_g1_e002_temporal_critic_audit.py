from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.networks import Critic


def _metrics(*, rank: float, nrmse: float):
    row = {"rank_correlation": rank, "nrmse": nrmse}
    return {
        group: {boundary: dict(row) for boundary in ("aggregate", "h0", "h24")}
        for group in ("combined", "carried", "repeated_current")
    }


def _tree_allclose(left, right) -> bool:
    return jax.tree.structure(left) == jax.tree.structure(right) and all(
        np.allclose(np.asarray(lhs), np.asarray(rhs), rtol=0.0, atol=0.0)
        for lhs, rhs in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def test_temporal_encoder_is_order_sensitive_and_residual_starts_exact_zero() -> None:
    from experiments.g1_e002_temporal_critic_audit.run import (
        TemporalHistoryEncoder,
        TemporalValueResidual,
        parameter_count,
    )

    histories = jnp.asarray(
        [
            [[0.0, 0.5], [1.0, -0.5], [2.0, 0.25]],
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    encoder = TemporalHistoryEncoder(
        frame_dim=2,
        history_len=3,
        hidden_dim=4,
    )
    encoder_params = encoder.init(jax.random.PRNGKey(1), histories)
    forward = encoder.apply(encoder_params, histories)
    reverse = encoder.apply(encoder_params, histories[:, ::-1])

    assert forward.shape == (2, 4)
    assert np.isfinite(forward).all()
    assert not np.allclose(forward, reverse)

    residual = TemporalValueResidual(
        frame_dim=2,
        history_len=3,
        hidden_dim=4,
    )
    residual_params = residual.init(jax.random.PRNGKey(2), histories)
    np.testing.assert_array_equal(residual.apply(residual_params, histories), 0.0)
    assert parameter_count(residual_params) == 93


def test_temporal_dataset_preserves_oldest_to_newest_flattening() -> None:
    from experiments.g1_e002_temporal_critic_audit.run import (
        validate_temporal_dataset,
    )

    rows = 12
    flat = np.arange(rows * 6, dtype=np.float32).reshape(rows, 6)
    arrays = {
        "normalized_current_critic_obs": np.ones((rows, 4), dtype=np.float32),
        "normalized_actor_history": flat,
        "realized_return": np.linspace(0.0, 1.0, rows),
        "arm": np.tile(np.asarray([0, 1], dtype=np.int8), rows // 2),
        "environment_id": np.repeat(np.arange(6, dtype=np.int32), 2),
        "time_index": np.tile(np.asarray([0, 24], dtype=np.int32), 6),
        "split_code": np.repeat(np.asarray([0, 0, 1, 1, 2, 2]), 2),
        "fit_row_selected": np.asarray(
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        ),
    }

    dataset = validate_temporal_dataset(
        arrays,
        critic_dim=4,
        history_len=3,
        frame_dim=2,
    )

    assert dataset["history"].shape == (rows, 3, 2)
    np.testing.assert_array_equal(dataset["history"].reshape(rows, 6), flat)
    assert not np.any(dataset["fit_mask"] & dataset["validation_mask"])
    assert not np.any(dataset["validation_mask"] & dataset["test_mask"])


def test_combined_optimizer_preserves_critic_adam_and_zeros_temporal_moments() -> None:
    from experiments.g1_e002_temporal_critic_audit.run import (
        TemporalValueResidual,
        graft_temporal_optimizer,
        optimizer_count,
    )

    critic = Critic(hidden=(4, 3))
    critic_params = critic.init(jax.random.PRNGKey(3), jnp.ones((5, 2)))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    critic_opt = optimizer.init(critic_params)
    for _ in range(3):
        updates, critic_opt = optimizer.update(
            jax.tree.map(jnp.ones_like, critic_params),
            critic_opt,
        )
        critic_params = optax.apply_updates(critic_params, updates)

    histories = jnp.ones((5, 2, 2), dtype=jnp.float32)
    temporal = TemporalValueResidual(frame_dim=2, history_len=2, hidden_dim=3)
    temporal_params = temporal.init(jax.random.PRNGKey(4), histories)
    combined_params = {"critic": critic_params, "temporal": temporal_params}
    combined_opt = graft_temporal_optimizer(
        critic_opt,
        optimizer.init(combined_params),
    )

    old_adam = critic_opt[1][0]
    combined_adam = combined_opt[1][0]
    assert optimizer_count(combined_opt) == 3
    assert _tree_allclose(combined_adam.mu["critic"], old_adam.mu)
    assert _tree_allclose(combined_adam.nu["critic"], old_adam.nu)
    assert all(
        np.count_nonzero(np.asarray(value)) == 0
        for value in jax.tree.leaves(combined_adam.mu["temporal"])
    )
    assert all(
        np.count_nonzero(np.asarray(value)) == 0
        for value in jax.tree.leaves(combined_adam.nu["temporal"])
    )


def test_temporal_fit_selects_validation_only_and_runs_exact_budget() -> None:
    from experiments.g1_e002_temporal_critic_audit.run import (
        TemporalValueResidual,
        fit_temporal_critic_with_validation,
        graft_temporal_optimizer,
    )

    block_returns = np.asarray([1.0, 2.0, 3.0])
    returns = np.tile(block_returns, 8).astype(np.float32)
    arms = np.repeat(np.asarray([0, 0, 1, 1, 0, 0, 1, 1]), 3)
    times = np.repeat(np.asarray([0, 24, 0, 24, 0, 24, 0, 24]), 3)
    current = np.stack((returns, np.linspace(-1.0, 1.0, returns.size)), axis=-1).astype(
        np.float32
    )
    history = np.stack((current, current[:, ::-1]), axis=1)
    fit_mask = np.zeros(returns.shape, dtype=bool)
    fit_mask[:12] = True
    validation_mask = ~fit_mask
    critic = Critic(hidden=(4, 3))
    temporal = TemporalValueResidual(frame_dim=2, history_len=2, hidden_dim=3)
    critic_params = critic.init(jax.random.PRNGKey(5), jnp.asarray(current))
    temporal_params = temporal.init(jax.random.PRNGKey(6), jnp.asarray(history))
    params = {"critic": critic_params, "temporal": temporal_params}
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    opt_state = graft_temporal_optimizer(
        optimizer.init(critic_params),
        optimizer.init(params),
    )

    selected_params, report = fit_temporal_critic_with_validation(
        critic,
        temporal,
        params,
        opt_state,
        current,
        history,
        returns,
        arms,
        times,
        fit_mask=fit_mask,
        validation_mask=validation_mask,
        steps=4,
        evaluation_interval=2,
        optimizer=optimizer,
        boundary_indices={"h0": 0, "h24": 24},
    )

    assert selected_params is not None
    assert [row["step"] for row in report["candidates"]] == [0, 2, 4]
    assert report["executed_steps"] == 4
    assert report["final_optimizer_count"] == report["initial_optimizer_count"] + 4
    assert report["selected"]["step"] in {0, 2, 4}
    assert tuple(report["selected"]["metrics"]["combined"]) == (
        "aggregate",
        "h0",
        "h24",
    )


def test_temporal_classification_requires_absolute_pass_or_strict_gain() -> None:
    from experiments.g1_e002_temporal_critic_audit.run import (
        classify_temporal_representation,
    )

    control = _metrics(rank=0.90, nrmse=0.27)
    assert (
        classify_temporal_representation(
            _metrics(rank=0.91, nrmse=0.24),
            control,
        )
        == "temporal-representation-adequate"
    )
    assert (
        classify_temporal_representation(
            _metrics(rank=0.91, nrmse=0.26),
            control,
        )
        == "temporal-improves-but-insufficient"
    )
    assert (
        classify_temporal_representation(
            _metrics(rank=0.89, nrmse=0.28),
            control,
        )
        == "temporal-no-gain"
    )
