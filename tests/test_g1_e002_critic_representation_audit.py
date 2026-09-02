from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
)
from src.core.networks import Actor, Critic


def _metrics(
    *,
    rank: float,
    nrmse: float,
    boundaries: tuple[str, ...] = ("aggregate", "h24"),
) -> dict[str, dict[str, float]]:
    row = {"rank_correlation": rank, "nrmse": nrmse}
    return {
        group: {boundary: dict(row) for boundary in boundaries}
        for group in ("combined", "carried", "repeated_current")
    }


def test_actor_history_latent_reconstructs_exact_base_actor_action() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        ActorHistoryLatent,
        reconstruct_base_actor_action,
    )

    observations = jax.random.normal(jax.random.PRNGKey(0), (7, 11))
    actor = Actor(
        action_dim=3,
        hidden=(5, 4, 2),
        squash=True,
        layer_norm=True,
        zero_output=False,
    )
    params = actor.init(jax.random.PRNGKey(1), observations)
    latent = ActorHistoryLatent(hidden=(5, 4, 2)).apply(params, observations)
    early_latent = ActorHistoryLatent(hidden=(5, 4, 2), output_index=0).apply(
        params,
        observations,
    )
    reconstructed = reconstruct_base_actor_action(params, latent, squash=True)

    np.testing.assert_allclose(
        reconstructed,
        actor.apply(params, observations),
        rtol=1e-6,
        atol=1e-6,
    )
    assert latent.shape == (7, 2)
    assert early_latent.shape == (7, 5)


def test_unwrap_base_actor_requires_preview_base_and_reports_depth() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        unwrap_base_actor,
    )

    base = FrozenPreviewResidualParams(
        parent={"params": {"Dense_0": {"kernel": jnp.zeros((2, 3))}}},
        adapter={"preview": jnp.zeros(1)},
    )
    nested = FrozenControllerResidualParams(
        parent=base,
        adapter={"controller": jnp.zeros(1)},
    )

    recovered, depth = unwrap_base_actor(nested)
    assert recovered is base
    assert depth == 1
    with pytest.raises(TypeError, match="preview-residual base"):
        unwrap_base_actor({"params": {}})


def test_e005_scan_carrier_uses_retained_rollout_shapes_without_noise() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        e005_scan_carrier,
    )

    source = {
        "carried_reward": np.zeros((5, 7), dtype=np.float64),
        "repeated_current_reward": np.ones((5, 7), dtype=np.float64),
    }

    carrier = e005_scan_carrier(
        source,
        horizon=5,
        population=7,
        action_dim=3,
    )

    assert carrier.shape == (5, 7, 3)
    assert carrier.dtype == jnp.float64
    np.testing.assert_array_equal(carrier, 0.0)
    with pytest.raises(ValueError, match="rollout shape"):
        e005_scan_carrier(
            {**source, "repeated_current_reward": np.ones((4, 7))},
            horizon=5,
            population=7,
            action_dim=3,
        )


def test_environment_group_splits_are_disjoint_complete_and_deterministic() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        environment_group_splits,
    )

    first = environment_group_splits(512, seed=20260902)
    second = environment_group_splits(512, seed=20260902)

    assert {name: len(rows) for name, rows in first.items()} == {
        "fit": 307,
        "validation": 102,
        "test": 103,
    }
    assert all(np.array_equal(first[name], second[name]) for name in first)
    joined = np.concatenate(tuple(first.values()))
    np.testing.assert_array_equal(np.sort(joined), np.arange(512))
    assert np.unique(joined).size == 512


def test_fit_mask_subsamples_time_without_splitting_environment_groups() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        fit_row_mask,
    )

    environment_id = np.asarray([0, 0, 0, 1, 1, 2, 2, 2])
    time_index = np.asarray([0, 1, 4, 0, 4, 0, 3, 4])
    alive = np.asarray([True, True, True, True, False, True, True, True])

    mask = fit_row_mask(
        environment_id,
        time_index,
        alive,
        fit_environment_ids=np.asarray([0, 2]),
        stride=4,
    )

    np.testing.assert_array_equal(
        mask, [True, False, True, False, False, True, False, True]
    )


def test_extra_feature_normalizer_uses_only_registered_fit_rows() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        normalize_extra_features,
    )

    values = np.asarray([[0.0, 2.0], [2.0, 4.0], [100.0, 200.0]])
    fit_mask = np.asarray([True, True, False])
    normalized, statistics = normalize_extra_features(values, fit_mask)

    np.testing.assert_allclose(statistics["mean"], [1.0, 3.0])
    np.testing.assert_allclose(statistics["std"], [1.0, 1.0])
    np.testing.assert_allclose(normalized[:2], [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(normalized[2], [99.0, 197.0])


def test_validation_candidate_key_requires_every_group_and_boundary() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        representation_adequate,
        validation_candidate_key,
    )

    passing = _metrics(rank=0.85, nrmse=0.20)
    failing = _metrics(rank=0.85, nrmse=0.20)
    failing["repeated_current"]["h24"]["nrmse"] = 0.26

    assert representation_adequate(passing)
    assert not representation_adequate(failing)
    assert validation_candidate_key(passing, step=40) > validation_candidate_key(
        failing, step=20
    )
    better = _metrics(rank=0.90, nrmse=0.15)
    assert validation_candidate_key(better, step=60) > validation_candidate_key(
        passing, step=40
    )
    assert validation_candidate_key(passing, step=20) > validation_candidate_key(
        passing, step=40
    )


def test_representation_summary_keeps_arm_and_h24_gates_separate() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        representation_adequate,
        summarize_representation_metrics,
    )

    targets = np.tile(np.asarray([1.0, 2.0, 3.0]), 4)
    predictions = targets.copy()
    arms = np.repeat(np.asarray([0, 0, 1, 1]), 3)
    times = np.repeat(np.asarray([0, 24, 0, 24]), 3)
    selected = np.ones(targets.shape, dtype=bool)

    summary = summarize_representation_metrics(
        predictions,
        targets,
        arms,
        times,
        selected,
    )
    assert representation_adequate(summary)
    assert summary["carried"]["h24"]["nrmse"] == 0.0
    assert summary["repeated_current"]["h24"]["rank_correlation"] == 1.0

    predictions[-3:] = predictions[-3:][::-1]
    changed = summarize_representation_metrics(
        predictions,
        targets,
        arms,
        times,
        selected,
    )
    assert changed["repeated_current"]["h24"]["rank_correlation"] == -1.0
    assert not representation_adequate(changed)


def test_representation_summary_can_add_h0_as_a_required_gate() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        representation_adequate,
        summarize_representation_metrics,
    )

    targets = np.tile(np.asarray([1.0, 2.0, 3.0]), 4)
    predictions = targets.copy()
    arms = np.repeat(np.asarray([0, 0, 1, 1]), 3)
    times = np.repeat(np.asarray([0, 24, 0, 24]), 3)
    boundary_indices = {"h0": 0, "h24": 24}

    summary = summarize_representation_metrics(
        predictions,
        targets,
        arms,
        times,
        np.ones(targets.shape, dtype=bool),
        boundary_indices=boundary_indices,
    )

    assert tuple(summary["combined"]) == ("aggregate", "h0", "h24")
    assert representation_adequate(
        summary,
        required_boundaries=("aggregate", "h0", "h24"),
    )
    summary["carried"]["h0"]["nrmse"] = 0.26
    assert not representation_adequate(
        summary,
        required_boundaries=("aggregate", "h0", "h24"),
    )


def test_representation_classification_prioritizes_latent_then_action() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        classify_representations,
    )

    passing = _metrics(rank=0.85, nrmse=0.20)
    failing = _metrics(rank=0.40, nrmse=0.80)
    improved = _metrics(rank=0.70, nrmse=0.40)

    assert (
        classify_representations(
            {
                "current_only": failing,
                "current_plus_action": passing,
                "current_plus_actor_latent": passing,
            }
        )
        == "actor-latent-representation-adequate"
    )
    assert (
        classify_representations(
            {
                "current_only": failing,
                "current_plus_action": passing,
                "current_plus_actor_latent": improved,
            }
        )
        == "policy-action-representation-adequate"
    )
    assert (
        classify_representations(
            {
                "current_only": passing,
                "current_plus_action": failing,
                "current_plus_actor_latent": failing,
            }
        )
        == "current-only-refit-adequate"
    )
    assert (
        classify_representations(
            {
                "current_only": passing,
                "current_plus_action": passing,
                "current_plus_actor_latent": passing,
            },
            current_only_first=True,
        )
        == "current-only-refit-adequate"
    )


def test_representation_classification_distinguishes_improvement_from_no_gain() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        classify_representations,
    )

    baseline = _metrics(rank=0.40, nrmse=0.80)
    improved = _metrics(rank=0.70, nrmse=0.40)
    regressed = _metrics(rank=0.30, nrmse=0.90)
    assert (
        classify_representations(
            {
                "current_only": baseline,
                "current_plus_action": regressed,
                "current_plus_actor_latent": improved,
            }
        )
        == "augmented-representation-improves-but-insufficient"
    )
    assert (
        classify_representations(
            {
                "current_only": baseline,
                "current_plus_action": regressed,
                "current_plus_actor_latent": regressed,
            }
        )
        == "tested-representations-insufficient"
    )


def test_early_latent_classification_prefers_simplest_then_richer_history() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        classify_early_latent_representations,
    )

    passing = _metrics(
        rank=0.85,
        nrmse=0.20,
        boundaries=("aggregate", "h0", "h24"),
    )
    failing = _metrics(
        rank=0.85,
        nrmse=0.30,
        boundaries=("aggregate", "h0", "h24"),
    )
    arms = {
        "current_only": failing,
        "current_plus_action": failing,
        "current_plus_actor_latent": failing,
        "current_plus_actor_early_latent": passing,
    }

    assert (
        classify_early_latent_representations(
            arms,
            required_boundaries=("aggregate", "h0", "h24"),
        )
        == "actor-early-latent-representation-adequate"
    )
    arms["current_only"] = passing
    assert (
        classify_early_latent_representations(
            arms,
            required_boundaries=("aggregate", "h0", "h24"),
        )
        == "current-only-refit-adequate"
    )


def test_raw_history_classification_requires_raw_arm_and_prefers_current() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        classify_raw_history_representations,
    )

    passing = _metrics(
        rank=0.85,
        nrmse=0.20,
        boundaries=("aggregate", "h0", "h24"),
    )
    failing = _metrics(
        rank=0.85,
        nrmse=0.30,
        boundaries=("aggregate", "h0", "h24"),
    )
    arms = {
        "current_only": failing,
        "current_plus_action": failing,
        "current_plus_actor_latent": failing,
        "current_plus_actor_history": passing,
    }

    assert (
        classify_raw_history_representations(
            arms,
            required_boundaries=("aggregate", "h0", "h24"),
        )
        == "actor-history-representation-adequate"
    )
    arms["current_only"] = passing
    assert (
        classify_raw_history_representations(
            arms,
            required_boundaries=("aggregate", "h0", "h24"),
        )
        == "current-only-refit-adequate"
    )
    del arms["current_plus_actor_history"]
    with pytest.raises(ValueError, match="raw-history representation test arms"):
        classify_raw_history_representations(
            arms,
            required_boundaries=("aggregate", "h0", "h24"),
        )


def test_representation_metrics_reject_missing_or_nonfinite_gates() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        representation_adequate,
    )

    missing = _metrics(rank=0.85, nrmse=0.20)
    del missing["carried"]["h24"]
    with pytest.raises(ValueError, match="representation metrics"):
        representation_adequate(missing)
    nonfinite = _metrics(rank=0.85, nrmse=0.20)
    nonfinite["combined"]["aggregate"]["nrmse"] = np.nan
    with pytest.raises(ValueError, match="representation metrics"):
        representation_adequate(nonfinite)


def test_critic_fit_selects_only_on_validation_and_runs_exact_budget() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        fit_critic_with_validation,
    )

    block_returns = np.asarray([1.0, 2.0, 3.0])
    returns = np.tile(block_returns, 8).astype(np.float32)
    arms = np.repeat(np.asarray([0, 0, 1, 1, 0, 0, 1, 1]), 3)
    times = np.repeat(np.asarray([0, 24, 0, 24, 0, 24, 0, 24]), 3)
    observations = np.stack(
        (returns, np.linspace(-1.0, 1.0, returns.size)), axis=-1
    ).astype(np.float32)
    fit_mask = np.zeros(returns.shape, dtype=bool)
    fit_mask[:12] = True
    validation_mask = ~fit_mask
    critic = Critic(hidden=(4, 3))
    params = critic.init(jax.random.PRNGKey(5), jnp.asarray(observations))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    opt_state = optimizer.init(params)

    selected_params, report = fit_critic_with_validation(
        critic,
        params,
        opt_state,
        observations,
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


def test_parser_makes_h0_an_explicit_opt_in() -> None:
    from experiments.g1_e002_critic_representation_audit.run import (
        build_parser,
        validate_representation_options,
    )

    base = [
        "--checkpoint",
        "checkpoint.pkl",
        "--reference-path",
        "reference.npz",
        "--alias-trace",
        "trace.npz",
        "--output-root",
        "output",
        "--code-commit",
        "0" * 40,
    ]

    defaults = build_parser().parse_args(base)
    assert defaults.include_h0 is False
    assert defaults.include_early_latent is False
    assert defaults.include_raw_history is False
    enabled = build_parser().parse_args(
        [*base, "--include-h0", "--include-early-latent"]
    )
    assert enabled.include_h0 is True
    assert enabled.include_early_latent is True
    assert enabled.include_raw_history is False
    raw = build_parser().parse_args([*base, "--include-h0", "--include-raw-history"])
    assert raw.include_h0 is True
    assert raw.include_early_latent is False
    assert raw.include_raw_history is True
    validate_representation_options(
        include_h0=True,
        include_early_latent=True,
        include_raw_history=False,
    )
    validate_representation_options(
        include_h0=True,
        include_early_latent=False,
        include_raw_history=True,
    )
    with pytest.raises(ValueError, match="requires H0"):
        validate_representation_options(
            include_h0=False,
            include_early_latent=True,
            include_raw_history=False,
        )
    with pytest.raises(ValueError, match="requires H0"):
        validate_representation_options(
            include_h0=False,
            include_early_latent=False,
            include_raw_history=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_representation_options(
            include_h0=True,
            include_early_latent=True,
            include_raw_history=True,
        )
