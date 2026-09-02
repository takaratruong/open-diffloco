from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest


@dataclass(frozen=True)
class _FakeState:
    data: object
    obs: np.ndarray
    reward: object
    done: object
    info: dict[str, object]
    metrics: object

    def replace(self, **updates):
        return replace(self, **updates)


def test_history_only_counterfactual_preserves_every_other_state_field() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        replace_with_repeated_current_history,
    )

    history = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    original = _FakeState(
        data=object(),
        obs=history.reshape(2, -1),
        reward=object(),
        done=object(),
        info={
            "actor_obs_history": history,
            "last_act": np.arange(6, dtype=np.float64).reshape(2, 3),
            "phase": np.asarray([4, 9]),
            "rng": np.arange(4, dtype=np.uint32).reshape(2, 2),
        },
        metrics=object(),
    )

    counterfactual = replace_with_repeated_current_history(original)
    expected = np.repeat(history[:, -1:, :], history.shape[1], axis=1)

    np.testing.assert_array_equal(counterfactual.info["actor_obs_history"], expected)
    np.testing.assert_array_equal(counterfactual.obs, expected.reshape(2, -1))
    assert counterfactual.data is original.data
    assert counterfactual.reward is original.reward
    assert counterfactual.done is original.done
    assert counterfactual.metrics is original.metrics
    assert counterfactual.info["last_act"] is original.info["last_act"]
    assert counterfactual.info["phase"] is original.info["phase"]
    assert counterfactual.info["rng"] is original.info["rng"]
    assert set(counterfactual.info) == set(original.info)


def test_history_only_counterfactual_rejects_incoherent_flat_observation() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        replace_with_repeated_current_history,
    )

    history = np.zeros((2, 3, 4), dtype=np.float64)
    state = _FakeState(
        data=None,
        obs=np.zeros((2, 11), dtype=np.float64),
        reward=None,
        done=None,
        info={"actor_obs_history": history},
        metrics=None,
    )

    with pytest.raises(ValueError, match="flattened actor observation"):
        replace_with_repeated_current_history(state)


def test_nonhistory_tree_delta_requires_exact_structure_and_dtype() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        _tree_max_abs_delta,
    )

    original = {
        "phase": np.asarray([1, 2], dtype=np.int32),
        "nested": {"value": np.asarray([0.25], dtype=np.float64)},
    }
    identical = {
        "phase": np.asarray([1, 2], dtype=np.int32),
        "nested": {"value": np.asarray([0.25], dtype=np.float64)},
    }
    changed = {
        "phase": np.asarray([1, 3], dtype=np.int32),
        "nested": {"value": np.asarray([0.25], dtype=np.float64)},
    }
    dtype_changed = {
        "phase": np.asarray([1, 2], dtype=np.int64),
        "nested": {"value": np.asarray([0.25], dtype=np.float64)},
    }

    assert _tree_max_abs_delta(original, identical) == 0.0
    assert _tree_max_abs_delta(original, changed) == 1.0
    assert np.isinf(_tree_max_abs_delta(original, dtype_changed))


def test_paired_divergence_uses_carried_scale_and_signed_treatment_delta() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        paired_divergence_metrics,
    )

    metrics = paired_divergence_metrics(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 0.0, 3.0]),
    )

    assert metrics["mean_carried"] == pytest.approx(2.0)
    assert metrics["mean_repeated"] == pytest.approx(5.0 / 3.0)
    assert metrics["mean_repeated_minus_carried"] == pytest.approx(-1.0 / 3.0)
    assert metrics["mean_absolute_difference"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics["carried_mean_absolute_normalized_rmse"] == pytest.approx(
        np.sqrt(5.0 / 3.0) / 2.0
    )


def test_policy_action_metrics_measure_executed_action_relative_rms() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        policy_action_metrics,
    )

    carried = np.asarray([[0.2, -0.2], [0.4, -0.4]])
    repeated = np.asarray([[0.1, -0.1], [0.4, -0.4]])
    metrics = policy_action_metrics(carried, repeated)

    assert metrics["global_difference_rms"] == pytest.approx(np.sqrt(0.02 / 4.0))
    assert metrics["carried_action_rms"] == pytest.approx(np.sqrt(0.4 / 4.0))
    assert metrics["relative_rms"] == pytest.approx(np.sqrt(0.02 / 0.4))
    assert metrics["fraction_env_rms_at_least_0p01"] == pytest.approx(0.5)


def test_alias_classification_requires_material_action_and_outcome_effects() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        classify_policy_memory_alias,
    )

    assert (
        classify_policy_memory_alias(
            action_relative_rms=0.05,
            action_env_fraction=0.2,
            return_nrmse=0.2,
            survival_mae=0.0,
        )
        == "policy-memory-alias-material"
    )
    assert (
        classify_policy_memory_alias(
            action_relative_rms=0.05,
            action_env_fraction=0.2,
            return_nrmse=0.01,
            survival_mae=0.5,
        )
        == "policy-memory-action-sensitive-return-immaterial"
    )
    assert (
        classify_policy_memory_alias(
            action_relative_rms=0.001,
            action_env_fraction=0.0,
            return_nrmse=2.0,
            survival_mae=100.0,
        )
        == "policy-memory-immaterial"
    )


def test_alias_action_gate_accepts_a_localized_affected_population() -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        classify_policy_memory_alias,
    )

    assert (
        classify_policy_memory_alias(
            action_relative_rms=0.001,
            action_env_fraction=0.1,
            return_nrmse=0.1,
            survival_mae=0.0,
        )
        == "policy-memory-alias-material"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_relative_rms", np.nan),
        ("action_env_fraction", -0.1),
        ("action_env_fraction", 1.1),
        ("return_nrmse", np.inf),
        ("survival_mae", -1.0),
    ],
)
def test_alias_classification_rejects_invalid_metrics(field: str, value: float) -> None:
    from experiments.g1_e002_policy_memory_alias_audit.run import (
        classify_policy_memory_alias,
    )

    values = {
        "action_relative_rms": 0.0,
        "action_env_fraction": 0.0,
        "return_nrmse": 0.0,
        "survival_mae": 0.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match="alias metrics"):
        classify_policy_memory_alias(**values)
