from __future__ import annotations

import numpy as np
import pytest


def _single_critic(value: float) -> dict[str, object]:
    return {
        "params": {
            "Dense_0": {
                "kernel": np.asarray([value], dtype=np.float32),
                "bias": np.asarray([value + 1.0], dtype=np.float32),
            }
        }
    }


def test_combine_single_critics_preserves_order_and_values() -> None:
    from experiments.g1_ahac_critic_migration_audit.run import (
        combine_single_critic_params,
    )

    online = _single_critic(1.0)
    delayed = _single_critic(2.0)
    combined = combine_single_critic_params(online, delayed)

    assert set(combined["params"]) == {"critic_0", "critic_1"}
    np.testing.assert_array_equal(
        combined["params"]["critic_0"]["Dense_0"]["kernel"],
        online["params"]["Dense_0"]["kernel"],
    )
    np.testing.assert_array_equal(
        combined["params"]["critic_1"]["Dense_0"]["kernel"],
        delayed["params"]["Dense_0"]["kernel"],
    )


def test_combine_single_critics_rejects_malformed_or_double_inputs() -> None:
    from experiments.g1_ahac_critic_migration_audit.run import (
        combine_single_critic_params,
    )

    with pytest.raises(ValueError, match="single critic"):
        combine_single_critic_params(
            {"params": {"critic_0": {}, "critic_1": {}}},
            _single_critic(2.0),
        )
    with pytest.raises(ValueError, match="single critic"):
        combine_single_critic_params(_single_critic(1.0), {"wrong": {}})


def test_migration_summary_selects_aligned_distinct_trained_heads() -> None:
    from experiments.g1_ahac_critic_migration_audit.run import (
        summarize_migration_candidates,
    )

    online = np.linspace(-4.0, 4.0, 512)
    delayed = online * 0.99 + 0.01 * np.sin(np.arange(512))
    duplicate = np.column_stack((online, online))
    diverse = np.column_stack((online, delayed))

    summary = summarize_migration_candidates(
        online,
        delayed,
        duplicate,
        diverse,
    )

    assert summary["classification"] == (
        "reuse-legacy-target-as-second-online-head"
    )
    assert summary["duplicate_online"]["head_disagreement_max"] == 0.0
    assert summary["online_plus_legacy_target"]["head_disagreement_max"] > 0.0
    assert summary["online_plus_legacy_target"]["pearson_correlation"] > 0.99


def test_migration_summary_rejects_large_or_unaligned_target_lag() -> None:
    from experiments.g1_ahac_critic_migration_audit.run import (
        summarize_migration_candidates,
    )

    online = np.linspace(-1.0, 1.0, 64)
    delayed = -10.0 * online
    duplicate = np.column_stack((online, online))
    diverse = np.column_stack((online, delayed))

    summary = summarize_migration_candidates(
        online,
        delayed,
        duplicate,
        diverse,
    )

    assert summary["classification"] == "legacy-target-lag-too-large"
    assert summary["safe_to_reuse_legacy_target"] is False


def test_upstream_double_critic_audit_detects_shared_module_objects() -> None:
    from experiments.g1_ahac_critic_migration_audit.run import (
        inspect_upstream_double_critic,
    )

    source = """
class DoubleCriticMLP(nn.Module):
    def __init__(self):
        modules = [nn.Linear(3, 1)]
        self.critic_1 = nn.Sequential(*modules)
        self.critic_2 = nn.Sequential(*modules)
"""
    audit = inspect_upstream_double_critic(source)

    assert audit["declares_two_critic_branches"] is True
    assert audit["branches_reuse_same_module_container"] is True
    assert audit["effective_parameter_independence"] is False
    assert audit["paper_public_code_double_critic_divergence"] is True
