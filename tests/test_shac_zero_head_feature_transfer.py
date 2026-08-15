from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _adapter_params(*, input_dim: int = 5, hidden_dim: int = 4, action_dim: int = 2):
    from src.algorithms.shac.residual_preview_adapter import PreviewResidualAdapter

    actor = PreviewResidualAdapter(action_dim=action_dim, hidden_dim=hidden_dim)
    params = actor.init(
        jax.random.PRNGKey(0), jnp.zeros((1, input_dim), dtype=jnp.float32)
    )
    return actor, params


def test_transplant_copies_only_hidden_features_and_keeps_zero_head():
    from src.algorithms.shac.residual_preview_adapter import (
        transplant_zero_head_recovery_features,
    )

    actor, template = _adapter_params()
    expert = copy.deepcopy(template)
    expert["params"]["Dense_0"]["kernel"] = jnp.arange(
        20, dtype=jnp.float32
    ).reshape(5, 4)
    expert["params"]["Dense_0"]["bias"] = jnp.arange(4, dtype=jnp.float32)
    expert["params"]["Dense_1"]["kernel"] = jnp.ones((4, 2))
    expert["params"]["Dense_1"]["bias"] = jnp.ones((2,))

    candidate, report = transplant_zero_head_recovery_features(
        template, expert
    )

    np.testing.assert_array_equal(
        candidate["params"]["Dense_0"]["kernel"],
        expert["params"]["Dense_0"]["kernel"],
    )
    np.testing.assert_array_equal(
        candidate["params"]["Dense_0"]["bias"],
        expert["params"]["Dense_0"]["bias"],
    )
    np.testing.assert_array_equal(
        candidate["params"]["Dense_1"]["kernel"],
        template["params"]["Dense_1"]["kernel"],
    )
    np.testing.assert_array_equal(
        candidate["params"]["Dense_1"]["bias"],
        template["params"]["Dense_1"]["bias"],
    )
    np.testing.assert_array_equal(
        actor.apply(candidate, jnp.ones((3, 5), dtype=jnp.float32)),
        jnp.zeros((3, 2), dtype=jnp.float32),
    )
    assert report == {
        "protocol": "g1-zero-head-recovery-feature-transfer-v1",
        "input_dim": 5,
        "hidden_dim": 4,
        "action_dim": 2,
        "hidden_kernel_exact": True,
        "hidden_bias_exact": True,
        "output_head_zero": True,
        "valid": True,
    }


@pytest.mark.parametrize(
    "mutation",
    ("shape", "mutual_shape", "nonfinite", "template_head", "negative_zero"),
)
def test_transplant_fails_closed_on_invalid_source_or_template(mutation: str):
    from src.algorithms.shac.residual_preview_adapter import (
        transplant_zero_head_recovery_features,
    )

    _, template = _adapter_params()
    expert = copy.deepcopy(template)
    if mutation == "shape":
        expert["params"]["Dense_0"]["kernel"] = jnp.zeros((6, 4))
    elif mutation == "mutual_shape":
        for params in (template, expert):
            params["params"]["Dense_0"]["bias"] = jnp.zeros((3,))
    elif mutation == "nonfinite":
        expert["params"]["Dense_0"]["bias"] = jnp.full((4,), jnp.nan)
    elif mutation == "template_head":
        template["params"]["Dense_1"]["bias"] = jnp.ones((2,))
    else:
        template["params"]["Dense_1"]["bias"] = jnp.full(
            (2,), -0.0, dtype=jnp.float32
        )

    with pytest.raises(ValueError, match="zero-head recovery feature"):
        transplant_zero_head_recovery_features(template, expert)


def test_feature_transfer_resume_settings_start_once_and_then_match_exactly():
    from src.algorithms.shac.residual_preview_adapter import (
        resolve_zero_head_feature_transfer_resume_setting,
    )

    path = "/tmp/e038.pkl"
    digest = "a" * 64
    assert resolve_zero_head_feature_transfer_resume_setting(
        {"actor_residual_preview_adapter": False},
        requested_path=path,
        requested_sha256=digest,
        residual_adapter_enabled=True,
        residual_adapter_upgrade=True,
        is_resume=True,
    ) == (path, digest)
    treated = {
        "actor_residual_preview_adapter": True,
        "actor_residual_preview_initial_adapter_path": path,
        "actor_residual_preview_initial_adapter_sha256": digest,
    }
    assert resolve_zero_head_feature_transfer_resume_setting(
        treated,
        requested_path=path,
        requested_sha256=digest,
        residual_adapter_enabled=True,
        residual_adapter_upgrade=False,
        is_resume=True,
    ) == (path, digest)

    with pytest.raises(ValueError, match="path and SHA"):
        resolve_zero_head_feature_transfer_resume_setting(
            treated,
            requested_path=path,
            requested_sha256=None,
            residual_adapter_enabled=True,
            residual_adapter_upgrade=False,
            is_resume=True,
        )
    with pytest.raises(ValueError, match="must match"):
        resolve_zero_head_feature_transfer_resume_setting(
            treated,
            requested_path="/tmp/other.pkl",
            requested_sha256=digest,
            residual_adapter_enabled=True,
            residual_adapter_upgrade=False,
            is_resume=True,
        )
    with pytest.raises(ValueError, match="requires residual adapter upgrade"):
        resolve_zero_head_feature_transfer_resume_setting(
            {"actor_residual_preview_adapter": True},
            requested_path=path,
            requested_sha256=digest,
            residual_adapter_enabled=True,
            residual_adapter_upgrade=False,
            is_resume=True,
        )
    with pytest.raises(ValueError, match="requires a resumed checkpoint"):
        resolve_zero_head_feature_transfer_resume_setting(
            None,
            requested_path=path,
            requested_sha256=digest,
            residual_adapter_enabled=True,
            residual_adapter_upgrade=False,
            is_resume=False,
        )
    with pytest.raises(ValueError, match="plain parent"):
        resolve_zero_head_feature_transfer_resume_setting(
            treated,
            requested_path=path,
            requested_sha256=digest,
            residual_adapter_enabled=True,
            residual_adapter_upgrade=True,
            is_resume=True,
        )
    with pytest.raises(ValueError, match="hexadecimal"):
        resolve_zero_head_feature_transfer_resume_setting(
            {"actor_residual_preview_adapter": False},
            requested_path=path,
            requested_sha256="z" * 64,
            residual_adapter_enabled=True,
            residual_adapter_upgrade=True,
            is_resume=True,
        )
