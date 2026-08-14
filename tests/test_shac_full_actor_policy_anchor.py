import inspect

import jax
import jax.numpy as jnp
import pytest


def test_policy_anchor_penalty_is_zero_for_identical_actions():
    from src.algorithms.shac.algorithm import policy_anchor_penalty

    action = jnp.asarray([0.2, -0.4, 0.7])
    assert float(policy_anchor_penalty(action, action, weight=1.0)) == 0.0


def test_policy_anchor_penalty_is_quadratic_with_finite_candidate_gradient():
    from src.algorithms.shac.algorithm import policy_anchor_penalty

    parent = jnp.asarray([0.0, 0.0])
    candidate = jnp.asarray([1.0, -1.0])
    value = policy_anchor_penalty(candidate, parent, weight=0.5)
    gradient = jax.grad(policy_anchor_penalty)(candidate, parent, weight=0.5)

    assert float(value) == pytest.approx(0.5)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, jnp.asarray([0.5, -0.5]))


def test_policy_anchor_configuration_is_default_off_and_fail_closed():
    from src.algorithms.shac.algorithm import (
        train,
        validate_actor_policy_anchor_configuration,
    )

    assert inspect.signature(train).parameters[
        "actor_policy_anchor_weight"
    ].default == 0.0
    validate_actor_policy_anchor_configuration(
        weight=0.0,
        initial_full_actor_policy=None,
        resume_from=None,
    )
    with pytest.raises(ValueError, match="full actor"):
        validate_actor_policy_anchor_configuration(
            weight=1.0,
            initial_full_actor_policy=None,
            resume_from=None,
        )
    with pytest.raises(ValueError, match="fresh run"):
        validate_actor_policy_anchor_configuration(
            weight=1.0,
            initial_full_actor_policy=object(),
            resume_from="checkpoint.pkl",
        )
    for invalid in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="non-negative and finite"):
            validate_actor_policy_anchor_configuration(
                weight=invalid,
                initial_full_actor_policy=object(),
                resume_from=None,
            )


def test_train_wires_policy_anchor_into_rollout_loss_and_hparams():
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)
    assert "policy_anchor_penalty(" in source
    assert '"actor_policy_anchor_squared_error"' in source
    assert '"actor_policy_anchor_weight": actor_policy_anchor_weight' in source


def test_preview_anchors_reuse_expanded_candidate_and_parent_actions():
    """The 154-wide frozen parent must never receive the 328-wide preview obs."""
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)
    assert "if actor_residual_preview_adapter or actor_preview_adapter:" in source
    assert "(action, parent_action)" in source


def test_policy_anchor_checkpoint_telemetry_is_finite_and_explicit():
    from src.algorithms.shac.algorithm import build_policy_anchor_telemetry

    assert build_policy_anchor_telemetry(
        {"actor_policy_anchor_squared_error": jnp.asarray(0.25)},
        weight=1.0,
    ) == {
        "actor_policy_anchor_weight": 1.0,
        "actor_policy_anchor_squared_error": 0.25,
        "actor_policy_anchor_valid": True,
    }
    with pytest.raises(ValueError, match="finite"):
        build_policy_anchor_telemetry(
            {"actor_policy_anchor_squared_error": jnp.asarray(float("nan"))},
            weight=1.0,
        )
