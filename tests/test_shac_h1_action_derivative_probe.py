import copy
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_h1_action_derivative_probe_defaults_disabled() -> None:
    from src.algorithms.shac.algorithm import train

    parameter = inspect.signature(train).parameters["actor_h1_action_derivative_probe"]
    assert parameter.default is False


def test_action_derivative_pair_matches_an_exact_smooth_objective() -> None:
    from src.algorithms.shac.algorithm import compute_action_derivative_pair

    action = jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float32)
    direction = jnp.asarray([0.5, -0.5, 0.5], dtype=jnp.float32)

    result = compute_action_derivative_pair(
        lambda candidate: jnp.sum(candidate * candidate),
        action,
        direction=direction,
        finite_difference_epsilon=1e-3,
    )

    assert bool(result["reverse_primal"] == result["forward_primal"])
    np.testing.assert_allclose(result["reverse_gradient"], 2.0 * action)
    np.testing.assert_allclose(result["forward_gradient"], 2.0 * action)
    np.testing.assert_allclose(
        result["forward_directional"], jnp.vdot(2.0 * action, direction)
    )
    np.testing.assert_allclose(
        result["finite_difference_directional"],
        result["forward_directional"],
        rtol=2e-4,
        atol=2e-4,
    )
    source = inspect.getsource(compute_action_derivative_pair)
    assert "jax.lax.scan(" in source
    assert "jax.vmap(pushforward)" not in source


def test_h1_action_derivative_contract_is_probe_only_and_one_step() -> None:
    from src.algorithms.shac.algorithm import (
        validate_actor_h1_action_derivative_probe_contract,
    )

    valid = {
        "enabled": True,
        "ahac": True,
        "unroll_length": 1,
        "ahac_horizon_min": 1,
        "ahac_horizon_max": 1,
        "ahac_semantics": "paper_equation_10_no_target",
        "actor_bootstrap_scale": 0.0,
        "actor_bootstrap_delay_steps": 0,
        "actor_bootstrap_graph_mode": "excised",
        "actor_forward_jvp_probe": False,
        "actor_cagrad": True,
        "has_unfactorized_actor_terms": False,
        "determinism_probe_output": "probe.json",
    }
    validate_actor_h1_action_derivative_probe_contract(**valid)
    validate_actor_h1_action_derivative_probe_contract(
        **{**valid, "enabled": False, "ahac": False}
    )

    invalid = (
        ("ahac", False, "requires AHAC"),
        ("ahac_semantics", "legacy_horizon_only_target", "paper AHAC"),
        ("unroll_length", 2, "one-step"),
        ("ahac_horizon_min", 2, "one-step"),
        ("ahac_horizon_max", 2, "one-step"),
        ("actor_bootstrap_scale", 1.0, "zero bootstrap"),
        ("actor_bootstrap_delay_steps", 1, "zero bootstrap"),
        ("actor_bootstrap_graph_mode", "connected", "excised"),
        ("actor_forward_jvp_probe", True, "mutually exclusive"),
        ("actor_cagrad", False, "CAGrad population"),
        ("has_unfactorized_actor_terms", True, "factorized objective"),
        ("determinism_probe_output", None, "probe-only"),
    )
    for field, value, message in invalid:
        with pytest.raises(ValueError, match=message):
            validate_actor_h1_action_derivative_probe_contract(
                **{**valid, field: value}
            )

    with pytest.raises(TypeError, match="boolean"):
        validate_actor_h1_action_derivative_probe_contract(**{**valid, "enabled": 1})


def _h1_metrics() -> dict[str, jax.Array]:
    return {
        "actor_cagrad_losses_by_env": jnp.asarray([1.0, 2.0, 3.0]),
        "actor_cagrad_gradient_finite_by_env": jnp.asarray([False, True, False]),
        "actor_h1_action_reverse_primal_by_env": jnp.asarray([1.0, 2.0, 3.0]),
        "actor_h1_action_forward_primal_by_env": jnp.asarray([1.0, 2.0, 3.0]),
        "actor_h1_action_reverse_gradient_by_env": jnp.asarray(
            [[jnp.nan, jnp.nan], [0.25, -0.5], [jnp.nan, 1.0]]
        ),
        "actor_h1_action_forward_gradient_by_env": jnp.asarray(
            [[0.5, 0.25], [0.25, -0.5], [0.75, 1.0]]
        ),
        "actor_h1_action_forward_directional_by_env": jnp.asarray([0.75, -0.25, 1.75]),
        "actor_h1_action_finite_difference_by_env": jnp.asarray([0.75, -0.25, 1.75]),
        "actor_h1_action_direction_fingerprint": jnp.asarray(
            [11, 12, 13, 14], dtype=jnp.uint32
        ),
        "actor_h1_action_direction_norm": jnp.asarray(1.0),
        "actor_h1_action_finite_difference_epsilon": jnp.asarray(1e-5),
        "actor_h1_action_dimension": jnp.asarray(2, dtype=jnp.int32),
    }


def test_h1_action_derivative_report_reconstructs_complete_matrices() -> None:
    from src.algorithms.shac.algorithm import (
        build_actor_h1_action_derivative_report,
    )

    first = _h1_metrics()
    report = build_actor_h1_action_derivative_report(first, copy.deepcopy(first))

    assert report["protocol"] == "shac-h1-action-derivative-pair-v1"
    assert report["valid"] is True
    assert report["population_size"] == 3
    assert report["action_dimension"] == 2
    assert report["source_primal_matches_action_reverse"] is True
    assert report["source_primal_matches_action_forward"] is True
    assert report["repeat_exact"] is True
    assert report["policy_reverse_finite_count"] == 1
    assert report["action_reverse_finite_count"] == 1
    assert report["action_forward_finite_count"] == 3
    assert report["forward_recovers_action_reverse_count"] == 2
    assert report["forward_recovers_policy_reverse_count"] == 2
    assert report["action_reverse_mask_matches_policy_reverse"] is True
    assert report["jointly_finite_count"] == 1
    assert report["jointly_finite_gradient_agreement_count"] == 1
    assert report["finite_difference_agreement_count"] == 3
    assert report["action_reverse_finite_by_env"] == [False, True, False]
    assert report["action_forward_finite_by_env"] == [True, True, True]
    assert report["action_reverse_gradients_by_env"] == [
        [None, None],
        [0.25, -0.5],
        [None, 1.0],
    ]
    assert report["action_forward_gradients_by_env"] == [
        [0.5, 0.25],
        [0.25, -0.5],
        [0.75, 1.0],
    ]


def test_h1_action_derivative_report_rejects_a_changed_forward_primal() -> None:
    from src.algorithms.shac.algorithm import (
        build_actor_h1_action_derivative_report,
    )

    first = _h1_metrics()
    first["actor_h1_action_forward_primal_by_env"] = jnp.asarray([1.0, 2.0, 3.000001])
    report = build_actor_h1_action_derivative_report(first, copy.deepcopy(first))

    assert report["valid"] is False
    assert report["source_primal_matches_action_forward"] is False
