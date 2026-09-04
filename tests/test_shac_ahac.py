from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def test_contact_stiffness_uses_matching_root_constraint_and_acceleration() -> None:
    from src.algorithms.shac.ahac import contact_stiffness

    constraint = jp.asarray([3.0, 4.0, 0.0, 12.0, 0.0, 0.0, 999.0])
    acceleration = jp.asarray([0.0, -2.0, 4.0, 3.0, 0.5, -0.5, 0.0])

    actual = contact_stiffness(constraint, acceleration)

    # Root-only ratios are [3, 2, 0, 4, 0, 0].
    np.testing.assert_allclose(actual, np.sqrt(29.0), rtol=0.0, atol=1e-6)


def test_contact_stiffness_rejects_wrong_or_nonfinite_inputs() -> None:
    from src.algorithms.shac.ahac import contact_stiffness

    with np.testing.assert_raises_regex(ValueError, "at least six"):
        contact_stiffness(jp.ones((5,)), jp.ones((5,)))
    with np.testing.assert_raises_regex(ValueError, "matching"):
        contact_stiffness(jp.ones((6,)), jp.ones((7,)))

    actual = contact_stiffness(
        jp.asarray([jp.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),
        jp.ones((6,)),
    )
    assert not bool(jp.isfinite(actual))


def test_all_body_spatial_contact_stiffness_matches_official_normalization() -> None:
    from src.core.contact import all_body_spatial_contact_stiffness

    contact_force = jp.asarray(
        [
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 12.0, 0.0, 0.0, 0.0],
        ]
    )
    spatial_acceleration = jp.asarray(
        [
            [-2.0, 4.0, 0.5, 1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 1.0, 1.0, 1.0],
        ]
    )

    # Official source uses max(acceleration, 1) elementwise, not abs.
    np.testing.assert_allclose(
        all_body_spatial_contact_stiffness(
            contact_force, spatial_acceleration
        ),
        np.sqrt(3.0**2 + 1.0**2 + 4.0**2),
        rtol=0.0,
        atol=1e-6,
    )


def test_all_body_spatial_contact_stiffness_validates_body_spatial_layout() -> None:
    from src.core.contact import all_body_spatial_contact_stiffness

    with np.testing.assert_raises_regex(ValueError, "matching"):
        all_body_spatial_contact_stiffness(
            jp.ones((2, 6)), jp.ones((3, 6))
        )
    with np.testing.assert_raises_regex(ValueError, "body-by-spatial"):
        all_body_spatial_contact_stiffness(jp.ones((6,)), jp.ones((6,)))
    with np.testing.assert_raises_regex(ValueError, "six"):
        all_body_spatial_contact_stiffness(
            jp.ones((2, 5)), jp.ones((2, 5))
        )


def test_active_horizon_mask_rounds_and_preserves_exact_bounds() -> None:
    from src.algorithms.shac.ahac import active_horizon_mask

    np.testing.assert_array_equal(
        active_horizon_mask(jp.asarray(8.49), 24),
        np.asarray([True] * 8 + [False] * 16),
    )
    np.testing.assert_array_equal(
        active_horizon_mask(jp.asarray(8.5), 24),
        np.asarray([True] * 9 + [False] * 15),
    )
    np.testing.assert_array_equal(
        active_horizon_mask(jp.asarray(24.0), 24),
        np.ones((24,), dtype=bool),
    )


def test_inactive_horizon_step_excises_an_undefined_zero_cotangent_pullback() -> None:
    from src.algorithms.shac.ahac import (
        evaluate_with_inactive_gradient_excision,
    )

    @jax.custom_vjp
    def undefined_pullback(value):
        return jp.square(value)

    def forward(value):
        return jp.square(value), None

    def backward(_, cotangent):
        return (jp.full_like(cotangent, jp.nan),)

    undefined_pullback.defvjp(forward, backward)

    inactive_value, inactive_gradient = jax.jit(
        jax.value_and_grad(
            lambda value: evaluate_with_inactive_gradient_excision(
                undefined_pullback,
                value,
                active=jp.asarray(False),
            )
        )
    )(jp.asarray(2.0))
    active_value, active_gradient = jax.jit(
        jax.value_and_grad(
            lambda value: evaluate_with_inactive_gradient_excision(
                undefined_pullback,
                value,
                active=jp.asarray(True),
            )
        )
    )(jp.asarray(2.0))

    np.testing.assert_allclose(inactive_value, 4.0)
    np.testing.assert_allclose(active_value, 4.0)
    np.testing.assert_allclose(inactive_gradient, 0.0)
    assert bool(jp.isnan(active_gradient))


def test_runtime_pullback_gate_shares_one_compiled_primal_and_skips_undefined_vjp() -> None:
    from src.algorithms.shac.ahac import evaluate_with_runtime_pullback_gate

    @jax.custom_vjp
    def undefined_pullback(value):
        return jp.square(value)

    def forward(value):
        return jp.square(value), None

    def backward(_, cotangent):
        return (jp.full_like(cotangent, jp.nan),)

    undefined_pullback.defvjp(forward, backward)

    compiled = jax.jit(
        jax.value_and_grad(
            lambda value, pullback_active: evaluate_with_runtime_pullback_gate(
                undefined_pullback,
                value,
                pullback_active=pullback_active,
            )
        )
    ).lower(jp.asarray(2.0), jp.asarray(False)).compile()

    inactive_value, inactive_gradient = compiled(
        jp.asarray(2.0), jp.asarray(False)
    )
    active_value, active_gradient = compiled(
        jp.asarray(2.0), jp.asarray(True)
    )

    np.testing.assert_array_equal(inactive_value, active_value)
    np.testing.assert_allclose(inactive_value, 4.0)
    np.testing.assert_allclose(inactive_gradient, 0.0)
    assert bool(jp.isnan(active_gradient))


def test_runtime_pullback_gate_returns_float0_for_integer_pytree_leaves() -> None:
    from src.algorithms.shac.ahac import evaluate_with_runtime_pullback_gate

    def mixed_tree_step(tree):
        return {
            "count": tree["count"] + 1,
            "position": jp.square(tree["position"]),
        }

    def loss(position, pullback_active):
        result = evaluate_with_runtime_pullback_gate(
            mixed_tree_step,
            {
                "count": jp.asarray(3, dtype=jp.int32),
                "position": position,
            },
            pullback_active=pullback_active,
        )
        return result["position"]

    compiled = jax.jit(jax.value_and_grad(loss)).lower(
        jp.asarray(2.0), jp.asarray(False)
    ).compile()

    inactive_value, inactive_gradient = compiled(
        jp.asarray(2.0), jp.asarray(False)
    )
    active_value, active_gradient = compiled(
        jp.asarray(2.0), jp.asarray(True)
    )

    np.testing.assert_array_equal(inactive_value, active_value)
    np.testing.assert_allclose(inactive_gradient, 0.0)
    np.testing.assert_allclose(active_gradient, 4.0)


def test_runtime_pullback_gate_removes_only_inactive_scan_transposes() -> None:
    from src.algorithms.shac.ahac import evaluate_with_runtime_pullback_gate

    @jax.custom_vjp
    def zero_cotangent_unsafe_square(value):
        return jp.square(value)

    def forward(value):
        return jp.square(value), value

    def backward(value, cotangent):
        derivative = 2.0 * value * cotangent
        return (jp.where(cotangent == 0.0, jp.nan, derivative),)

    zero_cotangent_unsafe_square.defvjp(forward, backward)

    def rollout_loss(parameter, excise_inactive):
        def step(state, index):
            active = index < 1
            candidate = evaluate_with_runtime_pullback_gate(
                zero_cotangent_unsafe_square,
                state,
                pullback_active=(~excise_inactive) | active,
            )
            next_state = jp.where(active, candidate, state)
            reward = jp.where(active, candidate, 0.0)
            return next_state, reward

        _, rewards = jax.lax.scan(step, parameter, jp.arange(3))
        return jp.sum(rewards)

    compiled = jax.jit(jax.value_and_grad(rollout_loss)).lower(
        jp.asarray(2.0), jp.asarray(False)
    ).compile()
    connected_value, connected_gradient = compiled(
        jp.asarray(2.0), jp.asarray(False)
    )
    excised_value, excised_gradient = compiled(
        jp.asarray(2.0), jp.asarray(True)
    )

    np.testing.assert_array_equal(connected_value, excised_value)
    np.testing.assert_allclose(connected_value, 4.0)
    assert bool(jp.isnan(connected_gradient))
    np.testing.assert_allclose(excised_gradient, 4.0)


def test_masked_mean_excludes_inactive_slots() -> None:
    from src.algorithms.shac.ahac import masked_mean

    values = jp.asarray([1.0, 3.0, 1_000_000.0])
    mask = jp.asarray([True, True, False])
    np.testing.assert_allclose(masked_mean(values, mask), 2.0)


def test_equation_10_contact_penalty_keeps_policy_gradient_and_stops_dual() -> None:
    import jax

    from src.algorithms.shac.ahac import adaptive_contact_penalty

    contact = jp.asarray([12.0, 8.0, 1_000.0])
    dual = jp.asarray([2.0, 3.0, 999.0])
    active = jp.asarray([True, True, False])

    penalty = adaptive_contact_penalty(
        contact_by_step=contact,
        dual=dual,
        active_mask=active,
        threshold=10.0,
    )
    contact_gradient = jax.grad(
        lambda values: adaptive_contact_penalty(
            contact_by_step=values,
            dual=dual,
            active_mask=active,
            threshold=10.0,
        )
    )(contact)
    dual_gradient = jax.grad(
        lambda values: adaptive_contact_penalty(
            contact_by_step=contact,
            dual=values,
            active_mask=active,
            threshold=10.0,
        )
    )(dual)

    np.testing.assert_allclose(penalty, -1.0)
    np.testing.assert_allclose(contact_gradient, [1.0, 1.5, 0.0])
    np.testing.assert_allclose(dual_gradient, [0.0, 0.0, 0.0])


def test_equation_10_contact_penalty_validates_vector_layout() -> None:
    from src.algorithms.shac.ahac import adaptive_contact_penalty

    with np.testing.assert_raises_regex(ValueError, "matching vectors"):
        adaptive_contact_penalty(
            contact_by_step=jp.ones((2,)),
            dual=jp.ones((3,)),
            active_mask=jp.ones((2,), dtype=bool),
            threshold=1.0,
        )


def test_dual_update_is_projected_and_only_active_slots_change() -> None:
    from src.algorithms.shac.ahac import update_horizon_dual

    result = update_horizon_dual(
        horizon=jp.asarray(8.0),
        dual=jp.zeros((4,)),
        contact_by_step=jp.asarray([8.0, 12.0, 9.0, 1_000.0]),
        active_mask=jp.asarray([True, True, True, False]),
        threshold=10.0,
        learning_rate=0.5,
        minimum=2,
        maximum=4,
    )

    # Negative violations project to zero; the inactive large value is ignored.
    np.testing.assert_allclose(result.dual, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.horizon, 4.0)
    assert bool(result.valid)


def test_dual_update_clips_horizon_and_flags_nonfinite_contact() -> None:
    from src.algorithms.shac.ahac import update_horizon_dual

    clipped = update_horizon_dual(
        horizon=jp.asarray(23.9),
        dual=jp.ones((2,)),
        contact_by_step=jp.asarray([20.0, 20.0]),
        active_mask=jp.asarray([True, True]),
        threshold=1.0,
        learning_rate=1.0,
        minimum=8,
        maximum=24,
    )
    np.testing.assert_allclose(clipped.horizon, 24.0)

    invalid = update_horizon_dual(
        horizon=jp.asarray(8.0),
        dual=jp.zeros((2,)),
        contact_by_step=jp.asarray([jp.nan, 2.0]),
        active_mask=jp.asarray([True, True]),
        threshold=1.0,
        learning_rate=0.1,
        minimum=1,
        maximum=2,
    )
    assert not bool(invalid.valid)


def test_critic_convergence_requires_five_finite_stable_losses() -> None:
    from src.algorithms.shac.ahac import critic_convergence

    assert bool(
        critic_convergence(jp.asarray([1.0, 0.95, 0.91, 0.88, 0.86]), 0.2)
    )
    assert not bool(critic_convergence(jp.asarray([5.0, 4.0, 3.0, 2.0]), 0.2))
    assert not bool(
        critic_convergence(jp.asarray([1.0, 0.9, jp.nan, 0.8, 0.7]), 0.2)
    )
    assert not bool(
        critic_convergence(jp.asarray([5.0, 4.0, 3.0, 2.0, 1.0]), 0.2)
    )
