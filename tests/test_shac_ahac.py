from __future__ import annotations

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


def test_masked_mean_excludes_inactive_slots() -> None:
    from src.algorithms.shac.ahac import masked_mean

    values = jp.asarray([1.0, 3.0, 1_000_000.0])
    mask = jp.asarray([True, True, False])
    np.testing.assert_allclose(masked_mean(values, mask), 2.0)


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
