from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.centroidal_objective import (
    centroidal_window_objective,
    pseudo_huber,
)


def _trajectory() -> tuple[jax.Array, jax.Array, jax.Array]:
    reference = jnp.zeros((9, 6), dtype=jnp.float64)
    actual = reference.at[4:, 0].set(jnp.arange(1.0, 6.0))
    quaternions = jnp.tile(
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64), (9, 1)
    )
    return actual, reference, quaternions


def test_four_step_window_has_no_padding_or_wrap() -> None:
    actual, reference, quaternions = _trajectory()

    result = centroidal_window_objective(
        actual,
        reference,
        quaternions,
        done=jnp.zeros(8, dtype=bool),
        active=jnp.ones(8, dtype=bool),
        window=4,
        linear_scale=2.0,
        angular_scale=3.0,
        delta=0.1,
    )

    assert result.valid.tolist() == [True, True, True, True, True]
    assert result.error.shape == (5, 6)
    assert int(result.valid_count) == 5


def test_window_crossing_terminal_or_inactive_transition_is_excluded() -> None:
    actual, reference, quaternions = _trajectory()
    done = jnp.asarray([False, False, True, False, False, False, False, False])
    active = jnp.asarray([True, True, True, True, True, True, True, False])

    result = centroidal_window_objective(
        actual,
        reference,
        quaternions,
        done=done,
        active=active,
        window=4,
        linear_scale=2.0,
        angular_scale=3.0,
        delta=0.1,
    )

    assert result.valid.tolist() == [False, False, False, True, False]
    assert int(result.valid_count) == 1


def test_pseudo_huber_is_zero_at_match_and_has_finite_nonzero_gradient() -> None:
    actual, reference, quaternions = _trajectory()

    matched = centroidal_window_objective(
        reference,
        reference,
        quaternions,
        done=jnp.zeros(8, dtype=bool),
        active=jnp.ones(8, dtype=bool),
        window=4,
        linear_scale=2.0,
        angular_scale=3.0,
        delta=0.1,
    )
    def loss_fn(value):
        return centroidal_window_objective(
            value,
            reference,
            quaternions,
            done=jnp.zeros(8, dtype=bool),
            active=jnp.ones(8, dtype=bool),
            window=4,
            linear_scale=2.0,
            angular_scale=3.0,
            delta=0.1,
        ).loss

    gradient = jax.grad(loss_fn)(actual)

    assert float(matched.loss) == 0.0
    assert jnp.isfinite(gradient).all()
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_pseudo_huber_matches_registered_closed_form() -> None:
    value = jnp.asarray([-0.2, 0.0, 0.2], dtype=jnp.float64)
    expected = 0.01 * (jnp.sqrt(1.0 + jnp.square(value / 0.1)) - 1.0)
    np.testing.assert_allclose(pseudo_huber(value, 0.1), expected)


def test_no_valid_windows_returns_finite_zero() -> None:
    actual, reference, quaternions = _trajectory()
    result = centroidal_window_objective(
        actual,
        reference,
        quaternions,
        done=jnp.ones(8, dtype=bool),
        active=jnp.ones(8, dtype=bool),
        window=4,
        linear_scale=2.0,
        angular_scale=3.0,
        delta=0.1,
    )

    assert float(result.loss) == 0.0
    assert float(result.p99_forward_abs) == 0.0
    assert int(result.valid_count) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"window": 3},
        {"delta": 0.2},
        {"linear_scale": 0.0},
        {"angular_scale": float("nan")},
    ],
)
def test_registered_contract_rejects_other_objectives(overrides) -> None:
    actual, reference, quaternions = _trajectory()
    kwargs = {
        "done": jnp.zeros(8, dtype=bool),
        "active": jnp.ones(8, dtype=bool),
        "window": 4,
        "linear_scale": 2.0,
        "angular_scale": 3.0,
        "delta": 0.1,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        centroidal_window_objective(
            actual, reference, quaternions, **kwargs
        )
