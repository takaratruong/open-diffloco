from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.capture_point_objective import capture_point_objective


def test_capture_objective_masks_inactive_states_without_padding() -> None:
    reference = jnp.zeros((5, 2), dtype=jnp.float64)
    actual = jnp.asarray(
        [[0.0, 0.0], [0.2, 0.0], [9.0, 9.0], [0.0, -0.4], [8.0, 8.0]],
        dtype=jnp.float64,
    )
    active = jnp.asarray([True, True, False, True, False])

    result = capture_point_objective(
        actual,
        reference,
        active=active,
        standing_height=2.0,
        delta=0.1,
    )

    assert result.valid.tolist() == active.tolist()
    assert int(result.valid_count) == 3
    np.testing.assert_allclose(
        result.normalized_error,
        np.asarray([[0.0, 0.0], [0.1, 0.0], [4.5, 4.5], [0.0, -0.2], [4.0, 4.0]]),
    )
    assert float(result.p99_norm) == pytest.approx(0.2)


def test_capture_objective_has_finite_nonzero_gradient() -> None:
    reference = jnp.zeros((4, 2), dtype=jnp.float64)
    actual = jnp.asarray(
        [[0.1, 0.0], [0.2, -0.1], [0.3, 0.2], [0.4, -0.2]],
        dtype=jnp.float64,
    )

    def loss_fn(value):
        return capture_point_objective(
            value,
            reference,
            active=jnp.ones(4, dtype=bool),
            standing_height=0.7,
            delta=0.1,
        ).loss

    gradient = jax.grad(loss_fn)(actual)
    assert jnp.isfinite(gradient).all()
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_no_valid_capture_states_returns_finite_zero() -> None:
    values = jnp.ones((3, 2), dtype=jnp.float64)
    result = capture_point_objective(
        values,
        jnp.zeros_like(values),
        active=jnp.zeros(3, dtype=bool),
        standing_height=0.7,
        delta=0.1,
    )

    assert float(result.loss) == 0.0
    assert float(result.p99_norm) == 0.0
    assert int(result.valid_count) == 0


@pytest.mark.parametrize(
    "standing_height,delta",
    [(0.0, 0.1), (0.7, 0.2), (float("nan"), 0.1)],
)
def test_capture_objective_enforces_registered_contract(
    standing_height, delta
) -> None:
    values = jnp.zeros((3, 2), dtype=jnp.float64)
    with pytest.raises(ValueError):
        capture_point_objective(
            values,
            values,
            active=jnp.ones(3, dtype=bool),
            standing_height=standing_height,
            delta=delta,
        )

