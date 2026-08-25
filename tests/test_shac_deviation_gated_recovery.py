import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.deviation_gated_recovery import (
    REGISTERED_DEVIATION_GATE,
    DeviationGate,
    compose_deviation_gated_recovery,
    deviation_recovery_gate,
)


def test_deviation_gate_has_exact_registered_endpoints():
    errors = jnp.asarray([0.0, 0.10, 0.15, 0.20, 1.0])

    gate = deviation_recovery_gate(errors, REGISTERED_DEVIATION_GATE)

    gate_array = np.asarray(gate)
    np.testing.assert_array_equal(gate_array[[0, 1]], [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(gate[2]), 0.5, atol=1e-7, rtol=0.0)
    np.testing.assert_array_equal(gate_array[[3, 4]], [1.0, 1.0])


@pytest.mark.parametrize("error", [np.nan, np.inf, -np.inf])
def test_deviation_gate_rejects_nonfinite_error(error):
    with pytest.raises(ValueError, match="finite"):
        deviation_recovery_gate(jnp.asarray(error), REGISTERED_DEVIATION_GATE)


@pytest.mark.parametrize(
    "contract",
    [
        DeviationGate(lower=0.2, upper=0.2),
        DeviationGate(lower=0.3, upper=0.2),
        DeviationGate(lower=-0.1, upper=0.2),
        DeviationGate(lower=np.nan, upper=0.2),
    ],
)
def test_deviation_gate_rejects_invalid_contract(contract):
    with pytest.raises(ValueError):
        deviation_recovery_gate(jnp.asarray(0.15), contract)


def test_composition_preserves_parent_at_zero_and_global_at_one():
    parent = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    residual = jnp.asarray([[0.5, -0.25], [-0.5, 0.25]])

    action, gated_residual, gate = compose_deviation_gated_recovery(
        parent,
        residual,
        jnp.asarray([0.0, 0.2]),
        REGISTERED_DEVIATION_GATE,
    )

    np.testing.assert_array_equal(np.asarray(action[0]), np.asarray(parent[0]))
    np.testing.assert_array_equal(
        np.asarray(gated_residual[0]), np.zeros(2, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(action[1]), np.asarray(parent[1] + residual[1])
    )
    np.testing.assert_array_equal(
        np.asarray(gated_residual[1]), np.asarray(residual[1])
    )
    np.testing.assert_array_equal(np.asarray(gate), [0.0, 1.0])


def test_composition_broadcasts_scalar_error_over_action_coordinates():
    parent = jnp.zeros((3, 29))
    residual = jnp.ones((3, 29))

    action, gated_residual, gate = compose_deviation_gated_recovery(
        parent,
        residual,
        jnp.asarray([0.1, 0.15, 0.2]),
        REGISTERED_DEVIATION_GATE,
    )

    np.testing.assert_allclose(
        np.asarray(action[:, 0]), [0.0, 0.5, 1.0], atol=1e-7, rtol=0.0
    )
    np.testing.assert_array_equal(np.asarray(action), np.asarray(gated_residual))
    assert gate.shape == (3,)


def test_composition_rejects_mismatched_action_shapes():
    with pytest.raises(ValueError, match="shapes"):
        compose_deviation_gated_recovery(
            jnp.zeros((2, 29)),
            jnp.zeros((2, 28)),
            jnp.zeros((2,)),
            REGISTERED_DEVIATION_GATE,
        )


def test_composition_rejects_error_shape_that_does_not_match_batch():
    with pytest.raises(ValueError, match="error shape"):
        compose_deviation_gated_recovery(
            jnp.zeros((2, 29)),
            jnp.zeros((2, 29)),
            jnp.zeros((3,)),
            REGISTERED_DEVIATION_GATE,
        )
