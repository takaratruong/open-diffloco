from dataclasses import dataclass, replace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from src.algorithms.shac.contact_compliance import (
    backward_from_compliant,
    with_contact_time_constant,
)


def test_backward_from_compliant_has_exact_hard_primal_and_soft_cotangent():
    hard = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    compliant = jnp.asarray([9.0, 8.0], dtype=jnp.float32)

    value = backward_from_compliant(hard, compliant)
    hard_gradient, compliant_gradient = jax.grad(
        lambda hard_value, compliant_value: jnp.sum(
            backward_from_compliant(hard_value, compliant_value) ** 2
        ),
        argnums=(0, 1),
    )(hard, compliant)

    np.testing.assert_array_equal(value, hard)
    np.testing.assert_array_equal(hard_gradient, jnp.zeros_like(hard))
    np.testing.assert_array_equal(compliant_gradient, 2.0 * hard)


def test_backward_from_compliant_preserves_hard_discrete_leaves():
    hard = {
        "continuous": jnp.asarray([1.0, 2.0]),
        "phase": jnp.asarray(3, dtype=jnp.int32),
        "done": jnp.asarray(False),
    }
    compliant = {
        "continuous": jnp.asarray([8.0, 9.0]),
        "phase": jnp.asarray(4, dtype=jnp.int32),
        "done": jnp.asarray(True),
    }

    value = backward_from_compliant(hard, compliant)

    np.testing.assert_array_equal(value["continuous"], hard["continuous"])
    np.testing.assert_array_equal(value["phase"], hard["phase"])
    np.testing.assert_array_equal(value["done"], hard["done"])


def test_backward_from_compliant_rejects_tree_shape_and_dtype_mismatch():
    with pytest.raises(ValueError, match="tree structure"):
        backward_from_compliant({"x": jnp.ones(2)}, {"y": jnp.ones(2)})
    with pytest.raises(ValueError, match="shape and dtype"):
        backward_from_compliant(jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="shape and dtype"):
        backward_from_compliant(
            jnp.ones(2, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.int32),
        )


@dataclass(frozen=True)
class _FakeModel:
    geom_solref: jax.Array
    marker: jax.Array

    def replace(self, **changes):
        return replace(self, **changes)


def test_with_contact_time_constant_changes_only_first_solref_column():
    model = _FakeModel(
        geom_solref=jnp.asarray([[0.02, 1.0], [0.02, 1.0]]),
        marker=jnp.asarray([4.0]),
    )

    actual = with_contact_time_constant(model, 0.05)

    np.testing.assert_allclose(actual.geom_solref, [[0.05, 1.0], [0.05, 1.0]])
    np.testing.assert_array_equal(actual.marker, model.marker)
    np.testing.assert_allclose(model.geom_solref, [[0.02, 1.0], [0.02, 1.0]])


@pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf])
def test_with_contact_time_constant_rejects_invalid_values(value):
    model = _FakeModel(jnp.asarray([[0.02, 1.0]]), jnp.asarray([4.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        with_contact_time_constant(model, value)


def test_with_contact_time_constant_requires_two_solref_columns():
    model = _FakeModel(jnp.ones((2, 3)), jnp.asarray([4.0]))
    with pytest.raises(ValueError, match="shape"):
        with_contact_time_constant(model, 0.05)
