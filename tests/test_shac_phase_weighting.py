import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.phase_weighting import (
    phase_bin_indices,
    phase_robust_weights,
)


def test_five_bins_cover_all_499_start_phases():
    phases = jnp.array([0, 99, 100, 199, 200, 299, 300, 399, 400, 498])
    np.testing.assert_array_equal(
        phase_bin_indices(phases, phase_count=499, bin_count=5),
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
    )


def test_equal_bin_losses_produce_unit_detached_weights():
    losses = jnp.ones(5)
    phases = jnp.array([0, 100, 200, 300, 400])
    result = phase_robust_weights(
        losses,
        phases,
        phase_count=499,
        bin_count=5,
        robust_fraction=0.5,
    )
    np.testing.assert_array_equal(result.bin_weights, np.ones(5))
    np.testing.assert_array_equal(result.env_weights, np.ones(5))
    np.testing.assert_array_equal(
        jax.jacrev(
            lambda values: phase_robust_weights(
                values,
                phases,
                phase_count=499,
                bin_count=5,
                robust_fraction=0.5,
            ).env_weights
        )(losses),
        np.zeros((5, 5)),
    )


def test_unique_hard_bin_is_largest_bounded_and_mean_one():
    losses = jnp.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0])
    phases = jnp.array([0, 50, 100, 200, 300, 400])
    result = jax.jit(
        lambda values: phase_robust_weights(
            values,
            phases,
            phase_count=499,
            bin_count=5,
            robust_fraction=0.5,
        )
    )(losses)
    assert int(jnp.argmax(result.bin_weights)) == 3
    assert float(result.bin_weights[result.bin_counts > 0].min()) > 0.5
    assert float(result.bin_weights.max()) <= 3.0
    np.testing.assert_allclose(result.env_weights.mean(), 1.0, atol=1e-7)


def test_equal_weights_reproduce_the_unweighted_gradient_mean():
    from src.algorithms.shac.phase_weighting import (
        aggregate_phase_weighted_gradients,
    )

    gradients = {"w": jnp.arange(10, dtype=jnp.float32).reshape(5, 2)}
    weighted = aggregate_phase_weighted_gradients(gradients, jnp.ones(5))
    np.testing.assert_allclose(weighted["w"], gradients["w"].mean(axis=0))


def test_nonfinite_gradient_elements_have_no_numerator_or_denominator():
    from src.algorithms.shac.phase_weighting import (
        aggregate_phase_weighted_gradients,
    )

    gradients = {
        "w": jnp.array([[2.0, jnp.inf], [4.0, 8.0], [jnp.nan, 14.0]])
    }
    weights = jnp.array([1.0, 2.0, 3.0])
    aggregate = aggregate_phase_weighted_gradients(gradients, weights)
    np.testing.assert_allclose(aggregate["w"], np.array([10.0 / 3.0, 11.6]))


def test_no_finite_gradient_contributor_emits_zero():
    from src.algorithms.shac.phase_weighting import (
        aggregate_phase_weighted_gradients,
    )

    aggregate = aggregate_phase_weighted_gradients(
        {"w": jnp.array([[jnp.nan], [jnp.inf]])}, jnp.ones(2)
    )
    np.testing.assert_array_equal(aggregate["w"], np.zeros(1))
