import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from src.algorithms.shac.cagrad import (
    accumulate_phase_gradients,
    combine_cagrad,
    finalize_phase_gradients,
    merge_phase_gradient_accumulators,
)


def _assert_trees_allclose(actual, expected, **kwargs):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(actual_leaf, expected_leaf, **kwargs)


def test_two_shards_merge_like_one_population_with_finite_aware_means():
    phases = jnp.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 499])
    per_env_gradients = jnp.array(
        [
            [1.0, jnp.nan],
            [3.0, jnp.inf],
            [7.0, 11.0],
            [jnp.inf, 13.0],
            [17.0, 19.0],
            [23.0, 29.0],
            [31.0, 37.0],
            [41.0, jnp.nan],
            [43.0, 47.0],
            [53.0, 59.0],
        ]
    )
    full = accumulate_phase_gradients(
        {"w": per_env_gradients},
        phases,
        phase_count=500,
        bin_count=5,
    )
    first = accumulate_phase_gradients(
        {"w": per_env_gradients[:4]},
        phases[:4],
        phase_count=500,
        bin_count=5,
    )
    second = accumulate_phase_gradients(
        {"w": per_env_gradients[4:]},
        phases[4:],
        phase_count=500,
        bin_count=5,
    )
    merged = merge_phase_gradient_accumulators(first, second)

    _assert_trees_allclose(merged.sums, full.sums)
    _assert_trees_allclose(merged.finite_counts, full.finite_counts)
    np.testing.assert_array_equal(
        merged.finite_counts["w"],
        np.array([[2, 0], [1, 2], [2, 2], [2, 1], [2, 2]]),
    )
    np.testing.assert_array_equal(merged.env_counts, full.env_counts)

    full_tasks, full_counts, full_valid = finalize_phase_gradients(full)
    merged_tasks, merged_counts, merged_valid = finalize_phase_gradients(merged)
    _assert_trees_allclose(merged_tasks, full_tasks)
    _assert_trees_allclose(merged_counts, full_counts)
    assert bool(full_valid)
    assert bool(merged_valid)
    np.testing.assert_allclose(
        merged_tasks["w"],
        np.array(
            [
                [2.0, 0.0],
                [7.0, 12.0],
                [20.0, 24.0],
                [36.0, 37.0],
                [48.0, 53.0],
            ]
        ),
    )


def test_identical_task_gradients_produce_collinear_combined_direction():
    task = jnp.array([2.0, -3.0, 4.0])
    task_gradients = {"w": jnp.broadcast_to(task, (5, task.size))}

    result = combine_cagrad(task_gradients, alpha=0.5, iterations=32)

    combined = result.combined_gradient["w"]
    uniform_mean = jnp.mean(task_gradients["w"], axis=0)
    cosine = jnp.vdot(combined, uniform_mean) / (
        jnp.linalg.norm(combined) * jnp.linalg.norm(uniform_mean)
    )
    np.testing.assert_allclose(cosine, 1.0, atol=1e-6)
    np.testing.assert_allclose(jnp.sum(result.weights), 1.0, atol=1e-6)
    assert bool(result.valid)


@pytest.mark.parametrize("seed", [3, 10])
def test_fixed_frank_wolfe_matches_scipy_simplex_oracle(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(12, 5)).astype(np.float32)
    gram = matrix.T @ matrix
    uniform = np.full(5, 1.0 / 5)
    eps = 1e-8
    coefficient = 0.5 * np.sqrt(uniform @ gram @ uniform)

    def objective(weights):
        return (
            weights @ gram @ uniform
            + coefficient * np.sqrt(weights @ gram @ weights + eps)
        )

    oracle = minimize(
        objective,
        uniform,
        bounds=[(0.0, 1.0)] * 5,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    assert oracle.success, oracle.message

    result = combine_cagrad(
        {"w": jnp.asarray(matrix.T)}, alpha=0.5, iterations=32
    )

    np.testing.assert_allclose(result.gram_matrix, gram, rtol=1e-5, atol=1e-5)
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(result)
    )
    assert abs(float(jnp.sum(result.weights)) - 1.0) <= 1e-6
    assert float(jnp.min(result.weights)) >= 0.0
    assert abs(float(result.objective) - oracle.fun) <= 1e-4
    assert bool(result.valid)
