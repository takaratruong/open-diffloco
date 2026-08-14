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


def test_missing_phase_bin_marks_finalized_gradients_invalid():
    accumulator = accumulate_phase_gradients(
        {"w": jnp.arange(8, dtype=jnp.float32).reshape(4, 2)},
        jnp.array([0, 100, 200, 300]),
        phase_count=500,
        bin_count=5,
    )

    _, counts, valid = finalize_phase_gradients(accumulator)

    np.testing.assert_array_equal(counts, np.array([1, 1, 1, 1, 0]))
    assert not bool(valid)


def test_phase_accumulator_clips_each_environment_before_bin_mean():
    gradients = {
        "w": jnp.array(
            [
                [3.0, 4.0],
                [300.0, 400.0],
                [jnp.nan, 1.0],
                [0.0, 2.0],
            ]
        )
    }
    accumulator = accumulate_phase_gradients(
        gradients,
        jnp.array([0, 0, 100, 100]),
        phase_count=200,
        bin_count=2,
        per_env_max_norm=10.0,
    )

    tasks, counts, valid = finalize_phase_gradients(accumulator)

    # Bin zero averages [3,4] with the clipped [6,8]. The nonfinite rollout
    # in bin one is removed as a whole, leaving only [0,2].
    np.testing.assert_allclose(tasks["w"], np.array([[4.5, 6.0], [0.0, 2.0]]))
    np.testing.assert_array_equal(counts, np.array([2, 1]))
    assert bool(valid)


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


def test_nonfinite_task_gradient_marks_cagrad_result_invalid():
    task_gradients = jnp.eye(5, dtype=jnp.float32).at[2, 3].set(jnp.nan)

    result = combine_cagrad(
        {"w": task_gradients}, alpha=0.5, iterations=32
    )

    assert not bool(result.valid)


@pytest.mark.parametrize("seed", [3, 10])
def test_fixed_frank_wolfe_matches_scipy_simplex_oracle(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(12, 5)).astype(np.float32)
    gram = matrix.T @ matrix
    uniform = np.full(5, 1.0 / 5)
    eps = 1e-8
    alpha = 0.5
    coefficient = alpha * np.sqrt(uniform @ gram @ uniform)

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

    task_gradients = matrix.T
    with jax.default_matmul_precision("highest"):
        result = combine_cagrad(
            {"w": jnp.asarray(task_gradients)}, alpha=alpha, iterations=32
        )

    np.testing.assert_allclose(result.gram_matrix, gram, rtol=1e-5, atol=1e-5)
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(result)
    )
    assert abs(float(jnp.sum(result.weights)) - 1.0) <= 1e-6
    assert float(jnp.min(result.weights)) >= 0.0
    assert float(jnp.max(result.weights)) <= 1.0
    assert abs(float(result.objective) - oracle.fun) <= 1e-4

    oracle_weighted_gradient = oracle.x @ task_gradients
    uniform_gradient = np.mean(task_gradients, axis=0)
    unscaled_oracle_gradient = (
        uniform_gradient
        + coefficient
        * oracle_weighted_gradient
        / (np.linalg.norm(oracle_weighted_gradient) + eps)
    )
    oracle_combined_gradient = unscaled_oracle_gradient / (1.0 + alpha**2)
    np.testing.assert_allclose(
        result.combined_gradient["w"],
        oracle_combined_gradient,
        rtol=5e-3,
        atol=5e-3,
    )
    assert not np.allclose(
        uniform_gradient,
        oracle_combined_gradient,
        rtol=5e-3,
        atol=5e-3,
    )
    assert not np.allclose(
        unscaled_oracle_gradient,
        oracle_combined_gradient,
        rtol=5e-3,
        atol=5e-3,
    )

    task_norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    expected_cosine_matrix = gram / (
        task_norms[:, None] * task_norms[None, :] + eps
    )
    np.testing.assert_allclose(
        result.cosine_matrix,
        expected_cosine_matrix,
        rtol=1e-5,
        atol=1e-5,
    )
    weights = np.asarray(result.weights)
    objective_gradient = (
        gram @ uniform
        + coefficient
        * (gram @ weights)
        / np.sqrt(weights @ gram @ weights + eps)
    )
    expected_dual_gap = weights @ objective_gradient - np.min(
        objective_gradient
    )
    np.testing.assert_allclose(result.dual_gap, expected_dual_gap, atol=1e-5)
    assert float(result.dual_gap) >= -1e-6

    combined_gradient = np.asarray(result.combined_gradient["w"])
    expected_uniform_combined_cosine = np.vdot(
        uniform_gradient, combined_gradient
    ) / (
        np.linalg.norm(uniform_gradient) * np.linalg.norm(combined_gradient)
        + eps
    )
    np.testing.assert_allclose(
        result.uniform_combined_cosine,
        expected_uniform_combined_cosine,
        atol=1e-6,
    )
    assert bool(result.valid)
