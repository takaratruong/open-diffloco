from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import hashlib

from src.algorithms.shac.centroidal_objective import (
    centroidal_window_objective,
    load_support_aware_impulse_target,
    pseudo_huber,
    support_aware_impulse_objective,
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


def test_support_aware_impulse_matches_phase_target_after_gravity() -> None:
    target_by_phase = (
        jnp.zeros((8, 6), dtype=jnp.float64)
        .at[1]
        .set(jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    )
    target_phase_valid = jnp.zeros(8, dtype=bool).at[1].set(True)
    gravity_impulse = jnp.asarray([0.0, 0.0, -2.0])
    actual = (
        jnp.zeros((5, 6), dtype=jnp.float64)
        .at[4]
        .set(target_by_phase[1].at[:3].add(gravity_impulse))
    )
    quaternions = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64), (5, 1))

    result = support_aware_impulse_objective(
        actual,
        quaternions,
        jnp.asarray([1, 2, 3, 4]),
        target_by_phase,
        target_phase_valid,
        done=jnp.zeros(4, dtype=bool),
        active=jnp.ones(4, dtype=bool),
        gravity_impulse=gravity_impulse,
        component_scales=jnp.ones(6, dtype=jnp.float64),
        window=4,
        reference_stride=1,
        delta=0.1,
    )

    assert result.valid.tolist() == [True]
    assert int(result.valid_count) == 1
    np.testing.assert_allclose(result.error, np.zeros((1, 6)), atol=1e-12)
    assert float(result.loss) == 0.0


def test_support_aware_impulse_excludes_uncovered_or_discontinuous_phase() -> None:
    actual = jnp.zeros((6, 6), dtype=jnp.float64)
    quaternions = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64), (6, 1))
    target_by_phase = jnp.zeros((8, 6), dtype=jnp.float64)
    target_phase_valid = jnp.zeros(8, dtype=bool).at[1].set(True)

    result = support_aware_impulse_objective(
        actual,
        quaternions,
        jnp.asarray([1, 2, 4, 5, 6]),
        target_by_phase,
        target_phase_valid,
        done=jnp.zeros(5, dtype=bool),
        active=jnp.ones(5, dtype=bool),
        gravity_impulse=jnp.zeros(3, dtype=jnp.float64),
        component_scales=jnp.ones(6, dtype=jnp.float64),
        window=4,
        reference_stride=1,
        delta=0.1,
    )

    assert result.valid.tolist() == [False, False]
    assert int(result.valid_count) == 0
    assert float(result.loss) == 0.0


def test_support_aware_target_loader_binds_hash_and_dense_phase_table(
    tmp_path,
) -> None:
    path = tmp_path / "target.npz"
    starts = np.arange(1, 126, dtype=np.int64)
    primary = np.arange(125 * 6, dtype=np.float64).reshape(125, 6)
    duplicate = primary + 0.25
    scales = np.asarray([2.0, 2.0, 2.0, 0.6, 0.6, 0.6])
    np.savez_compressed(
        path,
        window_start_transitions=starts,
        window_end_transitions_inclusive=starts + 3,
        component_scales=scales,
        support_projected_full_a=primary,
        support_projected_full_b=duplicate,
        support_projection_feasible_full_a=np.ones(125, dtype=bool),
        support_projection_feasible_full_b=np.ones(125, dtype=bool),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    target, report = load_support_aware_impulse_target(
        path,
        expected_sha256=digest,
        reference_length=272,
        expected_component_scales=scales,
    )

    assert target.phase_valid.shape == (272,)
    assert target.phase_valid[:127].tolist() == ([False] + [True] * 125 + [False])
    np.testing.assert_array_equal(target.primary_by_phase[1], primary[0])
    np.testing.assert_array_equal(target.primary_by_phase[125], primary[-1])
    np.testing.assert_array_equal(target.duplicate_by_phase[1], duplicate[0])
    np.testing.assert_array_equal(target.component_scales, scales)
    assert report["artifact_sha256"] == digest
    assert report["primary_replica"] == "full-a"
    assert report["heldout_replica"] == "full-b"


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
