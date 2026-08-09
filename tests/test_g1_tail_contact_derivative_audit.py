import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.g1_tail_contact_derivative_audit import (
    canonical_forward_scalar_gradient,
    classify_derivative_cases,
    compare_derivative_case,
    seeded_unbounded_unit_direction,
    select_rank_zero_fragments,
)


def shard_inputs():
    phases = (np.arange(64, dtype=np.int32) % 5) * 100
    losses = {}
    initial_phases = {}
    for seed in range(4):
        values = np.arange(64, dtype=np.float64) + 1000.0 * seed
        values[55] = values[60] = 10_000.0 + seed
        losses[seed] = values
        initial_phases[seed] = phases.copy()
    return losses, initial_phases


def valid_case(seed, phase_bin, *, reverse_gradient=None, directional_fd=None):
    forward = np.linspace(1.0, 2.0, 29, dtype=np.float64)
    direction = np.zeros(29, dtype=np.float64)
    direction[0] = 1.0
    if reverse_gradient is None:
        reverse_gradient = forward.copy()
    if directional_fd is None:
        directional_fd = float(forward[0])
    return compare_derivative_case(
        shard_seed=seed,
        phase_bin=phase_bin,
        forward_gradients=np.stack((forward, forward, forward)),
        reverse_gradient=reverse_gradient,
        finite_difference_direction=direction,
        directional_finite_difference=directional_fd,
    )


class RankZeroFragmentSelectionTest(unittest.TestCase):
    def test_selects_rank_zero_per_shard_and_bin_in_frozen_order(self):
        losses, phases = shard_inputs()

        selected = select_rank_zero_fragments(losses, phases)

        self.assertEqual(len(selected), 20)
        self.assertEqual(
            [(item.shard_seed, item.phase_bin) for item in selected],
            [(seed, phase_bin) for seed in range(4) for phase_bin in range(5)],
        )
        expected_indices = (55, 61, 62, 63, 59)
        for seed in range(4):
            shard = selected[seed * 5 : (seed + 1) * 5]
            self.assertEqual(
                tuple(item.environment_index for item in shard), expected_indices
            )
            self.assertEqual(shard[0].environment_index, 55)
            self.assertEqual(shard[0].initial_phase, 0)

    def test_rejects_malformed_or_incomplete_shards(self):
        losses, phases = shard_inputs()
        invalid = []
        invalid.append(({seed: losses[seed] for seed in range(3)}, phases, "seeds"))
        invalid.append(
            (
                {**losses, 4: losses[0]},
                {**phases, 4: phases[0]},
                "seeds",
            )
        )
        invalid.append(({**losses, 0: losses[0][:-1]}, phases, "shape"))
        invalid.append(({**losses, 0: losses[0].reshape(8, 8)}, phases, "shape"))
        invalid.append(
            ({**losses, 0: losses[0].copy()}, phases, "finite")
        )
        invalid[-1][0][0][3] = np.nan
        invalid.append(
            (losses, {**phases, 0: phases[0].astype(np.float64)}, "integer")
        )
        out_of_range = phases[0].copy()
        out_of_range[3] = 500
        invalid.append((losses, {**phases, 0: out_of_range}, "range"))
        missing_bin = phases[0].copy()
        missing_bin[missing_bin == 400] = 300
        invalid.append((losses, {**phases, 0: missing_bin}, "nonempty"))

        for malformed_losses, malformed_phases, message in invalid:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                select_rank_zero_fragments(malformed_losses, malformed_phases)


class CanonicalForwardGradientTest(unittest.TestCase):
    def test_assembles_all_29_scalar_jvps_in_canonical_order(self):
        action = jnp.linspace(-1.0, 1.0, 29, dtype=jnp.float64)
        tangents = []

        def directional(current_action, tangent):
            tangents.append(np.asarray(tangent))
            return jnp.sum(current_action**2), jnp.vdot(2.0 * current_action, tangent)

        primal, gradient = canonical_forward_scalar_gradient(directional, action)

        np.testing.assert_array_equal(primal, jnp.sum(action**2))
        np.testing.assert_array_equal(gradient, 2.0 * action)
        np.testing.assert_array_equal(np.stack(tangents), np.eye(29))

    def test_rejects_disagreeing_directional_primals(self):
        calls = []

        def directional(action, tangent):
            calls.append(None)
            return jnp.asarray(float(len(calls))), jnp.vdot(action, tangent)

        with self.assertRaisesRegex(ValueError, "shared primal"):
            canonical_forward_scalar_gradient(
                directional, jnp.ones(29, dtype=jnp.float32)
            )


class UnboundedDirectionTest(unittest.TestCase):
    def test_is_seeded_repeatable_unit_norm_and_dtype_exact(self):
        action = np.linspace(-20.0, 30.0, 29, dtype=np.float64)

        first = seeded_unbounded_unit_direction(action, seed=12001)
        second = seeded_unbounded_unit_direction(action, seed=12001)
        different = seeded_unbounded_unit_direction(action, seed=12002)
        expected = jax.random.normal(
            jax.random.PRNGKey(12001), (29,), dtype=jnp.float64
        )
        expected = expected / jnp.linalg.norm(expected)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.asarray(expected))
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(first.dtype, action.dtype)
        self.assertTrue(np.isfinite(first).all())
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(action + 0.001 * first)))
        self.assertTrue(np.all(np.isfinite(action - 0.001 * first)))

    def test_casts_the_float64_jax_direction_only_after_normalization(self):
        action = np.ones(29, dtype=np.float32)
        actual = seeded_unbounded_unit_direction(action, seed=17)
        expected = jax.random.normal(
            jax.random.PRNGKey(17), (29,), dtype=jnp.float64
        )
        expected = np.asarray(expected / jnp.linalg.norm(expected), dtype=np.float32)

        self.assertEqual(actual.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_nonfinite_wrong_shape_or_nonfloating_actions(self):
        invalid = (
            np.ones(28, dtype=np.float32),
            np.full(29, np.nan, dtype=np.float32),
            np.ones(29, dtype=np.int32),
        )
        for action in invalid:
            with (
                self.subTest(dtype=action.dtype, shape=action.shape),
                self.assertRaises((TypeError, ValueError)),
            ):
                seeded_unbounded_unit_direction(action, seed=1)


class DerivativeComparisonTest(unittest.TestCase):
    def test_applies_forward_repeat_fd_and_reverse_parity_gates(self):
        comparison = valid_case(0, 0)

        self.assertTrue(comparison.forward_finite)
        self.assertEqual(comparison.forward_repeat_maximum_absolute_error, 0.0)
        self.assertTrue(comparison.forward_repeat_valid)
        self.assertTrue(comparison.forward_fd_valid)
        self.assertTrue(comparison.forward_valid)
        self.assertTrue(comparison.reverse_finite)
        self.assertTrue(comparison.reverse_parity_valid)

    def test_repeat_gate_uses_maximum_coordinate_drift_across_three_sweeps(self):
        gradient = np.zeros(29, dtype=np.float64)
        gradient[0] = 1.0
        direction = np.zeros(29, dtype=np.float64)
        direction[0] = 1.0
        within_tolerance = np.stack((gradient, gradient.copy(), gradient.copy()))
        within_tolerance[1, 1] += 0.5e-6
        within_tolerance[2, 1] -= 0.5e-6
        beyond_tolerance = np.stack((gradient, gradient.copy(), gradient.copy()))
        beyond_tolerance[1, 1] += 0.55e-6
        beyond_tolerance[2, 1] -= 0.55e-6

        accepted = compare_derivative_case(
            shard_seed=0,
            phase_bin=0,
            forward_gradients=within_tolerance,
            reverse_gradient=gradient,
            finite_difference_direction=direction,
            directional_finite_difference=float(gradient[0]),
        )
        rejected = compare_derivative_case(
            shard_seed=0,
            phase_bin=0,
            forward_gradients=beyond_tolerance,
            reverse_gradient=gradient,
            finite_difference_direction=direction,
            directional_finite_difference=float(gradient[0]),
        )

        self.assertAlmostEqual(
            accepted.forward_repeat_maximum_absolute_error, 1e-6
        )
        self.assertTrue(accepted.forward_repeat_valid)
        self.assertTrue(accepted.forward_valid)
        self.assertAlmostEqual(
            rejected.forward_repeat_maximum_absolute_error, 1.1e-6
        )
        self.assertFalse(rejected.forward_repeat_valid)
        self.assertFalse(rejected.forward_valid)

    def test_small_finite_difference_uses_absolute_error_gate(self):
        direction = np.zeros(29, dtype=np.float64)
        direction[0] = 1.0
        passing = np.zeros(29, dtype=np.float64)
        passing[0] = 0.5e-6
        failing = passing.copy()
        failing[0] = 2.0e-6

        accepted = compare_derivative_case(
            shard_seed=0,
            phase_bin=0,
            forward_gradients=np.stack((passing, passing, passing)),
            reverse_gradient=passing,
            finite_difference_direction=direction,
            directional_finite_difference=0.0,
        )
        rejected = compare_derivative_case(
            shard_seed=0,
            phase_bin=0,
            forward_gradients=np.stack((failing, failing, failing)),
            reverse_gradient=failing,
            finite_difference_direction=direction,
            directional_finite_difference=0.0,
        )

        self.assertTrue(accepted.forward_fd_valid)
        self.assertFalse(rejected.forward_fd_valid)

    def test_requires_exactly_three_complete_forward_sweeps(self):
        gradient = np.ones(29, dtype=np.float64)
        direction = np.zeros(29, dtype=np.float64)
        direction[0] = 1.0

        for repeat_count in (2, 4):
            with (
                self.subTest(repeat_count=repeat_count),
                self.assertRaisesRegex(ValueError, r"shape \(3, 29\)"),
            ):
                compare_derivative_case(
                    shard_seed=0,
                    phase_bin=0,
                    forward_gradients=np.stack((gradient,) * repeat_count),
                    reverse_gradient=gradient,
                    finite_difference_direction=direction,
                    directional_finite_difference=1.0,
                )

    def test_classifies_complete_case_matrix_with_fail_closed_precedence(self):
        cases = [
            valid_case(seed, phase_bin)
            for seed in range(4)
            for phase_bin in range(5)
        ]
        reverse_invalid = np.linspace(1.0, 2.0, 29, dtype=np.float64)
        reverse_invalid[1::2] *= -1.0
        rescued = list(cases)
        rescued[7] = valid_case(1, 2, reverse_gradient=reverse_invalid)
        forward_invalid = list(rescued)
        forward_invalid[3] = valid_case(0, 3, directional_fd=2.0)

        self.assertEqual(
            classify_derivative_cases(cases), "reverse-and-forward-valid"
        )
        self.assertEqual(
            classify_derivative_cases(rescued),
            "forward-rescues-reverse-tail-adjoint",
        )
        self.assertEqual(
            classify_derivative_cases(forward_invalid),
            "forward-contact-derivative-invalid",
        )
        self.assertEqual(
            classify_derivative_cases(cases[:-1]), "invalid-execution"
        )
        self.assertEqual(
            classify_derivative_cases(cases, execution_valid=False),
            "invalid-execution",
        )


if __name__ == "__main__":
    unittest.main()
