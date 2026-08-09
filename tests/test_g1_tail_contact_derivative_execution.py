import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.g1_gradient_audit import FirstActionObjective
from src.algorithms.shac.g1_tail_contact_derivative_execution import evaluate_case


class FakeFirstActionObjective:
    def __init__(self, *, change_done=False, change_support=False):
        self.action = jnp.linspace(-0.5, 0.5, 29, dtype=jnp.float64)
        self.objective_calls = 0
        self.rollout_calls = 0
        self.change_done = change_done
        self.change_support = change_support
        dones = jnp.zeros((48,), dtype=jnp.bool_)
        actions = jnp.tile(self.action[None, :], (48, 1))
        self.nominal_trajectory = SimpleNamespace(dones=dones, actions=actions)

    def objective(self, action):
        self.objective_calls += 1
        return jnp.sum(jnp.square(action))

    def rollout(self, action):
        self.rollout_calls += 1
        dones = self.nominal_trajectory.dones
        actions = jnp.tile(action[None, :], (48, 1))
        if self.change_done and bool(action[0] > self.action[0]):
            dones = dones.at[0].set(True)
        if self.change_support and bool(action[0] > self.action[0]):
            actions = actions.at[1, 0].set(jnp.nan)
        return SimpleNamespace(dones=dones, actions=actions), object()

    def build(self):
        diagnostic = FirstActionObjective(
            nominal_trajectory=self.nominal_trajectory,
            nominal_final_state=object(),
            nominal_first_action=self.action,
            nominal_objective=jnp.sum(jnp.square(self.action)),
            rollout=self.rollout,
            objective=self.objective,
        )
        self.objective_calls = 0
        self.rollout_calls = 0
        return diagnostic


class EvaluateTailContactDerivativeCaseTest(unittest.TestCase):
    def test_executes_one_reverse_and_three_complete_forward_sweeps(self):
        fake = FakeFirstActionObjective()

        result = evaluate_case(
            fake.build(), shard_seed=2, phase_bin=3, direction_seed=12014
        )

        self.assertEqual(fake.objective_calls, 1 + 3 * 29 + 2)
        self.assertEqual(fake.rollout_calls, 2)
        self.assertEqual(result.shard_seed, 2)
        self.assertEqual(result.phase_bin, 3)
        self.assertEqual(result.direction_seed, 12014)
        np.testing.assert_array_equal(result.nominal_action, fake.action)
        self.assertEqual(result.reverse_gradient.shape, (29,))
        self.assertEqual(result.forward_gradients.shape, (3, 29))
        self.assertEqual(result.forward_primals.shape, (3,))
        for gradient in result.forward_gradients:
            np.testing.assert_array_equal(gradient, 2.0 * fake.action)
        np.testing.assert_array_equal(result.reverse_gradient, 2.0 * fake.action)
        self.assertTrue(result.comparison.forward_valid)
        self.assertTrue(result.comparison.reverse_parity_valid)

    def test_uses_frozen_jax_direction_and_centered_point_zero_zero_one_probes(self):
        fake = FakeFirstActionObjective()

        result = evaluate_case(
            fake.build(), shard_seed=0, phase_bin=0, direction_seed=12001
        )
        expected_direction = jax.random.normal(
            jax.random.PRNGKey(12001), (29,), dtype=jnp.float64
        )
        expected_direction = expected_direction / jnp.linalg.norm(expected_direction)

        np.testing.assert_array_equal(result.direction, expected_direction)
        np.testing.assert_array_equal(
            result.positive_action,
            fake.action + jnp.asarray(0.001) * expected_direction,
        )
        np.testing.assert_array_equal(
            result.negative_action,
            fake.action - jnp.asarray(0.001) * expected_direction,
        )
        expected_fd = (
            jnp.sum(jnp.square(result.positive_action))
            - jnp.sum(jnp.square(result.negative_action))
        ) / 0.002
        np.testing.assert_array_equal(
            result.directional_finite_difference, expected_fd
        )
        self.assertEqual(result.finite_difference_epsilon, 0.001)

    def test_returns_complete_finite_timing_and_preservation_receipts(self):
        result = evaluate_case(
            FakeFirstActionObjective().build(),
            shard_seed=3,
            phase_bin=4,
            direction_seed=12020,
        )

        for array in (
            result.nominal_action,
            result.reverse_gradient,
            result.forward_gradients,
            result.forward_primals,
            result.direction,
            result.positive_action,
            result.negative_action,
            result.probe_objectives,
        ):
            self.assertTrue(np.isfinite(np.asarray(array)).all())
        self.assertEqual(result.forward_durations_seconds.shape, (3,))
        self.assertEqual(result.probe_durations_seconds.shape, (2,))
        self.assertGreaterEqual(result.reverse_duration_seconds, 0.0)
        self.assertTrue(np.all(result.forward_durations_seconds >= 0.0))
        self.assertTrue(np.all(result.probe_durations_seconds >= 0.0))
        self.assertTrue(result.positive_done_exact)
        self.assertTrue(result.negative_done_exact)
        self.assertTrue(result.positive_support_exact)
        self.assertTrue(result.negative_support_exact)
        self.assertTrue(result.probes_preserve_done_and_support)

    def test_done_or_support_change_is_forward_invalid_not_execution_error(self):
        for change in ("done", "support"):
            fake = FakeFirstActionObjective(
                change_done=change == "done",
                change_support=change == "support",
            )
            with self.subTest(change=change):
                result = evaluate_case(
                    fake.build(),
                    shard_seed=1,
                    phase_bin=2,
                    direction_seed=12008,
                )
                self.assertFalse(result.probes_preserve_done_and_support)
                self.assertFalse(result.comparison.forward_valid)
                self.assertTrue(result.execution_valid)
                self.assertEqual(
                    result.case_outcome, "forward-contact-derivative-invalid"
                )

    def test_rejects_malformed_nominal_contract_before_derivatives(self):
        fake = FakeFirstActionObjective()
        diagnostic = fake.build()._replace(
            nominal_first_action=jnp.zeros((28,), dtype=jnp.float64)
        )

        with self.assertRaisesRegex(ValueError, "nominal action.*29"):
            evaluate_case(
                diagnostic, shard_seed=0, phase_bin=0, direction_seed=12001
            )
        self.assertEqual(fake.objective_calls, 0)
        self.assertEqual(fake.rollout_calls, 0)

    def test_rejects_noncanonical_direction_seed_before_derivatives(self):
        fake = FakeFirstActionObjective()

        with self.assertRaisesRegex(ValueError, "canonical direction seed"):
            evaluate_case(
                fake.build(), shard_seed=2, phase_bin=3, direction_seed=12015
            )

        self.assertEqual(fake.objective_calls, 0)
        self.assertEqual(fake.rollout_calls, 0)


if __name__ == "__main__":
    unittest.main()
