import unittest
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from src.envs.g1_tracking.action_shooting import (
    ShootingConfig,
    canonical_forward_gradient,
    capture_actor_window_without_reset,
    directional_fd_audit,
    project_trust_box,
    rollout_actions_without_reset,
    run_projected_armijo,
    shooting_objective,
    support_switch_count,
    validate_action_sequence,
)


class G1ActionShootingTest(unittest.TestCase):
    def test_registered_config_has_fixed_contract(self):
        config = ShootingConfig()

        self.assertEqual(config.start_phase, 105)
        self.assertEqual(config.horizon, 12)
        self.assertEqual(config.action_dim, 29)
        self.assertEqual(config.iterations, 3)
        self.assertEqual(config.trust_radius, 0.02)
        self.assertEqual(config.line_search_alphas, (1.0, 0.5, 0.25, 0.125))

        actions = np.zeros((12, 29), dtype=np.float64)
        np.testing.assert_array_equal(
            validate_action_sequence(actions, config), actions
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_action_sequence(np.zeros((11, 29)), config)
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_action_sequence(actions, config)

    def test_support_switch_counts_left_right_set_changes(self):
        support = np.asarray(
            [
                [True, True],
                [True, True],
                [False, True],
                [False, True],
                [True, False],
            ],
            dtype=bool,
        )

        self.assertEqual(support_switch_count(support), 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            support_switch_count(np.ones((4, 3), dtype=bool))

    def test_trust_projection_preserves_unbounded_actor_center(self):
        nominal = np.asarray([[2.0, -1.6]], dtype=np.float64)
        proposed = np.asarray([[4.0, -4.0]], dtype=np.float64)

        projected = project_trust_box(proposed, nominal, radius=0.02)

        np.testing.assert_allclose(projected, [[2.02, -1.62]])

    def test_projected_armijo_uses_fixed_order_for_exactly_three_iterations(self):
        config = ShootingConfig()
        initial = np.full((12, 29), 1.5, dtype=np.float64)
        evaluations = []

        def objective_and_gate(actions):
            value = float(np.mean(np.square(actions - 1.48)))
            evaluations.append((np.array(actions, copy=True), value))
            return value, True

        def gradient(actions):
            return 2.0 * (actions - 1.48) / actions.size

        selected, trace = run_projected_armijo(
            initial,
            objective_and_gate=objective_and_gate,
            gradient_fn=gradient,
            config=config,
        )

        self.assertEqual(len(trace), 3)
        self.assertTrue(all(row.accepted for row in trace))
        self.assertTrue(all(row.alpha == 1.0 for row in trace))
        self.assertLess(float(np.mean(np.square(selected - 1.48))), 1e-12)
        self.assertLessEqual(np.max(np.abs(selected - initial)), 0.02 + 1e-12)
        self.assertGreaterEqual(len(evaluations), 4)

    def test_projected_armijo_aborts_on_first_infeasible_contact_trial(self):
        config = ShootingConfig()
        initial = np.zeros((12, 29), dtype=np.float64)
        candidate_calls = []

        def objective_and_gate(actions):
            displacement = float(np.max(np.abs(actions)))
            candidate_calls.append(displacement)
            return -displacement, displacement <= 0.01 + 1e-12

        def gradient(actions):
            del actions
            return -np.ones((12, 29), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "Armijo candidate.*infeasible"):
            run_projected_armijo(
                initial,
                objective_and_gate=objective_and_gate,
                gradient_fn=gradient,
                config=config,
            )

        self.assertEqual(candidate_calls, [0.0, 0.02])

    def test_no_reset_rollout_carries_full_data_cache_across_all_steps(self):
        class FakeData(NamedTuple):
            qpos: jnp.ndarray
            qvel: jnp.ndarray
            cache: jnp.ndarray

        class FakeEnv:
            reward_scale = 1.0
            termination_margin_weight = 0.0
            soft_joint_lower = jnp.full((29,), -10.0)
            soft_joint_upper = jnp.full((29,), 10.0)

            def advance_physics(self, data, action):
                next_cache = data.cache + 1.0
                next_qpos = data.qpos.at[0].add(next_cache)
                next_data = FakeData(next_qpos, data.qvel, next_cache)
                return next_data, action, jnp.zeros((29,))

            def _body_state(self, data):
                body_pos = jnp.asarray([[0.0, 0.0, data.qpos[0]]])
                body_quat = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
                velocity = jnp.zeros((1, 3))
                return body_pos, body_quat, velocity, velocity

            def _tracking_reward_from_body_state(
                self, info, body_pos, body_quat, body_lin_vel, body_ang_vel
            ):
                del info, body_quat, body_lin_vel, body_ang_vel
                return body_pos[0, 2], {}

            def _termination(self, data, info, body_pos, body_quat):
                del data, info, body_pos, body_quat
                return jnp.asarray(0.0), jnp.asarray(0.0)

            def termination_errors(self, *, phase, body_pos, body_quat):
                del phase, body_pos, body_quat
                return {
                    "anchor_z_error": jnp.asarray(0.0),
                    "anchor_xy_error": jnp.asarray(0.0),
                    "gravity_z_error": jnp.asarray(0.0),
                    "distal_z_error": jnp.asarray(0.0),
                }

        initial_data = FakeData(
            jnp.zeros((36,), dtype=jnp.float64),
            jnp.zeros((35,), dtype=jnp.float64),
            jnp.asarray(0.0),
        )
        actions = jnp.zeros((12, 29), dtype=jnp.float64)

        rollout = rollout_actions_without_reset(
            FakeEnv(),
            initial_data,
            start_phase=105,
            initial_previous_action=jnp.zeros((29,), dtype=jnp.float64),
            actions=actions,
        )

        np.testing.assert_allclose(
            np.asarray(rollout.qpos[:, 0]),
            np.cumsum(np.arange(1.0, 13.0)),
        )
        np.testing.assert_array_equal(rollout.phases, np.arange(106, 118))
        self.assertEqual(float(rollout.final_data.cache), 12.0)
        self.assertEqual(np.asarray(rollout.terminal).sum(), 0)

    def test_actor_window_initial_state_is_full_phase_zero_carry(self):
        class FakeData(NamedTuple):
            qpos: jnp.ndarray
            qvel: jnp.ndarray
            cache: jnp.ndarray

        class FakeEnv:
            def advance_physics(self, data, action):
                del action
                cache = data.cache + 1.0
                return (
                    FakeData(
                        data.qpos.at[0].set(cache),
                        data.qvel,
                        cache,
                    ),
                    jnp.full((29,), cache),
                    jnp.zeros((29,)),
                )

            def _body_state(self, data):
                del data
                return (
                    jnp.zeros((1, 3)),
                    jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
                    jnp.zeros((1, 3)),
                    jnp.zeros((1, 3)),
                )

            def _termination(self, data, info, body_pos, body_quat):
                del data, info, body_pos, body_quat
                return jnp.asarray(0.0), jnp.asarray(0.0)

            def _get_actor_obs(self, data, info):
                return jnp.asarray([data.cache, info["phase"]])

            def reset(self, *args):
                del args
                raise AssertionError("carried actor window must never reset")

        phase_zero_state = SimpleNamespace(
            data=FakeData(
                jnp.zeros((36,), dtype=jnp.float64),
                jnp.zeros((35,), dtype=jnp.float64),
                jnp.asarray(0.0),
            ),
            info={
                "phase": jnp.asarray(0, dtype=jnp.int32),
                "last_act": jnp.zeros((29,), dtype=jnp.float64),
                "actor_obs_history": jnp.asarray([[0.0, 0.0]]),
            },
        )

        window = capture_actor_window_without_reset(
            FakeEnv(),
            phase_zero_state,
            lambda obs: jnp.full((29,), obs[0]),
            start_phase=2,
            horizon=2,
        )

        self.assertEqual(float(window.initial_data.cache), 2.0)
        self.assertEqual(float(window.initial_previous_action[0]), 2.0)
        np.testing.assert_array_equal(window.actions[:, 0], [2.0, 3.0])
        np.testing.assert_array_equal(window.phases, [3, 4])
        self.assertEqual(float(window.final_data.cache), 4.0)
        self.assertEqual(np.asarray(window.prefix_terminal).sum(), 0)
        self.assertEqual(np.asarray(window.terminal).sum(), 0)

    def test_shooting_objective_is_exact_reward_plus_declared_regularizer(self):
        rollout = SimpleNamespace(rewards=jnp.asarray([0.08, 0.10]))
        nominal = jnp.zeros((2, 2), dtype=jnp.float64)
        actions = jnp.ones((2, 2), dtype=jnp.float64)

        value = shooting_objective(
            rollout,
            actions,
            nominal,
            action_deviation_weight=1e-3,
        )

        self.assertAlmostEqual(float(value), -0.09 + 0.001, places=12)

    def test_canonical_forward_gradient_assembles_all_348_coordinates_in_order(self):
        actions = jnp.linspace(-0.2, 0.3, 12 * 29).reshape(12, 29)
        weights = jnp.arange(1.0, 12 * 29 + 1.0).reshape(12, 29)

        report = canonical_forward_gradient(
            lambda value: jnp.sum(value * weights),
            actions,
            identity_tolerance=1e-8,
        )

        self.assertEqual(report.scalar_jvps, 348)
        np.testing.assert_array_equal(report.gradient, weights)
        self.assertEqual(report.maximum_primal_error, 0.0)
        repeated = canonical_forward_gradient(
            lambda value: jnp.sum(value * weights),
            actions,
            identity_tolerance=1e-8,
        )
        np.testing.assert_array_equal(report.gradient, repeated.gradient)

    def test_canonical_forward_gradient_rejects_changed_or_nonfinite_jvp(self):
        actions = jnp.zeros((12, 29), dtype=jnp.float64)

        def changed_primal(value, direction):
            return jnp.sum(value) + 1e-3, jnp.sum(direction)

        with self.assertRaisesRegex(ValueError, "primal identity"):
            canonical_forward_gradient(
                jnp.sum,
                actions,
                identity_tolerance=1e-8,
                directional_jvp=changed_primal,
            )

        def nonfinite_tangent(value, direction):
            return jnp.sum(value), jnp.asarray(jnp.nan)

        with self.assertRaisesRegex(ValueError, "finite"):
            canonical_forward_gradient(
                jnp.sum,
                actions,
                identity_tolerance=1e-8,
                directional_jvp=nonfinite_tangent,
            )

    def test_canonical_forward_gradient_checks_every_physical_primal_row(self):
        actions = jnp.zeros((12, 29), dtype=jnp.float64)

        def physical(value):
            return value[:, :2]

        def changed_physical_primal(value, direction):
            return (
                (jnp.sum(value), physical(value) + 1e-3),
                (jnp.sum(direction), jnp.zeros_like(physical(value))),
            )

        with self.assertRaisesRegex(ValueError, "physical primal identity"):
            canonical_forward_gradient(
                jnp.sum,
                actions,
                physical_fn=physical,
                identity_tolerance=1e-8,
                directional_jvp=changed_physical_primal,
            )

    def test_directional_fd_audit_matches_forward_gradient(self):
        actions = jnp.linspace(-0.3, 0.2, 12 * 29).reshape(12, 29)
        objective = lambda value: jnp.sum(jnp.square(value))
        gradient = 2.0 * actions

        audit = directional_fd_audit(
            objective,
            actions,
            gradient,
            epsilon=1e-3,
            seed=20260808,
        )

        self.assertEqual(audit.direction.shape, (12, 29))
        self.assertAlmostEqual(float(np.linalg.norm(audit.direction)), 1.0)
        self.assertLess(audit.relative_error, 1e-9)

    def test_directional_fd_audit_checks_both_physical_support_probes(self):
        actions = jnp.zeros((12, 29), dtype=jnp.float64)
        probes = []

        def support_gate(probe):
            probes.append(np.asarray(probe))
            return len(probes) == 1

        audit = directional_fd_audit(
            lambda value: jnp.sum(jnp.square(value)),
            actions,
            np.zeros((12, 29), dtype=np.float64),
            epsilon=1e-3,
            seed=20260808,
            support_gate=support_gate,
        )

        self.assertEqual(len(probes), 2)
        self.assertTrue(audit.positive_support_safe)
        self.assertFalse(audit.negative_support_safe)
        np.testing.assert_allclose(probes[0], -probes[1])


if __name__ == "__main__":
    unittest.main()
