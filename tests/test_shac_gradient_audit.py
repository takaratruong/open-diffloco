import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.gradient_audit import (
    PHASE_BINS,
    apply_functional_actor_step,
    assert_matching_pytree_leaf_order,
    detached_gaussian_score_loss,
    discounted_return_to_go,
    pytree_leaf_order,
    summarize_per_env_gradient_geometry,
)


def legacy_functional_actor_step(
    actor_apply, params, direction, observations, *, target_rms
):
    """Reproduce the pre-bracketing calibration for bit-exact regressions."""
    baseline_outputs = actor_apply(params, observations)
    _, output_tangent = jax.jvp(
        lambda value: actor_apply(value, observations),
        (params,),
        (direction,),
    )
    direction_rms = jnp.sqrt(jnp.mean(jnp.square(output_tangent)))
    target = jnp.asarray(target_rms, dtype=direction_rms.dtype)
    scale = target / direction_rms
    evaluated = []

    def evaluate(candidate_scale):
        candidate = jax.tree_util.tree_map(
            lambda value, delta: value + candidate_scale * delta,
            params,
            direction,
        )
        action_delta = actor_apply(candidate, observations) - baseline_outputs
        output_rms = jnp.sqrt(jnp.mean(jnp.square(action_delta)))
        evaluated.append((candidate_scale, output_rms))
        return candidate, action_delta, output_rms

    for _ in range(8):
        _, _, output_rms = evaluate(scale)
        scale = scale * target / output_rms
    candidate, action_delta, output_rms = evaluate(scale)
    return candidate, {
        "scale": scale,
        "linearized_rms": jnp.sqrt(jnp.mean(jnp.square(scale * output_tangent))),
        "output_rms": output_rms,
        "max_action_change": jnp.max(jnp.abs(action_delta)),
    }, evaluated


class DiscountedReturnToGoTest(unittest.TestCase):
    def test_accumulates_normal_transitions_to_fragment_end(self):
        returns = discounted_return_to_go(
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([False, False, False]),
            gamma=0.5,
        )

        np.testing.assert_array_equal(returns, jnp.array([2.75, 1.75, 0.75]))

    def test_done_cuts_off_future_episode_rewards(self):
        returns = discounted_return_to_go(
            jnp.array([1.0, 2.0, 100.0, 4.0]),
            jnp.array([False, True, False, False]),
            gamma=0.5,
        )

        np.testing.assert_array_equal(returns, jnp.array([2.0, 1.0, 102.0, 2.0]))

    def test_final_terminal_keeps_its_immediate_reward(self):
        returns = discounted_return_to_go(
            jnp.array([1.0, 7.0]),
            jnp.array([False, True]),
            gamma=0.99,
        )

        np.testing.assert_array_equal(returns, jnp.array([7.93, 6.93]))


class DetachedGaussianScoreLossTest(unittest.TestCase):
    def test_gradient_has_analytic_gaussian_score_sign(self):
        mean = jnp.array([[0.25]])
        action = jnp.array([[0.75]])
        returns = jnp.array([2.0])

        gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                value, action, returns, std=0.5
            )
        )(mean)

        np.testing.assert_array_equal(gradient, jnp.array([[-4.0]]))

    def test_detaching_reparameterized_action_prevents_gradient_cancellation(
        self,
    ):
        mean = jnp.array([[0.25]])
        action = mean + 0.5 * jnp.array([[1.0]])
        returns = jnp.array([2.0])

        detached_gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                value, action, returns, std=0.5
            )
        )(mean)

        def incorrectly_attached_loss(value):
            reparameterized_action = value + 0.5 * jnp.array([[1.0]])
            log_probability = -0.5 * jnp.square(
                (reparameterized_action - value) / 0.5
            )
            return -jnp.sum(returns[:, None] * log_probability)

        attached_gradient = jax.grad(incorrectly_attached_loss)(mean)

        np.testing.assert_array_equal(detached_gradient, jnp.array([[-4.0]]))
        np.testing.assert_array_equal(attached_gradient, jnp.zeros_like(mean))

    def test_return_coefficients_are_detached(self):
        mean = jnp.array([[0.25], [0.75]])
        actions = jnp.array([[0.75], [0.25]])
        returns = jnp.array([2.0, -3.0])

        gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                mean, actions, value, std=0.5
            )
        )(returns)

        np.testing.assert_array_equal(gradient, jnp.zeros_like(returns))

    def test_ratio_one_ppo_clipped_surrogate_gradient_equals_score_loss(self):
        mean = jnp.array([[-0.25], [0.75]])
        actions = jnp.array([[0.25], [0.25]])
        returns = jnp.array([2.0, -3.0])
        std = 0.5

        direct_gradient = jax.grad(
            lambda value: detached_gaussian_score_loss(
                value, actions, returns, std=std
            )
        )(mean)

        stopped_actions = jax.lax.stop_gradient(actions)
        stopped_returns = jax.lax.stop_gradient(returns)

        def log_probability(value):
            squared_normalized_error = jnp.square(
                (stopped_actions - value) / std
            )
            return -0.5 * jnp.sum(
                squared_normalized_error, axis=-1
            )

        old_log_probability = jax.lax.stop_gradient(log_probability(mean))

        def ratio_one_ppo_loss(value):
            ratio = jnp.exp(log_probability(value) - old_log_probability)
            unclipped = ratio * stopped_returns
            clipped = jnp.clip(ratio, 0.8, 1.2) * stopped_returns
            return -jnp.mean(jnp.minimum(unclipped, clipped))

        ratio = jnp.exp(log_probability(mean) - old_log_probability)
        ppo_gradient = jax.grad(ratio_one_ppo_loss)(mean)

        np.testing.assert_array_equal(ratio, jnp.ones_like(ratio))
        np.testing.assert_array_equal(ppo_gradient, direct_gradient)


class GradientGeometryTest(unittest.TestCase):
    def test_sanitizes_a_nonfinite_environment_as_a_whole_before_clipping(self):
        summary = summarize_per_env_gradient_geometry(
            {
                "w": jnp.array([[2.0, 0.0], [jnp.nan, 4.0]]),
                "b": jnp.array([[0.0], [3.0]]),
            },
            max_norm=1.0,
            initial_phases=jnp.array([0, 0]),
        )

        np.testing.assert_allclose(summary.raw_mean["w"], [2.0, 0.0])
        np.testing.assert_allclose(summary.raw_mean["b"], [0.0])
        np.testing.assert_allclose(summary.clipped_mean["w"], [0.5, 0.0])
        np.testing.assert_allclose(summary.clipped_mean["b"], [0.0])
        np.testing.assert_allclose(summary.finite_by_env, [True, False])
        self.assertAlmostEqual(float(summary.finite_fraction), 0.5)
        self.assertAlmostEqual(float(summary.clipping_fraction), 1.0)
        self.assertEqual(int(summary.phase_bins[0].count), 2)
        self.assertEqual(int(summary.phase_bins[0].finite_count), 1)
        self.assertAlmostEqual(float(summary.phase_bins[0].raw_mean_norm), 2.0)
        self.assertAlmostEqual(
            float(summary.phase_bins[0].clipped_mean_norm), 0.5
        )

    def test_clipping_fraction_is_zero_when_no_environment_is_finite(self):
        summary = summarize_per_env_gradient_geometry(
            {"w": jnp.array([[jnp.nan], [jnp.inf]])},
            max_norm=1.0,
            initial_phases=jnp.array([0, 100]),
        )

        self.assertEqual(float(summary.clipping_fraction), 0.0)
        np.testing.assert_array_equal(summary.raw_mean["w"], [0.0])
        np.testing.assert_array_equal(summary.clipped_mean["w"], [0.0])

    def test_reports_raw_and_clipped_population_geometry(self):
        summary = summarize_per_env_gradient_geometry(
            {"w": jnp.array([[3.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])},
            max_norm=1.0,
            initial_phases=jnp.array([4, 104, 204]),
        )

        np.testing.assert_allclose(summary.raw_mean["w"], [1.0, 0.0])
        np.testing.assert_allclose(
            summary.clipped_mean["w"], [1.0 / 3.0, 0.0]
        )
        np.testing.assert_allclose(summary.raw_norm_by_env, [3.0, 1.0, 1.0])
        np.testing.assert_allclose(summary.clipped_norm_by_env, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(float(summary.clipping_fraction), 1.0 / 3.0)
        self.assertAlmostEqual(float(summary.raw_trace_variance), 8.0 / 3.0)
        self.assertAlmostEqual(float(summary.raw_snr), 1.0 / np.sqrt(8.0 / 3.0))
        self.assertAlmostEqual(float(summary.clipped_trace_variance), 8.0 / 9.0)
        self.assertAlmostEqual(
            float(summary.clipped_snr), (1.0 / 3.0) / np.sqrt(8.0 / 9.0)
        )
        self.assertAlmostEqual(float(summary.negative_cosine_fraction), 1.0 / 3.0)
        np.testing.assert_allclose(summary.contribution_to_aggregate_cosines, [1, -1, 1])
        self.assertAlmostEqual(float(summary.raw_vs_clipped_cosine), 1.0)

    def test_groups_contributions_by_the_five_fixed_initial_phase_bins(self):
        summary = summarize_per_env_gradient_geometry(
            {"w": jnp.eye(5)},
            max_norm=1.0,
            initial_phases=jnp.array([0, 100, 200, 300, 400]),
        )

        self.assertEqual(PHASE_BINS, ((0, 100), (100, 200), (200, 300), (300, 400), (400, 500)))
        self.assertEqual(len(summary.phase_bins), 5)
        self.assertEqual([int(item.count) for item in summary.phase_bins], [1] * 5)
        self.assertEqual(
            tuple((item.start, item.stop) for item in summary.phase_bins),
            PHASE_BINS,
        )
        np.testing.assert_allclose(
            [float(item.clipped_mean_norm) for item in summary.phase_bins],
            np.ones(5),
        )
        self.assertTrue(
            all(
                np.isfinite(float(value))
                for item in summary.phase_bins
                for value in (item.raw_snr, item.clipped_snr)
            )
        )

    def test_empty_and_singleton_phase_bin_snr_is_finite(self):
        summary = summarize_per_env_gradient_geometry(
            {"w": jnp.array([[2.0]])},
            max_norm=1.0,
            initial_phases=jnp.array([0]),
        )

        self.assertEqual(int(summary.phase_bins[0].finite_count), 1)
        self.assertTrue(np.isfinite(float(summary.phase_bins[0].raw_snr)))
        self.assertTrue(np.isfinite(float(summary.phase_bins[0].clipped_snr)))
        for empty_bin in summary.phase_bins[1:]:
            self.assertEqual(int(empty_bin.count), 0)
            self.assertEqual(int(empty_bin.finite_count), 0)
            self.assertTrue(np.isfinite(float(empty_bin.raw_snr)))
            self.assertTrue(np.isfinite(float(empty_bin.clipped_snr)))


class PyTreeOrderAndFunctionalScalingTest(unittest.TestCase):
    def test_leaf_order_is_stable_and_mismatches_fail_closed(self):
        first = {"b": (jnp.array([1.0]),), "a": jnp.array([2.0, 3.0])}
        same_structure = {"a": jnp.array([4.0, 5.0]), "b": (jnp.array([6.0]),)}
        different_structure = {"a": (jnp.array([4.0, 5.0]),), "b": jnp.array([6.0])}

        self.assertEqual(pytree_leaf_order(first), ("['a']", "['b'][0]"))
        self.assertEqual(pytree_leaf_order(first), pytree_leaf_order(same_structure))
        assert_matching_pytree_leaf_order(first, same_structure)
        with self.assertRaisesRegex(ValueError, "leaf order"):
            assert_matching_pytree_leaf_order(first, different_structure)

    def test_jvp_scaling_matches_the_frozen_actor_output_rms(self):
        params = {
            "w": jnp.array([[1.0, -2.0], [0.5, 3.0]]),
            "b": jnp.array([0.25, -0.5]),
        }
        observations = jnp.array([[1.0, 2.0], [-3.0, 0.5], [0.25, -1.0]])

        def actor_apply(value, obs):
            return obs @ value["w"] + value["b"]

        first_direction = {
            "w": jnp.array([[3.0, 0.0], [0.0, 0.0]]),
            "b": jnp.array([0.0, 0.0]),
        }
        second_direction = {
            "w": jnp.array([[0.0, 0.0], [0.0, -4.0]]),
            "b": jnp.array([0.0, 2.0]),
        }

        first_params, first_summary = apply_functional_actor_step(
            actor_apply, params, first_direction, observations, target_rms=0.01
        )
        second_params, second_summary = apply_functional_actor_step(
            actor_apply, params, second_direction, observations, target_rms=0.01
        )

        self.assertNotEqual(float(first_summary.scale), float(second_summary.scale))
        self.assertAlmostEqual(float(first_summary.linearized_rms), 0.01, places=7)
        self.assertAlmostEqual(float(second_summary.linearized_rms), 0.01, places=7)
        self.assertAlmostEqual(float(first_summary.output_rms), 0.01, places=7)
        self.assertAlmostEqual(float(second_summary.output_rms), 0.01, places=7)
        self.assertAlmostEqual(
            float(first_summary.max_action_change),
            float(jnp.max(jnp.abs(actor_apply(first_params, observations) - actor_apply(params, observations)))),
            places=7,
        )
        self.assertAlmostEqual(
            float(second_summary.max_action_change),
            float(jnp.max(jnp.abs(actor_apply(second_params, observations) - actor_apply(params, observations)))),
            places=7,
        )

    def test_exact_rms_calibration_handles_nonlinear_actor_output(self):
        params = {"p": jnp.array(1.0, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(jnp.square(value["p"]), (1, 1))

        candidate, summary = apply_functional_actor_step(
            actor_apply,
            params,
            direction,
            jnp.zeros((1, 1)),
            target_rms=0.01,
        )

        exact_delta = actor_apply(candidate, None) - actor_apply(params, None)
        np.testing.assert_allclose(
            jnp.sqrt(jnp.mean(exact_delta**2)), 0.01, rtol=2e-5
        )
        np.testing.assert_allclose(summary.output_rms, 0.01, rtol=2e-5)

    def test_exact_rms_calibration_accepts_float32_parameter_quantization(self):
        params = {"p": jnp.array(4.0, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(value["p"], (1, 1))

        candidate, summary = apply_functional_actor_step(
            actor_apply,
            params,
            direction,
            jnp.zeros((1, 1), dtype=jnp.float32),
            target_rms=0.01,
        )

        exact_delta = actor_apply(candidate, None) - actor_apply(params, None)
        relative_error = jnp.abs(summary.output_rms - 0.01) / 0.01
        np.testing.assert_array_equal(summary.output_rms, jnp.abs(exact_delta[0, 0]))
        self.assertGreater(float(relative_error), 2e-5)
        self.assertLessEqual(float(relative_error), 5e-5)

    def test_bracketed_bisection_selects_passing_global_best_when_final_correction_fails(
        self,
    ):
        params = {"p": jnp.array(0.01, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(jnp.square(value["p"]), (1, 1))

        legacy_candidate, legacy_summary, legacy_evaluated = (
            legacy_functional_actor_step(
                actor_apply,
                params,
                direction,
                jnp.zeros((1, 1), dtype=jnp.float32),
                target_rms=0.01,
            )
        )
        del legacy_candidate
        legacy_relative_error = (
            jnp.abs(legacy_summary["output_rms"] - 0.01) / 0.01
        )
        self.assertEqual(len(legacy_evaluated), 9)
        self.assertGreater(float(legacy_relative_error), 5e-5)

        candidate, summary = apply_functional_actor_step(
            actor_apply,
            params,
            direction,
            jnp.zeros((1, 1), dtype=jnp.float32),
            target_rms=0.01,
        )

        relative_error = jnp.abs(summary.output_rms - 0.01) / 0.01
        self.assertLessEqual(float(relative_error), 5e-5)
        exact_rms = jnp.sqrt(
            jnp.mean(jnp.square(actor_apply(candidate, None) - actor_apply(params, None)))
        )
        np.testing.assert_array_equal(summary.output_rms, exact_rms)

    def test_fails_when_corrections_do_not_form_an_under_over_bracket(self):
        params = {"p": jnp.array(0.0, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(jnp.clip(value["p"], -0.001, 0.001), (1, 1))

        with self.assertRaisesRegex(ValueError, "under/over bracket"):
            apply_functional_actor_step(
                actor_apply,
                params,
                direction,
                jnp.zeros((1, 1), dtype=jnp.float32),
                target_rms=0.01,
            )

    def test_bracketed_calibration_is_deterministic_and_dtype_exact(self):
        params = {"p": jnp.array(0.01, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}
        observations = jnp.zeros((1, 1), dtype=jnp.float32)

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(jnp.square(value["p"]), (1, 1))

        first_candidate, first_summary = apply_functional_actor_step(
            actor_apply, params, direction, observations, target_rms=0.01
        )
        second_candidate, second_summary = apply_functional_actor_step(
            actor_apply, params, direction, observations, target_rms=0.01
        )

        np.testing.assert_array_equal(first_candidate["p"], second_candidate["p"])
        self.assertEqual(first_candidate["p"].dtype, params["p"].dtype)
        self.assertEqual(first_summary.scale.dtype, jnp.dtype(jnp.float32))
        for first, second in zip(first_summary, second_summary, strict=True):
            np.testing.assert_array_equal(first, second)

    def test_passing_legacy_final_point_remains_bit_exact(self):
        params = {"p": jnp.array(1.0, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}
        observations = jnp.zeros((1, 1), dtype=jnp.float32)

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(jnp.square(value["p"]), (1, 1))

        expected_candidate, expected_summary, _ = legacy_functional_actor_step(
            actor_apply,
            params,
            direction,
            observations,
            target_rms=0.01,
        )
        candidate, summary = apply_functional_actor_step(
            actor_apply,
            params,
            direction,
            observations,
            target_rms=0.01,
        )

        np.testing.assert_array_equal(candidate["p"], expected_candidate["p"])
        for field, expected in expected_summary.items():
            np.testing.assert_array_equal(getattr(summary, field), expected)

    def test_earlier_passing_legacy_global_best_is_selected_when_final_fails(self):
        params = {"p": jnp.array(1.9999312162399292, dtype=jnp.float32)}
        direction = {"p": jnp.array(1.0, dtype=jnp.float32)}
        observations = jnp.zeros((1, 1), dtype=jnp.float32)

        def actor_apply(value, observations):
            del observations
            return jnp.reshape(jnp.square(value["p"]), (1, 1))

        _, _, legacy_evaluated = legacy_functional_actor_step(
            actor_apply,
            params,
            direction,
            observations,
            target_rms=0.01,
        )
        target = jnp.asarray(0.01, dtype=jnp.float32)
        relative_errors = [
            jnp.abs(output_rms - target) / target
            for _, output_rms in legacy_evaluated
        ]
        best_index = min(
            range(len(relative_errors)), key=lambda index: float(relative_errors[index])
        )
        self.assertLess(best_index, len(legacy_evaluated) - 1)
        self.assertLessEqual(float(relative_errors[best_index]), 5e-5)
        self.assertGreater(float(relative_errors[-1]), 5e-5)

        candidate, summary = apply_functional_actor_step(
            actor_apply,
            params,
            direction,
            observations,
            target_rms=0.01,
        )
        expected_scale = legacy_evaluated[best_index][0]
        expected_candidate = jax.tree_util.tree_map(
            lambda value, delta: value + expected_scale * delta,
            params,
            direction,
        )

        np.testing.assert_array_equal(summary.scale, expected_scale)
        np.testing.assert_array_equal(candidate["p"], expected_candidate["p"])
        np.testing.assert_array_equal(
            summary.output_rms, legacy_evaluated[best_index][1]
        )

    def test_rejects_nonfinite_actor_unused_direction_leaf(self):
        params = {"used": jnp.array(1.0), "unused": jnp.array(0.0)}
        direction = {"used": jnp.array(1.0), "unused": jnp.array(jnp.nan)}

        with self.assertRaisesRegex(ValueError, "direction.*nonfinite"):
            apply_functional_actor_step(
                lambda value, observations: value["used"] * observations,
                params,
                direction,
                jnp.ones((1, 1)),
            )

    def test_rejects_nonfinite_baseline_actor_output(self):
        with self.assertRaisesRegex(ValueError, "baseline.*nonfinite"):
            apply_functional_actor_step(
                lambda value, observations: jnp.array([jnp.nan]),
                {"p": jnp.array(1.0)},
                {"p": jnp.array(1.0)},
                jnp.ones((1, 1)),
            )

    def test_rejects_nonfinite_candidate_actor_output(self):
        def actor_apply(value, observations):
            del observations
            return jnp.where(value["p"] == 1.0, value["p"], jnp.nan)

        with self.assertRaisesRegex(ValueError, "candidate.*nonfinite"):
            apply_functional_actor_step(
                actor_apply,
                {"p": jnp.array(1.0)},
                {"p": jnp.array(1.0)},
                jnp.ones((1, 1)),
            )

    def test_rejects_nonfinite_exact_action_delta(self):
        largest = jnp.array(jnp.finfo(jnp.float32).max, dtype=jnp.float32)

        def actor_apply(value, observations):
            del observations
            discontinuous = jnp.where(value["p"] == 0.0, largest, -largest)
            return jnp.stack((discontinuous, value["p"]))

        with self.assertRaisesRegex(ValueError, "delta.*nonfinite"):
            apply_functional_actor_step(
                actor_apply,
                {"p": jnp.array(0.0, dtype=jnp.float32)},
                {"p": jnp.array(1.0, dtype=jnp.float32)},
                jnp.ones((1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
