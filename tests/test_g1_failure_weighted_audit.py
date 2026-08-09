import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np


class PhasePreservingLowerTailWeightsTest(unittest.TestCase):
    def test_preserves_phase_mass_and_selects_stable_worst_quartile(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            phase_preserving_lower_tail_weights,
        )

        phases = jnp.asarray(
            [
                *range(4),
                *range(100, 117),
                *range(200, 219),
                *range(300, 308),
                *range(400, 416),
            ],
            dtype=jnp.int32,
        )
        losses = jnp.asarray(np.arange(64, dtype=np.float64) % 7)

        receipt = phase_preserving_lower_tail_weights(losses, phases)

        np.testing.assert_array_equal(receipt.bin_counts, [4, 17, 19, 8, 16])
        np.testing.assert_array_equal(receipt.tail_counts, [1, 4, 4, 2, 4])
        self.assertEqual(int(jnp.sum(receipt.selected)), 15)
        self.assertAlmostEqual(float(jnp.sum(receipt.weights)), 1.0, places=12)
        for phase_bin, expected_mass in enumerate(np.array([4, 17, 19, 8, 16]) / 64):
            actual_mass = jnp.sum(
                jnp.where(receipt.phase_bins == phase_bin, receipt.weights, 0.0)
            )
            self.assertAlmostEqual(float(actual_mass), expected_mass, places=12)

        losses_host = np.asarray(losses)
        phases_host = np.asarray(phases)
        selected_host = np.asarray(receipt.selected)
        for phase_bin, tail_count in enumerate([1, 4, 4, 2, 4]):
            indices = np.flatnonzero(phases_host // 100 == phase_bin)
            expected = sorted(indices, key=lambda index: (-losses_host[index], index))[
                :tail_count
            ]
            np.testing.assert_array_equal(
                np.flatnonzero(selected_host & (phases_host // 100 == phase_bin)),
                sorted(expected),
            )

    def test_rejects_malformed_losses_and_phases(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            phase_preserving_lower_tail_weights,
        )

        valid_losses = jnp.arange(64, dtype=jnp.float64)
        valid_phases = jnp.arange(64, dtype=jnp.int32) * 499 // 63
        invalid_cases = (
            (valid_losses[:-1], valid_phases, "shape"),
            (valid_losses.at[3].set(jnp.nan), valid_phases, "finite"),
            (valid_losses, valid_phases.astype(jnp.float64), "integer"),
            (valid_losses, valid_phases.at[4].set(500), "range"),
        )
        for losses, phases, message in invalid_cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                phase_preserving_lower_tail_weights(losses, phases)

    def test_weights_are_detached_from_losses(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            phase_preserving_lower_tail_weights,
        )

        phases = jnp.arange(64, dtype=jnp.int32) * 499 // 63

        def weighted_probe(losses):
            weights = phase_preserving_lower_tail_weights(losses, phases).weights
            return jnp.vdot(weights, jnp.arange(64, dtype=jnp.float64))

        gradient = jax.grad(weighted_probe)(jnp.arange(64, dtype=jnp.float64))

        np.testing.assert_array_equal(gradient, np.zeros(64))


class WeightedGradientMeanTest(unittest.TestCase):
    def test_applies_weights_to_already_clipped_per_environment_gradients(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            weighted_mean_environment_gradients,
        )

        weights = jnp.arange(1, 65, dtype=jnp.float64)
        weights = weights / jnp.sum(weights)
        gradients = {
            "kernel": jnp.arange(128, dtype=jnp.float64).reshape(64, 2),
            "bias": jnp.ones((64, 1), dtype=jnp.float64),
        }

        result = weighted_mean_environment_gradients(gradients, weights)

        np.testing.assert_allclose(
            result["kernel"],
            np.sum(
                np.asarray(gradients["kernel"]) * np.asarray(weights)[:, None], axis=0
            ),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(result["bias"], [1.0], rtol=0.0, atol=1e-12)

    def test_preserves_each_gradient_leaf_dtype(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            weighted_mean_environment_gradients,
        )

        weights = jnp.arange(1, 65, dtype=jnp.float64)
        weights = weights / jnp.sum(weights)
        gradients = {
            "f32": jnp.ones((64, 2), dtype=jnp.float32),
            "f64": jnp.ones((64, 2), dtype=jnp.float64),
        }

        result = weighted_mean_environment_gradients(gradients, weights)

        self.assertEqual(result["f32"].dtype, gradients["f32"].dtype)
        self.assertEqual(result["f64"].dtype, gradients["f64"].dtype)

    def test_fails_closed_on_invalid_weights_or_gradients(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            weighted_mean_environment_gradients,
        )

        valid_weights = jnp.full((64,), 1.0 / 64, dtype=jnp.float64)
        valid_gradients = {"w": jnp.ones((64, 2), dtype=jnp.float64)}
        invalid_cases = (
            (valid_gradients, valid_weights[:-1], "shape"),
            (valid_gradients, valid_weights.at[0].set(-0.1), "nonnegative"),
            (valid_gradients, valid_weights * 0.5, "sum"),
            (
                {"w": valid_gradients["w"].at[3, 0].set(jnp.inf)},
                valid_weights,
                "finite",
            ),
            ({"w": jnp.ones((63, 2), dtype=jnp.float64)}, valid_weights, "leading"),
        )
        for gradients, weights, message in invalid_cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                weighted_mean_environment_gradients(gradients, weights)


class FailureWeightedAggregationTest(unittest.TestCase):
    @staticmethod
    def _result(seed):
        phases = jnp.asarray(
            [
                *range(4),
                *range(100, 117),
                *range(200, 219),
                *range(300, 308),
                *range(400, 416),
            ],
            dtype=jnp.int32,
        )
        losses = (jnp.arange(64, dtype=jnp.float64) * (seed + 1)) % 11
        gradients = {
            "w": jnp.stack(
                (losses + 1.0, jnp.arange(64, dtype=jnp.float64) + seed),
                axis=1,
            )
        }
        trajectory = SimpleNamespace(
            initial_phase=phases,
            normalized_observations=jnp.full((64, 2, 3), seed, dtype=jnp.float32),
        )
        return SimpleNamespace(
            losses=losses,
            trajectory=trajectory,
            pathwise_effective_gradients=gradients,
        )

    def test_tail_aggregate_dtypes_match_uniform_gradient_dtypes(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            aggregate_failure_weighted_shards,
        )

        results = []
        for seed in range(4):
            result = self._result(seed)
            result.pathwise_effective_gradients["w"] = (
                result.pathwise_effective_gradients["w"].astype(jnp.float32)
            )
            results.append(result)

        aggregation = aggregate_failure_weighted_shards(results)

        self.assertEqual(aggregation.uniform_mean["w"].dtype, jnp.float32)
        self.assertEqual(aggregation.tail_mean["w"].dtype, jnp.float32)

    def test_aggregates_uniform_and_failure_weighted_directions_from_same_shards(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            aggregate_failure_weighted_shards,
            phase_preserving_lower_tail_weights,
            weighted_mean_environment_gradients,
        )

        results = tuple(self._result(seed) for seed in range(4))

        aggregation = aggregate_failure_weighted_shards(results)

        expected_uniform = np.mean(
            [
                np.mean(np.asarray(result.pathwise_effective_gradients["w"]), axis=0)
                for result in results
            ],
            axis=0,
        )
        expected_tail = np.mean(
            [
                np.asarray(
                    weighted_mean_environment_gradients(
                        result.pathwise_effective_gradients,
                        phase_preserving_lower_tail_weights(
                            result.losses, result.trajectory.initial_phase
                        ).weights,
                    )["w"]
                )
                for result in results
            ],
            axis=0,
        )
        np.testing.assert_array_equal(aggregation.uniform_mean["w"], expected_uniform)
        np.testing.assert_allclose(aggregation.tail_mean["w"], expected_tail)
        self.assertEqual(aggregation.normalized_observations.shape, (512, 3))
        self.assertEqual(len(aggregation.weighting_receipts), 4)
        for seed, receipt in enumerate(aggregation.weighting_receipts):
            self.assertEqual(receipt["seed"], seed)
            self.assertEqual(len(receipt["losses"]), 64)
            self.assertEqual(len(receipt["weights"]), 64)
            self.assertAlmostEqual(sum(receipt["weights"]), 1.0, places=12)
        self.assertIn("cross_shard_pairwise_cosines", aggregation.geometry)
        self.assertIn("leave_one_out_cosine_to_full", aggregation.geometry)
        self.assertIn("cross_direction", aggregation.geometry)

    def test_rejects_wrong_shard_count_and_incomplete_results(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            aggregate_failure_weighted_shards,
        )

        with self.assertRaisesRegex(ValueError, "four shards"):
            aggregate_failure_weighted_shards((self._result(0),) * 3)
        incomplete = SimpleNamespace(
            losses=jnp.ones(64),
            trajectory=self._result(0).trajectory,
        )
        with self.assertRaisesRegex((TypeError, ValueError), "gradient"):
            aggregate_failure_weighted_shards(
                (incomplete, self._result(1), self._result(2), self._result(3))
            )

    def test_independently_recomputes_published_weights_and_aggregates(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            aggregate_failure_weighted_shards,
            validate_failure_weighted_aggregation,
        )

        results = tuple(self._result(seed) for seed in range(4))
        aggregation = aggregate_failure_weighted_shards(results)

        receipt = validate_failure_weighted_aggregation(results, aggregation)

        self.assertTrue(receipt["weight_receipts_exact"])
        self.assertTrue(receipt["uniform_reproduction_exact"])
        self.assertLessEqual(
            receipt["independent_host_recomputation"]["tail_maximum_absolute_error"],
            1e-12,
        )

        changed_receipts = copy.deepcopy(aggregation.weighting_receipts)
        changed_receipts[0]["weights"][0] += 1e-3
        with self.assertRaisesRegex(ValueError, "weight receipt"):
            validate_failure_weighted_aggregation(
                results,
                replace(aggregation, weighting_receipts=changed_receipts),
            )

        changed_tail = {"w": aggregation.tail_mean["w"].at[0].add(1.0)}
        with self.assertRaisesRegex(ValueError, "tail aggregate"):
            validate_failure_weighted_aggregation(
                results,
                replace(aggregation, tail_mean=changed_tail),
            )

        float32_results = []
        for result in results:
            result.pathwise_effective_gradients["w"] = (
                result.pathwise_effective_gradients["w"].astype(jnp.float32)
            )
            float32_results.append(result)
        float32_aggregation = aggregate_failure_weighted_shards(float32_results)
        tiny_tail_change = {"w": float32_aggregation.tail_mean["w"].at[0].add(1e-5)}
        with self.assertRaisesRegex(ValueError, "exact producer tail aggregate"):
            validate_failure_weighted_aggregation(
                float32_results,
                replace(float32_aggregation, tail_mean=tiny_tail_change),
            )
        changed_tail_shards = list(float32_aggregation.tail_shard_means)
        changed_tail_shards[0] = {"w": changed_tail_shards[0]["w"].at[0].add(4e-5)}
        with self.assertRaisesRegex(ValueError, "tail shard 0"):
            validate_failure_weighted_aggregation(
                float32_results,
                replace(
                    float32_aggregation,
                    tail_shard_means=tuple(changed_tail_shards),
                ),
            )

    def test_independent_host_oracle_catches_a_shared_producer_bug(self):
        import src.algorithms.shac.g1_failure_weighted_audit as audit

        results = tuple(self._result(seed) for seed in range(4))
        production_helper = audit.phase_preserving_lower_tail_weights

        def wrong_producer(losses, phases):
            receipt = production_helper(losses, phases)
            return receipt._replace(weights=jnp.roll(receipt.weights, 1))

        with patch.object(
            audit,
            "phase_preserving_lower_tail_weights",
            side_effect=wrong_producer,
        ):
            aggregation = audit.aggregate_failure_weighted_shards(results)
            with self.assertRaisesRegex(ValueError, "independent host"):
                audit.validate_failure_weighted_aggregation(results, aggregation)

    def test_accepts_only_summary_roundoff_for_uneven_phase_bins(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            aggregate_failure_weighted_shards,
            validate_failure_weighted_aggregation,
        )

        phases = jnp.asarray(
            [
                *range(13),
                *range(100, 113),
                *range(200, 213),
                *range(300, 313),
                *range(400, 412),
            ],
            dtype=jnp.int32,
        )
        results = []
        for seed in range(4):
            result = self._result(seed)
            result.trajectory.initial_phase = phases
            results.append(result)

        aggregation = aggregate_failure_weighted_shards(results)
        receipt = validate_failure_weighted_aggregation(results, aggregation)

        self.assertTrue(receipt["weight_receipts_exact"])
        for published in aggregation.weighting_receipts:
            np.testing.assert_allclose(
                published["phase_weight_mass"],
                np.asarray([13, 13, 13, 13, 12]) / 64,
                rtol=0.0,
                atol=3e-14,
            )

        changed_receipts = copy.deepcopy(aggregation.weighting_receipts)
        changed_receipts[0]["phase_weight_mass"][0] += 1e-10
        with self.assertRaisesRegex(ValueError, "summary"):
            validate_failure_weighted_aggregation(
                results,
                replace(aggregation, weighting_receipts=changed_receipts),
            )

    def test_binds_stored_shard_means_and_geometry_to_verified_directions(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            aggregate_failure_weighted_shards,
            validate_failure_weighted_aggregation,
        )

        results = tuple(self._result(seed) for seed in range(4))
        aggregation = aggregate_failure_weighted_shards(results)
        changed_uniform_shards = list(aggregation.uniform_shard_means)
        changed_uniform_shards[2] = {
            "w": changed_uniform_shards[2]["w"].at[0].add(1e-8)
        }
        with self.assertRaisesRegex(ValueError, "uniform shard 2"):
            validate_failure_weighted_aggregation(
                results,
                replace(
                    aggregation,
                    uniform_shard_means=tuple(changed_uniform_shards),
                ),
            )

        changed_geometry = copy.deepcopy(aggregation.geometry)
        changed_geometry["cross_shard_pairwise_cosines"]["tail"]["pairs"][0][
            "cosine"
        ] = 10.0
        with self.assertRaisesRegex(ValueError, "cosine|geometry"):
            validate_failure_weighted_aggregation(
                results,
                replace(aggregation, geometry=changed_geometry),
            )

    def test_cosine_mean_summary_admits_only_e008_float32_reduction_roundoff(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            _validate_cosine_summary,
        )

        values = [
            0.41276466846466064,
            0.3933444619178772,
            0.5381776690483093,
            0.4716799855232239,
            0.5448299646377563,
            0.44469526410102844,
        ]
        e008_host_f64_mean = float(np.mean(values))
        self.assertEqual(e008_host_f64_mean, 0.46758200228214264)
        e008_producer_f32_mean = 0.4675820469856262
        receipt = {
            "minimum": min(values),
            "mean": e008_producer_f32_mean,
            "maximum": max(values),
        }

        _validate_cosine_summary(receipt, values, label="E008 uniform pairwise")

        corrupted = dict(receipt)
        corrupted["mean"] += 2.0 * np.finfo(np.float32).eps
        with self.assertRaisesRegex(ValueError, "mean summary"):
            _validate_cosine_summary(
                corrupted,
                values,
                label="E008 uniform pairwise",
            )


class FailureWeightedCandidatesTest(unittest.TestCase):
    def test_builds_equal_size_candidates_and_reports_their_separation(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            build_failure_weighted_candidates,
        )

        params = {"w": jnp.asarray([[1.0, -0.5], [0.25, 0.75]], dtype=jnp.float32)}
        observations = jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.5, 0.5]],
            dtype=jnp.float32,
        )

        def actor_apply(value, obs):
            return obs @ value["w"]

        candidates = build_failure_weighted_candidates(
            actor_apply=actor_apply,
            actor_params=params,
            uniform_gradient={
                "w": jnp.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
            },
            tail_gradient={
                "w": jnp.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=jnp.float32)
            },
            normalized_observations=observations,
            target_rms=0.01,
        )

        self.assertIs(candidates.baseline, params)
        baseline_output = actor_apply(params, observations)
        for label in ("uniform", "tail"):
            self.assertEqual(getattr(candidates, label)["w"].dtype, params["w"].dtype)
            output = actor_apply(getattr(candidates, label), observations)
            output_rms = jnp.sqrt(jnp.mean((output - baseline_output) ** 2))
            self.assertAlmostEqual(float(output_rms), 0.01, places=6)
            self.assertAlmostEqual(
                candidates.functional_steps[label]["output_rms"], 0.01, places=6
            )
        expected_separation = jnp.sqrt(
            jnp.mean(
                (
                    actor_apply(candidates.tail, observations)
                    - actor_apply(candidates.uniform, observations)
                )
                ** 2
            )
        )
        self.assertAlmostEqual(
            candidates.functional_steps["candidate_output_rms"]["tail_vs_uniform"],
            float(expected_separation),
        )

        with self.assertRaisesRegex(ValueError, "dtype"):
            build_failure_weighted_candidates(
                actor_apply=actor_apply,
                actor_params=params,
                uniform_gradient={"w": jnp.ones((2, 2), dtype=jnp.float64)},
                tail_gradient={"w": jnp.ones((2, 2), dtype=jnp.float32)},
                normalized_observations=observations,
                target_rms=0.01,
            )


class FailureWeightedOutcomeTest(unittest.TestCase):
    phases = (0, 100, 200, 300, 400)

    @staticmethod
    def _validity(**overrides):
        validity = {
            "frozen_hashes": True,
            "weight_receipts_exact": True,
            "uniform_reproduction_exact": True,
            "tail_reproduction_exact": True,
            "stability_evidence_exact": True,
            "aggregate_gradients_finite_nonzero": True,
            "candidate_trees_finite_nonzero": True,
            "functional_steps_valid": True,
            "rollouts_fresh_replay_free_complete_finite": True,
        }
        validity.update(overrides)
        return validity

    @staticmethod
    def _geometry(
        *,
        uniform_mean=0.40,
        uniform_min=0.30,
        tail_mean=0.38,
        tail_min=0.28,
        uniform_loo_min=0.50,
        tail_loo_min=0.47,
    ):
        return {
            "cross_shard_pairwise_cosines": {
                "uniform": {"mean": uniform_mean, "minimum": uniform_min},
                "tail": {"mean": tail_mean, "minimum": tail_min},
            },
            "leave_one_out_cosine_to_full": {
                "uniform": {"minimum": uniform_loo_min},
                "tail": {"minimum": tail_loo_min},
            },
        }

    def _evaluation(
        self,
        *,
        baseline_returns=(1.0,) * 5,
        uniform_returns=(1.0,) * 5,
        tail_returns=(1.002,) * 5,
        baseline_survival=(110, 78, 74, 76, 58),
        uniform_survival=(110, 78, 74, 76, 58),
        tail_survival=(110, 78, 74, 76, 58),
    ):
        metrics = {
            "baseline": (baseline_returns, baseline_survival),
            "uniform": (uniform_returns, uniform_survival),
            "tail": (tail_returns, tail_survival),
        }
        candidates = {
            label: [
                {
                    "phase": phase,
                    "return": returns[index],
                    "survival": survivals[index],
                    "complete": True,
                    "replay_free": True,
                }
                for index, phase in enumerate(self.phases)
            ]
            for label, (returns, survivals) in metrics.items()
        }
        return {
            "mode": "single-deterministic-five-phase-grid",
            "seed": 0,
            "phases": self.phases,
            "per_seed": [{"seed": 0, "candidates": candidates}],
        }

    @staticmethod
    def _steps(separation=0.005):
        return {
            "uniform": {"output_rms": 0.01},
            "tail": {"output_rms": 0.01},
            "candidate_output_rms": {"tail_vs_uniform": separation},
        }

    def _classify(self, **overrides):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            classify_failure_weighted_outcome,
        )

        arguments = {
            "geometry": self._geometry(),
            "evaluation": self._evaluation(),
            "functional_steps": self._steps(),
            "validity": self._validity(),
        }
        arguments.update(overrides)
        return classify_failure_weighted_outcome(**arguments)

    def test_support_requires_mean_gain_safety_and_two_phase_wins(self):
        supported = self._classify()

        self.assertEqual(supported["verdict"], "failure-aware-supported")
        self.assertEqual(
            supported["decision_metrics"]["material_phase_wins_over_uniform"], 5
        )

        one_phase = self._evaluation(tail_returns=(1.002, 1.0, 1.0, 1.0, 1.0))
        not_material = self._classify(evaluation=one_phase)
        self.assertEqual(not_material["verdict"], "failure-aware-not-material")

        unsafe = self._evaluation(tail_survival=(98, 78, 74, 76, 58))
        unsafe_result = self._classify(evaluation=unsafe)
        self.assertEqual(unsafe_result["verdict"], "failure-aware-not-material")

    def test_stability_regression_has_distinct_verdict(self):
        geometry = self._geometry(tail_mean=0.34, tail_min=0.24)

        result = self._classify(geometry=geometry)

        self.assertEqual(result["verdict"], "failure-aware-unstable")

    def test_leave_one_out_regression_is_independently_gated(self):
        geometry = self._geometry(tail_loo_min=0.449)

        result = self._classify(geometry=geometry)

        self.assertEqual(result["verdict"], "failure-aware-unstable")

    def test_exact_stability_tolerance_boundary_passes(self):
        geometry = self._geometry(
            tail_mean=0.35,
            tail_min=0.25,
            tail_loo_min=0.45,
        )

        result = self._classify(geometry=geometry)

        self.assertEqual(result["verdict"], "failure-aware-supported")

    def test_small_candidate_separation_is_descriptive_only(self):
        result = self._classify(functional_steps=self._steps(separation=0.000099))

        self.assertEqual(result["verdict"], "failure-aware-supported")
        self.assertEqual(
            result["decision_metrics"]["tail_vs_uniform_output_rms"], 0.000099
        )

    def test_malformed_or_invalid_evidence_fails_closed(self):
        malformed_seed = self._evaluation()
        malformed_seed["per_seed"] = [None]
        malformed_rows = self._evaluation()
        malformed_rows["per_seed"][0]["candidates"]["tail"] = {"not": "a row list"}
        invalid_cases = (
            ({"validity": self._validity(weight_receipts_exact=False)}, "validity"),
            ({"validity": {"frozen_hashes": True}}, "key mismatch"),
            ({"validity": None}, "mapping"),
            (
                {
                    "evaluation": self._evaluation(
                        baseline_survival=(109, 78, 74, 76, 58)
                    )
                },
                "baseline",
            ),
            (
                {"functional_steps": self._steps(separation=float("nan"))},
                "nonfinite",
            ),
            ({"evaluation": malformed_seed}, "seed row"),
            ({"evaluation": malformed_rows}, "five rows"),
            (
                {"geometry": self._geometry(tail_mean=10.0)},
                "cosine",
            ),
        )
        for overrides, reason in invalid_cases:
            with self.subTest(reason=reason):
                result = self._classify(**overrides)
                self.assertEqual(result["verdict"], "invalid")
                self.assertIn(reason, result["reason"])


if __name__ == "__main__":
    unittest.main()
