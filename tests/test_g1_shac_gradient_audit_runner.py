import json
import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.g1_gradient_audit_runner import (
    CandidateActors,
    OutcomeThresholds,
    aggregate_four_shards,
    bootstrap_direction_confidence,
    build_descent_candidates,
    classify_preregistered_outcome,
    evaluate_held_out_candidates,
    gradient_tree_geometry,
    mean_pytrees,
    pairwise_tree_cosines,
    to_finite_json,
)


class TreeAggregationTest(unittest.TestCase):
    def test_mean_pytrees_matches_structure_and_values(self):
        trees = (
            {"dense": {"kernel": jnp.array([1.0, 3.0])}, "bias": jnp.array(1.0)},
            {"dense": {"kernel": jnp.array([3.0, 1.0])}, "bias": jnp.array(3.0)},
        )

        result = mean_pytrees(trees)

        np.testing.assert_array_equal(result["dense"]["kernel"], [2.0, 2.0])
        np.testing.assert_array_equal(result["bias"], 2.0)
        with self.assertRaisesRegex(ValueError, "pytree"):
            mean_pytrees((trees[0], {"other": jnp.array([1.0])}))

    def test_reports_per_layer_and_aggregate_geometry(self):
        left = {
            "layer_0": jnp.array([1.0, -2.0, 0.0]),
            "layer_1": jnp.array([1.0, 1.0]),
        }
        right = {
            "layer_0": jnp.array([1.0, 2.0, 0.0]),
            "layer_1": jnp.array([1.0, -1.0]),
        }

        geometry = gradient_tree_geometry(left, right)

        self.assertAlmostEqual(geometry["aggregate_cosine"], -3.0 / 7.0)
        self.assertEqual(
            [layer["path"] for layer in geometry["layers"]],
            ["['layer_0']", "['layer_1']"],
        )
        self.assertAlmostEqual(geometry["layers"][0]["cosine"], -0.6)
        self.assertAlmostEqual(
            geometry["layers"][0]["sign_agreement_fraction"], 2.0 / 3.0
        )
        self.assertAlmostEqual(geometry["layers"][1]["cosine"], 0.0)
        self.assertAlmostEqual(
            geometry["layers"][1]["sign_agreement_fraction"], 0.5
        )

    def test_pairwise_cosines_have_stable_order_and_summary(self):
        trees = (
            {"w": jnp.array([1.0, 0.0])},
            {"w": jnp.array([0.0, 1.0])},
            {"w": jnp.array([1.0, 1.0])},
            {"w": jnp.array([-1.0, 0.0])},
        )

        result = pairwise_tree_cosines(trees, labels=(0, 1, 2, 3))

        self.assertEqual(
            [(row["left"], row["right"]) for row in result["pairs"]],
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        )
        self.assertAlmostEqual(result["pairs"][0]["cosine"], 0.0)
        self.assertAlmostEqual(result["pairs"][2]["cosine"], -1.0)
        self.assertEqual(result["count"], 6)
        json.dumps(result, allow_nan=False)

    def test_bootstrap_direction_confidence_is_deterministic_and_finite(self):
        trees = tuple({"w": jnp.array([1.0, 0.0])} for _ in range(4))

        first = bootstrap_direction_confidence(trees, confidence_level=0.95)
        second = bootstrap_direction_confidence(trees, confidence_level=0.95)

        self.assertEqual(first, second)
        self.assertEqual(first["resample_count"], 256)
        self.assertEqual(
            first["cosine_to_full_mean"],
            {"mean": 1.0, "lower": 1.0, "upper": 1.0},
        )
        json.dumps(first, allow_nan=False)

    def test_bootstrap_uses_same_finite_count_weights_as_score_aggregate(self):
        trees = (
            {"w": jnp.array([100.0])},
            {"w": jnp.array([0.0])},
            {"w": jnp.array([0.0])},
            {"w": jnp.array([0.0])},
        )

        weighted = bootstrap_direction_confidence(
            trees,
            weights=(1, 3, 3, 3),
            confidence_level=0.95,
        )
        unweighted = bootstrap_direction_confidence(
            trees,
            confidence_level=0.95,
        )

        self.assertAlmostEqual(weighted["full_aggregate_norm"], 10.0)
        self.assertAlmostEqual(unweighted["full_aggregate_norm"], 25.0)


class FourShardRunnerTest(unittest.TestCase):
    @staticmethod
    def _fake_result(seed):
        pathwise_raw = {
            "w": jnp.array(
                [[100.0 + seed], [200.0 + seed], [300.0 + seed]]
            )
        }
        pathwise_effective = {
            "w": jnp.array([[1.0 + seed], [3.0 + seed], [5.0 + seed]])
        }
        score = {
            "w": jnp.array([[10.0 + seed], [jnp.nan], [14.0 + seed]])
        }
        trajectory = SimpleNamespace(
            initial_phase=jnp.array([0, 100, 200]),
            normalized_observations=jnp.full((3, 2, 1), float(seed)),
        )
        return SimpleNamespace(
            pathwise_raw_gradients=pathwise_raw,
            pathwise_effective_gradients=pathwise_effective,
            score_gradients=score,
            trajectory=trajectory,
        )

    def test_runs_four_shards_sequentially_and_uses_estimator_specific_means(self):
        calls = []

        def estimate_shared_gradients(*, seed, marker):
            calls.append((seed, marker))
            return self._fake_result(seed)

        result = aggregate_four_shards(
            shard_seeds=(0, 1, 2, 3),
            estimate_shared_gradients=estimate_shared_gradients,
            estimate_kwargs={"marker": "same-shape-jit"},
            pathwise_clip_norm=1.0,
        )

        self.assertEqual(calls, [(seed, "same-shape-jit") for seed in range(4)])
        np.testing.assert_allclose(
            [tree["w"][0] for tree in result.pathwise_shard_means],
            [3.0, 4.0, 5.0, 6.0],
        )
        np.testing.assert_allclose(
            [tree["w"][0] for tree in result.score_shard_means],
            [12.0, 13.0, 14.0, 15.0],
        )
        np.testing.assert_allclose(result.pathwise_mean["w"], [4.5])
        np.testing.assert_allclose(result.score_mean["w"], [13.5])
        self.assertEqual(result.normalized_observations.shape, (24, 1))
        self.assertEqual(result.geometry["score_finite_count_by_shard"], [2] * 4)
        self.assertEqual(
            result.geometry["cross_shard_pairwise_cosines"]["pathwise"]["count"],
            6,
        )
        self.assertAlmostEqual(
            result.geometry["cross_estimator"]["aggregate_cosine"], 1.0
        )
        json.dumps(result.geometry, allow_nan=False)

    def test_requires_exactly_four_distinct_shards(self):
        for seeds in ((0, 1, 2), (0, 1, 2, 2), (0, 1, 2, 4)):
            with self.subTest(seeds=seeds):
                with self.assertRaisesRegex(ValueError, "shard seeds"):
                    aggregate_four_shards(
                        shard_seeds=seeds,
                        estimate_shared_gradients=lambda **_: None,
                        estimate_kwargs={},
                        pathwise_clip_norm=1.0,
                    )

    def test_weights_score_shards_by_their_finite_environment_counts(self):
        def estimate_shared_gradients(*, seed):
            result = self._fake_result(seed)
            if seed == 0:
                score = {"w": jnp.array([[100.0], [jnp.nan], [jnp.nan]])}
            else:
                score = {"w": jnp.zeros((3, 1))}
            return result.__class__(**{**vars(result), "score_gradients": score})

        result = aggregate_four_shards(
            shard_seeds=(0, 1, 2, 3),
            estimate_shared_gradients=estimate_shared_gradients,
            estimate_kwargs={},
            pathwise_clip_norm=1.0,
        )

        np.testing.assert_allclose(result.score_mean["w"], [10.0])
        self.assertEqual(
            result.geometry["score_finite_count_by_shard"], [1, 3, 3, 3]
        )


class JsonAndCandidateTest(unittest.TestCase):
    def test_json_conversion_materializes_arrays_and_rejects_nonfinite_values(self):
        converted = to_finite_json(
            {"scalar": jnp.array(2.5), "vector": jnp.array([1, 2])}
        )

        self.assertEqual(converted, {"scalar": 2.5, "vector": [1, 2]})
        json.dumps(converted, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            to_finite_json({"bad": jnp.array([jnp.inf])})

    def test_builds_equal_functional_size_descent_candidates(self):
        params = {"w": jnp.array([[1.0]])}
        observations = jnp.ones((8, 1))

        def actor_apply(value, obs):
            return obs @ value["w"]

        candidates = build_descent_candidates(
            actor_apply=actor_apply,
            actor_params=params,
            pathwise_gradient={"w": jnp.array([[2.0]])},
            score_gradient={"w": jnp.array([[4.0]])},
            normalized_observations=observations,
            target_rms=0.01,
        )

        self.assertIs(candidates.baseline, params)
        self.assertLess(float(candidates.pathwise["w"][0, 0]), 1.0)
        self.assertLess(float(candidates.score["w"][0, 0]), 1.0)
        for label in ("pathwise", "score"):
            candidate = getattr(candidates, label)
            delta = actor_apply(candidate, observations) - actor_apply(
                params, observations
            )
            self.assertAlmostEqual(float(jnp.sqrt(jnp.mean(delta**2))), 0.01)
            self.assertAlmostEqual(
                candidates.functional_steps[label]["output_rms"], 0.01
            )
        json.dumps(candidates.functional_steps, allow_nan=False)


class HeldOutEvaluationAndDecisionTest(unittest.TestCase):
    phases = (0, 100, 200, 300, 400)
    seeds = (4, 5, 6, 7)

    @staticmethod
    def _candidates():
        params = {"w": jnp.array([0.0])}
        return CandidateActors(
            baseline=params,
            pathwise={"w": jnp.array([-1.0])},
            score={"w": jnp.array([-2.0])},
            functional_steps={"pathwise": {}, "score": {}},
        )

    def _evaluation(
        self,
        *,
        baseline=(10.0, 100.0),
        pathwise=(9.0, 100.0),
        score=(12.0, 95.0),
        score_phase_survival=None,
        replay_free=True,
        complete=True,
    ):
        def evaluate_seed(*, seed, candidates, phases):
            self.assertEqual(tuple(candidates), ("baseline", "pathwise", "score"))
            score_survival = score_phase_survival or [score[1]] * len(phases)
            metrics = {
                "baseline": [baseline] * len(phases),
                "pathwise": [pathwise] * len(phases),
                "score": list(zip([score[0]] * len(phases), score_survival)),
            }
            return {
                label: [
                    {
                        "phase": phase,
                        "return": values[index][0] + 0.0 * seed,
                        "survival": values[index][1],
                        "replay_free": replay_free,
                        "complete": complete,
                    }
                    for index, phase in enumerate(phases)
                ]
                for label, values in metrics.items()
            }

        return evaluate_held_out_candidates(
            candidates=self._candidates(),
            held_out_seeds=self.seeds,
            phases=self.phases,
            evaluate_seed=evaluate_seed,
        )

    @staticmethod
    def _geometry(pathwise, score, alignment=0.0, *, confidence=True):
        geometry = {
            "cross_shard_pairwise_cosines": {
                "pathwise": {"mean": pathwise},
                "score": {"mean": score},
            },
            "cross_estimator": {"aggregate_cosine": alignment},
        }
        if confidence:
            geometry["bootstrap_direction_confidence"] = {
                "pathwise": {
                    "method": "exhaustive-four-shard-percentile",
                    "confidence_level": 0.95,
                    "resample_count": 256,
                    "cosine_to_full_mean": {
                        "lower": pathwise - 0.05,
                        "upper": pathwise + 0.05,
                    }
                },
                "score": {
                    "method": "exhaustive-four-shard-percentile",
                    "confidence_level": 0.95,
                    "resample_count": 256,
                    "cosine_to_full_mean": {
                        "lower": score - 0.05,
                        "upper": score + 0.05,
                    }
                },
            }
        return geometry

    @staticmethod
    def _validity(**updates):
        validity = {
            "frozen_hashes": True,
            "aggregate_gradients_finite_nonzero": True,
            "candidate_trees_finite_nonzero": True,
            "analytic_gaussian_sign": True,
            "detachment": True,
            "done_boundary_return": True,
            "ppo_ratio_one": True,
            "pytree_order": True,
            "smooth_toy_convergence": True,
            "rollouts_fresh_replay_free_complete_finite": True,
        }
        validity.update(updates)
        return validity

    @staticmethod
    def _thresholds():
        return OutcomeThresholds(
            minimum_stability=0.5,
            material_stability_advantage=0.2,
            minimum_alignment=0.7,
            minimum_return_improvement=1.0,
            minimum_survival_improvement=5.0,
            stability_tolerance=0.01,
            return_tolerance=0.1,
            survival_tolerance=1.0,
            maximum_phase_survival_loss_fraction=0.1,
            bootstrap_confidence_level=0.95,
        )

    def test_evaluates_all_candidates_once_per_common_random_seed(self):
        calls = []

        def evaluate_seed(*, seed, candidates, phases):
            calls.append((seed, tuple(candidates), tuple(phases)))
            return {
                label: [
                    {
                        "phase": phase,
                        "return": float(seed),
                        "survival": 50.0,
                        "replay_free": True,
                        "complete": True,
                    }
                    for phase in phases
                ]
                for label in candidates
            }

        result = evaluate_held_out_candidates(
            candidates=self._candidates(),
            held_out_seeds=self.seeds,
            phases=self.phases,
            evaluate_seed=evaluate_seed,
        )

        self.assertEqual([call[0] for call in calls], list(self.seeds))
        self.assertTrue(
            all(call[1] == ("baseline", "pathwise", "score") for call in calls)
        )
        self.assertEqual(len(result["per_seed"]), 4)
        json.dumps(result, allow_nan=False)

    def test_rejects_nonfrozen_held_out_seeds_or_phases(self):
        for seeds, phases in (
            ((4, 5, 6, 8), self.phases),
            (self.seeds, (0, 100, 200, 300, 401)),
        ):
            with self.subTest(seeds=seeds, phases=phases):
                with self.assertRaisesRegex(ValueError, "held-out seeds|phases"):
                    evaluate_held_out_candidates(
                        candidates=self._candidates(),
                        held_out_seeds=seeds,
                        phases=phases,
                        evaluate_seed=lambda **_: {},
                    )

    def test_classifies_pathwise_quality_limited_with_phase_survival_gate(self):
        result = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.4, score=0.9),
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )

        self.assertEqual(result["verdict"], "pathwise-quality-limited")
        self.assertTrue(result["decision_metrics"]["score_phase_survival_gate"])

        gated = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.4, score=0.9),
            evaluation=self._evaluation(
                score=(12.0, 105.0),
                score_phase_survival=[89.0, 110.0, 110.0, 110.0, 110.0],
            ),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(gated["verdict"], "inconclusive")
        self.assertFalse(
            gated["decision_metrics"]["score_phase_survival_gate"]
        )

    def test_classifies_shared_objective_and_pathwise_supported(self):
        shared = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.8, score=0.8, alignment=0.9),
            evaluation=self._evaluation(
                pathwise=(10.1, 101.0), score=(10.05, 100.5)
            ),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(shared["verdict"], "shared-objective-limited")

        supported = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.9, score=0.8, alignment=0.2),
            evaluation=self._evaluation(
                baseline=(10.0, 100.0),
                pathwise=(12.0, 105.0),
                score=(11.0, 103.0),
            ),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(supported["verdict"], "pathwise-supported")

    def test_classifies_invalid_and_inconclusive_paths(self):
        invalid = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.8, score=0.8),
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity=self._validity(frozen_hashes=False),
        )
        self.assertEqual(invalid["verdict"], "invalid")

        incomplete = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.8, score=0.8),
            evaluation=self._evaluation(complete=False),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(incomplete["verdict"], "invalid")

        noisy_score = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.8, score=0.2),
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(noisy_score["verdict"], "inconclusive")

        missing_validity = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.8, score=0.8),
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity={},
        )
        self.assertEqual(missing_validity["verdict"], "invalid")

    def test_confidence_intervals_can_force_inconclusive(self):
        missing = classify_preregistered_outcome(
            geometry=self._geometry(pathwise=0.4, score=0.9, confidence=False),
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(missing["verdict"], "inconclusive")

        overlapping_geometry = self._geometry(pathwise=0.4, score=0.9)
        overlapping_geometry["bootstrap_direction_confidence"]["pathwise"][
            "cosine_to_full_mean"
        ]["upper"] = 0.65
        overlapping_geometry["bootstrap_direction_confidence"]["score"][
            "cosine_to_full_mean"
        ]["lower"] = 0.7
        overlapping = classify_preregistered_outcome(
            geometry=overlapping_geometry,
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(overlapping["verdict"], "inconclusive")

        forged_geometry = self._geometry(pathwise=0.4, score=0.9)
        forged_geometry["bootstrap_direction_confidence"]["score"][
            "method"
        ] = "unregistered-random-bootstrap"
        forged = classify_preregistered_outcome(
            geometry=forged_geometry,
            evaluation=self._evaluation(),
            thresholds=self._thresholds(),
            validity=self._validity(),
        )
        self.assertEqual(forged["verdict"], "invalid")


if __name__ == "__main__":
    unittest.main()
