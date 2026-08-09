"""Pure orchestration helpers for the bounded G1 gradient-quality audit.

Simulator construction, checkpoint loading, JIT compilation, and evidence I/O
belong to the experiment runtime.  This module sequences injected shard and
evaluation callables and turns their results into deterministic finite data.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.batch_gradients import tree_cosine, tree_norm
from src.algorithms.shac.gradient_audit import (
    apply_functional_actor_step,
    assert_matching_pytree_leaf_order,
    summarize_per_env_gradient_geometry,
)


PyTree = Any
_CANDIDATE_LABELS = ("baseline", "pathwise", "score")
_FIXED_SHARD_SEEDS = (0, 1, 2, 3)
_FIXED_HELD_OUT_SEEDS = (4, 5, 6, 7)
_FIXED_PHASES = (0, 100, 200, 300, 400)
_REQUIRED_VALIDITY_KEYS = frozenset(
    {
        "frozen_hashes",
        "aggregate_gradients_finite_nonzero",
        "candidate_trees_finite_nonzero",
        "analytic_gaussian_sign",
        "detachment",
        "done_boundary_return",
        "ppo_ratio_one",
        "pytree_order",
        "smooth_toy_convergence",
        "rollouts_fresh_replay_free_complete_finite",
    }
)


@dataclass(frozen=True)
class FourShardAggregation:
    """Estimator directions, observations, and JSON geometry from four shards."""

    shard_seeds: tuple[int, int, int, int]
    pathwise_shard_means: tuple[PyTree, PyTree, PyTree, PyTree]
    score_shard_means: tuple[PyTree, PyTree, PyTree, PyTree]
    pathwise_mean: PyTree
    score_mean: PyTree
    normalized_observations: jax.Array
    geometry: dict[str, Any]


@dataclass(frozen=True)
class CandidateActors:
    """Frozen baseline and equal-functional-size descent candidates."""

    baseline: PyTree
    pathwise: PyTree
    score: PyTree
    functional_steps: dict[str, Any]


@dataclass(frozen=True)
class OutcomeThresholds:
    """Explicit preregistered numerical gates for the mechanism verdict."""

    minimum_stability: float
    material_stability_advantage: float
    minimum_alignment: float
    minimum_return_improvement: float
    minimum_survival_improvement: float
    stability_tolerance: float
    return_tolerance: float
    survival_tolerance: float
    maximum_phase_survival_loss_fraction: float
    bootstrap_confidence_level: float


def _require_finite_tree(tree: PyTree, *, label: str) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{label} pytree must not be empty")
    if any(not bool(jp.all(jp.isfinite(leaf))) for leaf in leaves):
        raise ValueError(f"{label} pytree contains nonfinite values")


def mean_pytrees(trees: Sequence[PyTree]) -> PyTree:
    """Return the leafwise arithmetic mean of matching nonempty PyTrees."""
    trees = tuple(trees)
    if not trees:
        raise ValueError("at least one pytree is required")
    try:
        assert_matching_pytree_leaf_order(*trees)
    except ValueError as error:
        raise ValueError("pytrees must have matching structure and leaf order") from error

    count = len(trees)
    result = jax.tree_util.tree_map(
        lambda *leaves: sum(leaves[1:], leaves[0]) / count,
        *trees,
    )
    _require_finite_tree(result, label="mean")
    return result


def _array_cosine(left: jax.Array, right: jax.Array) -> jax.Array:
    left = jp.ravel(left)
    right = jp.ravel(right)
    numerator = jp.vdot(left, right).real
    denominator = jp.linalg.norm(left) * jp.linalg.norm(right)
    return jp.where(
        denominator > 0.0,
        numerator / jp.maximum(denominator, jp.finfo(denominator.dtype).tiny),
        jp.zeros_like(numerator),
    )


def gradient_tree_geometry(left: PyTree, right: PyTree) -> dict[str, Any]:
    """Report aggregate and per-leaf cosine and elementwise sign agreement."""
    try:
        assert_matching_pytree_leaf_order(left, right)
    except ValueError as error:
        raise ValueError("gradient pytrees must match") from error
    _require_finite_tree(left, label="left gradient")
    _require_finite_tree(right, label="right gradient")

    left_with_paths, _ = jax.tree_util.tree_flatten_with_path(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    layers = []
    total_agreements = 0
    total_elements = 0
    for (path, left_leaf), right_leaf in zip(
        left_with_paths, right_leaves, strict=True
    ):
        agreement = jp.sign(left_leaf) == jp.sign(right_leaf)
        size = int(left_leaf.size)
        layers.append(
            {
                "path": jax.tree_util.keystr(path),
                "size": size,
                "cosine": _array_cosine(left_leaf, right_leaf),
                "sign_agreement_fraction": jp.mean(agreement.astype(jp.float64)),
            }
        )
        total_agreements += int(jp.sum(agreement))
        total_elements += size

    return to_finite_json(
        {
            "aggregate_cosine": tree_cosine(left, right),
            "aggregate_sign_agreement_fraction": total_agreements
            / total_elements,
            "left_norm": tree_norm(left),
            "right_norm": tree_norm(right),
            "layers": layers,
        }
    )


def pairwise_tree_cosines(
    trees: Sequence[PyTree],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic upper-triangle cosine pairs and finite summaries."""
    trees = tuple(trees)
    if len(trees) < 2:
        raise ValueError("at least two pytrees are required")
    try:
        assert_matching_pytree_leaf_order(*trees)
    except ValueError as error:
        raise ValueError("pairwise pytrees must match") from error
    for index, tree in enumerate(trees):
        _require_finite_tree(tree, label=f"tree {index}")

    if labels is None:
        labels = tuple(range(len(trees)))
    else:
        labels = tuple(labels)
    if len(labels) != len(trees):
        raise ValueError("labels must align with pytrees")

    pairs = [
        {
            "left": labels[left],
            "right": labels[right],
            "cosine": tree_cosine(trees[left], trees[right]),
        }
        for left in range(len(trees))
        for right in range(left + 1, len(trees))
    ]
    values = jp.asarray([pair["cosine"] for pair in pairs])
    return to_finite_json(
        {
            "count": len(pairs),
            "mean": jp.mean(values),
            "minimum": jp.min(values),
            "maximum": jp.max(values),
            "pairs": pairs,
        }
    )


def bootstrap_direction_confidence(
    trees: Sequence[PyTree],
    *,
    confidence_level: float,
    weights: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Exhaustively bootstrap four shard directions against their full mean.

    All ``4**4 == 256`` ordered resamples are enumerated, avoiding an
    unregistered bootstrap RNG seed.  The percentile interval summarizes the
    cosine of each resampled aggregate to the aggregate of all four shards.
    """
    trees = tuple(trees)
    if len(trees) != 4:
        raise ValueError("bootstrap confidence requires exactly four pytrees")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    try:
        assert_matching_pytree_leaf_order(*trees)
    except ValueError as error:
        raise ValueError("bootstrap pytrees must match") from error
    if weights is None:
        weights = (1, 1, 1, 1)
    else:
        weights = tuple(int(weight) for weight in weights)
    if len(weights) != 4 or any(weight <= 0 for weight in weights):
        raise ValueError("bootstrap weights must contain four positive counts")
    full_mean = _weighted_mean_pytrees(trees, weights)
    cosines = []
    norms = []
    for indices in product(range(4), repeat=4):
        resampled_mean = _weighted_mean_pytrees(
            tuple(trees[index] for index in indices),
            tuple(weights[index] for index in indices),
        )
        cosines.append(float(tree_cosine(resampled_mean, full_mean)))
        norms.append(float(tree_norm(resampled_mean)))

    lower_percentile = 50.0 * (1.0 - confidence_level)
    upper_percentile = 100.0 - lower_percentile

    def summarize(values):
        return {
            "mean": float(np.mean(values)),
            "lower": float(np.percentile(values, lower_percentile)),
            "upper": float(np.percentile(values, upper_percentile)),
        }

    return to_finite_json(
        {
            "method": "exhaustive-four-shard-percentile",
            "confidence_level": confidence_level,
            "resample_count": len(cosines),
            "full_aggregate_norm": tree_norm(full_mean),
            "cosine_to_full_mean": summarize(cosines),
            "aggregate_norm": summarize(norms),
        }
    )


def to_finite_json(value: Any, *, path: str = "document") -> Any:
    """Convert nested JAX/NumPy values to finite JSON-native Python values."""
    if dataclasses.is_dataclass(value):
        return to_finite_json(dataclasses.asdict(value), path=path)
    if hasattr(value, "_asdict"):
        return to_finite_json(value._asdict(), path=path)
    if isinstance(value, Mapping):
        return {
            str(key): to_finite_json(child, path=f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(jax.device_get(value))
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite value at {path}")
        if array.ndim == 0:
            return to_finite_json(array.item(), path=path)
        return to_finite_json(array.tolist(), path=path)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            to_finite_json(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value at {path}")
        return float(value)
    raise TypeError(f"value at {path} is not JSON serializable: {type(value)!r}")


def _per_environment_finite_mask(per_env_tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(per_env_tree)
    if not leaves or any(leaf.ndim == 0 for leaf in leaves):
        raise ValueError("per-environment gradients need a leading environment axis")
    population = leaves[0].shape[0]
    if population == 0 or any(leaf.shape[0] != population for leaf in leaves):
        raise ValueError("gradient leaves must share a nonempty environment axis")
    finite = jp.ones((population,), dtype=jp.bool_)
    for leaf in leaves:
        finite = finite & jp.all(
            jp.isfinite(leaf), axis=tuple(range(1, leaf.ndim))
        )
    return finite


def _mean_all_environments(per_env_tree: PyTree, *, label: str) -> PyTree:
    _per_environment_finite_mask(per_env_tree)
    _require_finite_tree(per_env_tree, label=label)
    return jax.tree_util.tree_map(lambda leaf: jp.mean(leaf, axis=0), per_env_tree)


def _finite_only_environment_mean(
    per_env_tree: PyTree,
) -> tuple[PyTree, int]:
    finite = _per_environment_finite_mask(per_env_tree)
    finite_count = int(jp.sum(finite))
    if finite_count == 0:
        raise ValueError("score gradients contain no finite environments")
    population = finite.shape[0]

    def mean_leaf(leaf):
        shape = (population,) + (1,) * (leaf.ndim - 1)
        sanitized = jp.where(jp.isfinite(leaf), leaf, 0.0)
        return jp.sum(sanitized * finite.reshape(shape), axis=0) / finite_count

    mean = jax.tree_util.tree_map(mean_leaf, per_env_tree)
    _require_finite_tree(mean, label="score finite-only mean")
    return mean, finite_count


def _weighted_mean_pytrees(
    trees: Sequence[PyTree], weights: Sequence[int]
) -> PyTree:
    trees = tuple(trees)
    weights = tuple(int(weight) for weight in weights)
    if len(trees) != len(weights) or not trees:
        raise ValueError("weights must align with a nonempty pytree sequence")
    if any(weight <= 0 for weight in weights):
        raise ValueError("pytree mean weights must be positive")
    try:
        assert_matching_pytree_leaf_order(*trees)
    except ValueError as error:
        raise ValueError("weighted pytrees must match") from error
    denominator = sum(weights)
    result = jax.tree_util.tree_map(
        lambda *leaves: sum(
            (leaf * weight for leaf, weight in zip(leaves, weights, strict=True)),
            jp.zeros_like(leaves[0]),
        )
        / denominator,
        *trees,
    )
    _require_finite_tree(result, label="weighted mean")
    return result


def _flatten_observations(observations: jax.Array) -> jax.Array:
    observations = jp.asarray(observations)
    if observations.ndim < 2 or observations.shape[-1] == 0:
        raise ValueError("normalized observations must have a nonempty feature axis")
    if not bool(jp.all(jp.isfinite(observations))):
        raise ValueError("normalized observations contain nonfinite values")
    return observations.reshape((-1, observations.shape[-1]))


def aggregate_four_shards(
    *,
    shard_seeds: Sequence[int],
    estimate_shared_gradients: Callable[..., Any],
    estimate_kwargs: Mapping[str, Any],
    pathwise_clip_norm: float,
    bootstrap_confidence_level: float = 0.95,
) -> FourShardAggregation:
    """Run four injected shards sequentially and aggregate exact estimator means."""
    shard_seeds = tuple(int(seed) for seed in shard_seeds)
    if shard_seeds != _FIXED_SHARD_SEEDS:
        raise ValueError(f"shard seeds must equal {_FIXED_SHARD_SEEDS}")
    if "seed" in estimate_kwargs:
        raise ValueError("estimate_kwargs must not override the shard seed")
    if not math.isfinite(pathwise_clip_norm) or pathwise_clip_norm <= 0.0:
        raise ValueError("pathwise_clip_norm must be finite and positive")

    pathwise_shard_means = []
    score_shard_means = []
    score_finite_counts = []
    normalized_observations = []
    per_shard_geometry = []
    for seed in shard_seeds:
        result = estimate_shared_gradients(seed=seed, **dict(estimate_kwargs))
        pathwise_mean = _mean_all_environments(
            result.pathwise_effective_gradients,
            label="pathwise training-effective gradients",
        )
        score_mean, score_finite_count = _finite_only_environment_mean(
            result.score_gradients
        )
        phases = jp.asarray(result.trajectory.initial_phase)
        pathwise_geometry = summarize_per_env_gradient_geometry(
            result.pathwise_raw_gradients,
            max_norm=pathwise_clip_norm,
            initial_phases=phases,
        )
        score_geometry = summarize_per_env_gradient_geometry(
            result.score_gradients,
            max_norm=pathwise_clip_norm,
            initial_phases=phases,
        )

        pathwise_shard_means.append(pathwise_mean)
        score_shard_means.append(score_mean)
        score_finite_counts.append(score_finite_count)
        normalized_observations.append(
            _flatten_observations(result.trajectory.normalized_observations)
        )
        per_shard_geometry.append(
            {
                "seed": seed,
                "pathwise": to_finite_json(pathwise_geometry),
                "score": to_finite_json(score_geometry),
                "cross_estimator": gradient_tree_geometry(
                    pathwise_mean, score_mean
                ),
            }
        )

    pathwise_shard_means = tuple(pathwise_shard_means)
    score_shard_means = tuple(score_shard_means)
    pathwise_mean = mean_pytrees(pathwise_shard_means)
    score_mean = _weighted_mean_pytrees(
        score_shard_means, score_finite_counts
    )
    observations = jp.concatenate(tuple(normalized_observations), axis=0)
    geometry = to_finite_json(
        {
            "per_shard": per_shard_geometry,
            "score_finite_count_by_shard": score_finite_counts,
            "cross_shard_pairwise_cosines": {
                "pathwise": pairwise_tree_cosines(
                    pathwise_shard_means, labels=shard_seeds
                ),
                "score": pairwise_tree_cosines(
                    score_shard_means, labels=shard_seeds
                ),
            },
            "bootstrap_direction_confidence": {
                "pathwise": bootstrap_direction_confidence(
                    pathwise_shard_means,
                    confidence_level=bootstrap_confidence_level,
                ),
                "score": bootstrap_direction_confidence(
                    score_shard_means,
                    confidence_level=bootstrap_confidence_level,
                    weights=score_finite_counts,
                ),
            },
            "cross_estimator": gradient_tree_geometry(
                pathwise_mean, score_mean
            ),
        }
    )
    return FourShardAggregation(
        shard_seeds=shard_seeds,
        pathwise_shard_means=pathwise_shard_means,
        score_shard_means=score_shard_means,
        pathwise_mean=pathwise_mean,
        score_mean=score_mean,
        normalized_observations=observations,
        geometry=geometry,
    )


def build_descent_candidates(
    *,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    actor_params: PyTree,
    pathwise_gradient: PyTree,
    score_gradient: PyTree,
    normalized_observations: jax.Array,
    target_rms: float,
    apply_step: Callable[..., Any] = apply_functional_actor_step,
) -> CandidateActors:
    """Build equal-functional-size candidates along both descent directions."""
    assert_matching_pytree_leaf_order(
        actor_params, pathwise_gradient, score_gradient
    )
    _require_finite_tree(pathwise_gradient, label="pathwise gradient")
    _require_finite_tree(score_gradient, label="score gradient")
    _flatten_observations(normalized_observations)
    pathwise_descent = jax.tree_util.tree_map(lambda value: -value, pathwise_gradient)
    score_descent = jax.tree_util.tree_map(lambda value: -value, score_gradient)
    pathwise_params, pathwise_summary = apply_step(
        actor_apply,
        actor_params,
        pathwise_descent,
        normalized_observations,
        target_rms=target_rms,
    )
    score_params, score_summary = apply_step(
        actor_apply,
        actor_params,
        score_descent,
        normalized_observations,
        target_rms=target_rms,
    )
    _require_finite_tree(pathwise_params, label="pathwise candidate")
    _require_finite_tree(score_params, label="score candidate")
    return CandidateActors(
        baseline=actor_params,
        pathwise=pathwise_params,
        score=score_params,
        functional_steps=to_finite_json(
            {"pathwise": pathwise_summary, "score": score_summary}
        ),
    )


def evaluate_held_out_candidates(
    *,
    candidates: CandidateActors,
    held_out_seeds: Sequence[int],
    phases: Sequence[int],
    evaluate_seed: Callable[..., Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Evaluate all three candidates together once for each of four CRN seeds."""
    held_out_seeds = tuple(int(seed) for seed in held_out_seeds)
    if held_out_seeds != _FIXED_HELD_OUT_SEEDS:
        raise ValueError(
            f"held-out seeds must equal {_FIXED_HELD_OUT_SEEDS}"
        )
    phases = tuple(int(phase) for phase in phases)
    if phases != _FIXED_PHASES:
        raise ValueError(f"phases must equal {_FIXED_PHASES}")
    candidate_mapping = {
        "baseline": candidates.baseline,
        "pathwise": candidates.pathwise,
        "score": candidates.score,
    }

    per_seed = []
    for seed in held_out_seeds:
        result = evaluate_seed(
            seed=seed,
            candidates=candidate_mapping,
            phases=phases,
        )
        if not isinstance(result, Mapping) or set(result) != set(_CANDIDATE_LABELS):
            raise ValueError("held-out evaluator must return all three candidates")
        normalized = {}
        for label in _CANDIDATE_LABELS:
            rows = tuple(result[label])
            if len(rows) != len(phases):
                raise ValueError(f"{label} must return one row per phase")
            normalized_rows = []
            for expected_phase, row in zip(phases, rows, strict=True):
                if not isinstance(row, Mapping):
                    raise ValueError("held-out rows must be mappings")
                required = {"phase", "return", "survival", "replay_free", "complete"}
                if not required.issubset(row):
                    raise ValueError(f"{label} held-out row is missing required fields")
                if int(row["phase"]) != expected_phase:
                    raise ValueError(f"{label} phases are not in the requested order")
                normalized_rows.append(to_finite_json(dict(row)))
            normalized[label] = normalized_rows
        per_seed.append({"seed": seed, "candidates": normalized})

    return to_finite_json(
        {
            "held_out_seeds": held_out_seeds,
            "phases": phases,
            "per_seed": per_seed,
        }
    )


def _invalid_outcome(reason: str) -> dict[str, Any]:
    return {
        "verdict": "invalid",
        "reason": reason,
        "decision_metrics": {},
    }


def _validate_thresholds(thresholds: OutcomeThresholds) -> None:
    values = dataclasses.asdict(thresholds)
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("outcome thresholds must be finite and nonnegative")
    if thresholds.maximum_phase_survival_loss_fraction > 1.0:
        raise ValueError("maximum phase survival loss fraction must not exceed one")
    if not 0.0 < thresholds.bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")


def classify_preregistered_outcome(
    *,
    geometry: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    thresholds: OutcomeThresholds,
    validity: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the explicit preregistered mechanism and validity decision gates."""
    _validate_thresholds(thresholds)
    missing_validity = sorted(_REQUIRED_VALIDITY_KEYS - set(validity))
    extra_validity = sorted(set(validity) - _REQUIRED_VALIDITY_KEYS)
    if missing_validity or extra_validity:
        return _invalid_outcome(
            "validity receipt key mismatch: "
            f"missing={missing_validity}, extra={extra_validity}"
        )
    failed_validity = sorted(
        key for key, valid in validity.items() if valid is not True
    )
    if failed_validity:
        return _invalid_outcome(
            "failed validity checks: " + ", ".join(failed_validity)
        )
    try:
        finite_geometry = to_finite_json(geometry)
        finite_evaluation = to_finite_json(evaluation)
    except (TypeError, ValueError) as error:
        return _invalid_outcome(str(error))

    rows_by_candidate = {label: [] for label in _CANDIDATE_LABELS}
    try:
        for seed_result in finite_evaluation["per_seed"]:
            candidates = seed_result["candidates"]
            if set(candidates) != set(_CANDIDATE_LABELS):
                return _invalid_outcome("evaluation candidate set is incomplete")
            for label in _CANDIDATE_LABELS:
                for row in candidates[label]:
                    if not row["complete"] or not row["replay_free"]:
                        return _invalid_outcome(
                            "held-out evaluation is incomplete or not replay-free"
                        )
                    rows_by_candidate[label].append(row)
        if any(not rows for rows in rows_by_candidate.values()):
            return _invalid_outcome("held-out evaluation has no rows")
        pathwise_stability = float(
            finite_geometry["cross_shard_pairwise_cosines"]["pathwise"]["mean"]
        )
        score_stability = float(
            finite_geometry["cross_shard_pairwise_cosines"]["score"]["mean"]
        )
        alignment = float(finite_geometry["cross_estimator"]["aggregate_cosine"])
    except (KeyError, TypeError, ValueError) as error:
        return _invalid_outcome(f"malformed decision evidence: {error}")

    try:
        confidence = finite_geometry["bootstrap_direction_confidence"]
    except (KeyError, TypeError):
        return {
            "verdict": "inconclusive",
            "reason": "bootstrap confidence evidence is missing or malformed",
            "decision_metrics": {},
        }
    try:
        pathwise_confidence = confidence["pathwise"]["cosine_to_full_mean"]
        score_confidence = confidence["score"]["cosine_to_full_mean"]
        for label in ("pathwise", "score"):
            receipt = confidence[label]
            if (
                receipt["method"] != "exhaustive-four-shard-percentile"
                or int(receipt["resample_count"]) != 256
                or float(receipt["confidence_level"])
                != thresholds.bootstrap_confidence_level
            ):
                return _invalid_outcome(
                    f"{label} bootstrap confidence receipt is not preregistered"
                )
        pathwise_lower = float(pathwise_confidence["lower"])
        pathwise_upper = float(pathwise_confidence["upper"])
        score_lower = float(score_confidence["lower"])
        score_upper = float(score_confidence["upper"])
        if pathwise_lower > pathwise_upper or score_lower > score_upper:
            return _invalid_outcome("bootstrap confidence interval bounds are reversed")
    except (KeyError, TypeError, ValueError) as error:
        return _invalid_outcome(f"malformed bootstrap confidence receipt: {error}")

    means = {
        label: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("return", "survival")
        }
        for label, rows in rows_by_candidate.items()
    }
    phases = sorted({row["phase"] for row in rows_by_candidate["baseline"]})
    phase_survival = {
        label: {
            str(phase): float(
                np.mean(
                    [
                        row["survival"]
                        for row in rows_by_candidate[label]
                        if row["phase"] == phase
                    ]
                )
            )
            for phase in phases
        }
        for label in _CANDIDATE_LABELS
    }
    score_phase_survival_gate = all(
        phase_survival["score"][str(phase)]
        >= phase_survival["baseline"][str(phase)]
        * (1.0 - thresholds.maximum_phase_survival_loss_fraction)
        for phase in phases
    )

    score_return_advantage = means["score"]["return"] - max(
        means["baseline"]["return"], means["pathwise"]["return"]
    )
    score_survival_advantage = means["score"]["survival"] - max(
        means["baseline"]["survival"], means["pathwise"]["survival"]
    )
    score_beats_both = (
        score_return_advantage >= thresholds.minimum_return_improvement
        or score_survival_advantage >= thresholds.minimum_survival_improvement
    )

    def improves_baseline(label: str) -> bool:
        return (
            means[label]["return"] - means["baseline"]["return"]
            >= thresholds.minimum_return_improvement
            or means[label]["survival"] - means["baseline"]["survival"]
            >= thresholds.minimum_survival_improvement
        )

    metrics = to_finite_json(
        {
            "pathwise_stability": pathwise_stability,
            "score_stability": score_stability,
            "pathwise_stability_confidence": {
                "lower": pathwise_lower,
                "upper": pathwise_upper,
            },
            "score_stability_confidence": {
                "lower": score_lower,
                "upper": score_upper,
            },
            "cross_estimator_alignment": alignment,
            "mean_by_candidate": means,
            "phase_survival_by_candidate": phase_survival,
            "score_return_advantage_over_both": score_return_advantage,
            "score_survival_advantage_over_both": score_survival_advantage,
            "score_phase_survival_gate": score_phase_survival_gate,
        }
    )

    if score_lower < thresholds.minimum_stability:
        verdict = "inconclusive"
        reason = "score estimator stability is below the preregistered minimum"
    elif (
        score_lower - pathwise_upper
        >= thresholds.material_stability_advantage
        and score_beats_both
        and score_phase_survival_gate
    ):
        verdict = "pathwise-quality-limited"
        reason = "score stability and equal-size transfer both clear their gates"
    elif (
        pathwise_lower >= thresholds.minimum_stability
        and score_lower >= thresholds.minimum_stability
        and alignment >= thresholds.minimum_alignment
        and not improves_baseline("pathwise")
        and not improves_baseline("score")
    ):
        verdict = "shared-objective-limited"
        reason = "both estimators are stable and aligned but neither transfers"
    elif (
        pathwise_lower >= thresholds.minimum_stability
        and pathwise_lower >= score_upper - thresholds.stability_tolerance
        and means["pathwise"]["return"]
        >= means["score"]["return"] - thresholds.return_tolerance
        and means["pathwise"]["survival"]
        >= means["score"]["survival"] - thresholds.survival_tolerance
    ):
        verdict = "pathwise-supported"
        reason = "pathwise stability and transfer are at least as strong as score"
    else:
        verdict = "inconclusive"
        reason = "the preregistered mechanism gates do not separate the estimators"
    return {
        "verdict": verdict,
        "reason": reason,
        "decision_metrics": metrics,
    }
