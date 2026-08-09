"""Pure helpers for the bounded E064 failure-weighted pathwise audit."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.batch_gradients import tree_cosine
from src.algorithms.shac.g1_gradient_audit_runner import (
    gradient_tree_geometry,
    mean_pytrees,
    pairwise_tree_cosines,
    to_finite_json,
)
from src.algorithms.shac.gradient_audit import (
    apply_functional_actor_step,
    assert_matching_pytree_leaf_order,
)

_POPULATION = 64
_PHASE_BIN_COUNT = 5
_FIXED_PHASES = (0, 100, 200, 300, 400)
_CANDIDATE_LABELS = ("baseline", "uniform", "tail")
_BASELINE_SURVIVAL_FLOORS = (110, 78, 74, 76, 58)
_RECEIPT_SUMMARY_ATOL = 128 * np.finfo(np.float64).eps
_REQUIRED_VALIDITY_KEYS = frozenset(
    {
        "frozen_hashes",
        "weight_receipts_exact",
        "uniform_reproduction_exact",
        "tail_reproduction_exact",
        "stability_evidence_exact",
        "aggregate_gradients_finite_nonzero",
        "candidate_trees_finite_nonzero",
        "functional_steps_valid",
        "rollouts_fresh_replay_free_complete_finite",
    }
)


class FailureWeightReceipt(NamedTuple):
    """Deterministic lower-tail membership and phase-preserving weights."""

    weights: jax.Array
    selected: jax.Array
    ranks: jax.Array
    phase_bins: jax.Array
    bin_counts: jax.Array
    tail_counts: jax.Array


@dataclass(frozen=True)
class FailureWeightedAggregation:
    """Uniform and lower-tail directions derived from identical four-shard data."""

    uniform_shard_means: tuple[Any, Any, Any, Any]
    tail_shard_means: tuple[Any, Any, Any, Any]
    uniform_mean: Any
    tail_mean: Any
    normalized_observations: jax.Array
    weighting_receipts: tuple[dict[str, Any], ...]
    geometry: dict[str, Any]


@dataclass(frozen=True)
class FailureWeightedCandidates:
    """Baseline and equal-functional-size uniform and lower-tail candidates."""

    baseline: Any
    uniform: Any
    tail: Any
    functional_steps: dict[str, Any]


def phase_preserving_lower_tail_weights(
    losses: jax.Array,
    initial_phases: jax.Array,
) -> FailureWeightReceipt:
    """Mix uniform mass with each phase bin's stable worst-return quartile."""

    losses = jp.asarray(losses)
    initial_phases = jp.asarray(initial_phases)
    if losses.shape != (_POPULATION,) or initial_phases.shape != (_POPULATION,):
        raise ValueError("losses and initial phases must each have shape (64,)")
    if not np.issubdtype(initial_phases.dtype, np.integer):
        raise TypeError("initial phases must have integer dtype")
    if not isinstance(losses, jax.core.Tracer) and not bool(
        jp.all(jp.isfinite(losses))
    ):
        raise ValueError("losses must be finite")
    if not isinstance(initial_phases, jax.core.Tracer) and not bool(
        jp.all((initial_phases >= 0) & (initial_phases < 500))
    ):
        raise ValueError("initial phases must be in range [0, 500)")

    losses = jax.lax.stop_gradient(losses.astype(jp.float64))
    phase_bins = initial_phases // 100
    indices = jp.arange(_POPULATION)
    same_bin = phase_bins[:, None] == phase_bins[None, :]
    strictly_worse = losses[None, :] > losses[:, None]
    stable_tie_before = (losses[None, :] == losses[:, None]) & (
        indices[None, :] < indices[:, None]
    )
    ranks = jp.sum(same_bin & (strictly_worse | stable_tie_before), axis=1)
    bin_counts = jp.bincount(phase_bins, length=_PHASE_BIN_COUNT)
    tail_counts = jp.where(bin_counts > 0, jp.maximum(1, bin_counts // 4), 0)
    selected = ranks < tail_counts[phase_bins]

    uniform = jp.asarray(0.5 / _POPULATION, dtype=jp.float64)
    tail = (
        jp.asarray(0.5 / _POPULATION, dtype=jp.float64)
        * bin_counts[phase_bins]
        / tail_counts[phase_bins]
    )
    weights = uniform + jp.where(selected, tail, 0.0)
    return FailureWeightReceipt(
        weights=weights,
        selected=selected,
        ranks=ranks,
        phase_bins=phase_bins,
        bin_counts=bin_counts,
        tail_counts=tail_counts,
    )


def weighted_mean_environment_gradients(
    per_environment_gradients,
    weights: jax.Array,
):
    """Average production-effective per-environment gradients with fixed weights."""

    weights = jp.asarray(weights, dtype=jp.float64)
    if weights.shape != (_POPULATION,):
        raise ValueError("gradient weights must have shape (64,)")
    if not bool(jp.all(jp.isfinite(weights))):
        raise ValueError("gradient weights must be finite")
    if not bool(jp.all(weights >= 0.0)):
        raise ValueError("gradient weights must be nonnegative")
    if not bool(jp.isclose(jp.sum(weights), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("gradient weights must sum to one")

    leaves = jax.tree_util.tree_leaves(per_environment_gradients)
    if not leaves:
        raise ValueError("per-environment gradients must not be empty")
    for leaf in leaves:
        if leaf.ndim < 1 or leaf.shape[0] != _POPULATION:
            raise ValueError("gradient leaves must have leading environment axis 64")
        if not bool(jp.all(jp.isfinite(leaf))):
            raise ValueError("per-environment gradients must be finite")

    return jax.tree_util.tree_map(
        lambda leaf: jp.tensordot(weights.astype(leaf.dtype), leaf, axes=((0,), (0,))),
        per_environment_gradients,
    )


def _uniform_mean_environment_gradients(per_environment_gradients):
    leaves = jax.tree_util.tree_leaves(per_environment_gradients)
    if not leaves or any(
        leaf.ndim < 1 or leaf.shape[0] != _POPULATION for leaf in leaves
    ):
        raise ValueError("pathwise effective gradient leaves need leading axis 64")
    if any(not bool(jp.all(jp.isfinite(leaf))) for leaf in leaves):
        raise ValueError("pathwise effective gradients must be finite")
    return jax.tree_util.tree_map(
        lambda leaf: jp.mean(leaf, axis=0), per_environment_gradients
    )


def _flatten_observations(observations: jax.Array) -> jax.Array:
    observations = jp.asarray(observations)
    if observations.ndim < 2 or observations.shape[0] != _POPULATION:
        raise ValueError("normalized observations need leading environment axis 64")
    if not bool(jp.all(jp.isfinite(observations))):
        raise ValueError("normalized observations must be finite")
    return observations.reshape((-1, observations.shape[-1]))


def _validated_candidate_observations(observations: jax.Array) -> jax.Array:
    observations = jp.asarray(observations)
    if observations.ndim != 2 or 0 in observations.shape:
        raise ValueError("candidate observations must be a nonempty matrix")
    if not bool(jp.all(jp.isfinite(observations))):
        raise ValueError("candidate observations must be finite")
    return observations


def _leave_one_out_cosines(
    shard_means: Sequence[Any], full_mean: Any
) -> dict[str, Any]:
    values = []
    for omitted in range(len(shard_means)):
        retained = tuple(
            value for index, value in enumerate(shard_means) if index != omitted
        )
        values.append(float(tree_cosine(mean_pytrees(retained), full_mean)))
    return {
        "values": values,
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def _weighting_receipt_json(
    *,
    seed: int,
    losses: jax.Array,
    phases: jax.Array,
    receipt: FailureWeightReceipt,
) -> dict[str, Any]:
    phase_mass = jp.asarray(
        [
            jp.sum(jp.where(receipt.phase_bins == phase_bin, receipt.weights, 0.0))
            for phase_bin in range(_PHASE_BIN_COUNT)
        ]
    )
    return to_finite_json(
        {
            "seed": seed,
            "losses": losses,
            "initial_phases": phases,
            "phase_bins": receipt.phase_bins,
            "ranks": receipt.ranks,
            "selected": receipt.selected,
            "weights": receipt.weights,
            "bin_counts": receipt.bin_counts,
            "tail_counts": receipt.tail_counts,
            "phase_weight_mass": phase_mass,
            "weight_sum": jp.sum(receipt.weights),
        }
    )


def _failure_weighted_geometry(
    uniform_shards: Sequence[Any],
    tail_shards: Sequence[Any],
) -> dict[str, Any]:
    uniform_shards = tuple(uniform_shards)
    tail_shards = tuple(tail_shards)
    uniform_mean = mean_pytrees(uniform_shards)
    tail_mean = mean_pytrees(tail_shards)
    return to_finite_json(
        {
            "cross_shard_pairwise_cosines": {
                "uniform": pairwise_tree_cosines(uniform_shards, labels=range(4)),
                "tail": pairwise_tree_cosines(tail_shards, labels=range(4)),
            },
            "leave_one_out_cosine_to_full": {
                "uniform": _leave_one_out_cosines(uniform_shards, uniform_mean),
                "tail": _leave_one_out_cosines(tail_shards, tail_mean),
            },
            "cross_direction": gradient_tree_geometry(uniform_mean, tail_mean),
        }
    )


def aggregate_failure_weighted_shards(
    results: Sequence[Any],
) -> FailureWeightedAggregation:
    """Aggregate four exact shared-rollout results under both actor reductions."""

    results = tuple(results)
    if len(results) != 4:
        raise ValueError("failure-weighted audit requires exactly four shards")

    uniform_shards = []
    tail_shards = []
    observations = []
    receipts = []
    for seed, result in enumerate(results):
        try:
            losses = result.losses
            phases = result.trajectory.initial_phase
            normalized = result.trajectory.normalized_observations
            gradients = result.pathwise_effective_gradients
        except AttributeError as error:
            raise ValueError("shard is missing pathwise gradient evidence") from error
        receipt = phase_preserving_lower_tail_weights(losses, phases)
        uniform_shards.append(_uniform_mean_environment_gradients(gradients))
        tail_shards.append(
            weighted_mean_environment_gradients(gradients, receipt.weights)
        )
        observations.append(_flatten_observations(normalized))
        receipts.append(
            _weighting_receipt_json(
                seed=seed,
                losses=losses,
                phases=phases,
                receipt=receipt,
            )
        )

    uniform_shards = tuple(uniform_shards)
    tail_shards = tuple(tail_shards)
    uniform_mean = mean_pytrees(uniform_shards)
    tail_mean = mean_pytrees(tail_shards)
    geometry = _failure_weighted_geometry(uniform_shards, tail_shards)
    return FailureWeightedAggregation(
        uniform_shard_means=uniform_shards,
        tail_shard_means=tail_shards,
        uniform_mean=uniform_mean,
        tail_mean=tail_mean,
        normalized_observations=jp.concatenate(tuple(observations), axis=0),
        weighting_receipts=tuple(receipts),
        geometry=geometry,
    )


def _assert_exact_tree(actual: Any, expected: Any, *, label: str) -> None:
    actual_leaves, actual_definition = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_definition = jax.tree_util.tree_flatten(expected)
    if actual_definition != expected_definition:
        raise ValueError(f"{label} pytree structure mismatch")
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array = np.asarray(jax.device_get(actual_leaf))
        expected_array = np.asarray(jax.device_get(expected_leaf))
        if (
            actual_array.dtype != expected_array.dtype
            or actual_array.shape != expected_array.shape
            or not np.array_equal(actual_array, expected_array)
        ):
            raise ValueError(f"{label} is not reproduced exactly")


def _tree_scaled_tolerance_error(
    actual: Any,
    expected: Any,
    *,
    tolerance_units: int,
    label: str,
) -> float:
    actual_leaves, actual_definition = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_definition = jax.tree_util.tree_flatten(expected)
    if actual_definition != expected_definition:
        raise ValueError(f"{label} pytree structure mismatch")
    maximum_error = 0.0
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array = np.asarray(jax.device_get(actual_leaf))
        expected_array = np.asarray(jax.device_get(expected_leaf))
        if (
            actual_array.dtype != expected_array.dtype
            or actual_array.shape != expected_array.shape
        ):
            raise ValueError(f"{label} dtype or shape mismatch")
        if not np.issubdtype(actual_array.dtype, np.floating):
            raise TypeError(f"{label} leaves must be floating point")
        errors = np.abs(
            actual_array.astype(np.float64) - expected_array.astype(np.float64)
        )
        scales = np.maximum(1.0, np.abs(expected_array.astype(np.float64)))
        tolerance = tolerance_units * np.finfo(actual_array.dtype).eps * scales
        if not np.all(errors <= tolerance):
            raise ValueError(f"{label} exceeds frozen scaled numerical tolerance")
        maximum_error = max(maximum_error, float(np.max(errors, initial=0.0)))
    return maximum_error


def _independent_host_weighting_receipt(
    *,
    seed: int,
    losses: Any,
    phases: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Recompute stable lower-tail membership without JAX producer helpers."""

    losses_host = np.asarray(jax.device_get(losses))
    phases_host = np.asarray(jax.device_get(phases))
    if losses_host.shape != (_POPULATION,) or phases_host.shape != (_POPULATION,):
        raise ValueError("independent host receipt requires 64 losses and phases")
    if not np.issubdtype(phases_host.dtype, np.integer):
        raise TypeError("independent host phases must be integers")
    if not np.all(np.isfinite(losses_host)):
        raise ValueError("independent host losses must be finite")
    if not np.all((phases_host >= 0) & (phases_host < 500)):
        raise ValueError("independent host phases must be in range [0, 500)")

    phase_bins = phases_host // 100
    bin_counts = np.bincount(phase_bins, minlength=_PHASE_BIN_COUNT)
    tail_counts = np.where(bin_counts > 0, np.maximum(1, bin_counts // 4), 0)
    ranks = np.empty((_POPULATION,), dtype=np.int64)
    selected = np.zeros((_POPULATION,), dtype=np.bool_)
    for phase_bin in range(_PHASE_BIN_COUNT):
        indices = np.flatnonzero(phase_bins == phase_bin)
        order = indices[np.lexsort((indices, -losses_host[indices]))]
        ranks[order] = np.arange(order.size, dtype=np.int64)
        selected[order[: tail_counts[phase_bin]]] = True

    weights = np.full((_POPULATION,), 0.5 / _POPULATION, dtype=np.float64)
    selected_indices = np.flatnonzero(selected)
    weights[selected_indices] += (
        0.5
        / _POPULATION
        * bin_counts[phase_bins[selected_indices]]
        / tail_counts[phase_bins[selected_indices]]
    )
    phase_mass = np.asarray(
        [
            weights[phase_bins == phase_bin].sum()
            for phase_bin in range(_PHASE_BIN_COUNT)
        ]
    )
    document = {
        "seed": seed,
        "losses": losses_host.tolist(),
        "initial_phases": phases_host.tolist(),
        "phase_bins": phase_bins.tolist(),
        "ranks": ranks.tolist(),
        "selected": selected.tolist(),
        "weights": weights.tolist(),
        "bin_counts": bin_counts.tolist(),
        "tail_counts": tail_counts.tolist(),
        "phase_weight_mass": phase_mass.tolist(),
        "weight_sum": float(weights.sum()),
    }
    return document, weights


def _independent_host_aggregate(
    results: Sequence[Any],
    weights_by_shard: Sequence[np.ndarray] | None,
) -> Any:
    """Reduce shard leaves on the host while retaining each source leaf dtype."""

    tree_definition = None
    per_leaf_shards: list[list[np.ndarray]] | None = None
    for shard_index, result in enumerate(results):
        leaves, definition = jax.tree_util.tree_flatten(
            result.pathwise_effective_gradients
        )
        if tree_definition is None:
            tree_definition = definition
            per_leaf_shards = [[] for _ in leaves]
        elif definition != tree_definition or len(leaves) != len(per_leaf_shards or ()):
            raise ValueError("independent host gradient pytrees do not match")
        for leaf_index, leaf in enumerate(leaves):
            leaf_host = np.asarray(jax.device_get(leaf))
            if leaf_host.ndim < 1 or leaf_host.shape[0] != _POPULATION:
                raise ValueError("independent host gradients need leading axis 64")
            if not np.issubdtype(leaf_host.dtype, np.floating):
                raise TypeError(
                    "independent host gradient leaves must be floating point"
                )
            if not np.all(np.isfinite(leaf_host)):
                raise ValueError("independent host gradients must be finite")
            if weights_by_shard is None:
                shard_mean = np.mean(leaf_host, axis=0, dtype=leaf_host.dtype)
            else:
                weights = np.asarray(weights_by_shard[shard_index]).astype(
                    leaf_host.dtype, copy=False
                )
                shard_mean = np.tensordot(weights, leaf_host, axes=((0,), (0,)))
            per_leaf_shards[leaf_index].append(shard_mean)

    if tree_definition is None or per_leaf_shards is None:
        raise ValueError("independent host aggregation needs nonempty results")
    aggregate_leaves = []
    for shard_leaves in per_leaf_shards:
        dtype = shard_leaves[0].dtype
        aggregate_leaves.append(np.mean(np.stack(shard_leaves), axis=0, dtype=dtype))
    return jax.tree_util.tree_unflatten(tree_definition, aggregate_leaves)


def _validate_published_weighting_receipt(
    published: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[float, float]:
    """Keep causal arrays exact while admitting only reduction-order roundoff."""

    published = to_finite_json(published)
    if set(published) != set(expected):
        raise ValueError(f"weight receipt {seed} differs from independent host key set")
    summary_fields = {"phase_weight_mass", "weight_sum"}
    for field in sorted(set(expected) - summary_fields):
        if published[field] != expected[field]:
            raise ValueError(
                f"weight receipt {seed} field {field} differs from independent host"
            )

    bin_counts = np.asarray(expected["bin_counts"], dtype=np.float64)
    canonical_phase_mass = bin_counts / _POPULATION
    phase_mass = np.asarray(published["phase_weight_mass"], dtype=np.float64)
    if phase_mass.shape != (_PHASE_BIN_COUNT,) or not np.all(np.isfinite(phase_mass)):
        raise ValueError(f"weight receipt {seed} phase-mass summary is malformed")
    phase_errors = np.abs(phase_mass - canonical_phase_mass)
    if not np.all(phase_errors <= _RECEIPT_SUMMARY_ATOL):
        raise ValueError(
            f"weight receipt {seed} phase-mass summary exceeds frozen tolerance"
        )

    weight_sum = float(published["weight_sum"])
    weight_sum_error = abs(weight_sum - 1.0)
    if not np.isfinite(weight_sum) or weight_sum_error > _RECEIPT_SUMMARY_ATOL:
        raise ValueError(
            f"weight receipt {seed} total-mass summary exceeds frozen tolerance"
        )
    return float(np.max(phase_errors, initial=0.0)), weight_sum_error


def _require_bounded_cosine(value: Any, *, label: str) -> float:
    cosine = float(value)
    if not np.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
        raise ValueError(f"{label} cosine must be finite and in [-1, 1]")
    return cosine


def _validate_cosine_summary(
    receipt: Mapping[str, Any],
    values: Sequence[float],
    *,
    label: str,
) -> None:
    values = tuple(values)
    if not values:
        raise ValueError(f"{label} cosine summary must not be empty")
    expected = {
        "minimum": min(values),
        "mean": float(np.mean(values)),
        "maximum": max(values),
    }
    for field, expected_value in expected.items():
        actual = _require_bounded_cosine(receipt[field], label=f"{label} {field}")
        if abs(actual - expected_value) > _RECEIPT_SUMMARY_ATOL:
            raise ValueError(f"{label} cosine {field} summary does not match values")


def _validate_stability_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every stored stability cosine, pair, and derived summary."""

    geometry = to_finite_json(geometry)
    if set(geometry) != {
        "cross_shard_pairwise_cosines",
        "leave_one_out_cosine_to_full",
        "cross_direction",
    }:
        raise ValueError("stability geometry key set is invalid")

    expected_pairs = [
        (left, right) for left in range(4) for right in range(left + 1, 4)
    ]
    pairwise = geometry["cross_shard_pairwise_cosines"]
    if not isinstance(pairwise, Mapping) or set(pairwise) != {"uniform", "tail"}:
        raise ValueError("pairwise stability geometry is malformed")
    for label in ("uniform", "tail"):
        receipt = pairwise[label]
        if not isinstance(receipt, Mapping) or int(receipt.get("count", -1)) != 6:
            raise ValueError(f"{label} pairwise cosine count must equal six")
        pairs = receipt.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 6:
            raise ValueError(f"{label} pairwise cosine pairs must contain six rows")
        values = []
        observed_pairs = []
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise TypeError(f"{label} pairwise cosine row is malformed")
            observed_pairs.append((int(pair["left"]), int(pair["right"])))
            values.append(
                _require_bounded_cosine(pair["cosine"], label=f"{label} pairwise")
            )
        if observed_pairs != expected_pairs:
            raise ValueError(f"{label} pairwise cosine labels are invalid")
        _validate_cosine_summary(receipt, values, label=f"{label} pairwise")

    leave_one_out = geometry["leave_one_out_cosine_to_full"]
    if not isinstance(leave_one_out, Mapping) or set(leave_one_out) != {
        "uniform",
        "tail",
    }:
        raise ValueError("leave-one-out stability geometry is malformed")
    for label in ("uniform", "tail"):
        receipt = leave_one_out[label]
        if not isinstance(receipt, Mapping):
            raise TypeError(f"{label} leave-one-out cosine receipt is malformed")
        raw_values = receipt.get("values")
        if not isinstance(raw_values, list) or len(raw_values) != 4:
            raise ValueError(f"{label} leave-one-out cosines must contain four values")
        values = [
            _require_bounded_cosine(value, label=f"{label} leave-one-out")
            for value in raw_values
        ]
        _validate_cosine_summary(receipt, values, label=f"{label} leave-one-out")

    cross_direction = geometry["cross_direction"]
    if not isinstance(cross_direction, Mapping):
        raise TypeError("cross-direction stability geometry is malformed")
    _require_bounded_cosine(
        cross_direction["aggregate_cosine"], label="cross-direction aggregate"
    )
    sign_agreement = float(cross_direction["aggregate_sign_agreement_fraction"])
    if not np.isfinite(sign_agreement) or not 0.0 <= sign_agreement <= 1.0:
        raise ValueError("cross-direction aggregate sign agreement is invalid")
    for norm_label in ("left_norm", "right_norm"):
        norm = float(cross_direction[norm_label])
        if not np.isfinite(norm) or norm < 0.0:
            raise ValueError(f"cross-direction {norm_label} is invalid")
    layers = cross_direction.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("cross-direction layer geometry must not be empty")
    for layer in layers:
        if not isinstance(layer, Mapping) or int(layer.get("size", 0)) <= 0:
            raise ValueError("cross-direction layer geometry is malformed")
        _require_bounded_cosine(layer["cosine"], label="cross-direction layer")
        layer_sign = float(layer["sign_agreement_fraction"])
        if not np.isfinite(layer_sign) or not 0.0 <= layer_sign <= 1.0:
            raise ValueError("cross-direction layer sign agreement is invalid")
    return geometry


def validate_failure_weighted_aggregation(
    results: Sequence[Any],
    aggregation: FailureWeightedAggregation,
    *,
    host_recomputation_tolerance_units: int = 256,
) -> dict[str, Any]:
    """Independently replay published weights and both aggregate reductions."""

    results = tuple(results)
    if len(results) != 4 or len(aggregation.weighting_receipts) != 4:
        raise ValueError("aggregation validation requires four shards and receipts")
    if host_recomputation_tolerance_units < 0:
        raise ValueError("host recomputation tolerance units must be nonnegative")

    published_weights_by_shard = []
    phase_mass_errors = []
    weight_sum_errors = []
    for seed, (result, published) in enumerate(
        zip(results, aggregation.weighting_receipts, strict=True)
    ):
        expected_document, expected_weights = _independent_host_weighting_receipt(
            seed=seed,
            losses=result.losses,
            phases=result.trajectory.initial_phase,
        )
        phase_mass_error, weight_sum_error = _validate_published_weighting_receipt(
            published,
            expected_document,
            seed=seed,
        )
        phase_mass_errors.append(phase_mass_error)
        weight_sum_errors.append(weight_sum_error)
        published_weights_by_shard.append(expected_weights)

    producer_uniform_shards = tuple(
        _uniform_mean_environment_gradients(result.pathwise_effective_gradients)
        for result in results
    )
    producer_tail_shards = tuple(
        weighted_mean_environment_gradients(
            result.pathwise_effective_gradients,
            jp.asarray(weights, dtype=jp.float64),
        )
        for result, weights in zip(results, published_weights_by_shard, strict=True)
    )
    for seed, (published_uniform, replayed_uniform) in enumerate(
        zip(
            aggregation.uniform_shard_means,
            producer_uniform_shards,
            strict=True,
        )
    ):
        _assert_exact_tree(
            published_uniform,
            replayed_uniform,
            label=f"uniform shard {seed}",
        )
    for seed, (published_tail, replayed_tail) in enumerate(
        zip(aggregation.tail_shard_means, producer_tail_shards, strict=True)
    ):
        _assert_exact_tree(
            published_tail,
            replayed_tail,
            label=f"tail shard {seed}",
        )

    producer_uniform = mean_pytrees(producer_uniform_shards)
    producer_tail = mean_pytrees(producer_tail_shards)
    _assert_exact_tree(
        aggregation.uniform_mean,
        producer_uniform,
        label="uniform aggregate",
    )
    _assert_exact_tree(
        aggregation.tail_mean,
        producer_tail,
        label="exact producer tail aggregate",
    )
    published_geometry = _validate_stability_geometry(aggregation.geometry)
    expected_geometry = _failure_weighted_geometry(
        producer_uniform_shards, producer_tail_shards
    )
    if published_geometry != expected_geometry:
        raise ValueError("stability geometry does not match verified shard directions")
    host_uniform = _independent_host_aggregate(results, None)
    uniform_host_error = _tree_scaled_tolerance_error(
        aggregation.uniform_mean,
        host_uniform,
        tolerance_units=host_recomputation_tolerance_units,
        label="independent host uniform aggregate",
    )
    host_tail = _independent_host_aggregate(results, published_weights_by_shard)
    tail_error = _tree_scaled_tolerance_error(
        aggregation.tail_mean,
        host_tail,
        tolerance_units=host_recomputation_tolerance_units,
        label="independent host tail aggregate",
    )
    return {
        "weight_receipts_exact": True,
        "uniform_reproduction_exact": True,
        "tail_reproduction_exact": True,
        "stability_evidence_exact": True,
        "receipt_summary_absolute_tolerance": _RECEIPT_SUMMARY_ATOL,
        "phase_mass_summary_maximum_absolute_error": max(phase_mass_errors),
        "weight_sum_summary_maximum_absolute_error": max(weight_sum_errors),
        "independent_host_recomputation": {
            "tolerance_units": host_recomputation_tolerance_units,
            "tolerance_model": (
                "absolute_error <= tolerance_units * dtype_epsilon * "
                "max(1, abs(expected))"
            ),
            "uniform_maximum_absolute_error": uniform_host_error,
            "tail_maximum_absolute_error": tail_error,
        },
    }


def build_failure_weighted_candidates(
    *,
    actor_apply: Callable[[Any, jax.Array], jax.Array],
    actor_params: Any,
    uniform_gradient: Any,
    tail_gradient: Any,
    normalized_observations: jax.Array,
    target_rms: float,
    apply_step: Callable[..., Any] = apply_functional_actor_step,
) -> FailureWeightedCandidates:
    """Build equal-size descent candidates and measure exact output separation."""

    assert_matching_pytree_leaf_order(actor_params, uniform_gradient, tail_gradient)
    parameter_dtypes = tuple(
        leaf.dtype for leaf in jax.tree_util.tree_leaves(actor_params)
    )
    for label, gradient in (
        ("uniform", uniform_gradient),
        ("tail", tail_gradient),
    ):
        gradient_dtypes = tuple(
            leaf.dtype for leaf in jax.tree_util.tree_leaves(gradient)
        )
        if gradient_dtypes != parameter_dtypes:
            raise ValueError(
                f"{label} gradient leaf dtypes must match actor parameters"
            )
    observations = _validated_candidate_observations(normalized_observations)
    uniform_descent = jax.tree_util.tree_map(lambda value: -value, uniform_gradient)
    tail_descent = jax.tree_util.tree_map(lambda value: -value, tail_gradient)
    uniform_params, uniform_summary = apply_step(
        actor_apply,
        actor_params,
        uniform_descent,
        observations,
        target_rms=target_rms,
    )
    tail_params, tail_summary = apply_step(
        actor_apply,
        actor_params,
        tail_descent,
        observations,
        target_rms=target_rms,
    )
    for label, candidate in (("uniform", uniform_params), ("tail", tail_params)):
        candidate_dtypes = tuple(
            leaf.dtype for leaf in jax.tree_util.tree_leaves(candidate)
        )
        if candidate_dtypes != parameter_dtypes:
            raise ValueError(
                f"{label} candidate leaf dtypes must match actor parameters"
            )
    uniform_output = actor_apply(uniform_params, observations)
    tail_output = actor_apply(tail_params, observations)
    separation = jp.sqrt(jp.mean(jp.square(tail_output - uniform_output)))
    if not bool(jp.isfinite(separation)):
        raise ValueError("candidate output separation must be finite")
    return FailureWeightedCandidates(
        baseline=actor_params,
        uniform=uniform_params,
        tail=tail_params,
        functional_steps=to_finite_json(
            {
                "uniform": uniform_summary,
                "tail": tail_summary,
                "candidate_output_rms": {"tail_vs_uniform": separation},
            }
        ),
    )


def _invalid_outcome(reason: str) -> dict[str, Any]:
    return {"verdict": "invalid", "reason": reason, "decision_metrics": {}}


def _validated_decision_rows(
    evaluation: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if evaluation.get("mode") != "single-deterministic-five-phase-grid":
        raise ValueError("ordinary evaluation mode is not preregistered")
    if int(evaluation.get("seed", -1)) != 0:
        raise ValueError("ordinary evaluation seed must equal zero")
    if tuple(evaluation.get("phases", ())) != _FIXED_PHASES:
        raise ValueError("ordinary evaluation phases are not preregistered")
    per_seed = evaluation.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != 1:
        raise ValueError("ordinary evaluation must contain one seed-zero grid")
    if not isinstance(per_seed[0], Mapping):
        raise TypeError("ordinary evaluation seed row must be a mapping")
    if int(per_seed[0].get("seed", -1)) != 0:
        raise ValueError("ordinary evaluation grid seed must equal zero")
    candidates = per_seed[0].get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != set(_CANDIDATE_LABELS):
        raise ValueError("evaluation candidate set is incomplete")

    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for label in _CANDIDATE_LABELS:
        rows = candidates[label]
        if not isinstance(rows, list) or len(rows) != len(_FIXED_PHASES):
            raise ValueError("ordinary evaluation needs five rows per candidate")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("ordinary evaluation rows must be mappings")
        if tuple(int(row.get("phase", -1)) for row in rows) != _FIXED_PHASES:
            raise ValueError("ordinary evaluation candidate phases are invalid")
        validated = []
        for row in rows:
            if row.get("complete") is not True or row.get("replay_free") is not True:
                raise ValueError(
                    "ordinary evaluation rows must be complete and replay-free"
                )
            return_value = float(row["return"])
            survival = float(row["survival"])
            if not np.isfinite(return_value) or not np.isfinite(survival):
                raise ValueError("ordinary evaluation contains nonfinite metrics")
            validated.append(
                {
                    "phase": int(row["phase"]),
                    "return": return_value,
                    "survival": survival,
                }
            )
        rows_by_candidate[label] = validated
    return rows_by_candidate


def classify_failure_weighted_outcome(
    *,
    geometry: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    functional_steps: Mapping[str, Any],
    validity: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the frozen failure-aware stability, safety, and transfer gates."""

    if not isinstance(validity, Mapping):
        return _invalid_outcome("validity receipt must be a mapping")
    missing = sorted(_REQUIRED_VALIDITY_KEYS - set(validity))
    extra = sorted(set(validity) - _REQUIRED_VALIDITY_KEYS)
    if missing or extra:
        return _invalid_outcome(
            f"validity receipt key mismatch: missing={missing}, extra={extra}"
        )
    failed = sorted(key for key, value in validity.items() if value is not True)
    if failed:
        return _invalid_outcome("failed validity checks: " + ", ".join(failed))
    for label, evidence in (
        ("geometry", geometry),
        ("evaluation", evaluation),
        ("functional steps", functional_steps),
    ):
        if not isinstance(evidence, Mapping):
            return _invalid_outcome(f"{label} evidence must be a mapping")

    try:
        geometry = to_finite_json(geometry)
        evaluation = to_finite_json(evaluation)
        functional_steps = to_finite_json(functional_steps)
        rows = _validated_decision_rows(evaluation)
        output_rms = {
            label: float(functional_steps[label]["output_rms"])
            for label in ("uniform", "tail")
        }
        separation = float(functional_steps["candidate_output_rms"]["tail_vs_uniform"])
        pairwise = geometry["cross_shard_pairwise_cosines"]
        leave_one_out = geometry["leave_one_out_cosine_to_full"]
        uniform_pair_mean = float(pairwise["uniform"]["mean"])
        uniform_pair_min = float(pairwise["uniform"]["minimum"])
        tail_pair_mean = float(pairwise["tail"]["mean"])
        tail_pair_min = float(pairwise["tail"]["minimum"])
        uniform_loo_min = float(leave_one_out["uniform"]["minimum"])
        tail_loo_min = float(leave_one_out["tail"]["minimum"])
        for label, cosine in (
            ("uniform pairwise mean", uniform_pair_mean),
            ("uniform pairwise minimum", uniform_pair_min),
            ("tail pairwise mean", tail_pair_mean),
            ("tail pairwise minimum", tail_pair_min),
            ("uniform leave-one-out minimum", uniform_loo_min),
            ("tail leave-one-out minimum", tail_loo_min),
        ):
            _require_bounded_cosine(cosine, label=label)
        for direction in ("uniform", "tail"):
            pairs = pairwise[direction].get("pairs", [])
            if not isinstance(pairs, list):
                raise TypeError(f"{direction} pairwise cosine rows are malformed")
            for pair in pairs:
                if not isinstance(pair, Mapping):
                    raise TypeError(f"{direction} pairwise cosine row is malformed")
                _require_bounded_cosine(pair["cosine"], label=f"{direction} pairwise")
            loo_values = leave_one_out[direction].get("values", [])
            if not isinstance(loo_values, list):
                raise TypeError(f"{direction} leave-one-out cosines are malformed")
            for cosine in loo_values:
                _require_bounded_cosine(cosine, label=f"{direction} leave-one-out")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return _invalid_outcome(f"malformed or nonfinite decision evidence: {error}")

    calibration_tolerance = 0.01 * 5e-5
    if any(abs(value - 0.01) > calibration_tolerance for value in output_rms.values()):
        return _invalid_outcome("functional candidate output RMS is not calibrated")

    baseline_survival = tuple(row["survival"] for row in rows["baseline"])
    if any(
        actual < floor
        for actual, floor in zip(
            baseline_survival, _BASELINE_SURVIVAL_FLOORS, strict=True
        )
    ):
        return _invalid_outcome("baseline survival is below an E066 competence floor")

    tolerance = 0.05
    boundary_epsilon = 1e-12
    stability_checks = {
        "pairwise_mean": uniform_pair_mean - tail_pair_mean
        <= tolerance + boundary_epsilon,
        "pairwise_minimum": uniform_pair_min - tail_pair_min
        <= tolerance + boundary_epsilon,
        "leave_one_out_minimum": uniform_loo_min - tail_loo_min
        <= tolerance + boundary_epsilon,
    }
    means = {
        label: {
            metric: float(np.mean([row[metric] for row in candidate_rows]))
            for metric in ("return", "survival")
        }
        for label, candidate_rows in rows.items()
    }
    phase_safety = all(
        tail["survival"] >= 0.90 * baseline["survival"]
        for baseline, tail in zip(rows["baseline"], rows["tail"], strict=True)
    )
    phase_wins = [
        tail["return"] - uniform["return"] >= 0.001
        or tail["survival"] - uniform["survival"] >= 5.0
        for uniform, tail in zip(rows["uniform"], rows["tail"], strict=True)
    ]
    mean_gain_over_both = (
        means["tail"]["return"]
        - max(means["baseline"]["return"], means["uniform"]["return"])
        >= 0.001
        or means["tail"]["survival"]
        - max(means["baseline"]["survival"], means["uniform"]["survival"])
        >= 5.0
    )
    metrics = to_finite_json(
        {
            "stability_checks": stability_checks,
            "pairwise_mean": {"uniform": uniform_pair_mean, "tail": tail_pair_mean},
            "pairwise_minimum": {"uniform": uniform_pair_min, "tail": tail_pair_min},
            "leave_one_out_minimum": {
                "uniform": uniform_loo_min,
                "tail": tail_loo_min,
            },
            "mean_by_candidate": means,
            "phase_survival_gate": phase_safety,
            "material_phase_wins_over_uniform": sum(phase_wins),
            "mean_gain_over_baseline_and_uniform": mean_gain_over_both,
            "tail_vs_uniform_output_rms": separation,
        }
    )
    if not all(stability_checks.values()):
        verdict = "failure-aware-unstable"
        reason = "lower-tail direction regresses a frozen stability gate"
    elif mean_gain_over_both and phase_safety and sum(phase_wins) >= 2:
        verdict = "failure-aware-supported"
        reason = "lower-tail weighting clears stability, safety, and transfer gates"
    else:
        verdict = "failure-aware-not-material"
        reason = "valid lower-tail candidate does not clear every materiality gate"
    return {"verdict": verdict, "reason": reason, "decision_metrics": metrics}
