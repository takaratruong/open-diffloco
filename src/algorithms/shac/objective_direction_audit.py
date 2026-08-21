"""Pure geometry and selection contracts for frozen objective-direction audits."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.cagrad import (
    accumulate_phase_gradients,
    combine_cagrad,
    finalize_phase_gradients,
)

PyTree = Any
ORDINARY_BASELINE_SURVIVAL = (116, 63, 49, 39, 47)


class AuditDirection(NamedTuple):
    """One training-effective direction and its phase diagnostics."""

    combined_gradient: PyTree
    task_gradients: PyTree
    env_counts: jp.ndarray
    gram_matrix: jp.ndarray
    cosine_matrix: jp.ndarray
    weights: jp.ndarray
    valid: bool


def _matching_numeric_leaves(first: PyTree, second: PyTree):
    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(second):
        raise ValueError("gradient PyTrees must have identical structure")
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    if not first_leaves:
        raise ValueError("numeric PyTree leaves are required")
    for left, right in zip(first_leaves, second_leaves, strict=True):
        if left.shape != right.shape:
            raise ValueError("gradient PyTree leaves must have identical shapes")
    return first_leaves, second_leaves


def _require_finite_tree(tree: PyTree) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves or any(not np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise ValueError("gradient PyTree must be finite and nonempty")


def tree_dot(first: PyTree, second: PyTree):
    """Return the Euclidean dot product over matching PyTree leaves."""
    first_leaves, second_leaves = _matching_numeric_leaves(first, second)
    return sum(
        jp.vdot(left.reshape(-1), right.reshape(-1))
        for left, right in zip(first_leaves, second_leaves, strict=True)
    )


def tree_l2_norm(tree: PyTree):
    """Return the Euclidean norm over all PyTree leaves."""
    _require_finite_tree(tree)
    return jp.sqrt(jp.maximum(tree_dot(tree, tree), 0.0))


def tree_cosine(first: PyTree, second: PyTree):
    """Return a finite cosine, rejecting either zero direction."""
    _require_finite_tree(first)
    _require_finite_tree(second)
    first_norm = tree_l2_norm(first)
    second_norm = tree_l2_norm(second)
    if float(first_norm) <= 0.0 or float(second_norm) <= 0.0:
        raise ValueError("cosine requires two nonzero directions")
    return tree_dot(first, second) / (first_norm * second_norm)


def normalized_descent_proposal(
    params: PyTree, gradient: PyTree, *, displacement: float
) -> PyTree:
    """Apply a raw-gradient descent proposal with an exact global L2 norm."""
    if not np.isfinite(displacement) or displacement <= 0.0:
        raise ValueError("proposal displacement must be finite and positive")
    _require_finite_tree(params)
    _require_finite_tree(gradient)
    _matching_numeric_leaves(params, gradient)
    norm = tree_l2_norm(gradient)
    if float(norm) <= 0.0:
        raise ValueError("proposal gradient must be nonzero")
    scale = jp.asarray(displacement, dtype=norm.dtype) / norm
    return jax.tree_util.tree_map(
        lambda parameter, grad: parameter - scale * grad, params, gradient
    )


def aggregate_audit_direction(
    per_env_gradients: PyTree,
    phases,
    *,
    phase_count: int,
    clip_norm: float,
    alpha: float,
    iterations: int,
) -> AuditDirection:
    """Clip per environment, average by phase, and form the CAGrad direction."""
    accumulator = accumulate_phase_gradients(
        per_env_gradients,
        jp.asarray(phases, dtype=jp.int32),
        phase_count=phase_count,
        bin_count=5,
        per_env_max_norm=clip_norm,
    )
    task_gradients, env_counts, phase_valid = finalize_phase_gradients(accumulator)
    result = combine_cagrad(
        task_gradients, alpha=alpha, iterations=iterations
    )
    valid = bool(np.asarray(phase_valid & result.valid))
    if not valid:
        raise ValueError("objective direction aggregation is invalid")
    _require_finite_tree(result.combined_gradient)
    if float(tree_l2_norm(result.combined_gradient)) <= 0.0:
        raise ValueError("objective direction aggregation is zero")
    return AuditDirection(
        combined_gradient=result.combined_gradient,
        task_gradients=task_gradients,
        env_counts=env_counts,
        gram_matrix=result.gram_matrix,
        cosine_matrix=result.cosine_matrix,
        weights=result.weights,
        valid=True,
    )


def _survival_counts(
    candidate: Sequence[int], baseline: Sequence[int]
) -> tuple[int, int, int, int]:
    candidate_array = np.asarray(candidate, dtype=np.int64)
    baseline_array = np.asarray(baseline, dtype=np.int64)
    if (
        candidate_array.ndim != 1
        or candidate_array.shape != baseline_array.shape
        or candidate_array.size == 0
        or np.any(candidate_array < 0)
        or np.any(baseline_array < 0)
    ):
        raise ValueError("survival vectors must be matching nonnegative vectors")
    improved = int(np.sum(candidate_array > baseline_array))
    unchanged = int(np.sum(candidate_array == baseline_array))
    regressed = int(np.sum(candidate_array < baseline_array))
    gain = int(np.sum(candidate_array - baseline_array))
    return improved, unchanged, regressed, gain


def select_carried_safe_candidate(
    rows: Sequence[Mapping[str, object]], *, baseline_survival: Sequence[int]
) -> dict[str, object] | None:
    """Select one componentwise-safe proposal using the registered ordering."""
    eligible: list[dict[str, object]] = []
    for row in rows:
        multiplier = float(row["multiplier"])
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("proposal multipliers must be finite and positive")
        candidate = list(row["candidate_survival"])  # type: ignore[arg-type]
        improved, unchanged, regressed, gain = _survival_counts(
            candidate, baseline_survival
        )
        decorated = {
            **row,
            "candidate_survival": candidate,
            "improved_count": improved,
            "unchanged_count": unchanged,
            "regressed_count": regressed,
            "total_survival_gain": gain,
        }
        if regressed == 0 and improved > 0:
            eligible.append(decorated)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            int(row["improved_count"]),
            int(row["unchanged_count"]),
            int(row["total_survival_gain"]),
            -float(row["multiplier"]),
        ),
    )


def ordinary_componentwise_safe(
    candidate: Sequence[int],
    baseline: Sequence[int] = ORDINARY_BASELINE_SURVIVAL,
) -> bool:
    """Return whether all ordinary suffix counts preserve the fixed baseline."""
    _, _, regressed, _ = _survival_counts(candidate, baseline)
    return regressed == 0


def classify_objective_direction_audit(
    *,
    execution_valid: bool,
    h24_tape_cosine: float,
    h24_tape_phase_cosines: Sequence[float],
    h24_h48_cosine: float,
    h24_h48_phase_cosines: Sequence[float],
    aggregate_pairwise_cosines: Sequence[float],
    full_gate_by_direction: Mapping[str, bool],
    h48_carried_strictly_better: bool,
    every_direction_mixed: bool,
) -> str:
    """Apply the preregistered ordered outcome map."""
    if not execution_valid:
        return "invalid-execution"
    required_directions = {"h24", "h48", "bootstrap"}
    if set(full_gate_by_direction) != required_directions:
        raise ValueError("full-gate mapping must contain all three directions")
    scalars = [
        h24_tape_cosine,
        h24_h48_cosine,
        *h24_tape_phase_cosines,
        *h24_h48_phase_cosines,
        *aggregate_pairwise_cosines,
    ]
    if (
        len(h24_tape_phase_cosines) != 5
        or len(h24_h48_phase_cosines) != 5
        or len(aggregate_pairwise_cosines) != 3
        or not np.isfinite(np.asarray(scalars, dtype=np.float64)).all()
        or any(value < -1.000001 or value > 1.000001 for value in scalars)
    ):
        raise ValueError("objective-direction cosine evidence is invalid")
    any_full_gate = any(full_gate_by_direction.values())
    if (
        h24_tape_cosine < 0.5
        or sum(value < 0.25 for value in h24_tape_phase_cosines) >= 3
    ) and not any_full_gate:
        return "stochastic-gradient-inconsistent"
    if full_gate_by_direction["h24"]:
        return "current-h24-direction-useful"
    if full_gate_by_direction["h48"] or (
        h24_h48_cosine <= 0.0 and h48_carried_strictly_better
    ):
        return "short-horizon-credit-misaligned"
    if full_gate_by_direction["bootstrap"]:
        return "terminal-bootstrap-useful"
    if not any_full_gate and (
        sum(value < 0.0 for value in h24_h48_phase_cosines) >= 2
        or every_direction_mixed
    ):
        return "cross-state-objective-conflict"
    if (
        not any_full_gate
        and h24_tape_cosine >= 0.5
        and all(value >= 0.8 for value in aggregate_pairwise_cosines)
    ):
        return "aligned-local-step-insufficient"
    return "direction-audit-inconclusive"
