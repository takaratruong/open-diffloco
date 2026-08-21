"""Evaluate PPO-style KL backtracking against protected E023 behavior."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import numpy as np


CANDIDATE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
E023_FLOORS = (116, 99, 67, 49, 24)
MEAN_KL_LIMIT = 0.01
P95_STATE_KL_LIMIT = 0.02


def _finite_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{label} leaves must be finite numeric arrays")
    return array


def interpolate_actor_tree(source: Any, proposal: Any, *, alpha: float) -> Any:
    """Interpolate one registered actor proposal from an immutable source."""
    if alpha not in CANDIDATE_ALPHAS:
        raise ValueError("alpha is not on the registered candidate grid")
    if jax.tree_util.tree_structure(source) != jax.tree_util.tree_structure(
        proposal
    ):
        raise ValueError("source and proposal actor structures differ")

    def interpolate(source_leaf: Any, proposal_leaf: Any) -> Any:
        source_array = _finite_array(source_leaf, label="source actor")
        proposal_array = _finite_array(proposal_leaf, label="proposal actor")
        if source_array.shape != proposal_array.shape:
            raise ValueError("source and proposal actor structures differ")
        if source_array.dtype != proposal_array.dtype:
            raise ValueError("source and proposal actor dtypes differ")
        if alpha == 1.0:
            return proposal_leaf
        return source_leaf + alpha * (proposal_leaf - source_leaf)

    return jax.tree_util.tree_map(interpolate, source, proposal)


def gaussian_proximity(
    old_mean: Any,
    new_mean: Any,
    std: Any,
    epsilon: Any,
) -> dict[str, Any]:
    """Measure fixed-variance Gaussian KL and common-sample log ratios."""
    old = _finite_array(old_mean, label="old mean").astype(np.float64)
    new = _finite_array(new_mean, label="new mean").astype(np.float64)
    noise = _finite_array(epsilon, label="epsilon").astype(np.float64)
    scale = np.asarray(std)
    if old.ndim != 2 or old.shape[0] == 0 or old.shape[1] == 0:
        raise ValueError("policy means must be a nonempty state-action matrix")
    if new.shape != old.shape or noise.shape != old.shape:
        raise ValueError("policy mean and epsilon shapes must match")
    if scale.shape != (old.shape[1],):
        raise ValueError("standard deviation shape does not match actions")
    if not np.isfinite(scale).all():
        raise ValueError("standard deviation must be finite")
    if np.any(scale <= 0):
        raise ValueError("standard deviation must be positive")
    scale = scale.astype(np.float64)

    delta = new - old
    kl_by_state_action = np.square(delta) / (2.0 * np.square(scale))
    kl_by_state = np.sum(kl_by_state_action, axis=1)
    sampled_action = old + scale * noise
    old_quadratic = np.square((sampled_action - old) / scale)
    new_quadratic = np.square((sampled_action - new) / scale)
    log_ratio_by_state = -0.5 * np.sum(new_quadratic - old_quadratic, axis=1)

    mean_kl = float(np.mean(kl_by_state_action))
    p95_state_kl = float(np.percentile(kl_by_state, 95))
    max_state_kl = float(np.max(kl_by_state))
    absolute_log_ratio = np.abs(log_ratio_by_state)
    return {
        "kl_by_state_action": kl_by_state_action,
        "kl_by_state": kl_by_state,
        "log_ratio_by_state": log_ratio_by_state,
        "mean_kl": mean_kl,
        "p95_state_kl": p95_state_kl,
        "max_state_kl": max_state_kl,
        "mean_log_ratio": float(np.mean(log_ratio_by_state)),
        "p95_abs_log_ratio": float(np.percentile(absolute_log_ratio, 95)),
        "max_abs_log_ratio": float(np.max(absolute_log_ratio)),
        "kl_feasible": (
            mean_kl <= MEAN_KL_LIMIT
            and p95_state_kl <= P95_STATE_KL_LIMIT
        ),
    }


def _validated_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if tuple(row.get("alpha") for row in rows) != CANDIDATE_ALPHAS:
        raise ValueError("rows do not match the registered candidate grid")
    validated: list[dict[str, Any]] = []
    for row in rows:
        if row.get("valid") is not True or not isinstance(
            row.get("kl_feasible"), bool
        ):
            raise ValueError("candidate row is invalid")
        survival = row.get("survival")
        if (
            not isinstance(survival, list)
            or len(survival) != len(E023_FLOORS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in survival
            )
        ):
            raise ValueError("candidate survival is invalid")
        validated.append(dict(row))
    return validated


def select_kl_behavioral_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_survival: Sequence[int] = E023_FLOORS,
) -> dict[str, Any]:
    """Select the largest KL-feasible candidate preserving every E023 phase."""
    if tuple(source_survival) != E023_FLOORS:
        raise ValueError("source survival does not match the protected E023 floor")
    candidates = _validated_rows(rows)
    kl_feasible = [row for row in candidates if row["kl_feasible"]]
    safe = [
        row
        for row in kl_feasible
        if all(
            value >= floor
            for value, floor in zip(row["survival"], E023_FLOORS, strict=True)
        )
    ]
    if safe:
        selected = safe[0]
        advances = any(
            value > floor
            for value, floor in zip(
                selected["survival"], E023_FLOORS, strict=True
            )
        )
        outcome = "kl-trust-advances" if advances else "kl-trust-preserves-only"
        selected_alpha: float | None = float(selected["alpha"])
    elif kl_feasible:
        outcome = "kl-feasible-behaviorally-unsafe"
        selected_alpha = None
    else:
        outcome = "no-nonzero-kl-candidate"
        selected_alpha = None
    return {
        "protocol": "g1-e023-kl-behavioral-selection-v1",
        "valid": True,
        "source_survival": list(E023_FLOORS),
        "outcome": outcome,
        "selected_alpha": selected_alpha,
    }
