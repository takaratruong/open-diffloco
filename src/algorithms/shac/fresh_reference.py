"""Fixed-count fresh-reference coverage at SHAC actor-update boundaries."""

from __future__ import annotations

import math
from typing import Any, Mapping

import jax
import jax.numpy as jp


def validate_fresh_reference_fraction(fraction: float) -> float:
    """Validate a fraction of the actor population refreshed per update."""

    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 <= float(fraction) <= 1.0
    ):
        raise ValueError(
            "actor_update_fresh_reference_fraction must be finite and in [0, 1]"
        )
    return float(fraction)


def fresh_reference_count(fraction: float, *, population_size: int) -> int:
    """Return the exact nearest-integer refresh cohort size."""

    fraction = validate_fresh_reference_fraction(fraction)
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < 1
    ):
        raise ValueError("population_size must be a positive integer")
    return int(math.floor(fraction * population_size + 0.5))


def sample_fixed_count_mask(
    key: jax.Array,
    population_size: int,
    *,
    count: int,
) -> jax.Array:
    """Sample exactly ``count`` population members without replacement."""

    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < 1
    ):
        raise ValueError("population_size must be a positive integer")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= population_size
    ):
        raise ValueError("count must be an integer in [0, population_size]")
    order = jax.random.permutation(key, population_size)
    return order < count


def _select_population_rows(mask, treatment, control):
    """Select complete batched pytree rows while preserving coherent state."""

    mask = jp.asarray(mask, dtype=jp.bool_)

    def select(treated, original):
        treated = jp.asarray(treated)
        original = jp.asarray(original)
        if treated.shape != original.shape:
            raise ValueError("fresh and carried population leaves must match")
        if not treated.shape or treated.shape[0] != mask.shape[0]:
            raise ValueError("population leaves must share the mask leading axis")
        expanded = mask.reshape(mask.shape + (1,) * (treated.ndim - 1))
        return jp.where(expanded, treated, original)

    return jax.tree_util.tree_map(select, treatment, control)


def refresh_reference_population(
    env,
    carried_state,
    *,
    mask_key: jax.Array,
    reset_key: jax.Array,
    difficulties: jax.Array,
    fraction: float,
    phase_sampler_failed_count: jax.Array | None = None,
):
    """Replace an exact random cohort with coherent states from ``env.reset``."""

    difficulties = jp.asarray(difficulties)
    if difficulties.ndim != 1 or difficulties.shape[0] < 1:
        raise ValueError("difficulties must contain one scalar per environment")
    population_size = difficulties.shape[0]
    count = fresh_reference_count(fraction, population_size=population_size)
    mask = sample_fixed_count_mask(mask_key, population_size, count=count)
    if count == 0:
        return carried_state, mask

    reset_keys = jax.random.split(reset_key, population_size)
    if phase_sampler_failed_count is None:
        fresh_state = jax.vmap(env.reset)(reset_keys, difficulties)
    else:
        if phase_sampler_failed_count.shape[0] != population_size:
            raise ValueError(
                "phase_sampler_failed_count must match the population"
            )
        fresh_state = jax.vmap(env.reset)(
            reset_keys,
            difficulties,
            phase_sampler_failed_count,
        )
    fresh_state = jax.lax.stop_gradient(fresh_state)
    return _select_population_rows(mask, fresh_state, carried_state), mask


def resolve_fresh_reference_resume_fraction(
    resumed_hparams: Mapping[str, Any] | None,
    *,
    is_resume: bool,
    requested: float,
    allow_change: bool,
) -> float:
    """Preserve the saved mixture unless an explicit resume treatment changes it."""

    requested = validate_fresh_reference_fraction(requested)
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_actor_update_fresh_reference_change must be boolean"
        )
    if not is_resume:
        return requested
    saved = validate_fresh_reference_fraction(
        0.0
        if resumed_hparams is None
        else resumed_hparams.get("actor_update_fresh_reference_fraction", 0.0)
    )
    if requested != saved and not allow_change:
        raise ValueError(
            "changing actor-update fresh-reference coverage requires explicit authority"
        )
    return requested if allow_change else saved
