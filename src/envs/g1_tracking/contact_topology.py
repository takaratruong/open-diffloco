"""Threshold-free grouped contact topology for G1 tracking."""

from __future__ import annotations

import jax
import jax.numpy as jp


def grouped_foot_support(
    contact_geom: jax.Array,
    contact_distance: jax.Array,
    geom_bodyid: jax.Array,
    foot_body_ids: jax.Array,
) -> jax.Array:
    """Return active left/right support from every collision geom per foot."""

    pairs = jp.asarray(contact_geom)
    distances = jp.asarray(contact_distance)
    body_for_geom = jp.asarray(geom_bodyid)
    feet = jp.asarray(foot_body_ids)
    if (
        pairs.ndim != 2
        or pairs.shape[-1] != 2
        or distances.shape != pairs.shape[:1]
        or body_for_geom.ndim != 1
        or body_for_geom.shape[0] < 1
        or feet.shape != (2,)
    ):
        raise ValueError("contact topology inputs have incompatible shapes")

    safe_pairs = jp.clip(pairs, 0, body_for_geom.shape[0] - 1)
    pair_bodies = body_for_geom[safe_pairs]
    active = distances <= 0.0
    return jp.any(
        active[None, :, None]
        & (pair_bodies[None, :, :] == feet[:, None, None]),
        axis=(1, 2),
    )


def grouped_body_pair_contacts(
    contact_geom: jax.Array,
    contact_distance: jax.Array,
    geom_bodyid: jax.Array,
    *,
    body_count: int,
) -> jax.Array:
    """Return one active-contact bit per unordered pair of model bodies."""

    pairs = jp.asarray(contact_geom)
    distances = jp.asarray(contact_distance)
    body_for_geom = jp.asarray(geom_bodyid)
    if (
        pairs.ndim != 2
        or pairs.shape[-1] != 2
        or distances.shape != pairs.shape[:1]
        or body_for_geom.ndim != 1
        or body_for_geom.shape[0] < 1
        or not isinstance(body_count, int)
        or body_count < 1
    ):
        raise ValueError("contact topology inputs have incompatible shapes")

    safe_pairs = jp.clip(pairs, 0, body_for_geom.shape[0] - 1)
    pair_bodies = body_for_geom[safe_pairs]
    lower = jp.minimum(pair_bodies[:, 0], pair_bodies[:, 1])
    upper = jp.maximum(pair_bodies[:, 0], pair_bodies[:, 1])
    active = (distances <= 0.0) & (lower != upper)
    flat_index = lower * body_count + upper
    counts = jp.zeros((body_count * body_count,), dtype=jp.int32).at[
        flat_index
    ].add(active.astype(jp.int32))
    return (counts > 0).reshape((body_count, body_count))


def contact_topology_event(
    previous: jax.Array,
    current: jax.Array,
    *,
    done: jax.Array,
) -> jax.Array:
    """Return true for a non-reset change between matching signatures."""

    previous_support = jp.asarray(previous, dtype=jp.bool_)
    current_support = jp.asarray(current, dtype=jp.bool_)
    if (
        previous_support.shape != current_support.shape
        or previous_support.size < 1
    ):
        raise ValueError("contact signatures must have matching nonempty shapes")
    return jp.any(previous_support != current_support) & ~jp.asarray(
        done, dtype=jp.bool_
    )
