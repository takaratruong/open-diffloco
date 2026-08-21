"""Threshold-free grouped foot-contact topology for G1 tracking."""

from __future__ import annotations

import jax
import jax.numpy as jp


def grouped_foot_support(
    contact_geom: jax.Array,
    efc_address: jax.Array,
    geom_bodyid: jax.Array,
    foot_body_ids: jax.Array,
) -> jax.Array:
    """Return active left/right support from every collision geom per foot."""

    pairs = jp.asarray(contact_geom)
    addresses = jp.asarray(efc_address)
    body_for_geom = jp.asarray(geom_bodyid)
    feet = jp.asarray(foot_body_ids)
    if (
        pairs.ndim != 2
        or pairs.shape[-1] != 2
        or addresses.shape != pairs.shape[:1]
        or body_for_geom.ndim != 1
        or body_for_geom.shape[0] < 1
        or feet.shape != (2,)
    ):
        raise ValueError("contact topology inputs have incompatible shapes")

    safe_pairs = jp.clip(pairs, 0, body_for_geom.shape[0] - 1)
    pair_bodies = body_for_geom[safe_pairs]
    active = addresses >= 0
    return jp.any(
        active[None, :, None]
        & (pair_bodies[None, :, :] == feet[:, None, None]),
        axis=(1, 2),
    )


def contact_topology_event(
    previous: jax.Array,
    current: jax.Array,
    *,
    done: jax.Array,
) -> jax.Array:
    """Return true for a non-reset touchdown, liftoff, or support swap."""

    previous_support = jp.asarray(previous, dtype=jp.bool_)
    current_support = jp.asarray(current, dtype=jp.bool_)
    if previous_support.shape != (2,) or current_support.shape != (2,):
        raise ValueError("foot support signatures must have shape (2,)")
    return jp.any(previous_support != current_support) & ~jp.asarray(
        done, dtype=jp.bool_
    )
