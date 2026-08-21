"""Contact-event pathwise gradient barriers for SHAC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp


def contact_gradient_barrier(
    tree: Any,
    event: jax.Array,
    *,
    enabled: bool | jax.Array,
) -> Any:
    """Preserve the primal tree and detach it at a contact event."""

    if isinstance(enabled, bool) and not enabled:
        return tree
    stopped_event = jax.lax.stop_gradient(
        jp.asarray(event, dtype=jp.bool_)
        & jp.asarray(enabled, dtype=jp.bool_)
    )
    return jax.tree_util.tree_map(
        lambda value: jp.where(
            stopped_event, jax.lax.stop_gradient(value), value
        ),
        tree,
    )


def contact_topology_event_from_info(
    info: Mapping[str, Any], *, enabled: bool
) -> jax.Array:
    """Read the G1-only event leaf without changing disabled Go2 rollouts."""

    if not enabled:
        return jp.asarray(False)
    key = "transition_contact_topology_event"
    if key not in info:
        raise KeyError(key)
    return jax.lax.stop_gradient(jp.asarray(info[key], dtype=jp.bool_))


def resolve_contact_topology_truncation_resume_setting(
    *,
    requested: bool,
    resumed_hparams: Mapping[str, object] | None,
    is_resume: bool,
    allow_change: bool,
) -> bool:
    """Resolve the treatment without silently changing resumed science."""

    if not isinstance(requested, bool) or not isinstance(allow_change, bool):
        raise ValueError("contact topology truncation settings must be boolean")
    if not is_resume:
        return requested
    key = "actor_contact_topology_gradient_truncation"
    if resumed_hparams is None or key not in resumed_hparams:
        if requested and not allow_change:
            raise ValueError(
                "starting contact topology truncation requires explicit "
                "resume authority"
            )
        return requested if allow_change else False
    saved = resumed_hparams[key]
    if not isinstance(saved, bool):
        raise ValueError("saved contact topology truncation must be boolean")
    if saved != requested and not allow_change:
        raise ValueError(
            "changing contact topology truncation requires explicit resume "
            "authority"
        )
    return requested if allow_change else saved
