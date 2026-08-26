"""A new joint residual over one exactly frozen complete E026 controller."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
from jax import lax
import jax.numpy as jp
import numpy as np
import optax

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    current_treatment_frame,
)


PyTree = Any
ParentApply = Callable[[FrozenPreviewResidualParams, jax.Array], jax.Array]


class FrozenControllerResidualParams(NamedTuple):
    """Complete immutable E026 controller plus one new trainable adapter."""

    parent: FrozenPreviewResidualParams
    adapter: PyTree


class FrozenControllerResidualOptState(NamedTuple):
    """Immutable E026 optimizer snapshot plus the new adapter optimizer."""

    parent_optimizer_state: PyTree
    adapter_optimizer_state: optax.OptState


def apply_frozen_controller_residual(
    parent_apply: ParentApply,
    adapter_actor: PreviewResidualAdapter,
    params: FrozenControllerResidualParams,
    normalized_observations: jax.Array,
    *,
    history_len: int,
    frame_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return complete-E026 action plus the new zero-head joint correction."""
    if not isinstance(params, FrozenControllerResidualParams) or not isinstance(
        params.parent, FrozenPreviewResidualParams
    ):
        raise ValueError("frozen controller residual requires an E026 parent")
    frozen_parent = jax.tree.map(lax.stop_gradient, params.parent)
    parent_action = parent_apply(frozen_parent, normalized_observations)
    frame = current_treatment_frame(
        normalized_observations,
        history_len=history_len,
        treatment_frame_dim=frame_dim,
    )
    residual_action = adapter_actor.apply(params.adapter, frame)
    if (
        parent_action.shape != residual_action.shape
        or parent_action.shape[-1] != 29
    ):
        raise ValueError("frozen controller residual requires 29 actions")
    return parent_action + residual_action, parent_action, residual_action


def migrate_frozen_controller_residual(
    *,
    parent_params: FrozenPreviewResidualParams,
    parent_optimizer_state: PyTree,
    parent_apply: ParentApply,
    adapter_actor: PreviewResidualAdapter,
    adapter_optimizer: optax.GradientTransformation,
    rng: jax.Array,
    normalized_observations: jax.Array,
    history_len: int,
    frame_dim: int,
) -> tuple[
    FrozenControllerResidualParams,
    FrozenControllerResidualOptState,
    dict[str, bool],
]:
    """Attach a zero-effect adapter while retaining all E026 state."""
    if not isinstance(parent_params, FrozenPreviewResidualParams):
        raise ValueError("frozen controller residual migration requires E026")
    frame = current_treatment_frame(
        normalized_observations,
        history_len=history_len,
        treatment_frame_dim=frame_dim,
    )
    adapter_params = adapter_actor.init(rng, frame)
    params = FrozenControllerResidualParams(
        parent=parent_params, adapter=adapter_params
    )
    optimizer_state = FrozenControllerResidualOptState(
        parent_optimizer_state=parent_optimizer_state,
        adapter_optimizer_state=adapter_optimizer.init(adapter_params),
    )
    action, parent_action, residual_action = apply_frozen_controller_residual(
        parent_apply,
        adapter_actor,
        params,
        normalized_observations,
        history_len=history_len,
        frame_dim=frame_dim,
    )
    expected = parent_apply(parent_params, normalized_observations)
    report = {
        "parent_action_exact": bool(
            np.array_equal(np.asarray(parent_action), np.asarray(expected))
        ),
        "residual_action_zero": bool(
            np.array_equal(
                np.asarray(residual_action),
                np.zeros_like(np.asarray(residual_action)),
            )
        ),
        "parent_optimizer_preserved": (
            optimizer_state.parent_optimizer_state is parent_optimizer_state
        ),
    }
    report["valid"] = bool(
        report["parent_action_exact"]
        and report["residual_action_zero"]
        and report["parent_optimizer_preserved"]
        and np.array_equal(np.asarray(action), np.asarray(expected))
    )
    if not report["valid"]:
        raise ValueError("frozen controller residual migration is not exact")
    return params, optimizer_state, report


def update_frozen_controller_residual(
    *,
    gradients: FrozenControllerResidualParams,
    optimizer_state: FrozenControllerResidualOptState,
    params: FrozenControllerResidualParams,
    adapter_optimizer: optax.GradientTransformation,
) -> tuple[FrozenControllerResidualParams, FrozenControllerResidualOptState]:
    """Return zero parent updates and ordinary updates for the new adapter."""
    if not isinstance(optimizer_state, FrozenControllerResidualOptState):
        raise ValueError("frozen controller residual optimizer state is invalid")
    if not isinstance(params, FrozenControllerResidualParams) or not isinstance(
        gradients, FrozenControllerResidualParams
    ):
        raise ValueError("frozen controller residual parameters are invalid")
    adapter_updates, adapter_optimizer_state = adapter_optimizer.update(
        gradients.adapter,
        optimizer_state.adapter_optimizer_state,
        params.adapter,
    )
    updates = FrozenControllerResidualParams(
        parent=jax.tree.map(jp.zeros_like, params.parent),
        adapter=adapter_updates,
    )
    return updates, FrozenControllerResidualOptState(
        parent_optimizer_state=optimizer_state.parent_optimizer_state,
        adapter_optimizer_state=adapter_optimizer_state,
    )
