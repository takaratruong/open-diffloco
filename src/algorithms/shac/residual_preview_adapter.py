"""Zero-effect nonlinear preview residual over an exactly frozen actor."""

from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jp
import numpy as np
import optax


PyTree = Any


class FrozenPreviewResidualParams(NamedTuple):
    """Composite actor parameters with an immutable parent subtree."""

    parent: PyTree
    adapter: PyTree


class PreviewResidualAdapter(nn.Module):
    """One-hidden-layer bounded action correction with a zero output head."""

    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, frame):
        hidden = nn.elu(nn.Dense(self.hidden_dim)(frame))
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(hidden)
        return jp.tanh(logits)


def current_treatment_frame(
    observations: jax.Array,
    *,
    history_len: int,
    treatment_frame_dim: int,
) -> jax.Array:
    """Extract the newest frame from a flattened chronological history."""
    values = jp.asarray(observations)
    expected_width = history_len * treatment_frame_dim
    if (
        history_len < 1
        or treatment_frame_dim < 1
        or values.ndim < 1
        or values.shape[-1] != expected_width
    ):
        raise ValueError(
            "observations do not match the residual preview history layout"
        )
    frames = values.reshape(
        values.shape[:-1] + (history_len, treatment_frame_dim)
    )
    return frames[..., -1, :]


def apply_frozen_preview_residual(
    parent_actor,
    residual_actor: PreviewResidualAdapter,
    params: FrozenPreviewResidualParams,
    normalized_observations: jax.Array,
    *,
    history_len: int,
    treatment_frame_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply a frozen parent plus a current-frame nonlinear correction."""
    if not isinstance(params, FrozenPreviewResidualParams):
        raise ValueError(
            "residual preview parameters require frozen parent and adapter"
        )
    frame = current_treatment_frame(
        normalized_observations,
        history_len=history_len,
        treatment_frame_dim=treatment_frame_dim,
    )
    frozen_parent = jax.tree_util.tree_map(lax.stop_gradient, params.parent)
    parent_action = parent_actor.apply(
        frozen_parent, normalized_observations
    )
    residual_action = residual_actor.apply(params.adapter, frame)
    if residual_action.shape != parent_action.shape:
        raise ValueError("parent and residual actions must have matching shapes")
    return (
        parent_action + residual_action,
        parent_action,
        residual_action,
    )


def build_residual_adapter_mask(
    params: FrozenPreviewResidualParams,
) -> FrozenPreviewResidualParams:
    """Select every adapter scalar and no frozen-parent scalar."""
    if not isinstance(params, FrozenPreviewResidualParams):
        raise ValueError(
            "residual preview parameters require frozen parent and adapter"
        )
    return FrozenPreviewResidualParams(
        parent=jax.tree_util.tree_map(
            lambda value: jp.zeros(value.shape, dtype=bool), params.parent
        ),
        adapter=jax.tree_util.tree_map(
            lambda value: jp.ones(value.shape, dtype=bool), params.adapter
        ),
    )


def _adam_state(optimizer_state: optax.OptState) -> optax.ScaleByAdamState:
    if (
        not isinstance(optimizer_state, (tuple, list))
        or len(optimizer_state) != 2
        or not isinstance(optimizer_state[1], (tuple, list))
        or len(optimizer_state[1]) != 2
        or not isinstance(optimizer_state[1][0], optax.ScaleByAdamState)
    ):
        raise ValueError(
            "residual preview requires clip_by_global_norm followed by Adam"
        )
    return optimizer_state[1][0]


def _same_tree_structure(left: PyTree, right: PyTree) -> bool:
    return jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(
        right
    )


def initialize_residual_adapter_optimizer(
    optimizer: optax.GradientTransformation,
    *,
    parent_optimizer_state: optax.OptState,
    composite_params: FrozenPreviewResidualParams,
) -> optax.OptState:
    """Wrap inherited parent moments with exact-zero adapter moments."""
    if not isinstance(composite_params, FrozenPreviewResidualParams):
        raise ValueError("composite parameters are required")
    parent_adam = _adam_state(parent_optimizer_state)
    if not _same_tree_structure(parent_adam.mu, composite_params.parent):
        raise ValueError("parent optimizer moments do not match parent parameters")
    template = optimizer.init(composite_params)
    template_adam = _adam_state(template)
    if not isinstance(template_adam.mu, FrozenPreviewResidualParams):
        raise ValueError("optimizer template does not preserve composite parameters")
    migrated_adam = template_adam._replace(
        count=parent_adam.count,
        mu=FrozenPreviewResidualParams(
            parent=parent_adam.mu,
            adapter=template_adam.mu.adapter,
        ),
        nu=FrozenPreviewResidualParams(
            parent=parent_adam.nu,
            adapter=template_adam.nu.adapter,
        ),
    )
    return (
        parent_optimizer_state[0],
        (migrated_adam, parent_optimizer_state[1][1]),
    )


def _tree_equal(left: PyTree, right: PyTree) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return left_structure == right_structure and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_finite(tree: PyTree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(leaves) and all(
        np.isfinite(np.asarray(leaf)).all() for leaf in leaves
    )


def _tree_zero(tree: PyTree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(leaves) and all(
        np.all(np.asarray(leaf) == 0.0) for leaf in leaves
    )


def residual_adapter_migration_report(
    *,
    parent_actor,
    residual_actor: PreviewResidualAdapter,
    parent_params: PyTree,
    parent_optimizer_state: optax.OptState,
    candidate_params: FrozenPreviewResidualParams,
    candidate_optimizer_state: optax.OptState,
    normalized_observations: jax.Array,
    history_len: int,
    treatment_frame_dim: int,
) -> dict[str, object]:
    """Return JSON-safe evidence for exact zero-effect residual migration."""
    if not isinstance(candidate_params, FrozenPreviewResidualParams):
        raise ValueError("candidate parameters are not a residual preview actor")
    parent_adam = _adam_state(parent_optimizer_state)
    candidate_adam = _adam_state(candidate_optimizer_state)
    if not isinstance(candidate_adam.mu, FrozenPreviewResidualParams):
        raise ValueError("candidate optimizer is not a residual preview state")
    parent_parameters_exact = _tree_equal(
        parent_params, candidate_params.parent
    )
    parent_mu_exact = _tree_equal(parent_adam.mu, candidate_adam.mu.parent)
    parent_nu_exact = _tree_equal(parent_adam.nu, candidate_adam.nu.parent)
    optimizer_count_exact = np.array_equal(
        np.asarray(parent_adam.count), np.asarray(candidate_adam.count)
    )
    optimizer_outer_state_exact = bool(
        _tree_equal(parent_optimizer_state[0], candidate_optimizer_state[0])
        and _tree_equal(
            parent_optimizer_state[1][1], candidate_optimizer_state[1][1]
        )
    )
    adapter_parameters_finite = _tree_finite(candidate_params.adapter)
    adapter_mu_zero = _tree_zero(candidate_adam.mu.adapter)
    adapter_nu_zero = _tree_zero(candidate_adam.nu.adapter)
    parent_action = parent_actor.apply(
        parent_params, normalized_observations
    )
    candidate_action, reconstructed_parent, residual_action = (
        apply_frozen_preview_residual(
            parent_actor,
            residual_actor,
            candidate_params,
            normalized_observations,
            history_len=history_len,
            treatment_frame_dim=treatment_frame_dim,
        )
    )
    difference = np.abs(
        np.asarray(candidate_action) - np.asarray(parent_action)
    )
    absolute_error = float(np.max(difference))
    relative_error = float(
        np.max(
            difference
            / np.maximum(np.abs(np.asarray(parent_action)), 1e-12)
        )
    )
    residual_zero = bool(np.all(np.asarray(residual_action) == 0.0))
    reconstructed_parent_exact = bool(
        np.array_equal(
            np.asarray(reconstructed_parent), np.asarray(parent_action)
        )
    )
    adapter_parameter_count = sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(candidate_params.adapter)
    )
    valid = bool(
        parent_parameters_exact
        and parent_mu_exact
        and parent_nu_exact
        and optimizer_count_exact
        and optimizer_outer_state_exact
        and adapter_parameters_finite
        and adapter_mu_zero
        and adapter_nu_zero
        and residual_zero
        and reconstructed_parent_exact
        and absolute_error <= 1e-7
        and relative_error <= 1e-7
    )
    return {
        "protocol": "g1-frozen-residual-preview-migration-v1",
        "parent_parameters_exact": parent_parameters_exact,
        "parent_mu_exact": parent_mu_exact,
        "parent_nu_exact": parent_nu_exact,
        "optimizer_count_exact": bool(optimizer_count_exact),
        "optimizer_outer_state_exact": optimizer_outer_state_exact,
        "adapter_parameters_finite": adapter_parameters_finite,
        "adapter_mu_zero": adapter_mu_zero,
        "adapter_nu_zero": adapter_nu_zero,
        "adapter_parameter_count": adapter_parameter_count,
        "residual_action_zero": residual_zero,
        "reconstructed_parent_exact": reconstructed_parent_exact,
        "max_action_absolute_error": absolute_error,
        "max_action_relative_error": relative_error,
        "valid": valid,
    }
