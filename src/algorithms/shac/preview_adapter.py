"""Exact optimization boundary for a current-frame future-reference adapter."""

from typing import Any

import flax
import jax
import jax.numpy as jp
import numpy as np
import optax


PyTree = Any


def _replace_dense_zero_kernel(tree: PyTree, kernel: jax.Array) -> PyTree:
    """Replace the first dense kernel while preserving the container type."""
    was_frozen = isinstance(tree, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(tree) if was_frozen else dict(tree)
    mutable["params"] = dict(mutable["params"])
    mutable["params"]["Dense_0"] = dict(mutable["params"]["Dense_0"])
    mutable["params"]["Dense_0"]["kernel"] = kernel
    return flax.core.freeze(mutable) if was_frozen else mutable


def build_current_preview_mask(
    params: PyTree,
    *,
    history_len: int,
    legacy_frame_dim: int,
    treatment_frame_dim: int,
) -> PyTree:
    """Select only the newest history frame's append-only preview rows."""
    if (
        history_len < 1
        or legacy_frame_dim < 1
        or treatment_frame_dim <= legacy_frame_dim
    ):
        raise ValueError(
            "preview dimensions must describe a positive append-only history"
        )
    try:
        kernel = params["params"]["Dense_0"]["kernel"]
    except (KeyError, TypeError) as error:
        raise ValueError("actor parameters require params/Dense_0/kernel") from error
    expected_rows = history_len * treatment_frame_dim
    if kernel.ndim != 2 or kernel.shape[0] != expected_rows:
        raise ValueError(
            "Dense_0 kernel does not match the preview history layout"
        )

    mask = jax.tree_util.tree_map(
        lambda value: jp.zeros(value.shape, dtype=bool), params
    )
    kernel_mask = jp.zeros(kernel.shape, dtype=bool).reshape(
        history_len, treatment_frame_dim, kernel.shape[1]
    )
    kernel_mask = kernel_mask.at[-1, legacy_frame_dim:, :].set(True)
    return _replace_dense_zero_kernel(mask, kernel_mask.reshape(kernel.shape))


def _assert_same_tree(first: PyTree, second: PyTree, *, label: str) -> None:
    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(second):
        raise ValueError(f"{label} PyTrees must have identical structures")


def masked_tree_l2_norm(tree: PyTree, mask: PyTree) -> jax.Array:
    """Return the finite selected-element L2 norm of a PyTree."""
    _assert_same_tree(tree, mask, label="value and mask")
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda value, selected: jp.sum(
                jp.square(
                    jp.where(selected & jp.isfinite(value), value, 0.0)
                )
            ),
            tree,
            mask,
        )
    )
    if not leaves:
        raise ValueError("value PyTree must not be empty")
    return jp.sqrt(jp.maximum(jp.sum(jp.stack(leaves)), 0.0))


def max_abs_outside_mask(tree: PyTree, mask: PyTree) -> jax.Array:
    """Return the maximum absolute value among unselected PyTree entries."""
    _assert_same_tree(tree, mask, label="value and mask")
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda value, selected: jp.max(
                jp.where(selected, 0.0, jp.abs(value))
            ),
            tree,
            mask,
        )
    )
    if not leaves:
        raise ValueError("value PyTree must not be empty")
    return jp.max(jp.stack(leaves))


def _adam_state(optimizer_state: optax.OptState) -> optax.ScaleByAdamState:
    """Extract Adam state from the fixed clip-then-Adam chain."""
    if (
        not isinstance(optimizer_state, (tuple, list))
        or len(optimizer_state) != 2
        or not isinstance(optimizer_state[1], (tuple, list))
        or len(optimizer_state[1]) != 2
        or not isinstance(optimizer_state[1][0], optax.ScaleByAdamState)
    ):
        raise ValueError(
            "preview adapter requires clip_by_global_norm followed by Adam"
        )
    return optimizer_state[1][0]


def apply_preview_adapter_update(
    optimizer: optax.GradientTransformation,
    gradients: PyTree,
    optimizer_state: optax.OptState,
    params: PyTree,
    mask: PyTree,
) -> tuple[PyTree, optax.OptState, dict[str, jax.Array]]:
    """Advance Adam while changing only selected parameters and moments."""
    for tree, label in (
        (gradients, "gradient and parameter"),
        (mask, "mask and parameter"),
    ):
        _assert_same_tree(tree, params, label=label)
    old_adam = _adam_state(optimizer_state)
    masked_gradients = jax.tree_util.tree_map(
        lambda gradient, selected: jp.where(selected, gradient, 0.0),
        gradients,
        mask,
    )
    proposed_updates, proposed_state = optimizer.update(
        masked_gradients, optimizer_state, params
    )
    new_adam = _adam_state(proposed_state)
    updates = jax.tree_util.tree_map(
        lambda update, selected: jp.where(selected, update, 0.0),
        proposed_updates,
        mask,
    )
    merged_adam = new_adam._replace(
        mu=jax.tree_util.tree_map(
            lambda new, old, selected: jp.where(selected, new, old),
            new_adam.mu,
            old_adam.mu,
            mask,
        ),
        nu=jax.tree_util.tree_map(
            lambda new, old, selected: jp.where(selected, new, old),
            new_adam.nu,
            old_adam.nu,
            mask,
        ),
    )
    merged_state = (
        proposed_state[0],
        (merged_adam, proposed_state[1][1]),
    )
    mu_drift = jax.tree_util.tree_map(
        jp.subtract, merged_adam.mu, old_adam.mu
    )
    nu_drift = jax.tree_util.tree_map(
        jp.subtract, merged_adam.nu, old_adam.nu
    )
    diagnostics = {
        "preview_gradient_norm": masked_tree_l2_norm(gradients, mask),
        "preview_update_norm": masked_tree_l2_norm(updates, mask),
        "frozen_update_max_abs": max_abs_outside_mask(updates, mask),
        "frozen_moment_drift_max_abs": jp.maximum(
            max_abs_outside_mask(mu_drift, mask),
            max_abs_outside_mask(nu_drift, mask),
        ),
    }
    return updates, merged_state, diagnostics


def _tree_max_abs_difference(
    first: PyTree, second: PyTree, mask: PyTree | None = None
) -> float:
    _assert_same_tree(first, second, label="audit")
    if mask is not None:
        _assert_same_tree(first, mask, label="audit mask")
    maxima = []
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    mask_leaves = (
        [None] * len(first_leaves)
        if mask is None
        else jax.tree_util.tree_leaves(mask)
    )
    for first_leaf, second_leaf, selected in zip(
        first_leaves, second_leaves, mask_leaves, strict=True
    ):
        difference = np.abs(np.asarray(first_leaf) - np.asarray(second_leaf))
        if selected is not None:
            difference = np.where(np.asarray(selected), 0.0, difference)
        maxima.append(float(np.max(difference)) if difference.size else 0.0)
    if not maxima:
        raise ValueError("audit PyTrees must not be empty")
    return max(maxima)


def frozen_preview_state_drift(
    parent_params: PyTree,
    candidate_params: PyTree,
    parent_optimizer_state: optax.OptState,
    candidate_optimizer_state: optax.OptState,
    parent_normalizer: Any,
    candidate_normalizer: Any,
    mask: PyTree,
) -> dict[str, float | bool]:
    """Audit a checkpoint directly against its migrated frozen parent."""
    parent_adam = _adam_state(parent_optimizer_state)
    candidate_adam = _adam_state(candidate_optimizer_state)
    parameter_drift = _tree_max_abs_difference(
        parent_params, candidate_params, mask
    )
    mu_drift = _tree_max_abs_difference(
        parent_adam.mu, candidate_adam.mu, mask
    )
    nu_drift = _tree_max_abs_difference(
        parent_adam.nu, candidate_adam.nu, mask
    )
    normalizer_drift = max(
        _tree_max_abs_difference(
            parent_normalizer.mean, candidate_normalizer.mean
        ),
        _tree_max_abs_difference(
            parent_normalizer.var, candidate_normalizer.var
        ),
        _tree_max_abs_difference(
            parent_normalizer.count, candidate_normalizer.count
        ),
    )
    valid = all(
        value == 0.0
        for value in (parameter_drift, mu_drift, nu_drift, normalizer_drift)
    )
    return {
        "frozen_parameter_max_abs": parameter_drift,
        "frozen_mu_max_abs": mu_drift,
        "frozen_nu_max_abs": nu_drift,
        "actor_normalizer_max_abs": normalizer_drift,
        "valid": valid,
    }
