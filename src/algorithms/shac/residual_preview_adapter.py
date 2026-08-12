"""Zero-effect nonlinear preview residual over an exactly frozen actor."""

from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
from flax.core import FrozenDict, freeze
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


class ResidualAdapterAuxParams(NamedTuple):
    """Residual parameters intentionally left on ordinary Adam."""

    dense0_bias: jax.Array
    dense1_kernel: jax.Array
    dense1_bias: jax.Array


class FrozenPreviewResidualMuonState(NamedTuple):
    """Frozen parent snapshot and the two trainable adapter optimizers."""

    parent_optimizer_state: optax.OptState
    muon_state: optax.OptState
    adam_state: optax.OptState


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
    assistance_scale: jax.Array | None = None,
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
    adapter_input_dim = int(
        _registered_adapter_arrays(params.adapter)[0].shape[0]
    )
    if adapter_input_dim == treatment_frame_dim + 1:
        leading_shape = frame.shape[:-1]
        if assistance_scale is None:
            scale = jp.zeros(leading_shape, dtype=frame.dtype)
        else:
            scale = jp.asarray(assistance_scale, dtype=frame.dtype)
            if scale.shape not in ((), leading_shape):
                raise ValueError("assistance scale shape must match observations")
            scale = jp.broadcast_to(scale, leading_shape)
        frame = jp.concatenate((frame, scale[..., None]), axis=-1)
    elif adapter_input_dim != treatment_frame_dim:
        raise ValueError("residual adapter input width is not registered")
    elif assistance_scale is not None:
        raise ValueError("legacy residual adapter cannot consume assistance scale")
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


def _registered_adapter_arrays(adapter_params: PyTree) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array
]:
    if not isinstance(adapter_params, (dict, FrozenDict)):
        raise ValueError("residual adapter parameters must be a mapping")
    if set(adapter_params) != {"params"}:
        raise ValueError("residual adapter must contain only the params collection")
    layers = adapter_params["params"]
    if not isinstance(layers, (dict, FrozenDict)) or set(layers) != {
        "Dense_0",
        "Dense_1",
    }:
        raise ValueError("residual adapter must contain exactly Dense_0 and Dense_1")
    dense0 = layers["Dense_0"]
    dense1 = layers["Dense_1"]
    if (
        not isinstance(dense0, (dict, FrozenDict))
        or set(dense0) != {"kernel", "bias"}
        or not isinstance(dense1, (dict, FrozenDict))
        or set(dense1) != {"kernel", "bias"}
    ):
        raise ValueError("residual dense layers must contain only kernel and bias")
    dense0_kernel = jp.asarray(dense0["kernel"])
    dense0_bias = jp.asarray(dense0["bias"])
    dense1_kernel = jp.asarray(dense1["kernel"])
    dense1_bias = jp.asarray(dense1["bias"])
    if dense0_kernel.ndim != 2:
        raise ValueError("Dense_0/kernel must be a matrix for Muon")
    return dense0_kernel, dense0_bias, dense1_kernel, dense1_bias


def split_residual_adapter_params(
    adapter_params: PyTree,
) -> tuple[jax.Array, ResidualAdapterAuxParams]:
    """Partition the registered input matrix from the Adam-only leaves."""
    kernel, dense0_bias, dense1_kernel, dense1_bias = (
        _registered_adapter_arrays(adapter_params)
    )
    return kernel, ResidualAdapterAuxParams(
        dense0_bias=dense0_bias,
        dense1_kernel=dense1_kernel,
        dense1_bias=dense1_bias,
    )


def merge_residual_adapter_params(
    template: PyTree,
    dense0_kernel: jax.Array,
    auxiliary: ResidualAdapterAuxParams,
) -> PyTree:
    """Rebuild the exact registered adapter tree and container type."""
    _registered_adapter_arrays(template)
    if not isinstance(auxiliary, ResidualAdapterAuxParams):
        raise ValueError("residual adapter auxiliary parameters are required")
    rebuilt = {
        "params": {
            "Dense_0": {
                "kernel": dense0_kernel,
                "bias": auxiliary.dense0_bias,
            },
            "Dense_1": {
                "kernel": auxiliary.dense1_kernel,
                "bias": auxiliary.dense1_bias,
            },
        }
    }
    return freeze(rebuilt) if isinstance(template, FrozenDict) else rebuilt


def build_residual_muon_optimizers(
    schedule,
) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Build official Muon for the input matrix and unchanged Adam elsewhere."""
    muon_optimizer = optax.contrib.muon(
        learning_rate=schedule,
        ns_steps=5,
        beta=0.95,
        weight_decay=0.0,
        nesterov=True,
        adaptive=False,
        preconditioning="frobenius",
        consistent_rms=0.2,
    )
    adam_optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    return muon_optimizer, adam_optimizer


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


_COUNTED_OPTIMIZER_STATES = (
    optax.contrib.MuonState,
    optax.ScaleByAdamState,
    optax.ScaleByScheduleState,
)


def _replace_optimizer_counts(state: optax.OptState, count) -> optax.OptState:
    def replace(value):
        if isinstance(value, _COUNTED_OPTIMIZER_STATES):
            return value._replace(count=count)
        return value

    return jax.tree_util.tree_map(
        replace,
        state,
        is_leaf=lambda value: isinstance(value, _COUNTED_OPTIMIZER_STATES),
    )


def _optimizer_states(state: optax.OptState, state_type: type) -> list[Any]:
    found: list[Any] = []

    def visit(value):
        if isinstance(value, state_type):
            found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(state)
    return found


def initialize_residual_muon_optimizer(
    *,
    muon_optimizer: optax.GradientTransformation,
    adam_optimizer: optax.GradientTransformation,
    parent_optimizer_state: optax.OptState,
    adapter_params: PyTree,
) -> FrozenPreviewResidualMuonState:
    """Initialize zero adapter moments at the inherited parent step count."""
    parent_count = _adam_state(parent_optimizer_state).count
    dense0_kernel, auxiliary = split_residual_adapter_params(adapter_params)
    muon_state = _replace_optimizer_counts(
        muon_optimizer.init(dense0_kernel), parent_count
    )
    adam_state = _replace_optimizer_counts(
        adam_optimizer.init(auxiliary), parent_count
    )
    return FrozenPreviewResidualMuonState(
        parent_optimizer_state=parent_optimizer_state,
        muon_state=muon_state,
        adam_state=adam_state,
    )


def residual_muon_migration_report(
    *,
    parent_optimizer_state: optax.OptState,
    candidate_optimizer_state: FrozenPreviewResidualMuonState,
) -> dict[str, object]:
    """Return JSON-safe evidence for exact Muon optimizer migration."""
    if not isinstance(
        candidate_optimizer_state, FrozenPreviewResidualMuonState
    ):
        raise ValueError("candidate optimizer is not a residual Muon state")
    parent_count = np.asarray(_adam_state(parent_optimizer_state).count)
    muon_states = _optimizer_states(
        candidate_optimizer_state.muon_state, optax.contrib.MuonState
    )
    adam_states = _optimizer_states(
        candidate_optimizer_state.adam_state, optax.ScaleByAdamState
    )
    count_states = []
    for state_type in _COUNTED_OPTIMIZER_STATES:
        count_states.extend(
            _optimizer_states(candidate_optimizer_state.muon_state, state_type)
        )
        count_states.extend(
            _optimizer_states(candidate_optimizer_state.adam_state, state_type)
        )
    parent_optimizer_snapshot_exact = _tree_equal(
        candidate_optimizer_state.parent_optimizer_state,
        parent_optimizer_state,
    )
    muon_momentum_zero = bool(muon_states) and all(
        _tree_zero(state.mu) for state in muon_states
    )
    adam_mu_zero = bool(adam_states) and all(
        _tree_zero(state.mu) for state in adam_states
    )
    adam_nu_zero = bool(adam_states) and all(
        _tree_zero(state.nu) for state in adam_states
    )
    optimizer_counts_exact = bool(count_states) and all(
        np.array_equal(np.asarray(state.count), parent_count)
        for state in count_states
    )
    valid = bool(
        parent_optimizer_snapshot_exact
        and muon_momentum_zero
        and adam_mu_zero
        and adam_nu_zero
        and optimizer_counts_exact
    )
    return {
        "protocol": "g1-frozen-residual-muon-migration-v1",
        "parent_optimizer_snapshot_exact": parent_optimizer_snapshot_exact,
        "muon_momentum_zero": muon_momentum_zero,
        "adam_mu_zero": adam_mu_zero,
        "adam_nu_zero": adam_nu_zero,
        "optimizer_counts_exact": optimizer_counts_exact,
        "valid": valid,
    }


def apply_residual_muon_update(
    *,
    muon_optimizer: optax.GradientTransformation,
    adam_optimizer: optax.GradientTransformation,
    gradients: FrozenPreviewResidualParams,
    optimizer_state: FrozenPreviewResidualMuonState,
    params: FrozenPreviewResidualParams,
) -> tuple[
    FrozenPreviewResidualParams,
    FrozenPreviewResidualMuonState,
    dict[str, jax.Array],
]:
    """Clip the adapter once, then update its matrix with Muon and rest with Adam."""
    if not isinstance(gradients, FrozenPreviewResidualParams) or not isinstance(
        params, FrozenPreviewResidualParams
    ):
        raise ValueError("residual Muon update requires composite parameters")
    if not isinstance(optimizer_state, FrozenPreviewResidualMuonState):
        raise ValueError("residual Muon optimizer state is required")
    if not _same_tree_structure(gradients, params):
        raise ValueError("gradient and parameter structures must match")

    clipped_adapter, _ = optax.clip_by_global_norm(1.0).update(
        gradients.adapter, optax.EmptyState()
    )
    kernel_gradient, auxiliary_gradient = split_residual_adapter_params(
        clipped_adapter
    )
    kernel_params, auxiliary_params = split_residual_adapter_params(
        params.adapter
    )
    kernel_update, muon_state = muon_optimizer.update(
        kernel_gradient,
        optimizer_state.muon_state,
        kernel_params,
    )
    auxiliary_update, adam_state = adam_optimizer.update(
        auxiliary_gradient,
        optimizer_state.adam_state,
        auxiliary_params,
    )
    adapter_update = merge_residual_adapter_params(
        params.adapter, kernel_update, auxiliary_update
    )
    parent_update = jax.tree_util.tree_map(jp.zeros_like, params.parent)
    updates = FrozenPreviewResidualParams(
        parent=parent_update,
        adapter=adapter_update,
    )
    new_state = FrozenPreviewResidualMuonState(
        parent_optimizer_state=optimizer_state.parent_optimizer_state,
        muon_state=muon_state,
        adam_state=adam_state,
    )
    diagnostics = {
        "preview_gradient_norm": optax.tree.norm(gradients.adapter),
        "preview_update_norm": optax.tree.norm(adapter_update),
        "frozen_update_max_abs": jp.asarray(0.0),
        "frozen_moment_drift_max_abs": jp.asarray(0.0),
        "muon_kernel_gradient_norm": optax.tree.norm(kernel_gradient),
        "muon_kernel_update_norm": optax.tree.norm(kernel_update),
        "aux_adam_gradient_norm": optax.tree.norm(auxiliary_gradient),
        "aux_adam_update_norm": optax.tree.norm(auxiliary_update),
    }
    return updates, new_state, diagnostics


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
