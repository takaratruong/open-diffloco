"""Pure Adaptive Horizon Actor-Critic utilities.

The functions in this module intentionally know nothing about the G1
environment or the SHAC trainer.  This keeps the contact-stiffness proxy and
dual update independently testable before they enter a compiled rollout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from src.core.contact import contact_stiffness as contact_stiffness


class HorizonDualUpdate(NamedTuple):
    """Result of one projected AHAC horizon/dual update."""

    horizon: jax.Array
    dual: jax.Array
    valid: jax.Array


class CriticValueLoss(NamedTuple):
    """Double- or single-head value-fit diagnostics."""

    total: jax.Array
    head_losses: jax.Array
    disagreement: jax.Array


def active_horizon_mask(horizon: jax.Array, maximum: int) -> jax.Array:
    """Return the static-scan mask for a rounded global AHAC horizon."""

    if maximum < 1:
        raise ValueError("maximum horizon must be positive")
    rounded = jp.floor(jp.asarray(horizon) + 0.5).astype(jp.int32)
    return jp.arange(maximum, dtype=jp.int32) < rounded


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over active leading-axis entries without inactive contamination."""

    values = jp.asarray(values)
    mask = jp.asarray(mask, dtype=jp.bool_)
    if values.shape[0] != mask.shape[0]:
        raise ValueError("mask must match the values leading axis")
    expanded_mask = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
    weights = expanded_mask.astype(values.dtype)
    denominator = jp.maximum(jp.sum(weights), jp.asarray(1.0, values.dtype))
    return jp.sum(jp.where(expanded_mask, values, 0.0)) / denominator


def update_horizon_dual(
    *,
    horizon: jax.Array,
    dual: jax.Array,
    contact_by_step: jax.Array,
    active_mask: jax.Array,
    threshold: float | jax.Array,
    learning_rate: float | jax.Array,
    minimum: int,
    maximum: int,
) -> HorizonDualUpdate:
    """Apply projected dual ascent and the resulting bounded horizon update."""

    dual = jp.asarray(dual)
    contact_by_step = jp.asarray(contact_by_step, dtype=dual.dtype)
    active_mask = jp.asarray(active_mask, dtype=jp.bool_)
    if dual.shape != contact_by_step.shape or dual.shape != active_mask.shape:
        raise ValueError("dual, contact, and active-mask shapes must match")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid AHAC horizon bounds")

    threshold_array = jp.asarray(threshold, dtype=dual.dtype)
    rate = jp.asarray(learning_rate, dtype=dual.dtype)
    violation = contact_by_step - threshold_array
    proposed_dual = jp.maximum(dual + rate * violation, 0.0)
    new_dual = jp.where(active_mask, proposed_dual, 0.0)
    new_horizon = jp.clip(
        jp.asarray(horizon, dtype=dual.dtype) + rate * jp.sum(new_dual),
        minimum,
        maximum,
    )
    valid = (
        jp.all(jp.isfinite(contact_by_step))
        & jp.all(jp.isfinite(dual))
        & jp.isfinite(jp.asarray(horizon))
        & jp.isfinite(threshold_array)
        & (threshold_array > 0.0)
        & jp.isfinite(rate)
        & (rate > 0.0)
        & jp.all(jp.isfinite(new_dual))
        & jp.isfinite(new_horizon)
    )
    return HorizonDualUpdate(
        horizon=new_horizon,
        dual=new_dual,
        valid=valid,
    )


def critic_convergence(
    loss_history: jax.Array,
    tolerance: float | jax.Array,
) -> jax.Array:
    """Whether five consecutive finite critic losses have stabilized."""

    losses = jp.asarray(loss_history)
    if losses.ndim != 1 or losses.shape[0] < 5:
        return jp.asarray(False)
    recent = losses[-5:]
    tolerance_array = jp.asarray(tolerance, dtype=recent.dtype)
    return (
        jp.all(jp.isfinite(recent))
        & jp.isfinite(tolerance_array)
        & (tolerance_array > 0.0)
        & (jp.mean(jp.abs(jp.diff(recent))) < tolerance_array)
    )


def conservative_value(values: jax.Array, *, double: bool) -> jax.Array:
    """Return AHAC's minimum head or the legacy singleton critic value."""

    values = jp.asarray(values)
    expected = 2 if double else 1
    if not values.shape or values.shape[-1] != expected:
        raise ValueError(f"critic values must end in exactly {expected} head(s)")
    return jp.min(values, axis=-1) if double else jp.squeeze(values, axis=-1)


def critic_value_loss(
    predictions: jax.Array,
    targets: jax.Array,
    *,
    double: bool,
    active_mask: jax.Array | None = None,
) -> CriticValueLoss:
    """Fit every critic head to one stopped-gradient TD(lambda) target."""

    predictions = jp.asarray(predictions)
    targets = jp.asarray(targets)
    expected = 2 if double else 1
    if predictions.shape[:-1] != targets.shape or predictions.shape[-1] != expected:
        raise ValueError("critic predictions and targets have incompatible shapes")
    squared_error = jp.square(predictions - targets[..., None])
    if active_mask is None:
        weights = jp.ones_like(targets)
    else:
        weights = jp.asarray(active_mask, dtype=predictions.dtype)
        if weights.shape != targets.shape:
            raise ValueError("critic active mask must match targets")
    denominator = jp.maximum(jp.sum(weights), 1.0)
    head_losses = jp.sum(
        squared_error * weights[..., None],
        axis=tuple(range(targets.ndim)),
    ) / denominator
    disagreement = (
        jp.sum(
            jp.abs(predictions[..., 0] - predictions[..., 1]) * weights
        )
        / denominator
        if double
        else jp.asarray(0.0, dtype=predictions.dtype)
    )
    return CriticValueLoss(
        total=jp.mean(head_losses),
        head_losses=head_losses,
        disagreement=disagreement,
    )


def select_active_tree(previous, candidate, active: jax.Array):
    """Choose a candidate pytree only for an active static-scan slot."""

    active = jp.asarray(active, dtype=jp.bool_)
    return jax.tree_util.tree_map(
        lambda old, new: jp.where(active, new, old),
        previous,
        candidate,
    )


AHAC_RESUME_KEYS = (
    "ahac",
    "ahac_horizon_min",
    "ahac_horizon_max",
    "ahac_contact_threshold",
    "ahac_dual_lr",
    "ahac_critic_max_iterations",
    "ahac_critic_tolerance",
)


def resolve_ahac_resume_settings(
    *,
    requested: dict[str, object],
    resumed_hparams: dict[str, object] | None,
    is_resume: bool,
    allow_change: bool,
) -> dict[str, object]:
    """Fail closed when a resumed checkpoint changes the AHAC treatment."""

    missing_requested = [key for key in AHAC_RESUME_KEYS if key not in requested]
    if missing_requested:
        raise ValueError(f"requested AHAC settings omit {missing_requested}")
    resolved = {key: requested[key] for key in AHAC_RESUME_KEYS}
    if not is_resume:
        return resolved
    metadata_incomplete = resumed_hparams is None or any(
        key not in resumed_hparams for key in AHAC_RESUME_KEYS
    )
    if metadata_incomplete:
        saved_ahac = False if resumed_hparams is None else resumed_hparams.get("ahac", False)
        if not bool(resolved["ahac"]) and not bool(saved_ahac):
            # Checkpoints created before AHAC existed have no AHAC metadata.  They
            # are unambiguously legacy SHAC when the resumed treatment also keeps
            # AHAC disabled, so preserve that path without requiring migration.
            return resolved
        if not allow_change:
            raise ValueError("AHAC resume metadata is missing or incomplete")
        return resolved
    saved = {key: resumed_hparams[key] for key in AHAC_RESUME_KEYS}
    if saved != resolved and not allow_change:
        raise ValueError("AHAC settings must match the checkpoint")
    return resolved
