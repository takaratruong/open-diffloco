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


AHAC_CONTACT_METRICS = (
    "root_generalized",
    "all_body_spatial",
)
AHAC_SEMANTICS = (
    "legacy_horizon_only_target",
    "paper_equation_10_no_target",
)


def active_horizon_mask(horizon: jax.Array, maximum: int) -> jax.Array:
    """Return the static-scan mask for a rounded global AHAC horizon."""

    if maximum < 1:
        raise ValueError("maximum horizon must be positive")
    rounded = jp.floor(jp.asarray(horizon) + 0.5).astype(jp.int32)
    return jp.arange(maximum, dtype=jp.int32) < rounded


def evaluate_with_inactive_gradient_excision(
    function, operand, *, active: jax.Array
):
    """Evaluate both paths identically while omitting an inactive pullback.

    AHAC uses a static maximum-length scan and a dynamic active horizon.  An
    inactive physics slot is still evaluated for forward telemetry, but its
    derivative must not enter the actor graph.  Placing the stopped evaluation
    inside ``lax.cond`` prevents an unsafe primitive transpose from receiving
    a zero cotangent on the inactive branch.
    """

    predicate = jp.asarray(active, dtype=jp.bool_)
    if predicate.ndim != 0:
        raise ValueError("active horizon predicate must be scalar")
    return jax.lax.cond(
        predicate,
        function,
        lambda value: jax.lax.stop_gradient(function(value)),
        operand,
    )


def evaluate_with_runtime_pullback_gate(
    function, operand, *, pullback_active: jax.Array
):
    """Evaluate one shared primal and select its pullback at runtime.

    Unlike :func:`evaluate_with_inactive_gradient_excision`, ``function`` is
    present exactly once in the custom forward rule.  The dynamic predicate is
    consulted only by the backward rule: an active call recomputes and applies
    the ordinary VJP, while an inactive call returns a structural zero without
    invoking that VJP.  This makes two calls to one compiled executable a valid
    exact-primal comparison even when a primitive's zero-cotangent transpose is
    undefined.
    """

    predicate = jp.asarray(pullback_active, dtype=jp.bool_)
    if predicate.ndim != 0:
        raise ValueError("pullback-active predicate must be scalar")

    @jax.custom_vjp
    def gated(value, current_predicate):
        del current_predicate
        return function(value)

    def forward(value, current_predicate):
        return function(value), (value, current_predicate)

    def backward(residual, cotangent):
        value, current_predicate = residual

        def apply_pullback(inputs):
            current_value, current_cotangent = inputs
            _, pullback = jax.vjp(function, current_value)
            return pullback(current_cotangent)[0]

        def zero_pullback(inputs):
            current_value, _ = inputs

            def zero_cotangent_like(leaf):
                array = jp.asarray(leaf)
                if jp.issubdtype(array.dtype, jp.inexact):
                    return jp.zeros_like(array)
                return jp.zeros(array.shape, dtype=jax.dtypes.float0)

            return jax.tree_util.tree_map(
                zero_cotangent_like, current_value
            )

        operand_cotangent = jax.lax.cond(
            current_predicate,
            apply_pullback,
            zero_pullback,
            (value, cotangent),
        )
        return operand_cotangent, None

    gated.defvjp(forward, backward)
    return gated(operand, predicate)


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


def adaptive_contact_penalty(
    *,
    contact_by_step: jax.Array,
    dual: jax.Array,
    active_mask: jax.Array,
    threshold: float | jax.Array,
) -> jax.Array:
    """Return the Equation 10 contact term for one actor trajectory.

    The trainer minimizes actor loss, so maximizing
    ``dual * (threshold - contact)`` becomes the positive loss term
    ``dual * (contact - threshold)``.  The shared dual is not an actor
    parameter, but contact remains differentiable with respect to the policy.
    """

    contact = jp.asarray(contact_by_step)
    coefficients = jp.asarray(dual, dtype=contact.dtype)
    active = jp.asarray(active_mask, dtype=jp.bool_)
    if contact.ndim != 1 or coefficients.shape != contact.shape:
        raise ValueError("AHAC contact and dual must be matching vectors")
    if active.shape != contact.shape:
        raise ValueError("AHAC active mask must match contact")
    threshold_array = jp.asarray(threshold, dtype=contact.dtype)
    weighted_violation = jax.lax.stop_gradient(coefficients) * (
        contact - threshold_array
    )
    return masked_mean(weighted_violation, active)


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


def duplicate_single_critic_params(params):
    """Duplicate one SHAC value network into exact AHAC critic heads."""

    if not isinstance(params, dict) or set(params) != {"params"}:
        raise ValueError(
            "AHAC migration requires one single critic parameter tree"
        )
    single = params["params"]
    if (
        not isinstance(single, dict)
        or not single
        or "critic_0" in single
        or "critic_1" in single
    ):
        raise ValueError(
            "AHAC migration requires one single critic parameter tree"
        )

    def copy_tree(tree):
        return jax.tree_util.tree_map(lambda leaf: leaf, tree)

    return {
        "params": {
            "critic_0": copy_tree(single),
            "critic_1": copy_tree(single),
        }
    }


def select_critic_bootstrap_params(
    online_params,
    delayed_target_params,
    *,
    semantics: str,
):
    """Select paper AHAC's online critic or the explicit legacy target."""

    if semantics not in AHAC_SEMANTICS:
        raise ValueError("unknown AHAC semantics")
    if semantics == "paper_equation_10_no_target":
        return online_params
    return delayed_target_params


AHAC_RESUME_KEYS = (
    "ahac",
    "ahac_horizon_min",
    "ahac_horizon_max",
    "ahac_contact_threshold",
    "ahac_dual_lr",
    "ahac_critic_max_iterations",
    "ahac_critic_tolerance",
    "ahac_contact_metric",
    "ahac_semantics",
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
    if requested["ahac_contact_metric"] not in AHAC_CONTACT_METRICS:
        raise ValueError(
            "AHAC contact metric must be 'root_generalized' or "
            "'all_body_spatial'"
        )
    if requested["ahac_semantics"] not in AHAC_SEMANTICS:
        raise ValueError(
            "AHAC semantics must be 'legacy_horizon_only_target' or "
            "'paper_equation_10_no_target'"
        )
    resolved = {key: requested[key] for key in AHAC_RESUME_KEYS}
    if not is_resume:
        return resolved
    normalized_resumed_hparams = resumed_hparams
    if (
        resumed_hparams is not None
        and bool(resumed_hparams.get("ahac", False))
    ):
        normalized_resumed_hparams = {
            **resumed_hparams,
        }
        # Older AHAC checkpoints unambiguously used the only metric and actor
        # semantics then implemented locally.  Materialize both identities so
        # exact legacy resume remains possible without silently upgrading it.
        normalized_resumed_hparams.setdefault(
            "ahac_contact_metric", "root_generalized"
        )
        normalized_resumed_hparams.setdefault(
            "ahac_semantics", "legacy_horizon_only_target"
        )
    metadata_incomplete = normalized_resumed_hparams is None or any(
        key not in normalized_resumed_hparams for key in AHAC_RESUME_KEYS
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
    saved = {
        key: normalized_resumed_hparams[key]
        for key in AHAC_RESUME_KEYS
    }
    if saved != resolved and not allow_change:
        raise ValueError("AHAC settings must match the checkpoint")
    return resolved
