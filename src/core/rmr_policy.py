"""Pure-JAX inference and residual-composition seam for a trained RMR actor.

This module reads array-like RSL-RL model and observation-normalizer state
dictionaries and exposes a small, JAX-jittable forward pass plus an exact
residual-composition helper. It has no PyTorch runtime dependency: state
dictionaries are consumed only as array-likes (anything ``jnp.asarray`` accepts).

Frozen contract:

* Actor layers are ordered by their numeric actor index (``actor.<i>.weight`` /
  ``actor.<i>.bias``), not by insertion or lexical order.
* Inference applies source normalization ``(obs - mean) / std``, then ELU
  hidden layers, then a final linear output layer.
* Both a single observation and leading batch dimensions are supported.
* Residual composition is an unclipped source action plus the learned residual,
  and stays jittable and differentiable with respect to the residual.
* Incomplete or malformed layer state fails closed with an actionable
  ``KeyError`` or ``ValueError``.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, NamedTuple, Tuple

from jax import lax
import jax.nn as jnn
import jax.numpy as jnp

__all__ = [
    "RmrPolicy",
    "rmr_policy_from_state_dict",
    "apply_rmr_policy",
    "compose_rmr_residual",
]

_ACTOR_KEY = re.compile(r"^actor\.(\d+)\.(weight|bias)$")


class RmrPolicy(NamedTuple):
    """An immutable, pytree-friendly container for the pure-JAX RMR actor.

    Being a ``NamedTuple`` makes instances valid JAX pytrees, so they can be
    passed through ``jax.jit`` / ``jax.jacobian`` as arguments.
    """

    mean: jnp.ndarray
    std: jnp.ndarray
    weights: Tuple[jnp.ndarray, ...]
    biases: Tuple[jnp.ndarray, ...]


def _require(mapping: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in mapping:
        raise KeyError(
            f"RMR state dict is missing required {what} key {key!r}; "
            f"available keys: {sorted(mapping)!r}"
        )
    return mapping[key]


def rmr_policy_from_state_dict(
    model_state_dict: Mapping[str, Any],
    normalizer_state_dict: Mapping[str, Any],
) -> RmrPolicy:
    """Build an :class:`RmrPolicy` from RSL-RL-style state dictionaries.

    ``model_state_dict`` must contain paired ``actor.<i>.weight`` and
    ``actor.<i>.bias`` entries; ``normalizer_state_dict`` must contain
    ``_mean`` and ``_std``. Both are consumed as array-likes only.
    """
    # Collect the numeric indices that have at least one actor.* entry.
    indices: set[int] = set()
    for key in model_state_dict:
        match = _ACTOR_KEY.match(key)
        if match is not None:
            indices.add(int(match.group(1)))

    if not indices:
        raise ValueError(
            "RMR model state dict contains no 'actor.<index>.weight/bias' "
            f"layers; available keys: {sorted(model_state_dict)!r}"
        )

    weights: list[jnp.ndarray] = []
    biases: list[jnp.ndarray] = []
    for idx in sorted(indices):  # numeric ordering, not lexical/insertion
        weight = _require(model_state_dict, f"actor.{idx}.weight", "actor weight")
        bias = _require(model_state_dict, f"actor.{idx}.bias", "actor bias")
        weight = jnp.asarray(weight)
        bias = jnp.asarray(bias)
        if weight.ndim != 2:
            raise ValueError(
                f"actor.{idx}.weight must be 2-D, got shape {tuple(weight.shape)}"
            )
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                f"actor.{idx}.bias shape {tuple(bias.shape)} is incompatible "
                f"with weight shape {tuple(weight.shape)}"
            )
        weights.append(weight)
        biases.append(bias)

    mean = jnp.asarray(
        _require(normalizer_state_dict, "_mean", "normalizer mean")
    )
    std = jnp.asarray(
        _require(normalizer_state_dict, "_std", "normalizer standard deviation")
    )
    if mean.ndim == 2 and mean.shape[0] == 1:
        mean = mean[0]
    if std.ndim == 2 and std.shape[0] == 1:
        std = std[0]
    if mean.ndim != 1:
        raise ValueError(
            "normalizer _mean must be 1-D or have one leading singleton "
            f"dimension, got shape {tuple(mean.shape)}"
        )
    if std.shape != mean.shape:
        raise ValueError(
            "normalizer _std shape "
            f"{tuple(std.shape)} does not match _mean shape {tuple(mean.shape)}"
        )
    expected_input_dim = mean.shape[0]
    for idx, weight in zip(sorted(indices), weights):
        if weight.shape[1] != expected_input_dim:
            raise ValueError(
                f"actor.{idx}.weight expects {weight.shape[1]} inputs, "
                f"but the preceding output has {expected_input_dim}"
            )
        expected_input_dim = weight.shape[0]

    return RmrPolicy(
        mean=mean,
        std=std,
        weights=tuple(weights),
        biases=tuple(biases),
    )


def apply_rmr_policy(policy: RmrPolicy, observations: Any) -> jnp.ndarray:
    """Run the pure-JAX RMR forward pass on one or a batch of observations.

    Applies ``(obs - mean) / std`` normalization, ELU hidden layers, and a
    final linear output layer. A single observation returns a 1-D action; a
    leading batch dimension is preserved.
    """
    x = jnp.asarray(observations)
    x = (x - policy.mean) / (policy.std + 1e-8)

    last = len(policy.weights) - 1
    for i, (weight, bias) in enumerate(zip(policy.weights, policy.biases)):
        # x @ weight.T + bias handles both unbatched (obs_dim,) and batched
        # (..., obs_dim) inputs via standard broadcasting.
        x = jnp.matmul(
            x, weight.T, precision=lax.Precision.HIGHEST
        ) + bias
        if i != last:
            x = jnn.elu(x)
    return x


def compose_rmr_residual(
    policy: RmrPolicy, observations: Any, residual: Any
) -> jnp.ndarray:
    """Return the unclipped source action plus the learned residual.

    The result is exactly ``apply_rmr_policy(policy, observations) + residual``,
    keeping the mapping jittable and differentiable with an identity Jacobian
    with respect to ``residual``.
    """
    source_action = apply_rmr_policy(policy, observations)
    return source_action + jnp.asarray(residual)
