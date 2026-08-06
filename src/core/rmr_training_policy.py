"""Randomly initialized pure-JAX networks for RMR-style G1 training.

The frozen-checkpoint :mod:`src.core.rmr_policy` seam deliberately owns source
normalization statistics.  From-scratch training must not load or optimize
those statistics, so this module keeps trainable MLP parameters separate from
the online actor and critic normalizers owned by the trainer.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import jax
from jax import lax
import jax.numpy as jnp

__all__ = [
    "GaussianRmrActorParams",
    "RMR_HIDDEN_DIMS",
    "RmrMlpParams",
    "apply_rmr_mlp",
    "gaussian_entropy",
    "init_gaussian_rmr_actor",
    "init_rmr_critic",
    "init_rmr_mlp",
    "rmr_mlp_parameter_count",
    "sample_rmr_action",
]


RMR_HIDDEN_DIMS = (2048, 2048, 1024, 1024, 512, 512)


class RmrMlpParams(NamedTuple):
    """Weight/bias-only ELU MLP in PyTorch linear-layer orientation."""

    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]


class GaussianRmrActorParams(NamedTuple):
    """RMR mean network plus a positive log-standard-deviation parameter."""

    mlp: RmrMlpParams
    log_std: jax.Array


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hidden_dimensions(values: Sequence[int]) -> tuple[int, ...]:
    dimensions = tuple(
        _positive_dimension(value, f"hidden_dims[{index}]")
        for index, value in enumerate(values)
    )
    if not dimensions:
        raise ValueError("hidden_dims must contain at least one layer")
    return dimensions


def init_rmr_mlp(
    key: jax.Array,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dtype=jnp.float32,
) -> RmrMlpParams:
    """Initializes an ELU MLP with PyTorch ``nn.Linear`` default bounds.

    PyTorch's default Kaiming-uniform call with ``a=sqrt(5)`` reduces to a
    symmetric bound of ``1 / sqrt(fan_in)`` for both weights and biases.
    JAX and PyTorch PRNG streams are not byte-equivalent; the distribution and
    layer topology are the frozen parity contract.
    """

    input_dim = _positive_dimension(input_dim, "input_dim")
    output_dim = _positive_dimension(output_dim, "output_dim")
    hidden_dims = _hidden_dimensions(hidden_dims)
    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError("dtype must be floating")

    dimensions = (input_dim, *hidden_dims, output_dim)
    keys = jax.random.split(key, 2 * (len(dimensions) - 1))
    weights: list[jax.Array] = []
    biases: list[jax.Array] = []
    for index, (fan_in, fan_out) in enumerate(
        zip(dimensions, dimensions[1:])
    ):
        bound = jnp.asarray(1.0 / math.sqrt(fan_in), dtype=dtype)
        weights.append(
            jax.random.uniform(
                keys[2 * index],
                (fan_out, fan_in),
                dtype=dtype,
                minval=-bound,
                maxval=bound,
            )
        )
        biases.append(
            jax.random.uniform(
                keys[2 * index + 1],
                (fan_out,),
                dtype=dtype,
                minval=-bound,
                maxval=bound,
            )
        )
    return RmrMlpParams(weights=tuple(weights), biases=tuple(biases))


def init_gaussian_rmr_actor(
    key: jax.Array,
    *,
    input_dim: int = 154,
    action_dim: int = 29,
    dtype=jnp.float32,
) -> GaussianRmrActorParams:
    """Initializes the exact RMR actor topology without learned source data."""

    key, network_key = jax.random.split(key)
    action_dim = _positive_dimension(action_dim, "action_dim")
    return GaussianRmrActorParams(
        mlp=init_rmr_mlp(
            network_key,
            input_dim=input_dim,
            hidden_dims=RMR_HIDDEN_DIMS,
            output_dim=action_dim,
            dtype=dtype,
        ),
        log_std=jnp.zeros(action_dim, dtype=dtype),
    )


def init_rmr_critic(
    key: jax.Array,
    *,
    input_dim: int = 286,
    dtype=jnp.float32,
) -> RmrMlpParams:
    """Initializes the exact RMR critic topology."""

    return init_rmr_mlp(
        key,
        input_dim=input_dim,
        hidden_dims=RMR_HIDDEN_DIMS,
        output_dim=1,
        dtype=dtype,
    )


def apply_rmr_mlp(
    params: RmrMlpParams,
    observations,
) -> jax.Array:
    """Applies the linear/ELU stack to one or leading-batched observations."""

    if not params.weights or len(params.weights) != len(params.biases):
        raise ValueError("RMR MLP must contain paired weight and bias layers")
    value = jnp.asarray(observations, dtype=params.weights[0].dtype)
    last = len(params.weights) - 1
    for index, (weight, bias) in enumerate(
        zip(params.weights, params.biases)
    ):
        value = (
            jnp.matmul(value, weight.T, precision=lax.Precision.HIGHEST)
            + bias
        )
        if index != last:
            value = jax.nn.elu(value)
    return value


def sample_rmr_action(
    params: GaussianRmrActorParams,
    normalized_observations,
    epsilon,
) -> jax.Array:
    """Returns a reparameterized Gaussian action from caller-owned noise."""

    mean = apply_rmr_mlp(params.mlp, normalized_observations)
    epsilon = jnp.asarray(epsilon, dtype=mean.dtype)
    if epsilon.shape != mean.shape:
        raise ValueError(
            f"epsilon shape {epsilon.shape} does not match action {mean.shape}"
        )
    return mean + jnp.exp(params.log_std) * epsilon


def gaussian_entropy(log_std) -> jax.Array:
    """Returns summed entropy for a diagonal Gaussian action distribution."""

    log_std = jnp.asarray(log_std)
    constant = 0.5 * (1.0 + math.log(2.0 * math.pi))
    return jnp.sum(log_std + constant)


def rmr_mlp_parameter_count(params: RmrMlpParams) -> int:
    """Counts MLP weight and bias scalars, excluding Gaussian log standard deviation."""

    return sum(
        int(weight.size + bias.size)
        for weight, bias in zip(params.weights, params.biases)
    )
