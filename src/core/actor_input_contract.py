"""Fail-closed validation for actor observation input dimensions."""

from __future__ import annotations


def validate_actor_input_contract(
    *,
    expected_input_dim: int,
    environment_input_dim: int,
    first_layer_input_dim: int,
) -> dict[str, int | bool]:
    """Require the environment and initialized actor to match registration."""
    values = (expected_input_dim, environment_input_dim, first_layer_input_dim)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("actor input dimensions must be integers")
    if expected_input_dim <= 0:
        raise ValueError("expected actor input dimension must be positive")
    if environment_input_dim != expected_input_dim:
        raise ValueError("environment actor input dimension does not match")
    if first_layer_input_dim != expected_input_dim:
        raise ValueError("actor first-layer input dimension does not match")
    return {
        "expected_actor_obs_dim": expected_input_dim,
        "environment_actor_obs_dim": environment_input_dim,
        "actor_first_layer_input_dim": first_layer_input_dim,
        "valid": True,
    }
