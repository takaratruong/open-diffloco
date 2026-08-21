"""Simulator contact measurements shared by environments and optimizers."""

from __future__ import annotations

import jax
import jax.numpy as jp


ROOT_DOF_COUNT = 6


def contact_stiffness(
    qfrc_constraint: jax.Array,
    qacc: jax.Array,
) -> jax.Array:
    """Return force-over-modified-acceleration on MuJoCo's floating base."""

    constraint = jp.asarray(qfrc_constraint)
    acceleration = jp.asarray(qacc)
    if constraint.shape != acceleration.shape:
        raise ValueError("constraint force and acceleration shapes must be matching")
    if not constraint.shape or constraint.shape[-1] < ROOT_DOF_COUNT:
        raise ValueError("contact stiffness inputs require at least six coordinates")
    root_force = constraint[..., :ROOT_DOF_COUNT]
    root_acceleration = acceleration[..., :ROOT_DOF_COUNT]
    modified_acceleration = jp.maximum(jp.abs(root_acceleration), 1.0)
    return jp.linalg.norm(root_force / modified_acceleration, axis=-1)
