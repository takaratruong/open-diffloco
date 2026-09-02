"""Simulator contact measurements shared by environments and optimizers."""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx


ROOT_DOF_COUNT = 6
SPATIAL_DIMENSION = 6


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


def all_body_spatial_contact_stiffness(
    contact_force: jax.Array,
    spatial_acceleration: jax.Array,
) -> jax.Array:
    """Return the official AHAC all-body normalized-contact norm.

    The pinned author implementation normalizes every spatial contact-force
    component by ``max(acceleration, 1)`` elementwise, then takes one norm over
    all bodies and all six spatial components.  Negative acceleration is
    intentionally not converted to an absolute value.
    """

    force = jp.asarray(contact_force)
    acceleration = jp.asarray(spatial_acceleration)
    if force.shape != acceleration.shape:
        raise ValueError(
            "contact force and spatial acceleration shapes must be matching"
        )
    if force.ndim < 2:
        raise ValueError("contact inputs must use a body-by-spatial layout")
    if force.shape[-1] != SPATIAL_DIMENSION:
        raise ValueError(
            "contact inputs must contain exactly six spatial components"
        )
    modified_acceleration = jp.maximum(acceleration, 1.0)
    return jp.linalg.norm(
        force / modified_acceleration,
        axis=(-2, -1),
    )


def mjx_all_body_contact_stiffness(
    model: mjx.Model,
    data: mjx.Data,
) -> jax.Array:
    """Measure contact-only all-body AHAC stiffness from complete MJX data."""

    contact_only_data = data.replace(
        xfrc_applied=jp.zeros_like(data.xfrc_applied)
    )
    complete = mjx.rne_postconstraint(model, contact_only_data)
    return all_body_spatial_contact_stiffness(
        complete._impl.cfrc_ext,
        complete._impl.cacc,
    )
