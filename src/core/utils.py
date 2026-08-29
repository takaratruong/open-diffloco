"""Shared math utilities."""

import jax
import jax.numpy as jp


def quat_inv(q):
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    return jp.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(v, q):
    """Rotate vector v by quaternion q."""
    q_vec = q[1:]
    a = q[0]
    return (
        2.0 * jp.dot(q_vec, v) * q_vec
        + (a * a - jp.dot(q_vec, q_vec)) * v
        + 2.0 * a * jp.cross(q_vec, v)
    )


def axis_angle_to_quat(axis, angle):
    """Convert axis-angle representation to quaternion."""
    axis = axis / (jp.linalg.norm(axis) + 1e-8)
    return jp.concatenate([jp.cos(0.5 * angle).reshape(1), jp.sin(0.5 * angle) * axis])


def compute_grad_norm(grads):
    """Compute global L2 norm of gradients."""
    leaves = jax.tree_util.tree_leaves(grads)
    return jp.sqrt(sum(jp.sum(jp.square(x)) for x in leaves))


def _mix_uint32(values):
    values = jp.asarray(values, dtype=jp.uint32)
    values = values ^ (values >> jp.uint32(16))
    values = values * jp.uint32(0x7FEB352D)
    values = values ^ (values >> jp.uint32(15))
    values = values * jp.uint32(0x846CA68B)
    return values ^ (values >> jp.uint32(16))


def tree_bit_fingerprint(tree) -> jax.Array:
    """Return a compact deterministic fingerprint of exact pytree leaf bits."""

    fingerprint = jp.zeros((4,), dtype=jp.uint32)
    for leaf_index, leaf in enumerate(jax.tree.leaves(tree)):
        array = jp.asarray(leaf)
        raw_bytes = (
            array.astype(jp.uint8)
            if array.dtype == jp.bool_
            else jax.lax.bitcast_convert_type(array, jp.uint8)
        )
        values = raw_bytes.reshape(-1).astype(jp.uint32)
        indices = jp.arange(values.size, dtype=jp.uint32)
        leaf_seed = jp.uint32(
            ((leaf_index + 1) * 0x9E3779B1) & 0xFFFFFFFF
        )
        mixed = _mix_uint32(
            values ^ (indices * jp.uint32(0x85EBCA77)) ^ leaf_seed
        )
        xor_lane = jax.lax.reduce(
            mixed,
            jp.uint32(0),
            jax.lax.bitwise_xor,
            dimensions=(0,),
        )
        remixed = _mix_uint32(mixed ^ jp.uint32(0xC2B2AE3D))
        leaf_fingerprint = jp.stack(
            (
                xor_lane,
                jp.sum(mixed, dtype=jp.uint32),
                jax.lax.reduce(
                    remixed,
                    jp.uint32(0),
                    jax.lax.bitwise_xor,
                    dimensions=(0,),
                ),
                jp.sum(remixed, dtype=jp.uint32),
            )
        )
        fingerprint = fingerprint ^ _mix_uint32(
            leaf_fingerprint + leaf_seed
        )
    return fingerprint


def cos_wave(t, step_period, scale):
    """Cosine swing profile from 0 to scale and back to 0."""
    wave = -jp.cos(((2 * jp.pi) / step_period) * t)
    return wave * (scale / 2) + (scale / 2)


def make_kinematic_ref(step_k, scale=0.3, dt=0.02):
    """Generate a 12-DoF trot joint-offset reference for Go2."""
    steps = jp.arange(step_k)
    step_period = step_k * dt
    wave = cos_wave(steps * dt, step_period, scale)

    leg_cmd = jp.concatenate(
        [
            jp.zeros((step_k, 1)),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )

    block1 = jp.concatenate(
        [
            jp.zeros((step_k, 3)),
            leg_cmd,
            leg_cmd,
            jp.zeros((step_k, 3)),
        ],
        axis=1,
    )
    block2 = jp.concatenate(
        [
            leg_cmd,
            jp.zeros((step_k, 3)),
            jp.zeros((step_k, 3)),
            leg_cmd,
        ],
        axis=1,
    )
    return jp.concatenate([block1, block2], axis=0)
