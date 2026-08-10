"""Open-DiffLoco-style domain randomization for G1 tracking."""

from dataclasses import dataclass

import jax
import jax.numpy as jp


@dataclass(frozen=True)
class G1RandomizationRanges:
    """Scalar ranges mapped from the canonical Go2 SHAC configuration."""

    friction: tuple[float, float]
    mass: tuple[float, float]
    kp_scale: tuple[float, float]
    kd_scale: tuple[float, float]
    com_offset: tuple[float, float, float]


CANONICAL_G1_RANDOMIZATION = G1RandomizationRanges(
    friction=(0.5, 2.0),
    mass=(0.85, 1.15),
    kp_scale=(25.0 / 35.0, 45.0 / 35.0),
    kd_scale=(0.3 / 0.5, 0.7 / 0.5),
    com_offset=(0.05, 0.05, 0.04),
)


def sample_g1_randomization(
    key: jax.Array,
    difficulty: jax.Array,
    ranges: G1RandomizationRanges,
) -> dict[str, jax.Array]:
    """Sample one episode's parameters with upstream curriculum semantics."""
    k_friction, k_mass, k_kp, k_kd, k_com = jax.random.split(key, 5)
    kp_raw = jax.random.uniform(
        k_kp,
        (),
        minval=ranges.kp_scale[0],
        maxval=ranges.kp_scale[1],
    )
    kd_raw = jax.random.uniform(
        k_kd,
        (),
        minval=ranges.kd_scale[0],
        maxval=ranges.kd_scale[1],
    )
    com_raw = (
        2.0 * jax.random.uniform(k_com, (3,)) - 1.0
    ) * jp.asarray(ranges.com_offset)
    return {
        "friction_scale": jax.random.uniform(
            k_friction,
            (),
            minval=ranges.friction[0],
            maxval=ranges.friction[1],
        ),
        "mass_scale": jax.random.uniform(
            k_mass,
            (),
            minval=ranges.mass[0],
            maxval=ranges.mass[1],
        ),
        "kp_scale": 1.0 + difficulty * (kp_raw - 1.0),
        "kd_scale": 1.0 + difficulty * (kd_raw - 1.0),
        "com_offset": difficulty * com_raw,
    }
