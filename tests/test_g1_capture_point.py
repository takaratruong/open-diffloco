from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from src.envs.g1_tracking.centroidal_momentum import (
    capture_point_position,
    cpu_capture_point,
    mjx_capture_point,
    reference_capture_points,
)


def _free_body_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <body name="root" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def test_capture_point_matches_closed_form_and_has_finite_gradient() -> None:
    com = jnp.asarray([1.0, 2.0, 0.8], dtype=jnp.float64)
    momentum = jnp.asarray([2.0, -4.0, 0.0], dtype=jnp.float64)

    value = capture_point_position(
        com, momentum, total_mass=2.0, gravity_magnitude=9.81
    )
    omega = np.sqrt(9.81 / 0.8)
    expected = np.asarray([1.0 + 1.0 / omega, 2.0 - 2.0 / omega])
    np.testing.assert_allclose(value, expected, rtol=0.0, atol=1e-12)

    gradient = jax.grad(
        lambda p: jnp.sum(
            capture_point_position(
                com, p, total_mass=2.0, gravity_magnitude=9.81
            )
        )
    )(momentum)
    assert jnp.isfinite(gradient).all()
    assert float(jnp.linalg.norm(gradient)) > 0.0


@pytest.mark.parametrize(
    "com,momentum,total_mass,gravity",
    [
        (jnp.ones(2), jnp.ones(3), 1.0, 9.81),
        (jnp.ones(3), jnp.ones(2), 1.0, 9.81),
        (jnp.asarray([0.0, 0.0, 0.0]), jnp.ones(3), 1.0, 9.81),
        (jnp.ones(3), jnp.ones(3), 0.0, 9.81),
        (jnp.ones(3), jnp.ones(3), 1.0, float("nan")),
    ],
)
def test_capture_point_rejects_invalid_inputs(
    com, momentum, total_mass, gravity
) -> None:
    with pytest.raises(ValueError):
        capture_point_position(
            com,
            momentum,
            total_mass=total_mass,
            gravity_magnitude=gravity,
        )


def test_cpu_and_mjx_capture_points_match() -> None:
    model = _free_body_model()
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray([0.2, -0.1, 0.8, 1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = np.asarray([0.7, -0.4, 0.1, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    cpu = cpu_capture_point(model, data.qpos, data.qvel, root_body_id=1)
    differentiable = mjx_capture_point(
        mjx_model,
        mjx_data,
        root_body_id=1,
        total_mass=float(model.body_subtreemass[1]),
        gravity_magnitude=9.81,
    )

    np.testing.assert_allclose(differentiable, cpu, rtol=0.0, atol=1e-9)


def test_reference_capture_points_preserve_frame_count() -> None:
    model = _free_body_model()
    qpos = np.tile(
        np.asarray([0.2, -0.1, 0.8, 1.0, 0.0, 0.0, 0.0]), (3, 1)
    )
    qvel = np.tile(
        np.asarray([0.7, -0.4, 0.1, 0.0, 0.0, 0.0]), (3, 1)
    )

    result = reference_capture_points(model, qpos, qvel, root_body_id=1)

    assert result.shape == (3, 2)
    np.testing.assert_allclose(
        result[1:], np.repeat(result[:1], 2, axis=0), rtol=0.0, atol=0.0
    )
