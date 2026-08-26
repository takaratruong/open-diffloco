from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from src.envs.g1_tracking.centroidal_momentum import (
    cpu_centroidal_momentum,
    mjx_centroidal_momentum,
    reference_centroidal_momentum,
    standing_com_height,
    yaw_frame_momentum,
    yaw_frame_vector,
)

jax.config.update("jax_enable_x64", True)


def _free_body_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <body name="root" pos="0 0 1">
              <freejoint/>
              <geom type="box" size="0.1 0.2 0.3" mass="4"/>
              <body name="child" pos="0.2 0 0">
                <joint axis="0 1 0"/>
                <geom type="sphere" size="0.1" mass="1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray([0.1, -0.2, 1.2, 1.0, 0.0, 0.0, 0.0, 0.3])
    data.qvel[:] = np.asarray([0.4, -0.3, 0.2, 0.1, 0.2, -0.1, 0.5])
    return data.qpos.copy(), data.qvel.copy()


def test_cpu_and_mjx_centroidal_momentum_match() -> None:
    model = _free_body_model()
    qpos, qvel = _state(model)
    root_body_id = model.body("root").id
    cpu = cpu_centroidal_momentum(model, qpos, qvel, root_body_id)

    mx = mjx.put_model(model)
    dx = mjx.make_data(model).replace(
        qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel)
    )
    dx = mjx.forward(mx, dx)
    actual = mjx_centroidal_momentum(
        mx,
        dx,
        root_body_id,
        float(model.body_subtreemass[root_body_id]),
    )

    np.testing.assert_allclose(actual, cpu, rtol=1e-9, atol=1e-9)


def test_yaw_frame_rotates_linear_and_angular_blocks_identically() -> None:
    quaternion = jnp.asarray(
        [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    )
    value = yaw_frame_momentum(
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), quaternion
    )

    np.testing.assert_allclose(
        value, [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], atol=1e-7
    )


def test_reference_preprocessing_returns_one_finite_row_per_frame() -> None:
    model = _free_body_model()
    qpos, qvel = _state(model)
    root_body_id = model.body("root").id

    rows = reference_centroidal_momentum(
        model,
        np.stack((qpos, qpos)),
        np.stack((qvel, qvel * 0.5)),
        root_body_id,
    )

    assert rows.shape == (2, 6)
    assert np.isfinite(rows).all()
    assert standing_com_height(model, qpos, root_body_id) > 0.0


def test_g1_environment_registers_reference_momentum_and_scales() -> None:
    from src.envs.g1_tracking.environment import G1TrackingEnv

    source = inspect.getsource(G1TrackingEnv.__init__)
    assert "self.root_body_id" in source
    assert "reference_centroidal_momentum(" in source
    assert "self.centroidal_linear_scale" in source
    assert "self.centroidal_angular_scale" in source


@pytest.mark.parametrize(
    ("vector", "quaternion", "message"),
    [
        (jnp.zeros(2), jnp.asarray([1.0, 0.0, 0.0, 0.0]), "shape"),
        (jnp.zeros(3), jnp.zeros(4), "nonzero norm"),
    ],
)
def test_yaw_frame_vector_rejects_invalid_inputs(
    vector: jax.Array, quaternion: jax.Array, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        yaw_frame_vector(vector, quaternion)
