from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.algorithms.shac.counterfactual_wrench_distillation import (
    LEG_ACTION_NAMES,
    counterfactual_target_change,
    counterfactual_transition_loss,
    resolve_leg_action_indices,
    scatter_leg_residual,
)
from src.algorithms.shac.frozen_controller_residual import (
    apply_frozen_controller_residual,
    migrate_frozen_controller_residual,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
)
from src.core.rmr_action_noise import RMR_ACTION_STD_JOINT_NAMES


def _parent() -> FrozenPreviewResidualParams:
    return FrozenPreviewResidualParams(
        parent={"weights": jnp.arange(8 * 29, dtype=jnp.float32).reshape(8, 29) / 1000},
        adapter={"bias": jnp.linspace(-0.1, 0.1, 29, dtype=jnp.float32)},
    )


def _parent_apply(params, observation):
    return observation @ params.parent["weights"] + params.adapter["bias"]


def test_leg_indices_follow_canonical_actor_order() -> None:
    indices = resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)

    assert len(indices) == 12
    assert tuple(RMR_ACTION_STD_JOINT_NAMES[index] for index in indices) == (
        LEG_ACTION_NAMES
    )


def test_leg_residual_scatter_is_zero_outside_canonical_twelve() -> None:
    indices = resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)
    residual = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)

    scattered = scatter_leg_residual(residual, indices, action_dim=29)

    assert scattered.shape == (2, 29)
    np.testing.assert_array_equal(np.asarray(scattered)[:, list(indices)], residual)
    non_leg = sorted(set(range(29)) - set(indices))
    np.testing.assert_array_equal(
        np.asarray(scattered)[:, non_leg], np.zeros((2, 17), dtype=np.float32)
    )


@pytest.mark.parametrize(
    ("joint_names", "message"),
    [
        (RMR_ACTION_STD_JOINT_NAMES[:-1], "29 canonical"),
        (RMR_ACTION_STD_JOINT_NAMES[:-1] + ("left_hip_pitch_joint",), "unique"),
        (tuple(reversed(RMR_ACTION_STD_JOINT_NAMES)), "canonical order"),
    ],
)
def test_leg_indices_fail_closed_on_wrong_actor_contract(joint_names, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_leg_action_indices(joint_names)


def test_scatter_rejects_wrong_width_and_nonfinite_values() -> None:
    indices = resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)
    with pytest.raises(ValueError, match="twelve"):
        scatter_leg_residual(jnp.zeros((11,)), indices, action_dim=29)
    with pytest.raises(ValueError, match="finite"):
        scatter_leg_residual(jnp.full((12,), jnp.nan), indices, action_dim=29)


def test_counterfactual_change_and_loss_are_zero_for_exact_match() -> None:
    before = jnp.linspace(-0.2, 0.2, 12, dtype=jnp.float32)
    after = before + jnp.linspace(0.01, 0.12, 12, dtype=jnp.float32)
    teacher = counterfactual_target_change(before, after)

    loss, telemetry = counterfactual_transition_loss(
        teacher,
        teacher,
        target_rms=jnp.ones((12,), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(teacher, after - before)
    assert float(loss) == 0.0
    assert float(telemetry["cosine"]) == pytest.approx(1.0, abs=1e-6)
    for name in ("base_linear", "base_angular", "centroidal_linear", "centroidal_angular"):
        assert float(telemetry[f"{name}_loss"]) == 0.0


def test_counterfactual_loss_is_finite_positive_and_jittable() -> None:
    teacher = jnp.arange(1, 13, dtype=jnp.float32) / 10
    student = teacher.at[3:6].multiply(0.5)
    target_rms = jnp.linspace(0.0, 1.1, 12, dtype=jnp.float32)

    loss, telemetry = jax.jit(counterfactual_transition_loss)(
        student, teacher, target_rms
    )

    assert np.isfinite(float(loss)) and float(loss) > 0.0
    assert np.isfinite(np.asarray(list(telemetry.values()), dtype=np.float64)).all()
    assert float(telemetry["base_angular_loss"]) > 0.0


def test_leg_only_frozen_controller_migration_is_zero_effect() -> None:
    parent = _parent()
    parent_optimizer = {"count": jnp.asarray(5, dtype=jnp.int32)}
    adapter = PreviewResidualAdapter(action_dim=12, hidden_dim=16)
    optimizer = optax.adam(1e-3)
    observations = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 20
    indices = resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)

    params, state, report = migrate_frozen_controller_residual(
        parent_params=parent,
        parent_optimizer_state=parent_optimizer,
        parent_apply=_parent_apply,
        adapter_actor=adapter,
        adapter_optimizer=optimizer,
        rng=jax.random.PRNGKey(7),
        normalized_observations=observations,
        history_len=2,
        frame_dim=4,
        residual_action_indices=indices,
    )
    action, parent_action, scattered = apply_frozen_controller_residual(
        _parent_apply,
        adapter,
        params,
        observations,
        history_len=2,
        frame_dim=4,
        residual_action_indices=indices,
    )

    expected = _parent_apply(parent, observations)
    np.testing.assert_array_equal(action, expected)
    np.testing.assert_array_equal(parent_action, expected)
    np.testing.assert_array_equal(scattered, np.zeros((2, 29), dtype=np.float32))
    assert report["valid"] is True
    assert state.parent_optimizer_state is parent_optimizer
