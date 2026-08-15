from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
)


def _fixture():
    actor = PreviewResidualAdapter(action_dim=3, hidden_dim=8)
    adapter = actor.init(jax.random.PRNGKey(0), jnp.zeros((1, 5)))
    params = FrozenPreviewResidualParams(
        parent={"w": jnp.asarray([2.0, -1.0])},
        adapter=adapter,
    )
    physics = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, 0.1), params
    )
    frames = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 20.0
    parent_action = jnp.zeros((4, 3), dtype=jnp.float32)
    correction = jnp.full((4, 3), 0.25, dtype=jnp.float32)
    return actor, params, physics, frames, parent_action, correction


def test_train_exposes_recovery_teacher_contract():
    from src.algorithms.shac.algorithm import train

    signature = inspect.signature(train)
    expected = {
        "actor_recovery_teacher_dataset_path": None,
        "actor_recovery_teacher_dataset_sha256": None,
        "actor_recovery_teacher_gradient_ratio": 0.0,
        "allow_resume_actor_recovery_teacher_change": False,
    }
    for name, default in expected.items():
        assert signature.parameters[name].default == default


def test_enabled_teacher_gradient_is_adapter_only_finite_and_bounded():
    from src.algorithms.shac.recovery_teacher import (
        mix_recovery_teacher_actor_gradient,
    )

    actor, params, physics, frames, parent_action, correction = _fixture()
    result = jax.jit(mix_recovery_teacher_actor_gradient, static_argnames=("residual_actor",))(
        physics,
        params,
        residual_actor=actor,
        teacher_frames=frames,
        parent_action=parent_action,
        teacher_correction=correction,
        teacher_effective_action=correction,
        max_ratio=0.5,
    )

    assert float(result.loss) > 0.0
    for leaf in jax.tree_util.tree_leaves(result.teacher_gradient.parent):
        np.testing.assert_array_equal(leaf, np.zeros_like(leaf))
    assert any(
        np.any(np.asarray(leaf) != 0.0)
        for leaf in jax.tree_util.tree_leaves(result.teacher_gradient.adapter)
    )
    assert float(result.mix.applied_teacher_norm) <= (
        0.5 * float(result.mix.physics_norm) + 1e-7
    )
    assert np.isfinite(
        np.asarray(jax.tree_util.tree_leaves(result.mix.combined_gradient)[0])
    ).all()
    assert float(result.parent_gradient_max_abs) == 0.0
    assert bool(result.valid)


def test_disabled_teacher_path_preserves_physics_gradient_exactly():
    from src.algorithms.shac.recovery_teacher import (
        mix_recovery_teacher_actor_gradient,
    )

    actor, params, physics, *_ = _fixture()
    result = mix_recovery_teacher_actor_gradient(
        physics,
        params,
        residual_actor=actor,
        teacher_frames=None,
        parent_action=None,
        teacher_correction=None,
        teacher_effective_action=None,
        max_ratio=0.0,
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(result.mix.combined_gradient),
        jax.tree_util.tree_leaves(physics),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert float(result.loss) == 0.0
    assert float(result.mix.applied_teacher_norm) == 0.0
    assert not bool(result.enabled)
    assert bool(result.valid)


def test_checkpoint_teacher_telemetry_is_complete_and_fails_closed():
    from src.algorithms.shac.algorithm import (
        build_checkpoint_recovery_teacher_telemetry,
    )

    metrics = {
        "actor_recovery_teacher_loss": 0.2,
        "actor_recovery_teacher_raw_gradient_norm": 4.0,
        "actor_recovery_teacher_projected_gradient_norm": 3.0,
        "actor_recovery_teacher_applied_gradient_norm": 1.0,
        "actor_recovery_teacher_physics_gradient_norm": 2.0,
        "actor_recovery_teacher_combined_gradient_norm": 2.5,
        "actor_recovery_teacher_physics_dot": -0.5,
        "actor_recovery_teacher_physics_cosine": -0.25,
        "actor_recovery_teacher_applied_scale": 1.0 / 3.0,
        "actor_recovery_teacher_parent_gradient_max_abs": 0.0,
        "actor_recovery_teacher_valid": True,
    }
    row = build_checkpoint_recovery_teacher_telemetry(
        metrics, max_ratio=0.5
    )
    assert row["actor_recovery_teacher_valid"] is True
    assert len(row) == 11

    with pytest.raises(ValueError, match="norm cap"):
        build_checkpoint_recovery_teacher_telemetry(
            {
                **metrics,
                "actor_recovery_teacher_applied_gradient_norm": 1.01,
            },
            max_ratio=0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        build_checkpoint_recovery_teacher_telemetry(
            {**metrics, "actor_recovery_teacher_loss": np.nan},
            max_ratio=0.5,
        )
