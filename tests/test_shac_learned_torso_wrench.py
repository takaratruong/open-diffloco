from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def test_zero_head_wrap_is_exactly_zero_and_preserves_controller_tree():
    from src.algorithms.shac.learned_torso_wrench import (
        FrozenControllerWrenchParams,
        LearnedTorsoWrenchHead,
        apply_learned_torso_wrench,
        build_learned_wrench_mask,
    )

    controller = {"controller": jp.asarray([1.0, -2.0])}
    head = LearnedTorsoWrenchHead(hidden_dim=8)
    params = head.init(jax.random.PRNGKey(0), jp.ones((3, 5), dtype=jp.float32))
    composite = FrozenControllerWrenchParams(controller=controller, wrench=params)

    normalized = apply_learned_torso_wrench(head, composite, jp.ones((3, 5)))
    np.testing.assert_array_equal(normalized, np.zeros((3, 6), np.float32))

    mask = build_learned_wrench_mask(composite)
    assert not any(
        bool(np.any(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(mask.controller)
    )
    assert all(
        bool(np.all(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(mask.wrench)
    )


def test_yaw_frame_wrench_rotates_to_world_and_obeys_norm_caps():
    from src.algorithms.shac.learned_torso_wrench import (
        normalized_yaw_wrench_to_world,
    )

    half = np.sqrt(0.5)
    root_quaternion = jp.asarray([half, 0.0, 0.0, half])
    normalized = jp.asarray([2.0, 0.0, 0.0, 0.0, -3.0, 0.0])
    world = normalized_yaw_wrench_to_world(
        normalized,
        root_quaternion=root_quaternion,
        force_cap=10.0,
        torque_cap=4.0,
        scale=1.0,
    )

    np.testing.assert_allclose(world[:3], [0.0, 10.0, 0.0], atol=2e-6)
    np.testing.assert_allclose(world[3:], [4.0, 0.0, 0.0], atol=2e-6)
    assert float(jp.linalg.norm(world[:3])) <= 10.0 + 1e-6
    assert float(jp.linalg.norm(world[3:])) <= 4.0 + 1e-6


def test_zero_scale_is_bit_exact_and_invalid_values_fail_closed():
    from src.algorithms.shac.learned_torso_wrench import (
        normalized_yaw_wrench_to_world,
    )

    root_quaternion = jp.asarray([1.0, 0.0, 0.0, 0.0])
    zero = normalized_yaw_wrench_to_world(
        jp.ones(6),
        root_quaternion=root_quaternion,
        force_cap=10.0,
        torque_cap=4.0,
        scale=0.0,
    )
    np.testing.assert_array_equal(zero, np.zeros(6, np.float32))

    invalid = normalized_yaw_wrench_to_world(
        jp.asarray([jp.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),
        root_quaternion=root_quaternion,
        force_cap=10.0,
        torque_cap=4.0,
        scale=1.0,
    )
    assert bool(jp.all(jp.isnan(invalid)))


def test_zero_wrench_boundary_has_finite_nonzero_jacobian():
    from src.algorithms.shac.learned_torso_wrench import (
        normalized_yaw_wrench_to_world,
    )

    def convert(values):
        return normalized_yaw_wrench_to_world(
            values,
            root_quaternion=jp.asarray([1.0, 0.0, 0.0, 0.0]),
            force_cap=10.0,
            torque_cap=4.0,
            scale=1.0,
        )

    jacobian = jax.jacrev(convert)(jp.zeros(6, dtype=jp.float32))
    assert bool(jp.all(jp.isfinite(jacobian)))
    np.testing.assert_allclose(
        jacobian,
        np.diag([10.0, 10.0, 10.0, 4.0, 4.0, 4.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_wrench_head_has_finite_nonzero_gradient_after_zero_initialization():
    from src.algorithms.shac.learned_torso_wrench import LearnedTorsoWrenchHead

    head = LearnedTorsoWrenchHead(hidden_dim=8)
    obs = jp.asarray([[0.5, -1.0, 2.0]], dtype=jp.float32)
    params = head.init(jax.random.PRNGKey(3), obs)

    def objective(candidate):
        wrench = head.apply(candidate, obs)
        return jp.sum((wrench - 0.25) ** 2)

    gradient = jax.grad(objective)(params)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(bool(jp.all(jp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jp.any(leaf != 0.0)) for leaf in leaves)


def test_continuous_wrench_scale_schedule_has_exact_endpoints():
    from src.algorithms.shac.learned_torso_wrench import (
        learned_wrench_scale_at_step,
    )

    assert float(
        learned_wrench_scale_at_step(
            100, start_step=100, end_step=200, start_scale=1.0, end_scale=0.0
        )
    ) == 1.0
    assert float(
        learned_wrench_scale_at_step(
            150, start_step=100, end_step=200, start_scale=1.0, end_scale=0.0
        )
    ) == 0.5
    assert float(
        learned_wrench_scale_at_step(
            200, start_step=100, end_step=200, start_scale=1.0, end_scale=0.0
        )
    ) == 0.0


def test_scale_conditioning_migration_preserves_legacy_output_exactly():
    from src.algorithms.shac.learned_torso_wrench import (
        FrozenControllerWrenchParams,
        LearnedTorsoWrenchHead,
        apply_learned_torso_wrench,
        migrate_learned_wrench_scale_conditioning,
    )

    frame = jp.asarray([[0.3, -0.2, 0.7]], dtype=jp.float32)
    legacy = LearnedTorsoWrenchHead(hidden_dim=8)
    legacy_params = legacy.init(jax.random.PRNGKey(7), frame)
    # Make the output nonzero so exact preservation is meaningful.
    legacy_params = jax.tree_util.tree_map(lambda value: value + 0.01, legacy_params)
    composite = FrozenControllerWrenchParams(
        controller={"parent": jp.ones(1)}, wrench=legacy_params
    )

    migrated = migrate_learned_wrench_scale_conditioning(composite)
    conditioned = LearnedTorsoWrenchHead(hidden_dim=8, condition_on_scale=True)
    before = apply_learned_torso_wrench(legacy, composite, frame)
    after = apply_learned_torso_wrench(
        conditioned, migrated, frame, assistance_scale=jp.asarray([1.0])
    )
    np.testing.assert_array_equal(after, before)


def test_anneal_mask_trains_residual_adapter_and_wrench_but_not_parent():
    from src.algorithms.shac.learned_torso_wrench import (
        FrozenControllerWrenchParams,
        build_learned_wrench_anneal_mask,
    )
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
    )

    params = FrozenControllerWrenchParams(
        controller=FrozenPreviewResidualParams(
            parent={"p": jp.ones(2)}, adapter={"a": jp.ones(3)}
        ),
        wrench={"w": jp.ones(4)},
    )
    mask = build_learned_wrench_anneal_mask(params)
    assert not bool(jp.any(mask.controller.parent["p"]))
    assert bool(jp.all(mask.controller.adapter["a"]))
    assert bool(jp.all(mask.wrench["w"]))
