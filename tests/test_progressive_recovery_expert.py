from __future__ import annotations

from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.progressive_recovery_expert import (
    RecoverySupport,
    apply_state_gated_recovery,
    build_recovery_support,
    compact_recovery_gate,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
)
from src.core.networks import Actor


def _support() -> RecoverySupport:
    return RecoverySupport(
        anchors=jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
        radius=jnp.asarray(1.0, dtype=jnp.float32),
        phase_min=10,
        phase_max=12,
        taper=2,
    )


def test_support_radius_is_half_nearest_protected_distance():
    positives = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float32
    )
    negatives = np.asarray([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

    support, report = build_recovery_support(
        positives,
        negatives,
        np.asarray([10, 11, 12]),
        taper=4,
        minimum_positive_coverage=3,
    )

    assert float(support.radius) == pytest.approx(0.9)
    assert support.phase_min == 10
    assert support.phase_max == 12
    assert report["positive_leave_one_out_coverage"] == 3
    assert report["protected_negative_max_gate"] == 0.0


def test_gate_is_exact_zero_outside_phase_or_state_support():
    gate = compact_recovery_gate(
        jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        jnp.asarray([7, 11, 15]),
        _support(),
    )

    np.testing.assert_array_equal(np.asarray(gate), [0.0, 0.0, 0.0])


def test_gate_is_one_on_anchor_inside_phase_plateau():
    gate = compact_recovery_gate(
        jnp.asarray([[0.0, 0.0]]), jnp.asarray([11]), _support()
    )

    np.testing.assert_array_equal(np.asarray(gate), [1.0])


@pytest.mark.parametrize(
    ("positives", "negatives", "phases", "message"),
    [
        (np.zeros((0, 2)), np.ones((1, 2)), np.zeros((0,)), "nonempty"),
        (np.zeros((2, 2)), np.ones((1, 3)), np.arange(2), "width"),
        (
            np.asarray([[0.0, np.nan], [0.1, 0.0]]),
            np.ones((1, 2)),
            np.arange(2),
            "finite",
        ),
    ],
)
def test_support_builder_rejects_invalid_corpora(
    positives, negatives, phases, message
):
    with pytest.raises(ValueError, match=message):
        build_recovery_support(
            positives,
            negatives,
            phases,
            minimum_positive_coverage=1,
        )


def _actors_and_params():
    parent = Actor(
        2,
        hidden=(4,),
        squash=False,
        layer_norm=False,
        zero_output=False,
    )
    expert = PreviewResidualAdapter(action_dim=2, hidden_dim=3)
    parent_params = parent.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    expert_params = expert.init(jax.random.PRNGKey(1), jnp.ones((1, 2)))
    mutable = unfreeze(expert_params)
    mutable["params"]["Dense_1"]["bias"] = jnp.asarray([0.4, -0.2])
    expert_params = freeze(mutable)
    return parent, expert, FrozenPreviewResidualParams(
        parent=parent_params, adapter=expert_params
    )


def test_gate_zero_is_bit_identical_to_parent_action():
    parent, expert, params = _actors_and_params()
    observations = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)

    action, parent_action, residual, gate = apply_state_gated_recovery(
        parent,
        expert,
        params,
        observations,
        jnp.asarray([20]),
        _support(),
        history_len=1,
        treatment_frame_dim=2,
    )

    np.testing.assert_array_equal(np.asarray(gate), [0.0])
    np.testing.assert_array_equal(np.asarray(residual), np.zeros((1, 2)))
    np.testing.assert_array_equal(np.asarray(action), np.asarray(parent_action))


def test_gate_one_matches_existing_residual_application():
    parent, expert, params = _actors_and_params()
    observations = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)
    expected, expected_parent, expected_residual = apply_frozen_preview_residual(
        parent,
        expert,
        params,
        observations,
        history_len=1,
        treatment_frame_dim=2,
    )

    action, parent_action, residual, gate = apply_state_gated_recovery(
        parent,
        expert,
        params,
        observations,
        jnp.asarray([11]),
        _support(),
        history_len=1,
        treatment_frame_dim=2,
    )

    np.testing.assert_array_equal(np.asarray(gate), [1.0])
    np.testing.assert_array_equal(np.asarray(action), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(parent_action), np.asarray(expected_parent))
    np.testing.assert_array_equal(np.asarray(residual), np.asarray(expected_residual))


def test_expert_gradient_is_zero_outside_support_and_nonzero_inside():
    parent, expert, params = _actors_and_params()
    observations = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)

    def loss(adapter_params, phase):
        composite = FrozenPreviewResidualParams(
            parent=params.parent, adapter=adapter_params
        )
        action, _, _, _ = apply_state_gated_recovery(
            parent,
            expert,
            composite,
            observations,
            jnp.asarray([phase]),
            _support(),
            history_len=1,
            treatment_frame_dim=2,
        )
        return jnp.sum(action)

    outside = jax.grad(loss)(params.adapter, 20)
    inside = jax.grad(loss)(params.adapter, 11)
    outside_norm = sum(
        float(jnp.linalg.norm(leaf)) for leaf in jax.tree_util.tree_leaves(outside)
    )
    inside_norm = sum(
        float(jnp.linalg.norm(leaf)) for leaf in jax.tree_util.tree_leaves(inside)
    )

    assert outside_norm == 0.0
    assert np.isfinite(inside_norm)
    assert inside_norm > 0.0
