from pathlib import Path

import jax
import numpy as np
import pytest


def test_root_velocity_continuation_has_exact_update_grid() -> None:
    from tools.run_g1_root_velocity_continuation import (
        END_STEP,
        ROOT_VELOCITY_WEIGHT,
        expected_checkpoint_steps,
    )

    assert ROOT_VELOCITY_WEIGHT == 1.0
    assert END_STEP == 2_162_688
    assert expected_checkpoint_steps() == (
        1_867_776,
        1_966_080,
        2_064_384,
        2_162_688,
    )


def test_root_velocity_kwargs_change_only_registered_treatment() -> None:
    from tools.run_g1_root_velocity_continuation import (
        build_root_velocity_kwargs,
    )

    checkpoint = Path("/tmp/checkpoint_step_1769472.pkl")
    treatment = build_root_velocity_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, enabled=True
    )
    control = build_root_velocity_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, enabled=False
    )

    assert treatment["tracking_root_velocity_weight"] == 1.0
    assert treatment["allow_resume_tracking_root_velocity_change"] is True
    assert treatment["actor_capture_point_tracking"] is False
    assert treatment["actor_centroidal_propulsion"] is False
    assert treatment["actor_counterfactual_wrench_distillation"] is False
    assert control["tracking_root_velocity_weight"] == 0.0
    assert control["allow_resume_tracking_root_velocity_change"] is False
    for key in (
        "tracking_root_velocity_weight",
        "allow_resume_tracking_root_velocity_change",
    ):
        treatment.pop(key)
        control.pop(key)
    assert treatment.keys() == control.keys()
    for key in treatment:
        treatment_leaves, treatment_tree = jax.tree_util.tree_flatten(
            treatment[key]
        )
        control_leaves, control_tree = jax.tree_util.tree_flatten(control[key])
        assert treatment_tree == control_tree, key
        for actual, expected in zip(
            treatment_leaves, control_leaves, strict=True
        ):
            np.testing.assert_equal(
                np.asarray(actual), np.asarray(expected), err_msg=key
            )


def _candidates(*survivals: tuple[int, int, int, int, int]):
    from tools.run_g1_root_velocity_continuation import expected_checkpoint_steps

    return {
        step: {"checkpoint_sha256": f"{index + 1:064x}", "survival": list(row)}
        for index, (step, row) in enumerate(
            zip(expected_checkpoint_steps(), survivals, strict=True)
        )
    }


def test_root_velocity_selector_retains_only_componentwise_safe_gain() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    result = classify_selection(
        _candidates(
            (131, 114, 74, 71, 74),
            (132, 114, 74, 71, 74),
            (131, 115, 75, 72, 75),
            (130, 200, 200, 200, 200),
        )
    )

    assert result["outcome"] == "root-velocity-advances"
    assert result["selected_step"] == 2_064_384
    assert result["selected_survival"] == [131, 115, 75, 72, 75]
    assert result["policy_retained"] is True


def test_root_velocity_selector_marks_aggregate_gain_with_regression() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    result = classify_selection(
        _candidates(
            (130, 120, 90, 90, 90),
            (120, 130, 100, 100, 100),
            (100, 100, 100, 100, 100),
            (131, 114, 74, 71, 74),
        )
    )

    assert result["outcome"] == "root-velocity-redistributes"
    assert result["selected_step"] is None
    assert result["policy_retained"] is False


def test_root_velocity_selector_rejects_bad_grid() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    with pytest.raises(ValueError, match="exact checkpoint grid"):
        classify_selection({})
