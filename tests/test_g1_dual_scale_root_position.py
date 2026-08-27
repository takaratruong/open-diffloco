from pathlib import Path

import jax
import numpy as np
import pytest


def test_pair_has_exact_short_checkpoint_grid() -> None:
    from tools.run_g1_dual_scale_root_position import (
        END_STEP,
        START_STEP,
        expected_checkpoint_steps,
    )

    assert START_STEP == 1_867_776
    assert END_STEP == 2_064_384
    assert expected_checkpoint_steps() == (
        1_916_928,
        1_966_080,
        2_015_232,
        2_064_384,
    )


def test_pair_arms_differ_only_in_registered_kernel_authority() -> None:
    from tools.run_g1_dual_scale_root_position import build_arm_kwargs

    checkpoint = Path("/tmp/checkpoint_step_1867776.pkl")
    control = build_arm_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, kernel="exponential"
    )
    treatment = build_arm_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, kernel="dual_scale"
    )

    assert control["tracking_anchor_position_kernel"] == "exponential"
    assert control["allow_resume_tracking_anchor_position_kernel_change"] is False
    assert treatment["tracking_anchor_position_kernel"] == "dual_scale"
    assert treatment["allow_resume_tracking_anchor_position_kernel_change"] is True
    assert treatment["tracking_root_velocity_weight"] == 1.0
    assert treatment["allow_resume_tracking_root_velocity_change"] is False
    for key in (
        "tracking_anchor_position_kernel",
        "allow_resume_tracking_anchor_position_kernel_change",
    ):
        control.pop(key)
        treatment.pop(key)
    assert control.keys() == treatment.keys()
    for key in control:
        control_leaves, control_tree = jax.tree_util.tree_flatten(control[key])
        treatment_leaves, treatment_tree = jax.tree_util.tree_flatten(
            treatment[key]
        )
        assert control_tree == treatment_tree, key
        for actual, expected in zip(
            control_leaves, treatment_leaves, strict=True
        ):
            np.testing.assert_array_equal(actual, expected, err_msg=key)


def _records(*survivals):
    from tools.run_g1_dual_scale_root_position import expected_checkpoint_steps

    return {
        step: {"checkpoint_sha256": f"{index + 1:064x}", "survival": list(row)}
        for index, (step, row) in enumerate(
            zip(expected_checkpoint_steps(), survivals, strict=True)
        )
    }


def test_selector_requires_safe_treatment_and_beats_control() -> None:
    from tools.run_g1_dual_scale_root_position import classify_pair

    result = classify_pair(
        control=_records(
            (136, 144, 84, 90, 79),
            (136, 145, 84, 90, 79),
            (135, 160, 100, 100, 100),
            (130, 170, 110, 110, 110),
        ),
        treatment=_records(
            (137, 145, 85, 91, 80),
            (138, 146, 86, 92, 81),
            (136, 144, 84, 90, 79),
            (135, 200, 200, 200, 200),
        ),
        source_survival=[136, 144, 84, 90, 79],
    )

    assert result["outcome"] == "dual-scale-advances"
    assert result["selected_treatment_step"] == 1_966_080
    assert result["selected_treatment_survival"] == [138, 146, 86, 92, 81]


def test_selector_rejects_uncorroborated_source() -> None:
    from tools.run_g1_dual_scale_root_position import classify_pair

    with pytest.raises(ValueError, match="source E002"):
        classify_pair(
            control=_records(*([(136, 144, 84, 90, 79)] * 4)),
            treatment=_records(*([(137, 145, 85, 91, 80)] * 4)),
            source_survival=[136, 144, 84, 90, 78],
        )
