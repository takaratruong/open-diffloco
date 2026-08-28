from pathlib import Path

import jax
import numpy as np
import pytest


def test_jave_checkpoint_grid_is_resume_relative():
    from tools.run_g1_jave_continuation import (
        SOURCE_STEP,
        TRANSITIONS_PER_UPDATE,
        expected_checkpoint_steps,
    )

    assert expected_checkpoint_steps(2) == (
        SOURCE_STEP + TRANSITIONS_PER_UPDATE,
        SOURCE_STEP + 2 * TRANSITIONS_PER_UPDATE,
    )
    with pytest.raises(ValueError, match="positive integer"):
        expected_checkpoint_steps(0)


def test_jave_kwargs_are_matched_except_for_jave_authority():
    from tools.run_g1_jave_continuation import (
        JAVE_VG_WEIGHT,
        SOURCE_STEP,
        TRANSITIONS_PER_UPDATE,
        build_jave_kwargs,
    )

    checkpoint = Path("/tmp/checkpoint_step_1867776.pkl")
    common = ("g1-4x5", "/tmp/reference.npz", 0, checkpoint)
    control = build_jave_kwargs(
        *common, enabled=False, updates=2, warmup_updates=1
    )
    treatment = build_jave_kwargs(
        *common, enabled=True, updates=2, warmup_updates=1
    )

    assert control["actor_bootstrap_scale"] == 1.0
    assert treatment["actor_bootstrap_scale"] == 1.0
    assert control["allow_resume_actor_bootstrap_scale_change"] is True
    assert treatment["allow_resume_actor_bootstrap_scale_change"] is True
    assert control["jave_vg_weight"] == 0.0
    assert treatment["jave_vg_weight"] == JAVE_VG_WEIGHT
    assert control["allow_resume_jave_start"] is False
    assert treatment["allow_resume_jave_start"] is True
    assert treatment["jave_vg_warmup_steps"] == TRANSITIONS_PER_UPDATE
    assert treatment["total_steps"] == SOURCE_STEP + 2 * TRANSITIONS_PER_UPDATE

    for key in ("jave_vg_weight", "allow_resume_jave_start"):
        control.pop(key)
        treatment.pop(key)
    assert control.keys() == treatment.keys()
    for key in control:
        control_leaves, control_tree = jax.tree.flatten(control[key])
        treatment_leaves, treatment_tree = jax.tree.flatten(treatment[key])
        assert control_tree == treatment_tree, key
        for expected, actual in zip(
            control_leaves, treatment_leaves, strict=True
        ):
            np.testing.assert_equal(
                np.asarray(actual), np.asarray(expected), err_msg=key
            )


def test_jave_warmup_must_leave_an_active_update():
    from tools.run_g1_jave_continuation import build_jave_kwargs

    with pytest.raises(ValueError, match="warmup_updates"):
        build_jave_kwargs(
            "g1-4x5",
            "/tmp/reference.npz",
            0,
            "/tmp/checkpoint.pkl",
            enabled=True,
            updates=2,
            warmup_updates=2,
        )
