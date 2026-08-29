from pathlib import Path

import jax
import numpy as np


def test_inprocess_pair_builds_one_common_warmup_and_two_active_updates():
    from tools.run_g1_jave_inprocess_pair import (
        JAVE_VG_WEIGHT,
        SOURCE_STEP,
        TRANSITIONS_PER_UPDATE,
        WARMUP_STEP,
        build_branch_kwargs,
        expected_branch_checkpoint_steps,
    )

    assert WARMUP_STEP == SOURCE_STEP + TRANSITIONS_PER_UPDATE
    expected = (
        WARMUP_STEP + TRANSITIONS_PER_UPDATE,
        WARMUP_STEP + 2 * TRANSITIONS_PER_UPDATE,
    )
    assert expected_branch_checkpoint_steps(2) == expected

    common = (
        "g1-4x5",
        Path("/tmp/reference.npz"),
        0,
        Path("/tmp/checkpoint_step_1880064.pkl"),
    )
    control = build_branch_kwargs(*common, enabled=False, active_updates=2)
    treatment = build_branch_kwargs(*common, enabled=True, active_updates=2)

    assert control["checkpoint_steps"] == expected
    assert treatment["checkpoint_steps"] == expected
    assert control["total_steps"] == expected[-1]
    assert treatment["total_steps"] == expected[-1]
    assert control["jave_vg_weight"] == 0.0
    assert treatment["jave_vg_weight"] == JAVE_VG_WEIGHT
    assert control["allow_resume_jave_weight_change"] is True
    assert treatment["allow_resume_jave_weight_change"] is True
    assert control["jave_vg_warmup_steps"] == 0
    assert treatment["jave_vg_warmup_steps"] == 0

    control.pop("jave_vg_weight")
    treatment.pop("jave_vg_weight")
    assert control.keys() == treatment.keys()
    for key in control:
        control_leaves, control_tree = jax.tree.flatten(control[key])
        treatment_leaves, treatment_tree = jax.tree.flatten(treatment[key])
        assert control_tree == treatment_tree, key
        for expected_leaf, actual_leaf in zip(
            control_leaves, treatment_leaves, strict=True
        ):
            np.testing.assert_equal(
                np.asarray(actual_leaf), np.asarray(expected_leaf), err_msg=key
            )


def test_control_repeatability_requires_exact_state_hashes_and_telemetry():
    from tools.run_g1_jave_inprocess_pair import compare_control_repeats

    control_a = {
        "checkpoint_steps": [1, 2],
        "state_sha256": ["a" * 64, "b" * 64],
        "telemetry": [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}],
    }
    control_b = {
        **control_a,
        "state_sha256": list(control_a["state_sha256"]),
        "telemetry": [dict(row) for row in control_a["telemetry"]],
    }
    assert compare_control_repeats(control_a, control_b)["valid"] is True

    control_b["state_sha256"][1] = "c" * 64
    report = compare_control_repeats(control_a, control_b)
    assert report["valid"] is False
    assert report["first_mismatch_step"] == 2


def test_jave_pair_selection_requires_control_and_retained_preservation():
    from tools.run_g1_jave_inprocess_pair import classify_pair

    steps = (10, 20)
    baseline = [136, 144, 84, 90, 79]
    control = {
        10: [120, 130, 80, 88, 75],
        20: [130, 140, 82, 89, 78],
    }
    treatment = {
        10: [121, 130, 81, 88, 76],
        20: [137, 145, 85, 91, 80],
    }

    result = classify_pair(
        control,
        treatment,
        checkpoint_steps=steps,
        retained_survival=baseline,
    )
    assert result["outcome"] == "jave-late-advances"
    assert result["selected_step"] == 20
    assert result["policy_retained"] is True
    assert result["checkpoints"][0]["eligible"] is False

    treatment[20] = [137, 139, 90, 92, 82]
    result = classify_pair(
        control,
        treatment,
        checkpoint_steps=steps,
        retained_survival=baseline,
    )
    assert result["outcome"] == "jave-late-mixed"
    assert result["selected_step"] is None
    assert result["policy_retained"] is False
