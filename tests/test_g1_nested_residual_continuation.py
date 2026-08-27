from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


def test_nested_residual_has_exact_short_update_grid() -> None:
    from tools.run_g1_nested_residual_continuation import (
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


def test_nested_residual_kwargs_change_only_depth_and_budget() -> None:
    from tools.run_g1_dual_scale_root_position import build_arm_kwargs
    from tools.run_g1_nested_residual_continuation import (
        build_nested_residual_kwargs,
        expected_checkpoint_steps,
    )

    source = "/tmp/e002.pkl"
    reference = "/tmp/reference.npz"
    legacy = build_arm_kwargs(
        "g1-4x5", reference, 0, source, kernel="exponential"
    )
    nested = build_nested_residual_kwargs(
        "g1-4x5", reference, 0, source
    )
    expected = deepcopy(legacy)
    expected.update(
        actor_frozen_controller_residual_depth=2,
        allow_resume_actor_frozen_controller_residual_start=True,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    assert nested.keys() == expected.keys()
    for key in nested:
        if key == "action_noise_std_end":
            np.testing.assert_array_equal(nested[key], expected[key])
        else:
            assert nested[key] == expected[key], key


def test_nested_selector_requires_componentwise_safe_strict_gain() -> None:
    from tools.run_g1_nested_residual_continuation import (
        E002_SURVIVAL,
        classify_selection,
        expected_checkpoint_steps,
    )

    steps = expected_checkpoint_steps()
    candidates = {
        steps[0]: {"checkpoint_sha256": "a" * 64, "survival": [136, 144, 84, 90, 79]},
        steps[1]: {"checkpoint_sha256": "b" * 64, "survival": [137, 144, 84, 90, 79]},
        steps[2]: {"checkpoint_sha256": "c" * 64, "survival": [180, 100, 100, 100, 100]},
        steps[3]: {"checkpoint_sha256": "d" * 64, "survival": [130, 130, 130, 130, 130]},
    }
    result = classify_selection(
        candidates, source_survival=list(E002_SURVIVAL)
    )
    assert result["outcome"] == "nested-residual-advances"
    assert result["selected_step"] == steps[1]
    assert result["policy_retained"] is True


def test_nested_selector_rejects_wrong_source_or_grid() -> None:
    from tools.run_g1_nested_residual_continuation import classify_selection

    with pytest.raises(ValueError, match="source E002"):
        classify_selection({}, source_survival=[1, 2, 3, 4, 5])
