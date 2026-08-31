from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


def test_support_aware_continuation_has_one_eight_update_checkpoint() -> None:
    from tools.run_g1_support_aware_impulse_continuation import (
        END_STEP,
        START_STEP,
        expected_checkpoint_steps,
    )

    assert START_STEP == 1_867_776
    assert END_STEP == 1_966_080
    assert expected_checkpoint_steps() == (1_966_080,)


def test_support_aware_kwargs_add_only_depth_two_target_and_budget() -> None:
    from tools.run_g1_dual_scale_root_position import build_arm_kwargs
    from tools.run_g1_support_aware_impulse_continuation import (
        END_STEP,
        SUPPORT_TARGET_SHA256,
        build_support_aware_kwargs,
        expected_checkpoint_steps,
    )

    source = "/tmp/e002.pkl"
    reference = "/tmp/reference.npz"
    target = "/tmp/support.npz"
    legacy = build_arm_kwargs("g1-4x5", reference, 0, source, kernel="exponential")
    treatment = build_support_aware_kwargs("g1-4x5", reference, 0, source, target)
    expected = deepcopy(legacy)
    expected.update(
        actor_frozen_controller_residual_depth=2,
        allow_resume_actor_frozen_controller_residual_start=True,
        actor_support_aware_impulse=True,
        actor_support_aware_impulse_path=target,
        actor_support_aware_impulse_sha256=SUPPORT_TARGET_SHA256,
        actor_support_aware_impulse_window=4,
        actor_support_aware_impulse_delta=0.1,
        actor_support_aware_impulse_weight=1.0,
        allow_resume_actor_support_aware_impulse_start=True,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    assert treatment.keys() == expected.keys()
    for key in treatment:
        if key == "action_noise_std_end":
            np.testing.assert_array_equal(treatment[key], expected[key])
        else:
            assert treatment[key] == expected[key], key


def test_support_aware_selector_requires_componentwise_safe_gain() -> None:
    from tools.run_g1_support_aware_impulse_continuation import (
        E002_SURVIVAL,
        classify_candidate,
    )

    advanced = classify_candidate(
        source_survival=list(E002_SURVIVAL),
        candidate_survival=[137, 144, 84, 90, 79],
        checkpoint_sha256="a" * 64,
    )
    assert advanced["outcome"] == "support-aware-joint-treatment-advances"
    assert advanced["policy_retained"] is True

    redistributed = classify_candidate(
        source_survival=list(E002_SURVIVAL),
        candidate_survival=[180, 100, 100, 100, 100],
        checkpoint_sha256="b" * 64,
    )
    assert redistributed["outcome"] == "support-aware-joint-treatment-redistributes"
    assert redistributed["policy_retained"] is False

    with pytest.raises(ValueError, match="source E002"):
        classify_candidate(
            source_survival=[1, 2, 3, 4, 5],
            candidate_survival=[137, 144, 84, 90, 79],
            checkpoint_sha256="c" * 64,
        )


def test_support_target_gate_rejects_incomplete_candidate_without_comparing_losses() -> None:
    from tools.run_g1_support_aware_impulse_continuation import (
        annotate_target_metric_coverage,
        classify_target_reachability,
    )

    source = annotate_target_metric_coverage(
        {
            "primary": {"loss": 0.0040, "valid_window_count": 125},
            "heldout": {"loss": 0.0041, "valid_window_count": 125},
        }
    )
    candidate = annotate_target_metric_coverage(
        {
            "primary": {"loss": 0.0027, "valid_window_count": 99},
            "heldout": {"loss": 0.0028, "valid_window_count": 99},
        }
    )

    assert source["complete_window_coverage"] is True
    assert candidate["complete_window_coverage"] is False
    gate = classify_target_reachability(source=source, candidate=candidate)
    assert gate == {
        "target_reached": False,
        "target_gate_reason": "candidate-incomplete-window-coverage",
        "expected_target_window_count": 125,
        "source_complete_window_coverage": True,
        "candidate_complete_window_coverage": False,
        "primary_target_loss_relative_improvement": None,
        "heldout_target_loss_relative_improvement": None,
    }
