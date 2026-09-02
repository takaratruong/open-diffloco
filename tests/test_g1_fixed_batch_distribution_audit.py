from __future__ import annotations

from copy import deepcopy

import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
)


def _adapter(value: float) -> dict[str, object]:
    return {
        "params": {
            "Dense_0": {
                "kernel": jnp.full((3, 2), value),
                "bias": jnp.full((2,), value + 1.0),
            },
            "Dense_1": {
                "kernel": jnp.full((2, 1), value + 2.0),
                "bias": jnp.full((1,), value + 3.0),
            },
        }
    }


def _depth_two_params() -> FrozenControllerResidualParams:
    base = FrozenPreviewResidualParams(
        parent={"base": jnp.array([1.0])},
        adapter=_adapter(2.0),
    )
    depth_one = FrozenControllerResidualParams(
        parent=base,
        adapter=_adapter(4.0),
    )
    return FrozenControllerResidualParams(
        parent=depth_one,
        adapter=_adapter(6.0),
    )


def test_zero_head_control_preserves_everything_except_newest_output_head():
    from experiments.g1_fixed_batch_distribution_audit.run import (
        zero_newest_residual_output_head,
    )
    from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256

    candidate = _depth_two_params()
    original_hash = parameter_tree_sha256(candidate)

    control = zero_newest_residual_output_head(candidate)

    assert parameter_tree_sha256(candidate) == original_hash
    assert parameter_tree_sha256(control.parent) == parameter_tree_sha256(
        candidate.parent
    )
    np.testing.assert_array_equal(
        control.adapter["params"]["Dense_0"]["kernel"],
        candidate.adapter["params"]["Dense_0"]["kernel"],
    )
    np.testing.assert_array_equal(
        control.adapter["params"]["Dense_0"]["bias"],
        candidate.adapter["params"]["Dense_0"]["bias"],
    )
    np.testing.assert_array_equal(
        control.adapter["params"]["Dense_1"]["kernel"],
        np.zeros((2, 1)),
    )
    np.testing.assert_array_equal(
        control.adapter["params"]["Dense_1"]["bias"],
        np.zeros((1,)),
    )


def test_fixed_batch_kwargs_resume_depth_two_for_exactly_one_update():
    from experiments.g1_fixed_batch_distribution_audit.run import (
        END_STEP,
        build_fixed_batch_audit_kwargs,
    )
    from tools.run_g1_support_aware_impulse_continuation import (
        build_support_aware_kwargs,
    )

    reference = "/tmp/reference.npz"
    resume = "/tmp/candidate.pkl"
    target = "/tmp/support.npz"
    expected = deepcopy(
        build_support_aware_kwargs(
            "g1-4x5", reference, 0, resume, target
        )
    )
    expected.update(
        total_steps=END_STEP,
        checkpoint_steps=(END_STEP,),
        diagnose=True,
    )

    actual = build_fixed_batch_audit_kwargs(
        "g1-4x5", reference, 0, resume, target
    )

    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "action_noise_std_end":
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key], key


def test_distribution_classifier_separates_reversal_failure_and_mixed_replica():
    from experiments.g1_fixed_batch_distribution_audit.run import (
        classify_fixed_batch_distribution,
    )

    deterministic = {
        "source_primary_loss": 0.0040,
        "source_heldout_loss": 0.0041,
        "candidate_primary_loss": 0.0054,
        "candidate_heldout_loss": 0.0055,
    }
    reversal = classify_fixed_batch_distribution(
        source_primary_loss=0.0030,
        source_heldout_loss=0.0031,
        candidate_primary_loss=0.0027,
        candidate_heldout_loss=0.0028,
        deterministic=deterministic,
    )
    assert (
        reversal["classification"]
        == "training-distribution-improves-deterministic-reverses"
    )
    assert reversal["fixed_batch_both_improve"] is True
    assert reversal["deterministic_both_reverse"] is True

    failure = classify_fixed_batch_distribution(
        source_primary_loss=0.0030,
        source_heldout_loss=0.0031,
        candidate_primary_loss=0.0031,
        candidate_heldout_loss=0.0032,
        deterministic=deterministic,
    )
    assert failure["classification"] == "fixed-batch-objective-not-improved"

    mixed = classify_fixed_batch_distribution(
        source_primary_loss=0.0030,
        source_heldout_loss=0.0031,
        candidate_primary_loss=0.0027,
        candidate_heldout_loss=0.0032,
        deterministic=deterministic,
    )
    assert mixed["classification"] == "replica-specific-objective-shift"
