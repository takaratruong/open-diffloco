from __future__ import annotations

from copy import deepcopy
import inspect

import numpy as np


def test_builder_changes_only_update_boundary_distribution_and_cadence(
    tmp_path,
) -> None:
    from experiments.g1_fresh_reference_mixture.run import (
        FRESH_REFERENCE_FRACTION,
        build_fresh_reference_kwargs,
        checkpoint_steps,
    )
    from tools.run_g1_support_aware_impulse_continuation import (
        build_support_aware_kwargs,
    )

    reference = tmp_path / "reference.npz"
    source = tmp_path / "source.pkl"
    target = tmp_path / "target.npz"
    control = build_support_aware_kwargs(
        "g1-4x5", reference, 0, source, target
    )
    treatment = build_fresh_reference_kwargs(
        "g1-4x5", reference, 0, source, target
    )
    expected = deepcopy(control)
    expected.update(
        actor_update_fresh_reference_fraction=FRESH_REFERENCE_FRACTION,
        allow_resume_actor_update_fresh_reference_change=True,
        checkpoint_steps=checkpoint_steps(),
    )
    assert treatment.keys() == expected.keys()
    for key in treatment:
        if key == "action_noise_std_end":
            np.testing.assert_array_equal(treatment[key], expected[key])
        else:
            assert treatment[key] == expected[key], key
    assert (
        treatment["actor_update_fresh_reference_fraction"]
        == FRESH_REFERENCE_FRACTION
        == 0.25
    )
    assert treatment["allow_resume_actor_update_fresh_reference_change"] is True
    assert treatment["checkpoint_steps"] == checkpoint_steps()


def test_checkpoint_selector_prefers_componentwise_safe_candidate() -> None:
    from experiments.g1_fresh_reference_mixture.run import (
        checkpoint_steps,
        select_checkpoint,
    )

    steps = checkpoint_steps()
    rows = [
        {"step": steps[0], "survival": [135, 150, 90, 100, 90]},
        {"step": steps[1], "survival": [136, 144, 84, 90, 80]},
        {"step": steps[2], "survival": [140, 145, 85, 90, 79]},
        {"step": steps[3], "survival": [130, 120, 90, 100, 100]},
    ]
    selected = select_checkpoint(rows)

    assert selected["step"] == steps[2]
    assert selected["componentwise_preserves_e002"] is True
    assert selected["strictly_improves_any_phase"] is True


def test_checkpoint_selector_minimizes_worst_forgetting_when_none_are_safe() -> None:
    from experiments.g1_fresh_reference_mixture.run import (
        checkpoint_steps,
        select_checkpoint,
    )

    steps = checkpoint_steps()
    rows = [
        {"step": steps[0], "survival": [120, 130, 80, 95, 90]},
        {"step": steps[1], "survival": [130, 120, 84, 100, 95]},
        {"step": steps[2], "survival": [116, 135, 83, 105, 100]},
        {"step": steps[3], "survival": [100, 140, 84, 100, 100]},
    ]
    selected = select_checkpoint(rows)

    assert selected["step"] == steps[0]
    assert selected["maximum_e002_deficit"] == 16
    assert selected["componentwise_preserves_e002"] is False


def test_mixture_classification_requires_behavior_and_target_consolidation() -> None:
    from experiments.g1_fresh_reference_mixture.run import classify_mixture

    assert (
        classify_mixture(
            componentwise_preserves=True,
            strictly_improves=True,
            target_reached=True,
            maximum_e002_deficit=0,
        )
        == ("fresh-mixture-consolidates", True)
    )
    assert (
        classify_mixture(
            componentwise_preserves=True,
            strictly_improves=True,
            target_reached=False,
            maximum_e002_deficit=0,
        )
        == ("fresh-mixture-preserves-without-target-learning", False)
    )
    assert (
        classify_mixture(
            componentwise_preserves=False,
            strictly_improves=True,
            target_reached=False,
            maximum_e002_deficit=20,
        )
        == ("fresh-mixture-mitigates-reversal", False)
    )
    assert (
        classify_mixture(
            componentwise_preserves=False,
            strictly_improves=True,
            target_reached=False,
            maximum_e002_deficit=40,
        )
        == ("fresh-mixture-redistributes", False)
    )


def test_training_validator_expects_fresh_vs_carried_gradient_evidence() -> None:
    from experiments.g1_fresh_reference_mixture import run

    source = inspect.getsource(run.validate_training_artifacts) + inspect.getsource(
        run._validate_gradient_decomposition
    )
    assert "actor_update_fresh_reference_count" in source
    assert "actor_grad_fresh_reference_bin_counts" in source
    assert "actor_grad_fresh_reference_bin_cosine_matrix" in source
    assert "between_variance_fraction" in source
