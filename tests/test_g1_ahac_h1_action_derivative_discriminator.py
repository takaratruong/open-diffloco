from __future__ import annotations

import inspect


def test_h1_probe_changes_only_the_registered_localization_fields(
    monkeypatch,
) -> None:
    from experiments.g1_ahac_h1_action_derivative_discriminator import run

    baseline = {
        "total_steps": 1_880_064,
        "checkpoint_steps": (1_880_064,),
        "unroll_length": 24,
        "ahac_horizon_min": 8,
        "ahac_horizon_max": 24,
        "actor_bootstrap_scale": 0.0,
        "actor_bootstrap_graph_mode": "connected",
        "actor_forward_jvp_probe": False,
        "actor_h1_action_derivative_probe": False,
        "determinism_probe_output": "/old/probe.json",
        "unchanged": object(),
    }
    monkeypatch.setattr(
        run,
        "build_source_probe_kwargs",
        lambda *args, **kwargs: dict(baseline),
    )

    kwargs = run.build_probe_kwargs(
        "g1-4x5",
        "/reference.npz",
        0,
        "/checkpoint.pkl",
        probe_output="/h1.json",
    )

    assert kwargs["total_steps"] == run.H1_END_STEP
    assert kwargs["checkpoint_steps"] == (run.H1_END_STEP,)
    assert kwargs["unroll_length"] == 1
    assert kwargs["ahac_horizon_min"] == 1
    assert kwargs["ahac_horizon_max"] == 1
    assert kwargs["actor_bootstrap_scale"] == 0.0
    assert kwargs["actor_bootstrap_graph_mode"] == "excised"
    assert kwargs["actor_forward_jvp_probe"] is False
    assert kwargs["actor_h1_action_derivative_probe"] is True
    assert kwargs["determinism_probe_output"] == "/h1.json"
    assert kwargs["unchanged"] is baseline["unchanged"]


def _report(**changes):
    baseline = {
        "protocol": "shac-h1-action-derivative-pair-v1",
        "valid": True,
        "population_size": 512,
        "action_dimension": 29,
        "source_primal_matches_action_reverse": True,
        "source_primal_matches_action_forward": True,
        "repeat_exact": True,
        "metadata_exact": True,
        "policy_reverse_finite_count": 491,
        "action_reverse_finite_count": 491,
        "action_forward_finite_count": 512,
        "forward_recovers_action_reverse_count": 21,
        "forward_recovers_policy_reverse_count": 21,
        "action_reverse_mask_matches_policy_reverse": True,
        "jointly_finite_count": 491,
        "jointly_finite_gradient_agreement_count": 491,
        "finite_difference_finite_count": 512,
        "finite_difference_agreement_count": 500,
    }
    return {**baseline, **changes}


def test_classification_distinguishes_reverse_transpose_from_shared_failure() -> None:
    from experiments.g1_ahac_h1_action_derivative_discriminator.run import (
        classify_action_derivatives,
    )

    bypass = classify_action_derivatives(_report())
    assert bypass["outcome"] == "forward-action-bypasses-reverse-transpose"
    assert bypass["scientifically_interpretable"] is True

    shared = classify_action_derivatives(
        _report(
            action_forward_finite_count=491,
            forward_recovers_action_reverse_count=0,
            forward_recovers_policy_reverse_count=0,
            jointly_finite_gradient_agreement_count=491,
            finite_difference_finite_count=491,
            finite_difference_agreement_count=480,
        )
    )
    assert shared["outcome"] == "forward-action-shares-reverse-failure"
    assert shared["scientifically_interpretable"] is True


def test_classification_fails_closed_on_any_primal_change() -> None:
    from experiments.g1_ahac_h1_action_derivative_discriminator.run import (
        classify_action_derivatives,
    )

    result = classify_action_derivatives(
        _report(valid=False, source_primal_matches_action_forward=False)
    )
    assert result["outcome"] == "invalid-source-primal"
    assert result["scientifically_interpretable"] is False


def test_runner_uses_one_nonretained_train_invocation() -> None:
    from experiments.g1_ahac_h1_action_derivative_discriminator import run

    source = inspect.getsource(run._run)
    assert source.count("train(**kwargs)") == 1
    assert "subprocess" not in source
    assert "policy_retained" in source
    assert "persisted_optimizer_updates" in source
