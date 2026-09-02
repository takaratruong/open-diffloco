import math


def test_gradient_audit_is_one_exact_e002_update() -> None:
    from experiments.g1_gradient_population_audit.run import (
        END_STEP,
        START_STEP,
        build_gradient_audit_kwargs,
    )

    kwargs = build_gradient_audit_kwargs(
        "g1-4x5",
        "/tmp/reference.npz",
        0,
        "/tmp/checkpoint.pkl",
    )

    assert END_STEP - START_STEP == 512 * 24
    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == (END_STEP,)
    assert kwargs["actor_cagrad"] is True
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["num_envs"] == 256
    assert kwargs["unroll_length"] == 24
    assert kwargs["tracking_anchor_position_kernel"] == "exponential"
    assert kwargs["tracking_root_velocity_weight"] == 1.0
    assert kwargs["diagnose"] is True


def test_gradient_distribution_summary_reconstructs_batch_esnr() -> None:
    from experiments.g1_gradient_population_audit.run import (
        summarize_gradient_distribution,
    )

    summary = summarize_gradient_distribution(
        {
            "step": 1_880_064,
            "actor_grad_population_mean_norm": 0.5,
            "actor_grad_population_rms_norm": 1.0,
            "actor_grad_population_variance_trace": 0.75,
            "actor_grad_population_cancellation_ratio": 0.5,
            "actor_grad_population_noise_scale": 3.0,
            "actor_grad_population_esnr": 512.0 / 3.0,
            "actor_grad_finite_fraction": 1.0,
            "actor_cagrad_valid": True,
        },
        expected_step=1_880_064,
    )

    assert summary["valid"] is True
    assert summary["optimizer_update_retained"] is False
    assert summary["estimated_batch_esnr"]["1"] == 1.0 / 3.0
    assert math.isclose(summary["estimated_batch_esnr"]["8"], 8.0 / 3.0)
    assert math.isclose(summary["estimated_batch_esnr"]["512"], 512.0 / 3.0)
    assert summary["classification"] == "population-mean-has-material-cancellation"
