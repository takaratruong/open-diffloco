from pathlib import Path


def test_full_actor_recovery_runner_has_exact_bounded_contract():
    from tools.run_g1_rmr_full_actor_recovery import (
        build_rmr_full_actor_recovery_kwargs,
    )

    reference = Path("/tmp/exact-rmr-motion.npz")
    source_actor = object()
    candidate = build_rmr_full_actor_recovery_kwargs(
        "g1-4x5",
        reference,
        seed=0,
        source_actor=source_actor,
    )

    assert candidate["total_steps"] == 49_152
    assert candidate["checkpoint_interval"] == 24_576
    assert candidate["num_envs"] == 128
    assert candidate["gradient_accumulation_steps"] == 2
    assert candidate["unroll_length"] == 12
    assert (
        candidate["total_steps"]
        // (
            candidate["num_envs"]
            * candidate["gradient_accumulation_steps"]
            * candidate["unroll_length"]
        )
        == 16
    )
    assert candidate["actor_lr"] == 1e-4
    assert candidate["actor_bootstrap_scale"] == 0.0
    assert candidate["action_noise_std_start"] == 0.05
    assert candidate["action_noise_std_end"] == 0.05
    assert candidate["actor_history_len"] == 1
    assert candidate["actor_reference_lookahead_steps"] == ()
    assert candidate["initial_full_actor_policy"] is source_actor
    assert candidate["env_variant"] == "g1_tracking_rmr_50hz_action_parity"
    assert candidate["reference_residual_control"] is True
    assert candidate["reference_residual_scale"] == 0.5
    assert candidate["domain_randomization"] is False
    assert candidate["actor_observation_noise"] is False
    assert candidate["reference_reset_noise_scale"] == 0.0
    assert candidate["friction_range"] == (1.0, 1.0)
    assert candidate["mass_range"] == (1.0, 1.0)
    assert candidate["kp_range"] == (35.0, 35.0)
    assert candidate["kd_range"] == (0.5, 0.5)
    assert candidate["com_offset_range"] == (0.0, 0.0, 0.0)
    assert candidate["push_velocity_range"] == (0.0, 0.0)
    assert candidate["actor_cagrad"] is True
    assert candidate["actor_phase_bin_count"] == 5


def test_full_actor_recovery_parser_requires_source_and_hides_tuning():
    from tools.run_g1_rmr_full_actor_recovery import build_parser

    parser = build_parser()
    for argv in (
        ["--solver-profile", "g1-4x5"],
        [
            "--solver-profile",
            "g1-4x5",
            "--source-policy-checkpoint",
            "/tmp/source.pt",
            "--actor-lr",
            "0.001",
        ],
    ):
        try:
            parser.parse_args(argv)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"parser unexpectedly accepted {argv}")
