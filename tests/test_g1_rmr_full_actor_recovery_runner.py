from pathlib import Path


def test_full_scale_phase_grid_contract_is_explicit_and_legacy_safe():
    from tools.evaluate_g1_rmr_phase_grid import (
        build_parser as build_phase_grid_parser,
        phase_grid_action_contract,
    )

    parser = build_phase_grid_parser()
    required = [
        "--source-policy-checkpoint",
        "/tmp/source.pt",
        "--output",
        "/tmp/grid.json",
    ]
    assert parser.parse_args(required).reference_residual_scale == 0.5
    args = parser.parse_args(
        [*required, "--reference-residual-scale", "1.0"]
    )
    assert args.reference_residual_scale == 1.0
    assert phase_grid_action_contract(1.0) == {
        "environment_variant": "g1_tracking_rmr_50hz_source_step",
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "squash_actor_actions": False,
    }


def test_full_scale_phase_grid_rejects_an_unregistered_scale():
    from tools.evaluate_g1_rmr_phase_grid import build_parser

    parser = build_parser()
    try:
        parser.parse_args(
            [
                "--source-policy-checkpoint",
                "/tmp/source.pt",
                "--output",
                "/tmp/grid.json",
                "--reference-residual-scale",
                "0.75",
            ]
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("phase grid accepted an unregistered scale")


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


def test_full_scale_recovery_is_explicit_and_preserves_action_parity():
    from tools.run_g1_rmr_full_actor_recovery import (
        build_rmr_full_actor_recovery_kwargs,
    )

    candidate = build_rmr_full_actor_recovery_kwargs(
        "g1-4x5",
        Path("/tmp/exact-rmr-motion.npz"),
        seed=0,
        source_actor=object(),
        reference_residual_scale=1.0,
    )

    assert candidate["reference_residual_scale"] == 1.0
    assert candidate["env_variant"] == "g1_tracking_rmr_50hz_action_parity"


def test_full_scale_recovery_parser_is_choices_constrained():
    from tools.run_g1_rmr_full_actor_recovery import build_parser

    parser = build_parser()
    required = [
        "--solver-profile",
        "g1-4x5",
        "--source-policy-checkpoint",
        "/tmp/source.pt",
    ]
    assert parser.parse_args(required).reference_residual_scale == 0.5
    assert (
        parser.parse_args(
            [*required, "--reference-residual-scale", "1.0"]
        ).reference_residual_scale
        == 1.0
    )


def test_full_actor_recovery_forwards_explicit_policy_anchor():
    from tools.run_g1_rmr_full_actor_recovery import (
        build_rmr_full_actor_recovery_kwargs,
    )

    candidate = build_rmr_full_actor_recovery_kwargs(
        "g1-4x5",
        Path("/tmp/exact-rmr-motion.npz"),
        seed=0,
        source_actor=object(),
        actor_policy_anchor_weight=1.0,
    )

    assert candidate["actor_policy_anchor_weight"] == 1.0


def test_full_actor_recovery_supports_a_preregistered_longer_update_budget():
    from tools.run_g1_rmr_full_actor_recovery import (
        build_parser,
        build_rmr_full_actor_recovery_kwargs,
    )

    candidate = build_rmr_full_actor_recovery_kwargs(
        "g1-4x5",
        Path("/tmp/exact-rmr-motion.npz"),
        seed=0,
        source_actor=object(),
        total_updates=128,
        checkpoint_updates=16,
    )

    assert candidate["total_steps"] == 393_216
    assert candidate["checkpoint_interval"] == 49_152
    args = build_parser().parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--source-policy-checkpoint",
            "/tmp/source.pt",
            "--total-updates",
            "128",
            "--checkpoint-updates",
            "16",
        ]
    )
    assert args.total_updates == 128
    assert args.checkpoint_updates == 16
