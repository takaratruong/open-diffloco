from pathlib import Path


def test_frozen_preview_recovery_changes_only_preview_parameterization():
    from tools.run_g1_rmr_frozen_preview_recovery import (
        build_rmr_frozen_preview_recovery_kwargs,
    )
    from tools.run_g1_rmr_full_actor_recovery import (
        build_rmr_full_actor_recovery_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    source_actor = object()
    parent = build_rmr_full_actor_recovery_kwargs(
        "g1-4x5",
        reference,
        seed=0,
        source_actor=source_actor,
        reference_residual_scale=1.0,
        total_updates=128,
        checkpoint_updates=16,
    )
    candidate = build_rmr_frozen_preview_recovery_kwargs(
        "g1-4x5",
        reference,
        seed=0,
        source_actor=source_actor,
    )

    changed = {
        key
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }
    assert changed == {
        "actor_preview_adapter",
        "actor_reference_lookahead_steps",
    }
    assert candidate["actor_preview_adapter"] is True
    assert candidate["actor_reference_lookahead_steps"] == (4, 8, 12)
    assert candidate["reference_residual_scale"] == 1.0
    assert candidate["initial_full_actor_policy"] is source_actor
    assert candidate["total_steps"] == 393_216
    assert candidate["checkpoint_interval"] == 49_152


def test_frozen_preview_recovery_parser_requires_source_and_hides_tuning():
    from tools.run_g1_rmr_frozen_preview_recovery import build_parser

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


def test_frozen_preview_recovery_supports_a_short_anchored_discriminator():
    from tools.run_g1_rmr_frozen_preview_recovery import (
        build_parser,
        build_rmr_frozen_preview_recovery_kwargs,
    )

    candidate = build_rmr_frozen_preview_recovery_kwargs(
        "g1-4x5",
        Path("/tmp/walk.npz"),
        seed=0,
        source_actor=object(),
        actor_policy_anchor_weight=1.0,
        total_updates=16,
        checkpoint_updates=8,
    )

    assert candidate["actor_policy_anchor_weight"] == 1.0
    assert candidate["total_steps"] == 49_152
    assert candidate["checkpoint_interval"] == 24_576
    args = build_parser().parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--source-policy-checkpoint",
            "/tmp/source.pt",
            "--actor-policy-anchor-weight",
            "1.0",
            "--total-updates",
            "16",
            "--checkpoint-updates",
            "8",
        ]
    )
    assert args.actor_policy_anchor_weight == 1.0
    assert args.total_updates == 16
    assert args.checkpoint_updates == 8


def test_frozen_preview_recovery_can_pin_short_lr_decay_in_long_run():
    from tools.run_g1_rmr_frozen_preview_recovery import (
        build_parser,
        build_rmr_frozen_preview_recovery_kwargs,
    )

    candidate = build_rmr_frozen_preview_recovery_kwargs(
        "g1-4x5",
        Path("/tmp/walk.npz"),
        seed=0,
        source_actor=object(),
        actor_policy_anchor_weight=1.0,
        total_updates=128,
        checkpoint_updates=8,
        lr_decay_updates=16,
    )

    assert candidate["total_steps"] == 393_216
    assert candidate["lr_decay_updates"] == 16
    args = build_parser().parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--source-policy-checkpoint",
            "/tmp/source.pt",
            "--lr-decay-updates",
            "16",
        ]
    )
    assert args.lr_decay_updates == 16
