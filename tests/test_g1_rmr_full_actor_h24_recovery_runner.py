from pathlib import Path


def test_h24_recovery_changes_only_horizon_population_pair():
    from tools.run_g1_rmr_full_actor_h24_recovery import (
        build_rmr_full_actor_h24_recovery_kwargs,
    )
    from tools.run_g1_rmr_full_actor_recovery import (
        build_rmr_full_actor_recovery_kwargs,
    )

    reference = Path("/tmp/exact-rmr-motion.npz")
    source_actor = object()
    parent = build_rmr_full_actor_recovery_kwargs(
        "g1-4x5", reference, seed=0, source_actor=source_actor
    )
    candidate = build_rmr_full_actor_h24_recovery_kwargs(
        "g1-4x5", reference, seed=0, source_actor=source_actor
    )

    changed = {
        key
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }
    assert changed == {"num_envs", "unroll_length"}
    assert candidate["num_envs"] == 64
    assert candidate["gradient_accumulation_steps"] == 2
    assert candidate["unroll_length"] == 24
    assert candidate["total_steps"] == 49_152
    assert candidate["checkpoint_interval"] == 24_576
    assert (
        candidate["num_envs"]
        * candidate["gradient_accumulation_steps"]
        * candidate["unroll_length"]
        == 3_072
    )


def test_h24_recovery_parser_requires_source_and_hides_tuning():
    from tools.run_g1_rmr_full_actor_h24_recovery import build_parser

    parser = build_parser()
    for argv in (
        ["--solver-profile", "g1-4x5"],
        [
            "--solver-profile",
            "g1-4x5",
            "--source-policy-checkpoint",
            "/tmp/source.pt",
            "--unroll-length",
            "12",
        ],
    ):
        try:
            parser.parse_args(argv)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"parser unexpectedly accepted {argv}")
