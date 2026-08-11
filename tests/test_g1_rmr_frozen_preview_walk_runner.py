from pathlib import Path


def test_walk_runner_preserves_canonical_contract_except_registered_boundary():
    from tools.run_canonical_g1_shac import build_canonical_kwargs
    from tools.run_g1_rmr_frozen_preview_walk import (
        build_rmr_frozen_preview_walk_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    source_actor = object()
    parent = build_canonical_kwargs("g1-4x5", reference, seed=0)
    candidate = build_rmr_frozen_preview_walk_kwargs(
        "g1-4x5", reference, seed=0, source_actor=source_actor
    )

    changed = {
        key
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }
    assert changed == {
        "total_steps",
        "checkpoint_interval",
        "gradient_accumulation_steps",
        "actor_history_len",
        "actor_reference_lookahead_steps",
        "actor_preview_adapter",
        "actor_cagrad",
        "actor_cagrad_alpha",
        "actor_cagrad_iterations",
        "actor_phase_bin_count",
        "initial_full_actor_policy",
    }
    assert candidate["initial_full_actor_policy"] is source_actor
    assert candidate["total_steps"] == 393_216
    assert candidate["checkpoint_interval"] == 49_152
    assert candidate["actor_history_len"] == 1
    assert candidate["actor_reference_lookahead_steps"] == (4, 8, 12)
    assert candidate["actor_preview_adapter"] is True
    assert candidate["actor_cagrad"] is True
    assert candidate["num_envs"] == 256
    assert candidate["gradient_accumulation_steps"] == 2
    assert candidate["unroll_length"] == 12


def test_walk_runner_parser_requires_source_actor_and_hides_overrides():
    from tools.run_g1_rmr_frozen_preview_walk import build_parser

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

