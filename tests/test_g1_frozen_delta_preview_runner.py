from pathlib import Path


def test_delta_preview_contract_changes_only_representation():
    from tools.run_g1_frozen_delta_preview_continuation import (
        build_frozen_delta_preview_kwargs,
    )
    from tools.run_g1_frozen_preview_dense_checkpoint_continuation import (
        build_frozen_preview_dense_checkpoint_kwargs,
    )

    reference = Path("/tmp/dance.npz")
    checkpoint = Path("/tmp/e008.pkl")
    parent = build_frozen_preview_dense_checkpoint_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_frozen_delta_preview_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )

    assert set(candidate) == set(parent) | {
        "actor_reference_preview_mode"
    }
    assert candidate["actor_reference_preview_mode"] == "delta"
    assert {
        key: value
        for key, value in candidate.items()
        if key != "actor_reference_preview_mode"
    } == parent
    assert candidate["termination_margin_weight"] == 0.0
    assert candidate["checkpoint_interval"] == 49_152
    assert candidate["total_steps"] == 1_572_864
    assert candidate["unroll_length"] == 12
    assert (
        candidate["num_envs"]
        * candidate["gradient_accumulation_steps"]
    ) == 512


def test_delta_preview_parser_has_no_scientific_overrides():
    from tools.run_g1_frozen_delta_preview_continuation import build_parser

    parser = build_parser()
    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e008.pkl",
    ]
    args = parser.parse_args(required)
    assert args.solver_profile == "g1-4x5"
    assert args.resume_from == Path("/tmp/e008.pkl")
    for override in (
        ["--actor-reference-preview-mode", "absolute"],
        ["--termination-margin-weight", "0.5"],
        ["--num-envs", "1024"],
        ["--unroll-length", "24"],
    ):
        try:
            parser.parse_args([*required, *override])
        except SystemExit:
            continue
        raise AssertionError(f"parser accepted scientific override {override}")
