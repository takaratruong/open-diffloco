from pathlib import Path


def test_residual_preview_contract_changes_only_adapter_kind_and_width():
    from tools.run_g1_frozen_delta_preview_continuation import (
        build_frozen_delta_preview_kwargs,
    )
    from tools.run_g1_frozen_residual_preview_continuation import (
        build_frozen_residual_preview_kwargs,
    )

    reference = Path("/tmp/dance.npz")
    checkpoint = Path("/tmp/e008.pkl")
    parent = build_frozen_delta_preview_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_frozen_residual_preview_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    changed = {
        "actor_preview_adapter",
        "actor_residual_preview_adapter",
        "actor_residual_preview_hidden",
    }

    assert candidate["actor_preview_adapter"] is False
    assert candidate["actor_residual_preview_adapter"] is True
    assert candidate["actor_residual_preview_hidden"] == 256
    assert candidate["actor_reference_preview_mode"] == "delta"
    assert {
        key: value for key, value in candidate.items() if key not in changed
    } == {key: value for key, value in parent.items() if key not in changed}
    assert candidate["checkpoint_interval"] == 49_152
    assert candidate["total_steps"] == 1_572_864


def test_residual_preview_parser_has_no_scientific_overrides():
    from tools.run_g1_frozen_residual_preview_continuation import build_parser

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
        ["--actor-residual-preview-hidden", "128"],
        ["--actor-reference-preview-mode", "absolute"],
        ["--num-envs", "512"],
        ["--unroll-length", "24"],
    ):
        try:
            parser.parse_args([*required, *override])
        except SystemExit:
            continue
        raise AssertionError(f"parser accepted scientific override {override}")
