from pathlib import Path


def test_muon_contract_changes_only_residual_optimizer():
    from tools.run_g1_frozen_residual_muon_continuation import (
        build_frozen_residual_muon_kwargs,
    )
    from tools.run_g1_frozen_residual_preview_continuation import (
        build_frozen_residual_preview_kwargs,
    )

    reference = Path("/tmp/dance.npz")
    checkpoint = Path("/tmp/e008.pkl")
    parent = build_frozen_residual_preview_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_frozen_residual_muon_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )

    assert candidate["actor_residual_preview_optimizer"] == "muon"
    changed = {"actor_residual_preview_optimizer"}
    assert {
        key: value for key, value in candidate.items() if key not in changed
    } == {key: value for key, value in parent.items() if key not in changed}


def test_muon_parser_matches_parent_and_has_no_scientific_overrides():
    from tools.run_g1_frozen_residual_muon_continuation import build_parser

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
    assert args.seed == 0
    assert args.reference_path is not None
    assert args.output_root == Path("g1_frozen_residual_muon_runs")
    for override in (
        ["--actor-residual-preview-optimizer", "adam"],
        ["--actor-residual-preview-hidden", "128"],
        ["--num-envs", "512"],
        ["--unroll-length", "24"],
    ):
        try:
            parser.parse_args([*required, *override])
        except SystemExit:
            continue
        raise AssertionError(f"parser accepted scientific override {override}")
