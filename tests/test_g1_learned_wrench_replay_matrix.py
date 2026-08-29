from pathlib import Path


def test_replay_matrix_is_complete_and_duplicate():
    from tools.evaluate_g1_learned_wrench_replay_matrix import MATRIX_CONDITIONS

    assert MATRIX_CONDITIONS == (
        ("legacy-default-a", "legacy", "default"),
        ("legacy-default-b", "legacy", "default"),
        ("legacy-deterministic-a", "legacy", "deterministic"),
        ("legacy-deterministic-b", "legacy", "deterministic"),
        ("current-default-a", "current", "default"),
        ("current-default-b", "current", "default"),
        ("current-deterministic-a", "current", "deterministic"),
        ("current-deterministic-b", "current", "deterministic"),
    )


def test_commands_preserve_hparams_and_adapt_only_current_mask_cli(tmp_path):
    from tools.evaluate_g1_learned_wrench_replay_matrix import (
        build_evaluator_command,
    )

    common = dict(
        python=tmp_path / "python",
        checkpoint=tmp_path / "checkpoint.pkl",
        reference=tmp_path / "reference.npz",
        output_dir=tmp_path / "output",
        phase=0,
        solver_profile="g1-4x5",
    )
    legacy = build_evaluator_command(
        evaluator=tmp_path / "legacy.py", source="legacy", **common
    )
    current = build_evaluator_command(
        evaluator=tmp_path / "current.py", source="current", **common
    )

    for command in (legacy, current):
        assert command[command.index("--env-variant") + 1] == (
            "g1_tracking_rmr_50hz_action_parity"
        )
        assert command[command.index("--reference-stride") + 1] == "1"
        assert command[command.index("--actor-history-len") + 1] == "10"
        assert command[command.index("--actor-reference-preview-mode") + 1] == (
            "delta"
        )
        assert "--reference-residual-control" in command
        assert command[command.index("--reference-residual-scale") + 1] == "1.0"
        assert command[command.index("--actor-reference-lookahead-steps") + 1 :] == [
            "4",
            "8",
            "12",
        ]
    assert "--learned-wrench-components" not in legacy
    assert current[current.index("--learned-wrench-components") + 1] == "full"


def test_child_environment_isolates_only_deterministic_xla():
    from tools.evaluate_g1_learned_wrench_replay_matrix import child_environment

    ambient = {
        "PATH": "/bin",
        "XLA_FLAGS": "ambient-is-forbidden",
        "CUDA_VISIBLE_DEVICES": "2",
    }
    default = child_environment(ambient, execution="default")
    deterministic = child_environment(ambient, execution="deterministic")

    assert "XLA_FLAGS" not in default
    assert deterministic["XLA_FLAGS"] == "--xla_gpu_exclude_nondeterministic_ops"
    assert default["CUDA_VISIBLE_DEVICES"] == "2"
    assert deterministic["CUDA_VISIBLE_DEVICES"] == "2"


def test_matrix_classification_requires_repeatable_completion():
    from tools.evaluate_g1_learned_wrench_replay_matrix import classify_matrix

    rows = {}
    for source in ("legacy", "current"):
        for execution in ("default", "deterministic"):
            rows[f"{source}-{execution}"] = {
                "steps": [271, 271],
                "content_exact": True,
            }
    result = classify_matrix(rows)
    assert result["outcome"] == "repeatable-positive-control-found"
    assert result["eligible_routes"] == [
        "legacy-default",
        "legacy-deterministic",
        "current-default",
        "current-deterministic",
    ]

    rows["current-deterministic"] = {
        "steps": [176, 176],
        "content_exact": True,
    }
    result = classify_matrix(rows)
    assert result["outcome"] == "repeatable-positive-control-found"
    assert "xla-effect-current" in result["attribution"]
    assert "current-deterministic" not in result["eligible_routes"]

    for row in rows.values():
        row["steps"] = [176, 176]
    result = classify_matrix(rows)
    assert result["outcome"] == "no-repeatable-positive-control"


def test_parser_requires_both_source_roots(tmp_path):
    from tools.evaluate_g1_learned_wrench_replay_matrix import build_parser

    args = build_parser().parse_args(
        [
            "--legacy-repository",
            str(tmp_path / "legacy"),
            "--current-repository",
            str(tmp_path / "current"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pkl"),
            "--checkpoint-sha256",
            "a" * 64,
            "--reference-path",
            str(tmp_path / "reference.npz"),
            "--reference-sha256",
            "b" * 64,
            "--historical-evaluation",
            str(tmp_path / "historical.npz"),
            "--historical-evaluation-sha256",
            "c" * 64,
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert args.legacy_repository == Path(tmp_path / "legacy")
    assert args.current_repository == Path(tmp_path / "current")
