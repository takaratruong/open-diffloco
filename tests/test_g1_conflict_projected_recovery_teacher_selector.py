from __future__ import annotations

import pytest


def test_selector_cli_is_a_pinned_e042_aggregate_wrapper():
    from tools.select_g1_conflict_projected_recovery_teacher import (
        OUTCOME_LABELS,
        build_parser,
    )

    assert OUTCOME_LABELS == {
        "solve": "teacher-objective-solve",
        "advance": "teacher-objective-advance",
        "insufficient": "teacher-objective-insufficient",
    }
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--evaluation-root", "/tmp/evaluations",
            "--training-validation", "/tmp/training.json",
            "--output", "/tmp/selection.json",
            "--code-commit", "a" * 40,
        ]
    )
    assert str(args.output) == "/tmp/selection.json"
