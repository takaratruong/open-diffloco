from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


CONTROL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}


def test_builder_changes_only_replay_and_execution_budget() -> None:
    from tools.run_g1_demonstration_replay_h24_walk import (
        TOTAL_STEPS,
        build_demonstration_replay_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = Path("/tmp/walk.npz")
    parent = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_demonstration_replay_kwargs(
        "g1-4x5", reference, 0
    )
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }

    assert changed == {"demonstration_replay_threshold", "total_steps"}
    assert treatment["demonstration_replay_threshold"] == 0.2
    assert treatment["total_steps"] == TOTAL_STEPS == 393_216
    assert treatment["action_noise_schedule_steps"] == 1_572_864
    assert expected_checkpoint_steps() == (196_608, 393_216)


@pytest.mark.parametrize(
    ("treatment", "fractions", "expected"),
    [
        (
            {16: (43, 36, 48, 49, 24), 32: CONTROL[32]},
            {16: 0.2, 32: 0.1},
            "demo-replay-early-advances",
        ),
        (CONTROL, {16: 0.2, 32: 0.1}, "demo-replay-early-parity"),
        (CONTROL, {16: 0.95, 32: 0.1}, "demo-replay-overassisted"),
        (
            {16: (46, 32, 48, 49, 24), 32: (45, 46, 53, 49, 24)},
            {16: 0.2, 32: 0.1},
            "demo-replay-early-mixed",
        ),
        (
            {16: (38, 34, 45, 46, 24), 32: (41, 46, 49, 45, 24)},
            {16: 0.2, 32: 0.1},
            "demo-replay-early-underperforms",
        ),
    ],
)
def test_classifier_covers_registered_outcomes(
    treatment, fractions, expected
) -> None:
    from tools.run_g1_demonstration_replay_h24_walk import (
        classify_demonstration_replay,
    )

    assert classify_demonstration_replay(treatment, fractions) == expected


def test_replay_telemetry_validator_requires_nontrivial_bounded_events(
    tmp_path: Path,
) -> None:
    from tools.run_g1_demonstration_replay_h24_walk import (
        validate_replay_telemetry,
    )

    path = tmp_path / "checkpoint_phase_metrics.json"
    rows = [
        {
            "step": 196_608,
            "demonstration_replay_threshold": 0.2,
            "demonstration_replay_count": 100,
            "demonstration_replay_fraction": 0.1,
            "demonstration_replay_valid": True,
        },
        {
            "step": 393_216,
            "demonstration_replay_threshold": 0.2,
            "demonstration_replay_count": 50,
            "demonstration_replay_fraction": 0.05,
            "demonstration_replay_valid": True,
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert validate_replay_telemetry(path) == {16: 0.1, 32: 0.05}
    rows[0]["demonstration_replay_fraction"] = 1.1
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded"):
        validate_replay_telemetry(path)


def test_parser_requires_code_commit() -> None:
    from tools.run_g1_demonstration_replay_h24_walk import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
