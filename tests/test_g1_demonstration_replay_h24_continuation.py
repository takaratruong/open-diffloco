from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_continuation_changes_only_resume_and_absolute_endpoint() -> None:
    from tools.run_g1_demonstration_replay_h24_continuation import (
        END_STEP,
        build_continuation_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_demonstration_replay_h24_walk import (
        build_demonstration_replay_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    resume = Path("/tmp/checkpoint_step_393216.pkl")
    parent = build_demonstration_replay_kwargs("g1-4x5", reference, 0)
    treatment = build_continuation_kwargs(
        "g1-4x5", reference, 0, resume
    )
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }
    assert changed == {"resume_from", "total_steps"}
    assert treatment["demonstration_replay_threshold"] == 0.2
    assert treatment["action_noise_schedule_steps"] == END_STEP == 1_572_864
    assert treatment["resume_from"] == str(resume.resolve())
    assert expected_checkpoint_steps() == (
        589_824,
        786_432,
        983_040,
        1_179_648,
        1_376_256,
        1_572_864,
    )


def test_continuation_replay_telemetry_requires_exact_rows(tmp_path: Path):
    from tools.run_g1_demonstration_replay_h24_continuation import (
        expected_checkpoint_steps,
        validate_continuation_replay_telemetry,
    )

    path = tmp_path / "checkpoint_phase_metrics.json"
    rows = [
        {
            "step": step,
            "demonstration_replay_threshold": 0.2,
            "demonstration_replay_count": 10,
            "demonstration_replay_fraction": 0.01,
            "demonstration_replay_valid": True,
        }
        for step in expected_checkpoint_steps()
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert len(validate_continuation_replay_telemetry(path)) == 6
    rows[-1]["demonstration_replay_count"] = 0
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_continuation_replay_telemetry(path)


def test_continuation_parser_requires_resume_and_commit() -> None:
    from tools.run_g1_demonstration_replay_h24_continuation import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
