from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def test_pair_changes_only_assistance_observability() -> None:
    from tools.run_g1_assistance_observability_pair import (
        ASSISTANCE_END_STEP,
        CONTINUATION_END_STEP,
        E012_SELECTED_STEP,
        build_assistance_observability_pair_kwargs,
    )

    aware, blind = build_assistance_observability_pair_kwargs(
        "g1-4x5",
        Path("reference.npz"),
        0,
        Path("checkpoint.pkl"),
    )
    changed = {
        key
        for key in aware
        if aware[key] != blind[key]
    }

    assert changed == {"actor_observe_torso_wrench_assistance"}
    assert aware["actor_observe_torso_wrench_assistance"] is True
    assert blind["actor_observe_torso_wrench_assistance"] is False
    for kwargs in (aware, blind):
        assert kwargs["total_steps"] == CONTINUATION_END_STEP
        assert kwargs["torso_wrench_assistance_start_step"] == E012_SELECTED_STEP
        assert kwargs["torso_wrench_assistance_end_step"] == ASSISTANCE_END_STEP
        assert kwargs["torso_wrench_assistance_zero_fraction"] == 0.25
        assert kwargs["torso_wrench_assistance_continuous"] is True
        assert kwargs["actor_torso_wrench_assistance_conditioning"] is True
        assert kwargs["allow_resume_assistance_conditioning_change"] is True
        assert kwargs["allow_resume_torso_wrench_assistance_change"] is True
        assert kwargs["resume_random_seed"] == 2
        assert kwargs["checkpoint_interval"] == 49_152
    assert (ASSISTANCE_END_STEP - E012_SELECTED_STEP) // 6_144 == 96
    assert (CONTINUATION_END_STEP - ASSISTANCE_END_STEP) // 6_144 == 32


def test_pair_parser_exposes_only_operational_paths_and_devices() -> None:
    from tools.run_g1_assistance_observability_pair import build_parser

    parser = build_parser()
    parser.parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--resume-from",
            "checkpoint.pkl",
            "--aware-device",
            "0",
            "--blind-device",
            "1",
        ]
    )
    for override in (
        ["--total-steps", "10"],
        ["--assistance-end-step", "10"],
        ["--zero-fraction", "0.5"],
        ["--continuous", "false"],
        ["--observed", "false"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--solver-profile",
                    "g1-4x5",
                    "--resume-from",
                    "checkpoint.pkl",
                    *override,
                ]
            )


def test_pair_preflight_requires_exact_e012_checkpoint(tmp_path: Path) -> None:
    from tools.run_g1_assistance_observability_pair import (
        E012_SELECTED_CHECKPOINT_SHA256,
        validate_parent_checkpoint,
    )

    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="E012 selected checkpoint SHA-256"):
        validate_parent_checkpoint(checkpoint)

    expected = hashlib.sha256(b"wrong").hexdigest()
    assert expected != E012_SELECTED_CHECKPOINT_SHA256
