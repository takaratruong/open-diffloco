from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_continuation_builder_preserves_e023_and_adds_128_updates() -> None:
    from tools.run_g1_rmr_noise_h24_continuation import (
        CONTINUATION_END_STEP,
        START_STEP,
        build_rmr_noise_h24_continuation_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = Path("/tmp/walk.npz")
    resume = Path("/tmp/checkpoint_step_1572864.pkl")
    parent = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_rmr_noise_h24_continuation_kwargs(
        "g1-4x5", reference, 0, resume
    )
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }

    assert changed == {"resume_from", "total_steps"}
    assert START_STEP == 1_572_864
    assert CONTINUATION_END_STEP == 3_145_728
    assert treatment["resume_from"] == str(resume.resolve())
    assert treatment["total_steps"] == CONTINUATION_END_STEP
    assert treatment["checkpoint_interval"] == 196_608
    assert treatment["action_noise_schedule_steps"] == START_STEP
    assert treatment["reference_reset_noise_scale"] == 0.0
    assert expected_checkpoint_steps() == tuple(
        range(1_769_472, CONTINUATION_END_STEP + 1, 196_608)
    )


def test_continuation_preflight_pins_parent_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_rmr_noise_h24_continuation as runner

    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"valid": True, "protocol": "parent"},
    )
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: (
            runner.EXPECTED_RESUME_HPARAMS_SHA256
            if path.name == "hparams.json"
            else runner.EXPECTED_RESUME_SHA256
        ),
    )
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        resume_from=Path("/tmp/checkpoint_step_1572864.pkl"),
        code_commit="a" * 40,
    )

    assert report["valid"] is True
    assert report["checkpoint_sha256"] == (
        "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f"
    )
    assert report["hparams_sha256"] == (
        "a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650"
    )
    assert report["scientific_delta"] == [
        "resume_from",
        "total_steps",
    ]


def test_continuation_parser_requires_resume_and_commit() -> None:
    from tools.run_g1_rmr_noise_h24_continuation import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
