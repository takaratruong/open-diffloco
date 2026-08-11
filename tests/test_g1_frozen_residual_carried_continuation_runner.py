import hashlib
from pathlib import Path

import pytest


def test_carried_continuation_changes_only_reset_distribution_and_endpoint(
    tmp_path: Path,
) -> None:
    from tools.run_g1_frozen_residual_carried_continuation import (
        build_frozen_residual_carried_kwargs,
    )
    from tools.run_g1_frozen_residual_preview_continuation import (
        build_frozen_residual_preview_kwargs,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e008.pkl"
    bank = tmp_path / "carried.npz"
    parent = build_frozen_residual_preview_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_frozen_residual_carried_kwargs(
        "g1-4x5", reference, 0, checkpoint, bank
    )
    changed = {
        "total_steps",
        "carried_reset_bank_path",
        "carried_reset_probability",
        "carried_reset_bank_start",
        "allow_resume_carried_reset_change",
        "actor_residual_preview_optimizer",
    }

    assert candidate["total_steps"] == 1_720_320
    assert candidate["checkpoint_interval"] == 49_152
    assert candidate["actor_residual_preview_adapter"] is True
    assert candidate["actor_residual_preview_hidden"] == 256
    assert candidate["actor_residual_preview_optimizer"] == "adam"
    assert parent.get("actor_residual_preview_optimizer", "adam") == "adam"
    assert candidate["carried_reset_bank_path"] == str(bank.resolve())
    assert candidate["carried_reset_probability"] == 0.5
    assert candidate["carried_reset_bank_start"] == 0
    assert candidate["allow_resume_carried_reset_change"] is True
    assert {
        key: value for key, value in candidate.items() if key not in changed
    } == {key: value for key, value in parent.items() if key not in changed}


def test_carried_continuation_parser_has_no_scientific_overrides() -> None:
    from tools.run_g1_frozen_residual_carried_continuation import build_parser

    parser = build_parser()
    required = [
        "--solver-profile",
        "g1-4x5",
        "--resume-from",
        "/tmp/e008.pkl",
        "--carried-reset-bank",
        "/tmp/carried.npz",
        "--carried-reset-bank-sha256",
        "0" * 64,
    ]
    args = parser.parse_args(required)
    assert args.carried_reset_bank == Path("/tmp/carried.npz")
    assert args.carried_reset_bank_sha256 == "0" * 64
    for override in (
        ["--carried-reset-probability", "0.75"],
        ["--actor-residual-preview-optimizer", "muon"],
        ["--num-envs", "512"],
        ["--unroll-length", "24"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, *override])


def test_carried_continuation_validates_bank_sha256(tmp_path: Path) -> None:
    from tools.run_g1_frozen_residual_carried_continuation import (
        validate_file_sha256,
    )

    bank = tmp_path / "carried.npz"
    bank.write_bytes(b"immutable carried bank")
    expected = hashlib.sha256(bank.read_bytes()).hexdigest()

    assert validate_file_sha256(bank, expected) == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_file_sha256(bank, "0" * 64)
