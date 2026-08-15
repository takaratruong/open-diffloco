from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_builder_changes_only_reference_reset_noise() -> None:
    from tools.run_g1_fresh_noisy_rsi_h24_walk import (
        build_fresh_noisy_rsi_h24_kwargs,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = Path("/tmp/walk.npz")
    baseline = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_fresh_noisy_rsi_h24_kwargs("g1-4x5", reference, 0)
    changed = {
        key
        for key in set(baseline) | set(treatment)
        if not np.array_equal(baseline.get(key), treatment.get(key))
    }

    assert changed == {"reference_reset_noise_scale"}
    assert treatment["reference_reset_noise_scale"] == 1.0
    assert treatment.get("resume_from") is None
    assert treatment["domain_randomization"] is False
    assert treatment["actor_observation_noise"] is False
    assert treatment["push_velocity_range"] == (0.0, 0.0)
    assert treatment["reference_root_reset_noise_multiplier"] == 1.0
    assert treatment["reference_root_reset_noise_probability"] == 0.0
    assert treatment["carried_reset_probability"] == 0.0
    assert treatment.get("adaptive_phase_sampling", False) is False


def test_preflight_records_fresh_noisy_rsi_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_fresh_noisy_rsi_h24_walk as runner

    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"protocol": "parent", "valid": True},
    )
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="abc",
    )

    assert report["valid"] is True
    assert report["protocol"] == "g1-fresh-noisy-rsi-h24-walk-preflight-v1"
    assert report["scientific_delta"] == ["reference_reset_noise_scale"]
    assert report["reference_reset_noise_scale"] == 1.0
    assert report["fresh_initialization"] is True


def test_parser_requires_the_provenance_commit() -> None:
    from tools.run_g1_fresh_noisy_rsi_h24_walk import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
