from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


EXPECTED_MODEL_999_STD = np.asarray(
    [
        0.42593249678611755,
        0.4301448464393616,
        0.377777636051178,
        0.4007538855075836,
        0.4021025598049164,
        0.45319169759750366,
        0.3858374059200287,
        0.37606239318847656,
        0.45214489102363586,
        0.41382259130477905,
        0.418007493019104,
        0.4800826609134674,
        0.4738956689834595,
        0.4416677951812744,
        0.4364272654056549,
        0.46375957131385803,
        0.46326711773872375,
        0.4293907582759857,
        0.42477574944496155,
        0.4214183986186981,
        0.419751912355423,
        0.4488624930381775,
        0.45025548338890076,
        0.45707160234451294,
        0.457524836063385,
        0.5523859858512878,
        0.5578638315200806,
        0.5592747926712036,
        0.5565704703330994,
    ],
    dtype=np.float32,
)


def test_pinned_walk_model_999_noise_preserves_exact_vector_and_provenance() -> None:
    from src.core.rmr_action_noise import (
        RMR_WALK_MODEL_999_ACTION_STD,
        RMR_WALK_MODEL_999_SHA256,
    )

    actual = np.asarray(RMR_WALK_MODEL_999_ACTION_STD)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, EXPECTED_MODEL_999_STD)
    assert RMR_WALK_MODEL_999_SHA256 == (
        "5db9d8371754a635d162c416e192b49ec2064d3133d20eea0df63463d1c8ae03"
    )


def test_builder_changes_only_the_rmr_noise_schedule() -> None:
    from tools.run_g1_fresh_full_action_h24_walk import (
        build_fresh_full_action_h24_kwargs,
    )
    from tools.run_g1_rmr_noise_h24_walk import (
        TOTAL_STEPS,
        build_rmr_noise_h24_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    baseline = build_fresh_full_action_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    changed = {
        key
        for key in set(baseline) | set(treatment)
        if not np.array_equal(baseline.get(key), treatment.get(key))
    }

    assert changed == {
        "action_noise_std_start",
        "action_noise_std_end",
        "action_noise_schedule_steps",
    }
    assert treatment["action_noise_std_start"] == 1.0
    np.testing.assert_array_equal(
        np.asarray(treatment["action_noise_std_end"]), EXPECTED_MODEL_999_STD
    )
    assert treatment["action_noise_schedule_steps"] == TOTAL_STEPS == 1_572_864
    assert treatment["unroll_length"] == 24
    assert treatment["num_envs"] == 256
    assert treatment["gradient_accumulation_steps"] == 2
    assert treatment["actor_observation_noise"] is False
    assert treatment["reference_reset_noise_scale"] == 0.0
    assert treatment["domain_randomization"] is False
    assert treatment["push_velocity_range"] == (0.0, 0.0)
    assert treatment["torso_wrench_assistance"] is False


def _valid_cagrad_row(*, action_noise_current) -> dict[str, object]:
    return {
        "actor_cagrad_valid": True,
        "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
        "actor_cagrad_bin_gradient_norms": [0.2] * 5,
        "actor_cagrad_bin_losses": [-0.1] * 5,
        "actor_cagrad_weights": [0.2] * 5,
        "actor_cagrad_gram_matrix": np.eye(5).tolist(),
        "actor_cagrad_cosine_matrix": np.eye(5).tolist(),
        "actor_cagrad_objective": 0.1,
        "actor_cagrad_dual_gap": 0.0,
        "actor_cagrad_uniform_combined_cosine": 0.5,
        "actor_cagrad_combined_norm": 0.2,
        "actor_bootstrap_scale_current": 0.0,
        "action_noise_current": action_noise_current,
        "actor_grad_raw_median": 0.1,
        "actor_grad_raw_max": 0.2,
        "actor_grad_finite_fraction": 1.0,
        "actor_grad_post_clip_median": 0.1,
        "actor_grad_post_clip_max": 0.2,
        "actor_grad_clipped_fraction": 0.0,
    }


def test_cagrad_validator_accepts_exact_vector_noise_and_rejects_drift() -> None:
    from tools.run_g1_fresh_ppo_action_contract_walk import _validate_cagrad_row

    expected = EXPECTED_MODEL_999_STD.tolist()
    _validate_cagrad_row(
        _valid_cagrad_row(action_noise_current=expected),
        step=1_572_864,
        expected_action_noise=expected,
    )
    drifted = expected.copy()
    drifted[0] += 1e-3
    with pytest.raises(ValueError, match="CAGrad telemetry"):
        _validate_cagrad_row(
            _valid_cagrad_row(action_noise_current=drifted),
            step=1_572_864,
            expected_action_noise=expected,
        )


def test_preflight_records_the_only_delta_and_source_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_rmr_noise_h24_walk as runner

    monkeypatch.setattr(
        runner,
        "validate_h24_preflight",
        lambda **_: {"protocol": "parent", "valid": True},
    )
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="abc",
    )

    assert report["valid"] is True
    assert report["scientific_delta"] == [
        "action_noise_std_start",
        "action_noise_std_end",
        "action_noise_schedule_steps",
    ]
    assert report["rmr_walk_model_999_sha256"] == (
        "5db9d8371754a635d162c416e192b49ec2064d3133d20eea0df63463d1c8ae03"
    )
    assert report["action_noise_std_start"] == 1.0
    np.testing.assert_array_equal(
        np.asarray(report["action_noise_std_end"]), EXPECTED_MODEL_999_STD
    )


def test_parser_requires_the_provenance_commit() -> None:
    from tools.run_g1_rmr_noise_h24_walk import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
