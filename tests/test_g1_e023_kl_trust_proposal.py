from __future__ import annotations

from pathlib import Path

import numpy as np


def test_proposal_builder_changes_only_one_update_endpoint_and_resume(tmp_path):
    from tools.run_g1_e023_kl_trust_proposal import (
        END_STEP,
        START_STEP,
        TRANSITIONS_PER_UPDATE,
        build_e023_kl_trust_proposal_kwargs,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = tmp_path / "reference.npz"
    checkpoint = tmp_path / "checkpoint_step_1572864.pkl"
    base = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_e023_kl_trust_proposal_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )

    assert END_STEP == START_STEP + TRANSITIONS_PER_UPDATE == 1_585_152
    assert treatment["resume_from"] == str(checkpoint.resolve())
    assert treatment["total_steps"] == END_STEP
    assert treatment["checkpoint_interval"] == TRANSITIONS_PER_UPDATE
    for key, value in base.items():
        if key in {"resume_from", "total_steps", "checkpoint_interval"}:
            continue
        if hasattr(value, "shape"):
            np.testing.assert_array_equal(treatment[key], value)
        else:
            assert treatment[key] == value


def test_proposal_preserves_e023_scientific_contract(tmp_path):
    from tools.run_g1_e023_kl_trust_proposal import (
        START_STEP,
        build_e023_kl_trust_proposal_kwargs,
        expected_checkpoint_steps,
    )

    kwargs = build_e023_kl_trust_proposal_kwargs(
        "g1-4x5",
        tmp_path / "reference.npz",
        0,
        tmp_path / "checkpoint.pkl",
    )

    assert expected_checkpoint_steps() == (1_585_152,)
    assert kwargs["action_noise_schedule_steps"] == START_STEP
    assert kwargs["tracking_velocity_kernel"] == "exponential"
    assert kwargs["num_envs"] == 256
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["unroll_length"] == 24
    assert kwargs["actor_cagrad"] is True
    assert kwargs["actor_per_env_grad_clip"] == 1.0
    assert kwargs["actor_bootstrap_scale"] == 0.0
    assert kwargs["reference_reset_noise_scale"] == 0.0
    assert kwargs["domain_randomization"] is False


def test_proposal_parser_requires_pinned_runtime_paths():
    from tools.run_g1_e023_kl_trust_proposal import build_parser

    args = build_parser().parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--reference-path",
            "/tmp/reference.npz",
            "--resume-from",
            "/tmp/source.pkl",
            "--output-root",
            "/tmp/output",
            "--code-commit",
            "a" * 40,
        ]
    )

    assert args.seed == 0
    assert args.reference_path == Path("/tmp/reference.npz")
    assert args.resume_from == Path("/tmp/source.pkl")
    assert args.code_commit == "a" * 40


def test_preflight_adds_exact_source_and_single_update_contract(monkeypatch, tmp_path):
    import tools.run_g1_e023_kl_trust_proposal as runner

    source = tmp_path / "checkpoint_step_1572864.pkl"
    hparams = tmp_path / "hparams.json"
    source.write_bytes(b"source")
    hparams.write_bytes(b"hparams")
    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"protocol": "parent", "code_commit": "b" * 40},
    )
    hashes = {
        source.resolve(): runner.EXPECTED_RESUME_SHA256,
        hparams.resolve(): runner.EXPECTED_RESUME_HPARAMS_SHA256,
    }
    monkeypatch.setattr(runner, "sha256_file", lambda path: hashes[path.resolve()])

    report = runner.validate_preflight(
        repository=tmp_path,
        reference_path=tmp_path / "reference.npz",
        resume_from=source,
        code_commit="b" * 40,
    )

    assert report["protocol"] == "g1-e023-kl-trust-proposal-preflight-v1"
    assert report["start_step"] == runner.START_STEP
    assert report["end_step"] == runner.END_STEP
    assert report["total_steps"] == runner.END_STEP
    assert report["total_updates"] == 129
    assert report["checkpoint_interval"] == runner.CHECKPOINT_INTERVAL
    assert report["checkpoint_updates"] == 1
    assert report["checkpoint_steps"] == [runner.END_STEP]
    assert report["scientific_delta"] == [
        "resume_from",
        "total_steps",
        "checkpoint_interval",
    ]
