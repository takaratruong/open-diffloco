from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_runner_changes_only_zero_head_feature_source():
    from tools.run_g1_e023_lafan_anchored_carried_recovery import (
        build_lafan_recovery_kwargs,
    )
    from tools.run_g1_zero_head_feature_transfer import (
        EXPERT_SHA256,
        build_zero_head_feature_transfer_kwargs,
    )

    args = ("g1-4x5", "/tmp/ref.npz", 0, "/tmp/parent.pkl", "/tmp/bank.npz")
    baseline = build_lafan_recovery_kwargs(*args)
    candidate = build_zero_head_feature_transfer_kwargs(
        *args, expert_path="/tmp/expert.pkl", expert_sha256=EXPERT_SHA256
    )

    assert set(candidate) == set(baseline) | {
        "actor_residual_preview_initial_adapter_path",
        "actor_residual_preview_initial_adapter_sha256",
    }
    for key, value in baseline.items():
        if hasattr(value, "shape"):
            np.testing.assert_array_equal(candidate[key], value)
        else:
            assert candidate[key] == value
    assert candidate["actor_residual_preview_initial_adapter_path"] == str(
        Path("/tmp/expert.pkl").resolve()
    )
    assert candidate["actor_residual_preview_initial_adapter_sha256"] == EXPERT_SHA256


def test_feature_transfer_report_requires_exact_source_and_zero_effect():
    from tools.run_g1_zero_head_feature_transfer import (
        EXPERT_SHA256,
        validate_feature_transfer_report,
    )

    report = {
        "protocol": "g1-zero-head-recovery-feature-transfer-v1",
        "source_sha256": EXPERT_SHA256,
        "input_dim": 328,
        "hidden_dim": 256,
        "action_dim": 29,
        "hidden_kernel_exact": True,
        "hidden_bias_exact": True,
        "output_head_zero": True,
        "parent_parameters_exact": True,
        "initial_action_exact": True,
        "adapter_optimizer_moments_zero": True,
        "valid": True,
    }

    assert validate_feature_transfer_report(report) == {
        "valid": True,
        "protocol": "g1-zero-head-feature-transfer-validation-v1",
    }
    report["initial_action_exact"] = False
    with pytest.raises(ValueError, match="feature transfer report"):
        validate_feature_transfer_report(report)


def test_runner_parser_requires_expert_and_pinned_inputs():
    from tools.run_g1_zero_head_feature_transfer import (
        build_parser,
        validate_registered_hash_arguments,
    )

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--solver-profile", "g1-4x5",
            "--reference-path", "/tmp/ref.npz",
            "--resume-from", "/tmp/parent.pkl",
            "--carried-reset-bank", "/tmp/bank.npz",
            "--carried-reset-bank-sha256", "a" * 64,
            "--carried-reset-bank-summary", "/tmp/bank.json",
            "--carried-reset-bank-summary-sha256", "b" * 64,
            "--expert-checkpoint", "/tmp/expert.pkl",
            "--expert-sha256", "c" * 64,
            "--output-root", "/tmp/out",
            "--code-commit", "d" * 40,
        ]
    )
    assert args.seed == 0
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--solver-profile", "g1-4x5",
                "--reference-path", "/tmp/ref.npz",
                "--resume-from", "/tmp/parent.pkl",
                "--carried-reset-bank", "/tmp/bank.npz",
                "--carried-reset-bank-sha256", "a" * 64,
                "--carried-reset-bank-summary", "/tmp/bank.json",
                "--carried-reset-bank-summary-sha256", "b" * 64,
                "--expert-checkpoint", "/tmp/expert.pkl",
                "--expert-sha256", "c" * 64,
                "--seed", "1",
                "--output-root", "/tmp/out",
                "--code-commit", "d" * 40,
            ]
        )
    with pytest.raises(ValueError, match="registered E027"):
        validate_registered_hash_arguments(
            bank_sha256="a" * 64,
            bank_summary_sha256="b" * 64,
            expert_sha256="c" * 64,
        )
