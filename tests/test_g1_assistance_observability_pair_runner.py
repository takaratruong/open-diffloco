from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

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
            "--code-commit",
            "a" * 40,
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
                    "--code-commit",
                    "a" * 40,
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


def test_pair_preflight_rejects_unknown_or_aliased_devices() -> None:
    from tools.run_g1_assistance_observability_pair import validate_device_ids

    assert validate_device_ids("5", "6", available=("0", "5", "6")) == (
        "5",
        "6",
    )
    with pytest.raises(ValueError, match="distinct GPU"):
        validate_device_ids("5", "5", available=("0", "5", "6"))
    with pytest.raises(ValueError, match="unavailable GPU"):
        validate_device_ids("5", "7", available=("0", "5", "6"))


def test_pair_preflight_binds_runtime_assets(tmp_path: Path) -> None:
    from tools.run_g1_assistance_observability_pair import (
        FROZEN_CONTROLLER_PATH,
        FROZEN_MODEL_PATH,
        FROZEN_REFERENCE_SHA256,
        validate_runtime_assets,
    )

    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="reference SHA-256"):
        validate_runtime_assets(reference)

    report = validate_runtime_assets(
        Path(
            "/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/"
            "artifacts/E-20260808-000/reference/"
            "dance1_subject2_f122_422_50hz.npz"
        )
    )
    assert report["reference_sha256"] == FROZEN_REFERENCE_SHA256
    assert report["model_path"] == str(FROZEN_MODEL_PATH.resolve())
    assert report["controller_path"] == str(FROZEN_CONTROLLER_PATH.resolve())


def test_pair_preflight_binds_exact_clean_code_commit() -> None:
    from tools.run_g1_assistance_observability_pair import (
        validate_code_provenance,
    )

    repository = Path(__file__).resolve().parents[1]
    import subprocess

    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    report = validate_code_provenance(expected)

    assert report["code_commit"] == expected
    assert report["dirty_patch_sha256"] == hashlib.sha256(b"").hexdigest()


def test_zero_tail_evaluation_requires_exact_registered_checkpoint_grid() -> None:
    from tools.run_g1_assistance_observability_pair import (
        ASSISTANCE_END_STEP,
        CONTINUATION_END_STEP,
        ZERO_TAIL_CHECKPOINT_STEPS,
        validate_zero_tail_checkpoints,
    )

    assert ZERO_TAIL_CHECKPOINT_STEPS == (
        2_310_144,
        2_359_296,
        2_408_448,
        2_457_600,
    )
    assert ZERO_TAIL_CHECKPOINT_STEPS[0] > ASSISTANCE_END_STEP
    assert ZERO_TAIL_CHECKPOINT_STEPS[-1] == CONTINUATION_END_STEP
    paths = [Path(f"checkpoint_step_{step}.pkl") for step in ZERO_TAIL_CHECKPOINT_STEPS]
    assert validate_zero_tail_checkpoints(paths, require_files=False) == tuple(paths)
    with pytest.raises(ValueError, match="zero-tail checkpoint grid"):
        validate_zero_tail_checkpoints(paths[:-1], require_files=False)


def test_worker_exit_without_queue_report_becomes_failure() -> None:
    from tools.run_g1_assistance_observability_pair import complete_worker_results

    results = complete_worker_results(
        [("aware", True, "/aware")],
        (
            ("aware", SimpleNamespace(exitcode=0)),
            ("blind", SimpleNamespace(exitcode=9)),
        ),
    )

    assert results[0] == ("aware", True, "/aware")
    assert results[1] == (
        "blind",
        False,
        "worker exited with code 9 without a report",
    )


def test_zero_tail_selector_ranks_minimum_median_mean_then_earliest() -> None:
    from tools.run_g1_assistance_observability_pair import (
        select_zero_tail_checkpoint,
    )

    def payload(sha: str, survival: list[int]) -> dict:
        return {
            "checkpoint_sha256": sha * 64,
            "reference_sha256": (
                "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
            ),
            "solver_profile": "g1-4x5",
            "actor_assistance_conditioning_scale": 0.0,
            "summary": {
                "phases": [0, 100, 200, 300, 400],
                "survival": survival,
            },
        }

    selected = select_zero_tail_checkpoint(
        {
            2_310_144: payload("a", [60, 61, 62, 63, 64]),
            2_359_296: payload("b", [60, 70, 70, 70, 70]),
        }
    )

    assert selected["selected_step"] == 2_359_296
    assert selected["selected_key"] == [60, 70.0, 68.0]
    assert selected["selected_survival"] == [60, 70, 70, 70, 70]


def test_zero_tail_selector_rejects_nonzero_conditioning() -> None:
    from tools.run_g1_assistance_observability_pair import (
        select_zero_tail_checkpoint,
    )

    payload = {
        "checkpoint_sha256": "a" * 64,
        "reference_sha256": (
            "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
        ),
        "solver_profile": "g1-4x5",
        "actor_assistance_conditioning_scale": 0.1,
        "summary": {
            "phases": [0, 100, 200, 300, 400],
            "survival": [60, 61, 62, 63, 64],
        },
    }
    with pytest.raises(ValueError, match="exact-zero assistance conditioning"):
        select_zero_tail_checkpoint({2_310_144: payload})


def test_pair_comparison_rejects_aggregate_gain_with_compensating_collapse() -> None:
    from tools.run_g1_assistance_observability_pair import compare_pair_selections

    aware = {
        "selected_key": [52, 52.0, 52.0],
        "selected_survival": [52, 52, 52, 52, 52],
    }
    blind = {
        "selected_key": [51, 100.0, 90.2],
        "selected_survival": [51, 100, 100, 100, 100],
    }

    result = compare_pair_selections(aware, blind)

    assert result["outcome"] == "scalar-observability-mixed-tradeoff"
    assert result["aware_no_compensating_collapse"] is False
    assert result["aware_phase_floors"] == [51, 61, 89, 51, 76]


def test_pair_comparison_accepts_scalar_edge_without_phase_collapse() -> None:
    from tools.run_g1_assistance_observability_pair import compare_pair_selections

    aware = {
        "selected_key": [70, 80.0, 90.0],
        "selected_survival": [70, 80, 100, 90, 110],
    }
    blind = {
        "selected_key": [60, 70.0, 80.0],
        "selected_survival": [60, 70, 90, 80, 100],
    }

    result = compare_pair_selections(aware, blind)

    assert result["outcome"] == "scalar-observability-advances"
    assert result["aware_no_compensating_collapse"] is True
