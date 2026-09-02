from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _provenance() -> dict[str, str]:
    return {
        "checkpoint_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "controller_sha256": "d" * 64,
        "code_commit": "e" * 40,
    }


def _signals() -> dict[int, list[float]]:
    return {
        0: [1.0, 2.0],
        25: [3.0],
        50: [4.0, 5.0],
        75: [6.0],
        100: [7.0, 8.0],
    }


def test_contact_calibration_uses_exact_grid_and_linear_p90() -> None:
    from tools.calibrate_g1_ahac_contact_threshold import (
        CALIBRATION_PHASES,
        build_calibration_payload,
        validate_calibration_payload,
    )

    payload = build_calibration_payload(_signals(), provenance=_provenance())

    assert CALIBRATION_PHASES == (0, 25, 50, 75, 100)
    assert payload["phases"] == list(CALIBRATION_PHASES)
    assert payload["quantile"] == 0.9
    assert payload["quantile_method"] == "linear"
    assert payload["threshold"] == pytest.approx(7.3)
    assert payload["sample_count"] == 8
    assert validate_calibration_payload(payload) == pytest.approx(7.3)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda payload: payload["signals_by_phase"].pop("50"), "phase"),
        (
            lambda payload: payload["signals_by_phase"]["25"].append(float("nan")),
            "finite",
        ),
        (lambda payload: payload.__setitem__("threshold", 99.0), "threshold"),
        (
            lambda payload: payload["provenance"].pop("model_sha256"),
            "provenance",
        ),
    ],
)
def test_contact_calibration_fails_closed(mutator, match: str) -> None:
    from tools.calibrate_g1_ahac_contact_threshold import (
        build_calibration_payload,
        validate_calibration_payload,
    )

    payload = build_calibration_payload(_signals(), provenance=_provenance())
    mutator(payload)
    with pytest.raises(ValueError, match=match):
        validate_calibration_payload(payload)


def test_ahac_builder_changes_only_registered_optimizer_family_fields() -> None:
    from tools.run_g1_ahac_walk import (
        AHAC_CHANGED_FIELDS,
        CHECKPOINT_INTERVAL,
        TOTAL_STEPS,
        build_ahac_walk_kwargs,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = Path("/tmp/walk.npz")
    baseline = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_ahac_walk_kwargs(
        "g1-4x5", reference, 0, contact_threshold=123.0
    )
    changed = {
        key
        for key in set(baseline) | set(treatment)
        if not np.array_equal(baseline.get(key), treatment.get(key))
    }

    assert changed == set(AHAC_CHANGED_FIELDS)
    assert treatment["ahac"] is True
    assert treatment["ahac_horizon_min"] == 8
    assert treatment["ahac_horizon_max"] == treatment["unroll_length"] == 24
    assert treatment["ahac_contact_threshold"] == 123.0
    assert treatment["ahac_dual_lr"] == 5e-4
    assert treatment["ahac_critic_max_iterations"] == 64
    assert treatment["ahac_critic_tolerance"] == 0.2
    assert treatment["ahac_contact_metric"] == "root_generalized"
    assert treatment["ahac_semantics"] == "legacy_horizon_only_target"
    assert treatment["actor_bootstrap_scale"] == 1.0
    assert treatment["actor_bootstrap_delay_steps"] == 0
    assert treatment["num_envs"] * treatment["gradient_accumulation_steps"] == 512
    assert TOTAL_STEPS == 1_572_864
    assert CHECKPOINT_INTERVAL == 196_608
    assert TOTAL_STEPS // (512 * 24) == 128


def _valid_ahac_row(step: int) -> dict[str, object]:
    from tests.test_g1_rmr_noise_h24_walk import _valid_cagrad_row

    return {
        "step": step,
        **_valid_cagrad_row(action_noise_current=[0.5] * 29),
        "actor_bootstrap_scale_current": 1.0,
        "ahac_valid": True,
        "ahac_horizon": 10.0,
        "ahac_horizon_before_update": 9.5,
        "ahac_active_transitions": 10,
        "ahac_dual_mean": 0.1,
        "ahac_dual_max": 0.2,
        "ahac_contact_stiffness_mean": 80.0,
        "ahac_contact_stiffness_max": 120.0,
        "ahac_contact_threshold": 100.0,
        "ahac_critic_iterations": 12,
        "ahac_critic_converged": True,
        "ahac_critic_loss_history": [1.0, 0.9, 0.8, 0.75, 0.7],
        "ahac_critic_head_losses": [0.7, 0.8],
        "ahac_critic_head_disagreement": 0.2,
    }


def test_ahac_telemetry_validator_rejects_bounds_and_nonfinite_values() -> None:
    from tools.run_g1_ahac_walk import validate_ahac_telemetry_row

    row = _valid_ahac_row(196_608)
    validate_ahac_telemetry_row(row, threshold=100.0)

    with pytest.raises(ValueError, match="AHAC telemetry"):
        validate_ahac_telemetry_row({**row, "ahac_horizon": 25.0}, threshold=100.0)
    with pytest.raises(ValueError, match="AHAC telemetry"):
        validate_ahac_telemetry_row(
            {**row, "ahac_critic_head_losses": [0.7, np.inf]},
            threshold=100.0,
        )


def test_runner_parser_requires_calibration_and_code_commit(tmp_path: Path) -> None:
    from tools.run_g1_ahac_walk import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
    args = parser.parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--reference-path",
            "/tmp/walk.npz",
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--code-commit",
            "f" * 40,
        ]
    )
    assert args.seed == 0


def test_atomic_calibration_round_trip(tmp_path: Path) -> None:
    from tools.calibrate_g1_ahac_contact_threshold import (
        build_calibration_payload,
        publish_calibration,
    )

    output = tmp_path / "nested" / "calibration.json"
    payload = build_calibration_payload(_signals(), provenance=_provenance())
    publish_calibration(output, payload)
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True
    assert not list(output.parent.glob("*.tmp"))
