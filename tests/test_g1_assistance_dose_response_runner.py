from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.g1_assistance_dose_response import (
    ASSISTANCE_SCALES,
    CHECKPOINT_LABELS,
    PHASES,
)
from tools.run_g1_assistance_dose_response import (
    build_aggregate,
    monitor_workers,
    parse_checkpoint_specs,
    stop_live_workers,
    validate_devices,
)


def test_parse_checkpoint_specs_requires_registered_order() -> None:
    specs = parse_checkpoint_specs(
        [f"{label}=/tmp/{label}.pkl" for label in CHECKPOINT_LABELS]
    )
    assert tuple(specs) == CHECKPOINT_LABELS
    assert specs["final"] == Path("/tmp/final.pkl")

    with pytest.raises(ValueError, match="registered checkpoint order"):
        parse_checkpoint_specs(
            [
                "parent=/tmp/parent.pkl",
                "assistance_end=/tmp/end.pkl",
                "midpoint=/tmp/mid.pkl",
                "final=/tmp/final.pkl",
            ]
        )


def test_validate_devices_requires_one_distinct_physical_gpu_per_checkpoint() -> None:
    inventory = {"1": "GPU-a", "3": "GPU-b", "5": "GPU-c", "6": "GPU-d"}
    assert validate_devices(("1", "3", "5", "6"), inventory=inventory) == {
        "1": "GPU-a",
        "3": "GPU-b",
        "5": "GPU-c",
        "6": "GPU-d",
    }
    with pytest.raises(ValueError, match="distinct physical GPU"):
        validate_devices(("1", "1", "5", "6"), inventory=inventory)
    with pytest.raises(ValueError, match="not present"):
        validate_devices(("1", "3", "5", "7"), inventory=inventory)


class _FakeProcess:
    def __init__(self, return_code: int | None) -> None:
        self.return_code = return_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = -15

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.return_code is None else self.return_code

    def kill(self) -> None:
        self.return_code = -9


def test_monitor_workers_terminates_peer_after_hard_crash() -> None:
    crashed = _FakeProcess(-9)
    peer = _FakeProcess(None)
    with pytest.raises(RuntimeError, match="parent.*-9"):
        monitor_workers(
            {"parent": crashed, "midpoint": peer}, poll_interval_seconds=0.0
        )
    assert peer.terminated


def test_stop_live_workers_terminates_partially_launched_worker() -> None:
    started = _FakeProcess(None)
    completed = _FakeProcess(0)
    stop_live_workers({"parent": started, "midpoint": completed})
    assert started.terminated
    assert not completed.terminated


def _worker(
    label: str,
    required: tuple[float | None, ...],
    *,
    device_uuid: str | None = None,
) -> dict:
    conditions = []
    for phase in PHASES:
        threshold = required[PHASES.index(phase)]
        for scale in ASSISTANCE_SCALES:
            completed = threshold is not None and scale >= threshold
            remaining = 499 - phase
            steps = remaining if completed else 1
            conditions.append(
                {
                    "phase": phase,
                    "scale": scale,
                    "valid": True,
                    "steps": steps,
                    "remaining_reference_transitions": remaining,
                    "terminal": not completed,
                    "completed_reference_suffix": completed,
                    "wrench": {
                        "steps": steps,
                        "finite": True,
                        "force_cap_compliant": True,
                        "torque_cap_compliant": True,
                        "exact_zero_wrench": scale == 0.0,
                    },
                }
            )
    return {
        "protocol": "g1-assistance-dose-response-worker-v1",
        "checkpoint_label": label,
        "provenance": {
            "checkpoint_sha256": label * 8,
            "reference_sha256": "reference",
            "code_commit": "commit",
            "solver_profile": "g1-4x5",
            "solver_iterations": 4,
            "solver_ls_iterations": 5,
            "runtime_assets": {
                "model_sha256": "model",
                "controller_sha256": "controller",
            },
        },
        "device": {
            "platform": "gpu",
            "device_count": 1,
            "physical_uuid": device_uuid or f"GPU-{label}",
        },
        "phases": list(PHASES),
        "assistance_scales": list(ASSISTANCE_SCALES),
        "conditions": conditions,
        "required_scales": {
            str(phase): value
            for phase, value in zip(PHASES, required, strict=True)
        },
    }


def test_build_aggregate_classifies_monotonic_threshold_reduction() -> None:
    workers = [
        _worker("parent", (1.0, None, 1.0, None, 0.5)),
        _worker("midpoint", (0.5, None, 1.0, 1.0, 0.5)),
        _worker("assistance_end", (0.5, 1.0, 0.5, 1.0, 0.25)),
        _worker("final", (0.25, 1.0, 0.5, 0.5, 0.25)),
    ]
    aggregate = build_aggregate(
        workers,
        expected_checkpoint_sha256={label: label * 8 for label in CHECKPOINT_LABELS},
        expected_device_uuid={label: f"GPU-{label}" for label in CHECKPOINT_LABELS},
        reference_sha256="reference",
        code_commit="commit",
        model_sha256="model",
        controller_sha256="controller",
    )
    assert aggregate["verdict"] == "assistance-requirement-decreases"
    assert [row["label"] for row in aggregate["checkpoints"]] == list(
        CHECKPOINT_LABELS
    )


def test_build_aggregate_rejects_invalid_worker_record() -> None:
    workers = [_worker(label, (1.0,) * 5) for label in CHECKPOINT_LABELS]
    workers[2]["conditions"][0]["valid"] = False
    with pytest.raises(ValueError, match="invalid.*condition"):
        build_aggregate(
            workers,
            expected_checkpoint_sha256={
                label: label * 8 for label in CHECKPOINT_LABELS
            },
            expected_device_uuid={
                label: f"GPU-{label}" for label in CHECKPOINT_LABELS
            },
            reference_sha256="reference",
            code_commit="commit",
            model_sha256="model",
            controller_sha256="controller",
        )


def test_build_aggregate_rejects_worker_threshold_inconsistent_with_conditions() -> None:
    workers = [_worker(label, (1.0,) * 5) for label in CHECKPOINT_LABELS]
    workers[1]["required_scales"]["200"] = 0.25
    with pytest.raises(ValueError, match="derived thresholds"):
        build_aggregate(
            workers,
            expected_checkpoint_sha256={
                label: label * 8 for label in CHECKPOINT_LABELS
            },
            expected_device_uuid={
                label: f"GPU-{label}" for label in CHECKPOINT_LABELS
            },
            reference_sha256="reference",
            code_commit="commit",
            model_sha256="model",
            controller_sha256="controller",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda worker: worker.update(device={"platform": "cpu", "device_count": 99}), "device"),
        (
            lambda worker: worker["provenance"]["runtime_assets"].update(
                model_sha256="bogus"
            ),
            "model",
        ),
        (
            lambda worker: worker["provenance"].update(solver_iterations=3),
            "solver",
        ),
        (
            lambda worker: worker["conditions"][0].pop("wrench"),
            "raw condition",
        ),
    ],
)
def test_build_aggregate_recomputes_provenance_and_rollout_validity(
    mutation, message: str
) -> None:
    workers = [_worker(label, (1.0,) * 5) for label in CHECKPOINT_LABELS]
    mutation(workers[0])
    with pytest.raises(ValueError, match=message):
        build_aggregate(
            workers,
            expected_checkpoint_sha256={
                label: label * 8 for label in CHECKPOINT_LABELS
            },
            expected_device_uuid={
                label: f"GPU-{label}" for label in CHECKPOINT_LABELS
            },
            reference_sha256="reference",
            code_commit="commit",
            model_sha256="model",
            controller_sha256="controller",
        )
