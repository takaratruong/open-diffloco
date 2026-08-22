"""Run bounded DiffMimic-style demonstration replay on E023 walking SHAC."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


TOTAL_UPDATES = 32
TRANSITIONS_PER_UPDATE = 512 * 24
TOTAL_STEPS = TOTAL_UPDATES * TRANSITIONS_PER_UPDATE
CHECKPOINT_INTERVAL = 16 * TRANSITIONS_PER_UPDATE
E023_TOTAL_STEPS = 1_572_864
DEMONSTRATION_REPLAY_THRESHOLD = 0.20
PHASE_CAPS = (124, 99, 74, 49, 24)
CONTROL_SURVIVAL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}


def expected_checkpoint_steps() -> tuple[int, ...]:
    return (CHECKPOINT_INTERVAL, TOTAL_STEPS)


def build_demonstration_replay_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Add only intra-rollout replay and the bounded execution budget to E023."""
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        demonstration_replay_threshold=DEMONSTRATION_REPLAY_THRESHOLD,
        total_steps=TOTAL_STEPS,
    )
    return kwargs


def _validated_survival(
    treatment: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    if set(treatment) != set(CONTROL_SURVIVAL):
        raise ValueError("survival evidence must contain updates 16 and 32")
    rows: dict[int, tuple[int, ...]] = {}
    for update in sorted(CONTROL_SURVIVAL):
        values = treatment[update]
        if len(values) != len(PHASE_CAPS):
            raise ValueError("each survival vector must contain five phases")
        row = []
        for value, cap in zip(values, PHASE_CAPS, strict=True):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError("survival values must be integers")
            value = int(value)
            if value < 0 or value > cap:
                raise ValueError("survival value lies outside its phase suffix")
            row.append(value)
        rows[update] = tuple(row)
    return rows


def _validated_fractions(
    fractions: Mapping[int, float],
) -> dict[int, float]:
    if set(fractions) != set(CONTROL_SURVIVAL):
        raise ValueError("replay fractions must contain updates 16 and 32")
    result = {}
    for update, value in fractions.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError("replay fractions must be finite and positive")
        result[int(update)] = float(value)
    return result


def classify_demonstration_replay(
    treatment: Mapping[int, Sequence[int]],
    replay_fractions: Mapping[int, float],
) -> str:
    """Classify replay-free competence and the assistance dose."""
    rows = _validated_survival(treatment)
    fractions = _validated_fractions(replay_fractions)
    if any(value >= 0.95 for value in fractions.values()):
        return "demo-replay-overassisted"
    deltas = {
        update: tuple(
            candidate - control
            for candidate, control in zip(
                rows[update], CONTROL_SURVIVAL[update], strict=True
            )
        )
        for update in rows
    }
    if any(
        all(delta >= 0 for delta in row)
        and any(delta > 0 for delta in row[:4])
        for row in deltas.values()
    ):
        return "demo-replay-early-advances"
    if any(all(abs(delta) <= 2 for delta in row) for row in deltas.values()):
        return "demo-replay-early-parity"
    if any(
        any(delta > 2 for delta in row) and any(delta < -2 for delta in row)
        for row in deltas.values()
    ):
        return "demo-replay-early-mixed"
    return "demo-replay-early-underperforms"


def select_checkpoint(treatment: Mapping[int, Sequence[int]]) -> int:
    rows = _validated_survival(treatment)

    def key(update: int) -> tuple[float, float, float, int]:
        values = rows[update][:4]
        return min(values), median(values), sum(values) / len(values), -update

    return max(rows, key=key)


def validate_replay_telemetry(path: Path) -> dict[int, float]:
    """Validate exact finite checkpoint replay evidence for classification."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(zip(expected_checkpoint_steps(), (16, 32), strict=True))
    if not isinstance(rows, list) or {row.get("step") for row in rows} != set(
        expected
    ):
        raise ValueError("replay telemetry has the wrong checkpoint rows")
    fractions = {}
    for row in rows:
        if (
            row.get("demonstration_replay_valid") is not True
            or row.get("demonstration_replay_threshold")
            != DEMONSTRATION_REPLAY_THRESHOLD
        ):
            raise ValueError("replay telemetry contract is invalid")
        count = row.get("demonstration_replay_count")
        fraction = row.get("demonstration_replay_fraction")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError(
                "replay telemetry must be finite, nontrivial, and bounded"
            )
        fractions[expected[row["step"]]] = float(fraction)
    return fractions


def validate_preflight(
    *, repository: Path, reference_path: Path, code_commit: str
) -> dict[str, Any]:
    from tools.run_g1_rmr_noise_h24_walk import (
        validate_preflight as validate_e023_preflight,
    )

    parent = validate_e023_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    return {
        **parent,
        "protocol": "g1-demonstration-replay-h24-walk-preflight-v1",
        "scientific_delta": ["demonstration_replay_threshold"],
        "demonstration_replay_threshold": DEMONSTRATION_REPLAY_THRESHOLD,
        "total_updates": TOTAL_UPDATES,
        "total_steps": TOTAL_STEPS,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "action_noise_schedule_steps": E023_TOTAL_STEPS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_demonstration_replay_h24_walk_runs"),
    )
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    from src.algorithms.shac.algorithm import train
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )
    from tools.run_g1_tracking_shac import configure_jax
    from tools.run_g1_zero_assistance_consolidation import (
        _write_json_atomically,
    )

    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path,
        code_commit=args.code_commit,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_demonstration_replay_kwargs(
        args.solver_profile, args.reference_path.resolve(), args.seed
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory,
        expected_kwargs=kwargs,
        expected_steps=expected_checkpoint_steps(),
        total_steps=TOTAL_STEPS,
        protocol="g1-demonstration-replay-h24-walk-training-v1",
    )
    fractions = validate_replay_telemetry(
        run_directory / "checkpoint_phase_metrics.json"
    )
    _write_json_atomically(
        output_root / "training_validation.json",
        {**validation, "demonstration_replay_fractions": fractions},
    )
    print(run_directory)


if __name__ == "__main__":
    main()
