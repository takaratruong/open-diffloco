"""Select fresh walking SHAC checkpoints against an exact zero residual."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path


METRICS = (
    "mean_reward",
    "mean_body_position_error",
    "mean_body_orientation_error",
    "mean_body_linear_velocity_error",
    "mean_body_angular_velocity_error",
    "mean_anchor_position_error",
    "mean_anchor_orientation_error",
)
SECONDARY_ERROR_METRICS = METRICS[2:]
CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pkl$")


def _finite_mean(results: list[dict], metric: str) -> float:
    if len(results) != 5:
        raise ValueError("paired evaluation requires five phase results")
    values = [float(row[metric]) for row in results]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired evaluation metrics must be finite")
    return sum(values) / len(values)


def _relative_change(candidate: float, source: float) -> float:
    if source <= 0.0:
        raise ValueError("paired source error must be positive")
    return candidate / source - 1.0


def _checkpoint_step(payload: dict) -> int:
    match = CHECKPOINT_PATTERN.search(str(payload.get("checkpoint_path", "")))
    if match is None:
        raise ValueError("checkpoint path does not encode its step")
    return int(match.group(1))


def _validate_training_row(row: dict) -> None:
    counts = row.get("actor_cagrad_bin_counts")
    norms = row.get("actor_cagrad_bin_gradient_norms")
    if (
        row.get("actor_cagrad_valid") is not True
        or not isinstance(counts, list)
        or len(counts) != 5
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in counts
        )
        or not isinstance(norms, list)
        or len(norms) != 5
        or any(
            not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0 + 1e-6
            for value in norms
        )
    ):
        raise ValueError("checkpoint CAGrad telemetry is invalid")
    for key in ("actor_grad", "actor_update_norm"):
        value = float(row.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("checkpoint actor update telemetry is invalid")


def select_fresh_walking_checkpoint(
    evaluations: list[dict], training_rows: list[dict]
) -> dict:
    """Apply the preregistered zero-residual improvement gate."""
    rows_by_step = {int(row["step"]): row for row in training_rows}
    if len(rows_by_step) != len(training_rows):
        raise ValueError("checkpoint telemetry contains duplicate steps")
    records = []
    for payload in evaluations:
        if payload.get("source_controller") != "zero_residual":
            raise ValueError("fresh SHAC selection requires zero residual source")
        step = _checkpoint_step(payload)
        if step not in rows_by_step:
            raise ValueError("checkpoint telemetry is missing")
        training_row = rows_by_step[step]
        _validate_training_row(training_row)
        source_results = payload["source"]["results"]
        candidate_results = payload["candidate"]["results"]
        source = {
            metric: _finite_mean(source_results, metric) for metric in METRICS
        }
        candidate = {
            metric: _finite_mean(candidate_results, metric)
            for metric in METRICS
        }
        complete = payload["candidate"]["summary"].get(
            "completed_suffix"
        ) == [True] * 5
        action_rms = float(
            payload["candidate"].get("action_root_mean_square", float("nan"))
        )
        if not math.isfinite(action_rms) or action_rms <= 0.0:
            raise ValueError("candidate learned residual must be finite and nonzero")
        reward_delta = candidate["mean_reward"] - source["mean_reward"]
        error_changes = {
            metric: _relative_change(candidate[metric], source[metric])
            for metric in METRICS[1:]
        }
        eligible = (
            complete
            and reward_delta > 0.0
            and error_changes["mean_body_position_error"] < 0.0
            and all(
                error_changes[metric] <= 0.05
                for metric in SECONDARY_ERROR_METRICS
            )
        )
        records.append(
            {
                "step": step,
                "checkpoint_path": payload["checkpoint_path"],
                "eligible": eligible,
                "completed_suffix": complete,
                "source": source,
                "candidate": candidate,
                "reward_delta": reward_delta,
                "error_relative_changes": error_changes,
                "action_root_mean_square": action_rms,
                "actor_grad": float(training_row["actor_grad"]),
                "actor_update_norm": float(training_row["actor_update_norm"]),
            }
        )
    eligible_records = [record for record in records if record["eligible"]]
    eligible_records.sort(
        key=lambda record: (
            -record["candidate"]["mean_reward"],
            record["candidate"]["mean_body_position_error"],
            record["step"],
        )
    )
    selected = eligible_records[0] if eligible_records else None
    return {
        "protocol": "g1-fresh-walking-shac-selection-v1",
        "outcome": (
            "fresh-shac-learns-walk"
            if selected is not None
            else "stable-partial-learning"
        ),
        "selected_checkpoint": selected,
        "records": sorted(records, key=lambda record: record["step"]),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.evaluation
    ]
    training_rows = json.loads(
        args.checkpoint_metrics.read_text(encoding="utf-8")
    )
    report = select_fresh_walking_checkpoint(evaluations, training_rows)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
