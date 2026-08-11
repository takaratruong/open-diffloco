"""Join checkpoint-aligned actor losses to replay-free phase survival."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.algorithms.shac.phase_weighting import phase_bin_indices
from tools.evaluate_g1_phase_grid import PHASES


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def _validated_record_map(
    records: list[dict[str, Any]], *, bin_count: int
) -> dict[int, dict[str, Any]]:
    record_map: dict[int, dict[str, Any]] = {}
    for record in records:
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("checkpoint record step must be an integer")
        if step in record_map:
            raise ValueError("checkpoint records contain a duplicate step")
        losses = np.asarray(record.get("actor_cagrad_bin_losses"), dtype=float)
        counts = np.asarray(record.get("actor_cagrad_bin_counts"), dtype=int)
        if losses.shape != (bin_count,) or counts.shape != (bin_count,):
            raise ValueError("checkpoint records require five phase bins")
        if not np.isfinite(losses).all():
            raise ValueError("checkpoint actor losses must be finite")
        if np.any(counts <= 0) or record.get("actor_preview_valid") is not True:
            raise ValueError("checkpoint preview telemetry must be valid")
        record_map[step] = record
    return record_map


def _validate_summary(summary: dict[str, Any], *, step: int) -> None:
    if summary.get("phases") != list(PHASES):
        raise ValueError(f"phase-grid phases must be exactly {PHASES}")
    steps = summary.get("steps")
    if not isinstance(steps, dict) or set(steps) != {
        str(phase) for phase in PHASES
    }:
        raise ValueError("phase-grid survival steps are incomplete")
    survival = np.asarray([steps[str(phase)] for phase in PHASES], dtype=float)
    if not np.isfinite(survival).all() or np.any(survival < 1):
        raise ValueError("phase-grid survival steps must be positive and finite")
    checkpoint_step = summary.get("checkpoint_step")
    if checkpoint_step is not None and checkpoint_step != step:
        raise ValueError("phase-grid checkpoint step does not match its key")
    if not isinstance(summary.get("checkpoint_sha256"), str):
        raise ValueError("phase-grid checkpoint SHA-256 is required")


def build_objective_survival_audit(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
    *,
    phase_count: int,
    bin_count: int = 5,
) -> dict[str, Any]:
    """Build ten immutable checkpoint-phase objective/survival cases."""
    if phase_count < 1 or bin_count != 5:
        raise ValueError("audit requires a positive phase count and five bins")
    record_map = _validated_record_map(records, bin_count=bin_count)
    if set(record_map) != set(summaries):
        raise ValueError("checkpoint record and phase-grid steps must match")
    if len(record_map) != 2:
        raise ValueError("audit requires exactly midpoint and final steps")

    phase_bins = np.asarray(
        phase_bin_indices(
            np.asarray(PHASES, dtype=np.int32),
            phase_count=phase_count,
            bin_count=bin_count,
        )
    )
    cases = []
    for step in sorted(record_map):
        summary = summaries[step]
        _validate_summary(summary, step=step)
        losses = record_map[step]["actor_cagrad_bin_losses"]
        for phase, bin_index in zip(PHASES, phase_bins, strict=True):
            cases.append(
                {
                    "checkpoint_step": step,
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "phase": phase,
                    "bin_index": int(bin_index),
                    "actor_loss": float(losses[int(bin_index)]),
                    "survival_frames": int(summary["steps"][str(phase)]),
                }
            )

    loss_ranks = _average_ranks(
        np.asarray([case["actor_loss"] for case in cases])
    )
    survival_ranks = _average_ranks(
        np.asarray([case["survival_frames"] for case in cases])
    )
    if np.std(loss_ranks) == 0.0 or np.std(survival_ranks) == 0.0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(loss_ranks, survival_ranks)[0, 1])
    return {
        "checkpoint_steps": sorted(record_map),
        "phases": list(PHASES),
        "phase_count": phase_count,
        "bin_count": bin_count,
        "cases": cases,
        "loss_survival_spearman": (
            correlation if math.isfinite(correlation) else None
        ),
        "interpretation": "lower-loss-higher-survival-is-negative",
        "valid": math.isfinite(correlation),
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen-preview actor loss against phase survival."
    )
    parser.add_argument(
        "--checkpoint-phase-metrics", type=Path, required=True
    )
    parser.add_argument(
        "--phase-grid",
        action="append",
        nargs=2,
        metavar=("STEP", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    phase_grid_inputs: dict[int, Path] = {}
    for raw_step, raw_path in args.phase_grid:
        step = int(raw_step)
        if step in phase_grid_inputs:
            raise ValueError("phase-grid inputs contain a duplicate step")
        phase_grid_inputs[step] = Path(raw_path)
    paths = [args.checkpoint_phase_metrics, *phase_grid_inputs.values()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = json.loads(args.checkpoint_phase_metrics.read_text())
    summaries = {
        step: json.loads(path.read_text())
        for step, path in phase_grid_inputs.items()
    }
    payload = build_objective_survival_audit(
        records, summaries, phase_count=499
    )
    payload["source_sha256"] = {
        str(path.resolve()): _sha256(path) for path in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
