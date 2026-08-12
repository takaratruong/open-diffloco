"""Select a checkpoint using immutable replay-free phase-grid summaries."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from tools.evaluate_g1_phase_grid import PHASES


def select_checkpoint(
    summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Rank checkpoints by minimum, median, mean, then earliest step."""
    if not summaries:
        raise ValueError("at least one phase-grid summary is required")
    records = []
    expected_step_keys = {str(phase) for phase in PHASES}
    for checkpoint_step, summary in sorted(summaries.items()):
        if isinstance(checkpoint_step, bool) or checkpoint_step < 0:
            raise ValueError("checkpoint steps must be non-negative integers")
        rollout_summary = summary.get("summary")
        if isinstance(rollout_summary, dict):
            phases = rollout_summary.get("phases")
            survival_values = rollout_summary.get("survival")
            raw_steps = (
                {
                    str(phase): value
                    for phase, value in zip(
                        phases or (), survival_values or (), strict=True
                    )
                }
                if isinstance(phases, list) and isinstance(survival_values, list)
                else None
            )
        else:
            phases = summary.get("phases")
            raw_steps = summary.get("steps")
        if phases != list(PHASES):
            raise ValueError("phase-grid phases do not match the protocol")
        if not isinstance(raw_steps, dict) or set(raw_steps) != expected_step_keys:
            raise ValueError("phase-grid survival steps are incomplete")
        survival = [float(raw_steps[str(phase)]) for phase in PHASES]
        if not all(math.isfinite(value) and value > 0 for value in survival):
            raise ValueError("survival steps must be positive and finite")
        embedded_step = summary.get("checkpoint_step")
        if embedded_step is not None and embedded_step != checkpoint_step:
            raise ValueError("embedded checkpoint step does not match its key")
        checkpoint_sha = summary.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            raise ValueError("checkpoint SHA-256 is required")
        integer_survival = [int(value) for value in survival]
        if any(value != converted for value, converted in zip(
            survival, integer_survival, strict=True
        )):
            raise ValueError("survival steps must be integers")
        records.append(
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint_sha256": checkpoint_sha,
                "survival": integer_survival,
                "minimum": min(integer_survival),
                "median": float(statistics.median(integer_survival)),
                "mean": float(statistics.fmean(integer_survival)),
            }
        )

    selected = max(
        records,
        key=lambda record: (
            record["minimum"],
            record["median"],
            record["mean"],
            -record["checkpoint_step"],
        ),
    )
    return {
        "protocol": "minimum-median-mean-earliest-v1",
        "phases": list(PHASES),
        "checkpoints": records,
        "selected_step": selected["checkpoint_step"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_survival": selected["survival"],
        "selected_minimum": selected["minimum"],
        "selected_median": selected["median"],
        "selected_mean": selected["mean"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one replay-free phase-grid checkpoint."
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
    summaries = {}
    for raw_step, raw_path in args.phase_grid:
        step = int(raw_step)
        if step in summaries:
            raise ValueError("phase-grid inputs contain a duplicate step")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries[step] = json.loads(path.read_text(encoding="utf-8"))
    payload = select_checkpoint(summaries)
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
