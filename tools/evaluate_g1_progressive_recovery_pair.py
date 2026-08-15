"""Select a progressive recovery expert from paired replay-free phase grids."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


FULL_SUFFIX = (499, 399, 299, 199, 99)


def _validate_repeats(
    values: Sequence[Sequence[int | float]], *, label: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(values) != 2:
        raise ValueError(f"{label} requires exactly two paired repeats")
    normalized = []
    for vector in values:
        if len(vector) != 5 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or int(value) != value
            for value in vector
        ):
            raise ValueError(f"{label} requires five finite survivals")
        normalized.append(tuple(int(value) for value in vector))
    return normalized[0], normalized[1]


def _candidate_record(
    parent: tuple[tuple[int, ...], tuple[int, ...]],
    candidate: tuple[tuple[int, ...], tuple[int, ...]],
) -> dict[str, object]:
    phase_zero_improvements = [
        candidate[repeat][0] - parent[repeat][0] for repeat in range(2)
    ]
    protected_deltas = [
        candidate[repeat][phase] - parent[repeat][phase]
        for repeat in range(2)
        for phase in range(1, 5)
    ]
    phase_zero_pass = min(phase_zero_improvements) >= 3
    protected_pass = min(protected_deltas) >= -2
    eligible = phase_zero_pass and protected_pass
    reason = (
        "eligible"
        if eligible
        else "phase-zero-improvement-insufficient"
        if not phase_zero_pass
        else "protected-phase-regression"
    )
    flattened = [value for vector in candidate for value in vector]
    rank = (
        min(phase_zero_improvements),
        min(flattened),
        median(flattened),
        sum(flattened) / len(flattened),
    )
    return {
        "eligible": eligible,
        "reason": reason,
        "survival": [list(vector) for vector in candidate],
        "phase_zero_improvements": phase_zero_improvements,
        "protected_phase_deltas": protected_deltas,
        "rank": list(rank),
    }


def select_progressive_candidate(
    parent_repeats: Sequence[Sequence[int | float]],
    candidate_repeats: Mapping[int, Sequence[Sequence[int | float]]],
) -> dict[str, object]:
    """Apply the preregistered paired componentwise selection rule."""
    parent = _validate_repeats(parent_repeats, label="parent")
    if not candidate_repeats:
        raise ValueError("at least one candidate is required")
    records: dict[str, dict[str, object]] = {}
    eligible: list[tuple[tuple[float, ...], int]] = []
    for update in sorted(candidate_repeats):
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError("candidate updates must be positive integers")
        candidate = _validate_repeats(
            candidate_repeats[update], label=f"candidate update {update}"
        )
        record = _candidate_record(parent, candidate)
        records[str(update)] = record
        if record["eligible"]:
            rank = tuple(record["rank"])
            eligible.append((rank, update))
    selected_update = (
        max(eligible, key=lambda item: (*item[0], -item[1]))[1]
        if eligible
        else None
    )
    selected_record = (
        records[str(selected_update)] if selected_update is not None else None
    )
    selected_vector = (
        [
            min(selected_record["survival"][0][phase], selected_record["survival"][1][phase])
            for phase in range(5)
        ]
        if selected_record is not None
        else None
    )
    return {
        "valid": True,
        "parent_survival": [list(vector) for vector in parent],
        "candidates": records,
        "selected_update": selected_update,
        "selected_vector": selected_vector,
    }


def classify_outcome(
    selection: Mapping[str, object] | None,
    *,
    support_separable: bool = True,
    execution_valid: bool = True,
) -> str:
    """Map the preregistered evidence state to one terminal outcome."""
    if not execution_valid:
        return "invalid-execution"
    if not support_separable:
        return "support-not-separable"
    if selection is None or selection.get("selected_vector") is None:
        return "gated-recovery-insufficient"
    vector = selection["selected_vector"]
    if list(vector) == list(FULL_SUFFIX):
        return "gated-recovery-solves"
    return "gated-recovery-advances"


def _read_survival(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != (
        "g1-flax-dance-replay-free-five-phase-v1"
    ):
        raise ValueError(f"invalid phase-grid artifact: {path}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"phase-grid summary missing: {path}")
    values = summary.get("survival")
    return _validate_repeats((values, values), label=str(path))[0]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-repeat", type=Path, action="append", required=True)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("UPDATE", "REPEAT0", "REPEAT1"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parent = [_read_survival(path.resolve()) for path in args.parent_repeat]
    candidates = {
        int(update): (
            _read_survival(Path(repeat0).resolve()),
            _read_survival(Path(repeat1).resolve()),
        )
        for update, repeat0, repeat1 in args.candidate
    }
    selection = select_progressive_candidate(parent, candidates)
    selection["outcome"] = classify_outcome(selection)
    _write_json(args.output.resolve(), selection)


if __name__ == "__main__":
    main()
