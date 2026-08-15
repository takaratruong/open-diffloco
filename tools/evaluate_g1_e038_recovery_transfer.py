"""Pure contracts for the preregistered E038 recovery transfer evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import numpy as np


SOURCE_PHASES = (0, 100, 200, 300, 400)
ROWS_PER_SOURCE = 24
BANK_ROWS = len(SOURCE_PHASES) * ROWS_PER_SOURCE
HORIZON = 32


def _zero_seed(value: str) -> int:
    """Parse the only seed registered for this deterministic evaluation."""
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("seed must be exactly zero")
    return seed


def validate_bank_layout(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Require the immutable E027 bank's five ordered 24-row source bands."""
    if "source_start_phase" not in arrays:
        raise ValueError("bank is missing source_start_phase")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in normalized.items():
        if name not in {"source_start_phase", "termination_thresholds"} and (
            value.ndim == 0 or value.shape[0] != BANK_ROWS
        ):
            raise ValueError(f"{name} must contain exactly 120 rows")
    source_start_phase = normalized["source_start_phase"]
    expected = np.repeat(
        np.asarray(SOURCE_PHASES, dtype=np.int32), ROWS_PER_SOURCE
    )
    try:
        is_exact = (
            source_start_phase.shape == (BANK_ROWS,)
            and np.isfinite(source_start_phase).all()
            and np.array_equal(source_start_phase, expected)
        )
    except TypeError:
        is_exact = False
    if not is_exact:
        raise ValueError("bank source groups must be five ordered 24-row bands")
    return normalized


def survival_from_terminals(terminals: np.ndarray) -> list[int]:
    """Report each H32 rollout's first terminal step, or H32 if it survives."""
    values = np.asarray(terminals)
    if values.ndim != 2 or values.shape[1] != HORIZON:
        raise ValueError("terminal evidence must have shape (rows, 32)")
    if not np.isfinite(values).all():
        raise ValueError("terminal evidence must be finite")
    return [
        int(indices[0]) if indices.size else HORIZON
        for indices in (np.flatnonzero(row) for row in values)
    ]


def _survival_vector(values: Sequence[int | float], *, label: str) -> np.ndarray:
    result = np.asarray(values)
    if (
        result.shape != (BANK_ROWS,)
        or not np.issubdtype(result.dtype, np.number)
        or not np.isfinite(result).all()
        or not np.equal(result, np.floor(result)).all()
        or np.any(result < 0)
        or np.any(result > HORIZON)
    ):
        raise ValueError(f"{label} survival evidence is malformed")
    return result.astype(np.int32)


def classify_transfer(
    parent_survival: Sequence[int | float],
    expert_survival: Sequence[int | float],
    source_start_phase: Sequence[int | float],
    execution_valid: bool,
) -> str:
    """Apply E038's preregistered transfer outcome map."""
    if not execution_valid:
        return "invalid-execution"
    parent = _survival_vector(parent_survival, label="parent")
    expert = _survival_vector(expert_survival, label="expert")
    phases = validate_bank_layout(
        {"source_start_phase": np.asarray(source_start_phase)}
    )["source_start_phase"]
    phase_zero = phases == 0
    untouched = ~phase_zero
    phase_zero_successes = int(np.sum(expert[phase_zero] >= HORIZON))
    no_regressions = bool(np.all(expert >= parent))
    newly_recovered = int(
        np.sum((expert[untouched] >= HORIZON) & (parent[untouched] < HORIZON))
    )
    median_regressed = bool(
        np.median(expert[untouched]) < np.median(parent[untouched])
    )
    has_improvements = bool(np.any(expert > parent))
    has_regressions = bool(np.any(expert < parent))

    if (
        phase_zero_successes < 10
        or median_regressed
        or (has_regressions and not has_improvements)
    ):
        return "recovery-expert-destructive"
    if phase_zero_successes in (10, 11) or (
        has_improvements and has_regressions
    ):
        return "recovery-expert-mixed-transfer"
    if phase_zero_successes >= 12 and no_regressions:
        if newly_recovered >= 10:
            return "recovery-expert-generalizes"
        return "recovery-expert-local-only"
    return "recovery-expert-mixed-transfer"


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded Task 1 parser, pinning the registered seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=_zero_seed, default=0)
    return parser


def main() -> None:
    build_parser().parse_args()


if __name__ == "__main__":
    main()
