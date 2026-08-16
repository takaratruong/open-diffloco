"""Run the bounded root-position-observation ablation on E023 walking SHAC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


TRANSITIONS_PER_UPDATE = 512 * 24
CHECKPOINT_INTERVAL = 16 * TRANSITIONS_PER_UPDATE
ACTOR_HISTORY_LEN = 10
ACTOR_FRAME_OBS_DIM = 331
EXPECTED_ACTOR_OBS_DIM = 3_310
E023_TOTAL_STEPS = 1_572_864
PHASE_CAPS = (124, 99, 74, 49, 24)
CONTROL_SURVIVAL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}
E004_EARLY_SURVIVAL = {
    16: (43, 38, 48, 49, 24),
    32: (51, 66, 55, 49, 24),
}
E023_FULL_SURVIVAL = {
    64: (71, 60, 51, 49, 24),
    128: (116, 99, 67, 49, 24),
}
FULL_EVIDENCE_UPDATES = (16, 32, 64, 128)


@dataclass(frozen=True)
class BudgetContract:
    """Immutable execution-only settings for the E004/E005 runner."""

    name: str
    total_updates: int
    checkpoint_updates: tuple[int, ...]

    @property
    def total_steps(self) -> int:
        return self.total_updates * TRANSITIONS_PER_UPDATE

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        return tuple(
            update * TRANSITIONS_PER_UPDATE for update in self.checkpoint_updates
        )


EARLY_BUDGET = BudgetContract(
    name="early",
    total_updates=32,
    checkpoint_updates=(16, 32),
)
FULL_BUDGET = BudgetContract(
    name="full",
    total_updates=128,
    checkpoint_updates=(16, 32, 48, 64, 80, 96, 112, 128),
)
_BUDGETS = {budget.name: budget for budget in (EARLY_BUDGET, FULL_BUDGET)}

# Preserve the public E004 constants for existing callers and evidence tooling.
TOTAL_UPDATES = EARLY_BUDGET.total_updates
TOTAL_STEPS = EARLY_BUDGET.total_steps


def resolve_budget(name: str) -> BudgetContract:
    """Resolve a registered execution budget and reject unknown aliases."""
    try:
        return _BUDGETS[name]
    except KeyError as error:
        raise ValueError(f"unknown execution budget: {name}") from error


def expected_checkpoint_steps(
    budget: BudgetContract = EARLY_BUDGET,
) -> tuple[int, ...]:
    """Return the exact archive steps for one immutable budget."""
    return budget.checkpoint_steps


def build_motion_anchor_position_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    *,
    budget: BudgetContract = EARLY_BUDGET,
) -> dict[str, Any]:
    """Apply the sole root-position treatment plus bounded-run metadata."""
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    kwargs = build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        actor_observe_motion_anchor_position=True,
        expected_actor_obs_dim=EXPECTED_ACTOR_OBS_DIM,
        total_steps=budget.total_steps,
    )
    return kwargs


def validate_e023_preflight(**kwargs: Any) -> dict[str, Any]:
    """Reuse E023's pure provenance and asset gate before trainer imports."""
    from tools.run_g1_one_frame_rmr_noise_h24_walk import validate_preflight

    return validate_preflight(**kwargs)


def validate_preflight(
    *,
    repository: Path,
    reference_path: Path,
    code_commit: str,
    budget: BudgetContract = EARLY_BUDGET,
) -> dict[str, Any]:
    """Bind the E023 runtime and the sole semantic treatment delta."""
    base = validate_e023_preflight(
        repository=repository,
        reference_path=reference_path,
        code_commit=code_commit,
    )
    treatment = {
        **base,
        "protocol": (
            "g1-motion-anchor-position-h24-walk-preflight-v1"
            if budget is EARLY_BUDGET
            else "g1-motion-anchor-position-h24-walk-preflight-full-v1"
        ),
        "scientific_delta": ["actor_observe_motion_anchor_position"],
        "actor_observe_motion_anchor_position": True,
        "actor_history_len": ACTOR_HISTORY_LEN,
        "actor_frame_obs_dim": ACTOR_FRAME_OBS_DIM,
        "expected_actor_obs_dim": EXPECTED_ACTOR_OBS_DIM,
        "actor_input_dim": EXPECTED_ACTOR_OBS_DIM,
        "total_updates": budget.total_updates,
        "total_steps": budget.total_steps,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_updates": list(budget.checkpoint_updates),
        "checkpoint_steps": list(expected_checkpoint_steps(budget)),
        "action_noise_schedule_steps": E023_TOTAL_STEPS,
    }
    if budget is FULL_BUDGET:
        treatment["budget"] = budget.name
    return treatment


def validate_e023_training_artifacts(
    run_directory: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load the reviewed E023 artifact validator without importing JAX early."""
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )

    return validate_training_artifacts(run_directory, **kwargs)


def validate_budget_training_artifacts(
    run_directory: Path,
    *,
    expected_kwargs: Mapping[str, Any],
    budget: BudgetContract = EARLY_BUDGET,
) -> dict[str, Any]:
    """Require the exact archive set and total for the selected budget."""
    protocol = "g1-motion-anchor-position-h24-walk-training-v1"
    if budget is FULL_BUDGET:
        protocol = "g1-motion-anchor-position-h24-walk-training-full-v1"
    return validate_e023_training_artifacts(
        run_directory,
        expected_kwargs=expected_kwargs,
        expected_steps=expected_checkpoint_steps(budget),
        total_steps=budget.total_steps,
        protocol=protocol,
    )


def _validated_survival_for_updates(
    treatment: Mapping[int, Sequence[int]],
    *,
    expected_updates: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    if set(treatment) != set(expected_updates):
        expected = ", ".join(str(update) for update in expected_updates)
        raise ValueError(f"survival evidence must contain exactly updates {expected}")
    validated: dict[int, tuple[int, ...]] = {}
    for update in expected_updates:
        values = treatment[update]
        if len(values) != len(PHASE_CAPS):
            raise ValueError("each survival vector must contain five phases")
        row: list[int] = []
        for value, cap in zip(values, PHASE_CAPS, strict=True):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("survival values must be integers")
            if value < 0 or value > cap:
                raise ValueError("survival value is outside its phase suffix")
            row.append(value)
        validated[update] = tuple(row)
    return validated


def _validated_survival(
    treatment: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    return _validated_survival_for_updates(
        treatment,
        expected_updates=tuple(CONTROL_SURVIVAL),
    )


def classify_root_position_ablation(
    treatment: Mapping[int, Sequence[int]],
) -> str:
    """Classify bounded treatment evidence against matched E023 checkpoints."""
    rows = _validated_survival(treatment)
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
        all(delta >= 0 for delta in row) and any(delta > 0 for delta in row[:4])
        for row in deltas.values()
    ):
        return "root-position-early-advances"
    if any(all(abs(delta) <= 2 for delta in row) for row in deltas.values()):
        return "root-position-early-parity"
    if any(
        any(delta > 2 for delta in row) and any(delta < -2 for delta in row)
        for row in deltas.values()
    ):
        return "root-position-early-mixed"
    return "root-position-early-underperforms"


def select_root_position_checkpoint(
    treatment: Mapping[int, Sequence[int]],
) -> int:
    """Select by first-four-phase min/median/mean, with earliest ties."""
    rows = _validated_survival(treatment)

    def key(update: int) -> tuple[float, float, float, int]:
        values = rows[update][:4]
        return min(values), median(values), sum(values) / len(values), -update

    return max(rows, key=key)


def classify_full_budget_root_position(
    treatment: Mapping[int, Sequence[int]],
) -> str:
    """Classify matched full-budget evidence after exact E004 corroboration."""
    rows = _validated_survival_for_updates(
        treatment,
        expected_updates=FULL_EVIDENCE_UPDATES,
    )
    for update, expected in E004_EARLY_SURVIVAL.items():
        if rows[update] != expected:
            raise ValueError(
                f"update {update} does not exactly corroborate E004 early evidence"
            )
    control = E023_FULL_SURVIVAL[128]
    deltas = tuple(
        candidate - baseline
        for candidate, baseline in zip(rows[128], control, strict=True)
    )
    if all(delta >= 0 for delta in deltas) and any(delta > 0 for delta in deltas[:4]):
        return "root-position-full-advances"
    if all(abs(delta) <= 2 for delta in deltas):
        return "root-position-full-parity"
    if any(delta > 2 for delta in deltas) and any(delta < -2 for delta in deltas):
        return "root-position-full-mixed"
    return "root-position-full-underperforms"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True, choices=("g1-4x5",))
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_motion_anchor_position_h24_walk_runs"),
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--budget", choices=tuple(_BUDGETS), default="early")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    budget = resolve_budget(args.budget)
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        reference_path=args.reference_path.resolve(),
        code_commit=args.code_commit,
        budget=budget,
    )
    from src.algorithms.shac.algorithm import train
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.run_g1_tracking_shac import configure_jax
    from tools.run_g1_zero_assistance_consolidation import _write_json_atomically

    _write_json_atomically(output_root / "preflight.json", preflight)
    configure_jax()
    kwargs = build_motion_anchor_position_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        budget=budget,
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_budget_training_artifacts(
        run_directory,
        expected_kwargs=kwargs,
        budget=budget,
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    print(run_directory)


if __name__ == "__main__":
    main()
