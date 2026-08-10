"""Named, auditable MJX solver configurations for G1 experiments."""

from contextlib import nullcontext
from dataclasses import dataclass

from src.envs.g1_tracking.fixed_solver import fixed_mjx_solver_outer_loop


@dataclass(frozen=True)
class SolverProfile:
    iterations: int
    ls_iterations: int
    fixed_scan: bool


SOLVER_PROFILES = {
    "upstream-1x5": SolverProfile(1, 5, False),
    "g1-4x5": SolverProfile(4, 5, True),
    "diagnostic-10x20": SolverProfile(10, 20, True),
}


def get_solver_profile(name: str) -> SolverProfile:
    """Resolve a registered profile without accepting silent near-matches."""
    try:
        return SOLVER_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SOLVER_PROFILES))
        raise ValueError(
            f"unknown solver profile {name!r}; choose one of: {choices}"
        ) from exc


def solver_context(profile: SolverProfile):
    """Return the process-local solver scope selected by ``profile``."""
    if profile.fixed_scan:
        return fixed_mjx_solver_outer_loop()
    return nullcontext()
