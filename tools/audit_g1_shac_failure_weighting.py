"""Thin CLI for the phase-preserving failure-weighted G1 audit."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def _load_run_audit() -> Callable[..., Any]:
    """Load the simulator-backed execution only after contract validation."""
    from src.algorithms.shac.g1_failure_weighted_audit_execution import run_audit

    return run_audit


def main(
    argv: Sequence[str] | None = None,
    *,
    shared_main_impl: Callable[..., Any] | None = None,
    run_audit_impl: Callable[..., Any] | None = None,
    load_run_audit_impl: Callable[[], Callable[..., Any]] | None = None,
) -> Any:
    """Reuse the E064 parser and immutable-contract validator unchanged."""
    from tools.audit_g1_shac_gradient_quality import main as shared_main

    if run_audit_impl is None:
        loader = load_run_audit_impl or _load_run_audit

        def run_audit_impl(*args: Any, **kwargs: Any) -> Any:
            return loader()(*args, **kwargs)

    return (shared_main_impl or shared_main)(
        argv,
        run_audit_impl=run_audit_impl,
    )


if __name__ == "__main__":
    main()
