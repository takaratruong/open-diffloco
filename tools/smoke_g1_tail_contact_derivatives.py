"""Fail-closed CLI for the single operational G1 contact-derivative smoke."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _reject_e012_output(output_dir: Path) -> None:
    resolved = Path(output_dir).resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "E-20260809-012" and candidate.parent.name == "runs":
            raise ValueError(
                "the operational smoke must not write within runs/E-20260809-012"
            )


def _load_run_smoke() -> Callable[..., Any]:
    """Import JAX/MJX execution only after all host validation succeeds."""

    from src.algorithms.shac.g1_tail_contact_derivative_smoke import (
        run_one_case_smoke,
    )

    return run_one_case_smoke


def main(
    argv: Sequence[str] | None = None,
    *,
    build_parser_impl: Callable[[], Any] | None = None,
    validate_contract_impl: Callable[[Any], Any] | None = None,
    validate_e011_impl: Callable[[Path], Path] | None = None,
    run_smoke_impl: Callable[..., Any] | None = None,
    load_run_smoke_impl: Callable[[], Callable[..., Any]] | None = None,
) -> Any:
    """Validate frozen inputs, run one case, and return its classification."""

    from tools.audit_g1_shac_gradient_quality import (
        build_parser,
        validate_audit_contract,
    )
    from tools.audit_g1_tail_contact_derivatives import validate_e011_run_dir

    parser = (build_parser_impl or build_parser)()
    parser.description = "Run the single canonical G1 contact-derivative smoke."
    parser.add_argument("--e011-run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        e011_run_dir = (validate_e011_impl or validate_e011_run_dir)(
            args.e011_run_dir
        )
        _reject_e012_output(args.output_dir)
        contract = (validate_contract_impl or validate_audit_contract)(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    implementation = run_smoke_impl or (load_run_smoke_impl or _load_run_smoke)()
    return implementation(contract, e011_run_dir)


if __name__ == "__main__":
    main()
