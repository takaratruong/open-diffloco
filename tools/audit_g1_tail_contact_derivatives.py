"""Thin fail-closed CLI for the E012 tail contact-derivative audit."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

E011_FROZEN_FILE_SHA256 = {
    Path("experiment.yaml"): (
        "4046d8e3b593308bc56a0ea944202ddd52d3db1a965b8de33d3bbc5efee07d4d"
    ),
    Path("run.json"): (
        "eba3a3eadaf476627b1176f69fb6a921ec1823944176990d6d66356575a07164"
    ),
    Path("seed-1/evidence/manifest.json"): (
        "f24f856e3173bc7fa9cb8ba1adb8fb78b482941fd57fbdf6d82f589bdac2e2ba"
    ),
    Path("seed-1/evidence/failure_weight_receipts.json"): (
        "c5f9f797abd13b149e5f1839948ef166ed78f7f86ae45fba504e12cd9cbbf9a3"
    ),
    Path("seed-1/evidence/estimator_receipts.json"): (
        "3c847f535b9ca6f49b37570f1e09dceede4e6bb08707403c6e5860544be4e5b3"
    ),
}


def validate_e011_run_dir(
    run_dir: Path,
    *,
    is_dir: Callable[[Path], bool] | None = None,
    is_file: Callable[[Path], bool] | None = None,
    sha256_file_impl: Callable[[Path], str] | None = None,
) -> Path:
    """Bind the immutable completed E011 run before E012 execution."""
    run_dir = Path(run_dir)
    directory_check = is_dir or Path.is_dir
    file_check = is_file or Path.is_file
    if not directory_check(run_dir):
        raise ValueError(f"E011 run directory does not exist: {run_dir}")
    if sha256_file_impl is None:
        from tools.audit_g1_shac_gradient_quality import sha256_file

        sha256_file_impl = sha256_file
    for relative, expected in E011_FROZEN_FILE_SHA256.items():
        path = run_dir / relative
        if not file_check(path):
            raise ValueError(f"E011 frozen file does not exist: {path}")
        if sha256_file_impl(path) != expected:
            raise ValueError(
                f"E011 {relative.as_posix()} SHA-256 differs from the frozen source"
            )
    return run_dir.resolve()


def _load_run_audit() -> Callable[..., Any]:
    """Import simulator execution only after every host contract passes."""
    from src.algorithms.shac.g1_tail_contact_derivative_execution import run_audit

    return run_audit


def main(
    argv: Sequence[str] | None = None,
    *,
    build_parser_impl: Callable[[], Any] | None = None,
    validate_contract_impl: Callable[[Any], Any] | None = None,
    validate_e011_impl: Callable[[Path], Path] | None = None,
    run_audit_impl: Callable[..., Any] | None = None,
    load_run_audit_impl: Callable[[], Callable[..., Any]] | None = None,
) -> Any:
    """Validate the frozen E064 inputs and exact E011 evidence, then delegate."""
    from tools.audit_g1_shac_gradient_quality import (
        build_parser,
        validate_audit_contract,
    )

    parser = (build_parser_impl or build_parser)()
    parser.add_argument("--e011-run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        e011_run_dir = (validate_e011_impl or validate_e011_run_dir)(args.e011_run_dir)
        contract = (validate_contract_impl or validate_audit_contract)(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    implementation = run_audit_impl or (load_run_audit_impl or _load_run_audit)()
    return implementation(contract=contract, e011_run_dir=e011_run_dir)


if __name__ == "__main__":
    main()
