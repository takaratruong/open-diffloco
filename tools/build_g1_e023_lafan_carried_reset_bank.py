"""Build E023 pre-failure carried resets on the pinned 10-second LAFAN walk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from tools.build_g1_e023_carried_reset_bank import (
    _sha256,
    collect_e023_bank,
    validate_code_commit,
    validate_e023_hparams,
)
from tools.build_g1_history_carried_reset_bank import (
    HISTORY_LEN,
    LOOKAHEAD_STEPS,
    _write_json_atomically,
    _write_npz_atomically,
    validate_history_bank,
)
from tools.run_g1_root_recovery_continuation import validate_runtime_assets


PROTOCOL = "g1-e023-lafan-history-carried-reset-bank-v1"
LAFAN_SOURCE_PHASES = (0, 100, 200, 300, 400)
E023_LAFAN_ZERO_SHOT = (118, 63, 49, 39, 46)
EXPECTED_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)


def build_lafan_bank_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    observed_survival: tuple[int, ...],
    frame_dim: int,
) -> dict[str, object]:
    """Require the preregistered five exact LAFAN pre-failure bands."""
    if observed_survival != E023_LAFAN_ZERO_SHOT:
        raise ValueError(
            "observed survival does not match the registered E023 LAFAN "
            "zero-shot baseline"
        )
    summary = validate_history_bank(
        arrays,
        expected_source_phases=LAFAN_SOURCE_PHASES,
        expected_survival=E023_LAFAN_ZERO_SHOT,
        history_len=HISTORY_LEN,
        frame_dim=frame_dim,
    )
    if summary["rows"] != 120 or summary["rows_per_source"] != [24] * 5:
        raise ValueError("LAFAN bank must contain five exact 24-row bands")
    return {**summary, "protocol": PROTOCOL}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--hparams-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    hparams = args.hparams.resolve()
    reference = args.reference_path.resolve()
    for path, expected in (
        (checkpoint, args.checkpoint_sha256),
        (hparams, args.hparams_sha256),
        (reference, args.reference_sha256),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"input SHA-256 mismatch: {path}")
    if args.reference_sha256 != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference SHA-256 is not the registered LAFAN walk")
    code_commit = validate_code_commit(repository, args.code_commit)
    hparams_payload = json.loads(hparams.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams_payload)
    runtime_assets = validate_runtime_assets(
        Path(str(hparams_payload["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    arrays, observed_survival = collect_e023_bank(
        checkpoint,
        hparams,
        reference,
        seed=args.seed,
        source_phases=LAFAN_SOURCE_PHASES,
        require_reference_match=False,
    )
    frame_dim = int(arrays["actor_obs_history"].shape[-1])
    summary = build_lafan_bank_summary(
        arrays,
        observed_survival=observed_survival,
        frame_dim=frame_dim,
    )
    output_npz = args.output_npz.resolve()
    _write_npz_atomically(output_npz, arrays)
    payload = {
        **summary,
        **runtime_assets,
        "code_commit": code_commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "hparams_path": str(hparams),
        "hparams_sha256": args.hparams_sha256,
        "reference_path": str(reference),
        "reference_sha256": args.reference_sha256,
        "bank_path": str(output_npz),
        "bank_sha256": _sha256(output_npz),
        "history_len": HISTORY_LEN,
        "actor_frame_obs_dim": frame_dim,
        "lookahead_steps": list(LOOKAHEAD_STEPS),
        "preview_mode": "delta",
        "solver_profile": "g1-4x5",
        "seed": args.seed,
    }
    _write_json_atomically(args.output_json.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
