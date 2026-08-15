"""Select E041 only from all registered carried and ordinary evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.evaluate_g1_zero_head_feature_transfer import aggregate_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--training-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = aggregate_selection(
        evaluation_root=args.evaluation_root,
        training_validation_path=args.training_validation,
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
