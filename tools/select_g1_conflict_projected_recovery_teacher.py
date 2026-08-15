"""Select E042 from its exact paired carried and ordinary evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.evaluate_g1_zero_head_feature_transfer import aggregate_selection


TRAINING_PROTOCOL = "g1-conflict-projected-recovery-teacher-training-v1"
SELECTION_PROTOCOL = "g1-conflict-projected-recovery-teacher-selection-v1"
OUTCOME_LABELS = {
    "solve": "teacher-objective-solve",
    "advance": "teacher-objective-advance",
    "insufficient": "teacher-objective-insufficient",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--training-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = aggregate_selection(
        evaluation_root=args.evaluation_root,
        training_validation_path=args.training_validation,
        output_path=args.output,
        expected_code_commit=args.code_commit,
        expected_training_protocol=TRAINING_PROTOCOL,
        output_protocol=SELECTION_PROTOCOL,
        outcome_labels=OUTCOME_LABELS,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
