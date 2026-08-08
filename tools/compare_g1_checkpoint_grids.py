"""Compare paired exact-reset and recovery-reset checkpoint grids."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def validate_pair(control: dict, treatment: dict) -> tuple[int, ...]:
    """Require aligned four-checkpoint grids on the same reference."""
    control_steps = tuple(control.get("checkpoint_steps", ()))
    treatment_steps = tuple(treatment.get("checkpoint_steps", ()))
    if control_steps != treatment_steps or len(control_steps) != 4:
        raise ValueError("paired checkpoint steps must match exactly")
    if control.get("reference_sha256") != treatment.get("reference_sha256"):
        raise ValueError("paired reference SHA-256 values must match")
    expected_keys = {str(step) for step in control_steps}
    for payload in (control, treatment):
        if set(payload.get("steps", {})) != expected_keys:
            raise ValueError("grid survival values are incomplete")
        if set(payload.get("completed_suffix", {})) != expected_keys:
            raise ValueError("grid completion values are incomplete")
    return control_steps


def classify_pair(
    *,
    control_steps: dict[int, int],
    treatment_steps: dict[int, int],
    control_completed: dict[int, bool],
    treatment_completed: dict[int, bool],
    minimum_gain: int,
) -> str:
    """Apply the preregistered paired recovery-reset decision."""
    keys = set(control_steps)
    if (
        not keys
        or set(treatment_steps) != keys
        or set(control_completed) != keys
        or set(treatment_completed) != keys
    ):
        raise ValueError("paired results must share checkpoint steps")
    if minimum_gain <= 0:
        raise ValueError("minimum gain must be positive")
    if any(treatment_completed.values()) and not any(
        control_completed.values()
    ):
        return "recovery-completion-advantage"
    gain = max(treatment_steps.values()) - max(control_steps.values())
    if gain >= minimum_gain:
        return "material-recovery-gain"
    return "no-material-recovery-gain"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--treatment-summary", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _integer_map(payload: dict, name: str) -> dict[int, int]:
    return {int(step): int(value) for step, value in payload[name].items()}


def _boolean_map(payload: dict, name: str) -> dict[int, bool]:
    return {int(step): bool(value) for step, value in payload[name].items()}


def main() -> None:
    args = build_parser().parse_args()
    control = json.loads(args.control_summary.read_text())
    treatment = json.loads(args.treatment_summary.read_text())
    checkpoint_steps = validate_pair(control, treatment)
    control_survival = _integer_map(control, "steps")
    treatment_survival = _integer_map(treatment, "steps")
    control_completed = _boolean_map(control, "completed_suffix")
    treatment_completed = _boolean_map(treatment, "completed_suffix")
    decision = classify_pair(
        control_steps=control_survival,
        treatment_steps=treatment_survival,
        control_completed=control_completed,
        treatment_completed=treatment_completed,
        minimum_gain=args.minimum_gain,
    )
    best_control = max(control_survival.values())
    best_treatment = max(treatment_survival.values())
    payload = {
        "checkpoint_steps": list(checkpoint_steps),
        "control_steps": {str(k): v for k, v in control_survival.items()},
        "treatment_steps": {
            str(k): v for k, v in treatment_survival.items()
        },
        "best_control_steps": best_control,
        "best_treatment_steps": best_treatment,
        "treatment_gain_steps": best_treatment - best_control,
        "minimum_gain_steps": args.minimum_gain,
        "decision": decision,
        "reference_sha256": control["reference_sha256"],
        "control_summary": str(args.control_summary.resolve()),
        "treatment_summary": str(args.treatment_summary.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "paired_checkpoint_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot(
        checkpoint_steps,
        [control_survival[step] for step in checkpoint_steps],
        marker="o",
        label="exact-reset control",
    )
    axis.plot(
        checkpoint_steps,
        [treatment_survival[step] for step in checkpoint_steps],
        marker="o",
        label="RMR recovery resets",
    )
    axis.set(
        xlabel="Global training transitions",
        ylabel="Strict phase-zero transitions survived",
        title="Paired G1 recovery-reset continuation",
    )
    axis.legend(frameon=False)
    figure.savefig(args.output_dir / "paired_checkpoint_survival.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
