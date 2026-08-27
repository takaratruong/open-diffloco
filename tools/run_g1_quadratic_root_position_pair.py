"""Run matched exponential and quadratic root-position continuations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from tools.run_g1_dual_scale_root_position import (
    REFERENCE_SHA256,
    classify_pair,
    sha256_file,
)
from tools.run_g1_root_velocity_continuation import _render_command
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


def build_arm_command(
    *,
    solver_profile: str,
    reference: Path,
    checkpoint: Path,
    output: Path,
    code_commit: str,
    kernel: str,
    seed: int,
) -> list[str]:
    """Build one immutable arm command."""
    if kernel not in {"exponential", "quadratic"}:
        raise ValueError("pair arm must be exponential or quadratic")
    return [
        sys.executable,
        "-m",
        "tools.run_g1_dual_scale_root_position",
        "--solver-profile",
        solver_profile,
        "--reference-path",
        str(reference),
        "--resume-from",
        str(checkpoint),
        "--output-root",
        str(output),
        "--code-commit",
        code_commit,
        "--kernel",
        kernel,
        "--seed",
        str(seed),
    ]


def classify_arm_payloads(
    control: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, object]:
    """Validate both arm summaries and apply the preregistered safe gate."""
    if control.get("kernel") != "exponential":
        raise ValueError("control arm kernel is invalid")
    if treatment.get("kernel") != "quadratic":
        raise ValueError("treatment arm kernel is invalid")
    if control.get("source_survival") != treatment.get("source_survival"):
        raise ValueError("paired arms disagree on source survival")
    control_candidates = control.get("candidates")
    treatment_candidates = treatment.get("candidates")
    if not isinstance(control_candidates, dict) or not isinstance(
        treatment_candidates, dict
    ):
        raise ValueError("paired arm candidates are missing")
    try:
        normalized_control = {
            int(step): record for step, record in control_candidates.items()
        }
        normalized_treatment = {
            int(step): record for step, record in treatment_candidates.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("paired arm checkpoint keys are invalid") from error
    return classify_pair(
        control=normalized_control,
        treatment=normalized_treatment,
        source_survival=control["source_survival"],
        treatment_label="quadratic",
    )


def _available_gpu_indices() -> set[str]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line.strip() for line in output.splitlines() if line.strip()}


def _run_arms(
    commands: dict[str, list[str]],
    devices: dict[str, str],
    output_root: Path,
) -> None:
    """Run both arms concurrently and terminate the peer on first failure."""
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs: dict[str, Any] = {}
    try:
        for label in ("control", "treatment"):
            log_path = output_root / f"{label}.log"
            logs[label] = log_path.open("wb")
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": devices[label],
                "JAX_ENABLE_X64": "true",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "MUJOCO_GL": "egl",
                "PYTHONPATH": ".",
            }
            processes[label] = subprocess.Popen(
                commands[label],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=logs[label],
                stderr=subprocess.STDOUT,
            )
        pending = set(processes)
        while pending:
            for label in tuple(pending):
                return_code = processes[label].poll()
                if return_code is None:
                    continue
                pending.remove(label)
                if return_code != 0:
                    for peer in pending:
                        processes[peer].terminate()
                    for peer in pending:
                        processes[peer].wait(timeout=30)
                    raise RuntimeError(
                        f"{label} arm failed with return code {return_code}"
                    )
            if pending:
                time.sleep(1.0)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in logs.values():
            stream.close()


def _plot_survival(selection: dict[str, object], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    steps = [record["checkpoint_step"] for record in selection["control"]]
    for index, (axis, phase) in enumerate(
        zip(axes, (0, 25, 50, 75, 100), strict=True)
    ):
        axis.axhline(
            selection["source_survival"][index],
            color="black",
            linestyle="--",
            linewidth=1,
            label="E002" if index == 0 else None,
        )
        axis.plot(
            steps,
            [record["survival"][index] for record in selection["control"]],
            marker="o",
            label="exponential" if index == 0 else None,
        )
        axis.plot(
            steps,
            [record["survival"][index] for record in selection["treatment"]],
            marker="o",
            label="quadratic" if index == 0 else None,
        )
        axis.set_title(f"phase {phase}")
        axis.set_xlabel("transition step")
        axis.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("steps survived")
    axes[0].legend(fontsize=8)
    figure.suptitle("Matched root-position reward kernels")
    figure.tight_layout()
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.replace(temporary, output)


def _rank(record: dict[str, object]) -> tuple[float, float, float, int]:
    survival = record["survival"]
    return (
        float(min(survival)),
        float(statistics.median(survival)),
        float(statistics.fmean(survival)),
        -int(record["checkpoint_step"]),
    )


def _publish_selection(
    *,
    output_root: Path,
    reference: Path,
    selection: dict[str, object],
) -> dict[str, object]:
    records = selection["treatment"]
    if selection["policy_retained"]:
        rendered = next(
            record
            for record in records
            if record["checkpoint_step"]
            == selection["selected_treatment_step"]
        )
        purpose = "retained-policy"
    else:
        rendered = max(records, key=_rank)
        purpose = "diagnostic-only"
    validation = json.loads(
        (output_root / "treatment" / "training_validation.json").read_text(
            encoding="utf-8"
        )
    )
    run_directory = Path(validation["run_directory"])
    checkpoint = run_directory / f"checkpoint_step_{rendered['checkpoint_step']}.pkl"
    preview = output_root / "diagnostic_preview"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "MUJOCO_GL": "egl",
    }
    subprocess.run(
        _render_command(
            checkpoint=checkpoint,
            reference=reference,
            output=preview,
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    summary_path = preview / "summary.json"
    video_path = preview / "evaluation.mp4"
    contact_sheet_path = preview / "contact_sheet.png"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("checkpoint_sha256") != rendered["checkpoint_sha256"]
        or summary.get("reference_sha256") != REFERENCE_SHA256
        or summary.get("tracking_anchor_position_kernel") != "quadratic"
        or summary.get("tracking_root_velocity_weight") != 1.0
        or not video_path.is_file()
        or not contact_sheet_path.is_file()
    ):
        raise ValueError("quadratic diagnostic preview is invalid")
    _plot_survival(selection, output_root / "learning_curves.png")
    selection.update(
        rendered_treatment_step=rendered["checkpoint_step"],
        render_purpose=purpose,
        diagnostic_summary_sha256=sha256_file(summary_path),
        diagnostic_mp4_sha256=sha256_file(video_path),
        diagnostic_contact_sheet_sha256=sha256_file(contact_sheet_path),
        learning_curves_sha256=sha256_file(output_root / "learning_curves.png"),
    )
    _write_json_atomically(output_root / "selection.json", selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--control-device", required=True)
    parser.add_argument("--treatment-device", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("paired experiment seed must equal zero")
    devices = {
        "control": args.control_device,
        "treatment": args.treatment_device,
    }
    if len(set(devices.values())) != 2:
        raise ValueError("paired arms require distinct GPUs")
    available = _available_gpu_indices()
    if not set(devices.values()) <= available:
        raise ValueError("paired experiment requested an unavailable GPU")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    common = {
        "solver_profile": args.solver_profile,
        "reference": args.reference_path.resolve(),
        "checkpoint": args.resume_from.resolve(),
        "code_commit": args.code_commit,
        "seed": args.seed,
    }
    commands = {
        "control": build_arm_command(
            **common,
            output=output_root / "control",
            kernel="exponential",
        ),
        "treatment": build_arm_command(
            **common,
            output=output_root / "treatment",
            kernel="quadratic",
        ),
    }
    _run_arms(commands, devices, output_root)
    control = json.loads(
        (output_root / "control" / "arm_results.json").read_text(
            encoding="utf-8"
        )
    )
    treatment = json.loads(
        (output_root / "treatment" / "arm_results.json").read_text(
            encoding="utf-8"
        )
    )
    selection = classify_arm_payloads(control, treatment)
    _publish_selection(
        output_root=output_root,
        reference=args.reference_path.resolve(),
        selection=selection,
    )


if __name__ == "__main__":
    main()
