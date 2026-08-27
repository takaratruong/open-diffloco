"""Train one new zero-head residual over the exact frozen E002 controller."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
import statistics
import subprocess
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualOptState,
    FrozenControllerResidualParams,
    frozen_controller_residual_depth,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.run_g1_dual_scale_root_position import (
    REFERENCE_SHA256,
    build_arm_kwargs,
    evaluate_arm,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_root_velocity_continuation import _render_command
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
UPDATES = 16
TRANSITIONS_PER_UPDATE = 512 * 24
CHECKPOINT_EVERY_UPDATES = 4
CHECKPOINT_INTERVAL = CHECKPOINT_EVERY_UPDATES * TRANSITIONS_PER_UPDATE
END_STEP = START_STEP + UPDATES * TRANSITIONS_PER_UPDATE
E002_SURVIVAL = (136, 144, 84, 90, 79)


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the four registered behavior checkpoints."""
    return tuple(
        range(START_STEP + CHECKPOINT_INTERVAL, END_STEP + 1, CHECKPOINT_INTERVAL)
    )


def build_nested_residual_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Change only the frozen-controller residual depth from one to two."""
    kwargs = build_arm_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        kernel="exponential",
    )
    kwargs.update(
        actor_frozen_controller_residual_depth=2,
        allow_resume_actor_frozen_controller_residual_start=True,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    return kwargs


def _rank(record: dict[str, object]) -> tuple[float, float, float, int]:
    survival = record["survival"]
    assert isinstance(survival, list)
    return (
        float(min(survival)),
        float(statistics.median(survival)),
        float(statistics.fmean(survival)),
        -int(record["checkpoint_step"]),
    )


def classify_selection(
    candidates: dict[int, dict[str, object]],
    *,
    source_survival: list[int],
) -> dict[str, object]:
    """Retain only a componentwise-safe strict improvement over E002."""
    if tuple(source_survival) != E002_SURVIVAL:
        raise ValueError("source E002 survival does not match the registered baseline")
    if set(candidates) != set(expected_checkpoint_steps()):
        raise ValueError("nested residual selection requires the exact checkpoint grid")
    records: list[dict[str, object]] = []
    for step in expected_checkpoint_steps():
        candidate = candidates[step]
        checkpoint_sha256 = candidate.get("checkpoint_sha256")
        survival = candidate.get("survival")
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or not isinstance(survival, list)
            or len(survival) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in survival
            )
        ):
            raise ValueError("nested residual candidate is invalid")
        preserves = all(
            value >= baseline
            for value, baseline in zip(survival, E002_SURVIVAL, strict=True)
        )
        improves = any(
            value > baseline
            for value, baseline in zip(survival, E002_SURVIVAL, strict=True)
        )
        records.append(
            {
                "checkpoint_step": step,
                "checkpoint_sha256": checkpoint_sha256,
                "survival": survival,
                "minimum": min(survival),
                "median": float(statistics.median(survival)),
                "mean": float(statistics.fmean(survival)),
                "eligible": preserves and improves,
                "improves_any_phase": improves,
            }
        )
    eligible = [record for record in records if record["eligible"]]
    selected = max(eligible, key=_rank) if eligible else None
    any_gain = any(record["improves_any_phase"] for record in records)
    return {
        "protocol": "g1-nested-residual-selection-v1",
        "phases": [0, 25, 50, 75, 100],
        "source_survival": source_survival,
        "checkpoints": records,
        "outcome": (
            "nested-residual-advances"
            if selected is not None
            else "nested-residual-redistributes"
            if any_gain
            else "nested-residual-insufficient"
        ),
        "selected_step": selected["checkpoint_step"] if selected else None,
        "selected_checkpoint_sha256": (
            selected["checkpoint_sha256"] if selected else None
        ),
        "selected_survival": selected["survival"] if selected else None,
        "policy_retained": selected is not None,
    }


def _finite_tree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(tree)
    )


def validate_training_artifacts(
    run_directory: Path,
    *,
    source_checkpoint: Path,
) -> dict[str, object]:
    """Prove that E002 stayed frozen and only the new adapter was updated."""
    hparams = json.loads(
        (run_directory / "hparams.json").read_text(encoding="utf-8")
    )
    required_hparams = {
        "actor_frozen_controller_residual": True,
        "actor_frozen_controller_residual_hidden": 256,
        "actor_frozen_controller_residual_depth": 2,
        "tracking_anchor_position_kernel": "exponential",
        "tracking_root_velocity_weight": 1.0,
        "total_steps": END_STEP,
    }
    if any(hparams.get(key) != value for key, value in required_hparams.items()):
        raise ValueError("nested residual hparams do not match the treatment")
    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    if frozen_controller_residual_depth(source.actor_params) != 1:
        raise ValueError("source checkpoint is not exact depth-one E002")
    source_actor_hash = parameter_tree_sha256(source.actor_params)
    source_opt_hash = parameter_tree_sha256(source.actor_opt)
    source_normalizer_hash = parameter_tree_sha256(source.normalizer)
    for step in expected_checkpoint_steps():
        checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        if (
            int(state.step) != step
            or not isinstance(state.actor_params, FrozenControllerResidualParams)
            or not isinstance(state.actor_opt, FrozenControllerResidualOptState)
            or frozen_controller_residual_depth(state.actor_params) != 2
            or not _finite_tree(state)
            or parameter_tree_sha256(state.actor_params.parent) != source_actor_hash
            or parameter_tree_sha256(state.actor_opt.parent_optimizer_state)
            != source_opt_hash
            or parameter_tree_sha256(state.normalizer) != source_normalizer_hash
        ):
            raise ValueError("nested residual checkpoint violates frozen E002")
    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    if [row.get("step") for row in rows] != list(expected_checkpoint_steps()):
        raise ValueError("nested residual telemetry grid is invalid")
    for row in rows:
        scalar_keys = (
            "actor_preview_gradient_norm",
            "actor_preview_update_norm",
        )
        if (
            row.get("actor_preview_valid") is not True
            or any(
                not isinstance(row.get(key), (int, float))
                or isinstance(row.get(key), bool)
                or not math.isfinite(float(row[key]))
                or float(row[key]) <= 0.0
                for key in scalar_keys
            )
            or row.get("actor_preview_frozen_parameter_drift_max_abs") != 0.0
            or row.get("actor_preview_frozen_moment_drift_max_abs") != 0.0
            or row.get("actor_preview_normalizer_drift_max_abs") != 0.0
            or row.get("actor_cagrad_valid") is not True
            or len(row.get("actor_cagrad_bin_counts", [])) != 5
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in row.get("actor_cagrad_bin_counts", [])
            )
        ):
            raise ValueError("nested residual checkpoint telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-nested-residual-training-v1",
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_actor_tree_sha256": source_actor_hash,
        "source_optimizer_tree_sha256": source_opt_hash,
        "source_normalizer_tree_sha256": source_normalizer_hash,
    }


def _plot_survival(selection: dict[str, object], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = selection["checkpoints"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, phase in enumerate(selection["phases"]):
        axis.axhline(E002_SURVIVAL[index], linestyle="--", linewidth=0.8)
        axis.plot(
            [record["checkpoint_step"] for record in records],
            [record["survival"][index] for record in records],
            marker="o",
            label=f"phase {phase}",
        )
    axis.set_title("Frozen E002 plus new zero-head residual")
    axis.set_xlabel("transition step")
    axis.set_ylabel("steps survived")
    axis.legend(fontsize=8)
    figure.tight_layout()
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.replace(temporary, output)


def evaluate_and_select(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    reference: Path,
    output_root: Path,
    code_commit: str,
) -> dict[str, object]:
    """Evaluate all candidates and render the retained or best diagnostic one."""
    arm = evaluate_arm(
        run_directory,
        source_checkpoint=source_checkpoint,
        reference=reference,
        output_root=output_root,
        code_commit=code_commit,
        kernel="exponential",
    )
    candidates = {int(step): value for step, value in arm["candidates"].items()}
    selection = classify_selection(
        candidates,
        source_survival=arm["source_survival"],
    )
    records = selection["checkpoints"]
    rendered = (
        next(
            record
            for record in records
            if record["checkpoint_step"] == selection["selected_step"]
        )
        if selection["policy_retained"]
        else max(records, key=_rank)
    )
    selection["rendered_step"] = rendered["checkpoint_step"]
    selection["render_purpose"] = (
        "retained-policy" if selection["policy_retained"] else "diagnostic-only"
    )
    _plot_survival(selection, output_root / "learning_curves.png")
    render_directory = output_root / "selected_preview"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "MUJOCO_GL": "egl",
    }
    checkpoint = run_directory / f"checkpoint_step_{rendered['checkpoint_step']}.pkl"
    subprocess.run(
        _render_command(
            checkpoint=checkpoint,
            reference=reference,
            output=render_directory,
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    summary_path = render_directory / "summary.json"
    video_path = render_directory / "evaluation.mp4"
    contact_sheet_path = render_directory / "contact_sheet.png"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("checkpoint_sha256") != rendered["checkpoint_sha256"]
        or summary.get("reference_sha256") != REFERENCE_SHA256
        or summary.get("tracking_root_velocity_weight") != 1.0
        or not video_path.is_file()
        or not contact_sheet_path.is_file()
    ):
        raise ValueError("nested residual preview is invalid")
    selection.update(
        render_checkpoint_sha256=summary["checkpoint_sha256"],
        render_summary_sha256=sha256_file(summary_path),
        render_mp4_sha256=sha256_file(video_path),
        render_contact_sheet_sha256=sha256_file(contact_sheet_path),
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
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("nested residual treatment seed must equal zero")
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-nested-residual-preflight-v1",
        start_step=START_STEP,
        end_step=END_STEP,
        updates=UPDATES,
        checkpoint_steps=list(expected_checkpoint_steps()),
        source_residual_depth=1,
        treatment_residual_depth=2,
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_nested_residual_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
    )
    configure_jax()
    previous = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile(args.solver_profile)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory,
        source_checkpoint=args.resume_from.resolve(),
    )
    _write_json_atomically(output_root / "training_validation.json", validation)
    evaluate_and_select(
        run_directory,
        source_checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        output_root=output_root,
        code_commit=args.code_commit,
    )
    print(run_directory)


if __name__ == "__main__":
    main()
