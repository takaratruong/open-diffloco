"""Run the guarded E002 support-objective fresh-reference mixture treatment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
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
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.run_g1_dual_scale_root_position import (
    E002_SURVIVAL,
    REFERENCE_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_root_velocity_continuation import (
    _phase_grid_command,
    _render_command,
)
from tools.run_g1_support_aware_impulse_continuation import (
    END_STEP,
    START_STEP,
    SUPPORT_TARGET_SHA256,
    build_support_aware_kwargs,
    classify_target_reachability,
    support_target_metrics,
    validate_target_artifact,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import (
    _write_json_atomically,
)


EFFECTIVE_POPULATION = 512
UNROLL_LENGTH = 24
STEPS_PER_UPDATE = EFFECTIVE_POPULATION * UNROLL_LENGTH
FRESH_REFERENCE_FRACTION = 0.25
FRESH_REFERENCE_COUNT = 128
CHECKPOINT_UPDATES = (1, 2, 4, 8)
NO_REFRESH_SURVIVAL = (98, 113, 78, 106, 95)
NO_REFRESH_MAXIMUM_E002_DEFICIT = 38


def checkpoint_steps() -> tuple[int, ...]:
    """Return the four preregistered transient-learning boundaries."""

    return tuple(
        START_STEP + update * STEPS_PER_UPDATE
        for update in CHECKPOINT_UPDATES
    )


def build_fresh_reference_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    target_path: str | Path,
) -> dict[str, Any]:
    """Change E031 only by adding the update-boundary reset mixture."""

    kwargs = build_support_aware_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        target_path,
    )
    kwargs.update(
        actor_update_fresh_reference_fraction=FRESH_REFERENCE_FRACTION,
        allow_resume_actor_update_fresh_reference_change=True,
        checkpoint_steps=checkpoint_steps(),
    )
    return kwargs


def _validate_survival(value: object) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in value
        )
    ):
        raise ValueError("phase-grid survival must contain five positive integers")
    return value


def _enrich_checkpoint(row: dict[str, object]) -> dict[str, object]:
    step = row.get("step")
    if step not in checkpoint_steps():
        raise ValueError("candidate checkpoint step is not registered")
    survival = _validate_survival(row.get("survival"))
    deficits = [
        max(baseline - value, 0)
        for value, baseline in zip(survival, E002_SURVIVAL, strict=True)
    ]
    gains = [
        max(value - baseline, 0)
        for value, baseline in zip(survival, E002_SURVIVAL, strict=True)
    ]
    preserves = not any(deficits)
    improves = any(gains)
    return {
        **row,
        "minimum": min(survival),
        "median": float(statistics.median(survival)),
        "mean": float(statistics.fmean(survival)),
        "e002_deficits": deficits,
        "e002_gains": gains,
        "maximum_e002_deficit": max(deficits),
        "total_e002_deficit": sum(deficits),
        "total_e002_gain": sum(gains),
        "componentwise_preserves_e002": preserves,
        "strictly_improves_any_phase": improves,
    }


def select_checkpoint(rows: list[dict[str, object]]) -> dict[str, object]:
    """Select safe gain first, otherwise the least-forgetting checkpoint."""

    if len(rows) != len(checkpoint_steps()):
        raise ValueError("all four checkpoint rows are required")
    enriched = [_enrich_checkpoint(row) for row in rows]
    if len({int(row["step"]) for row in enriched}) != len(enriched):
        raise ValueError("candidate checkpoint steps must be unique")
    safe = [
        row
        for row in enriched
        if bool(row["componentwise_preserves_e002"])
    ]
    if safe:
        return max(
            safe,
            key=lambda row: (
                int(row["total_e002_gain"]),
                int(row["minimum"]),
                float(row["median"]),
                float(row["mean"]),
                -int(row["step"]),
            ),
        )
    return min(
        enriched,
        key=lambda row: (
            int(row["maximum_e002_deficit"]),
            int(row["total_e002_deficit"]),
            -int(row["minimum"]),
            -float(row["median"]),
            -float(row["mean"]),
            int(row["step"]),
        ),
    )


def classify_mixture(
    *,
    componentwise_preserves: bool,
    strictly_improves: bool,
    target_reached: bool,
    maximum_e002_deficit: int,
) -> tuple[str, bool]:
    """Apply the behavior-first distribution-preservation outcome map."""

    if componentwise_preserves:
        if strictly_improves and target_reached:
            return "fresh-mixture-consolidates", True
        if target_reached:
            return "fresh-mixture-target-parity", False
        return "fresh-mixture-preserves-without-target-learning", False
    if maximum_e002_deficit < NO_REFRESH_MAXIMUM_E002_DEFICIT:
        return "fresh-mixture-mitigates-reversal", False
    if strictly_improves:
        return "fresh-mixture-redistributes", False
    return "fresh-mixture-insufficient", False


def _finite_tree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(tree)
    )


def _finite_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("training telemetry array is incomplete or non-finite")
    return array


def _validate_gradient_decomposition(row: dict[str, object]) -> dict[str, object]:
    counts = _finite_array(
        row.get("actor_grad_fresh_reference_bin_counts"), (2,)
    )
    if not np.array_equal(counts, [384, 128]):
        raise ValueError("fresh/carried gradient counts are not exact")
    cosine = _finite_array(
        row.get("actor_grad_fresh_reference_bin_cosine_matrix"), (2, 2)
    )
    scalar_names = (
        "within_variance_trace",
        "between_variance_trace",
        "total_variance_trace",
        "within_variance_fraction",
        "between_variance_fraction",
    )
    values = {}
    for name in scalar_names:
        value = row.get(f"actor_grad_fresh_reference_{name}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("fresh/carried gradient variance is invalid")
        values[name] = float(value)
    tolerance = 1e-5 * max(values["total_variance_trace"], 1.0)
    population_variance = float(row["actor_grad_population_variance_trace"])
    if (
        abs(
            values["within_variance_trace"]
            + values["between_variance_trace"]
            - values["total_variance_trace"]
        )
        > tolerance
        or abs(values["total_variance_trace"] - population_variance)
        > tolerance
        or abs(
            values["within_variance_fraction"]
            + values["between_variance_fraction"]
            - 1.0
        )
        > 1e-5
    ):
        raise ValueError("fresh/carried gradient variance does not close")
    return {
        "counts": counts.astype(int).tolist(),
        "cosine": float(cosine[0, 1]),
        "between_variance_fraction": values[
            "between_variance_fraction"
        ],
    }


def validate_training_artifacts(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    target_path: Path,
) -> dict[str, object]:
    """Verify exact lineage, reset dose, and fresh-vs-carried gradients."""

    root = run_directory.resolve()
    hparams = json.loads((root / "hparams.json").read_text(encoding="utf-8"))
    required_hparams = {
        "actor_frozen_controller_residual": True,
        "actor_frozen_controller_residual_depth": 2,
        "actor_support_aware_impulse": True,
        "actor_support_aware_impulse_path": str(target_path.resolve()),
        "actor_support_aware_impulse_sha256": SUPPORT_TARGET_SHA256,
        "actor_support_aware_impulse_window": 4,
        "actor_support_aware_impulse_delta": 0.1,
        "actor_support_aware_impulse_weight": 1.0,
        "actor_update_fresh_reference_fraction": FRESH_REFERENCE_FRACTION,
        "actor_update_fresh_reference_count": FRESH_REFERENCE_COUNT,
        "allow_resume_actor_update_fresh_reference_change": True,
        "actor_centroidal_propulsion": False,
        "actor_capture_point_tracking": False,
        "actor_counterfactual_wrench_distillation": False,
        "torso_wrench_assistance": False,
        "actor_learned_torso_wrench": False,
        "domain_randomization": False,
        "reference_reset_noise_scale": 0.0,
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "gradient_accumulation_steps": 2,
        "unroll_length": UNROLL_LENGTH,
        "total_steps": END_STEP,
        "checkpoint_steps": list(checkpoint_steps()),
    }
    if any(hparams.get(key) != value for key, value in required_hparams.items()):
        raise ValueError("fresh-reference hparams do not match the treatment")
    report = hparams.get("actor_support_aware_impulse_target_report")
    if (
        not isinstance(report, dict)
        or report.get("valid") is not True
        or report.get("artifact_sha256") != SUPPORT_TARGET_SHA256
    ):
        raise ValueError("support target load report is invalid")

    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    if frozen_controller_residual_depth(source.actor_params) != 1:
        raise ValueError("source checkpoint is not exact depth-one E002")
    source_actor_hash = parameter_tree_sha256(source.actor_params)
    source_opt_hash = parameter_tree_sha256(source.actor_opt)
    source_normalizer_hash = parameter_tree_sha256(source.normalizer)

    rows = json.loads(
        (root / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    if [row.get("step") for row in rows] != list(checkpoint_steps()):
        raise ValueError("fresh-reference checkpoint telemetry grid is invalid")
    summaries = []
    for row in rows:
        step = int(row["step"])
        checkpoint = root / f"checkpoint_step_{step}.pkl"
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        if (
            int(state.step) != step
            or not isinstance(state.actor_params, FrozenControllerResidualParams)
            or not isinstance(state.actor_opt, FrozenControllerResidualOptState)
            or frozen_controller_residual_depth(state.actor_params) != 2
            or not _finite_tree(state)
            or parameter_tree_sha256(state.actor_params.parent)
            != source_actor_hash
            or parameter_tree_sha256(state.actor_opt.parent_optimizer_state)
            != source_opt_hash
            or parameter_tree_sha256(state.normalizer)
            != source_normalizer_hash
        ):
            raise ValueError("fresh-reference checkpoint violates frozen E002")
        phase_counts = _finite_array(
            row.get("actor_update_fresh_reference_phase_bin_counts"), (5,)
        )
        if (
            row.get("actor_update_fresh_reference_count")
            != FRESH_REFERENCE_COUNT
            or row.get("actor_update_fresh_reference_actual_fraction")
            != FRESH_REFERENCE_FRACTION
            or int(phase_counts.sum()) != FRESH_REFERENCE_COUNT
            or not np.all(phase_counts > 0)
            or row.get("actor_preview_valid") is not True
            or row.get("actor_cagrad_valid") is not True
            or row.get("actor_preview_frozen_parameter_drift_max_abs") != 0.0
            or row.get("actor_preview_frozen_moment_drift_max_abs") != 0.0
            or row.get("actor_preview_normalizer_drift_max_abs") != 0.0
        ):
            raise ValueError("fresh-reference treatment telemetry is invalid")
        for name in (
            "actor_preview_gradient_norm",
            "actor_preview_update_norm",
            "actor_support_aware_impulse_valid_window_count",
        ):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError("fresh-reference update is not finite and live")
        gradient = _validate_gradient_decomposition(row)
        summaries.append(
            {
                "step": step,
                "checkpoint_sha256": sha256_file(checkpoint),
                "fresh_phase_bin_counts": phase_counts.astype(int).tolist(),
                "support_loss": float(
                    row["actor_support_aware_impulse_loss"]
                ),
                "support_heldout_loss": float(
                    row["actor_support_aware_impulse_heldout_loss"]
                ),
                "support_valid_window_count": int(
                    row[
                        "actor_support_aware_impulse_valid_window_count"
                    ]
                ),
                "gradient": gradient,
            }
        )
    return {
        "valid": True,
        "protocol": "g1-fresh-reference-mixture-training-v1",
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_actor_tree_sha256": source_actor_hash,
        "source_optimizer_tree_sha256": source_opt_hash,
        "source_normalizer_tree_sha256": source_normalizer_hash,
        "fresh_reference_fraction": FRESH_REFERENCE_FRACTION,
        "fresh_reference_count": FRESH_REFERENCE_COUNT,
        "carried_state_count": EFFECTIVE_POPULATION - FRESH_REFERENCE_COUNT,
        "checkpoints": summaries,
    }


def _plot_selection(selection: dict[str, object], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = np.asarray([0, 25, 50, 75, 100])
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(phases, E002_SURVIVAL, marker="o", label="retained E002")
    axis.plot(
        phases,
        NO_REFRESH_SURVIVAL,
        marker="o",
        label="no-refresh update 8",
    )
    for row in selection["checkpoints"]:
        axis.plot(
            phases,
            row["survival"],
            marker=".",
            alpha=0.65,
            label=f"25% fresh update {row['update']}",
        )
    axis.set_xlabel("reference start phase")
    axis.set_ylabel("transitions survived")
    axis.set_title("Fresh-reference mixture behavior gate")
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
    target_path: Path,
    reference: Path,
    output_root: Path,
    code_commit: str,
) -> dict[str, object]:
    """Evaluate all transient checkpoints and gate the least-forgetting one."""

    phase_root = output_root / "phase_grid"
    phase_root.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "MUJOCO_GL": "egl",
    }
    repository = Path(__file__).resolve().parents[2]
    source_output = phase_root / "source_e002.json"
    subprocess.run(
        _phase_grid_command(
            checkpoint=source_checkpoint,
            reference=reference,
            output=source_output,
            code_commit=code_commit,
        ),
        cwd=repository,
        env=environment,
        check=True,
    )
    source_payload = json.loads(source_output.read_text(encoding="utf-8"))
    if source_payload.get("summary", {}).get("survival") != list(E002_SURVIVAL):
        raise ValueError("current source evaluator does not reproduce E002")

    candidates = []
    for update, step in zip(CHECKPOINT_UPDATES, checkpoint_steps(), strict=True):
        checkpoint = run_directory / f"checkpoint_step_{step}.pkl"
        output = phase_root / f"checkpoint_step_{step}.json"
        subprocess.run(
            _phase_grid_command(
                checkpoint=checkpoint,
                reference=reference,
                output=output,
                code_commit=code_commit,
            ),
            cwd=repository,
            env=environment,
            check=True,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if (
            payload.get("checkpoint_sha256") != sha256_file(checkpoint)
            or payload.get("reference_sha256") != REFERENCE_SHA256
            or payload.get("summary", {}).get("phases")
            != [0, 25, 50, 75, 100]
        ):
            raise ValueError("candidate phase-grid evidence is invalid")
        candidates.append(
            {
                "step": step,
                "update": update,
                "checkpoint_sha256": sha256_file(checkpoint),
                "survival": _validate_survival(
                    payload.get("summary", {}).get("survival")
                ),
                "phase_grid_sha256": sha256_file(output),
            }
        )
    selected = select_checkpoint(candidates)
    selected_checkpoint = run_directory / (
        f"checkpoint_step_{selected['step']}.pkl"
    )

    previews = {}
    for label, checkpoint in (
        ("source_e002", source_checkpoint),
        ("selected_candidate", selected_checkpoint),
    ):
        output = output_root / label
        subprocess.run(
            _render_command(
                checkpoint=checkpoint,
                reference=reference,
                output=output,
            ),
            cwd=repository,
            env=environment,
            check=True,
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("checkpoint_sha256") != sha256_file(checkpoint)
            or summary.get("reference_sha256") != REFERENCE_SHA256
            or not (output / "evaluation.mp4").is_file()
            or not (output / "contact_sheet.png").is_file()
        ):
            raise ValueError("fresh-reference preview evidence is invalid")
        previews[label] = {
            "target_metrics": support_target_metrics(
                output / "evaluation.npz", target_path=target_path
            ),
            "summary_sha256": sha256_file(output / "summary.json"),
            "evaluation_npz_sha256": sha256_file(output / "evaluation.npz"),
            "evaluation_mp4_sha256": sha256_file(output / "evaluation.mp4"),
            "contact_sheet_sha256": sha256_file(
                output / "contact_sheet.png"
            ),
        }
    target_gate = classify_target_reachability(
        source=previews["source_e002"]["target_metrics"],
        candidate=previews["selected_candidate"]["target_metrics"],
    )
    outcome, retained = classify_mixture(
        componentwise_preserves=bool(
            selected["componentwise_preserves_e002"]
        ),
        strictly_improves=bool(selected["strictly_improves_any_phase"]),
        target_reached=bool(target_gate["target_reached"]),
        maximum_e002_deficit=int(selected["maximum_e002_deficit"]),
    )
    selection = {
        "protocol": "g1-fresh-reference-mixture-selection-v1",
        "phases": [0, 25, 50, 75, 100],
        "source_survival": list(E002_SURVIVAL),
        "no_refresh_update8_survival": list(NO_REFRESH_SURVIVAL),
        "no_refresh_maximum_e002_deficit": (
            NO_REFRESH_MAXIMUM_E002_DEFICIT
        ),
        "fresh_reference_fraction": FRESH_REFERENCE_FRACTION,
        "checkpoints": [_enrich_checkpoint(row) for row in candidates],
        "selected": selected,
        **target_gate,
        "source_target_metrics": previews["source_e002"]["target_metrics"],
        "candidate_target_metrics": previews["selected_candidate"][
            "target_metrics"
        ],
        "source_preview": previews["source_e002"],
        "candidate_preview": previews["selected_candidate"],
        "outcome": outcome,
        "policy_retained": retained,
        "retained_checkpoint": (
            str(selected_checkpoint.resolve()) if retained else None
        ),
    }
    _plot_selection(selection, output_root / "learning_curves.png")
    selection["learning_curves_sha256"] = sha256_file(
        output_root / "learning_curves.png"
    )
    _write_json_atomically(output_root / "selection.json", selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--support-target", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("fresh-reference mixture seed must equal zero")
    repository = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.resume_from.resolve(),
        reference=args.reference_path.resolve(),
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-fresh-reference-mixture-preflight-v1",
        start_step=START_STEP,
        end_step=END_STEP,
        checkpoint_updates=list(CHECKPOINT_UPDATES),
        checkpoint_steps=list(checkpoint_steps()),
        effective_population=EFFECTIVE_POPULATION,
        fresh_reference_fraction=FRESH_REFERENCE_FRACTION,
        fresh_reference_count=FRESH_REFERENCE_COUNT,
        carried_state_count=EFFECTIVE_POPULATION - FRESH_REFERENCE_COUNT,
        no_refresh_survival=list(NO_REFRESH_SURVIVAL),
        support_target=validate_target_artifact(args.support_target),
    )
    _write_json_atomically(output_root / "preflight.json", preflight)

    kwargs = build_fresh_reference_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
        args.resume_from.resolve(),
        args.support_target.resolve(),
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
        target_path=args.support_target.resolve(),
    )
    validation["run_directory"] = str(run_directory)
    _write_json_atomically(output_root / "training_validation.json", validation)
    evaluate_and_select(
        run_directory,
        source_checkpoint=args.resume_from.resolve(),
        target_path=args.support_target.resolve(),
        reference=args.reference_path.resolve(),
        output_root=output_root,
        code_commit=args.code_commit,
    )
    print(run_directory)


if __name__ == "__main__":
    main()
