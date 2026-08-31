"""Run the eight-update zero-wrench support-aware joint-policy discriminator."""

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
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.centroidal_objective import (
    load_support_aware_impulse_target,
    support_aware_impulse_objective,
)
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualOptState,
    FrozenControllerResidualParams,
    frozen_controller_residual_depth,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.run_g1_dual_scale_root_position import (
    E002_SURVIVAL,
    REFERENCE_SHA256,
    build_arm_kwargs,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_root_velocity_continuation import (
    _phase_grid_command,
    _render_command,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
UPDATES = 8
TRANSITIONS_PER_UPDATE = 512 * 24
END_STEP = START_STEP + UPDATES * TRANSITIONS_PER_UPDATE
SUPPORT_TARGET_SHA256 = (
    "66f67b9669336e104b2e6a5345b42e4eff976bc31eb6945c6d5a84d1a0ffe980"
)
SUPPORT_COMPONENT_SCALES = np.asarray(
    [26.166128241600006] * 3 + [7.849838472480002] * 3,
    dtype=np.float64,
)
TARGET_LOSS_RELATIVE_IMPROVEMENT = 0.01


def expected_checkpoint_steps() -> tuple[int, ...]:
    """Return the sole registered early-gate checkpoint."""
    return (END_STEP,)


def build_support_aware_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    target_path: str | Path,
) -> dict[str, Any]:
    """Add one zero-head layer and one immutable four-step target to E002."""
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
        actor_support_aware_impulse=True,
        actor_support_aware_impulse_path=str(target_path),
        actor_support_aware_impulse_sha256=SUPPORT_TARGET_SHA256,
        actor_support_aware_impulse_window=4,
        actor_support_aware_impulse_delta=0.1,
        actor_support_aware_impulse_weight=1.0,
        allow_resume_actor_support_aware_impulse_start=True,
        total_steps=END_STEP,
        checkpoint_steps=expected_checkpoint_steps(),
    )
    return kwargs


def classify_candidate(
    *,
    source_survival: list[int],
    candidate_survival: list[int],
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Apply the unchanged componentwise E002 behavior gate."""
    if tuple(source_survival) != E002_SURVIVAL:
        raise ValueError("source E002 survival does not match the registered baseline")
    if (
        not isinstance(candidate_survival, list)
        or len(candidate_survival) != 5
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in candidate_survival
        )
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
    ):
        raise ValueError("support-aware candidate evidence is invalid")
    preserves = all(
        value >= baseline
        for value, baseline in zip(candidate_survival, E002_SURVIVAL, strict=True)
    )
    improves = any(
        value > baseline
        for value, baseline in zip(candidate_survival, E002_SURVIVAL, strict=True)
    )
    return {
        "protocol": "g1-support-aware-joint-selection-v1",
        "phases": [0, 25, 50, 75, 100],
        "source_survival": source_survival,
        "checkpoint_step": END_STEP,
        "checkpoint_sha256": checkpoint_sha256,
        "candidate_survival": candidate_survival,
        "minimum": min(candidate_survival),
        "median": float(statistics.median(candidate_survival)),
        "mean": float(statistics.fmean(candidate_survival)),
        "componentwise_preserves_e002": preserves,
        "strictly_improves_any_phase": improves,
        "outcome": (
            "support-aware-joint-treatment-advances"
            if preserves and improves
            else "support-aware-joint-treatment-redistributes"
            if improves
            else "support-aware-joint-treatment-insufficient"
        ),
        "policy_retained": preserves and improves,
    }


def validate_target_artifact(path: Path) -> dict[str, object]:
    """Verify the exact immutable target before environment construction."""
    resolved = path.resolve()
    if not resolved.is_file() or sha256_file(resolved) != SUPPORT_TARGET_SHA256:
        raise ValueError("support-aware target SHA-256 mismatch")
    with np.load(resolved, allow_pickle=False) as archive:
        starts = np.asarray(archive["window_start_transitions"])
        ends = np.asarray(archive["window_end_transitions_inclusive"])
        scales = np.asarray(archive["component_scales"])
        feasible_a = np.asarray(archive["support_projection_feasible_full_a"])
        feasible_b = np.asarray(archive["support_projection_feasible_full_b"])
        target_a = np.asarray(archive["support_projected_full_a"])
        target_b = np.asarray(archive["support_projected_full_b"])
    if (
        not np.array_equal(starts, np.arange(1, 126))
        or not np.array_equal(ends, starts + 3)
        or not np.allclose(scales, SUPPORT_COMPONENT_SCALES, rtol=0.0, atol=1e-12)
        or feasible_a.shape != (125,)
        or feasible_b.shape != (125,)
        or not feasible_a.all()
        or not feasible_b.all()
        or target_a.shape != (125, 6)
        or target_b.shape != (125, 6)
        or not np.isfinite(target_a).all()
        or not np.isfinite(target_b).all()
    ):
        raise ValueError("support-aware target contract is invalid")
    return {
        "valid": True,
        "path": str(resolved),
        "sha256": SUPPORT_TARGET_SHA256,
        "primary_replica": "full-a",
        "heldout_replica": "full-b",
        "window_start_first": 1,
        "window_start_last": 125,
        "window_count": 125,
        "component_scales": scales.tolist(),
    }


def _finite_tree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(tree)
    )


def _finite_vector(value: object, shape: tuple[int, ...]) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == shape and np.isfinite(array).all())


def validate_training_artifacts(
    run_directory: Path,
    *,
    source_checkpoint: Path,
    target_path: Path,
) -> dict[str, object]:
    """Prove frozen E002 lineage, zero wrench, and live target gradients."""
    hparams = json.loads((run_directory / "hparams.json").read_text(encoding="utf-8"))
    required_hparams = {
        "actor_frozen_controller_residual": True,
        "actor_frozen_controller_residual_depth": 2,
        "actor_support_aware_impulse": True,
        "actor_support_aware_impulse_path": str(target_path.resolve()),
        "actor_support_aware_impulse_sha256": SUPPORT_TARGET_SHA256,
        "actor_support_aware_impulse_window": 4,
        "actor_support_aware_impulse_delta": 0.1,
        "actor_support_aware_impulse_weight": 1.0,
        "actor_centroidal_propulsion": False,
        "actor_capture_point_tracking": False,
        "actor_counterfactual_wrench_distillation": False,
        "torso_wrench_assistance": False,
        "actor_learned_torso_wrench": False,
        "domain_randomization": False,
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "gradient_accumulation_steps": 2,
        "unroll_length": 24,
        "tracking_anchor_position_kernel": "exponential",
        "tracking_root_velocity_weight": 1.0,
        "total_steps": END_STEP,
        "checkpoint_steps": list(expected_checkpoint_steps()),
    }
    if any(hparams.get(key) != value for key, value in required_hparams.items()):
        raise ValueError("support-aware hparams do not match the treatment")
    report = hparams.get("actor_support_aware_impulse_target_report")
    if (
        not isinstance(report, dict)
        or report.get("valid") is not True
        or report.get("artifact_sha256") != SUPPORT_TARGET_SHA256
        or report.get("primary_replica") != "full-a"
        or report.get("heldout_replica") != "full-b"
        or report.get("window_count") != 125
    ):
        raise ValueError("support-aware target load report is invalid")

    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    if frozen_controller_residual_depth(source.actor_params) != 1:
        raise ValueError("source checkpoint is not exact depth-one E002")
    source_actor_hash = parameter_tree_sha256(source.actor_params)
    source_opt_hash = parameter_tree_sha256(source.actor_opt)
    source_normalizer_hash = parameter_tree_sha256(source.normalizer)
    checkpoint = run_directory / f"checkpoint_step_{END_STEP}.pkl"
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != END_STEP
        or not isinstance(state.actor_params, FrozenControllerResidualParams)
        or not isinstance(state.actor_opt, FrozenControllerResidualOptState)
        or frozen_controller_residual_depth(state.actor_params) != 2
        or not _finite_tree(state)
        or parameter_tree_sha256(state.actor_params.parent) != source_actor_hash
        or parameter_tree_sha256(state.actor_opt.parent_optimizer_state)
        != source_opt_hash
        or parameter_tree_sha256(state.normalizer) != source_normalizer_hash
    ):
        raise ValueError("support-aware checkpoint violates frozen E002")

    rows = json.loads(
        (run_directory / "checkpoint_phase_metrics.json").read_text(encoding="utf-8")
    )
    if [row.get("step") for row in rows] != [END_STEP]:
        raise ValueError("support-aware telemetry grid is invalid")
    row = rows[0]
    scalar_positive = (
        "actor_preview_gradient_norm",
        "actor_preview_update_norm",
        "actor_support_aware_impulse_valid_window_count",
    )
    scalar_finite = (
        "actor_support_aware_impulse_loss",
        "actor_support_aware_impulse_heldout_loss",
        "actor_support_aware_impulse_p99_forward_abs",
    )
    if (
        row.get("actor_preview_valid") is not True
        or row.get("actor_cagrad_valid") is not True
        or row.get("actor_preview_frozen_parameter_drift_max_abs") != 0.0
        or row.get("actor_preview_frozen_moment_drift_max_abs") != 0.0
        or row.get("actor_preview_normalizer_drift_max_abs") != 0.0
        or any(
            not isinstance(row.get(key), (int, float))
            or isinstance(row.get(key), bool)
            or not math.isfinite(float(row[key]))
            or float(row[key]) <= 0.0
            for key in scalar_positive
        )
        or any(
            not isinstance(row.get(key), (int, float))
            or isinstance(row.get(key), bool)
            or not math.isfinite(float(row[key]))
            or float(row[key]) < 0.0
            for key in scalar_finite
        )
        or not _finite_vector(
            row.get("actor_support_aware_impulse_component_rms"), (6,)
        )
        or not _finite_vector(
            row.get("actor_support_aware_impulse_heldout_component_rms"),
            (6,),
        )
    ):
        raise ValueError("support-aware training telemetry is invalid")
    return {
        "valid": True,
        "protocol": "g1-support-aware-joint-training-v1",
        "checkpoint_step": END_STEP,
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_actor_tree_sha256": source_actor_hash,
        "source_optimizer_tree_sha256": source_opt_hash,
        "source_normalizer_tree_sha256": source_normalizer_hash,
        "primary_training_loss": row["actor_support_aware_impulse_loss"],
        "heldout_training_loss": row["actor_support_aware_impulse_heldout_loss"],
        "valid_training_window_count": row[
            "actor_support_aware_impulse_valid_window_count"
        ],
    }


def support_target_metrics(
    evaluation_npz: Path,
    *,
    target_path: Path,
) -> dict[str, object]:
    """Measure both target replicas on one deterministic phase-zero rollout."""
    with np.load(evaluation_npz, allow_pickle=False) as archive:
        columns = [str(value) for value in archive["columns"].tolist()]
        values = np.asarray(archive["values"], dtype=np.float64)
        momentum = np.asarray(archive["centroidal_momentum"], dtype=np.float64)
        quaternions = np.asarray(
            archive["centroidal_root_quaternion"], dtype=np.float64
        )
    phase_index = columns.index("phase")
    done_index = columns.index("done")
    phases = np.rint(values[:, phase_index]).astype(np.int32)
    target, _ = load_support_aware_impulse_target(
        target_path,
        expected_sha256=SUPPORT_TARGET_SHA256,
        reference_length=272,
        expected_component_scales=SUPPORT_COMPONENT_SCALES,
    )
    kwargs = {
        "done": jnp.asarray(values[:, done_index] > 0.5),
        "active": jnp.ones((len(values),), dtype=bool),
        "gravity_impulse": jnp.asarray(
            [0.0, 0.0, -SUPPORT_COMPONENT_SCALES[0]], dtype=jnp.float64
        ),
        "component_scales": target.component_scales,
        "window": 4,
        "reference_stride": 1,
        "delta": 0.1,
    }
    primary = support_aware_impulse_objective(
        jnp.asarray(momentum),
        jnp.asarray(quaternions),
        jnp.asarray(phases),
        target.primary_by_phase,
        target.phase_valid,
        **kwargs,
    )
    heldout = support_aware_impulse_objective(
        jnp.asarray(momentum),
        jnp.asarray(quaternions),
        jnp.asarray(phases),
        target.duplicate_by_phase,
        target.phase_valid,
        **kwargs,
    )

    def summarize(result) -> dict[str, object]:
        valid = np.asarray(result.valid)
        normalized = np.asarray(result.normalized_error)
        count = int(np.asarray(result.valid_count))
        component_rms = np.sqrt(
            np.sum(np.where(valid[:, None], np.square(normalized), 0.0), axis=0)
            / max(count, 1)
        )
        return {
            "loss": float(np.asarray(result.loss)),
            "valid_window_count": count,
            "p99_forward_abs": float(np.asarray(result.p99_forward_abs)),
            "component_rms": component_rms.tolist(),
        }

    output = {
        "protocol": "g1-support-aware-evaluation-v1",
        "evaluation_npz": str(evaluation_npz.resolve()),
        "evaluation_npz_sha256": sha256_file(evaluation_npz),
        "primary": summarize(primary),
        "heldout": summarize(heldout),
    }
    if (
        output["primary"]["valid_window_count"] != 125
        or output["heldout"]["valid_window_count"] != 125
    ):
        raise ValueError("support-aware phase-zero evaluation lacks all 125 windows")
    return output


def _plot_survival(selection: dict[str, object], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = selection["phases"]
    source = selection["source_survival"]
    candidate = selection["candidate_survival"]
    x = np.arange(len(phases))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, source, marker="o", label="retained E002")
    axis.plot(x, candidate, marker="o", label="support-aware update 8")
    axis.set_xticks(x, [str(phase) for phase in phases])
    axis.set_xlabel("start phase")
    axis.set_ylabel("steps survived")
    axis.set_title("Support-aware joint-policy early gate")
    axis.legend()
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
    """Run matched behavior and target-space checks on source and candidate."""
    evaluation_root = output_root / "phase_grid"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "MUJOCO_GL": "egl",
    }
    source_phase_grid = evaluation_root / "source_e002.json"
    candidate_phase_grid = evaluation_root / f"checkpoint_step_{END_STEP}.json"
    checkpoint = run_directory / f"checkpoint_step_{END_STEP}.pkl"
    for current_checkpoint, output in (
        (source_checkpoint, source_phase_grid),
        (checkpoint, candidate_phase_grid),
    ):
        subprocess.run(
            _phase_grid_command(
                checkpoint=current_checkpoint,
                reference=reference,
                output=output,
                code_commit=code_commit,
            ),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
        )
    source_payload = json.loads(source_phase_grid.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_phase_grid.read_text(encoding="utf-8"))
    source_survival = source_payload.get("summary", {}).get("survival")
    candidate_survival = candidate_payload.get("summary", {}).get("survival")
    if (
        source_survival != list(E002_SURVIVAL)
        or candidate_payload.get("checkpoint_sha256") != sha256_file(checkpoint)
        or candidate_payload.get("reference_sha256") != REFERENCE_SHA256
        or candidate_payload.get("summary", {}).get("phases") != [0, 25, 50, 75, 100]
    ):
        raise ValueError("support-aware phase-grid evidence is invalid")
    selection = classify_candidate(
        source_survival=source_survival,
        candidate_survival=candidate_survival,
        checkpoint_sha256=sha256_file(checkpoint),
    )

    previews = {}
    for label, current_checkpoint in (
        ("source_e002", source_checkpoint),
        ("candidate", checkpoint),
    ):
        output = output_root / f"{label}_preview"
        subprocess.run(
            _render_command(
                checkpoint=current_checkpoint,
                reference=reference,
                output=output,
            ),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("checkpoint_sha256") != sha256_file(current_checkpoint)
            or summary.get("reference_sha256") != REFERENCE_SHA256
            or not (output / "evaluation.mp4").is_file()
            or not (output / "contact_sheet.png").is_file()
        ):
            raise ValueError("support-aware preview evidence is invalid")
        previews[label] = {
            "target_metrics": support_target_metrics(
                output / "evaluation.npz", target_path=target_path
            ),
            "summary_sha256": sha256_file(output / "summary.json"),
            "evaluation_mp4_sha256": sha256_file(output / "evaluation.mp4"),
            "contact_sheet_sha256": sha256_file(output / "contact_sheet.png"),
        }

    source_primary = previews["source_e002"]["target_metrics"]["primary"]["loss"]
    source_heldout = previews["source_e002"]["target_metrics"]["heldout"]["loss"]
    candidate_primary = previews["candidate"]["target_metrics"]["primary"]["loss"]
    candidate_heldout = previews["candidate"]["target_metrics"]["heldout"]["loss"]
    primary_relative = (source_primary - candidate_primary) / source_primary
    heldout_relative = (source_heldout - candidate_heldout) / source_heldout
    target_reached = (
        primary_relative >= TARGET_LOSS_RELATIVE_IMPROVEMENT
        and heldout_relative >= TARGET_LOSS_RELATIVE_IMPROVEMENT
    )
    behavior_safe = bool(selection["policy_retained"])
    if behavior_safe and target_reached:
        final_outcome = "support-aware-joint-treatment-advances"
        policy_retained = True
    elif target_reached and selection["strictly_improves_any_phase"]:
        final_outcome = "support-target-reachable-but-redistributes"
        policy_retained = False
    elif target_reached:
        final_outcome = "support-target-reachable-no-survival-consolidation"
        policy_retained = False
    elif behavior_safe:
        final_outcome = "behavior-gain-not-support-target-mediated"
        policy_retained = False
    else:
        final_outcome = selection["outcome"]
        policy_retained = False
    selection.update(
        outcome=final_outcome,
        policy_retained=policy_retained,
        target_reached=target_reached,
        target_loss_relative_improvement_floor=(TARGET_LOSS_RELATIVE_IMPROVEMENT),
        primary_target_loss_relative_improvement=primary_relative,
        heldout_target_loss_relative_improvement=heldout_relative,
        source_target_metrics=previews["source_e002"]["target_metrics"],
        candidate_target_metrics=previews["candidate"]["target_metrics"],
        source_preview=previews["source_e002"],
        candidate_preview=previews["candidate"],
        source_phase_grid_sha256=sha256_file(source_phase_grid),
        candidate_phase_grid_sha256=sha256_file(candidate_phase_grid),
    )
    _plot_survival(selection, output_root / "learning_curves.png")
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
        raise ValueError("support-aware treatment seed must equal zero")
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
        protocol="g1-support-aware-joint-preflight-v1",
        start_step=START_STEP,
        end_step=END_STEP,
        updates=UPDATES,
        checkpoint_steps=list(expected_checkpoint_steps()),
        source_residual_depth=1,
        treatment_residual_depth=2,
        exact_zero_external_wrench=True,
        model_friction=1.0,
        support_target=validate_target_artifact(args.support_target),
    )
    _write_json_atomically(output_root / "preflight.json", preflight)
    kwargs = build_support_aware_kwargs(
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
