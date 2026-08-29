"""SHAC training for Go2 locomotion."""

import hashlib
import os
import time
import pickle
import json
import shutil
import math
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import optax
import numpy as np

from src.core.data_structures import (
    Normalizer,
    TrainState,
    add_to_replay_buffer,
    init_replay_buffer,
    sample_replay_buffer,
)
from src.core.actor_input_contract import validate_actor_input_contract
from src.core.networks import Actor, Critic, DoubleCritic, LearnedDynamicsModel
from src.algorithms.jave.gradient_bellman import (
    denormalize_jave_observation,
    gradient_bellman_loss,
    gradient_bellman_targets,
    learned_dynamics_loss,
    normalize_jave_observation,
)
from src.core.rmr_policy import (
    RmrPolicy,
    apply_trainable_rmr_policy,
    compose_bounded_rmr_residual,
)
from src.core.rmr_action_noise import (
    action_noise_std_hparam,
    resolve_action_noise_resume_settings,
)
from src.envs.go2.environment import Go2Env
from src.envs.go2.terrain import differentiated_ou_foot_forces
from src.envs.g1_tracking.demonstration_replay import (
    apply_demonstration_replay,
    resolve_demonstration_replay_resume_setting,
)
from src.core.utils import compute_grad_norm, tree_bit_fingerprint
from src.algorithms.shac.gradients import (
    aggregate_per_env_gradients,
    per_env_gradient_statistics,
)
from src.algorithms.shac.ahac import (
    active_horizon_mask,
    conservative_value,
    critic_convergence,
    critic_value_loss,
    resolve_ahac_resume_settings,
    select_active_tree,
    update_horizon_dual,
)
from src.algorithms.shac.cagrad import (
    accumulate_phase_gradients,
    combine_cagrad,
    finalize_phase_gradients,
)
from src.algorithms.shac.contact_truncation import (
    contact_gradient_barrier,
    contact_topology_event_from_info,
    resolve_contact_topology_truncation_resume_setting,
)
from src.algorithms.shac.phase_weighting import (
    aggregate_phase_weighted_gradients,
    phase_bin_indices,
    phase_robust_weights,
)
from src.algorithms.shac.microbatch import (
    flatten_population,
    mean_shard_trees,
    reshape_population,
    summarize_shard_stats,
)
from src.algorithms.shac.torso_wrench_curriculum import (
    assistance_scale_at_step,
    resolve_assistance_conditioning_resume_settings,
    resolve_torso_wrench_assistance_resume_settings,
    sample_assistance_scales,
    torso_wrench_assistance_diagnostics,
    validate_assistance_conditioning_configuration,
    validate_torso_wrench_assistance_configuration,
)
from src.algorithms.shac.learned_torso_wrench import (
    FrozenControllerWrenchParams,
    LearnedTorsoWrenchHead,
    apply_learned_torso_wrench,
    build_learned_wrench_anneal_mask,
    build_learned_wrench_mask,
    learned_wrench_scale_at_step,
    migrate_learned_wrench_scale_conditioning,
    normalized_yaw_wrench_to_world,
)
from src.algorithms.shac.initialization import (
    canonicalize_normalizer_dtype,
    canonicalize_step_dtype,
    canonicalize_tree_like,
    commit_tree_to_local_device,
)
from src.algorithms.shac.future_reference_migration import (
    future_reference_migration_report,
    migrate_future_reference_train_state,
    validate_future_reference_migration_report,
)
from src.algorithms.shac.centroidal_objective import (
    centroidal_window_objective,
)
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualOptState,
    FrozenControllerResidualParams,
    apply_frozen_controller_residual,
    frozen_controller_residual_depth,
    migrate_frozen_controller_residual,
    update_frozen_controller_residual,
)
from src.algorithms.shac.counterfactual_wrench_distillation import (
    counterfactual_target_change,
    counterfactual_transition_loss,
    load_counterfactual_feasibility,
    parameter_tree_sha256,
    resolve_counterfactual_wrench_resume_setting,
    resolve_leg_action_indices,
)
from src.algorithms.shac.preview_adapter import (
    apply_preview_adapter_update,
    build_current_preview_mask,
    build_rmr_preview_mask,
    migrate_rmr_preview_policy,
    phase_binned_action_deviation,
    rmr_preview_migration_report,
    zero_current_preview,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualMuonState,
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_residual_muon_update,
    apply_frozen_preview_residual,
    build_residual_adapter_mask,
    build_residual_muon_optimizers,
    initialize_residual_adapter_optimizer,
    initialize_residual_muon_optimizer,
    migrate_residual_adapter_assistance_conditioning,
    residual_adapter_migration_report,
    residual_muon_migration_report,
    resolve_zero_head_feature_transfer_resume_setting,
    split_residual_adapter_params,
    transplant_zero_head_recovery_features,
    current_treatment_frame,
)
from src.algorithms.shac.recovery_teacher import (
    load_recovery_teacher_batch,
    mix_recovery_teacher_actor_gradient,
    resolve_recovery_teacher_resume_settings,
)
from src.algorithms.shac.progressive_recovery_expert import (
    RecoverySupport,
    apply_state_gated_recovery,
)
from src.algorithms.shac.resume_randomness import (
    apply_resume_randomness_setting,
    persist_resume_randomness_audit,
)
from src.evaluation.g1_torso_wrench_oracle import (
    compute_environment_torso_wrench,
    torso_wrench_parameters_from_environment,
    write_torso_wrench,
)
from src.envs.g1_tracking.training_distribution import (
    PhaseSamplerState,
    init_phase_sampler,
    phase_sampling_probabilities,
    update_phase_sampler,
)
from src.envs.g1_tracking.centroidal_momentum import (
    mjx_capture_point,
    mjx_centroidal_momentum,
)
from src.algorithms.shac.capture_point_objective import (
    capture_point_objective,
    capture_state_validity,
)


# Set to True to enable per-foot normal force logging.
DEBUG_FOOT_CONTACTS = False

DETERMINISM_BOUNDARIES = (
    "random_inputs",
    "first_actor_action",
    "first_mjx_substep",
    "first_mjx_control_step",
    "first_env_step",
    "rollout",
    "actor_cagrad",
    "learned_dynamics",
    "critic",
)

FIRST_MJX_SUBSTEP_COMPONENTS = (
    "integrated_state",
    "acceleration_state",
    "constraint_force",
    "contact_state",
)

FIRST_MJX_SUBSTEP_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "qfrc_applied",
    "qfrc_passive",
    "position_input_qpos",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "xanchor",
    "xaxis",
    "subtree_com",
    "subtree_local_position",
    "subtree_local_mass",
    "subtree_scan_position",
    "subtree_scan_mass",
    "subtree_divided",
    "subtree_selected",
    "rne_input_qvel",
    "cdof",
    "cdof_dot",
    "cvel",
    "cinert",
    "qfrc_bias",
    "qfrc_actuator",
    "actuator_force",
    "xfrc_applied",
    "xfrc_accumulated",
    "qfrc_smooth",
    "qfrc_constraint",
    "efc_force",
    "contact",
)


def numeric_tree_sha256(tree: object) -> str:
    """Hash stable leaf paths, dtypes, shapes, and exact numeric bytes."""

    digest = hashlib.sha256()
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    digest.update(str(len(paths_and_leaves)).encode("ascii"))
    for path, value in paths_and_leaves:
        digest.update(repr(path).encode("utf-8"))
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_determinism_probe(compiled_step, state) -> dict[str, object]:
    """Invoke one compiled update twice from one state and compare exactly."""

    first_state, first_metrics = compiled_step(state)
    jax.block_until_ready((first_state, first_metrics))
    second_state, second_metrics = compiled_step(state)
    jax.block_until_ready((second_state, second_metrics))

    boundaries: dict[str, dict[str, object]] = {}
    first_mismatch = None
    for name in DETERMINISM_BOUNDARIES:
        key = f"determinism_{name}_fingerprint"
        first = np.asarray(first_metrics[key], dtype=np.uint32)
        second = np.asarray(second_metrics[key], dtype=np.uint32)
        exact = bool(np.array_equal(first, second))
        boundaries[name] = {
            "first": first.tolist(),
            "second": second.tolist(),
            "exact": exact,
        }
        if not exact and first_mismatch is None:
            first_mismatch = name

    first_mjx_substep_components: dict[str, dict[str, object]] = {}
    mismatching_first_mjx_substep_components = []
    for name in FIRST_MJX_SUBSTEP_COMPONENTS:
        key = f"determinism_first_mjx_substep_{name}_fingerprint"
        first = np.asarray(first_metrics[key], dtype=np.uint32)
        second = np.asarray(second_metrics[key], dtype=np.uint32)
        exact = bool(np.array_equal(first, second))
        first_mjx_substep_components[name] = {
            "first": first.tolist(),
            "second": second.tolist(),
            "exact": exact,
        }
        if not exact:
            mismatching_first_mjx_substep_components.append(name)

    first_mjx_substep_fields: dict[str, dict[str, object]] = {}
    mismatching_first_mjx_substep_fields = []
    for name in FIRST_MJX_SUBSTEP_FIELDS:
        key = f"determinism_first_mjx_substep_field_{name}_fingerprint"
        first = np.asarray(first_metrics[key], dtype=np.uint32)
        second = np.asarray(second_metrics[key], dtype=np.uint32)
        exact = bool(np.array_equal(first, second))
        first_mjx_substep_fields[name] = {
            "first": first.tolist(),
            "second": second.tolist(),
            "exact": exact,
        }
        if not exact:
            mismatching_first_mjx_substep_fields.append(name)

    production_key = (
        "determinism_first_mjx_substep_field_subtree_com_fingerprint"
    )
    reconstructed_key = (
        "determinism_first_mjx_substep_field_subtree_selected_fingerprint"
    )
    first_subtree_consistent = bool(
        np.array_equal(
            np.asarray(first_metrics[production_key], dtype=np.uint32),
            np.asarray(first_metrics[reconstructed_key], dtype=np.uint32),
        )
    )
    second_subtree_consistent = bool(
        np.array_equal(
            np.asarray(second_metrics[production_key], dtype=np.uint32),
            np.asarray(second_metrics[reconstructed_key], dtype=np.uint32),
        )
    )
    subtree_com_probe_consistency = {
        "first": first_subtree_consistent,
        "second": second_subtree_consistent,
        "valid": first_subtree_consistent and second_subtree_consistent,
    }

    first_state_sha256 = numeric_tree_sha256(first_state)
    second_state_sha256 = numeric_tree_sha256(second_state)
    first_metrics_sha256 = numeric_tree_sha256(first_metrics)
    second_metrics_sha256 = numeric_tree_sha256(second_metrics)
    full_state_exact = first_state_sha256 == second_state_sha256
    metrics_exact = first_metrics_sha256 == second_metrics_sha256
    valid = (
        all(item["exact"] for item in boundaries.values())
        and all(
            item["exact"]
            for item in first_mjx_substep_components.values()
        )
        and all(
            item["exact"] for item in first_mjx_substep_fields.values()
        )
        and subtree_com_probe_consistency["valid"]
        and full_state_exact
        and metrics_exact
    )
    return {
        "protocol": "shac-compiled-update-determinism-v9",
        "valid": valid,
        "boundaries": boundaries,
        "first_mismatch_boundary": first_mismatch,
        "first_mjx_substep_components": first_mjx_substep_components,
        "mismatching_first_mjx_substep_components": (
            mismatching_first_mjx_substep_components
        ),
        "first_mjx_substep_fields": first_mjx_substep_fields,
        "mismatching_first_mjx_substep_fields": (
            mismatching_first_mjx_substep_fields
        ),
        "subtree_com_probe_consistency": subtree_com_probe_consistency,
        "full_state_exact": full_state_exact,
        "metrics_exact": metrics_exact,
        "first_state_sha256": first_state_sha256,
        "second_state_sha256": second_state_sha256,
        "first_metrics_sha256": first_metrics_sha256,
        "second_metrics_sha256": second_metrics_sha256,
    }


def load_checkpoint(path: str):
    """
    Load a training checkpoint.

    Args:
        path: Path to a .pkl file, or a training folder containing one.
              When given a folder, searches in order:
              checkpoint_latest.pkl, policy_best.pkl, policy_final.pkl

    Returns:
        Tuple of (state, hparams, step) where:
            - state: TrainState object
            - hparams: dict of hyperparameters (or None)
            - step: training step count
    """
    if os.path.isdir(path):
        for name in ["checkpoint_latest.pkl", "policy_best.pkl", "policy_final.pkl"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {path}")

    print(f"Loading checkpoint from {path}")
    with open(path, "rb") as f:
        state = pickle.load(f)

    # Try to load hyperparameters from same directory
    hparams = None
    hparams_path = os.path.join(os.path.dirname(path), "hparams.json")
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)

    return state, hparams, int(state.step)


def save_periodic_checkpoint(state, save_dir: str | Path, step: int) -> Path:
    """Archive a step-addressed checkpoint and atomically advance latest."""
    directory = Path(save_dir)
    step_path = directory / f"checkpoint_step_{step:06d}.pkl"
    step_temp = directory / f".{step_path.name}.tmp"
    with step_temp.open("wb") as stream:
        pickle.dump(state, stream)
    os.replace(step_temp, step_path)

    latest_path = directory / "checkpoint_latest.pkl"
    latest_temp = directory / ".checkpoint_latest.pkl.tmp"
    shutil.copyfile(step_path, latest_temp)
    os.replace(latest_temp, latest_path)
    return step_path


def persist_run_hparams(
    save_dir: str | Path, hparams: dict[str, object]
) -> Path:
    """Atomically persist resume metadata before a checkpoint references it."""
    path = Path(save_dir) / "hparams.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w") as stream:
        json.dump(hparams, stream, indent=2)
    os.replace(temp_path, path)
    return path


def persist_determinism_probe_report(
    output_path: str | Path, report: dict[str, object]
) -> Path:
    """Atomically publish one create-only compiled-update probe report."""

    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"determinism probe output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_future_reference_migration_report(
    save_dir: str | Path, report: dict[str, object]
) -> Path:
    """Validate and atomically publish append-only migration evidence."""
    validate_future_reference_migration_report(report)
    path = Path(save_dir) / "migration_equivalence.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_residual_adapter_migration_report(
    save_dir: str | Path, report: dict[str, object]
) -> Path:
    """Validate and atomically publish zero-effect adapter migration evidence."""
    if report.get("valid") is not True:
        raise ValueError("residual adapter migration equivalence failed")
    path = Path(save_dir) / "residual_adapter_migration.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_zero_head_feature_transfer_report(
    save_dir: str | Path, report: dict[str, object]
) -> Path:
    """Atomically publish the hash-bound zero-head feature transplant audit."""
    if report.get("valid") is not True:
        raise ValueError("zero-head recovery feature transfer failed")
    path = Path(save_dir) / "zero_head_feature_transfer.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_residual_muon_migration_report(
    save_dir: str | Path, report: dict[str, object]
) -> Path:
    """Validate and atomically publish Muon optimizer migration evidence."""
    if report.get("valid") is not True:
        raise ValueError("residual Muon optimizer migration failed")
    path = Path(save_dir) / "residual_muon_migration.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_assistance_conditioning_migration_report(
    save_dir: str | Path, report: dict[str, object]
) -> Path:
    """Atomically publish exact-zero scalar-conditioning migration evidence."""
    if report.get("valid") is not True:
        raise ValueError("assistance conditioning migration failed")
    path = Path(save_dir) / "assistance_conditioning_migration.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_reference_path_migration_report(
    save_dir: str | Path, report: dict[str, object]
) -> Path:
    """Atomically publish an explicitly authorized reference migration."""
    if report.get("valid") is not True:
        raise ValueError("reference path migration failed")
    path = Path(save_dir) / "reference_path_migration.json"
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def persist_checkpoint_phase_metric(
    save_dir: str | Path, row: dict[str, object]
) -> Path:
    """Atomically upsert one checkpoint-aligned phase diagnostic row."""
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint phase metric requires a nonnegative step")
    path = Path(save_dir) / "checkpoint_phase_metrics.json"
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, list):
            raise ValueError("checkpoint phase metrics must contain a JSON list")
        rows = loaded
    by_step = {int(existing["step"]): existing for existing in rows}
    by_step[step] = row
    ordered = [by_step[key] for key in sorted(by_step)]
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(ordered, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)
    return path


def build_checkpoint_cagrad_telemetry(metrics) -> dict[str, object]:
    """Serialize the complete checkpoint-aligned CAGrad validity contract."""
    return {
        "actor_bootstrap_scale_current": float(
            metrics["actor_bootstrap_scale_current"]
        ),
        "actor_cagrad_bin_counts": np.asarray(
            metrics["actor_cagrad_bin_counts"]
        ).tolist(),
        "actor_cagrad_bin_gradient_norms": np.asarray(
            metrics["actor_cagrad_bin_gradient_norms"]
        ).tolist(),
        "actor_cagrad_bin_losses": np.asarray(
            metrics["actor_cagrad_bin_losses"]
        ).tolist(),
        "actor_cagrad_weights": np.asarray(
            metrics["actor_cagrad_weights"]
        ).tolist(),
        "actor_cagrad_gram_matrix": np.asarray(
            metrics["actor_cagrad_gram_matrix"]
        ).tolist(),
        "actor_cagrad_cosine_matrix": np.asarray(
            metrics["actor_cagrad_cosine_matrix"]
        ).tolist(),
        "actor_cagrad_objective": float(metrics["actor_cagrad_objective"]),
        "actor_cagrad_dual_gap": float(metrics["actor_cagrad_dual_gap"]),
        "actor_cagrad_uniform_combined_cosine": float(
            metrics["actor_cagrad_uniform_combined_cosine"]
        ),
        "actor_cagrad_combined_norm": float(
            metrics["actor_cagrad_combined_norm"]
        ),
        "actor_cagrad_valid": bool(metrics["actor_cagrad_valid"]),
    }


def build_checkpoint_demonstration_replay_telemetry(
    metrics, *, threshold: float
) -> dict[str, object]:
    """Serialize checkpoint-aligned demonstration-replay evidence."""
    count = int(metrics["demonstration_replay_count"])
    fraction = float(metrics["demonstration_replay_fraction"])
    if count < 0 or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("demonstration replay telemetry is invalid")
    return {
        "demonstration_replay_threshold": float(threshold),
        "demonstration_replay_count": count,
        "demonstration_replay_fraction": fraction,
        "demonstration_replay_valid": True,
    }


def build_checkpoint_ahac_telemetry(metrics) -> dict[str, object]:
    """Serialize and fail closed on checkpoint-aligned AHAC evidence."""

    float_names = (
        "horizon",
        "horizon_before_update",
        "dual_mean",
        "dual_max",
        "contact_stiffness_mean",
        "contact_stiffness_max",
        "contact_threshold",
        "critic_head_disagreement",
    )
    values = {
        name: float(metrics[f"ahac_{name}"])
        for name in float_names
    }
    loss_history = np.asarray(
        metrics["ahac_critic_loss_history"], dtype=np.float64
    )
    head_losses = np.asarray(
        metrics["ahac_critic_head_losses"], dtype=np.float64
    )
    active_transitions = int(metrics["ahac_active_transitions"])
    critic_iterations_executed = int(metrics["ahac_critic_iterations"])
    if (
        not all(math.isfinite(value) for value in values.values())
        or loss_history.shape != (5,)
        or not np.all(np.isfinite(loss_history))
        or head_losses.shape != (2,)
        or not np.all(np.isfinite(head_losses))
    ):
        raise ValueError("AHAC telemetry must be finite and complete")
    if active_transitions < 1 or critic_iterations_executed < 1:
        raise ValueError("AHAC telemetry counts must be positive")
    valid = bool(metrics["ahac_horizon_valid"])
    if not valid:
        raise ValueError("AHAC horizon/dual telemetry is invalid")
    return {
        **{f"ahac_{name}": value for name, value in values.items()},
        "ahac_active_transitions": active_transitions,
        "ahac_critic_iterations": critic_iterations_executed,
        "ahac_critic_converged": bool(metrics["ahac_critic_converged"]),
        "ahac_critic_loss_history": loss_history.tolist(),
        "ahac_critic_head_losses": head_losses.tolist(),
        "ahac_valid": True,
    }


def build_checkpoint_recovery_teacher_telemetry(
    metrics, *, max_ratio: float
) -> dict[str, object]:
    """Serialize and fail closed on the recovery-teacher gradient contract."""
    names = (
        "loss",
        "raw_gradient_norm",
        "projected_gradient_norm",
        "applied_gradient_norm",
        "physics_gradient_norm",
        "combined_gradient_norm",
        "physics_dot",
        "physics_cosine",
        "applied_scale",
        "parent_gradient_max_abs",
    )
    values = {
        name: float(metrics[f"actor_recovery_teacher_{name}"])
        for name in names
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("recovery teacher telemetry must be finite")
    if values["parent_gradient_max_abs"] != 0.0:
        raise ValueError("recovery teacher changed the frozen parent gradient")
    if values["applied_gradient_norm"] > (
        max_ratio * values["physics_gradient_norm"] + 1e-7
    ):
        raise ValueError("recovery teacher gradient exceeded its norm cap")
    if not bool(metrics["actor_recovery_teacher_valid"]):
        raise ValueError("recovery teacher telemetry is invalid")
    return {
        **{
            f"actor_recovery_teacher_{name}": value
            for name, value in values.items()
        },
        "actor_recovery_teacher_valid": True,
    }


def build_counterfactual_wrench_telemetry(metrics) -> dict[str, object]:
    """Serialize and fail closed on the leg-only teacher objective."""
    names = (
        "loss",
        "base_linear_loss",
        "base_angular_loss",
        "centroidal_linear_loss",
        "centroidal_angular_loss",
        "cosine",
        "student_rms",
        "teacher_rms",
        "normalized_error_rms",
        "residual_rms",
        "residual_max_abs",
        "residual_bound_fraction",
        "nonleg_max_abs",
        "student_wrench_max_abs",
        "teacher_wrench_rms",
    )
    values = {
        name: float(metrics[f"actor_counterfactual_{name}"])
        for name in names
    }
    valid_count = int(metrics["actor_counterfactual_valid_count"])
    invalid_count = int(metrics["actor_counterfactual_invalid_count"])
    done_mismatch_count = int(
        metrics["actor_counterfactual_done_mismatch_count"]
    )
    if (
        not all(math.isfinite(value) for value in values.values())
        or valid_count < 1
        or invalid_count != 0
        or done_mismatch_count < 0
        or values["residual_rms"] < 0.0
        or values["residual_max_abs"] < 0.0
        or not 0.0 <= values["residual_bound_fraction"] <= 1.0
        or values["nonleg_max_abs"] != 0.0
        or values["student_wrench_max_abs"] != 0.0
        or not bool(metrics["actor_counterfactual_valid"])
    ):
        raise ValueError("counterfactual wrench telemetry is invalid")
    return {
        **{
            f"actor_counterfactual_{name}": value
            for name, value in values.items()
        },
        "actor_counterfactual_valid_count": valid_count,
        "actor_counterfactual_invalid_count": invalid_count,
        "actor_counterfactual_done_mismatch_count": done_mismatch_count,
        "actor_counterfactual_valid": True,
    }


def build_policy_anchor_telemetry(
    metrics, *, weight: float
) -> dict[str, object]:
    """Serialize and validate checkpoint-aligned proximal-policy evidence."""
    error = float(metrics["actor_policy_anchor_squared_error"])
    if not math.isfinite(error) or error < 0.0:
        raise ValueError("actor policy anchor error must be finite")
    return {
        "actor_policy_anchor_weight": weight,
        "actor_policy_anchor_squared_error": error,
        "actor_policy_anchor_valid": True,
    }


def should_persist_checkpoint_metrics(
    checkpoint_path: Path | None,
    *,
    actor_cagrad: bool,
    frozen_preview_treatment: bool,
    ahac: bool = False,
    actor_contact_topology_gradient_truncation: bool = False,
    demonstration_replay: bool = False,
    jave: bool = False,
) -> bool:
    """Persist validity evidence for every checkpointed CAGrad treatment."""
    return checkpoint_path is not None and (
        actor_cagrad
        or frozen_preview_treatment
        or ahac
        or actor_contact_topology_gradient_truncation
        or demonstration_replay
        or jave
    )


def _has_adaptive_phase_state(state) -> bool:
    env_state = getattr(state, "env_state", None)
    info = getattr(env_state, "info", {})
    return "phase_sampler_failed_count" in info


def validate_adaptive_phase_resume_metadata(
    resumed_state,
    resumed_hparams: dict[str, object] | None,
    *,
    requested_adaptive_phase_sampling: bool,
) -> None:
    """Reject adaptive state that cannot restore its sampling treatment."""
    if not _has_adaptive_phase_state(resumed_state):
        if requested_adaptive_phase_sampling:
            return
        return
    required = {
        "adaptive_phase_sampling",
        "adaptive_phase_uniform_ratio",
        "adaptive_phase_alpha",
    }
    if (
        resumed_hparams is None
        or resumed_hparams.get("adaptive_phase_sampling") is not True
        or not required.issubset(resumed_hparams)
    ):
        raise ValueError(
            "adaptive checkpoint requires complete resume metadata with "
            "adaptive_phase_sampling=true"
        )
    uniform_ratio = resumed_hparams["adaptive_phase_uniform_ratio"]
    alpha = resumed_hparams["adaptive_phase_alpha"]
    if (
        isinstance(uniform_ratio, bool)
        or not isinstance(uniform_ratio, (int, float))
        or not math.isfinite(uniform_ratio)
        or not 0.0 <= uniform_ratio <= 1.0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0.0 < alpha <= 1.0
    ):
        raise ValueError(
            "adaptive checkpoint resume metadata contains invalid sampling settings"
        )


def resolve_adaptive_phase_resume_settings(
    resumed_state,
    resumed_hparams: dict[str, object] | None,
    *,
    requested_adaptive_phase_sampling: bool,
    requested_uniform_ratio: float,
    requested_alpha: float,
) -> tuple[bool, float, float]:
    """Resolve exact adaptive resumes while allowing legacy treatment starts."""
    validate_adaptive_phase_resume_metadata(
        resumed_state,
        resumed_hparams,
        requested_adaptive_phase_sampling=requested_adaptive_phase_sampling,
    )
    if _has_adaptive_phase_state(resumed_state):
        return (
            True,
            float(resumed_hparams["adaptive_phase_uniform_ratio"]),
            float(resumed_hparams["adaptive_phase_alpha"]),
        )
    if requested_adaptive_phase_sampling or resumed_hparams is None:
        return (
            requested_adaptive_phase_sampling,
            requested_uniform_ratio,
            requested_alpha,
        )
    return (
        bool(resumed_hparams.get("adaptive_phase_sampling", False)),
        float(
            resumed_hparams.get(
                "adaptive_phase_uniform_ratio", requested_uniform_ratio
            )
        ),
        float(resumed_hparams.get("adaptive_phase_alpha", requested_alpha)),
    )


def archive_periodic_checkpoint_if_due(
    state,
    save_dir: str | Path,
    last_checkpoint_step: int,
    checkpoint_interval: int,
    *,
    current_step: int | None = None,
    hparams: dict[str, object] | None = None,
) -> tuple[int, Path | None]:
    """Archive a due checkpoint without coupling persistence to log cadence."""
    step = int(state.step) if current_step is None else current_step
    if step - last_checkpoint_step < checkpoint_interval:
        return last_checkpoint_step, None
    if _has_adaptive_phase_state(state):
        validate_adaptive_phase_resume_metadata(
            state,
            hparams,
            requested_adaptive_phase_sampling=True,
        )
    if hparams is not None:
        persist_run_hparams(save_dir, hparams)
    checkpoint_path = save_periodic_checkpoint(state, save_dir, step)
    return step, checkpoint_path


def resolve_action_noise_schedule_steps(
    *,
    total_steps: int,
    resumed_step: int,
    resumed_hparams: dict | None,
    requested_schedule_steps: int | None = None,
    allow_resume_action_noise_change: bool = False,
) -> int:
    """Keep a resumed run on its schedule unless an authorized endpoint is set."""
    if total_steps < resumed_step:
        raise ValueError(
            "total_steps must be at least the resumed checkpoint step"
        )
    if requested_schedule_steps is not None and (
        isinstance(requested_schedule_steps, bool)
        or not isinstance(requested_schedule_steps, int)
        or requested_schedule_steps <= 0
    ):
        raise ValueError("action-noise schedule steps must be a positive integer")
    if not isinstance(allow_resume_action_noise_change, bool):
        raise ValueError("allow_resume_action_noise_change must be boolean")
    if resumed_hparams is None:
        return (
            total_steps
            if requested_schedule_steps is None
            else requested_schedule_steps
        )
    schedule_steps = resumed_hparams.get(
        "action_noise_schedule_steps",
        resumed_hparams.get("total_steps", total_steps),
    )
    if (
        isinstance(schedule_steps, bool)
        or not isinstance(schedule_steps, int)
        or schedule_steps <= 0
    ):
        raise ValueError("action-noise schedule steps must be positive integer")
    if requested_schedule_steps is not None and (
        requested_schedule_steps != schedule_steps
        and not allow_resume_action_noise_change
    ):
        raise ValueError(
            "action-noise schedule steps must match the checkpoint unless "
            "allow_resume_action_noise_change is enabled"
        )
    if requested_schedule_steps is not None:
        return requested_schedule_steps
    return schedule_steps


def resolve_future_reference_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_steps: tuple[int, ...],
    allow_upgrade: bool,
) -> tuple[tuple[int, ...], bool]:
    """Resolve exact preview resumes or one authorized legacy upgrade."""
    if not isinstance(requested_steps, tuple):
        raise ValueError("requested future reference steps must be a tuple")
    if not isinstance(allow_upgrade, bool):
        raise ValueError("future reference upgrade authority must be boolean")
    if resumed_hparams is None:
        return requested_steps, False
    saved_steps = tuple(
        resumed_hparams.get("actor_reference_lookahead_steps", ())
    )
    if saved_steps == requested_steps:
        return requested_steps, False
    if not saved_steps and requested_steps:
        if not allow_upgrade:
            raise ValueError(
                "future reference resume requires explicit upgrade authority"
            )
        return requested_steps, True
    raise ValueError(
        "future reference lookahead steps must match the checkpoint"
    )


def resolve_future_reference_preview_mode(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_mode: str,
    future_reference_upgrade: bool,
) -> str:
    """Resolve a preview representation without silently changing inputs."""
    valid_modes = {"absolute", "delta"}
    if requested_mode not in valid_modes:
        raise ValueError("future reference preview mode is invalid")
    if resumed_hparams is None:
        return requested_mode
    saved_steps = tuple(
        resumed_hparams.get("actor_reference_lookahead_steps", ())
    )
    if not saved_steps:
        if requested_mode == "delta" and not future_reference_upgrade:
            raise ValueError(
                "delta preview requires explicit upgrade authority"
            )
        return requested_mode
    saved_mode = resumed_hparams.get(
        "actor_reference_preview_mode", "absolute"
    )
    if saved_mode not in valid_modes:
        raise ValueError("saved future reference preview mode is invalid")
    if saved_mode != requested_mode:
        raise ValueError(
            "future reference preview mode must match the checkpoint"
        )
    return saved_mode


def resolve_preview_adapter_resume_setting(
    resumed_hparams: dict[str, object] | None, *, requested: bool
) -> bool:
    """Allow an explicit legacy treatment start and exact treated resumes."""
    if not isinstance(requested, bool):
        raise ValueError("actor_preview_adapter must be boolean")
    if not resumed_hparams or "actor_preview_adapter" not in resumed_hparams:
        return requested
    saved = resumed_hparams["actor_preview_adapter"]
    if not isinstance(saved, bool):
        raise ValueError("actor_preview_adapter resume metadata must be boolean")
    if saved != requested:
        raise ValueError("actor_preview_adapter must match the checkpoint")
    return saved


def resolve_actor_observe_motion_anchor_position_resume_setting(
    resumed_hparams: dict[str, object] | None, *, requested: bool
) -> bool:
    """Keep the motion-anchor observation boundary identical on resume."""
    if not isinstance(requested, bool):
        raise ValueError("actor_observe_motion_anchor_position must be boolean")
    saved = (
        False
        if resumed_hparams is None
        else resumed_hparams.get("actor_observe_motion_anchor_position", False)
    )
    if not isinstance(saved, bool):
        raise ValueError(
            "checkpoint actor_observe_motion_anchor_position must be boolean"
        )
    if requested != saved:
        raise ValueError(
            "actor_observe_motion_anchor_position must match the checkpoint"
        )
    return saved


def resolve_tracking_velocity_kernel_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: str,
    allow_change: bool,
    is_resume: bool,
) -> str:
    """Restore the exact velocity objective unless change is authorized."""
    valid = {"exponential", "pseudo_huber"}
    if requested not in valid:
        raise ValueError("requested tracking velocity kernel is invalid")
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_tracking_velocity_kernel_change must be boolean"
        )
    if not isinstance(is_resume, bool):
        raise ValueError("is_resume must be boolean")
    if not is_resume:
        return requested
    if resumed_hparams is None:
        raise ValueError("resume hparams are required for the velocity kernel")
    saved = resumed_hparams.get("tracking_velocity_kernel", "exponential")
    if saved not in valid:
        raise ValueError("checkpoint tracking velocity kernel is invalid")
    if saved != requested and not allow_change:
        raise ValueError(
            "tracking velocity kernel must match the checkpoint unless "
            "allow_resume_tracking_velocity_kernel_change is enabled"
        )
    return requested


def resolve_anchor_position_kernel_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: str,
    allow_change: bool,
    is_resume: bool,
) -> str:
    """Restore the exact anchor-position objective unless authorized."""
    valid = {"exponential", "dual_scale", "quadratic"}
    if requested not in valid:
        raise ValueError("requested anchor position kernel is invalid")
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_tracking_anchor_position_kernel_change must be boolean"
        )
    if not isinstance(is_resume, bool):
        raise ValueError("is_resume must be boolean")
    if not is_resume:
        return requested
    if resumed_hparams is None:
        raise ValueError(
            "resume hparams are required for the anchor position kernel"
        )
    saved = resumed_hparams.get(
        "tracking_anchor_position_kernel", "exponential"
    )
    if saved not in valid:
        raise ValueError("checkpoint anchor position kernel is invalid")
    if saved != requested and not allow_change:
        raise ValueError(
            "anchor position kernel must match the checkpoint unless "
            "allow_resume_tracking_anchor_position_kernel_change is enabled"
        )
    return requested


def resolve_tracking_torso_orientation_resume_weight(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: float,
    allow_change: bool,
    is_resume: bool,
) -> float:
    """Restore the torso objective exactly unless a change is authorized."""
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not math.isfinite(requested)
        or requested < 0.0
    ):
        raise ValueError(
            "tracking_torso_orientation_weight must be non-negative and finite"
        )
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_tracking_torso_orientation_change must be boolean"
        )
    if not isinstance(is_resume, bool):
        raise ValueError("is_resume must be boolean")
    if not is_resume:
        return float(requested)
    if resumed_hparams is None:
        raise ValueError(
            "resume hparams are required for the torso orientation objective"
        )
    saved = resumed_hparams.get("tracking_torso_orientation_weight", 0.0)
    if (
        isinstance(saved, bool)
        or not isinstance(saved, (int, float))
        or not math.isfinite(saved)
        or saved < 0.0
    ):
        raise ValueError(
            "checkpoint tracking_torso_orientation_weight is invalid"
        )
    if float(saved) != float(requested) and not allow_change:
        raise ValueError(
            "tracking_torso_orientation_weight must match the checkpoint unless "
            "allow_resume_tracking_torso_orientation_change is enabled"
        )
    return float(requested)


def resolve_tracking_root_velocity_resume_weight(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: float,
    allow_change: bool,
    is_resume: bool,
) -> float:
    """Restore the explicit root-velocity objective unless authorized."""
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not math.isfinite(requested)
        or requested < 0.0
    ):
        raise ValueError(
            "tracking_root_velocity_weight must be non-negative and finite"
        )
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_tracking_root_velocity_change must be boolean"
        )
    if not isinstance(is_resume, bool):
        raise ValueError("is_resume must be boolean")
    if not is_resume:
        return float(requested)
    if resumed_hparams is None:
        raise ValueError(
            "resume hparams are required for the root velocity objective"
        )
    saved = resumed_hparams.get("tracking_root_velocity_weight", 0.0)
    if (
        isinstance(saved, bool)
        or not isinstance(saved, (int, float))
        or not math.isfinite(saved)
        or saved < 0.0
    ):
        raise ValueError("checkpoint tracking_root_velocity_weight is invalid")
    if float(saved) != float(requested) and not allow_change:
        raise ValueError(
            "tracking_root_velocity_weight must match the checkpoint unless "
            "allow_resume_tracking_root_velocity_change is enabled"
        )
    return float(requested)


def resolve_residual_preview_adapter_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: bool,
    requested_hidden: int,
    requested_optimizer: str,
    future_reference_upgrade: bool,
    allow_start: bool = False,
) -> tuple[bool, int, str]:
    """Allow one explicit legacy start and exact nonlinear treated resumes."""
    if not isinstance(requested, bool):
        raise ValueError("actor_residual_preview_adapter must be boolean")
    if (
        isinstance(requested_hidden, bool)
        or not isinstance(requested_hidden, int)
        or requested_hidden < 1
    ):
        raise ValueError("actor_residual_preview_hidden must be a positive integer")
    if requested_optimizer not in {"adam", "muon"}:
        raise ValueError("actor residual preview optimizer is invalid")
    if not isinstance(allow_start, bool):
        raise ValueError(
            "allow_resume_actor_residual_preview_adapter_start must be boolean"
        )
    if (
        not resumed_hparams
        or "actor_residual_preview_adapter" not in resumed_hparams
    ):
        if requested and not future_reference_upgrade:
            raise ValueError(
                "residual preview treatment requires a future-reference upgrade"
            )
        return requested, requested_hidden, requested_optimizer
    saved = resumed_hparams["actor_residual_preview_adapter"]
    if not isinstance(saved, bool):
        raise ValueError(
            "actor_residual_preview_adapter resume metadata must be boolean"
        )
    saved_hidden = resumed_hparams.get(
        "actor_residual_preview_hidden", requested_hidden
    )
    saved_optimizer = resumed_hparams.get(
        "actor_residual_preview_optimizer", "adam"
    )
    if saved_optimizer not in {"adam", "muon"}:
        raise ValueError("saved actor residual preview optimizer is invalid")
    if not saved and requested:
        if not allow_start:
            raise ValueError(
                "residual preview treatment requires explicit start authority"
            )
        return requested, requested_hidden, requested_optimizer
    if (
        saved != requested
        or (saved and saved_hidden != requested_hidden)
        or saved_optimizer != requested_optimizer
    ):
        raise ValueError(
            "actor residual preview settings must match the checkpoint"
        )
    return saved, int(saved_hidden), saved_optimizer


def resolve_frozen_controller_residual_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: bool,
    requested_hidden: int,
    requested_depth: int,
    allow_start: bool,
    is_resume: bool,
) -> tuple[bool, int, int, bool]:
    """Resolve at most one explicit parent-preserving residual upgrade."""
    if not isinstance(requested, bool) or not isinstance(allow_start, bool):
        raise ValueError("frozen controller residual settings must be boolean")
    if (
        isinstance(requested_hidden, bool)
        or not isinstance(requested_hidden, int)
        or requested_hidden < 1
    ):
        raise ValueError("frozen controller residual hidden width is invalid")
    if (
        isinstance(requested_depth, bool)
        or not isinstance(requested_depth, int)
        or requested_depth < 1
    ):
        raise ValueError("frozen controller residual depth is invalid")
    if not is_resume:
        if requested:
            raise ValueError("frozen controller residual requires an E026 resume")
        return False, requested_hidden, requested_depth, False
    if resumed_hparams is None:
        raise ValueError("frozen controller residual resume hparams are required")
    saved = resumed_hparams.get("actor_frozen_controller_residual", False)
    if not isinstance(saved, bool):
        raise ValueError("saved frozen controller residual setting is invalid")
    saved_hidden = resumed_hparams.get(
        "actor_frozen_controller_residual_hidden", requested_hidden
    )
    saved_depth = resumed_hparams.get(
        "actor_frozen_controller_residual_depth", 1 if saved else 0
    )
    if (
        isinstance(saved_depth, bool)
        or not isinstance(saved_depth, int)
        or saved_depth < 0
    ):
        raise ValueError("saved frozen controller residual depth is invalid")
    upgrade = bool(requested and requested_depth == saved_depth + 1)
    if upgrade and not allow_start:
        raise ValueError("frozen controller residual requires explicit authority")
    if requested and requested_depth > saved_depth + 1:
        raise ValueError("frozen controller residual can add only one layer")
    if saved != requested and not upgrade:
        raise ValueError("frozen controller residual must match the checkpoint")
    if requested and requested_depth < saved_depth:
        raise ValueError("frozen controller residual depth cannot decrease")
    if requested and requested_depth == saved_depth and not saved:
        raise ValueError("frozen controller residual checkpoint is invalid")
    if saved and saved_hidden != requested_hidden:
        raise ValueError("frozen controller residual width must match the checkpoint")
    return requested, requested_hidden, requested_depth, upgrade


def requires_plain_residual_preview_resume_validation(
    *,
    actor_residual_preview_adapter: bool,
    actor_frozen_controller_residual: bool,
    actor_learned_torso_wrench: bool,
    learned_wrench_state: bool,
) -> bool:
    """Identify resumes whose top-level actor is the plain preview residual."""
    return bool(
        actor_residual_preview_adapter
        and not actor_frozen_controller_residual
        and not (actor_learned_torso_wrench and learned_wrench_state)
    )


def resolve_centroidal_propulsion_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_enabled: bool,
    requested_window: int,
    requested_delta: float,
    requested_weight: float,
    allow_start: bool,
    is_resume: bool,
) -> tuple[bool, int, float, float, bool]:
    """Resolve the single registered four-step propulsion treatment."""
    if not isinstance(requested_enabled, bool) or not isinstance(allow_start, bool):
        raise ValueError("centroidal propulsion settings must be boolean")
    if requested_window != 4 or requested_delta != 0.1 or requested_weight != 1.0:
        raise ValueError("centroidal propulsion settings must match the registry")
    if not is_resume:
        if requested_enabled:
            raise ValueError("centroidal propulsion requires an E026 resume")
        return False, 4, 0.1, 1.0, False
    if resumed_hparams is None:
        raise ValueError("centroidal propulsion resume hparams are required")
    saved = resumed_hparams.get("actor_centroidal_propulsion", False)
    if not isinstance(saved, bool):
        raise ValueError("saved centroidal propulsion setting is invalid")
    upgrade = bool(requested_enabled and not saved)
    if upgrade and not allow_start:
        raise ValueError("centroidal propulsion requires explicit authority")
    if saved != requested_enabled and not upgrade:
        raise ValueError("centroidal propulsion must match the checkpoint")
    if saved:
        saved_contract = (
            resumed_hparams.get("actor_centroidal_window"),
            resumed_hparams.get("actor_centroidal_delta"),
            resumed_hparams.get("actor_centroidal_weight"),
        )
        if saved_contract != (4, 0.1, 1.0):
            raise ValueError("saved centroidal propulsion contract is invalid")
    return requested_enabled, 4, 0.1, 1.0, upgrade


def validate_centroidal_propulsion_configuration(
    *,
    enabled: bool,
    window: int,
    delta: float,
    weight: float,
    frozen_controller_residual: bool,
    actor_residual_preview_adapter: bool,
    torso_wrench_assistance: bool,
    actor_learned_torso_wrench: bool,
    unroll_length: int,
    env_variant: str,
) -> None:
    """Fail closed around the preregistered unassisted G1 treatment."""
    if not enabled:
        return
    if (window, delta, weight) != (4, 0.1, 1.0):
        raise ValueError("centroidal propulsion contract is not registered")
    if not frozen_controller_residual or not actor_residual_preview_adapter:
        raise ValueError("centroidal propulsion requires frozen complete E026")
    if torso_wrench_assistance or actor_learned_torso_wrench:
        raise ValueError("centroidal propulsion requires exact-zero torso wrench")
    if unroll_length < window:
        raise ValueError("centroidal window does not fit the actor horizon")
    if not env_variant.startswith("g1_tracking"):
        raise ValueError("centroidal propulsion requires G1 tracking")


def resolve_capture_point_tracking_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_enabled: bool,
    requested_delta: float,
    requested_weight: float,
    allow_start: bool,
    is_resume: bool,
) -> tuple[bool, float, float, bool]:
    """Resolve one explicit capture-point treatment over frozen E026."""
    if not isinstance(requested_enabled, bool) or not isinstance(allow_start, bool):
        raise ValueError("capture-point tracking settings must be boolean")
    if requested_delta != 0.1:
        raise ValueError("capture-point delta must equal the registered 0.1")
    if not math.isfinite(requested_weight) or requested_weight <= 0.0:
        raise ValueError("capture-point weight must be positive and finite")
    if not is_resume:
        if requested_enabled:
            raise ValueError("capture-point tracking requires an E026 resume")
        return False, 0.1, requested_weight, False
    if resumed_hparams is None:
        raise ValueError("capture-point tracking resume hparams are required")
    saved = resumed_hparams.get("actor_capture_point_tracking", False)
    if not isinstance(saved, bool):
        raise ValueError("saved capture-point tracking setting is invalid")
    upgrade = bool(requested_enabled and not saved)
    if upgrade and not allow_start:
        raise ValueError("capture-point tracking requires explicit authority")
    if saved != requested_enabled and not upgrade:
        raise ValueError("capture-point tracking must match the checkpoint")
    if saved:
        saved_contract = (
            resumed_hparams.get("actor_capture_point_delta"),
            resumed_hparams.get("actor_capture_point_weight"),
        )
        if saved_contract != (0.1, requested_weight):
            raise ValueError("saved capture-point tracking contract is invalid")
    return requested_enabled, 0.1, requested_weight, upgrade


def validate_capture_point_tracking_configuration(
    *,
    enabled: bool,
    delta: float,
    weight: float,
    frozen_controller_residual: bool,
    actor_residual_preview_adapter: bool,
    torso_wrench_assistance: bool,
    actor_learned_torso_wrench: bool,
    unroll_length: int,
    env_variant: str,
) -> None:
    """Fail closed around the unassisted capture-point treatment."""
    if not enabled:
        return
    if delta != 0.1 or not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("capture-point tracking contract is invalid")
    if not frozen_controller_residual or not actor_residual_preview_adapter:
        raise ValueError("capture-point tracking requires frozen complete E026")
    if torso_wrench_assistance or actor_learned_torso_wrench:
        raise ValueError("capture-point tracking requires exact-zero torso wrench")
    if unroll_length != 24:
        raise ValueError("capture-point tracking requires the E026 H24 horizon")
    if not env_variant.startswith("g1_tracking"):
        raise ValueError("capture-point tracking requires G1 tracking")


def validate_first_frozen_controller_update(
    *, gradient_norm: float, update_norm: float
) -> None:
    """Reject a dead/nonfinite first update before a treatment can continue."""
    values = (gradient_norm, update_norm)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in values
    ):
        raise RuntimeError(
            "first frozen-controller residual update must be finite and nonzero"
        )


def select_initial_training_state(*, initialized_state, resumed_state):
    """Select the full saved state for an exact continuation."""
    return resumed_state if resumed_state is not None else initialized_state


def migrate_env_state_metrics(resumed_env_state, initialized_env_state):
    """Align diagnostic metric and info leaves without changing physical state."""
    resumed_metrics = resumed_env_state.metrics
    initialized_metrics = initialized_env_state.metrics
    metrics = {
        name: resumed_metrics.get(name, initial_value)
        for name, initial_value in initialized_metrics.items()
    }
    resumed_info = resumed_env_state.info
    initialized_info = initialized_env_state.info
    info = {
        name: resumed_info.get(name, initial_value)
        for name, initial_value in initialized_info.items()
    }
    return resumed_env_state.replace(metrics=metrics, info=info)


def update_adaptive_phase_state(
    *,
    failed_count: jax.Array,
    transition_phases: jax.Array,
    terminals: jax.Array,
    reference_length: int,
    alpha: float,
) -> jax.Array:
    """Update one failure EMA from an arbitrarily sharded transition batch."""
    return update_phase_sampler(
        PhaseSamplerState(failed_count=jp.asarray(failed_count)),
        phases=transition_phases,
        terminals=terminals,
        reference_length=reference_length,
        alpha=alpha,
    ).failed_count


def migrate_adaptive_phase_env_state(env_state, *, reference_length: int):
    """Add the adaptive EMA leaf to a legacy batched environment state."""
    if "phase_sampler_failed_count" in env_state.info:
        return env_state
    initial = init_phase_sampler(reference_length).failed_count
    leading_shape = jp.asarray(env_state.info["phase"]).shape
    failed_count = jp.broadcast_to(initial, leading_shape + initial.shape)
    return env_state.replace(
        info={
            **env_state.info,
            "phase_sampler_failed_count": failed_count,
        }
    )


def broadcast_adaptive_phase_state(env_state, failed_count: jax.Array):
    """Broadcast one completed EMA into every batched environment state."""
    current = env_state.info["phase_sampler_failed_count"]
    broadcast = jp.broadcast_to(jp.asarray(failed_count), current.shape)
    return env_state.replace(
        info={
            **env_state.info,
            "phase_sampler_failed_count": broadcast,
        }
    )


def transition_phase_before_reset(
    phases: jax.Array,
    *,
    reference_stride: int,
    reference_length: int,
) -> jax.Array:
    """Return the transition phase that env.step evaluates before reset."""
    return jp.minimum(
        jp.asarray(phases, dtype=jp.int32) + reference_stride,
        reference_length - 1,
    )


def adaptive_phase_diagnostics(
    *,
    failed_count: jax.Array,
    transition_phases: jax.Array,
    terminals: jax.Array,
    reference_length: int,
    uniform_ratio: float,
) -> dict[str, jax.Array]:
    """Build fixed-shape adaptive reset telemetry and its validity bit."""
    failed_count = jp.asarray(failed_count)
    transition_phases = jp.asarray(transition_phases, dtype=jp.int32)
    terminals = jp.asarray(terminals, dtype=jp.float32)
    if transition_phases.shape != terminals.shape:
        raise ValueError(
            "transition_phases and terminals must have identical shapes"
        )
    bin_count = failed_count.shape[0]
    bins = jp.clip(
        transition_phases * bin_count // reference_length,
        min=0,
        max=bin_count - 1,
    )
    terminal_bin_counts = jp.bincount(
        bins.reshape(-1),
        weights=terminals.reshape(-1),
        length=bin_count,
    )
    probabilities = phase_sampling_probabilities(
        PhaseSamplerState(failed_count=failed_count),
        uniform_ratio=uniform_ratio,
    )
    minimum_probability = jp.min(probabilities)
    probability_floor = 0.5 / float(bin_count) - 1e-7
    valid = (
        jp.all(jp.isfinite(failed_count))
        & jp.all(jp.isfinite(probabilities))
        & jp.all(jp.isfinite(terminal_bin_counts))
        & jp.all(probabilities >= 0.0)
        & jp.isclose(jp.sum(probabilities), 1.0, atol=1e-6, rtol=0.0)
        & (minimum_probability >= probability_floor)
    )
    return {
        "failure_ema": failed_count,
        "probabilities": probabilities,
        "terminal_bin_counts": terminal_bin_counts,
        "minimum_probability": minimum_probability,
        "valid": valid,
    }


def reduce_cagrad_shard_accumulators(
    sharded_accumulators,
    *,
    alpha: float,
    iterations: int,
) -> dict[str, object]:
    """Merge reduced physical shards and form one CAGrad actor direction."""
    accumulator = jax.tree_util.tree_map(
        lambda leaf: jp.sum(leaf, axis=0), sharded_accumulators
    )
    task_gradients, bin_counts, bins_valid = finalize_phase_gradients(
        accumulator
    )
    result = combine_cagrad(
        task_gradients,
        alpha=alpha,
        iterations=iterations,
    )

    bin_squared_norms = None
    for leaf in jax.tree_util.tree_leaves(task_gradients):
        axes = tuple(range(1, leaf.ndim))
        leaf_squared_norms = jp.sum(jp.square(leaf), axis=axes)
        bin_squared_norms = (
            leaf_squared_norms
            if bin_squared_norms is None
            else bin_squared_norms + leaf_squared_norms
        )
    bin_gradient_norms = jp.sqrt(jp.maximum(bin_squared_norms, 0.0))
    valid = bins_valid & result.valid & jp.all(jp.isfinite(bin_gradient_norms))
    return {
        "accumulator": accumulator,
        "task_gradients": task_gradients,
        "bin_counts": bin_counts,
        "bin_gradient_norms": bin_gradient_norms,
        "result": result,
        "valid": valid,
    }


def cagrad_phase_loss_diagnostics(
    *,
    losses: jax.Array,
    phases: jax.Array,
    phase_count: int,
    bin_count: int,
) -> dict[str, jax.Array]:
    """Reduce detached actor losses over the CAGrad task bins."""
    losses = jax.lax.stop_gradient(jp.asarray(losses))
    phases = jp.asarray(phases, dtype=jp.int32)
    if losses.ndim != 1 or phases.shape != losses.shape:
        raise ValueError("losses and phases must be matching vectors")
    bins = phase_bin_indices(
        phases,
        phase_count=phase_count,
        bin_count=bin_count,
    )
    bin_counts = jp.zeros((bin_count,), dtype=jp.int32).at[bins].add(1)
    finite_losses = jp.isfinite(losses)
    sums = jp.zeros((bin_count,), dtype=losses.dtype).at[bins].add(
        jp.where(finite_losses, losses, 0.0)
    )
    bin_losses = jp.where(
        bin_counts > 0,
        sums / jp.maximum(bin_counts, 1),
        jp.nan,
    )
    valid = (
        jp.all(bin_counts > 0)
        & jp.all(finite_losses)
        & jp.all(jp.isfinite(bin_losses))
    )
    return {
        "bin_counts": bin_counts,
        "bin_losses": bin_losses,
        "valid": valid,
    }


def should_log_training_iteration(
    iteration: int,
    *,
    start_iteration: int,
    interval: int = 10,
) -> bool:
    """Log the first update of this invocation and each relative interval."""
    if interval < 1:
        raise ValueError("logging interval must be positive")
    return iteration >= start_iteration and (
        (iteration - start_iteration) % interval == 0
    )


def squeeze_value_head(values):
    """Remove only the critic output axis, preserving batch/time axes."""
    return jp.squeeze(values, axis=-1)


def actor_bootstrap_scale_at_step(
    step: jax.Array,
    target_scale: float,
    delay_steps: int,
) -> jax.Array:
    """Keeps actor value bootstrapping off until the critic warm-up ends."""
    # Target-critic values are float32.  Keep the scheduled scalar in the
    # same dtype so delay=0 preserves the original weak-scalar arithmetic.
    target = jp.asarray(target_scale, dtype=jp.float32)
    return jp.where(step >= delay_steps, target, jp.zeros_like(target))


def resolve_actor_bootstrap_resume_scale(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_scale: float,
    allow_change: bool,
) -> float:
    """Restore actor bootstrap scale or admit one explicit objective change."""
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_actor_bootstrap_scale_change must be boolean"
        )
    if (
        isinstance(requested_scale, bool)
        or not math.isfinite(requested_scale)
        or requested_scale < 0.0
    ):
        raise ValueError("actor_bootstrap_scale must be finite and non-negative")
    resumed_scale = (
        1.0
        if not resumed_hparams
        else resumed_hparams.get("actor_bootstrap_scale", 1.0)
    )
    if (
        isinstance(resumed_scale, bool)
        or not math.isfinite(resumed_scale)
        or resumed_scale < 0.0
    ):
        raise ValueError(
            "checkpoint actor_bootstrap_scale must be finite and non-negative"
        )
    if resumed_scale != requested_scale and not allow_change:
        raise ValueError(
            "actor_bootstrap_scale must match the checkpoint unless "
            "allow_resume_actor_bootstrap_scale_change is enabled"
        )
    return float(requested_scale if allow_change else resumed_scale)


def validate_jave_resume_settings(
    saved_settings: tuple[object, ...],
    requested_settings: tuple[object, ...],
    *,
    allow_weight_change: bool,
) -> None:
    """Allow an authorized dynamic JAVE weight branch and nothing else."""

    if not isinstance(allow_weight_change, bool):
        raise ValueError("allow_weight_change must be boolean")
    settings_match = (
        saved_settings[1:] == requested_settings[1:]
        if allow_weight_change
        else saved_settings == requested_settings
    )
    if (
        not saved_settings
        or len(saved_settings) != len(requested_settings)
        or not settings_match
    ):
        raise ValueError("JAVE continuation settings must match the checkpoint")


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_zero_head_recovery_feature_adapter(
    path: str | Path,
    *,
    expected_sha256: str,
    template_params,
) -> tuple[object, dict[str, object]]:
    """Load one immutable expert and copy only its hidden representation."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError("zero-head recovery feature source does not exist")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or _sha256_file(resolved) != expected_sha256
    ):
        raise ValueError("zero-head recovery feature source SHA-256 does not match")
    with resolved.open("rb") as stream:
        expert_params = pickle.load(stream)
    candidate, report = transplant_zero_head_recovery_features(
        template_params, expert_params
    )
    report = {
        **report,
        "source_path": str(resolved),
        "source_sha256": expected_sha256,
    }
    return candidate, report


def load_recovery_support_artifact(
    path: str | Path, *, expected_sha256: str
) -> tuple[RecoverySupport, dict[str, object]]:
    """Load and validate one immutable compact recovery support artifact."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError("recovery support artifact does not exist")
    if len(expected_sha256) != 64 or _sha256_file(resolved) != expected_sha256:
        raise ValueError("recovery support artifact SHA-256 does not match")
    required = {
        "anchors",
        "radius",
        "phase_min",
        "phase_max",
        "taper",
        "positive_leave_one_out_distances",
        "protected_negative_distances",
    }
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("recovery support artifact keys do not match")
        arrays = {name: archive[name] for name in archive.files}
    anchors = np.asarray(arrays["anchors"], dtype=np.float32)
    radius = float(np.asarray(arrays["radius"]))
    phase_min = int(np.asarray(arrays["phase_min"]))
    phase_max = int(np.asarray(arrays["phase_max"]))
    taper = int(np.asarray(arrays["taper"]))
    positive_distances = np.asarray(
        arrays["positive_leave_one_out_distances"], dtype=np.float32
    )
    protected_distances = np.asarray(
        arrays["protected_negative_distances"], dtype=np.float32
    )
    if (
        anchors.shape != (24, 328)
        or not np.isfinite(anchors).all()
        or not math.isfinite(radius)
        or radius <= 0.0
        or phase_min < 0
        or phase_max < phase_min
        or taper != 4
        or positive_distances.shape != (24,)
        or protected_distances.ndim != 1
        or protected_distances.size < 1
        or not np.isfinite(positive_distances).all()
        or not np.isfinite(protected_distances).all()
    ):
        raise ValueError("recovery support artifact contract is invalid")
    positive_coverage = int(np.sum(positive_distances < radius))
    if positive_coverage < 20:
        raise ValueError("recovery support positive coverage is insufficient")
    if np.any(protected_distances < 2.0 * radius - 1e-6):
        raise ValueError("recovery support activates on protected negatives")
    support = RecoverySupport(
        anchors=jp.asarray(anchors),
        radius=jp.asarray(radius, dtype=jp.float32),
        phase_min=phase_min,
        phase_max=phase_max,
        taper=taper,
    )
    return support, {
        "valid": True,
        "path": str(resolved),
        "sha256": expected_sha256,
        "radius": radius,
        "phase_min": phase_min,
        "phase_max": phase_max,
        "taper": taper,
        "positive_coverage": positive_coverage,
        "protected_negative_count": int(protected_distances.size),
        "protected_negative_max_gate": 0.0,
    }


def resolve_recovery_support_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_path: str | None,
    requested_sha256: str | None,
    allow_start: bool,
    is_resume: bool,
) -> tuple[str | None, str | None]:
    """Allow one explicit gated-recovery start and exact treated resumes."""
    if not isinstance(allow_start, bool) or not isinstance(is_resume, bool):
        raise ValueError("recovery support authority must be boolean")
    requested_enabled = requested_path is not None or requested_sha256 is not None
    if requested_enabled and (
        requested_path is None
        or requested_sha256 is None
        or len(requested_sha256) != 64
    ):
        raise ValueError("recovery support requires path and SHA-256")
    if not is_resume:
        return requested_path, requested_sha256
    if resumed_hparams is None:
        raise ValueError("recovery support resume metadata is missing")
    saved_enabled = resumed_hparams.get("actor_state_gated_recovery", False)
    if not isinstance(saved_enabled, bool):
        raise ValueError("saved recovery support metadata is invalid")
    if not saved_enabled:
        if requested_enabled and not allow_start:
            raise ValueError("recovery support requires explicit start authority")
        return requested_path, requested_sha256
    saved_path = resumed_hparams.get(
        "actor_state_gated_recovery_support_path"
    )
    saved_sha256 = resumed_hparams.get(
        "actor_state_gated_recovery_support_sha256"
    )
    if (
        not requested_enabled
        or saved_path != requested_path
        or saved_sha256 != requested_sha256
    ):
        raise ValueError("recovery support settings must match the checkpoint")
    return str(saved_path), str(saved_sha256)


def resolve_reference_path_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_path: str | None,
    allow_change: bool,
    is_resume: bool,
) -> tuple[str | None, dict[str, object] | None]:
    """Resolve an exact reference resume or one explicit path treatment."""
    if not isinstance(allow_change, bool):
        raise ValueError("reference path change authority must be boolean")
    if not isinstance(is_resume, bool):
        raise ValueError("reference path resume state must be boolean")
    if not is_resume:
        return requested_path, None
    if resumed_hparams is None:
        raise ValueError("reference path resume metadata is required")
    saved_path = resumed_hparams.get("reference_path")
    if not isinstance(saved_path, str) or not saved_path:
        raise ValueError("reference path resume metadata is invalid")
    saved = Path(saved_path).expanduser().resolve()
    if requested_path is None:
        return str(saved), None
    if not isinstance(requested_path, str) or not requested_path:
        raise ValueError("requested reference path is invalid")
    requested = Path(requested_path).expanduser().resolve()
    if requested == saved:
        return str(saved), None
    if not allow_change:
        raise ValueError("reference path change requires explicit authority")
    if not saved.is_file() or not requested.is_file():
        raise ValueError("reference path migration inputs must be files")
    report: dict[str, object] = {
        "protocol": "g1-reference-path-migration-v1",
        "valid": True,
        "previous_reference_path": str(saved),
        "previous_reference_sha256": _sha256_file(saved),
        "requested_reference_path": str(requested),
        "requested_reference_sha256": _sha256_file(requested),
        "environment_state_reinitialized": True,
    }
    return str(requested), report


def reference_hparams_for_env(env) -> dict[str, object]:
    """Returns the immutable G1 reference contract recorded with a run."""
    reference_path = Path(env.reference_path).resolve()
    hparams = {
        "reference_path": str(reference_path),
        "reference_sha256": _sha256_file(reference_path),
        "reference_fps": env.reference.fps,
        "reference_stride": int(env.reference_stride),
        "reference_states": int(env.reference.qpos.shape[0]),
        "reference_transitions": int(env.reference_transitions),
        "randomization_com_body_name": getattr(
            env, "randomization_com_body_name", None
        ),
        "randomization_uses_curriculum": getattr(
            env, "randomization_uses_curriculum", None
        ),
    }
    if getattr(env, "carried_reset_bank_path", None) is not None:
        bank_path = Path(env.carried_reset_bank_path).resolve()
        hparams.update(
            {
                "carried_reset_bank_path": str(bank_path),
                "carried_reset_bank_sha256": _sha256_file(bank_path),
                "carried_reset_bank_start": int(env.carried_reset_bank_start),
                "carried_reset_bank_size": int(env.carried_reset_bank_size),
            }
        )
    return hparams


def validate_termination_margin_resume(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_weight: float,
    allow_change: bool,
) -> None:
    """Require an explicit treatment flag before changing a resumed objective."""
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_termination_margin_change must be boolean"
        )
    resumed_weight = (
        0.0
        if not resumed_hparams
        else resumed_hparams.get("termination_margin_weight", 0.0)
    )
    if resumed_weight != requested_weight and not allow_change:
        raise ValueError(
            "termination_margin_weight must match the checkpoint unless "
            "allow_resume_termination_margin_change is enabled"
        )


def resolve_carried_reset_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_bank_path: str | None,
    requested_probability: float,
    requested_start: int,
    allow_change: bool,
) -> tuple[str | None, float, int]:
    """Restore carried resets or admit one explicit distribution treatment."""
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_carried_reset_change must be boolean"
        )
    requested = (
        requested_bank_path,
        requested_probability,
        requested_start,
    )
    if not resumed_hparams:
        return requested
    resumed = (
        resumed_hparams.get("carried_reset_bank_path"),
        resumed_hparams.get("carried_reset_probability", 0.0),
        resumed_hparams.get("carried_reset_bank_start", 0),
    )
    if requested == (None, 0.0, 0) and not allow_change:
        return resumed
    if requested != resumed and not allow_change:
        raise ValueError(
            "carried reset settings must match the checkpoint unless "
            "allow_resume_carried_reset_change is enabled"
        )
    return requested


def resolve_root_reset_noise_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_multiplier: float,
    requested_probability: float,
    allow_change: bool,
) -> tuple[float, float]:
    """Restore root-reset settings or admit one explicit distribution change."""
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_reference_root_reset_noise_change must be boolean"
        )
    requested = (requested_multiplier, requested_probability)
    resumed = (
        1.0
        if not resumed_hparams
        else resumed_hparams.get("reference_root_reset_noise_multiplier", 1.0),
        0.0
        if not resumed_hparams
        else resumed_hparams.get("reference_root_reset_noise_probability", 0.0),
    )
    if requested == (1.0, 0.0) and not allow_change:
        return resumed
    if requested != resumed and not allow_change:
        raise ValueError(
            "root reset noise settings must match the checkpoint unless "
            "allow_resume_reference_root_reset_noise_change is enabled"
        )
    return requested


def resolve_actor_per_env_grad_clip_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: float | None,
    allow_change: bool,
) -> float | None:
    """Restore a resumed clip unless an experiment explicitly changes it."""
    if not isinstance(allow_change, bool):
        raise ValueError(
            "allow_resume_actor_per_env_grad_clip_change must be boolean"
        )
    if (
        resumed_hparams is None
        or "actor_per_env_grad_clip" not in resumed_hparams
    ):
        raise ValueError(
            "actor_per_env_grad_clip requires complete resume metadata"
        )
    resumed = resumed_hparams["actor_per_env_grad_clip"]
    if requested is None and not allow_change:
        return resumed
    if requested != resumed and not allow_change:
        raise ValueError(
            "changing actor_per_env_grad_clip requires explicit resume authority"
        )
    return requested if allow_change else resumed


def resolve_cagrad_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_actor_cagrad: bool,
    requested_alpha: float,
    requested_iterations: int,
    requested_bin_count: int,
    allow_change: bool = False,
) -> tuple[bool, float, int, int]:
    """Restore CAGrad checkpoints while allowing legacy treatment starts."""
    if not isinstance(allow_change, bool):
        raise ValueError("CAGrad resume change authority must be boolean")
    if not resumed_hparams or "actor_cagrad" not in resumed_hparams:
        return (
            requested_actor_cagrad,
            requested_alpha,
            requested_iterations,
            requested_bin_count,
        )
    resumed_actor_cagrad = resumed_hparams["actor_cagrad"]
    if not isinstance(resumed_actor_cagrad, bool):
        raise ValueError("CAGrad checkpoint contains invalid resume metadata")
    if not resumed_actor_cagrad:
        return (
            requested_actor_cagrad,
            requested_alpha,
            requested_iterations,
            requested_bin_count,
        )
    required = {
        "actor_cagrad",
        "actor_cagrad_alpha",
        "actor_cagrad_iterations",
        "actor_phase_bin_count",
    }
    if not required.issubset(resumed_hparams):
        raise ValueError("CAGrad checkpoint requires complete resume metadata")
    alpha = resumed_hparams["actor_cagrad_alpha"]
    iterations = resumed_hparams["actor_cagrad_iterations"]
    bin_count = resumed_hparams["actor_phase_bin_count"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha < 0.0
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("CAGrad checkpoint contains invalid resume metadata")
    if isinstance(bin_count, bool) or bin_count != 5:
        raise ValueError("CAGrad checkpoint requires exactly five phase bins")
    if not requested_actor_cagrad:
        if not allow_change:
            return True, float(alpha), iterations, bin_count
        return (
            False,
            requested_alpha,
            requested_iterations,
            requested_bin_count,
        )
    return True, float(alpha), iterations, bin_count


def validate_actor_cagrad_configuration(
    *,
    actor_cagrad: bool,
    alpha: float,
    iterations: int,
    adaptive_phase_sampling: bool,
    actor_phase_robust_weighting: bool,
    env_variant: str,
    actor_per_env_grad_clip: float | None,
    gradient_accumulation_steps: int,
    actor_phase_bin_count: int,
) -> None:
    """Validate the fixed effective-512 CAGrad treatment contract."""
    if not isinstance(actor_cagrad, bool):
        raise ValueError("actor_cagrad must be boolean")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha < 0.0
    ):
        raise ValueError(
            "actor_cagrad_alpha must be non-negative and finite"
        )
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError(
            "actor_cagrad_iterations must be a positive integer"
        )
    if actor_cagrad and adaptive_phase_sampling:
        raise ValueError(
            "actor CAGrad cannot combine with adaptive phase sampling"
        )
    if actor_cagrad and actor_phase_robust_weighting:
        raise ValueError(
            "actor CAGrad cannot combine with phase-robust weighting"
        )
    if actor_cagrad and not env_variant.startswith("g1_tracking"):
        raise ValueError("actor CAGrad requires G1 reference phases")
    if actor_per_env_grad_clip is not None and (
        isinstance(actor_per_env_grad_clip, bool)
        or not isinstance(actor_per_env_grad_clip, (int, float))
        or not math.isfinite(actor_per_env_grad_clip)
        or actor_per_env_grad_clip <= 0.0
    ):
        raise ValueError("per-env gradient clip must be positive and finite")
    if actor_cagrad and gradient_accumulation_steps != 2:
        raise ValueError("actor CAGrad requires exactly two population shards")
    if actor_cagrad and actor_phase_bin_count != 5:
        raise ValueError("actor CAGrad requires exactly five phase bins")


def validate_preview_adapter_configuration(
    *,
    enabled: bool,
    actor_reference_lookahead_steps: tuple[int, ...],
    actor_cagrad: bool,
    history_len: int,
    source_actor_policy,
    initial_full_actor_policy,
    env_variant: str,
) -> None:
    """Validate the bounded frozen-parent preview treatment contract."""
    if not isinstance(enabled, bool):
        raise ValueError("actor_preview_adapter must be boolean")
    if not enabled:
        return
    if not actor_reference_lookahead_steps or not actor_cagrad:
        raise ValueError(
            "actor preview adapter requires future-reference CAGrad"
        )
    if source_actor_policy is not None:
        raise ValueError("actor preview adapter requires a plain Flax actor")
    if initial_full_actor_policy is None and history_len != 10:
        raise ValueError("actor preview adapter requires ten-frame history")
    if initial_full_actor_policy is not None and history_len != 1:
        raise ValueError(
            "full RMR actor preview adapter requires one-frame history"
        )
    if not env_variant.startswith("g1_tracking"):
        raise ValueError("actor preview adapter requires G1 tracking")


def validate_residual_preview_adapter_configuration(
    *,
    enabled: bool,
    hidden_dim: int,
    optimizer_name: str,
    linear_preview_enabled: bool,
    actor_reference_lookahead_steps: tuple[int, ...],
    actor_reference_preview_mode: str,
    actor_cagrad: bool,
    history_len: int,
    source_actor_policy,
    initial_full_actor_policy,
    env_variant: str,
) -> None:
    """Validate the isolated nonlinear frozen-parent treatment."""
    if not isinstance(enabled, bool):
        raise ValueError("actor_residual_preview_adapter must be boolean")
    if (
        isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim < 1
    ):
        raise ValueError("actor_residual_preview_hidden must be a positive integer")
    if optimizer_name not in {"adam", "muon"}:
        raise ValueError("actor residual preview optimizer is invalid")
    if optimizer_name == "muon" and not enabled:
        raise ValueError("Muon requires the residual preview adapter")
    if not enabled:
        return
    if linear_preview_enabled:
        raise ValueError(
            "linear and residual preview adapters are mutually exclusive"
        )
    if not actor_reference_lookahead_steps or not actor_cagrad:
        raise ValueError(
            "residual preview adapter requires future-reference CAGrad"
        )
    if actor_reference_preview_mode != "delta":
        raise ValueError("residual preview adapter requires delta preview")
    if history_len != 10:
        raise ValueError("residual preview adapter requires ten-frame history")
    if source_actor_policy is not None or initial_full_actor_policy is not None:
        raise ValueError("residual preview adapter requires a plain Flax actor")
    if not env_variant.startswith("g1_tracking"):
        raise ValueError("residual preview adapter requires G1 tracking")


def policy_anchor_penalty(
    candidate_action: jax.Array,
    parent_action: jax.Array,
    *,
    weight: float,
) -> jax.Array:
    """Quadratic proximal penalty around a frozen parent policy action."""
    return weight * jp.mean(
        jp.square(candidate_action - jax.lax.stop_gradient(parent_action))
    )


def validate_policy_anchor_source_configuration(
    *,
    path: str | os.PathLike[str] | None,
    sha256: str | None,
    weight: float,
    actor_residual_preview_adapter: bool,
) -> None:
    """Validate an optional immutable residual-policy anchor source."""
    if (path is None) != (sha256 is None):
        raise ValueError(
            "policy anchor source path and SHA-256 must be provided together"
        )
    if path is None:
        return
    if not actor_residual_preview_adapter:
        raise ValueError(
            "policy anchor source requires the residual preview adapter"
        )
    if weight <= 0.0:
        raise ValueError("policy anchor source requires a positive anchor weight")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("policy anchor source SHA-256 is invalid")


def resolve_policy_anchor_source_resume_setting(
    resumed_hparams: dict,
    *,
    requested_path: str | None,
    requested_sha256: str | None,
    allow_change: bool,
) -> tuple[str | None, str | None]:
    """Preserve or explicitly authorize a hash-bound source-anchor change."""
    keys = (
        "actor_policy_anchor_source_path",
        "actor_policy_anchor_source_sha256",
    )
    if not isinstance(resumed_hparams, dict) or not all(
        key in resumed_hparams for key in keys
    ):
        if allow_change:
            return requested_path, requested_sha256
        raise ValueError("policy anchor source resume metadata is incomplete")
    saved = tuple(resumed_hparams[key] for key in keys)
    requested = (requested_path, requested_sha256)
    if saved != requested and not allow_change:
        raise ValueError(
            "policy anchor source change requires explicit authority"
        )
    return requested if saved != requested else saved


def load_policy_anchor_source(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> FrozenPreviewResidualParams:
    """Load one hash-bound frozen residual policy used as a proximal target."""
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ValueError("policy anchor source checkpoint does not exist")
    actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("policy anchor source checkpoint SHA-256 mismatch")
    with source_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    actor_params = getattr(checkpoint, "actor_params", None)
    if not isinstance(actor_params, FrozenPreviewResidualParams):
        raise ValueError(
            "policy anchor source is not a frozen residual preview checkpoint"
        )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, actor_params)


def validate_policy_anchor_source_parent(
    source: FrozenPreviewResidualParams,
    candidate: FrozenPreviewResidualParams,
) -> None:
    """Require source and candidate residuals to share one frozen parent."""
    source_leaves, source_tree = jax.tree_util.tree_flatten(source.parent)
    candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(
        candidate.parent
    )
    if source_tree != candidate_tree or len(source_leaves) != len(
        candidate_leaves
    ):
        raise ValueError("policy anchor source frozen parent does not match")
    if any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(source_leaves, candidate_leaves)
    ):
        raise ValueError("policy anchor source frozen parent does not match")


def validate_actor_policy_anchor_configuration(
    *,
    weight: float,
    initial_full_actor_policy,
    resume_from,
    actor_residual_preview_adapter: bool = False,
    allow_resume_actor_residual_preview_adapter_start: bool = False,
) -> None:
    """Fail closed unless proximal anchoring has an immutable fresh parent."""
    if (
        isinstance(weight, bool)
        or not math.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError(
            "actor policy anchor weight must be non-negative and finite"
        )
    if weight == 0.0:
        return
    if initial_full_actor_policy is not None:
        if resume_from is not None:
            raise ValueError(
                "actor policy anchoring currently requires a fresh run"
            )
        return
    if resume_from is not None and actor_residual_preview_adapter:
        if not allow_resume_actor_residual_preview_adapter_start:
            raise ValueError(
                "resumed policy anchoring requires an explicit frozen residual "
                "start"
            )
        return
    raise ValueError("actor policy anchoring requires a full actor parent")


def train(
    # General
    total_steps: int = 100_000,
    unroll_length: int = 12,
    num_envs: int = 256,
    gradient_accumulation_steps: int = 1,
    actor_lr: float = 5e-3,
    critic_lr: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    target_update_rate: float = 0.01,
    critic_iterations: int = 16,
    use_lr_decay: bool = False,
    xml_path: str = "src/envs/go2/models/scene_mjx.xml",
    action_scale: float = 0.5,
    # Commands
    cmd_vel_x_range: tuple = (-2.0, 2.0),
    cmd_vel_y_range: tuple = (-1.0, 1.0),
    cmd_yaw_rate_range: tuple = (-1.5, 1.5),
    cmd_zero_prob: tuple = (0.1, 0.7, 0.5),
    cmd_ctrl_interval_range: tuple = (60, 140),
    # Randomization
    action_noise_std_start: float | list[float] = 0.5,
    action_noise_std_end: float | list[float] = 0.32,
    allow_resume_action_noise_change: bool = False,
    action_noise_schedule_steps: int | None = None,
    friction_range: tuple = (0.5, 2.0),
    mass_range: tuple = (0.85, 1.15),
    kp_range: tuple = (25.0, 45.0),
    kd_range: tuple = (0.3, 0.7),
    com_offset_range: tuple = (0.05, 0.05, 0.04),
    push_velocity_range: tuple = (-1.0, 1.0),
    push_interval_s: float = 4.0,
    terrain_flat_prob: float = 0.2,
    terrain_slope_max: float = 5.0,
    terrain_bump_std: float = 0.4,
    terrain_bump_decay: float = 0.4,
    terrain: bool = False,
    domain_randomization: bool = False,
    solver_profile: str | None = None,
    solver_iterations: int = 1,
    solver_ls_iterations: int = 5,
    # Annealing
    zero_difficulty_frac: float = 0.0,
    curriculum_grace: int = None,
    curriculum_steps: int = None,
    # Misc.
    diagnose: bool = False,
    seed: int = 0,
    resume_from: str = None,
    resume_random_seed: int | None = None,
    checkpoint_interval: int = 10_000,
    checkpoint_steps: tuple[int, ...] | None = None,
    max_episode_length: int = 5000,
    actor_history_len: int = 10,
    expected_actor_obs_dim: int | None = None,
    actor_observation_noise: bool = False,
    actor_observe_motion_anchor_position: bool = False,
    actor_reference_lookahead_steps: tuple[int, ...] = (),
    actor_reference_preview_mode: str = "absolute",
    allow_resume_actor_reference_lookahead_upgrade: bool = False,
    actor_preview_adapter: bool = False,
    actor_residual_preview_adapter: bool = False,
    actor_residual_preview_hidden: int = 256,
    actor_residual_preview_optimizer: str = "adam",
    allow_resume_actor_residual_preview_adapter_start: bool = False,
    actor_frozen_controller_residual: bool = False,
    actor_frozen_controller_residual_hidden: int = 256,
    actor_frozen_controller_residual_depth: int = 1,
    allow_resume_actor_frozen_controller_residual_start: bool = False,
    actor_counterfactual_wrench_distillation: bool = False,
    actor_counterfactual_wrench_teacher_path: str | None = None,
    actor_counterfactual_wrench_teacher_sha256: str | None = None,
    actor_counterfactual_wrench_feasibility_path: str | None = None,
    actor_counterfactual_wrench_feasibility_sha256: str | None = None,
    allow_resume_actor_counterfactual_wrench_distillation_start: bool = False,
    actor_centroidal_propulsion: bool = False,
    actor_centroidal_window: int = 4,
    actor_centroidal_delta: float = 0.1,
    actor_centroidal_weight: float = 1.0,
    allow_resume_actor_centroidal_propulsion_start: bool = False,
    actor_capture_point_tracking: bool = False,
    actor_capture_point_delta: float = 0.1,
    actor_capture_point_weight: float = 1.0,
    allow_resume_actor_capture_point_tracking_start: bool = False,
    actor_residual_preview_initial_adapter_path: str | None = None,
    actor_residual_preview_initial_adapter_sha256: str | None = None,
    actor_recovery_teacher_dataset_path: str | None = None,
    actor_recovery_teacher_dataset_sha256: str | None = None,
    actor_recovery_teacher_gradient_ratio: float = 0.0,
    allow_resume_actor_recovery_teacher_change: bool = False,
    actor_state_gated_recovery_support_path: str | None = None,
    actor_state_gated_recovery_support_sha256: str | None = None,
    allow_resume_actor_state_gated_recovery_start: bool = False,
    env_variant: str = "blind_nolinvel_nokinref",
    actor_per_env_grad_clip: float = None,
    allow_resume_actor_per_env_grad_clip_change: bool = False,
    actor_contact_topology_gradient_truncation: bool = False,
    allow_resume_actor_contact_topology_gradient_truncation_change: bool = False,
    critic_per_env_grad_clip: float = None,
    actor_phase_robust_weighting: bool = False,
    actor_phase_bin_count: int = 5,
    actor_phase_robust_fraction: float = 0.5,
    actor_cagrad: bool = False,
    actor_cagrad_alpha: float = 0.5,
    actor_cagrad_iterations: int = 32,
    allow_resume_actor_cagrad_change: bool = False,
    adaptive_phase_sampling: bool = False,
    adaptive_phase_uniform_ratio: float = 0.5,
    adaptive_phase_alpha: float = 0.001,
    actor_bootstrap_scale: float = 1.0,
    actor_bootstrap_delay_steps: int = 0,
    allow_resume_actor_bootstrap_scale_change: bool = False,
    ahac: bool = False,
    ahac_horizon_min: int = 8,
    ahac_horizon_max: int = 24,
    ahac_contact_threshold: float = 500.0,
    ahac_dual_lr: float = 5e-4,
    ahac_critic_max_iterations: int = 64,
    ahac_critic_tolerance: float = 0.2,
    allow_resume_ahac_change: bool = False,
    actor_hidden: tuple[int, ...] = (512, 256, 128),
    actor_layer_norm: bool = True,
    actor_zero_output: bool = True,
    source_actor_policy=None,
    initial_full_actor_policy=None,
    actor_policy_anchor_weight: float = 0.0,
    actor_policy_anchor_source_path: str | None = None,
    actor_policy_anchor_source_sha256: str | None = None,
    allow_resume_actor_policy_anchor_source_change: bool = False,
    residual_action_scale: float = 0.0,
    differentiate_source_feedback: bool = True,
    effort_limit_scale: float = 1.0,
    termination_margin_weight: float = 0.0,
    allow_resume_termination_margin_change: bool = False,
    tracking_velocity_kernel: str = "exponential",
    allow_resume_tracking_velocity_kernel_change: bool = False,
    tracking_anchor_position_kernel: str = "exponential",
    allow_resume_tracking_anchor_position_kernel_change: bool = False,
    tracking_torso_orientation_weight: float = 0.0,
    allow_resume_tracking_torso_orientation_change: bool = False,
    tracking_root_velocity_weight: float = 0.0,
    allow_resume_tracking_root_velocity_change: bool = False,
    jave_vg_weight: float = 0.0,
    jave_vg_warmup_steps: int = 5_000,
    jave_ldm_hidden: tuple[int, ...] = (256, 256),
    jave_ldm_lr: float = 3e-4,
    jave_ldm_iterations: int = 4,
    jave_ldm_batch_size: int = 256,
    jave_vg_batch_size: int = 256,
    jave_ldm_buffer_capacity: int = 100_000,
    jave_reward_feature_scale: float = 8.0,
    jave_collect_transitions: bool = False,
    allow_resume_jave_start: bool = False,
    allow_resume_jave_weight_change: bool = False,
    reference_reset_noise_scale: float = 0.0,
    reference_root_reset_noise_multiplier: float = 1.0,
    reference_root_reset_noise_probability: float = 0.0,
    allow_resume_reference_root_reset_noise_change: bool = False,
    demonstration_replay_threshold: float | None = None,
    allow_resume_demonstration_replay_change: bool = False,
    reference_residual_control: bool = False,
    reference_residual_scale: float = 0.5,
    carried_reset_bank_path: str | None = None,
    carried_reset_probability: float = 0.0,
    carried_reset_bank_start: int = 0,
    allow_resume_carried_reset_change: bool = False,
    torso_wrench_assistance: bool = False,
    torso_wrench_assistance_start_step: int = 0,
    torso_wrench_assistance_end_step: int = 1,
    torso_wrench_assistance_zero_fraction: float = 0.0,
    allow_resume_torso_wrench_assistance_change: bool = False,
    torso_wrench_assistance_continuous: bool = False,
    actor_torso_wrench_assistance_conditioning: bool = False,
    actor_observe_torso_wrench_assistance: bool = False,
    allow_resume_assistance_conditioning_change: bool = False,
    actor_learned_torso_wrench: bool = False,
    actor_learned_torso_wrench_hidden: int = 256,
    actor_learned_torso_wrench_scale: float = 1.0,
    actor_learned_torso_wrench_scale_end: float | None = None,
    actor_learned_torso_wrench_scale_start_step: int = 0,
    actor_learned_torso_wrench_scale_end_step: int = 1,
    actor_learned_torso_wrench_condition_on_scale: bool = False,
    actor_learned_torso_wrench_train_controller: bool = False,
    actor_learned_torso_wrench_penalty: float = 0.0,
    allow_resume_actor_learned_torso_wrench_start: bool = False,
    allow_resume_actor_learned_torso_wrench_change: bool = False,
    reference_path: str | None = None,
    allow_resume_reference_path_change: bool = False,
    reference_stride: int | None = None,
    determinism_probe_output: str | None = None,
):
    """
    Train a quadruped locomotion policy using SHAC.

    Args:
        total_steps: Total environment steps to train
        unroll_length: Number of steps per trajectory rollout (short horizon h)
        num_envs: Number of parallel environments (N)
        gradient_accumulation_steps: Equal microbatches averaged per update
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        gamma: Discount factor
        target_update_rate: Soft update rate for target critic (1-alpha)
        critic_iterations: Number of critic gradient steps per actor update
        use_lr_decay: Linear LR decay to 62.5% over training
        action_scale: Scale factor for actions
        cmd_vel_x_range: (min, max) for forward velocity command (m/s)
        cmd_vel_y_range: (min, max) for lateral velocity command (m/s)
        cmd_yaw_rate_range: (min, max) for yaw rate command (rad/s)
        cmd_zero_prob: Per-component probability of zeroing (vx, vy, yaw)
        cmd_ctrl_interval_range: (min, max) steps between random command samples
        action_noise_std_start: Std dev of Gaussian action noise at step 0
        action_noise_std_end: Std dev of Gaussian action noise at total_steps
        friction_range: (lo, hi) multiplicative factor for geom_friction per episode
        mass_range: (lo, hi) multiplicative factor for body_mass per episode
        effort_limit_scale: Fixed G1 controller torque-limit multiplier.
        termination_margin_weight: Optional differentiable G1 termination-
                                   margin surrogate weight.
        allow_resume_termination_margin_change: Explicitly permit a resumed
                                                 objective-treatment change.
        tracking_velocity_kernel: Kernel used by G1 linear/angular velocity
                                  tracking terms.
        allow_resume_tracking_velocity_kernel_change: Explicitly permit a
                                                       resumed kernel change.
        tracking_anchor_position_kernel: Kernel used by the G1 anchor-position
                                         tracking term.
        allow_resume_tracking_anchor_position_kernel_change: Explicitly permit
                                                               its resumed change.
        tracking_torso_orientation_weight: Optional direct pseudo-Huber torso
                                            orientation reward weight.
        allow_resume_tracking_torso_orientation_change: Explicitly permit a
                                                          resumed objective change.
        tracking_root_velocity_weight: Optional explicit pseudo-Huber pelvis
                                       linear/angular velocity reward weight.
        allow_resume_tracking_root_velocity_change: Explicitly permit a
                                                     resumed objective change.
        reference_reset_noise_scale: Scale of upstream RMR reference-reset
                                     perturbations; zero preserves exact RSI.
        reference_root_reset_noise_multiplier: Multiplier applied only to root
                                               reset pose and velocity limits.
        reference_root_reset_noise_probability: Per-reset probability of the
                                                root-focused recovery cohort.
        allow_resume_reference_root_reset_noise_change: Explicitly permit a
                                                        resumed reset treatment.
        carried_reset_bank_path: Optional NPZ containing rollout qpos/qvel/phase.
        carried_reset_probability: Fraction of resets sampled from that bank.
        carried_reset_bank_start: Leading bank rows excluded from sampling.
        allow_resume_carried_reset_change: Explicitly permit a resumed reset-
                                           distribution treatment.
        torso_wrench_assistance: Apply the bounded analytic torso wrench during
                                 actor rollouts only.
        torso_wrench_assistance_start_step: Absolute step with assistance one.
        torso_wrench_assistance_end_step: Absolute step reaching exact zero.
        torso_wrench_assistance_zero_fraction: Fraction of environments held at
                                               exact zero assistance per unroll.
        allow_resume_torso_wrench_assistance_change: Explicitly permit changing
                                                     a resumed assistance treatment.
        torso_wrench_assistance_continuous: Sample assisted environments
                                             continuously from zero to the cap.
        actor_torso_wrench_assistance_conditioning: Add one scalar input only
                                                     to the residual adapter.
        actor_observe_torso_wrench_assistance: Route the true assistance scalar;
                                               false routes exact zero.
        allow_resume_assistance_conditioning_change: Explicitly authorize a
                                                      resumed scalar-boundary change.
        allow_resume_reference_path_change: Explicitly authorize a resumed
                                            reference treatment.
        actor_observe_motion_anchor_position: Add the motion-anchor position
                                               error to every G1 actor frame.
        kp_range: (lo, hi) absolute range for actuator position gain per episode
        kd_range: (lo, hi) absolute range for actuator velocity gain per episode
        push_velocity_range: Interval root x/y velocity disturbance range.
        push_interval_s: Seconds between velocity pushes.
        terrain_flat_prob: Fraction of terrain episodes that use nominal gravity.
        terrain: Enable implicit tilted-gravity terrain randomization.
        zero_difficulty_frac: Fraction of envs that are held at difficulty=0 each
                              unroll, regardless of curriculum progress. These envs
                              see nominal gravity, nominal gains, and no COM offset.
        curriculum_grace: Steps at difficulty=0 before ramping starts.
        curriculum_steps: Steps over which difficulty ramps 0->1 (after grace).
        diagnose: Enable detailed diagnostic logging
        seed: Random seed
        resume_from: Path to checkpoint .pkl file or training folder to resume from
        resume_random_seed: Optional independent RNG stream for exact resume.
                            Changes only trainer and per-environment RNG keys.
        checkpoint_interval: Save checkpoint every N steps
        checkpoint_steps: Optional exact checkpoint step schedule. When set,
            it replaces the periodic interval.
        allow_resume_actor_bootstrap_scale_change: Explicitly permit changing
            actor terminal-value scale when resuming a checkpoint.
        actor_bootstrap_delay_steps: Environment steps before the actor uses
            target-critic terminal value estimates.
        determinism_probe_output: Optional create-only JSON path. When set,
            compile one update, invoke that same callable twice from the same
            input state, publish exact boundary comparisons, and return without
            entering the ordinary training loop.

    Returns:
        Tuple of (final_state, save_directory)
    """
    if (
        source_actor_policy is not None
        and initial_full_actor_policy is not None
    ):
        raise ValueError(
            "source_actor_policy and initial_full_actor_policy are "
            "mutually exclusive"
        )
    if not isinstance(allow_resume_actor_state_gated_recovery_start, bool):
        raise ValueError(
            "allow_resume_actor_state_gated_recovery_start must be boolean"
        )
    if not isinstance(allow_resume_actor_cagrad_change, bool):
        raise ValueError("allow_resume_actor_cagrad_change must be boolean")
    requested_gated_recovery = (
        actor_state_gated_recovery_support_path is not None
        or actor_state_gated_recovery_support_sha256 is not None
    )
    if (
        actor_state_gated_recovery_support_path is None
    ) != (actor_state_gated_recovery_support_sha256 is None):
        raise ValueError(
            "state-gated recovery support path and SHA-256 are required together"
        )
    if requested_gated_recovery and not actor_residual_preview_adapter:
        raise ValueError(
            "state-gated recovery requires actor_residual_preview_adapter"
        )
    if requested_gated_recovery and actor_torso_wrench_assistance_conditioning:
        raise ValueError(
            "state-gated recovery does not support assistance-conditioned adapters"
        )
    validate_actor_policy_anchor_configuration(
        weight=actor_policy_anchor_weight,
        initial_full_actor_policy=initial_full_actor_policy,
        resume_from=resume_from,
        actor_residual_preview_adapter=actor_residual_preview_adapter,
        allow_resume_actor_residual_preview_adapter_start=(
            allow_resume_actor_residual_preview_adapter_start
        ),
    )
    validate_policy_anchor_source_configuration(
        path=actor_policy_anchor_source_path,
        sha256=actor_policy_anchor_source_sha256,
        weight=actor_policy_anchor_weight,
        actor_residual_preview_adapter=actor_residual_preview_adapter,
    )
    if not isinstance(
        allow_resume_actor_policy_anchor_source_change, bool
    ):
        raise ValueError(
            "allow_resume_actor_policy_anchor_source_change must be boolean"
        )
    if initial_full_actor_policy is not None and residual_action_scale != 0.0:
        raise ValueError(
            "initial_full_actor_policy is standalone and cannot use "
            "residual_action_scale"
        )
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps < 1
    ):
        raise ValueError(
            "gradient_accumulation_steps must be a positive integer"
        )
    if not isinstance(actor_phase_robust_weighting, bool):
        raise ValueError("actor_phase_robust_weighting must be boolean")
    if (
        isinstance(actor_phase_bin_count, bool)
        or not isinstance(actor_phase_bin_count, int)
        or actor_phase_bin_count < 1
    ):
        raise ValueError("actor_phase_bin_count must be positive")
    if (
        isinstance(actor_phase_robust_fraction, bool)
        or not math.isfinite(actor_phase_robust_fraction)
        or not 0.0 <= actor_phase_robust_fraction <= 1.0
    ):
        raise ValueError("actor_phase_robust_fraction must be in [0, 1]")
    if actor_phase_robust_weighting and gradient_accumulation_steps != 1:
        raise ValueError(
            "phase-robust weighting requires one population shard"
        )
    if actor_phase_robust_weighting and actor_per_env_grad_clip is not None:
        raise ValueError(
            "phase-robust weighting cannot combine with per-env clipping"
        )
    if actor_phase_robust_weighting and not env_variant.startswith(
        "g1_tracking"
    ):
        raise ValueError(
            "phase-robust weighting requires G1 reference phases"
        )
    validate_actor_cagrad_configuration(
        actor_cagrad=actor_cagrad,
        alpha=actor_cagrad_alpha,
        iterations=actor_cagrad_iterations,
        adaptive_phase_sampling=adaptive_phase_sampling,
        actor_phase_robust_weighting=actor_phase_robust_weighting,
        env_variant=env_variant,
        actor_per_env_grad_clip=actor_per_env_grad_clip,
        gradient_accumulation_steps=gradient_accumulation_steps,
        actor_phase_bin_count=actor_phase_bin_count,
    )
    validate_preview_adapter_configuration(
        enabled=actor_preview_adapter,
        actor_reference_lookahead_steps=actor_reference_lookahead_steps,
        actor_cagrad=actor_cagrad,
        history_len=actor_history_len,
        source_actor_policy=source_actor_policy,
        initial_full_actor_policy=initial_full_actor_policy,
        env_variant=env_variant,
    )
    validate_residual_preview_adapter_configuration(
        enabled=actor_residual_preview_adapter,
        hidden_dim=actor_residual_preview_hidden,
        optimizer_name=actor_residual_preview_optimizer,
        linear_preview_enabled=actor_preview_adapter,
        actor_reference_lookahead_steps=actor_reference_lookahead_steps,
        actor_reference_preview_mode=actor_reference_preview_mode,
        actor_cagrad=(actor_cagrad or actor_learned_torso_wrench),
        history_len=actor_history_len,
        source_actor_policy=source_actor_policy,
        initial_full_actor_policy=initial_full_actor_policy,
        env_variant=env_variant,
    )
    validate_centroidal_propulsion_configuration(
        enabled=actor_centroidal_propulsion,
        window=actor_centroidal_window,
        delta=actor_centroidal_delta,
        weight=actor_centroidal_weight,
        frozen_controller_residual=actor_frozen_controller_residual,
        actor_residual_preview_adapter=actor_residual_preview_adapter,
        torso_wrench_assistance=torso_wrench_assistance,
        actor_learned_torso_wrench=actor_learned_torso_wrench,
        unroll_length=unroll_length,
        env_variant=env_variant,
    )
    validate_capture_point_tracking_configuration(
        enabled=actor_capture_point_tracking,
        delta=actor_capture_point_delta,
        weight=actor_capture_point_weight,
        frozen_controller_residual=actor_frozen_controller_residual,
        actor_residual_preview_adapter=actor_residual_preview_adapter,
        torso_wrench_assistance=torso_wrench_assistance,
        actor_learned_torso_wrench=actor_learned_torso_wrench,
        unroll_length=unroll_length,
        env_variant=env_variant,
    )
    if not isinstance(adaptive_phase_sampling, bool):
        raise ValueError("adaptive_phase_sampling must be boolean")
    if (
        isinstance(adaptive_phase_uniform_ratio, bool)
        or not math.isfinite(adaptive_phase_uniform_ratio)
        or not 0.0 <= adaptive_phase_uniform_ratio <= 1.0
    ):
        raise ValueError(
            "adaptive_phase_uniform_ratio must be finite and in [0, 1]"
        )
    if (
        isinstance(adaptive_phase_alpha, bool)
        or not math.isfinite(adaptive_phase_alpha)
        or not 0.0 < adaptive_phase_alpha <= 1.0
    ):
        raise ValueError("adaptive_phase_alpha must be finite and in (0, 1]")
    if adaptive_phase_sampling and not env_variant.startswith("g1_tracking"):
        raise ValueError(
            "adaptive phase sampling requires G1 reference phases"
        )
    if (
        isinstance(actor_bootstrap_scale, bool)
        or not math.isfinite(actor_bootstrap_scale)
        or actor_bootstrap_scale < 0.0
    ):
        raise ValueError(
            "actor_bootstrap_scale must be finite and non-negative"
        )
    if not isinstance(allow_resume_actor_bootstrap_scale_change, bool):
        raise ValueError(
            "allow_resume_actor_bootstrap_scale_change must be boolean"
        )
    if (
        isinstance(actor_bootstrap_delay_steps, bool)
        or not isinstance(actor_bootstrap_delay_steps, int)
        or actor_bootstrap_delay_steps < 0
    ):
        raise ValueError(
            "actor_bootstrap_delay_steps must be a non-negative integer"
        )
    if not isinstance(ahac, bool):
        raise ValueError("ahac must be boolean")
    if not isinstance(allow_resume_ahac_change, bool):
        raise ValueError("allow_resume_ahac_change must be boolean")
    for name, value in (
        ("ahac_horizon_min", ahac_horizon_min),
        ("ahac_horizon_max", ahac_horizon_max),
        ("ahac_critic_max_iterations", ahac_critic_max_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if ahac_horizon_max < ahac_horizon_min:
        raise ValueError("AHAC maximum horizon must be at least its minimum")
    if ahac and unroll_length != ahac_horizon_max:
        raise ValueError("AHAC unroll length must equal its maximum horizon")
    for name, value in (
        ("ahac_contact_threshold", ahac_contact_threshold),
        ("ahac_dual_lr", ahac_dual_lr),
        ("ahac_critic_tolerance", ahac_critic_tolerance),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if ahac and (actor_bootstrap_scale != 1.0 or actor_bootstrap_delay_steps != 0):
        raise ValueError(
            "AHAC requires actor bootstrap scale 1.0 with zero delay"
        )
    if (
        isinstance(termination_margin_weight, bool)
        or not math.isfinite(termination_margin_weight)
        or termination_margin_weight < 0.0
    ):
        raise ValueError(
            "termination_margin_weight must be non-negative and finite"
        )
    if not isinstance(allow_resume_termination_margin_change, bool):
        raise ValueError(
            "allow_resume_termination_margin_change must be boolean"
        )
    if tracking_velocity_kernel not in {"exponential", "pseudo_huber"}:
        raise ValueError("tracking velocity kernel is invalid")
    if not isinstance(allow_resume_tracking_velocity_kernel_change, bool):
        raise ValueError(
            "allow_resume_tracking_velocity_kernel_change must be boolean"
        )
    if tracking_anchor_position_kernel not in {
        "exponential",
        "dual_scale",
        "quadratic",
    }:
        raise ValueError("tracking anchor position kernel is invalid")
    if not isinstance(
        allow_resume_tracking_anchor_position_kernel_change, bool
    ):
        raise ValueError(
            "allow_resume_tracking_anchor_position_kernel_change must be boolean"
        )
    tracking_torso_orientation_weight = (
        resolve_tracking_torso_orientation_resume_weight(
            None,
            requested=tracking_torso_orientation_weight,
            allow_change=allow_resume_tracking_torso_orientation_change,
            is_resume=False,
        )
    )
    tracking_root_velocity_weight = resolve_tracking_root_velocity_resume_weight(
        None,
        requested=tracking_root_velocity_weight,
        allow_change=allow_resume_tracking_root_velocity_change,
        is_resume=False,
    )
    if not isinstance(allow_resume_carried_reset_change, bool):
        raise ValueError(
            "allow_resume_carried_reset_change must be boolean"
        )
    if not isinstance(allow_resume_reference_root_reset_noise_change, bool):
        raise ValueError(
            "allow_resume_reference_root_reset_noise_change must be boolean"
        )
    if not isinstance(allow_resume_actor_per_env_grad_clip_change, bool):
        raise ValueError(
            "allow_resume_actor_per_env_grad_clip_change must be boolean"
        )
    if not isinstance(allow_resume_torso_wrench_assistance_change, bool):
        raise ValueError(
            "allow_resume_torso_wrench_assistance_change must be boolean"
        )
    if not isinstance(allow_resume_assistance_conditioning_change, bool):
        raise ValueError(
            "allow_resume_assistance_conditioning_change must be boolean"
        )
    if not isinstance(allow_resume_reference_path_change, bool):
        raise ValueError(
            "allow_resume_reference_path_change must be boolean"
        )
    if (
        isinstance(reference_reset_noise_scale, bool)
        or not math.isfinite(reference_reset_noise_scale)
        or reference_reset_noise_scale < 0.0
    ):
        raise ValueError(
            "reference_reset_noise_scale must be non-negative and finite"
        )
    if (
        isinstance(reference_root_reset_noise_multiplier, bool)
        or not math.isfinite(reference_root_reset_noise_multiplier)
        or reference_root_reset_noise_multiplier < 1.0
    ):
        raise ValueError(
            "reference_root_reset_noise_multiplier must be finite and at least one"
        )
    if (
        isinstance(reference_root_reset_noise_probability, bool)
        or not math.isfinite(reference_root_reset_noise_probability)
        or not 0.0 <= reference_root_reset_noise_probability <= 1.0
    ):
        raise ValueError(
            "reference_root_reset_noise_probability must be finite and in [0, 1]"
        )
    if (
        isinstance(carried_reset_probability, bool)
        or not math.isfinite(carried_reset_probability)
        or not 0.0 <= carried_reset_probability <= 1.0
    ):
        raise ValueError(
            "carried_reset_probability must be finite and in [0, 1]"
        )
    if (
        isinstance(carried_reset_bank_start, bool)
        or not isinstance(carried_reset_bank_start, int)
        or carried_reset_bank_start < 0
    ):
        raise ValueError(
            "carried_reset_bank_start must be a non-negative integer"
        )
    if carried_reset_probability > 0.0 and carried_reset_bank_path is None:
        raise ValueError(
            "carried_reset_bank_path is required when "
            "carried_reset_probability is positive"
        )
    if carried_reset_bank_path is not None and carried_reset_probability == 0.0:
        raise ValueError(
            "carried_reset_probability must be positive when "
            "carried_reset_bank_path is set"
        )
    if reference_stride is not None and (
        isinstance(reference_stride, bool)
        or not isinstance(reference_stride, int)
        or reference_stride < 1
    ):
        raise ValueError("reference_stride must be a positive integer")
    if not isinstance(actor_reference_lookahead_steps, tuple):
        raise ValueError(
            "actor_reference_lookahead_steps must be a tuple"
        )
    if actor_reference_preview_mode not in {"absolute", "delta"}:
        raise ValueError("actor_reference_preview_mode is invalid")
    if (
        actor_reference_preview_mode == "delta"
        and not actor_reference_lookahead_steps
    ):
        raise ValueError(
            "delta actor_reference_preview_mode requires lookahead steps"
        )
    if not isinstance(
        allow_resume_actor_reference_lookahead_upgrade, bool
    ):
        raise ValueError(
            "allow_resume_actor_reference_lookahead_upgrade must be boolean"
        )
    for name, value in (
        ("solver_iterations", solver_iterations),
        ("solver_ls_iterations", solver_ls_iterations),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(jave_vg_weight, bool)
        or not np.isfinite(jave_vg_weight)
        or jave_vg_weight < 0.0
    ):
        raise ValueError("jave_vg_weight must be non-negative and finite")
    if (
        isinstance(jave_vg_warmup_steps, bool)
        or not isinstance(jave_vg_warmup_steps, int)
        or jave_vg_warmup_steps < 0
    ):
        raise ValueError("jave_vg_warmup_steps must be a non-negative integer")
    if (
        not isinstance(jave_ldm_hidden, tuple)
        or not jave_ldm_hidden
        or any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            for width in jave_ldm_hidden
        )
    ):
        raise ValueError("jave_ldm_hidden must contain positive integers")
    if (
        isinstance(jave_ldm_lr, bool)
        or not np.isfinite(jave_ldm_lr)
        or jave_ldm_lr <= 0.0
    ):
        raise ValueError("jave_ldm_lr must be positive and finite")
    for name, value in (
        ("jave_ldm_iterations", jave_ldm_iterations),
        ("jave_ldm_batch_size", jave_ldm_batch_size),
        ("jave_vg_batch_size", jave_vg_batch_size),
        ("jave_ldm_buffer_capacity", jave_ldm_buffer_capacity),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if jave_ldm_buffer_capacity < max(
        jave_ldm_batch_size, jave_vg_batch_size
    ):
        raise ValueError("JAVE replay capacity must cover both batch sizes")
    if (
        isinstance(jave_reward_feature_scale, bool)
        or not np.isfinite(jave_reward_feature_scale)
        or jave_reward_feature_scale <= 0.0
    ):
        raise ValueError(
            "jave_reward_feature_scale must be positive and finite"
        )
    if not isinstance(allow_resume_jave_start, bool):
        raise ValueError("allow_resume_jave_start must be boolean")
    if not isinstance(allow_resume_jave_weight_change, bool):
        raise ValueError("allow_resume_jave_weight_change must be boolean")
    if not isinstance(jave_collect_transitions, bool):
        raise ValueError("jave_collect_transitions must be boolean")
    if jave_vg_weight > 0.0 and not jave_collect_transitions:
        raise ValueError(
            "positive jave_vg_weight requires jave_collect_transitions=True"
        )
    jave_enabled = jave_collect_transitions
    effective_num_envs = num_envs * gradient_accumulation_steps
    steps_per_actor_update = effective_num_envs * unroll_length
    if checkpoint_steps is not None:
        if (
            not isinstance(checkpoint_steps, tuple)
            or not checkpoint_steps
            or any(
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 1
                for step in checkpoint_steps
            )
            or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps
            or any(step % steps_per_actor_update for step in checkpoint_steps)
            or checkpoint_steps[-1] != total_steps
        ):
            raise ValueError(
                "checkpoint_steps must be increasing actor-update steps ending at total_steps"
            )

    # Handle checkpoint resumption
    resumed_state = None
    resumed_step = 0
    resumed_hparams = None
    future_reference_upgrade = False
    residual_adapter_upgrade = False
    frozen_controller_residual_upgrade = False
    counterfactual_wrench_upgrade = False
    centroidal_propulsion_upgrade = False
    reference_path_migration_report = None
    recovery_support = None
    recovery_support_report = None

    if resume_from:
        resumed_state, resumed_hparams, resumed_step = load_checkpoint(resume_from)
        tracking_velocity_kernel = (
            resolve_tracking_velocity_kernel_resume_setting(
                resumed_hparams,
                requested=tracking_velocity_kernel,
                allow_change=allow_resume_tracking_velocity_kernel_change,
                is_resume=True,
            )
        )
        tracking_anchor_position_kernel = (
            resolve_anchor_position_kernel_resume_setting(
                resumed_hparams,
                requested=tracking_anchor_position_kernel,
                allow_change=(
                    allow_resume_tracking_anchor_position_kernel_change
                ),
                is_resume=True,
            )
        )
        tracking_torso_orientation_weight = (
            resolve_tracking_torso_orientation_resume_weight(
                resumed_hparams,
                requested=tracking_torso_orientation_weight,
                allow_change=allow_resume_tracking_torso_orientation_change,
                is_resume=True,
            )
        )
        tracking_root_velocity_weight = (
            resolve_tracking_root_velocity_resume_weight(
                resumed_hparams,
                requested=tracking_root_velocity_weight,
                allow_change=allow_resume_tracking_root_velocity_change,
                is_resume=True,
            )
        )
        (
            reference_path,
            reference_path_migration_report,
        ) = resolve_reference_path_resume_setting(
            resumed_hparams,
            requested_path=reference_path,
            allow_change=allow_resume_reference_path_change,
            is_resume=True,
        )
        actor_bootstrap_scale = resolve_actor_bootstrap_resume_scale(
            resumed_hparams,
            requested_scale=actor_bootstrap_scale,
            allow_change=allow_resume_actor_bootstrap_scale_change,
        )
        (
            actor_reference_lookahead_steps,
            future_reference_upgrade,
        ) = resolve_future_reference_resume_settings(
            resumed_hparams if resumed_hparams is not None else {},
            requested_steps=actor_reference_lookahead_steps,
            allow_upgrade=(
                allow_resume_actor_reference_lookahead_upgrade
            ),
        )
        actor_reference_preview_mode = (
            resolve_future_reference_preview_mode(
                resumed_hparams,
                requested_mode=actor_reference_preview_mode,
                future_reference_upgrade=future_reference_upgrade,
            )
        )
        actor_preview_adapter = resolve_preview_adapter_resume_setting(
            resumed_hparams,
            requested=actor_preview_adapter,
        )
        actor_observe_motion_anchor_position = (
            resolve_actor_observe_motion_anchor_position_resume_setting(
                resumed_hparams,
                requested=actor_observe_motion_anchor_position,
            )
        )
        (
            actor_residual_preview_adapter,
            actor_residual_preview_hidden,
            actor_residual_preview_optimizer,
        ) = resolve_residual_preview_adapter_resume_setting(
            resumed_hparams,
            requested=actor_residual_preview_adapter,
            requested_hidden=actor_residual_preview_hidden,
            requested_optimizer=actor_residual_preview_optimizer,
            future_reference_upgrade=future_reference_upgrade,
            allow_start=(
                allow_resume_actor_residual_preview_adapter_start
            ),
        )
        (
            actor_frozen_controller_residual,
            actor_frozen_controller_residual_hidden,
            actor_frozen_controller_residual_depth,
            frozen_controller_residual_upgrade,
        ) = resolve_frozen_controller_residual_resume_setting(
            resumed_hparams,
            requested=actor_frozen_controller_residual,
            requested_hidden=actor_frozen_controller_residual_hidden,
            requested_depth=actor_frozen_controller_residual_depth,
            allow_start=allow_resume_actor_frozen_controller_residual_start,
            is_resume=True,
        )
        (
            actor_centroidal_propulsion,
            actor_centroidal_window,
            actor_centroidal_delta,
            actor_centroidal_weight,
            centroidal_propulsion_upgrade,
        ) = resolve_centroidal_propulsion_resume_settings(
            resumed_hparams,
            requested_enabled=actor_centroidal_propulsion,
            requested_window=actor_centroidal_window,
            requested_delta=actor_centroidal_delta,
            requested_weight=actor_centroidal_weight,
            allow_start=allow_resume_actor_centroidal_propulsion_start,
            is_resume=True,
        )
        (
            actor_capture_point_tracking,
            actor_capture_point_delta,
            actor_capture_point_weight,
            _capture_point_upgrade,
        ) = resolve_capture_point_tracking_resume_settings(
            resumed_hparams,
            requested_enabled=actor_capture_point_tracking,
            requested_delta=actor_capture_point_delta,
            requested_weight=actor_capture_point_weight,
            allow_start=allow_resume_actor_capture_point_tracking_start,
            is_resume=True,
        )
        (
            actor_policy_anchor_source_path,
            actor_policy_anchor_source_sha256,
        ) = resolve_policy_anchor_source_resume_setting(
            resumed_hparams,
            requested_path=actor_policy_anchor_source_path,
            requested_sha256=actor_policy_anchor_source_sha256,
            allow_change=allow_resume_actor_policy_anchor_source_change,
        )
        residual_adapter_upgrade = bool(
            actor_residual_preview_adapter
            and resumed_hparams is not None
            and resumed_hparams.get("actor_residual_preview_adapter") is False
            and allow_resume_actor_residual_preview_adapter_start
        )
        (
            actor_residual_preview_initial_adapter_path,
            actor_residual_preview_initial_adapter_sha256,
        ) = resolve_zero_head_feature_transfer_resume_setting(
            resumed_hparams,
            requested_path=(
                actor_residual_preview_initial_adapter_path
            ),
            requested_sha256=(
                actor_residual_preview_initial_adapter_sha256
            ),
            residual_adapter_enabled=actor_residual_preview_adapter,
            residual_adapter_upgrade=residual_adapter_upgrade,
            is_resume=True,
        )
        (
            actor_state_gated_recovery_support_path,
            actor_state_gated_recovery_support_sha256,
        ) = resolve_recovery_support_resume_setting(
            resumed_hparams,
            requested_path=actor_state_gated_recovery_support_path,
            requested_sha256=actor_state_gated_recovery_support_sha256,
            allow_start=allow_resume_actor_state_gated_recovery_start,
            is_resume=True,
        )
        (
            adaptive_phase_sampling,
            adaptive_phase_uniform_ratio,
            adaptive_phase_alpha,
        ) = resolve_adaptive_phase_resume_settings(
            resumed_state,
            resumed_hparams,
            requested_adaptive_phase_sampling=adaptive_phase_sampling,
            requested_uniform_ratio=adaptive_phase_uniform_ratio,
            requested_alpha=adaptive_phase_alpha,
        )
        (
            actor_cagrad,
            actor_cagrad_alpha,
            actor_cagrad_iterations,
            actor_phase_bin_count,
        ) = resolve_cagrad_resume_settings(
            resumed_hparams,
            requested_actor_cagrad=actor_cagrad,
            requested_alpha=actor_cagrad_alpha,
            requested_iterations=actor_cagrad_iterations,
            requested_bin_count=actor_phase_bin_count,
            allow_change=allow_resume_actor_cagrad_change,
        )
        (
            carried_reset_bank_path,
            carried_reset_probability,
            carried_reset_bank_start,
        ) = resolve_carried_reset_resume_settings(
            resumed_hparams,
            requested_bank_path=carried_reset_bank_path,
            requested_probability=carried_reset_probability,
            requested_start=carried_reset_bank_start,
            allow_change=allow_resume_carried_reset_change,
        )
        (
            reference_root_reset_noise_multiplier,
            reference_root_reset_noise_probability,
        ) = resolve_root_reset_noise_resume_settings(
            resumed_hparams,
            requested_multiplier=reference_root_reset_noise_multiplier,
            requested_probability=reference_root_reset_noise_probability,
            allow_change=allow_resume_reference_root_reset_noise_change,
        )

        actor_per_env_grad_clip = (
            resolve_actor_per_env_grad_clip_resume_setting(
                resumed_hparams,
                requested=actor_per_env_grad_clip,
                allow_change=allow_resume_actor_per_env_grad_clip_change,
            )
        )
        (
            torso_wrench_assistance,
            torso_wrench_assistance_start_step,
            torso_wrench_assistance_end_step,
            torso_wrench_assistance_zero_fraction,
        ) = resolve_torso_wrench_assistance_resume_settings(
            resumed_hparams,
            requested_enabled=torso_wrench_assistance,
            requested_start_step=torso_wrench_assistance_start_step,
            requested_end_step=torso_wrench_assistance_end_step,
            requested_zero_fraction=torso_wrench_assistance_zero_fraction,
            allow_change=allow_resume_torso_wrench_assistance_change,
        )
        (
            torso_wrench_assistance_continuous,
            actor_torso_wrench_assistance_conditioning,
            actor_observe_torso_wrench_assistance,
        ) = resolve_assistance_conditioning_resume_settings(
            resumed_hparams,
            requested_continuous=torso_wrench_assistance_continuous,
            requested_conditioning=actor_torso_wrench_assistance_conditioning,
            requested_observed=actor_observe_torso_wrench_assistance,
            allow_change=allow_resume_assistance_conditioning_change,
        )
        if resumed_hparams:
            print(f"Resuming from step {resumed_step}")
            resumed_accumulation_steps = resumed_hparams.get(
                "gradient_accumulation_steps", 1
            )
            if resumed_accumulation_steps != gradient_accumulation_steps:
                raise ValueError(
                    "gradient_accumulation_steps must match the checkpoint"
                )
            resumed_bootstrap_delay = resumed_hparams.get(
                "actor_bootstrap_delay_steps", 0
            )
            if resumed_bootstrap_delay != actor_bootstrap_delay_steps:
                raise ValueError(
                    "actor_bootstrap_delay_steps must match the checkpoint"
                )
            validate_termination_margin_resume(
                resumed_hparams,
                requested_weight=termination_margin_weight,
                allow_change=allow_resume_termination_margin_change,
            )
            print(
                f"  Loaded hparams: action_scale={resumed_hparams.get('action_scale')}"
            )
            action_scale = resumed_hparams.get("action_scale", action_scale)
            xml_path = resumed_hparams.get("xml_path", xml_path)
            env_variant = resumed_hparams.get("env_variant", env_variant)
            reference_stride = resumed_hparams.get(
                "reference_stride", reference_stride
            )
            reference_reset_noise_scale = resumed_hparams.get(
                "reference_reset_noise_scale", reference_reset_noise_scale
            )
            domain_randomization = resumed_hparams.get(
                "domain_randomization", domain_randomization
            )
            solver_profile = resumed_hparams.get(
                "solver_profile", solver_profile
            )
            solver_iterations = resumed_hparams.get(
                "solver_iterations", solver_iterations
            )
            solver_ls_iterations = resumed_hparams.get(
                "solver_ls_iterations", solver_ls_iterations
            )
            if "kp_range" in resumed_hparams:
                kp_range = tuple(resumed_hparams["kp_range"])
            if "friction_range" in resumed_hparams:
                friction_range = tuple(resumed_hparams["friction_range"])
            if "mass_range" in resumed_hparams:
                mass_range = tuple(resumed_hparams["mass_range"])
            if "kd_range" in resumed_hparams:
                kd_range = tuple(resumed_hparams["kd_range"])
            if "com_offset_range" in resumed_hparams:
                com_offset_range = tuple(resumed_hparams["com_offset_range"])
            if "push_velocity_range" in resumed_hparams:
                push_velocity_range = tuple(resumed_hparams["push_velocity_range"])
            if "push_interval_s" in resumed_hparams:
                push_interval_s = resumed_hparams["push_interval_s"]
            if "terrain_bump_std" in resumed_hparams:
                terrain_bump_std = resumed_hparams["terrain_bump_std"]
            if "terrain_bump_decay" in resumed_hparams:
                terrain_bump_decay = resumed_hparams["terrain_bump_decay"]
            if "cmd_ctrl_interval_range" in resumed_hparams:
                cmd_ctrl_interval_range = tuple(
                    resumed_hparams["cmd_ctrl_interval_range"]
                )
            if "zero_difficulty_frac" in resumed_hparams:
                zero_difficulty_frac = resumed_hparams["zero_difficulty_frac"]
            if "curriculum_grace" in resumed_hparams:
                curriculum_grace = resumed_hparams["curriculum_grace"]
            if "curriculum_steps" in resumed_hparams:
                curriculum_steps = resumed_hparams["curriculum_steps"]
            if "max_episode_length" in resumed_hparams:
                max_episode_length = resumed_hparams["max_episode_length"]
            if "actor_history_len" in resumed_hparams:
                actor_history_len = resumed_hparams["actor_history_len"]
            if "actor_observation_noise" in resumed_hparams:
                actor_observation_noise = resumed_hparams[
                    "actor_observation_noise"
                ]
            if "critic_per_env_grad_clip" in resumed_hparams:
                critic_per_env_grad_clip = resumed_hparams[
                    "critic_per_env_grad_clip"
                ]
    (
        actor_counterfactual_wrench_distillation,
        counterfactual_wrench_upgrade,
    ) = resolve_counterfactual_wrench_resume_setting(
        resumed_hparams,
        requested=actor_counterfactual_wrench_distillation,
        teacher_sha256=actor_counterfactual_wrench_teacher_sha256,
        feasibility_sha256=actor_counterfactual_wrench_feasibility_sha256,
        allow_start=(
            allow_resume_actor_counterfactual_wrench_distillation_start
        ),
        is_resume=resume_from is not None,
    )
    if (
        actor_counterfactual_wrench_distillation
        and counterfactual_wrench_upgrade != frozen_controller_residual_upgrade
    ):
        raise ValueError(
            "counterfactual and leg-residual upgrades must start together"
        )
    demonstration_replay_threshold = (
        resolve_demonstration_replay_resume_setting(
            resumed_hparams,
            is_resume=resume_from is not None,
            requested=demonstration_replay_threshold,
            allow_change=allow_resume_demonstration_replay_change,
        )
    )
    if demonstration_replay_threshold is not None and not (
        env_variant.startswith("g1_tracking")
    ):
        raise ValueError(
            "demonstration replay requires a G1 tracking environment"
        )
    saved_jave_vg_weight = float(
        resumed_hparams.get("jave_vg_weight", 0.0)
        if resumed_hparams
        else 0.0
    )
    saved_jave_enabled = bool(
        resumed_hparams.get(
            "jave_collect_transitions", saved_jave_vg_weight > 0.0
        )
        if resumed_hparams
        else False
    )
    if resume_from is not None:
        if jave_enabled and not saved_jave_enabled:
            if not allow_resume_jave_start:
                raise ValueError(
                    "starting JAVE from a non-JAVE checkpoint requires "
                    "allow_resume_jave_start=True"
                )
        elif jave_enabled != saved_jave_enabled:
            raise ValueError(
                "JAVE cannot be disabled when resuming a JAVE checkpoint"
            )
        elif jave_enabled:
            saved_jave_settings = (
                saved_jave_vg_weight,
                int(resumed_hparams.get("jave_vg_warmup_steps", 5_000)),
                tuple(resumed_hparams.get("jave_ldm_hidden", (256, 256))),
                float(resumed_hparams.get("jave_ldm_lr", 3e-4)),
                int(resumed_hparams.get("jave_ldm_iterations", 4)),
                int(resumed_hparams.get("jave_ldm_batch_size", 256)),
                int(resumed_hparams.get("jave_vg_batch_size", 256)),
                int(
                    resumed_hparams.get(
                        "jave_ldm_buffer_capacity", 100_000
                    )
                ),
                float(
                    resumed_hparams.get("jave_reward_feature_scale", 8.0)
                ),
            )
            requested_jave_settings = (
                float(jave_vg_weight),
                jave_vg_warmup_steps,
                jave_ldm_hidden,
                float(jave_ldm_lr),
                jave_ldm_iterations,
                jave_ldm_batch_size,
                jave_vg_batch_size,
                jave_ldm_buffer_capacity,
                float(jave_reward_feature_scale),
            )
            validate_jave_resume_settings(
                saved_jave_settings,
                requested_jave_settings,
                allow_weight_change=allow_resume_jave_weight_change,
            )
    if jave_enabled:
        if not env_variant.startswith("g1_tracking"):
            raise ValueError("JAVE augmentation requires a G1 tracking task")
        if ahac:
            raise ValueError("JAVE augmentation does not support AHAC")
        if demonstration_replay_threshold is not None:
            raise ValueError(
                "JAVE augmentation does not support demonstration replay"
            )
        if torso_wrench_assistance or actor_learned_torso_wrench:
            raise ValueError(
                "JAVE augmentation requires the unmodified environment reward"
            )
    jave_start_step = (
        int(resumed_hparams.get("jave_start_step", 0))
        if saved_jave_enabled and resumed_hparams
        else (resumed_step if jave_enabled and resume_from is not None else 0)
    )
    if resume_from:
        validate_actor_cagrad_configuration(
            actor_cagrad=actor_cagrad,
            alpha=actor_cagrad_alpha,
            iterations=actor_cagrad_iterations,
            adaptive_phase_sampling=adaptive_phase_sampling,
            actor_phase_robust_weighting=actor_phase_robust_weighting,
            env_variant=env_variant,
            actor_per_env_grad_clip=actor_per_env_grad_clip,
            gradient_accumulation_steps=gradient_accumulation_steps,
            actor_phase_bin_count=actor_phase_bin_count,
        )
        validate_preview_adapter_configuration(
            enabled=actor_preview_adapter,
            actor_reference_lookahead_steps=actor_reference_lookahead_steps,
            actor_cagrad=actor_cagrad,
            history_len=actor_history_len,
            source_actor_policy=source_actor_policy,
            initial_full_actor_policy=initial_full_actor_policy,
            env_variant=env_variant,
        )

        validate_residual_preview_adapter_configuration(
            enabled=actor_residual_preview_adapter,
            hidden_dim=actor_residual_preview_hidden,
            optimizer_name=actor_residual_preview_optimizer,
            linear_preview_enabled=actor_preview_adapter,
            actor_reference_lookahead_steps=(
                actor_reference_lookahead_steps
            ),
            actor_reference_preview_mode=actor_reference_preview_mode,
            actor_cagrad=(actor_cagrad or actor_learned_torso_wrench),
            history_len=actor_history_len,
            source_actor_policy=source_actor_policy,
            initial_full_actor_policy=initial_full_actor_policy,
            env_variant=env_variant,
        )

    resolve_ahac_resume_settings(
        requested={
            "ahac": ahac,
            "ahac_horizon_min": ahac_horizon_min,
            "ahac_horizon_max": ahac_horizon_max,
            "ahac_contact_threshold": ahac_contact_threshold,
            "ahac_dual_lr": ahac_dual_lr,
            "ahac_critic_max_iterations": ahac_critic_max_iterations,
            "ahac_critic_tolerance": ahac_critic_tolerance,
        },
        resumed_hparams=resumed_hparams,
        is_resume=resume_from is not None,
        allow_change=allow_resume_ahac_change,
    )
    actor_contact_topology_gradient_truncation = (
        resolve_contact_topology_truncation_resume_setting(
            requested=actor_contact_topology_gradient_truncation,
            resumed_hparams=resumed_hparams,
            is_resume=resume_from is not None,
            allow_change=(
                allow_resume_actor_contact_topology_gradient_truncation_change
            ),
        )
    )
    if (
        actor_contact_topology_gradient_truncation
        and not env_variant.startswith("g1_tracking")
    ):
        raise ValueError(
            "contact topology gradient truncation requires a G1 tracking task"
        )

    if resume_from is None:
        (
            actor_residual_preview_initial_adapter_path,
            actor_residual_preview_initial_adapter_sha256,
        ) = resolve_zero_head_feature_transfer_resume_setting(
            None,
            requested_path=(
                actor_residual_preview_initial_adapter_path
            ),
            requested_sha256=(
                actor_residual_preview_initial_adapter_sha256
            ),
            residual_adapter_enabled=actor_residual_preview_adapter,
            residual_adapter_upgrade=False,
            is_resume=False,
        )

    (
        actor_recovery_teacher_dataset_path,
        actor_recovery_teacher_dataset_sha256,
        actor_recovery_teacher_gradient_ratio,
    ) = resolve_recovery_teacher_resume_settings(
        requested_path=actor_recovery_teacher_dataset_path,
        requested_sha256=actor_recovery_teacher_dataset_sha256,
        requested_ratio=actor_recovery_teacher_gradient_ratio,
        resumed_hparams=resumed_hparams,
        is_resume=resume_from is not None,
        allow_change=allow_resume_actor_recovery_teacher_change,
    )
    recovery_teacher_enabled = actor_recovery_teacher_dataset_path is not None

    if actor_state_gated_recovery_support_path is not None:
        if actor_state_gated_recovery_support_sha256 is None:
            raise ValueError("state-gated recovery support SHA-256 is required")
        recovery_support, recovery_support_report = (
            load_recovery_support_artifact(
                actor_state_gated_recovery_support_path,
                expected_sha256=actor_state_gated_recovery_support_sha256,
            )
        )

    validate_torso_wrench_assistance_configuration(
        enabled=torso_wrench_assistance,
        start_step=torso_wrench_assistance_start_step,
        end_step=torso_wrench_assistance_end_step,
        zero_fraction=torso_wrench_assistance_zero_fraction,
        env_variant=env_variant,
    )
    validate_assistance_conditioning_configuration(
        assistance_enabled=torso_wrench_assistance,
        continuous=torso_wrench_assistance_continuous,
        conditioning=actor_torso_wrench_assistance_conditioning,
        observed=actor_observe_torso_wrench_assistance,
        residual_adapter_enabled=actor_residual_preview_adapter,
    )

    if actor_reference_lookahead_steps and not env_variant.startswith(
        "g1_tracking"
    ):
        raise ValueError(
            "future reference observations require a G1 tracking task"
        )

    action_noise_schedule_steps = resolve_action_noise_schedule_steps(
        total_steps=total_steps,
        resumed_step=resumed_step,
        resumed_hparams=resumed_hparams,
        requested_schedule_steps=action_noise_schedule_steps,
        allow_resume_action_noise_change=allow_resume_action_noise_change,
    )

    # Compute curriculum defaults relative to total_steps
    if curriculum_grace is None:
        curriculum_grace = total_steps // 10  # 10% grace at difficulty=0
    if curriculum_steps is None:
        curriculum_steps = int(total_steps * 0.8)  # ramp over 80%

    _curriculum_steps = max(curriculum_steps, 1)  # avoid division by zero

    _curriculum_grace_jax = jp.array(curriculum_grace, dtype=jp.int32)
    _curriculum_steps_jax = jp.array(_curriculum_steps, dtype=jp.float32)

    g1_environment_kwargs = {}
    if env_variant.startswith("g1_tracking"):
        g1_environment_kwargs.update(
            {
                "effort_limit_scale": effort_limit_scale,
                "termination_margin_weight": termination_margin_weight,
                "tracking_velocity_kernel": tracking_velocity_kernel,
                "tracking_anchor_position_kernel": (
                    tracking_anchor_position_kernel
                ),
                "tracking_torso_orientation_weight": (
                    tracking_torso_orientation_weight
                ),
                "tracking_root_velocity_weight": tracking_root_velocity_weight,
                "jave_enabled": jave_enabled,
                "determinism_probe": determinism_probe_output is not None,
                "reference_reset_noise_scale": reference_reset_noise_scale,
                "reference_root_reset_noise_multiplier": (
                    reference_root_reset_noise_multiplier
                ),
                "reference_root_reset_noise_probability": (
                    reference_root_reset_noise_probability
                ),
                "reference_residual_control": reference_residual_control,
                "reference_residual_scale": reference_residual_scale,
                "domain_randomization": domain_randomization,
                "actor_observation_noise": actor_observation_noise,
                "actor_observe_motion_anchor_position": (
                    actor_observe_motion_anchor_position
                ),
                "actor_reference_lookahead_steps": (
                    actor_reference_lookahead_steps
                ),
                "actor_reference_preview_mode": (
                    actor_reference_preview_mode
                ),
                "solver_iterations": solver_iterations,
                "solver_ls_iterations": solver_ls_iterations,
                "carried_reset_bank_path": carried_reset_bank_path,
                "carried_reset_probability": carried_reset_probability,
                "carried_reset_bank_start": carried_reset_bank_start,
                "adaptive_phase_sampling": adaptive_phase_sampling,
                "adaptive_phase_uniform_ratio": (
                    adaptive_phase_uniform_ratio
                ),
            }
        )
        if reference_path is not None:
            g1_environment_kwargs["reference_path"] = str(reference_path)
        if reference_stride is not None:
            g1_environment_kwargs["reference_stride"] = reference_stride

    env = Go2Env(
        variant=env_variant,
        xml_path=xml_path,
        action_scale=action_scale,
        cmd_vel_x_range=cmd_vel_x_range,
        cmd_vel_y_range=cmd_vel_y_range,
        cmd_yaw_rate_range=cmd_yaw_rate_range,
        cmd_zero_prob=cmd_zero_prob,
        cmd_ctrl_interval_range=cmd_ctrl_interval_range,
        friction_range=friction_range,
        mass_range=mass_range,
        kp_range=kp_range,
        kd_range=kd_range,
        com_offset_range=com_offset_range,
        terrain_flat_prob=terrain_flat_prob,
        terrain_slope_max=terrain_slope_max if terrain else 0.0,
        max_episode_length=max_episode_length,
        actor_history_len=actor_history_len,
        **g1_environment_kwargs,
    )
    action_noise_std_start, action_noise_std_end = (
        resolve_action_noise_resume_settings(
            resumed_hparams,
            is_resume=resume_from is not None,
            requested_start=action_noise_std_start,
            requested_end=action_noise_std_end,
            allow_change=allow_resume_action_noise_change,
            action_dim=env.action_dim,
            actor_joint_names=getattr(env, "actor_joint_names", ()),
        )
    )
    reference_hparams = {}
    if env_variant.startswith("g1_tracking"):
        max_episode_length = env.reference_transitions
        reference_hparams = reference_hparams_for_env(env)
    counterfactual_feasibility = None
    counterfactual_teacher_params = None
    counterfactual_target_rms = None
    counterfactual_leg_indices = None
    counterfactual_nonleg_mask = None
    if actor_counterfactual_wrench_distillation:
        if (
            not actor_frozen_controller_residual
            or not actor_cagrad
            or actor_phase_bin_count != 5
            or gradient_accumulation_steps != 2
            or actor_residual_preview_adapter is False
            or torso_wrench_assistance
            or actor_learned_torso_wrench
            or demonstration_replay_threshold is not None
            or domain_randomization
            or actor_observation_noise
            or not env_variant.startswith("g1_tracking")
        ):
            raise ValueError(
                "counterfactual distillation requires the exact zero-wrench "
                "E026 five-phase CAGrad treatment"
            )
        if actor_counterfactual_wrench_teacher_path is None:
            raise ValueError("counterfactual teacher checkpoint is required")
        if actor_counterfactual_wrench_feasibility_path is None:
            raise ValueError("counterfactual feasibility artifact is required")
        counterfactual_feasibility = load_counterfactual_feasibility(
            actor_counterfactual_wrench_feasibility_path,
            expected_sha256=(
                actor_counterfactual_wrench_feasibility_sha256
            ),
        )
        teacher_path = Path(
            actor_counterfactual_wrench_teacher_path
        ).expanduser().resolve()
        if (
            not teacher_path.is_file()
            or _sha256_file(teacher_path)
            != actor_counterfactual_wrench_teacher_sha256
            or actor_counterfactual_wrench_teacher_sha256
            != counterfactual_feasibility.teacher_checkpoint_sha256
        ):
            raise ValueError("counterfactual teacher checkpoint hash mismatch")
        teacher_state, teacher_hparams, teacher_step = load_checkpoint(
            str(teacher_path)
        )
        if (
            not isinstance(teacher_state.actor_params, FrozenControllerWrenchParams)
            or not isinstance(
                teacher_state.actor_params.controller,
                FrozenPreviewResidualParams,
            )
            or teacher_step != 1_966_080
            or not isinstance(teacher_hparams, dict)
            or teacher_hparams.get("actor_learned_torso_wrench") is not True
        ):
            raise ValueError("counterfactual teacher checkpoint is invalid")
        counterfactual_teacher_params = jax.tree.map(
            jax.lax.stop_gradient, teacher_state.actor_params
        )
        resumed_counterfactual_parent = (
            resumed_state.actor_params.parent
            if isinstance(
                resumed_state.actor_params, FrozenControllerResidualParams
            )
            else resumed_state.actor_params
        ) if resumed_state is not None else None
        if (
            parameter_tree_sha256(counterfactual_teacher_params)
            != counterfactual_feasibility.teacher_tree_sha256
            or parameter_tree_sha256(counterfactual_teacher_params.controller)
            != counterfactual_feasibility.e026_tree_sha256
            or parameter_tree_sha256(counterfactual_teacher_params.wrench)
            != counterfactual_feasibility.wrench_tree_sha256
            or resumed_state is None
            or parameter_tree_sha256(resumed_counterfactual_parent)
            != counterfactual_feasibility.e026_tree_sha256
            or parameter_tree_sha256(resumed_state.normalizer)
            != parameter_tree_sha256(teacher_state.normalizer)
        ):
            raise ValueError("counterfactual frozen-tree provenance mismatch")
        counterfactual_target_rms = jp.asarray(
            counterfactual_feasibility.target_rms, dtype=jp.float64
        )
        counterfactual_leg_indices = resolve_leg_action_indices(
            env.actor_joint_names
        )
        counterfactual_nonleg_mask = jp.ones((env.action_dim,), dtype=bool)
        counterfactual_nonleg_mask = counterfactual_nonleg_mask.at[
            jp.asarray(counterfactual_leg_indices, dtype=jp.int32)
        ].set(False)
    torso_body_id = -1
    torso_slot = -1
    torso_wrench_parameters = None
    if torso_wrench_assistance and actor_learned_torso_wrench:
        raise ValueError(
            "analytic and learned torso wrench treatments are mutually exclusive"
        )
    if not isinstance(actor_learned_torso_wrench, bool):
        raise ValueError("actor_learned_torso_wrench must be boolean")
    if (
        isinstance(actor_learned_torso_wrench_hidden, bool)
        or not isinstance(actor_learned_torso_wrench_hidden, int)
        or actor_learned_torso_wrench_hidden < 1
    ):
        raise ValueError("actor_learned_torso_wrench_hidden must be positive")
    if (
        not math.isfinite(actor_learned_torso_wrench_scale)
        or not 0.0 <= actor_learned_torso_wrench_scale <= 1.0
    ):
        raise ValueError("actor_learned_torso_wrench_scale must be in [0, 1]")
    if actor_learned_torso_wrench_scale_end is None:
        actor_learned_torso_wrench_scale_end = actor_learned_torso_wrench_scale
    if (
        isinstance(actor_learned_torso_wrench_scale_end, bool)
        or not isinstance(actor_learned_torso_wrench_scale_end, (int, float))
        or not math.isfinite(actor_learned_torso_wrench_scale_end)
        or not 0.0 <= actor_learned_torso_wrench_scale_end <= 1.0
    ):
        raise ValueError("actor_learned_torso_wrench_scale_end must be in [0, 1]")
    if (
        isinstance(actor_learned_torso_wrench_scale_start_step, bool)
        or not isinstance(actor_learned_torso_wrench_scale_start_step, int)
        or actor_learned_torso_wrench_scale_start_step < 0
    ):
        raise ValueError("learned wrench scale start step must be non-negative")
    if (
        isinstance(actor_learned_torso_wrench_scale_end_step, bool)
        or not isinstance(actor_learned_torso_wrench_scale_end_step, int)
        or actor_learned_torso_wrench_scale_end_step
        <= actor_learned_torso_wrench_scale_start_step
    ):
        raise ValueError("learned wrench scale end step must exceed start step")
    for name, value in (
        (
            "actor_learned_torso_wrench_condition_on_scale",
            actor_learned_torso_wrench_condition_on_scale,
        ),
        (
            "actor_learned_torso_wrench_train_controller",
            actor_learned_torso_wrench_train_controller,
        ),
        (
            "allow_resume_actor_learned_torso_wrench_change",
            allow_resume_actor_learned_torso_wrench_change,
        ),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    if (
        not math.isfinite(actor_learned_torso_wrench_penalty)
        or actor_learned_torso_wrench_penalty < 0.0
    ):
        raise ValueError("actor_learned_torso_wrench_penalty must be non-negative")
    if not isinstance(allow_resume_actor_learned_torso_wrench_start, bool):
        raise ValueError(
            "allow_resume_actor_learned_torso_wrench_start must be boolean"
        )
    resumed_learned_wrench = bool(
        (resumed_hparams or {}).get("actor_learned_torso_wrench", False)
    )
    learned_wrench_resume_settings = (
        float(actor_learned_torso_wrench_scale),
        float(actor_learned_torso_wrench_scale_end),
        actor_learned_torso_wrench_scale_start_step,
        actor_learned_torso_wrench_scale_end_step,
        actor_learned_torso_wrench_condition_on_scale,
        actor_learned_torso_wrench_train_controller,
    )
    if resumed_learned_wrench:
        saved_scale = float(resumed_hparams["actor_learned_torso_wrench_scale"])
        saved_settings = (
            saved_scale,
            float(
                resumed_hparams.get(
                    "actor_learned_torso_wrench_scale_end", saved_scale
                )
            ),
            int(
                resumed_hparams.get(
                    "actor_learned_torso_wrench_scale_start_step", 0
                )
            ),
            int(
                resumed_hparams.get(
                    "actor_learned_torso_wrench_scale_end_step", 1
                )
            ),
            bool(
                resumed_hparams.get(
                    "actor_learned_torso_wrench_condition_on_scale", False
                )
            ),
            bool(
                resumed_hparams.get(
                    "actor_learned_torso_wrench_train_controller", False
                )
            ),
        )
        if (
            learned_wrench_resume_settings != saved_settings
            and not allow_resume_actor_learned_torso_wrench_change
        ):
            raise ValueError(
                "learned wrench continuation settings differ from the checkpoint "
                "without explicit resume authority"
            )
    if actor_learned_torso_wrench and not actor_residual_preview_adapter:
        raise ValueError(
            "learned torso wrench requires the frozen residual preview controller"
        )
    if actor_learned_torso_wrench and resume_from is None:
        raise ValueError("learned torso wrench currently requires a resumed controller")
    if (
        actor_learned_torso_wrench != resumed_learned_wrench
        and not allow_resume_actor_learned_torso_wrench_start
    ):
        raise ValueError(
            "learned torso wrench setting differs from the checkpoint without "
            "explicit resume authority"
        )
    if (
        torso_wrench_assistance
        or actor_learned_torso_wrench
        or actor_counterfactual_wrench_distillation
    ):
        torso_body_id, torso_wrench_parameters = (
            torso_wrench_parameters_from_environment(env)
        )
        try:
            torso_slot = env.body_ids.index(torso_body_id)
        except ValueError as error:
            raise ValueError(
                "G1 reference body slots do not include torso_link"
            ) from error
        if torso_slot != 7:
            raise ValueError("torso_link must occupy reference body slot 7")
    actor_norm = Normalizer(env.actor_frame_obs_dim)
    critic_norm = Normalizer(env.critic_obs_dim)

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_name = "ahac" if ahac else "shac"
    save_dir = f"training_runs/{algorithm_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Algorithm: {algorithm_name.upper()}, Save dir: {save_dir}")
    print(
        f"Domain randomization: action_noise={action_noise_std_start}->{action_noise_std_end}, "
        f"friction={friction_range}, mass={mass_range}, "
        f"kp={kp_range}, kd={kd_range}, "
        f"com_offset=+/-{com_offset_range}, "
        f"velocity_push={push_velocity_range} every {push_interval_s}s"
    )
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum={curriculum_steps} steps, "
        f"terrain={'ON' if terrain else 'OFF'}, "
        f"terrain_flat_prob={terrain_flat_prob}, "
        f"terrain_slope_max={terrain_slope_max} deg, "
        f"terrain_bump_std={terrain_bump_std if terrain else 0.0}"
    )

    # Initialize random keys
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Initialize networks
    squash_actor_actions = getattr(env, "squash_actor_actions", True)
    squash_actor_mean = getattr(
        env, "squash_actor_mean", squash_actor_actions
    )
    clip_sampled_actor_actions = getattr(
        env, "clip_sampled_actor_actions", squash_actor_actions
    )
    actor = Actor(
        env.action_dim,
        hidden=actor_hidden,
        squash=squash_actor_mean,
        layer_norm=actor_layer_norm,
        zero_output=actor_zero_output,
    )
    residual_preview_actor = PreviewResidualAdapter(
        action_dim=env.action_dim,
        hidden_dim=actor_residual_preview_hidden,
    )
    frozen_controller_residual_actor = PreviewResidualAdapter(
        action_dim=(12 if actor_counterfactual_wrench_distillation else env.action_dim),
        hidden_dim=actor_frozen_controller_residual_hidden,
    )
    learned_torso_wrench_actor = LearnedTorsoWrenchHead(
        hidden_dim=actor_learned_torso_wrench_hidden,
        condition_on_scale=actor_learned_torso_wrench_condition_on_scale,
    )
    critic = DoubleCritic() if ahac else Critic()

    actor_dummy = jp.zeros((1, env.actor_obs_dim), dtype=jp.float32)
    critic_dummy = jp.zeros((1, env.critic_obs_dim), dtype=jp.float32)
    actor_params = (
        initial_full_actor_policy
        if initial_full_actor_policy is not None
        else actor.init(k1, actor_dummy)
    )
    if expected_actor_obs_dim is not None:
        try:
            actor_first_layer_input_dim = int(
                actor_params["params"]["Dense_0"]["kernel"].shape[0]
            )
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError(
                "actor parameters do not expose a standard first input layer"
            ) from error
        validate_actor_input_contract(
            expected_input_dim=expected_actor_obs_dim,
            environment_input_dim=int(env.actor_obs_dim),
            first_layer_input_dim=actor_first_layer_input_dim,
        )
    migration_report = None
    residual_adapter_report = None
    residual_muon_report = None
    conditioning_report = None
    preview_adapter_mask = None
    preview_legacy_frame_dim = 0
    preview_trainable_parameter_count = 0
    frozen_preview_treatment = bool(
        actor_preview_adapter
        or actor_residual_preview_adapter
        or actor_frozen_controller_residual
    )
    residual_muon_treatment = bool(
        actor_residual_preview_adapter
        and actor_residual_preview_optimizer == "muon"
    )
    recovery_teacher_batch = None
    if recovery_teacher_enabled:
        if not actor_residual_preview_adapter:
            raise ValueError(
                "recovery teacher requires the residual preview adapter"
            )
        if not actor_cagrad or actor_phase_bin_count != 5:
            raise ValueError(
                "recovery teacher requires five-bin actor CAGrad"
            )
        if gradient_accumulation_steps != 2:
            raise ValueError(
                "recovery teacher requires two actor gradient shards"
            )
        if actor_recovery_teacher_gradient_ratio != 0.5:
            raise ValueError(
                "recovery teacher gradient ratio must equal 0.5"
            )
        recovery_teacher_batch = load_recovery_teacher_batch(
            actor_recovery_teacher_dataset_path,
            expected_sha256=actor_recovery_teacher_dataset_sha256,
        )
        if recovery_teacher_batch.actor_obs.shape[1] != env.actor_obs_dim:
            raise ValueError(
                "recovery teacher observations do not match the actor"
            )
    if actor_preview_adapter:
        preview_legacy_frame_dim = (
            env.actor_frame_obs_dim - env.actor_future_reference_dim
        )
        if initial_full_actor_policy is not None:
            if not isinstance(actor_params, RmrPolicy):
                raise ValueError(
                    "full actor preview adapter requires an RmrPolicy"
                )
            reference_phases = jp.arange(
                env.reference_length, dtype=jp.int32
            )
            preview_table = jax.vmap(env._future_reference_command)(
                reference_phases
            )
            preview_mean = jp.mean(preview_table, axis=0).astype(
                actor_params.mean.dtype
            )
            preview_std = jp.sqrt(
                jp.maximum(jp.var(preview_table, axis=0), 1e-8)
            ).astype(actor_params.std.dtype)
            parent_actor_params = actor_params
            actor_params = migrate_rmr_preview_policy(
                parent_actor_params,
                preview_mean=preview_mean,
                preview_std=preview_std,
            )
            migration_report = rmr_preview_migration_report(
                parent_actor_params, actor_params
            )
            preview_adapter_mask = build_rmr_preview_mask(
                actor_params,
                legacy_obs_dim=preview_legacy_frame_dim,
                treatment_obs_dim=env.actor_frame_obs_dim,
            )
            expected_preview_parameters = (
                env.actor_future_reference_dim
                * actor_params.weights[0].shape[0]
            )
        else:
            preview_adapter_mask = build_current_preview_mask(
                actor_params,
                history_len=actor_history_len,
                legacy_frame_dim=preview_legacy_frame_dim,
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            expected_preview_parameters = (
                env.actor_future_reference_dim
                * actor_params["params"]["Dense_0"]["kernel"].shape[1]
            )
        preview_trainable_parameter_count = sum(
            int(np.count_nonzero(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(preview_adapter_mask)
        )
        if preview_trainable_parameter_count != expected_preview_parameters:
            raise ValueError(
                "actor preview adapter trainable parameter count does not "
                "match its first-layer preview boundary"
            )
    critic_params = critic.init(k2, critic_dummy)
    target_critic_params = critic_params

    actor_normalizer = actor_norm.init()
    critic_normalizer = critic_norm.init()
    learned_dynamics_model = None
    ldm_params = None
    replay_buffer = None
    if jave_enabled:
        learned_dynamics_model = LearnedDynamicsModel(
            obs_dim=env.jave_obs_dim,
            hidden=jave_ldm_hidden,
        )
        jave_dummy = jp.zeros(
            (1, env.jave_obs_dim), dtype=jp.float32
        )
        action_dummy = jp.zeros((1, env.action_dim), dtype=jp.float32)
        ldm_params = learned_dynamics_model.init(
            jax.random.fold_in(k3, 0x4A415645),
            jave_dummy,
            action_dummy,
        )
        replay_buffer = init_replay_buffer(
            jave_ldm_buffer_capacity,
            env.jave_obs_dim,
            env.action_dim,
        )

    # Linear LR decay
    if use_lr_decay:
        total_iters = total_steps // steps_per_actor_update
        lr_floor = 0.62
        actor_schedule = optax.linear_schedule(
            init_value=actor_lr,
            end_value=actor_lr * lr_floor,
            transition_steps=total_iters,
        )

        critic_schedule = optax.linear_schedule(
            init_value=critic_lr,
            end_value=critic_lr * lr_floor,
            transition_steps=total_iters * critic_iterations,
        )
        print(
            f"LR decay: linear over {total_iters} iters, "
            f"actor {actor_lr:.1e} --> {actor_lr * lr_floor:.1e} ({total_iters} steps), "
            f"critic {critic_lr:.1e} --> {critic_lr * lr_floor:.1e} ({total_iters * critic_iterations} steps)"
        )
    else:
        actor_schedule = actor_lr
        critic_schedule = critic_lr

    # Initialize optimizers
    actor_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(actor_schedule))
    residual_muon_opt = None
    residual_adam_opt = None
    if actor_residual_preview_optimizer == "muon":
        residual_muon_opt, residual_adam_opt = (
            build_residual_muon_optimizers(actor_schedule)
        )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(critic_schedule)
    )
    ldm_optimizer = (
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(jave_ldm_lr),
        )
        if jave_enabled
        else None
    )

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    ldm_opt_state = (
        ldm_optimizer.init(ldm_params) if jave_enabled else None
    )

    # Initialize environments at difficulty=0 (flat ground)
    env_keys = jax.random.split(k3, effective_num_envs)
    env_state = jax.vmap(env.reset)(
        env_keys, jp.zeros(effective_num_envs)
    )
    actor_normalizer = canonicalize_normalizer_dtype(
        actor_normalizer, env_state.obs.dtype
    )
    critic_normalizer = canonicalize_normalizer_dtype(
        critic_normalizer, env_state.info["bootstrap_critic_obs"].dtype
    )

    recovery_teacher_actor_obs = None
    recovery_teacher_parent_action = None
    recovery_teacher_correction = None
    recovery_teacher_effective_action = None
    if recovery_teacher_batch is not None:
        recovery_teacher_actor_obs = jp.asarray(
            recovery_teacher_batch.actor_obs, dtype=jp.float64
        )
        recovery_teacher_parent_action = jp.asarray(
            recovery_teacher_batch.parent_action, dtype=jp.float32
        )
        recovery_teacher_correction = jp.asarray(
            recovery_teacher_batch.teacher_correction, dtype=jp.float32
        )
        recovery_teacher_effective_action = jp.asarray(
            recovery_teacher_batch.teacher_effective_action,
            dtype=jp.float32,
        )

    _push_interval_steps = max(int(round(push_interval_s / env.dt)), 1)
    _push_velocity_lo = jp.array(push_velocity_range[0], dtype=jp.float64)
    _push_velocity_hi = jp.array(push_velocity_range[1], dtype=jp.float64)
    _foot_body_ids = env._foot_body_ids
    _nominal_weight = env.nominal_total_mass * env.base_gravity_mag
    _terrain_bump_std = terrain_bump_std if terrain else 0.0
    policy_anchor_source_params = (
        load_policy_anchor_source(
            actor_policy_anchor_source_path,
            expected_sha256=actor_policy_anchor_source_sha256,
        )
        if actor_policy_anchor_source_path is not None
        else None
    )

    def actor_loss(
        actor_params,
        target_critic_params,
        actor_norm_state,
        critic_norm_state,
        env_state,
        randomization,
        current_noise_std,
        current_actor_bootstrap_scale,
        current_ahac_horizon,
        current_learned_wrench_scale,
    ):
        """Short-horizon actor objective with sampled perturbations."""
        (
            action_noise,
            velocity_pushes,
            terrain_bump_innovations,
            assistance_scale,
        ) = randomization

        def rollout_step(carry, inputs):
            state, foot_bump_ou = carry
            noise_t, velocity_push_t, terrain_bump_innov_t, rollout_index = inputs
            active = (
                rollout_index
                < jp.floor(current_ahac_horizon + 0.5).astype(jp.int32)
                if ahac
                else jp.asarray(True)
            )

            push_due = (state.info["step"] > 0) & (
                (state.info["step"] % _push_interval_steps) == 0
            )
            pushed_qvel = state.data.qvel.at[:2].set(velocity_push_t)
            state = state.replace(
                data=state.data.replace(
                    qvel=jp.where(push_due, pushed_qvel, state.data.qvel)
                )
            )

            foot_bump_ou, terrain_bump_forces = differentiated_ou_foot_forces(
                foot_bump_ou,
                terrain_bump_innov_t,
                jax.lax.stop_gradient(state.info["foot_normal_forces"]),
                difficulty=state.info["difficulty"],
                std=_terrain_bump_std,
                decay=terrain_bump_decay,
                robot_weight=_nominal_weight,
            )
            xfrc = state.data.xfrc_applied
            for i in range(4):
                xfrc = xfrc.at[_foot_body_ids[i], :3].add(terrain_bump_forces[i])
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

            if torso_wrench_assistance:
                torso_wrench, _, _ = compute_environment_torso_wrench(
                    env,
                    state,
                    torso_slot=torso_slot,
                    parameters=torso_wrench_parameters,
                    scale=assistance_scale,
                )
                xfrc = write_torso_wrench(
                    state.data.xfrc_applied,
                    torso_body_id=torso_body_id,
                    world_wrench=torso_wrench,
                )
                state = state.replace(
                    data=state.data.replace(xfrc_applied=xfrc)
                )

            # Actor sees noisy observations; critic/training targets keep raw obs.
            obs_rng, env_rng = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_rng})
            actor_obs = env._apply_obs_noise(state.obs, obs_rng)

            # Compute action with reparameterized noise
            obs_norm = env.normalize_actor_obs(
                actor_norm, actor_norm_state, actor_obs
            ).astype(jp.float32)
            controller_actor_params = (
                actor_params.controller
                if actor_learned_torso_wrench
                else actor_params
            )
            residual_logits = None
            if actor_frozen_controller_residual:
                def frozen_parent_apply(parent_params, observations):
                    if isinstance(parent_params, FrozenPreviewResidualParams):
                        return apply_frozen_preview_residual(
                            actor,
                            residual_preview_actor,
                            parent_params,
                            observations,
                            history_len=actor_history_len,
                            treatment_frame_dim=env.actor_frame_obs_dim,
                        )[0]
                    return apply_frozen_controller_residual(
                        frozen_parent_apply,
                        frozen_controller_residual_actor,
                        parent_params,
                        observations,
                        history_len=actor_history_len,
                        frame_dim=env.actor_frame_obs_dim,
                    )[0]

                action, parent_action, _residual_action = (
                    apply_frozen_controller_residual(
                        frozen_parent_apply,
                        frozen_controller_residual_actor,
                        controller_actor_params,
                        obs_norm,
                        history_len=actor_history_len,
                        frame_dim=env.actor_frame_obs_dim,
                        residual_action_indices=counterfactual_leg_indices,
                    )
                )
                action = action.astype(jp.float64)
                parent_action = parent_action.astype(jp.float64)
            elif actor_residual_preview_adapter:
                if recovery_support is not None:
                    (
                        action,
                        parent_action,
                        _residual_action,
                        recovery_gate,
                    ) = apply_state_gated_recovery(
                        actor,
                        residual_preview_actor,
                        controller_actor_params,
                        obs_norm,
                        state.info["phase"],
                        recovery_support,
                        history_len=actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                    )
                else:
                    action, parent_action, _residual_action = (
                        apply_frozen_preview_residual(
                            actor,
                            residual_preview_actor,
                            controller_actor_params,
                            obs_norm,
                            history_len=actor_history_len,
                            treatment_frame_dim=env.actor_frame_obs_dim,
                            assistance_scale=(
                                assistance_scale
                                if actor_observe_torso_wrench_assistance
                                else jp.zeros_like(assistance_scale)
                            )
                            if actor_torso_wrench_assistance_conditioning
                            else None,
                        )
                    )
                action = action.astype(jp.float64)
                parent_action = parent_action.astype(jp.float64)
            else:
                residual_logits = (
                    actor.apply(actor_params, obs_norm)
                    if initial_full_actor_policy is None
                    else None
                )
            if actor_preview_adapter:
                if initial_full_actor_policy is not None:
                    parent_action = apply_trainable_rmr_policy(
                        initial_full_actor_policy,
                        actor_obs[..., :preview_legacy_frame_dim],
                    ).astype(jp.float64)
                else:
                    parent_obs = zero_current_preview(
                        obs_norm,
                        history_len=actor_history_len,
                        legacy_frame_dim=preview_legacy_frame_dim,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                    )
                    parent_action = actor.apply(
                        actor_params, parent_obs
                    ).astype(jp.float64)
            if actor_frozen_controller_residual or actor_residual_preview_adapter:
                pass
            elif initial_full_actor_policy is not None:
                action = apply_trainable_rmr_policy(
                    actor_params, actor_obs
                ).astype(jp.float64)
            elif source_actor_policy is None:
                action = residual_logits.astype(jp.float64)
            else:
                action = compose_bounded_rmr_residual(
                    source_actor_policy,
                    actor_obs,
                    residual_logits,
                    action_scale=residual_action_scale,
                    differentiate_source_feedback=(
                        differentiate_source_feedback
                    ),
                ).astype(jp.float64)

            if actor_policy_anchor_weight > 0.0:
                anchor_obs = jax.lax.stop_gradient(actor_obs)
                if policy_anchor_source_params is not None:
                    anchor_parent_action, _, _ = apply_frozen_preview_residual(
                        actor,
                        residual_preview_actor,
                        policy_anchor_source_params,
                        obs_norm,
                        history_len=actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                        assistance_scale=(
                            assistance_scale
                            if actor_observe_torso_wrench_assistance
                            else jp.zeros_like(assistance_scale)
                        )
                        if actor_torso_wrench_assistance_conditioning
                        else None,
                    )
                    anchor_candidate_action = action
                elif actor_residual_preview_adapter or actor_preview_adapter:
                    (
                        anchor_candidate_action,
                        anchor_parent_action,
                    ) = (action, parent_action)
                else:
                    anchor_candidate_action = apply_trainable_rmr_policy(
                        actor_params, anchor_obs
                    ).astype(jp.float64)
                    anchor_parent_action = apply_trainable_rmr_policy(
                        initial_full_actor_policy, anchor_obs
                    ).astype(jp.float64)
                actor_policy_anchor_squared_error = jp.mean(
                    jp.square(
                        anchor_candidate_action
                        - jax.lax.stop_gradient(anchor_parent_action)
                    )
                )
                actor_policy_anchor_step_penalty = policy_anchor_penalty(
                    anchor_candidate_action,
                    anchor_parent_action,
                    weight=actor_policy_anchor_weight,
                )
            else:
                actor_policy_anchor_squared_error = jp.asarray(
                    0.0, dtype=jp.float64
                )
                actor_policy_anchor_step_penalty = jp.asarray(
                    0.0, dtype=jp.float64
                )

            # Reparameterized action noise
            noisy_action = action + noise_t.astype(jp.float64) * current_noise_std
            if clip_sampled_actor_actions:
                noisy_action = jp.clip(noisy_action, -1.0, 1.0)

            counterfactual_teacher_next_state = None
            counterfactual_before_features = None
            teacher_world_wrench = jp.zeros((6,), dtype=state.data.qpos.dtype)
            if actor_counterfactual_wrench_distillation:
                state = state.replace(
                    data=state.data.replace(
                        xfrc_applied=jp.zeros_like(state.data.xfrc_applied)
                    )
                )
                counterfactual_before_features = jp.concatenate(
                    (
                        state.data.qvel[:6],
                        mjx_centroidal_momentum(
                            env.mjx_model,
                            state.data,
                            env.root_body_id,
                            env.nominal_total_mass,
                        ),
                    )
                )
                teacher_frame = current_treatment_frame(
                    obs_norm,
                    history_len=actor_history_len,
                    treatment_frame_dim=env.actor_frame_obs_dim,
                )
                teacher_normalized_wrench = apply_learned_torso_wrench(
                    learned_torso_wrench_actor,
                    counterfactual_teacher_params,
                    teacher_frame,
                )
                teacher_world_wrench = normalized_yaw_wrench_to_world(
                    teacher_normalized_wrench.astype(state.data.qpos.dtype),
                    root_quaternion=state.data.qpos[3:7],
                    force_cap=torso_wrench_parameters.force_cap,
                    torque_cap=torso_wrench_parameters.torque_cap,
                    scale=1.0,
                )
                teacher_xfrc = write_torso_wrench(
                    jp.zeros_like(state.data.xfrc_applied),
                    torso_body_id=torso_body_id,
                    world_wrench=teacher_world_wrench,
                )
                teacher_state = state.replace(
                    data=state.data.replace(xfrc_applied=teacher_xfrc)
                )
                teacher_noisy_action = noisy_action - _residual_action
                teacher_next_state = env.step(
                    teacher_state, teacher_noisy_action
                )
                counterfactual_teacher_next_state = jax.lax.stop_gradient(
                    teacher_next_state
                )

            learned_torso_wrench = jp.zeros((6,), dtype=state.data.qpos.dtype)
            learned_torso_wrench_normalized = jp.zeros(
                (6,), dtype=jp.float32
            )
            learned_torso_wrench_step_penalty = jp.asarray(
                0.0, dtype=state.reward.dtype
            )
            if actor_learned_torso_wrench:
                wrench_frame = current_treatment_frame(
                    obs_norm,
                    history_len=actor_history_len,
                    treatment_frame_dim=env.actor_frame_obs_dim,
                )
                learned_torso_wrench_normalized = apply_learned_torso_wrench(
                    learned_torso_wrench_actor,
                    actor_params,
                    wrench_frame,
                    assistance_scale=(
                        current_learned_wrench_scale
                        if actor_learned_torso_wrench_condition_on_scale
                        else None
                    ),
                )
                learned_torso_wrench = normalized_yaw_wrench_to_world(
                    learned_torso_wrench_normalized.astype(state.data.qpos.dtype),
                    root_quaternion=state.data.qpos[3:7],
                    force_cap=torso_wrench_parameters.force_cap,
                    torque_cap=torso_wrench_parameters.torque_cap,
                    scale=current_learned_wrench_scale,
                )
                xfrc = write_torso_wrench(
                    state.data.xfrc_applied,
                    torso_body_id=torso_body_id,
                    world_wrench=learned_torso_wrench,
                )
                state = state.replace(
                    data=state.data.replace(xfrc_applied=xfrc)
                )
                learned_torso_wrench_step_penalty = (
                    actor_learned_torso_wrench_penalty
                    * jp.mean(jp.square(learned_torso_wrench_normalized))
                ).astype(state.reward.dtype)

            if adaptive_phase_sampling or frozen_preview_treatment:
                transition_phase = transition_phase_before_reset(
                    state.info["phase"],
                    reference_stride=env.reference_stride,
                    reference_length=env.reference_length,
                )
            candidate_unreplayed_state = env.step(state, noisy_action)
            if determinism_probe_output is not None:
                determinism_mjx_substep_fingerprint = (
                    candidate_unreplayed_state.info[
                        "determinism_mjx_substep_fingerprint"
                    ]
                )
                determinism_mjx_substep_component_fingerprints = {
                    name: candidate_unreplayed_state.info[
                        f"determinism_mjx_substep_{name}_fingerprint"
                    ]
                    for name in FIRST_MJX_SUBSTEP_COMPONENTS
                }
                determinism_mjx_substep_field_fingerprints = {
                    name: candidate_unreplayed_state.info[
                        "determinism_mjx_substep_"
                        f"field_{name}_fingerprint"
                    ]
                    for name in FIRST_MJX_SUBSTEP_FIELDS
                }
                determinism_mjx_control_step_fingerprint = (
                    candidate_unreplayed_state.info[
                        "determinism_mjx_control_step_fingerprint"
                    ]
                )
                candidate_unreplayed_state = (
                    candidate_unreplayed_state.replace(
                        info={
                            key: value
                            for key, value in (
                                candidate_unreplayed_state.info.items()
                            )
                            if key
                            not in {
                                "determinism_mjx_substep_fingerprint",
                                "determinism_mjx_control_step_fingerprint",
                                *(
                                    f"determinism_mjx_substep_{name}_fingerprint"
                                    for name in FIRST_MJX_SUBSTEP_COMPONENTS
                                ),
                                *(
                                    "determinism_mjx_substep_"
                                    f"field_{name}_fingerprint"
                                    for name in FIRST_MJX_SUBSTEP_FIELDS
                                ),
                            }
                        }
                    )
                )
                prepared_action = env._prepare_action(noisy_action)
                position_target = env.position_target(
                    state, prepared_action, prepared=True
                )
                determinism_actor_step_fingerprint = tree_bit_fingerprint(
                    (
                        actor_obs,
                        obs_norm,
                        action,
                        noisy_action,
                        prepared_action,
                        position_target,
                    )
                )
                env_step_values = (
                    candidate_unreplayed_state.data.qpos,
                    candidate_unreplayed_state.data.qvel,
                    candidate_unreplayed_state.data.qacc,
                    candidate_unreplayed_state.data.qacc_warmstart,
                    candidate_unreplayed_state.data.qfrc_constraint,
                    candidate_unreplayed_state.data._impl.efc_force,
                    candidate_unreplayed_state.data._impl.contact,
                    candidate_unreplayed_state.done,
                    candidate_unreplayed_state.reward,
                    candidate_unreplayed_state.info["bootstrap_obs"],
                    candidate_unreplayed_state.info[
                        "bootstrap_critic_obs"
                    ],
                    candidate_unreplayed_state.info[
                        "transition_contact_stiffness"
                    ],
                    candidate_unreplayed_state.info[
                        "transition_contact_topology_event"
                    ],
                )
                if jave_enabled:
                    env_step_values = (
                        *env_step_values,
                        candidate_unreplayed_state.info[
                            "bootstrap_jave_obs"
                        ],
                    )
                determinism_env_step_fingerprint = tree_bit_fingerprint(
                    env_step_values
                )
            counterfactual_step_loss = jp.asarray(
                0.0, dtype=candidate_unreplayed_state.reward.dtype
            )
            counterfactual_step_valid = jp.asarray(True)
            counterfactual_step_telemetry = {
                "base_linear_loss": counterfactual_step_loss,
                "base_angular_loss": counterfactual_step_loss,
                "centroidal_linear_loss": counterfactual_step_loss,
                "centroidal_angular_loss": counterfactual_step_loss,
                "cosine": counterfactual_step_loss,
                "student_rms": counterfactual_step_loss,
                "teacher_rms": counterfactual_step_loss,
                "normalized_error_rms": counterfactual_step_loss,
                "valid": jp.asarray(1.0),
            }
            if actor_counterfactual_wrench_distillation:
                student_features = jp.concatenate(
                    (
                        candidate_unreplayed_state.data.qvel[:6],
                        mjx_centroidal_momentum(
                            env.mjx_model,
                            candidate_unreplayed_state.data,
                            env.root_body_id,
                            env.nominal_total_mass,
                        ),
                    )
                )
                teacher_features = jp.concatenate(
                    (
                        counterfactual_teacher_next_state.data.qvel[:6],
                        mjx_centroidal_momentum(
                            env.mjx_model,
                            counterfactual_teacher_next_state.data,
                            env.root_body_id,
                            env.nominal_total_mass,
                        ),
                    )
                )
                student_change = counterfactual_target_change(
                    counterfactual_before_features, student_features
                )
                teacher_change = counterfactual_target_change(
                    counterfactual_before_features, teacher_features
                )
                (
                    counterfactual_step_loss,
                    counterfactual_step_telemetry,
                ) = counterfactual_transition_loss(
                    student_change,
                    teacher_change,
                    counterfactual_target_rms,
                )
                counterfactual_done_match = (
                    candidate_unreplayed_state.done
                    == counterfactual_teacher_next_state.done
                )
                counterfactual_step_integrity = (
                    jp.asarray(
                        counterfactual_step_telemetry["valid"], dtype=bool
                    )
                )
                counterfactual_step_valid = (
                    (candidate_unreplayed_state.done == 0)
                    & (counterfactual_teacher_next_state.done == 0)
                    & counterfactual_step_integrity
                )
            if demonstration_replay_threshold is not None:
                candidate_next_state, demonstration_replay = (
                    apply_demonstration_replay(
                        env,
                        candidate_unreplayed_state,
                        threshold=demonstration_replay_threshold,
                    )
                )
            else:
                candidate_next_state = candidate_unreplayed_state
                demonstration_replay = jp.array(False)
            contact_topology_event = contact_topology_event_from_info(
                candidate_unreplayed_state.info,
                enabled=actor_contact_topology_gradient_truncation,
            )
            gradient_next_state = contact_gradient_barrier(
                candidate_next_state,
                contact_topology_event,
                enabled=actor_contact_topology_gradient_truncation,
            )
            candidate_foot_bump_ou = jp.where(
                candidate_unreplayed_state.done,
                jp.zeros((4, 3)),
                foot_bump_ou,
            )
            next_state = (
                select_active_tree(state, gradient_next_state, active)
                if ahac
                else gradient_next_state
            )
            foot_bump_ou = jp.where(
                active, candidate_foot_bump_ou, foot_bump_ou
            )

            transition = {
                "reward": jp.where(
                    active,
                    gradient_next_state.reward
                    - learned_torso_wrench_step_penalty,
                    0.0,
                ),
                "done": jp.where(
                    active, candidate_unreplayed_state.done, False
                ),
                "terminal": jp.where(
                    active,
                    candidate_unreplayed_state.info["terminal"],
                    False,
                ),
                "actor_obs": state.obs,
                "critic_obs": env._get_critic_obs(state.data, state.info),
                "bootstrap_critic_obs": gradient_next_state.info[
                    "bootstrap_critic_obs"
                ],
                "vel_x": candidate_unreplayed_state.metrics["vel_x"],
                "vel_y": candidate_unreplayed_state.metrics["vel_y"],
                "yaw_rate": candidate_unreplayed_state.metrics["yaw_rate"],
                "cmd_x": candidate_unreplayed_state.metrics["cmd_x"],
                "cmd_y": candidate_unreplayed_state.metrics["cmd_y"],
                "cmd_yaw": candidate_unreplayed_state.metrics["cmd_yaw"],
                "height": candidate_unreplayed_state.metrics["height"],
                "tilt": candidate_unreplayed_state.metrics["tilt"],
                "foot_normal_FL": candidate_unreplayed_state.metrics[
                    "foot_normal_FL"
                ],
                "foot_normal_FR": candidate_unreplayed_state.metrics[
                    "foot_normal_FR"
                ],
                "foot_normal_RL": candidate_unreplayed_state.metrics[
                    "foot_normal_RL"
                ],
                "foot_normal_RR": candidate_unreplayed_state.metrics[
                    "foot_normal_RR"
                ],
                "demonstration_replay": jp.where(
                    active, demonstration_replay, False
                ),
                "actor_policy_anchor_squared_error": (
                    jp.where(active, actor_policy_anchor_squared_error, 0.0)
                ),
                "actor_policy_anchor_penalty": (
                    jp.where(active, actor_policy_anchor_step_penalty, 0.0)
                ),
                "ahac_active": active,
                "ahac_contact_stiffness": jp.where(
                    active,
                    candidate_unreplayed_state.info[
                        "transition_contact_stiffness"
                    ],
                    0.0,
                ),
                "contact_topology_event": jp.where(
                    active, contact_topology_event, False
                ),
            }
            if determinism_probe_output is not None:
                transition.update(
                    {
                        "determinism_actor_step_fingerprint": (
                            determinism_actor_step_fingerprint
                        ),
                        "determinism_mjx_substep_fingerprint": (
                            determinism_mjx_substep_fingerprint
                        ),
                        "determinism_mjx_control_step_fingerprint": (
                            determinism_mjx_control_step_fingerprint
                        ),
                        "determinism_env_step_fingerprint": (
                            determinism_env_step_fingerprint
                        ),
                        **{
                            f"determinism_mjx_substep_{name}_fingerprint": (
                                fingerprint
                            )
                            for name, fingerprint in (
                                determinism_mjx_substep_component_fingerprints.items()
                            )
                        },
                        **{
                            "determinism_mjx_substep_"
                            f"field_{name}_fingerprint": fingerprint
                            for name, fingerprint in (
                                determinism_mjx_substep_field_fingerprints.items()
                            )
                        },
                    }
                )
            if jave_enabled:
                transition.update(
                    {
                        "jave_obs": env._get_jave_obs(
                            state.data, state.info
                        ),
                        "bootstrap_jave_obs": (
                            candidate_unreplayed_state.info[
                                "bootstrap_jave_obs"
                            ]
                        ),
                        "jave_action": noisy_action,
                    }
                )
            if adaptive_phase_sampling:
                transition["transition_phase"] = transition_phase
            if frozen_preview_treatment:
                transition.update(
                    {
                        "candidate_action": action,
                        "parent_action": parent_action,
                        "transition_phase": transition_phase,
                    }
                )
            if actor_centroidal_propulsion:
                transition.update(
                    {
                        "centroidal_momentum": mjx_centroidal_momentum(
                            env.mjx_model,
                            state.data,
                            env.root_body_id,
                            env.nominal_total_mass,
                        ),
                        "reference_centroidal_momentum": (
                            env.reference_centroidal_momentum[
                                state.info["phase"]
                            ]
                        ),
                        "root_quaternion": state.data.xquat[
                            env.anchor_body_id
                        ],
                    }
                )
            if actor_capture_point_tracking:
                transition.update(
                    {
                        "capture_point": mjx_capture_point(
                            env.mjx_model,
                            state.data,
                            env.root_body_id,
                            env.nominal_total_mass,
                            env.centroidal_gravity,
                        ),
                        "reference_capture_point": env.reference_capture_point[
                            state.info["phase"]
                        ],
                    }
                )
            if recovery_support is not None:
                transition["recovery_gate"] = recovery_gate
                transition["gated_residual_action"] = _residual_action
                transition["reset_was_carried"] = state.info[
                    "reset_was_carried"
                ]
            if torso_wrench_assistance:
                transition["torso_wrench"] = torso_wrench
            if actor_learned_torso_wrench:
                transition["learned_torso_wrench"] = learned_torso_wrench
                transition["learned_torso_wrench_normalized"] = (
                    learned_torso_wrench_normalized
                )
                transition["learned_torso_wrench_penalty"] = (
                    learned_torso_wrench_step_penalty
                )
            if actor_counterfactual_wrench_distillation:
                transition.update(
                    {
                        "counterfactual_loss": jp.where(
                            active & counterfactual_step_valid,
                            counterfactual_step_loss,
                            0.0,
                        ),
                        "counterfactual_valid": (
                            active & counterfactual_step_valid
                        ),
                        "counterfactual_integrity": (
                            (~active) | counterfactual_step_integrity
                        ),
                        "counterfactual_done_mismatch": (
                            active & (~counterfactual_done_match)
                        ),
                        "counterfactual_residual_action": _residual_action,
                        "counterfactual_teacher_wrench": teacher_world_wrench,
                        "counterfactual_student_wrench_max": jp.max(
                            jp.abs(state.data.xfrc_applied)
                        ),
                        **{
                            f"counterfactual_{name}": jp.where(
                                active & counterfactual_step_valid,
                                value,
                                0.0,
                            )
                            for name, value in (
                                counterfactual_step_telemetry.items()
                            )
                            if name != "valid"
                        },
                    }
                )
            return (next_state, foot_bump_ou), transition

        env_state = jax.lax.stop_gradient(env_state)

        (final_state, final_foot_bump_ou), traj = jax.lax.scan(
            rollout_step,
            (env_state, env_state.info["foot_bump_ou"]),
            (
                action_noise,
                velocity_pushes,
                terrain_bump_innovations,
                jp.arange(unroll_length, dtype=jp.int32),
            ),
            length=unroll_length,
        )
        final_state = final_state.replace(
            info={**final_state.info, "foot_bump_ou": final_foot_bump_ou}
        )
        centroidal_result = None
        if actor_centroidal_propulsion:
            final_centroidal = mjx_centroidal_momentum(
                env.mjx_model,
                final_state.data,
                env.root_body_id,
                env.nominal_total_mass,
            )
            actual_momentum = jp.concatenate(
                (traj["centroidal_momentum"], final_centroidal[None]),
                axis=0,
            )
            reference_momentum = jp.concatenate(
                (
                    traj["reference_centroidal_momentum"],
                    env.reference_centroidal_momentum[
                        final_state.info["phase"]
                    ][None],
                ),
                axis=0,
            )
            root_quaternion = jp.concatenate(
                (
                    traj["root_quaternion"],
                    final_state.data.xquat[env.anchor_body_id][None],
                ),
                axis=0,
            )
            centroidal_result = centroidal_window_objective(
                actual_momentum,
                reference_momentum,
                root_quaternion,
                done=traj["done"],
                active=traj["ahac_active"],
                window=actor_centroidal_window,
                linear_scale=env.centroidal_linear_scale,
                angular_scale=env.centroidal_angular_scale,
                delta=actor_centroidal_delta,
            )
            valid_denominator = jp.maximum(
                centroidal_result.valid_count, 1
            ).astype(centroidal_result.error.dtype)
            component_rms = jp.sqrt(
                jp.sum(
                    jp.where(
                        centroidal_result.valid[:, None],
                        jp.square(centroidal_result.normalized_error),
                        0.0,
                    ),
                    axis=0,
                )
                / valid_denominator
            )
            traj = {
                **traj,
                "actor_centroidal_loss": centroidal_result.loss,
                "actor_centroidal_valid_window_count": (
                    centroidal_result.valid_count
                ),
                "actor_centroidal_p99_forward_abs": (
                    centroidal_result.p99_forward_abs
                ),
                "actor_centroidal_component_rms": component_rms,
            }
        capture_point_result = None
        if actor_capture_point_tracking:
            final_capture_point = mjx_capture_point(
                env.mjx_model,
                final_state.data,
                env.root_body_id,
                env.nominal_total_mass,
                env.centroidal_gravity,
            )
            actual_capture_point = jp.concatenate(
                (traj["capture_point"], final_capture_point[None]), axis=0
            )
            reference_capture_point = jp.concatenate(
                (
                    traj["reference_capture_point"],
                    env.reference_capture_point[
                        final_state.info["phase"]
                    ][None],
                ),
                axis=0,
            )
            capture_active = capture_state_validity(
                traj["ahac_active"], traj["done"]
            )
            capture_point_result = capture_point_objective(
                actual_capture_point,
                reference_capture_point,
                active=capture_active,
                standing_height=env.standing_com_height,
                delta=actor_capture_point_delta,
            )
            valid_denominator = jp.maximum(
                capture_point_result.valid_count, 1
            ).astype(capture_point_result.error.dtype)
            component_rms = jp.sqrt(
                jp.sum(
                    jp.where(
                        capture_point_result.valid[:, None],
                        jp.square(capture_point_result.normalized_error),
                        0.0,
                    ),
                    axis=0,
                )
                / valid_denominator
            )
            traj = {
                **traj,
                "actor_capture_point_loss": capture_point_result.loss,
                "actor_capture_point_valid_count": (
                    capture_point_result.valid_count
                ),
                "actor_capture_point_p99_norm": capture_point_result.p99_norm,
                "actor_capture_point_component_rms": component_rms,
            }

        bootstrap_obs = critic_norm.normalize(
            critic_norm_state, traj["bootstrap_critic_obs"]
        ).astype(jp.float32)
        bootstrap_v = conservative_value(
            critic.apply(target_critic_params, bootstrap_obs),
            double=ahac,
        )

        # Accumulate discounted returns, handling episode boundaries. Time-limit
        # truncations bootstrap from the pre-reset observation stored by env.step.
        def accum_return(carry, x):
            total, running, discount = carry
            r, done, terminal, v_next, active = x
            next_discount = jp.where(active, discount * gamma, discount)
            running = running + discount * r
            trunc_bootstrap = (
                current_actor_bootstrap_scale
                * (1.0 - terminal)
                * next_discount
                * v_next
            )
            total = total + jp.where(done, running + trunc_bootstrap, 0.0)
            running = jp.where(done, 0.0, running)
            discount = jp.where(done, 1.0, next_discount)
            return (total, running, discount), None

        (total_ret, running, final_discount), _ = jax.lax.scan(
            accum_return,
            (0.0, 0.0, 1.0),
            (
                traj["reward"],
                traj["done"],
                traj["terminal"],
                bootstrap_v,
                traj["ahac_active"],
            ),
        )

        final_obs = critic_norm.normalize(
            critic_norm_state,
            env._get_critic_obs(final_state.data, final_state.info),
        ).astype(jp.float32)
        final_v = conservative_value(
            critic.apply(target_critic_params, final_obs),
            double=ahac,
        )
        final_bootstrap = jp.where(
            traj["done"][-1],
            0.0,
            current_actor_bootstrap_scale * final_discount * final_v,
        )

        total_ret = total_ret + running + final_bootstrap

        active_count = jp.maximum(jp.sum(traj["ahac_active"]), 1)
        anchor_loss = (
            jp.sum(traj["actor_policy_anchor_penalty"]) / active_count
        )
        actor_objective = -total_ret / active_count + anchor_loss
        if actor_counterfactual_wrench_distillation:
            counterfactual_valid_count = jp.maximum(
                jp.sum(traj["counterfactual_valid"]), 1
            )
            counterfactual_task_loss = (
                jp.sum(traj["counterfactual_loss"])
                / counterfactual_valid_count
            )
            counterfactual_residual = traj[
                "counterfactual_residual_action"
            ]
            counterfactual_leg_residual = counterfactual_residual[
                ..., jp.asarray(counterfactual_leg_indices, dtype=jp.int32)
            ]
            counterfactual_residual_magnitude = jp.mean(
                jp.square(counterfactual_leg_residual)
            )
            counterfactual_residual_change = jp.diff(
                counterfactual_leg_residual, axis=0
            )
            counterfactual_residual_temporal_weight = 0.001
            counterfactual_objective = (
                counterfactual_task_loss
                + 0.01 * counterfactual_residual_magnitude
                + counterfactual_residual_temporal_weight
                * (
                    jp.sum(jp.square(counterfactual_residual_change))
                    / jp.maximum(counterfactual_residual_change.size, 1)
                )
            )
            actor_objective = actor_objective + counterfactual_objective
        if centroidal_result is not None:
            actor_objective = (
                actor_objective
                + actor_centroidal_weight * centroidal_result.loss
            )
        if capture_point_result is not None:
            actor_objective = (
                actor_objective
                + actor_capture_point_weight * capture_point_result.loss
            )
        return actor_objective, (traj, final_state)

    def critic_fit_from_data(
        critic_params,
        target_critic_params,
        critic_norm_state,
        traj_obs,
        traj_rewards,
        traj_dones,
        traj_terminals,
        traj_bootstrap_obs,
        final_obs,
        traj_active,
    ):
        """
        Critic TD(lambda) loss using trajectory data collected by the actor.

        Implements Eq. 7 from the SHAC paper (Xu et al., ICLR 2022).
        All in float32 precision.
        """

        flat_obs = traj_obs.reshape(-1, env.critic_obs_dim)
        flat_bootstrap_obs = traj_bootstrap_obs.reshape(-1, env.critic_obs_dim)
        flat_obs_norm = critic_norm.normalize(critic_norm_state, flat_obs).astype(
            jp.float32
        )
        flat_bootstrap_obs_norm = critic_norm.normalize(
            critic_norm_state, flat_bootstrap_obs
        ).astype(jp.float32)
        final_obs_norm = critic_norm.normalize(critic_norm_state, final_obs).astype(
            jp.float32
        )

        # Predicted values V(s_t)
        value_predictions = critic.apply(critic_params, flat_obs_norm)

        next_v = conservative_value(
            critic.apply(target_critic_params, flat_bootstrap_obs_norm),
            double=ahac,
        )
        final_v = conservative_value(
            critic.apply(target_critic_params, final_obs_norm),
            double=ahac,
        )  # scalar

        rewards = traj_rewards.reshape(-1).astype(jp.float32)  # (H,)
        dones = traj_dones.reshape(-1).astype(jp.float32)  # (H,)
        terminals = traj_terminals.reshape(-1).astype(jp.float32)  # (H,)
        active = traj_active.reshape(-1).astype(jp.bool_)

        def scan_fn(g_next, inputs):
            r"""TD(lambda) backward scan."""
            r, done, terminal, v_next, is_active = inputs
            g_normal = r + gamma * (
                (1.0 - gae_lambda) * v_next + gae_lambda * g_next
            )  # Normal step
            g_trunc = r + gamma * v_next  # Time-limit trunc.
            g_term = r  # true term.
            g = jp.where(terminal, g_term, jp.where(done, g_trunc, g_normal))
            g = jp.where(is_active, g, g_next)
            return g, g

        _, targets_reversed = jax.lax.scan(
            scan_fn,
            final_v,  # float32 scalar (determines the carry dtype)
            (
                rewards[::-1],
                dones[::-1],
                terminals[::-1],
                next_v[::-1],
                active[::-1],
            ),
        )
        targets = targets_reversed[::-1]

        return critic_value_loss(
            value_predictions,
            jax.lax.stop_gradient(targets),
            double=ahac,
            active_mask=active,
        )

    def aggregate_env_gradients(per_env_grads, max_norm):
        """Aggregates one equal-sized environment shard."""
        if max_norm is not None:
            return aggregate_per_env_gradients(
                per_env_grads, max_norm=max_norm
            )

        grads = jax.tree_util.tree_map(
            lambda grad: jp.nanmean(grad, axis=0), per_env_grads
        )
        grads = jax.tree_util.tree_map(
            lambda grad: jp.where(jp.isfinite(grad), grad, 0.0), grads
        )
        return grads, per_env_gradient_statistics(per_env_grads)

    @jax.jit
    def train_step(state: TrainState):
        (
            key,
            noise_key,
            push_key,
            bump_key,
            diff_mask_key,
            assistance_mask_key,
        ) = jax.random.split(state.key, 6)
        # Keep every rollout random stream paired with a JAVE-off control.
        # JAVE-only sampling is deterministically namespaced from the input key.
        jave_key = (
            jax.random.fold_in(state.key, 0x4A415645)
            if jave_enabled
            else None
        )

        # Curriculum: difficulty=0 during grace, then ramp to 1
        difficulty = jp.clip(
            (state.step - _curriculum_grace_jax).astype(jp.float32)
            / _curriculum_steps_jax,
            0.0,
            1.0,
        )

        # Per-env difficulty: a fixed fraction of envs are held at difficulty=0
        # The mask is resampled every unroll
        zero_diff_mask = (
            jax.random.uniform(diff_mask_key, (effective_num_envs,))
            < zero_difficulty_frac
        )
        per_env_difficulty = jp.where(
            zero_diff_mask,
            jp.zeros(effective_num_envs),
            jp.full((effective_num_envs,), difficulty),
        )

        # Inject per-env difficulty into all non-zeroed-out env states
        updated_env_state = state.env_state.replace(
            info={**state.env_state.info, "difficulty": per_env_difficulty}
        )
        if actor_phase_robust_weighting or actor_cagrad:
            actor_start_phases = jax.lax.stop_gradient(
                updated_env_state.info["phase"]
            )

        # Pre-sample all stochastic inputs (reparameterization)
        all_action_noise = jax.random.normal(
            noise_key, (effective_num_envs, unroll_length, env.action_dim)
        )
        all_velocity_pushes = jax.random.uniform(
            push_key,
            (effective_num_envs, unroll_length, 2),
            minval=_push_velocity_lo,
            maxval=_push_velocity_hi,
        )
        all_terrain_bump_innovations = jax.random.normal(
            bump_key, (effective_num_envs, unroll_length, 4, 3)
        )
        if torso_wrench_assistance:
            current_torso_wrench_assistance_scale = (
                assistance_scale_at_step(
                    state.step,
                    start_step=torso_wrench_assistance_start_step,
                    end_step=torso_wrench_assistance_end_step,
                )
            )
            all_torso_wrench_assistance_scales = (
                sample_assistance_scales(
                    assistance_mask_key,
                    num_envs=effective_num_envs,
                    scheduled_scale=current_torso_wrench_assistance_scale,
                    zero_fraction=torso_wrench_assistance_zero_fraction,
                    continuous=torso_wrench_assistance_continuous,
                )
            )
        else:
            current_torso_wrench_assistance_scale = jp.asarray(
                0.0, dtype=jp.float32
            )
            all_torso_wrench_assistance_scales = jp.zeros(
                (effective_num_envs,), dtype=jp.float32
            )
        all_randomization = (
            all_action_noise,
            all_velocity_pushes,
            all_terrain_bump_innovations,
            all_torso_wrench_assistance_scales,
        )
        if determinism_probe_output is not None:
            random_inputs_fingerprint = tree_bit_fingerprint(
                (
                    state.key,
                    key,
                    noise_key,
                    push_key,
                    bump_key,
                    diff_mask_key,
                    assistance_mask_key,
                    jave_key,
                    updated_env_state.info["rng"],
                    zero_diff_mask,
                    per_env_difficulty,
                    all_randomization,
                )
            )

        # Preserve the checkpoint's original schedule on exact continuation.
        progress = jp.clip(
            state.step / action_noise_schedule_steps, 0.0, 1.0
        )
        current_noise_std = action_noise_std_start + progress * (
            action_noise_std_end - action_noise_std_start
        )
        current_actor_bootstrap_scale = actor_bootstrap_scale_at_step(
            state.step,
            actor_bootstrap_scale,
            actor_bootstrap_delay_steps,
        )
        current_ahac_horizon = (
            state.ahac_horizon
            if ahac
            else jp.asarray(unroll_length, dtype=jp.float32)
        )
        current_learned_wrench_scale = (
            learned_wrench_scale_at_step(
                state.step,
                start_step=actor_learned_torso_wrench_scale_start_step,
                end_step=actor_learned_torso_wrench_scale_end_step,
                start_scale=actor_learned_torso_wrench_scale,
                end_scale=actor_learned_torso_wrench_scale_end,
            )
            if actor_learned_torso_wrench
            else jp.asarray(0.0, dtype=jp.float32)
        )

        # Actor update
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        if gradient_accumulation_steps == 1:
            (losses, (trajs, final_states)), per_env_grads = jax.vmap(
                actor_grad_fn,
                in_axes=(None, None, None, None, 0, 0, None, None, None, None),
            )(
                state.actor_params,
                state.target_critic_params,
                state.normalizer,
                state.critic_normalizer,
                updated_env_state,
                all_randomization,
                current_noise_std,
                current_actor_bootstrap_scale,
                current_ahac_horizon,
                current_learned_wrench_scale,
            )
            if actor_learned_torso_wrench:
                per_env_grads = jax.tree_util.tree_map(
                    lambda gradient, selected: jp.where(
                        selected, gradient, 0.0
                    ),
                    per_env_grads,
                    preview_adapter_mask,
                )
            if actor_phase_robust_weighting:
                phase_weighting = phase_robust_weights(
                    losses,
                    actor_start_phases,
                    phase_count=int(env.reference_transitions),
                    bin_count=actor_phase_bin_count,
                    robust_fraction=actor_phase_robust_fraction,
                )
                grads = aggregate_phase_weighted_gradients(
                    per_env_grads, phase_weighting.env_weights
                )
                actor_grad_stats = per_env_gradient_statistics(
                    per_env_grads
                )
            else:
                grads, actor_grad_stats = aggregate_env_gradients(
                    per_env_grads, actor_per_env_grad_clip
                )
        else:
            sharded_env_state = reshape_population(
                updated_env_state,
                accumulation_steps=gradient_accumulation_steps,
                microbatch_size=num_envs,
            )
            sharded_randomization = reshape_population(
                all_randomization,
                accumulation_steps=gradient_accumulation_steps,
                microbatch_size=num_envs,
            )
            if actor_cagrad:
                sharded_actor_start_phases = reshape_population(
                    actor_start_phases,
                    accumulation_steps=gradient_accumulation_steps,
                    microbatch_size=num_envs,
                )
                actor_shard_inputs = (
                    sharded_env_state,
                    sharded_randomization,
                    sharded_actor_start_phases,
                )
            else:
                actor_shard_inputs = (
                    sharded_env_state,
                    sharded_randomization,
                )

            def actor_microbatch_step(_, inputs):
                shard_env_state, shard_randomization = inputs[:2]
                (
                    shard_losses,
                    (shard_trajs, shard_final_states),
                ), shard_per_env_grads = jax.vmap(
                    actor_grad_fn,
                    in_axes=(
                        None,
                        None,
                        None,
                        None,
                        0,
                        0,
                        None,
                        None,
                        None,
                        None,
                    ),
                )(
                    state.actor_params,
                    state.target_critic_params,
                    state.normalizer,
                    state.critic_normalizer,
                    shard_env_state,
                    shard_randomization,
                    current_noise_std,
                    current_actor_bootstrap_scale,
                    current_ahac_horizon,
                    current_learned_wrench_scale,
                )
                if actor_learned_torso_wrench:
                    shard_per_env_grads = jax.tree_util.tree_map(
                        lambda gradient, selected: jp.where(
                            selected, gradient, 0.0
                        ),
                        shard_per_env_grads,
                        preview_adapter_mask,
                    )
                if actor_cagrad:
                    shard_grad_stats = per_env_gradient_statistics(
                        shard_per_env_grads
                    )
                    shard_reduction = accumulate_phase_gradients(
                        shard_per_env_grads,
                        inputs[2],
                        phase_count=int(env.reference_transitions),
                        bin_count=actor_phase_bin_count,
                        per_env_max_norm=actor_per_env_grad_clip,
                    )
                else:
                    shard_reduction, shard_grad_stats = (
                        aggregate_env_gradients(
                            shard_per_env_grads, actor_per_env_grad_clip
                        )
                    )
                return None, (
                    shard_losses,
                    shard_trajs,
                    shard_final_states,
                    shard_reduction,
                    {
                        "finite_by_env": shard_grad_stats[
                            "finite_by_env"
                        ],
                        "raw_norm_by_env": shard_grad_stats[
                            "raw_norm_by_env"
                        ],
                    },
                )

            _, actor_shard_outputs = jax.lax.scan(
                actor_microbatch_step,
                None,
                actor_shard_inputs,
            )
            (
                shard_losses,
                shard_trajs,
                shard_final_states,
                shard_reductions,
                shard_grad_stats,
            ) = actor_shard_outputs
            losses = flatten_population(shard_losses)
            trajs = flatten_population(shard_trajs)
            final_states = flatten_population(shard_final_states)
            if actor_cagrad:
                cagrad_reduction = reduce_cagrad_shard_accumulators(
                    shard_reductions,
                    alpha=actor_cagrad_alpha,
                    iterations=actor_cagrad_iterations,
                )
                grads = cagrad_reduction["result"].combined_gradient
            else:
                grads = mean_shard_trees(shard_reductions)
            actor_grad_stats = summarize_shard_stats(shard_grad_stats)

        if actor_cagrad:
            cagrad_loss_diagnostics = cagrad_phase_loss_diagnostics(
                losses=losses,
                phases=actor_start_phases,
                phase_count=int(env.reference_transitions),
                bin_count=actor_phase_bin_count,
            )
        if frozen_preview_treatment:
            preview_action_diagnostics = phase_binned_action_deviation(
                trajs["candidate_action"],
                trajs["parent_action"],
                trajs["transition_phase"],
                phase_count=int(env.reference_transitions),
                bin_count=actor_phase_bin_count,
            )
        if recovery_support is not None:
            recovery_gate_values = trajs["recovery_gate"]
            gated_residual_values = trajs["gated_residual_action"]
            carried_mask = trajs["reset_was_carried"].astype(jp.bool_)
            reference_mask = ~carried_mask
            active_mask = recovery_gate_values > 0.0
            carried_count = jp.sum(carried_mask)
            reference_count = jp.sum(reference_mask)
            recovery_diagnostics = {
                "activation_fraction": jp.mean(
                    recovery_gate_values > 0.0
                ),
                "gate_max": jp.max(recovery_gate_values),
                "carried_activation_fraction": jp.where(
                    carried_count > 0,
                    jp.sum(active_mask & carried_mask) / carried_count,
                    0.0,
                ),
                "reference_activation_fraction": jp.where(
                    reference_count > 0,
                    jp.sum(active_mask & reference_mask) / reference_count,
                    0.0,
                ),
                "gated_residual_rms": jp.sqrt(
                    jp.mean(jp.square(gated_residual_values))
                ),
                "gated_residual_max_abs": jp.max(
                    jp.abs(gated_residual_values)
                ),
                "valid": (
                    jp.all(jp.isfinite(recovery_gate_values))
                    & jp.all(jp.isfinite(gated_residual_values))
                    & jp.all(recovery_gate_values >= 0.0)
                    & jp.all(recovery_gate_values <= 1.0)
                ),
            }
        if torso_wrench_assistance:
            torso_wrench_diagnostics = (
                torso_wrench_assistance_diagnostics(
                    trajs["torso_wrench"],
                    assistance_scales=(
                        all_torso_wrench_assistance_scales
                    ),
                    force_cap=torso_wrench_parameters.force_cap,
                    torque_cap=torso_wrench_parameters.torque_cap,
                )
            )

        if adaptive_phase_sampling:
            completed_failed_count = update_adaptive_phase_state(
                failed_count=updated_env_state.info[
                    "phase_sampler_failed_count"
                ][0],
                transition_phases=trajs["transition_phase"],
                terminals=trajs["terminal"],
                reference_length=env.reference_length,
                alpha=adaptive_phase_alpha,
            )
            adaptive_diagnostics = adaptive_phase_diagnostics(
                failed_count=completed_failed_count,
                transition_phases=trajs["transition_phase"],
                terminals=trajs["terminal"],
                reference_length=env.reference_length,
                uniform_ratio=adaptive_phase_uniform_ratio,
            )
            final_states = broadcast_adaptive_phase_state(
                final_states, completed_failed_count
            )

        physics_actor_grad_norm = compute_grad_norm(grads)
        recovery_teacher_gradient = None
        if recovery_teacher_enabled:
            normalized_teacher_history = env.normalize_actor_obs(
                actor_norm,
                state.normalizer,
                recovery_teacher_actor_obs,
            ).astype(jp.float32)
            recovery_teacher_frames = current_treatment_frame(
                normalized_teacher_history,
                history_len=actor_history_len,
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            recovery_teacher_gradient = (
                mix_recovery_teacher_actor_gradient(
                    grads,
                    state.actor_params,
                    residual_actor=residual_preview_actor,
                    teacher_frames=recovery_teacher_frames,
                    parent_action=recovery_teacher_parent_action,
                    teacher_correction=recovery_teacher_correction,
                    teacher_effective_action=(
                        recovery_teacher_effective_action
                    ),
                    max_ratio=actor_recovery_teacher_gradient_ratio,
                )
            )
            grads = recovery_teacher_gradient.mix.combined_gradient

        actor_grad_norm = compute_grad_norm(grads)

        if actor_frozen_controller_residual:
            updates, new_actor_opt = update_frozen_controller_residual(
                gradients=grads,
                optimizer_state=state.actor_opt,
                params=state.actor_params,
                adapter_optimizer=actor_opt,
            )
            preview_update_diagnostics = {
                "preview_gradient_norm": compute_grad_norm(grads.adapter),
                "preview_update_norm": compute_grad_norm(updates.adapter),
                "frozen_update_max_abs": jp.asarray(0.0),
                "frozen_moment_drift_max_abs": jp.asarray(0.0),
            }
        elif (
            actor_residual_preview_adapter
            and actor_residual_preview_optimizer == "muon"
        ):
            if residual_muon_opt is None or residual_adam_opt is None:
                raise ValueError("residual Muon optimizers are unavailable")
            updates, new_actor_opt, preview_update_diagnostics = (
                apply_residual_muon_update(
                    muon_optimizer=residual_muon_opt,
                    adam_optimizer=residual_adam_opt,
                    gradients=grads,
                    optimizer_state=state.actor_opt,
                    params=state.actor_params,
                )
            )
        elif frozen_preview_treatment:
            updates, new_actor_opt, preview_update_diagnostics = (
                apply_preview_adapter_update(
                    actor_opt,
                    grads,
                    state.actor_opt,
                    state.actor_params,
                    preview_adapter_mask,
                )
            )
        else:
            updates, new_actor_opt = actor_opt.update(
                grads, state.actor_opt
            )
        actor_update_norm = compute_grad_norm(updates)
        new_actor_params = optax.apply_updates(state.actor_params, updates)
        if determinism_probe_output is not None:
            first_actor_action_fingerprint = tree_bit_fingerprint(
                trajs["determinism_actor_step_fingerprint"][:, 0]
            )
            first_mjx_substep_fingerprint = tree_bit_fingerprint(
                trajs["determinism_mjx_substep_fingerprint"][:, 0]
            )
            first_mjx_substep_component_fingerprints = {
                name: tree_bit_fingerprint(
                    trajs[
                        f"determinism_mjx_substep_{name}_fingerprint"
                    ][:, 0]
                )
                for name in FIRST_MJX_SUBSTEP_COMPONENTS
            }
            first_mjx_substep_field_fingerprints = {
                name: tree_bit_fingerprint(
                    trajs[
                        "determinism_mjx_substep_"
                        f"field_{name}_fingerprint"
                    ][:, 0]
                )
                for name in FIRST_MJX_SUBSTEP_FIELDS
            }
            first_mjx_control_step_fingerprint = tree_bit_fingerprint(
                trajs["determinism_mjx_control_step_fingerprint"][:, 0]
            )
            first_env_step_fingerprint = tree_bit_fingerprint(
                trajs["determinism_env_step_fingerprint"][:, 0]
            )
            rollout_fingerprint = tree_bit_fingerprint((trajs, final_states))
            actor_cagrad_fingerprint = tree_bit_fingerprint(
                (grads, updates, new_actor_params, new_actor_opt)
            )

        new_ldm_params = state.ldm_params
        new_ldm_opt = state.ldm_opt
        new_replay_buffer = state.replay_buffer
        jave_ldm_loss = jp.asarray(0.0, dtype=jp.float32)
        jave_vg_active = jp.asarray(False)
        jave_vg_targets = None
        jave_vg_observations = None
        if jave_enabled:
            ldm_train_key, vg_sample_key = jax.random.split(jave_key)
            flat_jave_obs = trajs["jave_obs"].reshape(
                -1, env.jave_obs_dim
            )
            flat_next_jave_obs = trajs["bootstrap_jave_obs"].reshape(
                -1, env.jave_obs_dim
            )
            flat_jave_actions = trajs["jave_action"].reshape(
                -1, env.action_dim
            )
            flat_jave_dones = trajs["done"].reshape(-1)
            normalized_jave_obs = normalize_jave_observation(
                flat_jave_obs,
                state.critic_normalizer,
                critic_dim=env.critic_obs_dim,
                reward_feature_scale=jave_reward_feature_scale,
                eps=critic_norm.eps,
            ).astype(jp.float32)
            normalized_next_jave_obs = normalize_jave_observation(
                flat_next_jave_obs,
                state.critic_normalizer,
                critic_dim=env.critic_obs_dim,
                reward_feature_scale=jave_reward_feature_scale,
                eps=critic_norm.eps,
            ).astype(jp.float32)
            new_replay_buffer = add_to_replay_buffer(
                state.replay_buffer,
                jax.lax.stop_gradient(normalized_jave_obs),
                jax.lax.stop_gradient(flat_jave_actions),
                jax.lax.stop_gradient(normalized_next_jave_obs),
                jax.lax.stop_gradient(flat_jave_dones),
                jave_ldm_buffer_capacity,
            )

            def ldm_update_step(carry, update_key):
                params, opt_state = carry
                batch = sample_replay_buffer(
                    new_replay_buffer,
                    update_key,
                    jave_ldm_batch_size,
                )
                loss, gradients = jax.value_and_grad(
                    lambda candidate: learned_dynamics_loss(
                        learned_dynamics_model, candidate, batch
                    )
                )(params)
                gradients = jax.tree.map(
                    lambda gradient: jp.where(
                        jp.isfinite(gradient), gradient, 0.0
                    ),
                    gradients,
                )
                updates, next_opt_state = ldm_optimizer.update(
                    gradients, opt_state
                )
                next_params = optax.apply_updates(params, updates)
                return (next_params, next_opt_state), loss

            ldm_update_keys = jax.random.split(
                ldm_train_key, jave_ldm_iterations
            )
            (
                (candidate_ldm_params, candidate_ldm_opt),
                jave_ldm_losses,
            ) = jax.lax.scan(
                ldm_update_step,
                (state.ldm_params, state.ldm_opt),
                ldm_update_keys,
            )
            jave_buffer_ready = (
                new_replay_buffer.size >= jave_ldm_batch_size
            )
            new_ldm_params = jax.tree.map(
                lambda candidate, current: jp.where(
                    jave_buffer_ready, candidate, current
                ),
                candidate_ldm_params,
                state.ldm_params,
            )
            new_ldm_opt = jax.tree.map(
                lambda candidate, current: jp.where(
                    jave_buffer_ready, candidate, current
                ),
                candidate_ldm_opt,
                state.ldm_opt,
            )
            jave_ldm_loss = jp.where(
                jave_buffer_ready,
                jave_ldm_losses[-1],
                jp.asarray(0.0, dtype=jp.float32),
            )
            (
                jave_vg_observations,
                jave_vg_actions,
                _,
                _,
            ) = sample_replay_buffer(
                new_replay_buffer,
                vg_sample_key,
                jave_vg_batch_size,
            )
            frozen_normalizer = jax.tree.map(
                jax.lax.stop_gradient, state.critic_normalizer
            )

            def analytical_reward(
                normalized_observation,
                normalized_next_observation,
                action,
            ):
                observation = denormalize_jave_observation(
                    normalized_observation,
                    frozen_normalizer,
                    critic_dim=env.critic_obs_dim,
                    reward_feature_scale=jave_reward_feature_scale,
                    eps=critic_norm.eps,
                )
                next_observation = denormalize_jave_observation(
                    normalized_next_observation,
                    frozen_normalizer,
                    critic_dim=env.critic_obs_dim,
                    reward_feature_scale=jave_reward_feature_scale,
                    eps=critic_norm.eps,
                )
                return env.compute_reward_from_jave_obs(
                    observation, next_observation, action
                )

            jave_vg_active = (
                jave_buffer_ready
                & (
                    state.step
                    >= jave_start_step + jave_vg_warmup_steps
                )
            )

            def compute_jave_targets(_):
                targets = gradient_bellman_targets(
                    dynamics_model=learned_dynamics_model,
                    dynamics_params=new_ldm_params,
                    critic=critic,
                    target_critic_params=state.target_critic_params,
                    normalized_observations=jave_vg_observations,
                    actions=jave_vg_actions,
                    critic_dim=env.critic_obs_dim,
                    gamma=gamma,
                    analytical_reward=analytical_reward,
                )
                return jp.where(jp.isfinite(targets), targets, 0.0)

            jave_vg_targets = jax.lax.cond(
                jave_vg_active,
                compute_jave_targets,
                lambda _: jp.zeros(
                    (jave_vg_batch_size, env.jave_obs_dim),
                    dtype=jp.float32,
                ),
                operand=None,
            )

        if determinism_probe_output is not None:
            learned_dynamics_fingerprint = tree_bit_fingerprint(
                (
                    new_ldm_params,
                    new_ldm_opt,
                    new_replay_buffer,
                    jave_vg_observations,
                    jave_vg_targets,
                )
            )

        if ahac:
            ahac_active_mask = active_horizon_mask(
                state.ahac_horizon, ahac_horizon_max
            )
            ahac_contact_by_step = jp.mean(
                trajs["ahac_contact_stiffness"], axis=0
            )
            ahac_update = update_horizon_dual(
                horizon=state.ahac_horizon,
                dual=state.ahac_dual,
                contact_by_step=ahac_contact_by_step,
                active_mask=ahac_active_mask,
                threshold=ahac_contact_threshold,
                learning_rate=ahac_dual_lr,
                minimum=ahac_horizon_min,
                maximum=ahac_horizon_max,
            )
        else:
            ahac_active_mask = jp.ones((unroll_length,), dtype=jp.bool_)
            ahac_contact_by_step = jp.zeros(
                (unroll_length,), dtype=jp.float64
            )
            ahac_update = None

        # Critic updates
        all_obs = trajs["critic_obs"]
        all_rewards = trajs["reward"]
        all_dones = trajs["done"]
        all_terminals = trajs["terminal"]
        all_bootstrap_obs = trajs["bootstrap_critic_obs"]
        all_ahac_active = trajs["ahac_active"]
        all_final_obs = jax.vmap(env._get_critic_obs)(
            final_states.data, final_states.info
        )

        def single_env_critic_loss(
            critic_params,
            target_critic_params,
            norm_state,
            obs,
            rewards,
            dones,
            terminals,
            bootstrap_obs,
            final_obs,
            active,
        ):
            return critic_fit_from_data(
                critic_params,
                target_critic_params,
                norm_state,
                obs,
                rewards,
                dones,
                terminals,
                bootstrap_obs,
                final_obs,
                active,
            ).total

        def critic_update_step(carry, _):
            c_params, c_opt_state = carry

            critic_grad_fn = jax.value_and_grad(
                single_env_critic_loss, argnums=0
            )
            if gradient_accumulation_steps == 1:
                c_losses, c_per_env_grads = jax.vmap(
                    critic_grad_fn,
                    in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
                )(
                    c_params,
                    state.target_critic_params,
                    state.critic_normalizer,
                    all_obs,
                    all_rewards,
                    all_dones,
                    all_terminals,
                    all_bootstrap_obs,
                    all_final_obs,
                    all_ahac_active,
                )
                c_grads, critic_grad_stats = aggregate_env_gradients(
                    c_per_env_grads, critic_per_env_grad_clip
                )
            else:
                sharded_critic_inputs = reshape_population(
                    (
                        all_obs,
                        all_rewards,
                        all_dones,
                        all_terminals,
                        all_bootstrap_obs,
                        all_final_obs,
                        all_ahac_active,
                    ),
                    accumulation_steps=gradient_accumulation_steps,
                    microbatch_size=num_envs,
                )

                def critic_microbatch_step(_, shard_inputs):
                    (
                        shard_obs,
                        shard_rewards,
                        shard_dones,
                        shard_terminals,
                        shard_bootstrap_obs,
                        shard_final_obs,
                        shard_active,
                    ) = shard_inputs
                    shard_losses, shard_per_env_grads = jax.vmap(
                        critic_grad_fn,
                        in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
                    )(
                        c_params,
                        state.target_critic_params,
                        state.critic_normalizer,
                        shard_obs,
                        shard_rewards,
                        shard_dones,
                        shard_terminals,
                        shard_bootstrap_obs,
                        shard_final_obs,
                        shard_active,
                    )
                    shard_grads, shard_grad_stats = (
                        aggregate_env_gradients(
                            shard_per_env_grads,
                            critic_per_env_grad_clip,
                        )
                    )
                    return None, (
                        shard_losses,
                        shard_grads,
                        {
                            "finite_by_env": shard_grad_stats[
                                "finite_by_env"
                            ],
                            "raw_norm_by_env": shard_grad_stats[
                                "raw_norm_by_env"
                            ],
                        },
                    )

                _, critic_shard_outputs = jax.lax.scan(
                    critic_microbatch_step,
                    None,
                    sharded_critic_inputs,
                )
                (
                    shard_c_losses,
                    shard_c_grads,
                    shard_critic_grad_stats,
                ) = critic_shard_outputs
                c_losses = flatten_population(shard_c_losses)
                c_grads = mean_shard_trees(shard_c_grads)
                critic_grad_stats = summarize_shard_stats(
                    shard_critic_grad_stats
                )

            if jave_enabled:
                def active_jave_gradient(_):
                    return jax.value_and_grad(
                        lambda candidate: gradient_bellman_loss(
                            critic=critic,
                            critic_params=candidate,
                            normalized_critic_observations=(
                                jave_vg_observations[
                                    ..., : env.critic_obs_dim
                                ]
                            ),
                            targets=jave_vg_targets,
                            critic_dim=env.critic_obs_dim,
                        )
                    )(c_params)

                jave_vg_loss_value, jave_vg_gradients = jax.lax.cond(
                    jave_vg_active,
                    active_jave_gradient,
                    lambda _: (
                        jp.asarray(0.0, dtype=jp.float32),
                        jax.tree.map(jp.zeros_like, c_params),
                    ),
                    operand=None,
                )
                effective_jave_weight = (
                    state.jave_vg_weight
                    * jave_vg_active.astype(jp.float32)
                )
                c_grads = jax.tree.map(
                    lambda td_gradient, jave_gradient: (
                        td_gradient
                        + effective_jave_weight * jave_gradient
                    ),
                    c_grads,
                    jave_vg_gradients,
                )
            else:
                jave_vg_loss_value = jp.asarray(0.0, dtype=jp.float32)

            c_grads = jax.tree.map(
                lambda gradient: jp.where(
                    jp.isfinite(gradient), gradient, 0.0
                ),
                c_grads,
            )

            c_updates, new_c_opt = critic_opt.update(c_grads, c_opt_state)
            new_c_params = optax.apply_updates(c_params, c_updates)

            return (new_c_params, new_c_opt), {
                "loss": jp.nanmean(c_losses),
                "finite_fraction": critic_grad_stats["finite_fraction"],
                "raw_norm_median": critic_grad_stats["raw_norm_median"],
                "raw_norm_max": critic_grad_stats["raw_norm_max"],
                "jave_vg_loss": jave_vg_loss_value,
            }

        if ahac:
            def ahac_critic_update_step(carry, _):
                (
                    current_params,
                    current_opt,
                    loss_history,
                    converged,
                    executed_iterations,
                ) = carry
                (candidate_params, candidate_opt), candidate_metrics = (
                    critic_update_step((current_params, current_opt), None)
                )
                apply_update = ~converged
                next_params = select_active_tree(
                    current_params, candidate_params, apply_update
                )
                next_opt = select_active_tree(
                    current_opt, candidate_opt, apply_update
                )
                next_history = jp.where(
                    apply_update,
                    jp.concatenate(
                        (loss_history[1:], candidate_metrics["loss"][None])
                    ),
                    loss_history,
                )
                next_executed = executed_iterations + apply_update.astype(jp.int32)
                next_converged = converged | (
                    (next_executed >= 5)
                    & critic_convergence(
                        next_history, ahac_critic_tolerance
                    )
                )
                return (
                    next_params,
                    next_opt,
                    next_history,
                    next_converged,
                    next_executed,
                ), {
                    **candidate_metrics,
                    "applied": apply_update,
                }

            initial_loss_history = jp.full(
                (5,), jp.inf, dtype=jp.float32
            )
            (
                new_critic_params,
                new_critic_opt,
                ahac_critic_loss_history,
                ahac_critic_converged,
                ahac_critic_iterations_executed,
            ), critic_update_metrics = jax.lax.scan(
                ahac_critic_update_step,
                (
                    state.critic_params,
                    state.critic_opt,
                    initial_loss_history,
                    jp.asarray(False),
                    jp.asarray(0, dtype=jp.int32),
                ),
                None,
                length=ahac_critic_max_iterations,
            )
        else:
            (new_critic_params, new_critic_opt), critic_update_metrics = (
                jax.lax.scan(
                    critic_update_step,
                    (state.critic_params, state.critic_opt),
                    None,
                    length=critic_iterations,
                )
            )
            ahac_critic_loss_history = jp.full(
                (5,), jp.nan, dtype=jp.float32
            )
            ahac_critic_converged = jp.asarray(False)
            ahac_critic_iterations_executed = jp.asarray(
                critic_iterations, dtype=jp.int32
            )

        # Soft target update
        new_target = (
            new_critic_params
            if ahac
            else optax.incremental_update(
                new_critic_params,
                state.target_critic_params,
                target_update_rate,
            )
        )
        if ahac:
            def single_env_critic_fit_diagnostics(
                obs,
                rewards,
                dones,
                terminals,
                bootstrap_obs,
                final_obs,
                active,
            ):
                fit = critic_fit_from_data(
                    new_critic_params,
                    state.target_critic_params,
                    state.critic_normalizer,
                    obs,
                    rewards,
                    dones,
                    terminals,
                    bootstrap_obs,
                    final_obs,
                    active,
                )
                return fit.head_losses, fit.disagreement

            (
                ahac_per_env_head_losses,
                ahac_per_env_head_disagreement,
            ) = jax.vmap(single_env_critic_fit_diagnostics)(
                all_obs,
                all_rewards,
                all_dones,
                all_terminals,
                all_bootstrap_obs,
                all_final_obs,
                all_ahac_active,
            )
            ahac_critic_head_losses = jp.mean(
                ahac_per_env_head_losses, axis=0
            )
            ahac_critic_head_disagreement = jp.mean(
                ahac_per_env_head_disagreement
            )
        else:
            ahac_critic_head_losses = jp.zeros((2,), dtype=jp.float32)
            ahac_critic_head_disagreement = jp.asarray(
                0.0, dtype=jp.float32
            )

        # Update actor and critic normalizers from their own observation streams.
        if frozen_preview_treatment:
            new_actor_norm = state.normalizer
            preview_normalizer_drift = jp.asarray(
                0.0, dtype=state.normalizer.mean.dtype
            )
        else:
            flat_actor_obs = trajs["actor_obs"].reshape(
                -1, env.actor_frame_obs_dim
            )
            safe_actor_obs = jp.where(
                jp.isfinite(flat_actor_obs),
                flat_actor_obs,
                state.normalizer.mean,
            )
            new_actor_norm = actor_norm.update(
                state.normalizer, safe_actor_obs
            )

        flat_critic_obs = trajs["critic_obs"].reshape(-1, env.critic_obs_dim)
        safe_critic_obs = jp.where(
            jp.isfinite(flat_critic_obs),
            flat_critic_obs,
            state.critic_normalizer.mean,
        )
        new_critic_norm = critic_norm.update(state.critic_normalizer, safe_critic_obs)
        if determinism_probe_output is not None:
            critic_fingerprint = tree_bit_fingerprint(
                (
                    new_critic_params,
                    new_critic_opt,
                    new_target,
                    new_critic_norm,
                )
            )

        new_state = state.replace(
            key=key,
            env_state=final_states,
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            target_critic_params=new_target,
            normalizer=new_actor_norm,
            critic_normalizer=new_critic_norm,
            actor_opt=new_actor_opt,
            critic_opt=new_critic_opt,
            step=state.step + steps_per_actor_update,
            ahac_horizon=(
                ahac_update.horizon if ahac else state.ahac_horizon
            ),
            ahac_dual=(ahac_update.dual if ahac else state.ahac_dual),
        )
        if jave_enabled:
            new_state = new_state.replace(
                ldm_params=new_ldm_params,
                ldm_opt=new_ldm_opt,
                replay_buffer=new_replay_buffer,
            )

        # Collect metrics
        metrics = {
            "reward": jp.mean(trajs["reward"]),
            "vel_x": jp.mean(trajs["vel_x"]),
            "vel_y": jp.mean(trajs["vel_y"]),
            "yaw_rate": jp.mean(trajs["yaw_rate"]),
            "cmd_x": jp.mean(trajs["cmd_x"]),
            "cmd_y": jp.mean(trajs["cmd_y"]),
            "cmd_yaw": jp.mean(trajs["cmd_yaw"]),
            "contact": jp.mean(final_states.metrics["contact_force"]),
            "actor_grad": actor_grad_norm,
            "actor_update_norm": actor_update_norm,
            "actor_grad_finite_fraction": actor_grad_stats["finite_fraction"],
            "actor_grad_raw_median": actor_grad_stats["raw_norm_median"],
            "actor_grad_raw_max": actor_grad_stats["raw_norm_max"],
            "critic_loss": critic_update_metrics["loss"][-1],
            "critic_grad_finite_fraction": critic_update_metrics[
                "finite_fraction"
            ][-1],
            "critic_grad_raw_median": critic_update_metrics["raw_norm_median"][
                -1
            ],
            "critic_grad_raw_max": critic_update_metrics["raw_norm_max"][-1],
            "actor_loss": jp.mean(losses),
            "actor_policy_anchor_squared_error": jp.mean(
                trajs["actor_policy_anchor_squared_error"]
            ),
            "actor_bootstrap_scale_current": current_actor_bootstrap_scale,
            "action_noise_current": current_noise_std,
            "track_vx": jp.mean(jp.abs(trajs["vel_x"] - trajs["cmd_x"])),
            "track_vy": jp.mean(jp.abs(trajs["vel_y"] - trajs["cmd_y"])),
            "track_yaw": jp.mean(jp.abs(trajs["yaw_rate"] - trajs["cmd_yaw"])),
            "track_vx_sq": jp.mean((trajs["vel_x"] - trajs["cmd_x"]) ** 2),
            "track_vy_sq": jp.mean((trajs["vel_y"] - trajs["cmd_y"]) ** 2),
            "track_yaw_sq": jp.mean((trajs["yaw_rate"] - trajs["cmd_yaw"]) ** 2),
            "rew_vel_x": jp.mean(final_states.metrics["rew_vel_x"]),
            "rew_vel_y": jp.mean(final_states.metrics["rew_vel_y"]),
            "rew_yaw": jp.mean(final_states.metrics["rew_yaw"]),
            "rew_vz": jp.mean(final_states.metrics["rew_vz"]),
            "pen_rate": jp.mean(final_states.metrics["pen_rate"]),
            "height": jp.mean(trajs["height"]),
            "tilt": jp.mean(trajs["tilt"]),
            "difficulty": difficulty,
            "foot_normal_FL": jp.mean(trajs["foot_normal_FL"]),
            "foot_normal_FR": jp.mean(trajs["foot_normal_FR"]),
            "foot_normal_RL": jp.mean(trajs["foot_normal_RL"]),
            "foot_normal_RR": jp.mean(trajs["foot_normal_RR"]),
            "contact_topology_event_count": jp.sum(
                trajs["contact_topology_event"]
            ),
            "contact_topology_event_fraction": jp.mean(
                trajs["contact_topology_event"]
            ),
            "demonstration_replay_count": jp.sum(
                trajs["demonstration_replay"]
            ),
            "demonstration_replay_fraction": jp.mean(
                trajs["demonstration_replay"]
            ),
        }
        if jave_enabled:
            metrics.update(
                {
                    "jave_ldm_loss": jave_ldm_loss,
                    "jave_vg_loss": critic_update_metrics[
                        "jave_vg_loss"
                    ][-1],
                    "jave_vg_active": (
                        jave_vg_active.astype(jp.float32)
                        * (state.jave_vg_weight > 0.0).astype(jp.float32)
                    ),
                    "jave_vg_target_norm": jp.mean(
                        jp.linalg.norm(jave_vg_targets, axis=-1)
                    ),
                    "jave_replay_size": new_replay_buffer.size,
                }
            )
        if determinism_probe_output is not None:
            metrics.update(
                {
                    "determinism_random_inputs_fingerprint": (
                        random_inputs_fingerprint
                    ),
                    "determinism_first_actor_action_fingerprint": (
                        first_actor_action_fingerprint
                    ),
                    "determinism_first_mjx_substep_fingerprint": (
                        first_mjx_substep_fingerprint
                    ),
                    **{
                        f"determinism_first_mjx_substep_{name}_fingerprint": (
                            fingerprint
                        )
                        for name, fingerprint in (
                            first_mjx_substep_component_fingerprints.items()
                        )
                    },
                    **{
                        "determinism_first_mjx_substep_"
                        f"field_{name}_fingerprint": fingerprint
                        for name, fingerprint in (
                            first_mjx_substep_field_fingerprints.items()
                        )
                    },
                    "determinism_first_mjx_control_step_fingerprint": (
                        first_mjx_control_step_fingerprint
                    ),
                    "determinism_first_env_step_fingerprint": (
                        first_env_step_fingerprint
                    ),
                    "determinism_rollout_fingerprint": rollout_fingerprint,
                    "determinism_actor_cagrad_fingerprint": (
                        actor_cagrad_fingerprint
                    ),
                    "determinism_learned_dynamics_fingerprint": (
                        learned_dynamics_fingerprint
                    ),
                    "determinism_critic_fingerprint": critic_fingerprint,
                }
            )
        if ahac:
            active_contact = jp.where(
                ahac_active_mask,
                ahac_contact_by_step,
                jp.nan,
            )
            metrics.update(
                {
                    "ahac_horizon": ahac_update.horizon,
                    "ahac_horizon_before_update": state.ahac_horizon,
                    "ahac_active_transitions": jp.sum(ahac_active_mask),
                    "ahac_dual_mean": jp.mean(ahac_update.dual),
                    "ahac_dual_max": jp.max(ahac_update.dual),
                    "ahac_contact_stiffness_mean": jp.nanmean(active_contact),
                    "ahac_contact_stiffness_max": jp.nanmax(active_contact),
                    "ahac_contact_threshold": jp.asarray(
                        ahac_contact_threshold, dtype=jp.float64
                    ),
                    "ahac_horizon_valid": ahac_update.valid,
                    "ahac_critic_iterations": ahac_critic_iterations_executed,
                    "ahac_critic_converged": ahac_critic_converged,
                    "ahac_critic_loss_history": ahac_critic_loss_history,
                    "ahac_critic_head_losses": ahac_critic_head_losses,
                    "ahac_critic_head_disagreement": (
                        ahac_critic_head_disagreement
                    ),
                }
            )
        if actor_phase_robust_weighting:
            metrics.update(
                {
                    "actor_phase_bin_counts": phase_weighting.bin_counts,
                    "actor_phase_bin_losses": phase_weighting.bin_losses,
                    "actor_phase_bin_weights": phase_weighting.bin_weights,
                    "actor_phase_weighting_valid": phase_weighting.valid,
                    "actor_loss_weighted": phase_weighting.weighted_loss,
                }
            )
        if torso_wrench_assistance:
            metrics.update(
                {
                    "torso_wrench_assistance_scale_current": (
                        current_torso_wrench_assistance_scale
                    ),
                    "torso_wrench_assistance_active_fraction": (
                        torso_wrench_diagnostics["active_fraction"]
                    ),
                    "torso_wrench_assistance_rms_force": (
                        torso_wrench_diagnostics["rms_force"]
                    ),
                    "torso_wrench_assistance_rms_torque": (
                        torso_wrench_diagnostics["rms_torque"]
                    ),
                    "torso_wrench_assistance_max_force": (
                        torso_wrench_diagnostics["max_force"]
                    ),
                    "torso_wrench_assistance_max_torque": (
                        torso_wrench_diagnostics["max_torque"]
                    ),
                    "torso_wrench_assistance_valid": (
                        torso_wrench_diagnostics["valid"]
                    ),
                }
            )
        if actor_learned_torso_wrench:
            learned_force_norm = jp.linalg.norm(
                trajs["learned_torso_wrench"][..., :3], axis=-1
            )
            learned_torque_norm = jp.linalg.norm(
                trajs["learned_torso_wrench"][..., 3:], axis=-1
            )
            learned_normalized = trajs[
                "learned_torso_wrench_normalized"
            ]
            metrics.update(
                {
                    "learned_torso_wrench_scale": jp.asarray(
                        current_learned_wrench_scale,
                        dtype=jp.float32,
                    ),
                    "learned_torso_wrench_rms_force": jp.sqrt(
                        jp.mean(jp.square(learned_force_norm))
                    ),
                    "learned_torso_wrench_rms_torque": jp.sqrt(
                        jp.mean(jp.square(learned_torque_norm))
                    ),
                    "learned_torso_wrench_max_force": jp.max(
                        learned_force_norm
                    ),
                    "learned_torso_wrench_max_torque": jp.max(
                        learned_torque_norm
                    ),
                    "learned_torso_wrench_normalized_rms": jp.sqrt(
                        jp.mean(jp.square(learned_normalized))
                    ),
                    "learned_torso_wrench_saturation_fraction": jp.mean(
                        jp.abs(learned_normalized) >= 0.95
                    ),
                    "learned_torso_wrench_penalty": jp.mean(
                        trajs["learned_torso_wrench_penalty"]
                    ),
                    "learned_torso_wrench_valid": (
                        jp.all(jp.isfinite(learned_normalized))
                        & jp.all(
                            jp.isfinite(trajs["learned_torso_wrench"])
                        )
                        & (
                            jp.max(learned_force_norm)
                            <= torso_wrench_parameters.force_cap + 1e-5
                        )
                        & (
                            jp.max(learned_torque_norm)
                            <= torso_wrench_parameters.torque_cap + 1e-5
                        )
                    ),
                }
            )
        if actor_counterfactual_wrench_distillation:
            counterfactual_valid_count = jp.sum(
                trajs["counterfactual_valid"]
            )
            counterfactual_invalid_count = jp.sum(
                ~trajs["counterfactual_integrity"]
            )
            counterfactual_done_mismatch_count = jp.sum(
                trajs["counterfactual_done_mismatch"]
            )
            counterfactual_residual = trajs[
                "counterfactual_residual_action"
            ]
            counterfactual_leg_residual = counterfactual_residual[
                ..., jp.asarray(counterfactual_leg_indices, dtype=jp.int32)
            ]
            counterfactual_nonleg_max = jp.max(
                jp.abs(
                    counterfactual_residual[..., counterfactual_nonleg_mask]
                )
            )
            counterfactual_teacher_wrench = trajs[
                "counterfactual_teacher_wrench"
            ]
            counterfactual_metric_names = (
                "base_linear_loss",
                "base_angular_loss",
                "centroidal_linear_loss",
                "centroidal_angular_loss",
                "cosine",
                "student_rms",
                "teacher_rms",
                "normalized_error_rms",
            )
            counterfactual_finite = jp.all(
                jp.stack(
                    [
                        jp.all(
                            jp.isfinite(
                                trajs[f"counterfactual_{name}"]
                            )
                        )
                        for name in counterfactual_metric_names
                    ]
                )
            )
            metrics.update(
                {
                    "actor_counterfactual_loss": jp.sum(
                        trajs["counterfactual_loss"]
                    )
                    / jp.maximum(counterfactual_valid_count, 1),
                    **{
                        f"actor_counterfactual_{name}": jp.sum(
                            trajs[f"counterfactual_{name}"]
                        )
                        / jp.maximum(counterfactual_valid_count, 1)
                        for name in counterfactual_metric_names
                    },
                    "actor_counterfactual_valid_count": (
                        counterfactual_valid_count
                    ),
                    "actor_counterfactual_invalid_count": (
                        counterfactual_invalid_count
                    ),
                    "actor_counterfactual_done_mismatch_count": (
                        counterfactual_done_mismatch_count
                    ),
                    "actor_counterfactual_residual_rms": jp.sqrt(
                        jp.mean(jp.square(counterfactual_leg_residual))
                    ),
                    "actor_counterfactual_residual_max_abs": jp.max(
                        jp.abs(counterfactual_leg_residual)
                    ),
                    "actor_counterfactual_residual_bound_fraction": jp.mean(
                        jp.abs(counterfactual_leg_residual) >= 0.95
                    ),
                    "actor_counterfactual_nonleg_max_abs": (
                        counterfactual_nonleg_max
                    ),
                    "actor_counterfactual_student_wrench_max_abs": jp.max(
                        trajs["counterfactual_student_wrench_max"]
                    ),
                    "actor_counterfactual_teacher_wrench_rms": jp.sqrt(
                        jp.mean(jp.square(counterfactual_teacher_wrench))
                    ),
                    "actor_counterfactual_valid": (
                        (counterfactual_valid_count > 0)
                        & (counterfactual_invalid_count == 0)
                        & counterfactual_finite
                        & jp.all(jp.isfinite(counterfactual_residual))
                        & (counterfactual_nonleg_max == 0.0)
                        & (
                            jp.max(
                                trajs[
                                    "counterfactual_student_wrench_max"
                                ]
                            )
                            == 0.0
                        )
                    ),
                }
            )
        if actor_cagrad:
            cagrad_result = cagrad_reduction["result"]
            cagrad_counts_match = jp.all(
                cagrad_loss_diagnostics["bin_counts"]
                == cagrad_reduction["bin_counts"]
            )
            metrics.update(
                {
                    "actor_cagrad_bin_counts": cagrad_reduction[
                        "bin_counts"
                    ],
                    "actor_cagrad_bin_gradient_norms": cagrad_reduction[
                        "bin_gradient_norms"
                    ],
                    "actor_cagrad_bin_losses": cagrad_loss_diagnostics[
                        "bin_losses"
                    ],
                    "actor_cagrad_weights": cagrad_result.weights,
                    "actor_cagrad_gram_matrix": cagrad_result.gram_matrix,
                    "actor_cagrad_cosine_matrix": cagrad_result.cosine_matrix,
                    "actor_cagrad_objective": cagrad_result.objective,
                    "actor_cagrad_dual_gap": cagrad_result.dual_gap,
                    "actor_cagrad_uniform_combined_cosine": (
                        cagrad_result.uniform_combined_cosine
                    ),
                    "actor_cagrad_combined_norm": physics_actor_grad_norm,
                    "actor_cagrad_valid": (
                        cagrad_reduction["valid"]
                        & cagrad_loss_diagnostics["valid"]
                        & cagrad_counts_match
                    ),
                }
            )
        if recovery_teacher_enabled:
            teacher_mix = recovery_teacher_gradient.mix
            metrics.update(
                {
                    "actor_recovery_teacher_loss": (
                        recovery_teacher_gradient.loss
                    ),
                    "actor_recovery_teacher_raw_gradient_norm": (
                        teacher_mix.raw_teacher_norm
                    ),
                    "actor_recovery_teacher_projected_gradient_norm": (
                        teacher_mix.projected_teacher_norm
                    ),
                    "actor_recovery_teacher_applied_gradient_norm": (
                        teacher_mix.applied_teacher_norm
                    ),
                    "actor_recovery_teacher_physics_gradient_norm": (
                        teacher_mix.physics_norm
                    ),
                    "actor_recovery_teacher_combined_gradient_norm": (
                        teacher_mix.combined_norm
                    ),
                    "actor_recovery_teacher_physics_dot": (
                        teacher_mix.physics_teacher_dot
                    ),
                    "actor_recovery_teacher_physics_cosine": (
                        teacher_mix.physics_teacher_cosine
                    ),
                    "actor_recovery_teacher_applied_scale": (
                        teacher_mix.applied_scale
                    ),
                    "actor_recovery_teacher_parent_gradient_max_abs": (
                        recovery_teacher_gradient.parent_gradient_max_abs
                    ),
                    "actor_recovery_teacher_valid": (
                        recovery_teacher_gradient.valid
                    ),
                }
            )
        if frozen_preview_treatment:
            preview_valid = (
                preview_action_diagnostics["valid"]
                & jp.isfinite(
                    preview_update_diagnostics["preview_gradient_norm"]
                )
                & jp.isfinite(
                    preview_update_diagnostics["preview_update_norm"]
                )
                & (
                    preview_update_diagnostics["frozen_update_max_abs"]
                    == 0.0
                )
                & (
                    preview_update_diagnostics[
                        "frozen_moment_drift_max_abs"
                    ]
                    == 0.0
                )
                & (preview_normalizer_drift == 0.0)
            )
            if recovery_support is not None:
                preview_valid = preview_valid & recovery_diagnostics["valid"]
            if residual_muon_treatment:
                preview_valid = preview_valid & jp.all(
                    jp.stack(
                        [
                            jp.isfinite(preview_update_diagnostics[key])
                            for key in (
                                "muon_kernel_gradient_norm",
                                "muon_kernel_update_norm",
                                "aux_adam_gradient_norm",
                                "aux_adam_update_norm",
                            )
                        ]
                    )
                )
            metrics.update(
                {
                    "actor_preview_gradient_norm": (
                        preview_update_diagnostics[
                            "preview_gradient_norm"
                        ]
                    ),
                    "actor_preview_update_norm": (
                        preview_update_diagnostics["preview_update_norm"]
                    ),
                    "actor_preview_frozen_parameter_drift_max_abs": (
                        preview_update_diagnostics[
                            "frozen_update_max_abs"
                        ]
                    ),
                    "actor_preview_frozen_moment_drift_max_abs": (
                        preview_update_diagnostics[
                            "frozen_moment_drift_max_abs"
                        ]
                    ),
                    "actor_preview_normalizer_drift_max_abs": (
                        preview_normalizer_drift
                    ),
                    "actor_preview_bin_counts": (
                        preview_action_diagnostics["bin_counts"]
                    ),
                    "actor_preview_bin_action_deviation_mean_abs": (
                        preview_action_diagnostics["mean_abs"]
                    ),
                    "actor_preview_bin_action_deviation_max_abs": (
                        preview_action_diagnostics["max_abs"]
                    ),
                    "actor_preview_valid": preview_valid,
                }
            )
            if residual_muon_treatment:
                metrics.update(
                    {
                        "actor_muon_kernel_gradient_norm": (
                            preview_update_diagnostics[
                                "muon_kernel_gradient_norm"
                            ]
                        ),
                        "actor_muon_kernel_update_norm": (
                            preview_update_diagnostics[
                                "muon_kernel_update_norm"
                            ]
                        ),
                        "actor_muon_aux_adam_gradient_norm": (
                            preview_update_diagnostics[
                                "aux_adam_gradient_norm"
                            ]
                        ),
                        "actor_muon_aux_adam_update_norm": (
                            preview_update_diagnostics[
                                "aux_adam_update_norm"
                            ]
                        ),
                    }
                )
            if recovery_support is not None:
                metrics.update(
                    {
                        "actor_recovery_gate_activation_fraction": (
                            recovery_diagnostics["activation_fraction"]
                        ),
                        "actor_recovery_gate_max": recovery_diagnostics[
                            "gate_max"
                        ],
                        "actor_recovery_carried_activation_fraction": (
                            recovery_diagnostics[
                                "carried_activation_fraction"
                            ]
                        ),
                        "actor_recovery_reference_activation_fraction": (
                            recovery_diagnostics[
                                "reference_activation_fraction"
                            ]
                        ),
                        "actor_recovery_gated_residual_rms": (
                            recovery_diagnostics["gated_residual_rms"]
                        ),
                        "actor_recovery_gated_residual_max_abs": (
                            recovery_diagnostics[
                                "gated_residual_max_abs"
                            ]
                        ),
                        "actor_recovery_valid": recovery_diagnostics[
                            "valid"
                        ],
                    }
                )
        if actor_centroidal_propulsion:
            metrics.update(
                {
                    "actor_centroidal_loss": jp.mean(
                        trajs["actor_centroidal_loss"]
                    ),
                    "actor_centroidal_valid_window_count": jp.sum(
                        trajs["actor_centroidal_valid_window_count"]
                    ),
                    "actor_centroidal_p99_forward_abs": jp.max(
                        trajs["actor_centroidal_p99_forward_abs"]
                    ),
                    "actor_centroidal_component_rms": jp.sqrt(
                        jp.mean(
                            jp.square(
                                trajs["actor_centroidal_component_rms"]
                            ),
                            axis=0,
                        )
                    ),
                }
            )
        if actor_capture_point_tracking:
            metrics.update(
                {
                    "actor_capture_point_loss": jp.mean(
                        trajs["actor_capture_point_loss"]
                    ),
                    "actor_capture_point_valid_count": jp.sum(
                        trajs["actor_capture_point_valid_count"]
                    ),
                    "actor_capture_point_p99_norm": jp.max(
                        trajs["actor_capture_point_p99_norm"]
                    ),
                    "actor_capture_point_component_rms": jp.sqrt(
                        jp.mean(
                            jp.square(
                                trajs["actor_capture_point_component_rms"]
                            ),
                            axis=0,
                        )
                    ),
                }
            )
        if adaptive_phase_sampling:
            metrics.update(
                {
                    "adaptive_phase_failure_ema": adaptive_diagnostics[
                        "failure_ema"
                    ],
                    "adaptive_phase_probabilities": adaptive_diagnostics[
                        "probabilities"
                    ],
                    "adaptive_phase_terminal_bin_counts": (
                        adaptive_diagnostics["terminal_bin_counts"]
                    ),
                    "adaptive_phase_min_probability": adaptive_diagnostics[
                        "minimum_probability"
                    ],
                    "adaptive_phase_sampling_valid": adaptive_diagnostics[
                        "valid"
                    ],
                }
            )

        return new_state, metrics

    initialized_state = TrainState(
        key=key,
        env_state=env_state,
        actor_params=actor_params,
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        normalizer=actor_normalizer,
        critic_normalizer=critic_normalizer,
        actor_opt=actor_opt_state,
        critic_opt=critic_opt_state,
        ldm_params=ldm_params,
        ldm_opt=ldm_opt_state,
        replay_buffer=replay_buffer,
        jave_vg_weight=(
            jp.asarray(jave_vg_weight, dtype=jp.float32)
            if jave_enabled
            else None
        ),
        step=canonicalize_step_dtype(0),
        ahac_horizon=(
            jp.asarray(ahac_horizon_min, dtype=jp.float32)
            if ahac
            else None
        ),
        ahac_dual=(
            jp.zeros((ahac_horizon_max,), dtype=jp.float32)
            if ahac
            else None
        ),
    )
    if migration_report is not None:
        persist_future_reference_migration_report(
            save_dir, migration_report
        )
    if reference_path_migration_report is not None:
        persist_reference_path_migration_report(
            save_dir, reference_path_migration_report
        )
    if resumed_state is not None:
        if ahac and (
            getattr(resumed_state, "ahac_horizon", None) is None
            or getattr(resumed_state, "ahac_dual", None) is None
        ):
            raise ValueError("AHAC checkpoint state is missing horizon or dual leaves")
        print(
            "Restoring complete training state from step "
            f"{resumed_step} (PRNG, environments, parameters, optimizers, "
            "and normalizers)"
        )
        resumed_state = resumed_state.replace(
            env_state=migrate_env_state_metrics(
                resumed_state.env_state,
                initialized_state.env_state,
            )
        )
        if reference_path_migration_report is not None:
            resumed_state = resumed_state.replace(
                env_state=initialized_state.env_state
            )
        if future_reference_upgrade:
            legacy_resumed_state = resumed_state
            resumed_state = migrate_future_reference_train_state(
                resumed_state,
                initialized_state,
                env,
                expected_history_len=actor_history_len,
            )
            migration_report = future_reference_migration_report(
                legacy_resumed_state,
                resumed_state,
                actor,
                legacy_frame_dim=int(
                    legacy_resumed_state.normalizer.mean.shape[0]
                ),
                treatment_frame_dim=env.actor_frame_obs_dim,
                history_len=actor_history_len,
            )
            persist_future_reference_migration_report(
                save_dir, migration_report
            )
        if requires_plain_residual_preview_resume_validation(
            actor_residual_preview_adapter=actor_residual_preview_adapter,
            actor_frozen_controller_residual=(
                actor_frozen_controller_residual
            ),
            actor_learned_torso_wrench=actor_learned_torso_wrench,
            learned_wrench_state=isinstance(
                resumed_state.actor_params, FrozenControllerWrenchParams
            ),
        ):
            if residual_adapter_upgrade or future_reference_upgrade:
                parent_params = resumed_state.actor_params
                parent_optimizer_state = resumed_state.actor_opt
                adapter_params = residual_preview_actor.init(
                    jax.random.fold_in(k1, 0x5250),
                    jp.zeros(
                        (1, env.actor_frame_obs_dim), dtype=jp.float32
                    ),
                )
                zero_head_feature_report = None
                if actor_residual_preview_initial_adapter_path is not None:
                    if actor_residual_preview_initial_adapter_sha256 is None:
                        raise ValueError(
                            "zero-head recovery feature SHA-256 is required"
                        )
                    (
                        adapter_params,
                        zero_head_feature_report,
                    ) = load_zero_head_recovery_feature_adapter(
                        actor_residual_preview_initial_adapter_path,
                        expected_sha256=(
                            actor_residual_preview_initial_adapter_sha256
                        ),
                        template_params=adapter_params,
                    )
                composite_params = FrozenPreviewResidualParams(
                    parent=parent_params,
                    adapter=adapter_params,
                )
                adapter_audit_optimizer_state = (
                    initialize_residual_adapter_optimizer(
                        actor_opt,
                        parent_optimizer_state=parent_optimizer_state,
                        composite_params=composite_params,
                    )
                )
                if actor_residual_preview_optimizer == "muon":
                    if residual_muon_opt is None or residual_adam_opt is None:
                        raise ValueError("residual Muon optimizers are unavailable")
                    composite_optimizer_state = (
                        initialize_residual_muon_optimizer(
                            muon_optimizer=residual_muon_opt,
                            adam_optimizer=residual_adam_opt,
                            parent_optimizer_state=parent_optimizer_state,
                            adapter_params=adapter_params,
                        )
                    )
                    residual_muon_report = residual_muon_migration_report(
                        parent_optimizer_state=parent_optimizer_state,
                        candidate_optimizer_state=composite_optimizer_state,
                    )
                else:
                    composite_optimizer_state = adapter_audit_optimizer_state
                normalized_observations = env.normalize_actor_obs(
                    actor_norm,
                    resumed_state.normalizer,
                    resumed_state.env_state.obs,
                ).astype(jp.float32)
                residual_adapter_report = (
                    residual_adapter_migration_report(
                        parent_actor=actor,
                        residual_actor=residual_preview_actor,
                        parent_params=parent_params,
                        parent_optimizer_state=parent_optimizer_state,
                        candidate_params=composite_params,
                        candidate_optimizer_state=(
                            adapter_audit_optimizer_state
                        ),
                        normalized_observations=normalized_observations,
                        history_len=actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                    )
                )
                resumed_state = resumed_state.replace(
                    actor_params=composite_params,
                    actor_opt=composite_optimizer_state,
                )
                persist_residual_adapter_migration_report(
                    save_dir, residual_adapter_report
                )
                if zero_head_feature_report is not None:
                    zero_head_feature_report = {
                        **zero_head_feature_report,
                        "parent_parameters_exact": (
                            residual_adapter_report["parent_parameters_exact"]
                        ),
                        "initial_action_exact": bool(
                            residual_adapter_report["residual_action_zero"]
                            and residual_adapter_report[
                                "reconstructed_parent_exact"
                            ]
                            and residual_adapter_report[
                                "max_action_absolute_error"
                            ]
                            == 0.0
                        ),
                        "adapter_optimizer_moments_zero": bool(
                            residual_adapter_report["adapter_mu_zero"]
                            and residual_adapter_report["adapter_nu_zero"]
                        ),
                    }
                    zero_head_feature_report["valid"] = bool(
                        zero_head_feature_report["valid"]
                        and zero_head_feature_report[
                            "parent_parameters_exact"
                        ]
                        and zero_head_feature_report["initial_action_exact"]
                        and zero_head_feature_report[
                            "adapter_optimizer_moments_zero"
                        ]
                    )
                    persist_zero_head_feature_transfer_report(
                        save_dir, zero_head_feature_report
                    )
                if residual_muon_report is not None:
                    persist_residual_muon_migration_report(
                        save_dir, residual_muon_report
                    )
            elif not isinstance(
                resumed_state.actor_params, FrozenPreviewResidualParams
            ):
                raise ValueError(
                    "resumed residual preview actor has invalid parameters"
                )
            elif (
                actor_residual_preview_optimizer == "muon"
                and not isinstance(
                    resumed_state.actor_opt, FrozenPreviewResidualMuonState
                )
            ):
                raise ValueError(
                    "resumed residual Muon actor has invalid optimizer state"
                )
            elif (
                actor_residual_preview_optimizer == "adam"
                and isinstance(
                    resumed_state.actor_opt, FrozenPreviewResidualMuonState
                )
            ):
                raise ValueError(
                    "resumed residual Adam actor has invalid optimizer state"
                )
            if actor_torso_wrench_assistance_conditioning:
                adapter_kernel, _ = split_residual_adapter_params(
                    resumed_state.actor_params.adapter
                )
                if adapter_kernel.shape[0] == env.actor_frame_obs_dim:
                    legacy_params = resumed_state.actor_params
                    legacy_optimizer = resumed_state.actor_opt
                    (
                        conditioned_params,
                        conditioned_optimizer,
                        conditioning_report,
                    ) = migrate_residual_adapter_assistance_conditioning(
                        params=legacy_params,
                        optimizer_state=legacy_optimizer,
                        expected_input_dim=env.actor_frame_obs_dim,
                    )
                    normalized_observations = env.normalize_actor_obs(
                        actor_norm,
                        resumed_state.normalizer,
                        resumed_state.env_state.obs,
                    ).astype(jp.float32)
                    legacy_action, _, _ = apply_frozen_preview_residual(
                        actor,
                        residual_preview_actor,
                        legacy_params,
                        normalized_observations,
                        history_len=actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                    )
                    conditioned_action, _, _ = apply_frozen_preview_residual(
                        actor,
                        residual_preview_actor,
                        conditioned_params,
                        normalized_observations,
                        history_len=actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                        assistance_scale=jp.asarray(0.0, dtype=jp.float32),
                    )
                    action_exact = bool(
                        np.array_equal(
                            np.asarray(legacy_action),
                            np.asarray(conditioned_action),
                        )
                    )
                    conditioning_report["zero_scale_action_exact"] = action_exact
                    conditioning_report["valid"] = bool(
                        conditioning_report["valid"] and action_exact
                    )
                    resumed_state = resumed_state.replace(
                        actor_params=conditioned_params,
                        actor_opt=conditioned_optimizer,
                    )
                    persist_assistance_conditioning_migration_report(
                        save_dir, conditioning_report
                    )
                elif adapter_kernel.shape[0] != env.actor_frame_obs_dim + 1:
                    raise ValueError(
                        "conditioned residual adapter input width is invalid"
                    )
            preview_adapter_mask = build_residual_adapter_mask(
                resumed_state.actor_params
            )
            preview_trainable_parameter_count = sum(
                int(np.count_nonzero(np.asarray(leaf)))
                for leaf in jax.tree_util.tree_leaves(
                    preview_adapter_mask
                )
            )
            expected_residual_parameters = (
                (
                    env.actor_frame_obs_dim
                    + int(actor_torso_wrench_assistance_conditioning)
                )
                * actor_residual_preview_hidden
                + actor_residual_preview_hidden
                + actor_residual_preview_hidden * env.action_dim
                + env.action_dim
            )
            if (
                preview_trainable_parameter_count
                != expected_residual_parameters
            ):
                raise ValueError(
                    "residual preview trainable parameter count is invalid"
                )
            if policy_anchor_source_params is not None:
                validate_policy_anchor_source_parent(
                    policy_anchor_source_params,
                    resumed_state.actor_params,
                )
        if actor_frozen_controller_residual:
            if frozen_controller_residual_upgrade:
                if frozen_controller_residual_depth(
                    resumed_state.actor_params
                ) != actor_frozen_controller_residual_depth - 1:
                    raise ValueError(
                        "frozen controller residual parent depth is invalid"
                    )
                normalized_observations = env.normalize_actor_obs(
                    actor_norm,
                    resumed_state.normalizer,
                    resumed_state.env_state.obs,
                ).astype(jp.float32)

                def frozen_parent_apply(parent_params, observations):
                    if isinstance(parent_params, FrozenPreviewResidualParams):
                        return apply_frozen_preview_residual(
                            actor,
                            residual_preview_actor,
                            parent_params,
                            observations,
                            history_len=actor_history_len,
                            treatment_frame_dim=env.actor_frame_obs_dim,
                        )[0]
                    return apply_frozen_controller_residual(
                        frozen_parent_apply,
                        frozen_controller_residual_actor,
                        parent_params,
                        observations,
                        history_len=actor_history_len,
                        frame_dim=env.actor_frame_obs_dim,
                    )[0]

                (
                    nested_params,
                    nested_optimizer_state,
                    frozen_controller_report,
                ) = migrate_frozen_controller_residual(
                    parent_params=resumed_state.actor_params,
                    parent_optimizer_state=resumed_state.actor_opt,
                    parent_apply=frozen_parent_apply,
                    adapter_actor=frozen_controller_residual_actor,
                    adapter_optimizer=actor_opt,
                    rng=jax.random.fold_in(k1, 0x434D),
                    normalized_observations=normalized_observations,
                    history_len=actor_history_len,
                    frame_dim=env.actor_frame_obs_dim,
                    residual_action_indices=counterfactual_leg_indices,
                )
                if not frozen_controller_report["valid"]:
                    raise ValueError(
                        "frozen controller residual migration is invalid"
                    )
                resumed_state = resumed_state.replace(
                    actor_params=nested_params,
                    actor_opt=nested_optimizer_state,
                )
            elif not isinstance(
                resumed_state.actor_params, FrozenControllerResidualParams
            ) or not isinstance(
                resumed_state.actor_opt, FrozenControllerResidualOptState
            ) or frozen_controller_residual_depth(
                resumed_state.actor_params
            ) != actor_frozen_controller_residual_depth:
                raise ValueError(
                    "resumed frozen controller residual state is invalid"
                )
            preview_trainable_parameter_count = sum(
                int(np.asarray(leaf).size)
                for leaf in jax.tree.leaves(
                    resumed_state.actor_params.adapter
                )
            )
        if actor_learned_torso_wrench:
            if isinstance(
                resumed_state.actor_params, FrozenControllerWrenchParams
            ):
                composite_wrench_params = resumed_state.actor_params
                if not isinstance(
                    composite_wrench_params.controller,
                    FrozenPreviewResidualParams,
                ):
                    raise ValueError(
                        "learned wrench continuation has an invalid controller"
                    )
                saved_conditioning = bool(
                    resumed_hparams.get(
                        "actor_learned_torso_wrench_condition_on_scale", False
                    )
                )
                if (
                    actor_learned_torso_wrench_condition_on_scale
                    and not saved_conditioning
                ):
                    composite_wrench_params = (
                        migrate_learned_wrench_scale_conditioning(
                            composite_wrench_params
                        )
                    )
                    resumed_state = resumed_state.replace(
                        actor_params=composite_wrench_params,
                        actor_opt=actor_opt.init(composite_wrench_params),
                    )
                elif (
                    saved_conditioning
                    != actor_learned_torso_wrench_condition_on_scale
                ):
                    raise ValueError(
                        "conditioned learned wrench heads cannot be downgraded"
                    )
            else:
                controller_params = resumed_state.actor_params
                wrench_init_frame = jp.zeros(
                    (1, env.actor_frame_obs_dim), dtype=jp.float32
                )
                wrench_params = learned_torso_wrench_actor.init(
                    jax.random.fold_in(k1, 0x5752),
                    wrench_init_frame,
                    (
                        jp.ones((1,), dtype=jp.float32)
                        if actor_learned_torso_wrench_condition_on_scale
                        else None
                    ),
                )
                composite_wrench_params = FrozenControllerWrenchParams(
                    controller=controller_params,
                    wrench=wrench_params,
                )
                resumed_state = resumed_state.replace(
                    actor_params=composite_wrench_params,
                    actor_opt=actor_opt.init(composite_wrench_params),
                )
            preview_adapter_mask = (
                build_learned_wrench_anneal_mask(composite_wrench_params)
                if actor_learned_torso_wrench_train_controller
                else build_learned_wrench_mask(composite_wrench_params)
            )
            preview_trainable_parameter_count = sum(
                int(np.count_nonzero(np.asarray(leaf)))
                for leaf in jax.tree_util.tree_leaves(preview_adapter_mask)
            )
            expected_wrench_parameters = (
                (
                    env.actor_frame_obs_dim
                    + int(actor_learned_torso_wrench_condition_on_scale)
                )
                * actor_learned_torso_wrench_hidden
                + actor_learned_torso_wrench_hidden
                + actor_learned_torso_wrench_hidden * 6
                + 6
            )
            if actor_learned_torso_wrench_train_controller:
                expected_wrench_parameters += (
                    env.actor_frame_obs_dim * actor_residual_preview_hidden
                    + actor_residual_preview_hidden
                    + actor_residual_preview_hidden * env.action_dim
                    + env.action_dim
                )
            if preview_trainable_parameter_count != expected_wrench_parameters:
                raise ValueError(
                    "learned torso wrench trainable parameter count is invalid"
                )
        if adaptive_phase_sampling:
            resumed_state = resumed_state.replace(
                env_state=migrate_adaptive_phase_env_state(
                    resumed_state.env_state,
                    reference_length=env.reference_length,
                )
            )
        if jave_enabled:
            if "bootstrap_jave_obs" not in resumed_state.env_state.info:
                resumed_jave_obs = jax.vmap(env._get_jave_obs)(
                    resumed_state.env_state.data,
                    resumed_state.env_state.info,
                )
                resumed_state = resumed_state.replace(
                    env_state=resumed_state.env_state.replace(
                        info={
                            **resumed_state.env_state.info,
                            "bootstrap_jave_obs": resumed_jave_obs,
                        }
                    )
                )
            resumed_has_jave_state = all(
                getattr(resumed_state, name, None) is not None
                for name in ("ldm_params", "ldm_opt", "replay_buffer")
            )
            if saved_jave_enabled and not resumed_has_jave_state:
                raise ValueError(
                    "JAVE checkpoint is missing learned-dynamics state"
                )
            if not resumed_has_jave_state:
                resumed_state = resumed_state.replace(
                    ldm_params=ldm_params,
                    ldm_opt=ldm_opt_state,
                    replay_buffer=replay_buffer,
                )
            resumed_state = resumed_state.replace(
                jave_vg_weight=jp.asarray(
                    jave_vg_weight, dtype=jp.float32
                )
            )
    resumed_state, resume_randomness_report = (
        apply_resume_randomness_setting(
            resumed_state,
            seed=resume_random_seed,
        )
    )
    if resume_randomness_report is not None:
        persist_resume_randomness_audit(
            save_dir,
            resume_randomness_report,
        )
        print(
            "Rekeyed resumed trainer and environment randomness with "
            f"resume_random_seed={resume_random_seed}; all non-RNG state "
            "is exact"
        )
    if actor_residual_preview_adapter and resumed_state is None:
        raise ValueError(
            "residual preview adapter requires an explicit resumed checkpoint"
        )
    state = select_initial_training_state(
        initialized_state=initialized_state,
        resumed_state=resumed_state,
    )

    # JAX distinguishes uncommitted and explicitly placed arrays in its JIT
    # cache key.  Commit the initial state before warm-up so the warm-up output
    # and the real update share one signature instead of compiling twice.
    state = commit_tree_to_local_device(state)

    print("Compiling...")
    start_comp_time = time.perf_counter()
    warmup_state, _ = train_step(state)
    jax.block_until_ready(warmup_state.step)
    compile_time = time.perf_counter() - start_comp_time
    print(f"Compilation took {compile_time:.1f}s")

    # Match compiled dtypes/placement without consuming a logical update. Fresh
    # runs retain the established one-step normalizer warm-up; exact resumes
    # preserve the saved normalizers byte-for-byte.
    state = canonicalize_tree_like(state, warmup_state)
    if resumed_state is None:
        state = state.replace(
            normalizer=warmup_state.normalizer,
            critic_normalizer=warmup_state.critic_normalizer,
        )

    print("Training...")

    if diagnose:
        header = (
            f"{'Step':>7} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'RewVx':>7} | {'RewVy':>7} | {'RewYaw':>7} | "
            f"{'PenRate':>7} | {'Height':>7} | "
            f"{'Tilt':>7} | {'Diff':>5} | {'AGrad':>7} | {'Status'}"
        )
    else:
        header = (
            f"{'Step':>7} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'AGrad':>7} | {'CLoss':>7} | "
            f"{'Diff':>5} | {'Status':>8}"
        )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    start = time.time()
    best_reward = (
        resumed_hparams.get("best_reward", -np.inf) if resumed_hparams else -np.inf
    )
    hparams = {
        "algorithm": algorithm_name,
        "total_steps": total_steps,
        "unroll_length": unroll_length,
        "num_envs": num_envs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_num_envs": effective_num_envs,
        "steps_per_actor_update": steps_per_actor_update,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_update_rate": target_update_rate,
        "critic_iterations": critic_iterations,
        "ahac": ahac,
        "ahac_horizon_min": ahac_horizon_min,
        "ahac_horizon_max": ahac_horizon_max,
        "ahac_contact_threshold": ahac_contact_threshold,
        "ahac_dual_lr": ahac_dual_lr,
        "ahac_critic_max_iterations": ahac_critic_max_iterations,
        "ahac_critic_tolerance": ahac_critic_tolerance,
        "allow_resume_ahac_change": allow_resume_ahac_change,
        "xml_path": xml_path,
        "action_scale": action_scale,
        "cmd_vel_x_range": list(cmd_vel_x_range),
        "cmd_vel_y_range": list(cmd_vel_y_range),
        "cmd_yaw_rate_range": list(cmd_yaw_rate_range),
        "cmd_zero_prob": list(cmd_zero_prob),
        "cmd_ctrl_interval_range": list(cmd_ctrl_interval_range),
        "action_noise_std_start": action_noise_std_hparam(action_noise_std_start),
        "action_noise_std_end": action_noise_std_hparam(action_noise_std_end),
        "allow_resume_action_noise_change": allow_resume_action_noise_change,
        "action_noise_schedule_steps": action_noise_schedule_steps,
        "friction_range": list(friction_range),
        "mass_range": list(mass_range),
        "effort_limit_scale": effort_limit_scale,
        "termination_margin_weight": termination_margin_weight,
        "allow_resume_termination_margin_change": (
            allow_resume_termination_margin_change
        ),
        "tracking_velocity_kernel": tracking_velocity_kernel,
        "allow_resume_tracking_velocity_kernel_change": (
            allow_resume_tracking_velocity_kernel_change
        ),
        "tracking_anchor_position_kernel": tracking_anchor_position_kernel,
        "allow_resume_tracking_anchor_position_kernel_change": (
            allow_resume_tracking_anchor_position_kernel_change
        ),
        "tracking_torso_orientation_weight": (
            tracking_torso_orientation_weight
        ),
        "allow_resume_tracking_torso_orientation_change": (
            allow_resume_tracking_torso_orientation_change
        ),
        "tracking_root_velocity_weight": tracking_root_velocity_weight,
        "allow_resume_tracking_root_velocity_change": (
            allow_resume_tracking_root_velocity_change
        ),
        "jave_vg_weight": jave_vg_weight,
        "jave_vg_warmup_steps": jave_vg_warmup_steps,
        "jave_ldm_hidden": list(jave_ldm_hidden),
        "jave_ldm_lr": jave_ldm_lr,
        "jave_ldm_iterations": jave_ldm_iterations,
        "jave_ldm_batch_size": jave_ldm_batch_size,
        "jave_vg_batch_size": jave_vg_batch_size,
        "jave_ldm_buffer_capacity": jave_ldm_buffer_capacity,
        "jave_reward_feature_scale": jave_reward_feature_scale,
        "jave_collect_transitions": jave_collect_transitions,
        "jave_start_step": jave_start_step,
        "allow_resume_jave_start": allow_resume_jave_start,
        "allow_resume_jave_weight_change": (
            allow_resume_jave_weight_change
        ),
        "reference_reset_noise_scale": reference_reset_noise_scale,
        "reference_root_reset_noise_multiplier": (
            reference_root_reset_noise_multiplier
        ),
        "reference_root_reset_noise_probability": (
            reference_root_reset_noise_probability
        ),
        "allow_resume_reference_root_reset_noise_change": (
            allow_resume_reference_root_reset_noise_change
        ),
        "demonstration_replay_threshold": demonstration_replay_threshold,
        "allow_resume_demonstration_replay_change": (
            allow_resume_demonstration_replay_change
        ),
        "reference_residual_control": reference_residual_control,
        "reference_residual_scale": reference_residual_scale,
        "carried_reset_bank_path": carried_reset_bank_path,
        "carried_reset_probability": carried_reset_probability,
        "carried_reset_bank_start": carried_reset_bank_start,
        "allow_resume_carried_reset_change": (
            allow_resume_carried_reset_change
        ),
        "torso_wrench_assistance": torso_wrench_assistance,
        "torso_wrench_assistance_start_step": (
            torso_wrench_assistance_start_step
        ),
        "torso_wrench_assistance_end_step": (
            torso_wrench_assistance_end_step
        ),
        "torso_wrench_assistance_zero_fraction": (
            torso_wrench_assistance_zero_fraction
        ),
        "allow_resume_torso_wrench_assistance_change": (
            allow_resume_torso_wrench_assistance_change
        ),
        "torso_wrench_assistance_continuous": (
            torso_wrench_assistance_continuous
        ),
        "actor_torso_wrench_assistance_conditioning": (
            actor_torso_wrench_assistance_conditioning
        ),
        "actor_observe_torso_wrench_assistance": (
            actor_observe_torso_wrench_assistance
        ),
        "actor_learned_torso_wrench": actor_learned_torso_wrench,
        "actor_learned_torso_wrench_hidden": (
            actor_learned_torso_wrench_hidden
        ),
        "actor_learned_torso_wrench_scale": (
            actor_learned_torso_wrench_scale
        ),
        "actor_learned_torso_wrench_scale_end": (
            actor_learned_torso_wrench_scale_end
        ),
        "actor_learned_torso_wrench_scale_start_step": (
            actor_learned_torso_wrench_scale_start_step
        ),
        "actor_learned_torso_wrench_scale_end_step": (
            actor_learned_torso_wrench_scale_end_step
        ),
        "actor_learned_torso_wrench_condition_on_scale": (
            actor_learned_torso_wrench_condition_on_scale
        ),
        "actor_learned_torso_wrench_train_controller": (
            actor_learned_torso_wrench_train_controller
        ),
        "actor_learned_torso_wrench_penalty": (
            actor_learned_torso_wrench_penalty
        ),
        "allow_resume_actor_learned_torso_wrench_start": (
            allow_resume_actor_learned_torso_wrench_start
        ),
        "allow_resume_actor_learned_torso_wrench_change": (
            allow_resume_actor_learned_torso_wrench_change
        ),
        "allow_resume_assistance_conditioning_change": (
            allow_resume_assistance_conditioning_change
        ),
        "allow_resume_reference_path_change": (
            allow_resume_reference_path_change
        ),
        "reference_path_migration_artifact": (
            "reference_path_migration.json"
            if reference_path_migration_report is not None
            else None
        ),
        "assistance_conditioning_migration_artifact": (
            "assistance_conditioning_migration.json"
            if conditioning_report is not None
            else None
        ),
        "kp_range": list(kp_range),
        "kd_range": list(kd_range),
        "com_offset_range": list(com_offset_range),
        "push_velocity_range": list(push_velocity_range),
        "push_interval_s": push_interval_s,
        "terrain_flat_prob": terrain_flat_prob,
        "terrain_slope_max": terrain_slope_max,
        "terrain_bump_std": terrain_bump_std,
        "terrain_bump_decay": terrain_bump_decay,
        "terrain": terrain,
        "domain_randomization": domain_randomization,
        "solver_profile": solver_profile,
        "solver_iterations": solver_iterations,
        "solver_ls_iterations": solver_ls_iterations,
        "zero_difficulty_frac": zero_difficulty_frac,
        "curriculum_grace": curriculum_grace,
        "curriculum_steps": curriculum_steps,
        "seed": seed,
        "resume_random_seed": resume_random_seed,
        "determinism_probe": determinism_probe_output is not None,
        "determinism_probe_output": (
            str(Path(determinism_probe_output).resolve())
            if determinism_probe_output is not None
            else None
        ),
        "best_reward": best_reward,
        "checkpoint_steps": (
            list(checkpoint_steps) if checkpoint_steps is not None else None
        ),
        "max_episode_length": max_episode_length,
        "actor_history_len": actor_history_len,
        "expected_actor_obs_dim": expected_actor_obs_dim,
        "actor_observation_noise": actor_observation_noise,
        "actor_observe_motion_anchor_position": (
            actor_observe_motion_anchor_position
        ),
        "actor_reference_lookahead_steps": list(
            actor_reference_lookahead_steps
        ),
        "actor_reference_preview_mode": actor_reference_preview_mode,
        "actor_preview_adapter": actor_preview_adapter,
        "actor_residual_preview_adapter": (
            actor_residual_preview_adapter
        ),
        "actor_residual_preview_hidden": (
            actor_residual_preview_hidden
        ),
        "actor_residual_preview_optimizer": (
            actor_residual_preview_optimizer
        ),
        "actor_frozen_controller_residual": (
            actor_frozen_controller_residual
        ),
        "actor_frozen_controller_residual_hidden": (
            actor_frozen_controller_residual_hidden
        ),
        "actor_frozen_controller_residual_depth": (
            actor_frozen_controller_residual_depth
        ),
        "actor_counterfactual_wrench_distillation": (
            actor_counterfactual_wrench_distillation
        ),
        "actor_counterfactual_wrench_teacher_path": (
            str(Path(actor_counterfactual_wrench_teacher_path).resolve())
            if actor_counterfactual_wrench_teacher_path is not None
            else None
        ),
        "actor_counterfactual_wrench_teacher_sha256": (
            actor_counterfactual_wrench_teacher_sha256
        ),
        "actor_counterfactual_wrench_feasibility_path": (
            str(Path(actor_counterfactual_wrench_feasibility_path).resolve())
            if actor_counterfactual_wrench_feasibility_path is not None
            else None
        ),
        "actor_counterfactual_wrench_feasibility_sha256": (
            actor_counterfactual_wrench_feasibility_sha256
        ),
        "actor_counterfactual_wrench_target_rms": (
            counterfactual_feasibility.target_rms.tolist()
            if counterfactual_feasibility is not None
            else None
        ),
        "actor_counterfactual_wrench_leg_indices": (
            list(counterfactual_leg_indices)
            if counterfactual_leg_indices is not None
            else None
        ),
        "actor_counterfactual_wrench_loss_weight": 1.0,
        "actor_counterfactual_wrench_residual_magnitude_weight": 0.01,
        "actor_counterfactual_wrench_residual_temporal_weight": 0.001,
        "actor_centroidal_propulsion": actor_centroidal_propulsion,
        "actor_centroidal_window": actor_centroidal_window,
        "actor_centroidal_delta": actor_centroidal_delta,
        "actor_centroidal_weight": actor_centroidal_weight,
        "actor_capture_point_tracking": actor_capture_point_tracking,
        "actor_capture_point_delta": actor_capture_point_delta,
        "actor_capture_point_weight": actor_capture_point_weight,
        "actor_residual_preview_initial_adapter_path": (
            actor_residual_preview_initial_adapter_path
        ),
        "actor_residual_preview_initial_adapter_sha256": (
            actor_residual_preview_initial_adapter_sha256
        ),
        "actor_recovery_teacher_enabled": recovery_teacher_enabled,
        "actor_recovery_teacher_dataset_path": (
            actor_recovery_teacher_dataset_path
        ),
        "actor_recovery_teacher_dataset_sha256": (
            actor_recovery_teacher_dataset_sha256
        ),
        "actor_recovery_teacher_gradient_ratio": (
            actor_recovery_teacher_gradient_ratio
        ),
        "allow_resume_actor_recovery_teacher_change": (
            allow_resume_actor_recovery_teacher_change
        ),
        "actor_state_gated_recovery": recovery_support is not None,
        "actor_state_gated_recovery_support_path": (
            actor_state_gated_recovery_support_path
        ),
        "actor_state_gated_recovery_support_sha256": (
            actor_state_gated_recovery_support_sha256
        ),
        "actor_state_gated_recovery_radius": (
            recovery_support_report["radius"]
            if recovery_support_report is not None
            else None
        ),
        "actor_state_gated_recovery_phase_min": (
            recovery_support_report["phase_min"]
            if recovery_support_report is not None
            else None
        ),
        "actor_state_gated_recovery_phase_max": (
            recovery_support_report["phase_max"]
            if recovery_support_report is not None
            else None
        ),
        "actor_state_gated_recovery_taper": (
            recovery_support_report["taper"]
            if recovery_support_report is not None
            else None
        ),
        "allow_resume_actor_state_gated_recovery_start": (
            allow_resume_actor_state_gated_recovery_start
        ),
        "allow_resume_actor_residual_preview_adapter_start": (
            allow_resume_actor_residual_preview_adapter_start
        ),
        "actor_residual_preview_muon_beta": 0.95,
        "actor_residual_preview_muon_ns_steps": 5,
        "actor_residual_preview_muon_nesterov": True,
        "actor_residual_preview_muon_preconditioning": "frobenius",
        "actor_residual_preview_muon_consistent_rms": 0.2,
        "actor_residual_preview_trainable_parameter_count": (
            preview_trainable_parameter_count
            if actor_residual_preview_adapter
            else 0
        ),
        "actor_preview_trainable_parameter_count": (
            preview_trainable_parameter_count
        ),
        "actor_normalizer_frozen": frozen_preview_treatment,
        "checkpoint_phase_metrics_artifact": (
            "checkpoint_phase_metrics.json"
            if frozen_preview_treatment
            else None
        ),
        "resume_future_reference_upgrade": future_reference_upgrade,
        "resume_residual_adapter_upgrade": residual_adapter_upgrade,
        "migration_equivalence_artifact": (
            "migration_equivalence.json"
            if migration_report is not None
            else None
        ),
        "residual_adapter_migration_artifact": (
            "residual_adapter_migration.json"
            if residual_adapter_report is not None
            else None
        ),
        "residual_muon_migration_artifact": (
            "residual_muon_migration.json"
            if residual_muon_report is not None
            else None
        ),
        "actor_per_env_grad_clip": actor_per_env_grad_clip,
        "allow_resume_actor_per_env_grad_clip_change": (
            allow_resume_actor_per_env_grad_clip_change
        ),
        "actor_contact_topology_gradient_truncation": (
            actor_contact_topology_gradient_truncation
        ),
        "allow_resume_actor_contact_topology_gradient_truncation_change": (
            allow_resume_actor_contact_topology_gradient_truncation_change
        ),
        "critic_per_env_grad_clip": critic_per_env_grad_clip,
        "actor_phase_robust_weighting": actor_phase_robust_weighting,
        "actor_phase_bin_count": actor_phase_bin_count,
        "actor_phase_robust_fraction": actor_phase_robust_fraction,
        "actor_cagrad": actor_cagrad,
        "actor_cagrad_alpha": actor_cagrad_alpha,
        "actor_cagrad_iterations": actor_cagrad_iterations,
        "allow_resume_actor_cagrad_change": allow_resume_actor_cagrad_change,
        "adaptive_phase_sampling": adaptive_phase_sampling,
        "adaptive_phase_uniform_ratio": adaptive_phase_uniform_ratio,
        "adaptive_phase_alpha": adaptive_phase_alpha,
        "actor_bootstrap_scale": actor_bootstrap_scale,
        "actor_bootstrap_delay_steps": actor_bootstrap_delay_steps,
        "allow_resume_actor_bootstrap_scale_change": (
            allow_resume_actor_bootstrap_scale_change
        ),
        "actor_hidden": list(actor_hidden),
        "actor_layer_norm": actor_layer_norm,
        "actor_zero_output": actor_zero_output,
        "source_actor_policy": source_actor_policy is not None,
        "actor_policy_anchor_weight": actor_policy_anchor_weight,
        "actor_policy_anchor_source_path": (
            str(Path(actor_policy_anchor_source_path).resolve())
            if actor_policy_anchor_source_path is not None
            else None
        ),
        "actor_policy_anchor_source_sha256": (
            actor_policy_anchor_source_sha256
        ),
        "allow_resume_actor_policy_anchor_source_change": (
            allow_resume_actor_policy_anchor_source_change
        ),
        "actor_kind": (
            "flax_frozen_controller_residual"
            if actor_frozen_controller_residual
            else "flax_residual_preview"
            if actor_residual_preview_adapter
            else "full_rmr"
            if initial_full_actor_policy is not None
            else "bounded_rmr_residual"
            if source_actor_policy is not None
            else "flax"
        ),
        "residual_action_scale": residual_action_scale,
        "differentiate_source_feedback": differentiate_source_feedback,
        "env_variant": env_variant,
        "squash_actor_actions": squash_actor_actions,
        "squash_actor_mean": squash_actor_mean,
        "clip_sampled_actor_actions": clip_sampled_actor_actions,
        **reference_hparams,
    }
    persist_run_hparams(save_dir, hparams)
    log = []
    diag_log = []
    last_checkpoint_step = state.step

    steps_per_iter = steps_per_actor_update
    start_iter = resumed_step // steps_per_iter
    total_iters = total_steps // steps_per_iter

    if determinism_probe_output is not None:
        probe_report = run_determinism_probe(train_step, state)
        probe_report.update(
            {
                "input_state_sha256": numeric_tree_sha256(state),
                "input_step": int(np.asarray(state.step)),
                "compiled_callable_reused": True,
                "ordinary_training_loop_entered": False,
            }
        )
        report_path = persist_determinism_probe_report(
            determinism_probe_output, probe_report
        )
        print(f"Determinism probe written to {report_path}")
        return state, save_dir

    for i in range(start_iter, total_iters):
        state, metrics = train_step(state)

        if actor_cagrad and not bool(metrics["actor_cagrad_valid"]):
            raise RuntimeError("actor CAGrad aggregation is invalid")
        if recovery_teacher_enabled and not bool(
            metrics["actor_recovery_teacher_valid"]
        ):
            raise RuntimeError("actor recovery teacher telemetry is invalid")
        if frozen_preview_treatment and not bool(metrics["actor_preview_valid"]):
            raise RuntimeError("actor preview adapter telemetry is invalid")
        if actor_frozen_controller_residual and i == start_iter:
            validate_first_frozen_controller_update(
                gradient_norm=float(metrics["actor_preview_gradient_norm"]),
                update_norm=float(metrics["actor_preview_update_norm"]),
            )
        if recovery_support is not None and not bool(
            metrics["actor_recovery_valid"]
        ):
            raise RuntimeError("actor recovery gate telemetry is invalid")
        if torso_wrench_assistance and not bool(
            metrics["torso_wrench_assistance_valid"]
        ):
            raise RuntimeError("torso wrench assistance telemetry is invalid")
        if actor_learned_torso_wrench and not bool(
            metrics["learned_torso_wrench_valid"]
        ):
            raise RuntimeError("learned torso wrench telemetry is invalid")
        if actor_counterfactual_wrench_distillation and not bool(
            metrics["actor_counterfactual_valid"]
        ):
            counterfactual_failure = {
                name: (
                    int(metrics[f"actor_counterfactual_{name}"])
                    if name in {"valid_count", "invalid_count"}
                    else float(metrics[f"actor_counterfactual_{name}"])
                )
                for name in (
                    "valid_count",
                    "invalid_count",
                    "done_mismatch_count",
                    "loss",
                    "normalized_error_rms",
                    "residual_max_abs",
                    "nonleg_max_abs",
                    "student_wrench_max_abs",
                )
            }
            raise RuntimeError(
                "counterfactual wrench telemetry is invalid: "
                f"{counterfactual_failure}"
            )
        if actor_policy_anchor_weight > 0.0 and not np.isfinite(
            float(metrics["actor_policy_anchor_squared_error"])
        ):
            raise RuntimeError("actor policy anchor telemetry is invalid")
        if jave_enabled:
            jave_telemetry = {
                "jave_ldm_loss": float(metrics["jave_ldm_loss"]),
                "jave_vg_loss": float(metrics["jave_vg_loss"]),
                "jave_vg_active": bool(metrics["jave_vg_active"]),
                "jave_vg_target_norm": float(
                    metrics["jave_vg_target_norm"]
                ),
                "jave_replay_size": int(metrics["jave_replay_size"]),
            }
            if (
                not all(
                    np.isfinite(jave_telemetry[name])
                    for name in (
                        "jave_ldm_loss",
                        "jave_vg_loss",
                        "jave_vg_target_norm",
                    )
                )
                or jave_telemetry["jave_replay_size"] < 1
            ):
                raise RuntimeError(
                    f"JAVE telemetry is invalid: {jave_telemetry}"
                )

        if should_log_training_iteration(i, start_iteration=start_iter):
            jax.block_until_ready(state.step)

            vel_x = float(metrics["vel_x"])
            vel_y = float(metrics["vel_y"])
            yaw_rate = float(metrics["yaw_rate"])
            reward = float(metrics["reward"])

            # Per-env tracking errors (proper: mean of |vel-cmd| per env)
            cmd_x = float(metrics["cmd_x"])
            cmd_y = float(metrics["cmd_y"])
            cmd_yaw = float(metrics["cmd_yaw"])
            trk_vx = float(metrics["track_vx"])
            trk_vy = float(metrics["track_vy"])
            trk_yaw = float(metrics["track_yaw"])
            diff = float(metrics["difficulty"])
            max_err = max(trk_vx, trk_vy, trk_yaw)

            if max_err < 0.1:
                status = "TRACK :D"
            elif max_err < 0.25:
                status = "CLOSE :)"
            elif max_err < 0.4:
                status = "TRYING"
            else:
                status = "LEARN"

            if diagnose:
                print(
                    f"{state.step:7d} | {reward:7.2f} | {trk_vx:7.3f} | {trk_vy:7.3f} | "
                    f"{trk_yaw:7.3f} | "
                    f"{metrics['rew_vel_x']:7.2f} | "
                    f"{metrics['rew_vel_y']:7.2f} | "
                    f"{metrics['rew_yaw']:7.2f} | "
                    f"{metrics['pen_rate']:7.3f} | {metrics['height']:7.3f} | "
                    f"{metrics['tilt']:7.2f} | {diff:5.2f} | {metrics['actor_grad']:7.1f} | {status}"
                )
                if actor_per_env_grad_clip is not None:
                    print(
                        " " * 9
                        + "raw actor grad "
                        + f"median={float(metrics['actor_grad_raw_median']):.2e} "
                        + f"max={float(metrics['actor_grad_raw_max']):.2e} "
                        + "finite="
                        + f"{float(metrics['actor_grad_finite_fraction']):.3f}"
                    )
                if critic_per_env_grad_clip is not None:
                    print(
                        " " * 9
                        + "raw critic grad "
                        + f"median={float(metrics['critic_grad_raw_median']):.2e} "
                        + f"max={float(metrics['critic_grad_raw_max']):.2e} "
                        + "finite="
                        + f"{float(metrics['critic_grad_finite_fraction']):.3f}"
                    )

                diag_entry = {
                    "step": int(state.step),
                    "reward": reward,
                    "difficulty": diff,
                    "vel_x": vel_x,
                    "vel_y": vel_y,
                    "yaw_rate": yaw_rate,
                    "cmd_x": cmd_x,
                    "cmd_y": cmd_y,
                    "cmd_yaw": cmd_yaw,
                    "track_vx": trk_vx,
                    "track_vy": trk_vy,
                    "track_yaw": trk_yaw,
                    "rew_vel_x": float(metrics["rew_vel_x"]),
                    "rew_vel_y": float(metrics["rew_vel_y"]),
                    "rew_yaw": float(metrics["rew_yaw"]),
                    "pen_rate": float(metrics["pen_rate"]),
                    "height": float(metrics["height"]),
                    "tilt": float(metrics["tilt"]),
                    "actor_grad": float(metrics["actor_grad"]),
                    "actor_update_norm": float(
                        metrics["actor_update_norm"]
                    ),
                    "actor_bootstrap_scale_current": float(
                        metrics["actor_bootstrap_scale_current"]
                    ),
                    "actor_grad_raw_median": float(
                        metrics["actor_grad_raw_median"]
                    ),
                    "actor_grad_raw_max": float(
                        metrics["actor_grad_raw_max"]
                    ),
                    "actor_grad_finite_fraction": float(
                        metrics["actor_grad_finite_fraction"]
                    ),
                    "critic_loss": float(metrics["critic_loss"]),
                    "critic_grad_raw_median": float(
                        metrics["critic_grad_raw_median"]
                    ),
                    "critic_grad_raw_max": float(
                        metrics["critic_grad_raw_max"]
                    ),
                    "critic_grad_finite_fraction": float(
                        metrics["critic_grad_finite_fraction"]
                    ),
                    "contact_topology_event_count": int(
                        metrics["contact_topology_event_count"]
                    ),
                    "contact_topology_event_fraction": float(
                        metrics["contact_topology_event_fraction"]
                    ),
                }
                if jave_enabled:
                    diag_entry.update(jave_telemetry)
                if ahac:
                    diag_entry.update(build_checkpoint_ahac_telemetry(metrics))
                if actor_policy_anchor_weight > 0.0:
                    diag_entry.update(
                        build_policy_anchor_telemetry(
                            metrics, weight=actor_policy_anchor_weight
                        )
                    )
                if actor_phase_robust_weighting:
                    diag_entry.update(
                        {
                            "actor_phase_bin_counts": np.asarray(
                                metrics["actor_phase_bin_counts"]
                            ).tolist(),
                            "actor_phase_bin_losses": np.asarray(
                                metrics["actor_phase_bin_losses"]
                            ).tolist(),
                            "actor_phase_bin_weights": np.asarray(
                                metrics["actor_phase_bin_weights"]
                            ).tolist(),
                            "actor_phase_weighting_valid": bool(
                                metrics["actor_phase_weighting_valid"]
                            ),
                            "actor_loss_unweighted": float(
                                metrics["actor_loss"]
                            ),
                            "actor_loss_weighted": float(
                                metrics["actor_loss_weighted"]
                            ),
                        }
                    )
                if actor_cagrad:
                    diag_entry.update(
                        {
                            "actor_cagrad_bin_counts": np.asarray(
                                metrics["actor_cagrad_bin_counts"]
                            ).tolist(),
                            "actor_cagrad_bin_gradient_norms": np.asarray(
                                metrics["actor_cagrad_bin_gradient_norms"]
                            ).tolist(),
                            "actor_cagrad_bin_losses": np.asarray(
                                metrics["actor_cagrad_bin_losses"]
                            ).tolist(),
                            "actor_cagrad_weights": np.asarray(
                                metrics["actor_cagrad_weights"]
                            ).tolist(),
                            "actor_cagrad_gram_matrix": np.asarray(
                                metrics["actor_cagrad_gram_matrix"]
                            ).tolist(),
                            "actor_cagrad_cosine_matrix": np.asarray(
                                metrics["actor_cagrad_cosine_matrix"]
                            ).tolist(),
                            "actor_cagrad_objective": float(
                                metrics["actor_cagrad_objective"]
                            ),
                            "actor_cagrad_dual_gap": float(
                                metrics["actor_cagrad_dual_gap"]
                            ),
                            "actor_cagrad_uniform_combined_cosine": float(
                                metrics[
                                    "actor_cagrad_uniform_combined_cosine"
                                ]
                            ),
                            "actor_cagrad_combined_norm": float(
                                metrics["actor_cagrad_combined_norm"]
                            ),
                            "actor_cagrad_valid": bool(
                                metrics["actor_cagrad_valid"]
                            ),
                        }
                    )
                if demonstration_replay_threshold is not None:
                    diag_entry.update(
                        build_checkpoint_demonstration_replay_telemetry(
                            metrics,
                            threshold=demonstration_replay_threshold,
                        )
                    )
                if recovery_teacher_enabled:
                    diag_entry.update(
                        build_checkpoint_recovery_teacher_telemetry(
                            metrics,
                            max_ratio=(
                                actor_recovery_teacher_gradient_ratio
                            ),
                        )
                    )
                if frozen_preview_treatment:
                    diag_entry.update(
                        {
                            "actor_preview_gradient_norm": float(
                                metrics["actor_preview_gradient_norm"]
                            ),
                            "actor_preview_update_norm": float(
                                metrics["actor_preview_update_norm"]
                            ),
                            "actor_preview_frozen_parameter_drift_max_abs": float(
                                metrics[
                                    "actor_preview_frozen_parameter_drift_max_abs"
                                ]
                            ),
                            "actor_preview_frozen_moment_drift_max_abs": float(
                                metrics[
                                    "actor_preview_frozen_moment_drift_max_abs"
                                ]
                            ),
                            "actor_preview_normalizer_drift_max_abs": float(
                                metrics[
                                    "actor_preview_normalizer_drift_max_abs"
                                ]
                            ),
                            "actor_preview_bin_counts": np.asarray(
                                metrics["actor_preview_bin_counts"]
                            ).tolist(),
                            "actor_preview_bin_action_deviation_mean_abs": np.asarray(
                                metrics[
                                    "actor_preview_bin_action_deviation_mean_abs"
                                ]
                            ).tolist(),
                            "actor_preview_bin_action_deviation_max_abs": np.asarray(
                                metrics[
                                    "actor_preview_bin_action_deviation_max_abs"
                                ]
                            ).tolist(),
                            "actor_preview_valid": bool(
                                metrics["actor_preview_valid"]
                            ),
                        }
                    )
                if recovery_support is not None:
                    diag_entry.update(
                        {
                            "actor_recovery_gate_activation_fraction": float(
                                metrics[
                                    "actor_recovery_gate_activation_fraction"
                                ]
                            ),
                            "actor_recovery_gate_max": float(
                                metrics["actor_recovery_gate_max"]
                            ),
                            "actor_recovery_carried_activation_fraction": float(
                                metrics[
                                    "actor_recovery_carried_activation_fraction"
                                ]
                            ),
                            "actor_recovery_reference_activation_fraction": float(
                                metrics[
                                    "actor_recovery_reference_activation_fraction"
                                ]
                            ),
                            "actor_recovery_gated_residual_rms": float(
                                metrics[
                                    "actor_recovery_gated_residual_rms"
                                ]
                            ),
                            "actor_recovery_gated_residual_max_abs": float(
                                metrics[
                                    "actor_recovery_gated_residual_max_abs"
                                ]
                            ),
                            "actor_recovery_valid": bool(
                                metrics["actor_recovery_valid"]
                            ),
                        }
                    )
                if residual_muon_treatment:
                    diag_entry.update(
                        {
                            "actor_muon_kernel_gradient_norm": float(
                                metrics["actor_muon_kernel_gradient_norm"]
                            ),
                            "actor_muon_kernel_update_norm": float(
                                metrics["actor_muon_kernel_update_norm"]
                            ),
                            "actor_muon_aux_adam_gradient_norm": float(
                                metrics[
                                    "actor_muon_aux_adam_gradient_norm"
                                ]
                            ),
                            "actor_muon_aux_adam_update_norm": float(
                                metrics["actor_muon_aux_adam_update_norm"]
                            ),
                        }
                    )
                if torso_wrench_assistance:
                    diag_entry.update(
                        {
                            "torso_wrench_assistance_scale_current": float(
                                metrics[
                                    "torso_wrench_assistance_scale_current"
                                ]
                            ),
                            "torso_wrench_assistance_active_fraction": float(
                                metrics[
                                    "torso_wrench_assistance_active_fraction"
                                ]
                            ),
                            "torso_wrench_assistance_rms_force": float(
                                metrics["torso_wrench_assistance_rms_force"]
                            ),
                            "torso_wrench_assistance_rms_torque": float(
                                metrics["torso_wrench_assistance_rms_torque"]
                            ),
                            "torso_wrench_assistance_max_force": float(
                                metrics["torso_wrench_assistance_max_force"]
                            ),
                            "torso_wrench_assistance_max_torque": float(
                                metrics["torso_wrench_assistance_max_torque"]
                            ),
                            "torso_wrench_assistance_valid": bool(
                                metrics["torso_wrench_assistance_valid"]
                            ),
                        }
                    )
                if actor_learned_torso_wrench:
                    diag_entry.update(
                        {
                            "learned_torso_wrench_scale": float(
                                metrics["learned_torso_wrench_scale"]
                            ),
                            "learned_torso_wrench_rms_force": float(
                                metrics["learned_torso_wrench_rms_force"]
                            ),
                            "learned_torso_wrench_rms_torque": float(
                                metrics["learned_torso_wrench_rms_torque"]
                            ),
                            "learned_torso_wrench_max_force": float(
                                metrics["learned_torso_wrench_max_force"]
                            ),
                            "learned_torso_wrench_max_torque": float(
                                metrics["learned_torso_wrench_max_torque"]
                            ),
                            "learned_torso_wrench_normalized_rms": float(
                                metrics[
                                    "learned_torso_wrench_normalized_rms"
                                ]
                            ),
                            "learned_torso_wrench_saturation_fraction": float(
                                metrics[
                                    "learned_torso_wrench_saturation_fraction"
                                ]
                            ),
                            "learned_torso_wrench_penalty": float(
                                metrics["learned_torso_wrench_penalty"]
                            ),
                            "learned_torso_wrench_valid": bool(
                                metrics["learned_torso_wrench_valid"]
                            ),
                        }
                    )
                if actor_counterfactual_wrench_distillation:
                    diag_entry.update(
                        build_counterfactual_wrench_telemetry(metrics)
                    )
                if adaptive_phase_sampling:
                    if not bool(metrics["adaptive_phase_sampling_valid"]):
                        raise RuntimeError(
                            "adaptive phase sampling telemetry is invalid"
                        )
                    diag_entry.update(
                        {
                            "adaptive_phase_failure_ema": np.asarray(
                                metrics["adaptive_phase_failure_ema"]
                            ).tolist(),
                            "adaptive_phase_probabilities": np.asarray(
                                metrics["adaptive_phase_probabilities"]
                            ).tolist(),
                            "adaptive_phase_terminal_bin_counts": np.asarray(
                                metrics[
                                    "adaptive_phase_terminal_bin_counts"
                                ]
                            ).tolist(),
                            "adaptive_phase_min_probability": float(
                                metrics["adaptive_phase_min_probability"]
                            ),
                            "adaptive_phase_sampling_valid": bool(
                                metrics["adaptive_phase_sampling_valid"]
                            ),
                        }
                    )
                diag_log.append(diag_entry)
            else:
                print(
                    f"{state.step:7d} | {reward:7.3f} | {trk_vx:7.3f} | {trk_vy:7.3f} | "
                    f"{trk_yaw:7.3f} | "
                    f"{metrics['actor_grad']:7.2f} | {metrics['critic_loss']:7.4f} | "
                    f"{diff:5.2f} | {status}"
                )
            if jave_enabled:
                print(
                    "         JAVE "
                    f"ldm={jave_telemetry['jave_ldm_loss']:.4f} "
                    f"vg={jave_telemetry['jave_vg_loss']:.4f} "
                    f"target={jave_telemetry['jave_vg_target_norm']:.4f} "
                    f"active={int(jave_telemetry['jave_vg_active'])} "
                    f"replay={jave_telemetry['jave_replay_size']}"
                )

            if DEBUG_FOOT_CONTACTS:
                print(
                    f"         foot GRF (N):  "
                    f"FL={float(metrics['foot_normal_FL']):7.2f}  "
                    f"FR={float(metrics['foot_normal_FR']):7.2f}  "
                    f"RL={float(metrics['foot_normal_RL']):7.2f}  "
                    f"RR={float(metrics['foot_normal_RR']):7.2f}"
                )

            log.append(
                [
                    state.step,
                    reward,
                    vel_x,
                    vel_y,
                    yaw_rate,
                    cmd_x,
                    cmd_y,
                    cmd_yaw,
                    float(metrics["actor_loss"]),
                    float(metrics["contact"]),
                    float(metrics["actor_grad"]),
                    float(metrics["critic_loss"]),
                    float(metrics["track_vx_sq"]),
                    float(metrics["track_vy_sq"]),
                    float(metrics["track_yaw_sq"]),
                    diff,
                ]
            )

            # Save best policy
            if (
                reward > best_reward
                and state.step > 5000
                and not torso_wrench_assistance
                and not actor_learned_torso_wrench
                and not actor_counterfactual_wrench_distillation
                and demonstration_replay_threshold is None
            ):
                best_reward = reward
                with open(f"{save_dir}/policy_best.pkl", "wb") as f:
                    pickle.dump(state, f)
                print(f"  >> New best! Reward: {best_reward:.3f}")

        current_step = (i + 1) * steps_per_iter
        hparams["best_reward"] = best_reward
        if checkpoint_steps is None:
            last_checkpoint_step, checkpoint_path = (
                archive_periodic_checkpoint_if_due(
                    state,
                    save_dir,
                    last_checkpoint_step,
                    checkpoint_interval,
                    current_step=current_step,
                    hparams=hparams,
                )
            )
        elif current_step in checkpoint_steps:
            persist_run_hparams(save_dir, hparams)
            checkpoint_path = save_periodic_checkpoint(
                state,
                save_dir,
                current_step,
            )
            last_checkpoint_step = current_step
        else:
            checkpoint_path = None
        if should_persist_checkpoint_metrics(
            checkpoint_path,
            actor_cagrad=actor_cagrad,
            frozen_preview_treatment=frozen_preview_treatment,
            ahac=ahac,
            actor_contact_topology_gradient_truncation=(
                actor_contact_topology_gradient_truncation
            ),
            demonstration_replay=(
                demonstration_replay_threshold is not None
            ),
            jave=jave_enabled,
        ):
            checkpoint_metrics = {
                "step": int(current_step),
                "action_noise_current": action_noise_std_hparam(
                    np.asarray(metrics["action_noise_current"])
                ),
            }
            if actor_cagrad:
                checkpoint_metrics.update(
                    build_checkpoint_cagrad_telemetry(metrics)
                )
            if jave_enabled:
                checkpoint_metrics.update(jave_telemetry)
            if ahac:
                checkpoint_metrics.update(
                    build_checkpoint_ahac_telemetry(metrics)
                )
            if actor_contact_topology_gradient_truncation:
                checkpoint_metrics.update(
                    {
                        "actor_contact_topology_gradient_truncation": True,
                        "contact_topology_event_count": int(
                            metrics["contact_topology_event_count"]
                        ),
                        "contact_topology_event_fraction": float(
                            metrics["contact_topology_event_fraction"]
                        ),
                    }
                )
            if demonstration_replay_threshold is not None:
                checkpoint_metrics.update(
                    build_checkpoint_demonstration_replay_telemetry(
                        metrics,
                        threshold=demonstration_replay_threshold,
                    )
                )
            if recovery_teacher_enabled:
                checkpoint_metrics.update(
                    build_checkpoint_recovery_teacher_telemetry(
                        metrics,
                        max_ratio=actor_recovery_teacher_gradient_ratio,
                    )
                )
            if actor_counterfactual_wrench_distillation:
                checkpoint_metrics.update(
                    build_counterfactual_wrench_telemetry(metrics)
                )
            if actor_policy_anchor_weight > 0.0:
                checkpoint_metrics.update(
                    build_policy_anchor_telemetry(
                        metrics, weight=actor_policy_anchor_weight
                    )
                )
            if frozen_preview_treatment:
                checkpoint_metrics.update(
                    {
                        "actor_preview_gradient_norm": float(
                            metrics["actor_preview_gradient_norm"]
                        ),
                        "actor_preview_update_norm": float(
                            metrics["actor_preview_update_norm"]
                        ),
                        "actor_preview_frozen_parameter_drift_max_abs": float(
                            metrics[
                                "actor_preview_frozen_parameter_drift_max_abs"
                            ]
                        ),
                        "actor_preview_frozen_moment_drift_max_abs": float(
                            metrics[
                                "actor_preview_frozen_moment_drift_max_abs"
                            ]
                        ),
                        "actor_preview_normalizer_drift_max_abs": float(
                            metrics[
                                "actor_preview_normalizer_drift_max_abs"
                            ]
                        ),
                        "actor_preview_bin_counts": np.asarray(
                            metrics["actor_preview_bin_counts"]
                        ).tolist(),
                        "actor_preview_bin_action_deviation_mean_abs": np.asarray(
                            metrics[
                                "actor_preview_bin_action_deviation_mean_abs"
                            ]
                        ).tolist(),
                        "actor_preview_bin_action_deviation_max_abs": np.asarray(
                            metrics[
                                "actor_preview_bin_action_deviation_max_abs"
                            ]
                        ).tolist(),
                        "actor_preview_valid": bool(
                            metrics["actor_preview_valid"]
                        ),
                        **(
                            {
                                "torso_wrench_assistance_scale_current": float(
                                    metrics[
                                        "torso_wrench_assistance_scale_current"
                                    ]
                                ),
                                "torso_wrench_assistance_active_fraction": float(
                                    metrics[
                                        "torso_wrench_assistance_active_fraction"
                                    ]
                                ),
                                "torso_wrench_assistance_rms_force": float(
                                    metrics[
                                        "torso_wrench_assistance_rms_force"
                                    ]
                                ),
                                "torso_wrench_assistance_rms_torque": float(
                                    metrics[
                                        "torso_wrench_assistance_rms_torque"
                                    ]
                                ),
                                "torso_wrench_assistance_max_force": float(
                                    metrics[
                                        "torso_wrench_assistance_max_force"
                                    ]
                                ),
                                "torso_wrench_assistance_max_torque": float(
                                    metrics[
                                        "torso_wrench_assistance_max_torque"
                                    ]
                                ),
                                "torso_wrench_assistance_valid": bool(
                                    metrics[
                                        "torso_wrench_assistance_valid"
                                    ]
                                ),
                            }
                            if torso_wrench_assistance
                            else {}
                        ),
                        **(
                            {
                                "learned_torso_wrench_scale": float(
                                    metrics["learned_torso_wrench_scale"]
                                ),
                                "learned_torso_wrench_rms_force": float(
                                    metrics[
                                        "learned_torso_wrench_rms_force"
                                    ]
                                ),
                                "learned_torso_wrench_rms_torque": float(
                                    metrics[
                                        "learned_torso_wrench_rms_torque"
                                    ]
                                ),
                                "learned_torso_wrench_max_force": float(
                                    metrics[
                                        "learned_torso_wrench_max_force"
                                    ]
                                ),
                                "learned_torso_wrench_max_torque": float(
                                    metrics[
                                        "learned_torso_wrench_max_torque"
                                    ]
                                ),
                                "learned_torso_wrench_normalized_rms": float(
                                    metrics[
                                        "learned_torso_wrench_normalized_rms"
                                    ]
                                ),
                                "learned_torso_wrench_saturation_fraction": float(
                                    metrics[
                                        "learned_torso_wrench_saturation_fraction"
                                    ]
                                ),
                                "learned_torso_wrench_penalty": float(
                                    metrics[
                                        "learned_torso_wrench_penalty"
                                    ]
                                ),
                                "learned_torso_wrench_valid": bool(
                                    metrics[
                                        "learned_torso_wrench_valid"
                                    ]
                                ),
                            }
                            if actor_learned_torso_wrench
                            else {}
                        ),
                        **(
                            {
                                "actor_muon_kernel_gradient_norm": float(
                                    metrics[
                                        "actor_muon_kernel_gradient_norm"
                                    ]
                                ),
                                "actor_muon_kernel_update_norm": float(
                                    metrics[
                                        "actor_muon_kernel_update_norm"
                                    ]
                                ),
                                "actor_muon_aux_adam_gradient_norm": float(
                                    metrics[
                                        "actor_muon_aux_adam_gradient_norm"
                                    ]
                                ),
                                "actor_muon_aux_adam_update_norm": float(
                                    metrics[
                                        "actor_muon_aux_adam_update_norm"
                                    ]
                                ),
                            }
                            if residual_muon_treatment
                            else {}
                        ),
                    }
                )
            if actor_capture_point_tracking:
                checkpoint_metrics.update(
                    {
                        "actor_capture_point_loss": float(
                            metrics["actor_capture_point_loss"]
                        ),
                        "actor_capture_point_valid_count": int(
                            metrics["actor_capture_point_valid_count"]
                        ),
                        "actor_capture_point_p99_norm": float(
                            metrics["actor_capture_point_p99_norm"]
                        ),
                        "actor_capture_point_component_rms": np.asarray(
                            metrics["actor_capture_point_component_rms"]
                        ).tolist(),
                    }
                )
            if recovery_support is not None:
                checkpoint_metrics.update(
                    {
                        "actor_recovery_gate_activation_fraction": float(
                            metrics[
                                "actor_recovery_gate_activation_fraction"
                            ]
                        ),
                        "actor_recovery_gate_max": float(
                            metrics["actor_recovery_gate_max"]
                        ),
                        "actor_recovery_carried_activation_fraction": float(
                            metrics[
                                "actor_recovery_carried_activation_fraction"
                            ]
                        ),
                        "actor_recovery_reference_activation_fraction": float(
                            metrics[
                                "actor_recovery_reference_activation_fraction"
                            ]
                        ),
                        "actor_recovery_gated_residual_rms": float(
                            metrics[
                                "actor_recovery_gated_residual_rms"
                            ]
                        ),
                        "actor_recovery_gated_residual_max_abs": float(
                            metrics[
                                "actor_recovery_gated_residual_max_abs"
                            ]
                        ),
                        "actor_recovery_valid": bool(
                            metrics["actor_recovery_valid"]
                        ),
                    }
                )
            persist_checkpoint_phase_metric(save_dir, checkpoint_metrics)
        if checkpoint_path is not None:
            print(f"  >> Checkpoint saved at step {current_step}")

    # Save final state and logs
    with open(f"{save_dir}/policy_final.pkl", "wb") as f:
        pickle.dump(state, f)
    np.save(f"{save_dir}/log.npy", np.array(log))

    if diagnose and diag_log:
        with open(f"{save_dir}/diag_log.json", "w") as f:
            json.dump(diag_log, f, indent=2)
        print(f"Diagnostic log saved to {save_dir}/diag_log.json")

    elapsed = time.time() - start
    cmd_str = (
        f"vx=[{cmd_vel_x_range[0]:.2f},{cmd_vel_x_range[1]:.2f}], "
        f"vy=[{cmd_vel_y_range[0]:.2f},{cmd_vel_y_range[1]:.2f}], "
        f"yaw=[{cmd_yaw_rate_range[0]:.2f},{cmd_yaw_rate_range[1]:.2f}] "
        f"| zero_prob={cmd_zero_prob} interval={cmd_ctrl_interval_range}"
    )
    print("=" * (160 if diagnose else 120))
    print(f"Training complete in {elapsed:.1f}s (compilation: {compile_time:.1f}s)")
    print(f"Command ranges: {cmd_str}")
    print("=" * 100)
    print(f"Training complete in {elapsed:.1f}s (compile: {compile_time:.1f}s)")
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum_steps={curriculum_steps}, "
        f"terrain={'ON' if terrain else 'OFF'}, terrain_flat_prob={terrain_flat_prob}, "
        f"terrain_slope_max={terrain_slope_max} deg, "
        f"terrain_bump_std={terrain_bump_std if terrain else 0.0}"
    )
    print(f"Best reward: {best_reward:.3f}")

    hparams["best_reward"] = best_reward
    persist_run_hparams(save_dir, hparams)

    return state, save_dir
