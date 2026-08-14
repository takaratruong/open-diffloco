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

from src.core.data_structures import Normalizer, TrainState
from src.core.networks import Actor, Critic
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
from src.core.utils import compute_grad_norm
from src.algorithms.shac.gradients import (
    aggregate_per_env_gradients,
    per_env_gradient_statistics,
)
from src.algorithms.shac.cagrad import (
    accumulate_phase_gradients,
    combine_cagrad,
    finalize_phase_gradients,
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
    split_residual_adapter_params,
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


# Set to True to enable per-foot normal force logging.
DEBUG_FOOT_CONTACTS = False


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
) -> bool:
    """Persist validity evidence for every checkpointed CAGrad treatment."""
    return checkpoint_path is not None and (
        actor_cagrad or frozen_preview_treatment
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


def resolve_residual_preview_adapter_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: bool,
    requested_hidden: int,
    requested_optimizer: str,
    future_reference_upgrade: bool,
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
    if (
        saved != requested
        or (saved and saved_hidden != requested_hidden)
        or saved_optimizer != requested_optimizer
    ):
        raise ValueError(
            "actor residual preview settings must match the checkpoint"
        )
    return saved, int(saved_hidden), saved_optimizer


def select_initial_training_state(*, initialized_state, resumed_state):
    """Select the full saved state for an exact continuation."""
    return resumed_state if resumed_state is not None else initialized_state


def migrate_env_state_metrics(resumed_env_state, initialized_env_state):
    """Align diagnostic metric leaves without changing physical state."""
    resumed_metrics = resumed_env_state.metrics
    initialized_metrics = initialized_env_state.metrics
    metrics = {
        name: resumed_metrics.get(name, initial_value)
        for name, initial_value in initialized_metrics.items()
    }
    return resumed_env_state.replace(metrics=metrics)


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


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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
) -> tuple[bool, float, int, int]:
    """Restore CAGrad checkpoints while allowing legacy treatment starts."""
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


def validate_actor_policy_anchor_configuration(
    *,
    weight: float,
    initial_full_actor_policy,
    resume_from,
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
    if weight > 0.0 and initial_full_actor_policy is None:
        raise ValueError("actor policy anchoring requires a full actor parent")
    if weight > 0.0 and resume_from is not None:
        raise ValueError("actor policy anchoring currently requires a fresh run")


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
    max_episode_length: int = 5000,
    actor_history_len: int = 10,
    actor_observation_noise: bool = False,
    actor_reference_lookahead_steps: tuple[int, ...] = (),
    actor_reference_preview_mode: str = "absolute",
    allow_resume_actor_reference_lookahead_upgrade: bool = False,
    actor_preview_adapter: bool = False,
    actor_residual_preview_adapter: bool = False,
    actor_residual_preview_hidden: int = 256,
    actor_residual_preview_optimizer: str = "adam",
    env_variant: str = "blind_nolinvel_nokinref",
    actor_per_env_grad_clip: float = None,
    allow_resume_actor_per_env_grad_clip_change: bool = False,
    critic_per_env_grad_clip: float = None,
    actor_phase_robust_weighting: bool = False,
    actor_phase_bin_count: int = 5,
    actor_phase_robust_fraction: float = 0.5,
    actor_cagrad: bool = False,
    actor_cagrad_alpha: float = 0.5,
    actor_cagrad_iterations: int = 32,
    adaptive_phase_sampling: bool = False,
    adaptive_phase_uniform_ratio: float = 0.5,
    adaptive_phase_alpha: float = 0.001,
    actor_bootstrap_scale: float = 1.0,
    actor_bootstrap_delay_steps: int = 0,
    allow_resume_actor_bootstrap_scale_change: bool = False,
    actor_hidden: tuple[int, ...] = (512, 256, 128),
    actor_layer_norm: bool = True,
    actor_zero_output: bool = True,
    source_actor_policy=None,
    initial_full_actor_policy=None,
    actor_policy_anchor_weight: float = 0.0,
    residual_action_scale: float = 0.0,
    differentiate_source_feedback: bool = True,
    effort_limit_scale: float = 1.0,
    termination_margin_weight: float = 0.0,
    allow_resume_termination_margin_change: bool = False,
    reference_reset_noise_scale: float = 0.0,
    reference_root_reset_noise_multiplier: float = 1.0,
    reference_root_reset_noise_probability: float = 0.0,
    allow_resume_reference_root_reset_noise_change: bool = False,
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
    reference_path: str | None = None,
    reference_stride: int | None = None,
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
        allow_resume_actor_bootstrap_scale_change: Explicitly permit changing
            actor terminal-value scale when resuming a checkpoint.
        actor_bootstrap_delay_steps: Environment steps before the actor uses
            target-critic terminal value estimates.

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
    validate_actor_policy_anchor_configuration(
        weight=actor_policy_anchor_weight,
        initial_full_actor_policy=initial_full_actor_policy,
        resume_from=resume_from,
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
        actor_cagrad=actor_cagrad,
        history_len=actor_history_len,
        source_actor_policy=source_actor_policy,
        initial_full_actor_policy=initial_full_actor_policy,
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
    effective_num_envs = num_envs * gradient_accumulation_steps
    steps_per_actor_update = effective_num_envs * unroll_length

    # Handle checkpoint resumption
    resumed_state = None
    resumed_step = 0
    resumed_hparams = None
    future_reference_upgrade = False

    if resume_from:
        resumed_state, resumed_hparams, resumed_step = load_checkpoint(resume_from)
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
            reference_path = resumed_hparams.get(
                "reference_path", reference_path
            )
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
            actor_cagrad=actor_cagrad,
            history_len=actor_history_len,
            source_actor_policy=source_actor_policy,
            initial_full_actor_policy=initial_full_actor_policy,
            env_variant=env_variant,
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
    torso_body_id = -1
    torso_slot = -1
    torso_wrench_parameters = None
    if torso_wrench_assistance:
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
    save_dir = f"training_runs/shac_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Algorithm: SHAC, Save dir: {save_dir}")
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
    critic = Critic()

    actor_dummy = jp.zeros((1, env.actor_obs_dim), dtype=jp.float32)
    critic_dummy = jp.zeros((1, env.critic_obs_dim), dtype=jp.float32)
    actor_params = (
        initial_full_actor_policy
        if initial_full_actor_policy is not None
        else actor.init(k1, actor_dummy)
    )
    migration_report = None
    residual_adapter_report = None
    residual_muon_report = None
    conditioning_report = None
    preview_adapter_mask = None
    preview_legacy_frame_dim = 0
    preview_trainable_parameter_count = 0
    frozen_preview_treatment = bool(
        actor_preview_adapter or actor_residual_preview_adapter
    )
    residual_muon_treatment = bool(
        actor_residual_preview_adapter
        and actor_residual_preview_optimizer == "muon"
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

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

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

    _push_interval_steps = max(int(round(push_interval_s / env.dt)), 1)
    _push_velocity_lo = jp.array(push_velocity_range[0], dtype=jp.float64)
    _push_velocity_hi = jp.array(push_velocity_range[1], dtype=jp.float64)
    _foot_body_ids = env._foot_body_ids
    _nominal_weight = env.nominal_total_mass * env.base_gravity_mag
    _terrain_bump_std = terrain_bump_std if terrain else 0.0

    def actor_loss(
        actor_params,
        target_critic_params,
        actor_norm_state,
        critic_norm_state,
        env_state,
        randomization,
        current_noise_std,
        current_actor_bootstrap_scale,
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
            noise_t, velocity_push_t, terrain_bump_innov_t = inputs

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
            residual_logits = None
            if actor_residual_preview_adapter:
                action, parent_action, _residual_action = (
                    apply_frozen_preview_residual(
                        actor,
                        residual_preview_actor,
                        actor_params,
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
            if actor_residual_preview_adapter:
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
                if actor_residual_preview_adapter or actor_preview_adapter:
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

            if adaptive_phase_sampling or frozen_preview_treatment:
                transition_phase = transition_phase_before_reset(
                    state.info["phase"],
                    reference_stride=env.reference_stride,
                    reference_length=env.reference_length,
                )
            next_state = env.step(state, noisy_action)
            foot_bump_ou = jp.where(next_state.done, jp.zeros((4, 3)), foot_bump_ou)

            transition = {
                "reward": next_state.reward,
                "done": next_state.done,
                "terminal": next_state.info["terminal"],
                "actor_obs": state.obs,
                "critic_obs": env._get_critic_obs(state.data, state.info),
                "bootstrap_critic_obs": next_state.info["bootstrap_critic_obs"],
                "vel_x": next_state.metrics["vel_x"],
                "vel_y": next_state.metrics["vel_y"],
                "yaw_rate": next_state.metrics["yaw_rate"],
                "cmd_x": next_state.metrics["cmd_x"],
                "cmd_y": next_state.metrics["cmd_y"],
                "cmd_yaw": next_state.metrics["cmd_yaw"],
                "height": next_state.metrics["height"],
                "tilt": next_state.metrics["tilt"],
                "foot_normal_FL": next_state.metrics["foot_normal_FL"],
                "foot_normal_FR": next_state.metrics["foot_normal_FR"],
                "foot_normal_RL": next_state.metrics["foot_normal_RL"],
                "foot_normal_RR": next_state.metrics["foot_normal_RR"],
                "actor_policy_anchor_squared_error": (
                    actor_policy_anchor_squared_error
                ),
                "actor_policy_anchor_penalty": (
                    actor_policy_anchor_step_penalty
                ),
            }
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
            if torso_wrench_assistance:
                transition["torso_wrench"] = torso_wrench
            return (next_state, foot_bump_ou), transition

        env_state = jax.lax.stop_gradient(env_state)

        (final_state, final_foot_bump_ou), traj = jax.lax.scan(
            rollout_step,
            (env_state, env_state.info["foot_bump_ou"]),
            (action_noise, velocity_pushes, terrain_bump_innovations),
            length=unroll_length,
        )
        final_state = final_state.replace(
            info={**final_state.info, "foot_bump_ou": final_foot_bump_ou}
        )

        bootstrap_obs = critic_norm.normalize(
            critic_norm_state, traj["bootstrap_critic_obs"]
        ).astype(jp.float32)
        bootstrap_v = squeeze_value_head(
            critic.apply(target_critic_params, bootstrap_obs)
        )

        # Accumulate discounted returns, handling episode boundaries. Time-limit
        # truncations bootstrap from the pre-reset observation stored by env.step.
        def accum_return(carry, x):
            total, running, discount = carry
            r, done, terminal, v_next = x
            next_discount = discount * gamma
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
            (traj["reward"], traj["done"], traj["terminal"], bootstrap_v),
        )

        final_obs = critic_norm.normalize(
            critic_norm_state,
            env._get_critic_obs(final_state.data, final_state.info),
        ).astype(jp.float32)
        final_v = squeeze_value_head(
            critic.apply(target_critic_params, final_obs)
        )
        final_bootstrap = jp.where(
            traj["done"][-1],
            0.0,
            current_actor_bootstrap_scale * final_discount * final_v,
        )

        total_ret = total_ret + running + final_bootstrap

        anchor_loss = jp.mean(traj["actor_policy_anchor_penalty"])
        return -total_ret / unroll_length + anchor_loss, (traj, final_state)

    def critic_loss_from_data(
        critic_params,
        target_critic_params,
        critic_norm_state,
        traj_obs,
        traj_rewards,
        traj_dones,
        traj_terminals,
        traj_bootstrap_obs,
        final_obs,
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
        values = squeeze_value_head(
            critic.apply(critic_params, flat_obs_norm)
        )  # (H,)

        next_v = squeeze_value_head(
            critic.apply(target_critic_params, flat_bootstrap_obs_norm)
        )
        final_v = squeeze_value_head(
            critic.apply(target_critic_params, final_obs_norm)
        )  # scalar

        rewards = traj_rewards.reshape(-1).astype(jp.float32)  # (H,)
        dones = traj_dones.reshape(-1).astype(jp.float32)  # (H,)
        terminals = traj_terminals.reshape(-1).astype(jp.float32)  # (H,)

        def scan_fn(g_next, inputs):
            r"""TD(lambda) backward scan."""
            r, done, terminal, v_next = inputs
            g_normal = r + gamma * (
                (1.0 - gae_lambda) * v_next + gae_lambda * g_next
            )  # Normal step
            g_trunc = r + gamma * v_next  # Time-limit trunc.
            g_term = r  # true term.
            g = jp.where(terminal, g_term, jp.where(done, g_trunc, g_normal))
            return g, g

        _, targets_reversed = jax.lax.scan(
            scan_fn,
            final_v,  # float32 scalar (determines the carry dtype)
            (rewards[::-1], dones[::-1], terminals[::-1], next_v[::-1]),
        )
        targets = targets_reversed[::-1]

        return jp.mean(jp.square(values - jax.lax.stop_gradient(targets)))

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

        # Actor update
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        if gradient_accumulation_steps == 1:
            (losses, (trajs, final_states)), per_env_grads = jax.vmap(
                actor_grad_fn,
                in_axes=(None, None, None, None, 0, 0, None, None),
            )(
                state.actor_params,
                state.target_critic_params,
                state.normalizer,
                state.critic_normalizer,
                updated_env_state,
                all_randomization,
                current_noise_std,
                current_actor_bootstrap_scale,
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
                    in_axes=(None, None, None, None, 0, 0, None, None),
                )(
                    state.actor_params,
                    state.target_critic_params,
                    state.normalizer,
                    state.critic_normalizer,
                    shard_env_state,
                    shard_randomization,
                    current_noise_std,
                    current_actor_bootstrap_scale,
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

        actor_grad_norm = compute_grad_norm(grads)

        if (
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

        # Critic updates
        all_obs = trajs["critic_obs"]
        all_rewards = trajs["reward"]
        all_dones = trajs["done"]
        all_terminals = trajs["terminal"]
        all_bootstrap_obs = trajs["bootstrap_critic_obs"]
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
        ):
            return critic_loss_from_data(
                critic_params,
                target_critic_params,
                norm_state,
                obs,
                rewards,
                dones,
                terminals,
                bootstrap_obs,
                final_obs,
            )

        def critic_update_step(carry, _):
            c_params, c_opt_state = carry

            critic_grad_fn = jax.value_and_grad(
                single_env_critic_loss, argnums=0
            )
            if gradient_accumulation_steps == 1:
                c_losses, c_per_env_grads = jax.vmap(
                    critic_grad_fn,
                    in_axes=(None, None, None, 0, 0, 0, 0, 0, 0),
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
                    ) = shard_inputs
                    shard_losses, shard_per_env_grads = jax.vmap(
                        critic_grad_fn,
                        in_axes=(None, None, None, 0, 0, 0, 0, 0, 0),
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

            c_updates, new_c_opt = critic_opt.update(c_grads, c_opt_state)
            new_c_params = optax.apply_updates(c_params, c_updates)

            return (new_c_params, new_c_opt), {
                "loss": jp.nanmean(c_losses),
                "finite_fraction": critic_grad_stats["finite_fraction"],
                "raw_norm_median": critic_grad_stats["raw_norm_median"],
                "raw_norm_max": critic_grad_stats["raw_norm_max"],
            }

        (new_critic_params, new_critic_opt), critic_update_metrics = jax.lax.scan(
            critic_update_step,
            (state.critic_params, state.critic_opt),
            None,
            length=critic_iterations,
        )

        # Soft target update
        new_target = optax.incremental_update(
            new_critic_params, state.target_critic_params, target_update_rate
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
        }
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
                    "actor_cagrad_combined_norm": actor_grad_norm,
                    "actor_cagrad_valid": (
                        cagrad_reduction["valid"]
                        & cagrad_loss_diagnostics["valid"]
                        & cagrad_counts_match
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
        step=canonicalize_step_dtype(0),
    )
    if migration_report is not None:
        persist_future_reference_migration_report(
            save_dir, migration_report
        )
    if resumed_state is not None:
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
        if actor_residual_preview_adapter:
            if future_reference_upgrade:
                parent_params = resumed_state.actor_params
                parent_optimizer_state = resumed_state.actor_opt
                adapter_params = residual_preview_actor.init(
                    jax.random.fold_in(k1, 0x5250),
                    jp.zeros(
                        (1, env.actor_frame_obs_dim), dtype=jp.float32
                    ),
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
        if adaptive_phase_sampling:
            resumed_state = resumed_state.replace(
                env_state=migrate_adaptive_phase_env_state(
                    resumed_state.env_state,
                    reference_length=env.reference_length,
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
        "algorithm": "shac",
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
        "allow_resume_assistance_conditioning_change": (
            allow_resume_assistance_conditioning_change
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
        "best_reward": best_reward,
        "max_episode_length": max_episode_length,
        "actor_history_len": actor_history_len,
        "actor_observation_noise": actor_observation_noise,
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
        "critic_per_env_grad_clip": critic_per_env_grad_clip,
        "actor_phase_robust_weighting": actor_phase_robust_weighting,
        "actor_phase_bin_count": actor_phase_bin_count,
        "actor_phase_robust_fraction": actor_phase_robust_fraction,
        "actor_cagrad": actor_cagrad,
        "actor_cagrad_alpha": actor_cagrad_alpha,
        "actor_cagrad_iterations": actor_cagrad_iterations,
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
        "actor_kind": (
            "flax_residual_preview"
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

    for i in range(start_iter, total_iters):
        state, metrics = train_step(state)

        if actor_cagrad and not bool(metrics["actor_cagrad_valid"]):
            raise RuntimeError("actor CAGrad aggregation is invalid")
        if frozen_preview_treatment and not bool(metrics["actor_preview_valid"]):
            raise RuntimeError("actor preview adapter telemetry is invalid")
        if torso_wrench_assistance and not bool(
            metrics["torso_wrench_assistance_valid"]
        ):
            raise RuntimeError("torso wrench assistance telemetry is invalid")
        if actor_policy_anchor_weight > 0.0 and not np.isfinite(
            float(metrics["actor_policy_anchor_squared_error"])
        ):
            raise RuntimeError("actor policy anchor telemetry is invalid")

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
                }
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
            ):
                best_reward = reward
                with open(f"{save_dir}/policy_best.pkl", "wb") as f:
                    pickle.dump(state, f)
                print(f"  >> New best! Reward: {best_reward:.3f}")

        current_step = (i + 1) * steps_per_iter
        hparams["best_reward"] = best_reward
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
        if should_persist_checkpoint_metrics(
            checkpoint_path,
            actor_cagrad=actor_cagrad,
            frozen_preview_treatment=frozen_preview_treatment,
        ):
            checkpoint_metrics = {
                "step": int(current_step),
                "action_noise_current": action_noise_std_hparam(
                    np.asarray(metrics["action_noise_current"])
                ),
                **build_checkpoint_cagrad_telemetry(metrics),
            }
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
