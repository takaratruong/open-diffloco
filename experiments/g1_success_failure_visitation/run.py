"""Capture successful-PPO and failing-DiffSim MJX visitation on one phase grid.

This is a read-only diagnostic.  It loads the frozen corrected-long PPO control
and retained E002 DiffSim policy, runs both under their exact observation/action
contracts on the same corrected reference and nominal MJX plant, and stores the
state, action-target, contact, reward, and termination-margin trajectories.  It
does not differentiate, update an optimizer, or retain a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from src.core.data_structures import Normalizer
from src.core.rmr_policy import apply_trainable_rmr_policy
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_rmr_phase_grid import select_rmr_policy_observation
from tools.evaluate_g1_tracking import (
    _load_policy,
    build_compiled_step,
    configure_jax,
    make_evaluation_env,
)
from tools.run_g1_tracking_rmr50_shac import load_source_actor_policy


PHASES = (0, 25, 50, 75, 100)
PPO_SURVIVAL = (271, 246, 221, 196, 171)
DIFFSIM_SURVIVAL = (136, 144, 84, 90, 79)
TERMINATION_LIMITS = np.asarray((0.25, 1.3, 0.8, 0.4), dtype=np.float64)
TERMINATION_METRICS = (
    "termination_anchor_z_error",
    "termination_anchor_xy_error",
    "termination_gravity_z_error",
    "termination_distal_z_error",
)
TRACKING_METRICS = (
    "anchor_position_error",
    "anchor_orientation_error",
    "body_position_error",
    "body_orientation_error",
    "body_linear_velocity_error",
    "body_angular_velocity_error",
)
TRANSITION_METRICS = (
    *TRACKING_METRICS,
    *TERMINATION_METRICS,
    "contact_force",
    "contact_stiffness",
)

REFERENCE_SHA256 = "f47d13b431d85a273eba6022f5a28bd55cae7c788112baf0778ab159914a039c"
PPO_CHECKPOINT_SHA256 = (
    "45b179d2107774d76e2337adbd800d21362cd431324eb9820389010821767703"
)
DIFFSIM_CHECKPOINT_SHA256 = (
    "52aa142dabf382671a5fe7e6b1f26954b77e4fde492bb413a25b85358a1c4325"
)
DIFFSIM_HPARAMS_SHA256 = (
    "79927f89ef75cf0a6fbfd5c92746a59db587c00319db780dcad702f0c3bbd5eb"
)
SOURCE_PPO_GRID_SHA256 = (
    "5bb86e70b49fca6571c200e29db9d48b1e3964d1d66997ae2797d7fe7b0e316a"
)
SOURCE_DIFFSIM_GRID_SHA256 = (
    "9f459abcdeb6982cd80839b1f1cf2d1ce3152cf9878bf4c79b1434513af60bd9"
)
SOURCE_E006_BACKEND_REPRODUCTION_SHA256 = (
    "e07b201f4d0b3ca8eabd6b0862ba29fdd0615bb82e40347a55c3d222835b2872"
)
SOURCE_E006_AUDIT_SHA256 = (
    "f375ed244a8b37c9bfcf2c693a84218e0dc7c084b6b656d435f922a2927cfb67"
)


def sha256_file(path: Path) -> str:
    """Hash one required source or output artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a strict JSON object."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write strict JSON atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed trajectory archive atomically."""
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def finite_float(value: object, name: str) -> float:
    """Return one JSON-safe finite scalar."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def first_true(values: np.ndarray) -> int | None:
    """Return the first true row or null."""
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(indices[0]) if indices.size else None


def rows_differ(left: np.ndarray, right: np.ndarray, *, name: str) -> np.ndarray:
    """Reduce an arbitrary per-row signature to one difference bit per row."""
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape or left_values.ndim < 2:
        raise ValueError(f"{name} traces have incompatible shapes")
    return np.any(
        left_values != right_values,
        axis=tuple(range(1, left_values.ndim)),
    )


def selected_boundary_indices(
    length: int, event_indices: tuple[int | None, ...]
) -> list[int]:
    """Select deterministic coverage rows plus any first contact events."""
    if length < 1:
        raise ValueError("boundary selection requires at least one row")
    last = length - 1
    indices = {
        0,
        last // 4,
        last // 2,
        (3 * last) // 4,
        last,
    }
    indices.update(
        index for index in event_indices if index is not None and 0 <= index < length
    )
    return sorted(indices)


def termination_margin(metric_values: np.ndarray) -> np.ndarray:
    """Return the smallest remaining hard-termination margin per transition."""
    values = np.asarray(metric_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < len(TRANSITION_METRICS):
        raise ValueError("transition metric matrix has the wrong shape")
    start = len(TRACKING_METRICS)
    errors = values[:, start : start + len(TERMINATION_METRICS)]
    return np.min(TERMINATION_LIMITS[None, :] - errors, axis=1)


def support_code(support: np.ndarray) -> np.ndarray:
    """Encode left/right support bits as none/left/right/double integers."""
    values = np.asarray(support, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("foot support must have shape (N, 2)")
    return values[:, 0].astype(np.int64) + 2 * values[:, 1].astype(np.int64)


def summarize_trace(
    trace: Mapping[str, np.ndarray], *, phase: int, remaining: int
) -> dict[str, object]:
    """Summarize one complete or terminal strict rollout."""
    phases = np.asarray(trace["phase"], dtype=np.int64)
    metrics = np.asarray(trace["metrics"], dtype=np.float64)
    terminal = np.asarray(trace["terminal"], dtype=bool)
    done = np.asarray(trace["done"], dtype=bool)
    actions = np.asarray(trace["model_action"], dtype=np.float64)
    targets = np.asarray(trace["position_target"], dtype=np.float64)
    support = support_code(np.asarray(trace["foot_support"], dtype=bool))
    if phases.shape != (len(metrics),) or len(phases) < 1:
        raise ValueError("trace rows are empty or misaligned")
    if not np.array_equal(phases, np.arange(phase, phase + len(phases))):
        raise ValueError("trace phases are not contiguous")
    if terminal.shape != phases.shape or done.shape != phases.shape:
        raise ValueError("trace terminal vectors are misaligned")
    if not all(
        np.isfinite(np.asarray(value)).all()
        for name, value in trace.items()
        if name not in {"foot_support", "contact_pairs", "done", "terminal"}
    ):
        raise ValueError("trace contains a nonfinite numeric value")
    margins = termination_margin(metrics)
    return {
        "phase": phase,
        "steps": int(len(phases)),
        "terminal": bool(terminal[-1]),
        "done": bool(done[-1]),
        "completed_suffix": bool(len(phases) == remaining and not terminal[-1]),
        "last_action_phase": int(phases[-1]),
        "minimum_termination_margin": finite_float(np.min(margins), "margin"),
        "final_termination_margin": finite_float(margins[-1], "margin"),
        "mean_body_position_error": finite_float(
            np.mean(metrics[:, TRANSITION_METRICS.index("body_position_error")]),
            "body position error",
        ),
        "mean_body_linear_velocity_error": finite_float(
            np.mean(metrics[:, TRANSITION_METRICS.index("body_linear_velocity_error")]),
            "body linear velocity error",
        ),
        "action_rms": finite_float(np.sqrt(np.mean(np.square(actions))), "action RMS"),
        "position_target_rms": finite_float(
            np.sqrt(np.mean(np.square(targets))), "target RMS"
        ),
        "contact_force_p95": finite_float(
            np.percentile(metrics[:, TRANSITION_METRICS.index("contact_force")], 95),
            "contact force p95",
        ),
        "contact_stiffness_p95": finite_float(
            np.percentile(
                metrics[:, TRANSITION_METRICS.index("contact_stiffness")], 95
            ),
            "contact stiffness p95",
        ),
        "support_counts": {
            name: int(np.sum(support == code))
            for code, name in enumerate(("none", "left", "right", "double"))
        },
    }


def compare_traces(
    ppo: Mapping[str, np.ndarray],
    diffsim: Mapping[str, np.ndarray],
    *,
    phase: int,
) -> tuple[dict[str, object], list[int]]:
    """Compare aligned visitation until the retained DiffSim rollout stops."""
    ppo_phase = np.asarray(ppo["phase"], dtype=np.int64)
    diffsim_phase = np.asarray(diffsim["phase"], dtype=np.int64)
    overlap = min(len(ppo_phase), len(diffsim_phase))
    if overlap < 1 or not np.array_equal(ppo_phase[:overlap], diffsim_phase[:overlap]):
        raise ValueError("policy traces do not share a contiguous phase prefix")
    if ppo_phase[0] != phase:
        raise ValueError("trace starts at the wrong phase")

    ppo_qpos = np.asarray(ppo["qpos"], dtype=np.float64)[:overlap]
    diffsim_qpos = np.asarray(diffsim["qpos"], dtype=np.float64)[:overlap]
    ppo_qvel = np.asarray(ppo["qvel"], dtype=np.float64)[:overlap]
    diffsim_qvel = np.asarray(diffsim["qvel"], dtype=np.float64)[:overlap]
    if not np.array_equal(ppo_qpos[0], diffsim_qpos[0]) or not np.array_equal(
        ppo_qvel[0], diffsim_qvel[0]
    ):
        raise ValueError("exact policy resets differ")

    qpos_delta = np.linalg.norm(ppo_qpos - diffsim_qpos, axis=1)
    qvel_delta = np.linalg.norm(ppo_qvel - diffsim_qvel, axis=1)
    target_delta = np.linalg.norm(
        np.asarray(ppo["position_target"], dtype=np.float64)[:overlap]
        - np.asarray(diffsim["position_target"], dtype=np.float64)[:overlap],
        axis=1,
    )
    action_delta = np.linalg.norm(
        np.asarray(ppo["model_action"], dtype=np.float64)[:overlap]
        - np.asarray(diffsim["model_action"], dtype=np.float64)[:overlap],
        axis=1,
    )
    support_diff = np.any(
        np.asarray(ppo["foot_support"], dtype=bool)[:overlap]
        != np.asarray(diffsim["foot_support"], dtype=bool)[:overlap],
        axis=1,
    )
    contact_diff = rows_differ(
        np.asarray(ppo["contact_pairs"], dtype=bool)[:overlap],
        np.asarray(diffsim["contact_pairs"], dtype=bool)[:overlap],
        name="contact-pair",
    )
    state_divergence = first_true((qpos_delta > 1e-6) | (qvel_delta > 1e-5))
    target_divergence = first_true(target_delta > 1e-6)
    support_divergence = first_true(support_diff)
    contact_divergence = first_true(contact_diff)
    selected = selected_boundary_indices(
        overlap,
        (state_divergence, target_divergence, support_divergence, contact_divergence),
    )
    ppo_margin = termination_margin(np.asarray(ppo["metrics"])[:overlap])
    diffsim_margin = termination_margin(np.asarray(diffsim["metrics"])[:overlap])
    return (
        {
            "phase": phase,
            "overlap_steps": overlap,
            "reset_qpos_exact": True,
            "reset_qvel_exact": True,
            "first_target_divergence_offset": target_divergence,
            "first_state_divergence_offset": state_divergence,
            "first_support_divergence_offset": support_divergence,
            "first_contact_pair_divergence_offset": contact_divergence,
            "first_target_divergence_phase": (
                None if target_divergence is None else phase + target_divergence
            ),
            "first_state_divergence_phase": (
                None if state_divergence is None else phase + state_divergence
            ),
            "first_support_divergence_phase": (
                None if support_divergence is None else phase + support_divergence
            ),
            "first_contact_pair_divergence_phase": (
                None if contact_divergence is None else phase + contact_divergence
            ),
            "qpos_delta_rms": finite_float(
                np.sqrt(np.mean(np.square(qpos_delta))), "qpos delta RMS"
            ),
            "qvel_delta_rms": finite_float(
                np.sqrt(np.mean(np.square(qvel_delta))), "qvel delta RMS"
            ),
            "position_target_delta_rms": finite_float(
                np.sqrt(np.mean(np.square(target_delta))), "target delta RMS"
            ),
            "model_action_delta_rms": finite_float(
                np.sqrt(np.mean(np.square(action_delta))), "action delta RMS"
            ),
            "support_divergence_fraction": finite_float(
                np.mean(support_diff), "support divergence"
            ),
            "contact_pair_divergence_fraction": finite_float(
                np.mean(contact_diff), "contact divergence"
            ),
            "diffsim_final_margin": finite_float(diffsim_margin[-1], "margin"),
            "ppo_margin_at_diffsim_final_phase": finite_float(ppo_margin[-1], "margin"),
            "selected_h1_offsets": selected,
            "selected_h1_absolute_phases": [phase + value for value in selected],
        },
        selected,
    )


def rollout_trace(
    env: object,
    action_fn: Callable[[object], jax.Array],
    *,
    phase: int,
    seed: int,
    max_steps: int,
    step_fn: Callable[[object, jax.Array], object],
) -> dict[str, np.ndarray]:
    """Run one exact reset and retain pre-step states plus transition metrics."""
    state = env.reset_at_phase(
        jax.random.PRNGKey(seed),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(phase, dtype=jnp.int32),
    )
    rows: dict[str, list[np.ndarray | float | int | bool]] = {
        "phase": [],
        "qpos": [],
        "qvel": [],
        "model_action": [],
        "position_target": [],
        "last_action": [],
        "foot_support": [],
        "contact_pairs": [],
        "constraint_force_root": [],
        "reward": [],
        "done": [],
        "terminal": [],
        "metrics": [],
    }
    actor_to_model = np.asarray(env.actor_to_model_permutation, dtype=np.int64)
    for _ in range(max_steps):
        current_phase = int(state.info["phase"])
        action = action_fn(state).astype(jnp.float64)
        model_action = np.asarray(action)[actor_to_model]
        rows["phase"].append(current_phase)
        rows["qpos"].append(np.asarray(state.data.qpos))
        rows["qvel"].append(np.asarray(state.data.qvel))
        rows["model_action"].append(model_action)
        rows["position_target"].append(np.asarray(env.position_target(state, action)))
        rows["last_action"].append(np.asarray(state.info["last_act"]))
        rows["foot_support"].append(np.asarray(env.foot_support_signature(state.data)))
        rows["contact_pairs"].append(np.asarray(env.contact_pair_signature(state.data)))
        rows["constraint_force_root"].append(np.asarray(state.data.qfrc_constraint[:6]))

        next_state = step_fn(state, action)
        rows["reward"].append(float(next_state.reward))
        rows["done"].append(bool(float(next_state.done) > 0.5))
        rows["terminal"].append(bool(float(next_state.info["terminal"]) > 0.5))
        rows["metrics"].append(
            np.asarray(
                [float(next_state.metrics[name]) for name in TRANSITION_METRICS],
                dtype=np.float64,
            )
        )
        state = next_state
        if rows["done"][-1]:
            break
    return {name: np.asarray(values) for name, values in rows.items()}


def validate_diffsim_hparams(value: Mapping[str, object]) -> None:
    """Require the retained E002 observation/action/dynamics contract."""
    expected = {
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "reference_stride": 1,
        "actor_history_len": 10,
        "actor_reference_lookahead_steps": [4, 8, 12],
        "actor_reference_preview_mode": "delta",
        "actor_observe_motion_anchor_position": False,
        "tracking_velocity_kernel": "exponential",
        "tracking_root_velocity_weight": 1.0,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
        "solver_profile": "g1-4x5",
        "domain_randomization": False,
        "friction_range": [1.0, 1.0],
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "com_offset_range": [0.0, 0.0, 0.0],
        "reference_reset_noise_scale": 0.0,
        "effort_limit_scale": 1.0,
        "squash_actor_actions": False,
        "clip_sampled_actor_actions": False,
    }
    mismatches = {
        name: (value.get(name), expected_value)
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(f"retained E002 hparams changed: {mismatches}")


def validate_source_grids(
    ppo: Mapping[str, object], diffsim: Mapping[str, object]
) -> None:
    """Bind the run to the two already-audited phase-grid outcomes."""
    ppo_summary = ppo.get("source", {})
    if not isinstance(ppo_summary, Mapping):
        raise TypeError("PPO source grid is missing")
    ppo_summary = ppo_summary.get("summary", {})
    diff_summary = diffsim.get("summary", {})
    if not isinstance(ppo_summary, Mapping) or not isinstance(diff_summary, Mapping):
        raise TypeError("source phase-grid summary is missing")
    if (
        ppo_summary.get("phases") != list(PHASES)
        or ppo_summary.get("survival") != list(PPO_SURVIVAL)
        or ppo_summary.get("completed_suffix") != [True] * len(PHASES)
        or diff_summary.get("phases") != list(PHASES)
        or diff_summary.get("survival") != list(DIFFSIM_SURVIVAL)
        or diff_summary.get("completed_suffix") != [False] * len(PHASES)
    ):
        raise ValueError("source phase-grid outcomes changed")


def classify_frozen_controls(
    ppo: list[Mapping[str, object]],
    diffsim: list[Mapping[str, object]],
) -> dict[str, object]:
    """Require one successful and one failing control on the active backend.

    The PPO source grid was produced on the same GPU path and therefore retains
    an exact survival-vector gate.  E002's archived phase grid was explicitly a
    CPU evaluation; its terminal indices are provenance, not an exact GPU gate.
    """
    if len(ppo) != len(PHASES) or len(diffsim) != len(PHASES):
        raise ValueError("frozen controls require the complete phase grid")
    ppo_survival = [int(row["steps"]) for row in ppo]
    diffsim_survival = [int(row["steps"]) for row in diffsim]
    if ppo_survival != list(PPO_SURVIVAL) or not all(
        row["completed_suffix"] is True and row["terminal"] is False for row in ppo
    ):
        raise ValueError("successful PPO control does not reproduce on this backend")
    if any(
        row["completed_suffix"] is True or row["terminal"] is not True
        for row in diffsim
    ):
        raise ValueError("failing DiffSim control does not terminate on every phase")
    return {
        "outcome": "paired-success-failure-visitation-captured",
        "ppo_survival": ppo_survival,
        "diffsim_survival": diffsim_survival,
    }


def validate_e006_backend_diagnosis(
    reproduction: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    """Require the audited CPU-versus-GPU gate diagnosis from E006."""
    evaluator = reproduction.get("canonical_evaluator_payload", {})
    evaluator_summary = (
        evaluator.get("summary", {}) if isinstance(evaluator, Mapping) else {}
    )
    if (
        reproduction.get("protocol")
        != "e006-current-gpu-canonical-evaluator-reproduction-v1"
        or reproduction.get("valid") is not True
        or reproduction.get("historical_cpu_survival") != list(DIFFSIM_SURVIVAL)
        or reproduction.get("current_gpu_survival") != [124, 135, 81, 92, 79]
        or reproduction.get("same_failure_category") is not True
        or evaluator_summary.get("survival") != [124, 135, 81, 92, 79]
        or evaluator_summary.get("terminal") != [True] * len(PHASES)
        or audit.get("protocol") != "e006-frozen-control-reproduction-failure-audit-v1"
        or audit.get("valid") is not True
        or audit.get("outcome") != "frozen-control-reproduction-failed"
        or audit.get("checks_passed") != audit.get("checks_total")
        or audit.get("backend_reproduction_sha256")
        != SOURCE_E006_BACKEND_REPRODUCTION_SHA256
        or audit.get("policy_retained") is not False
    ):
        raise ValueError("E006 backend diagnosis is invalid")


def repository_preflight(repository: Path, code_commit: str) -> dict[str, object]:
    """Require the exact clean pushed code revision named by registration."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "rev-parse", "takaratruong/research/g1-nested-residual-20260826"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != code_commit or remote != code_commit or tracked_status:
        raise ValueError("code commit is not the exact clean pushed revision")
    return {"repository": str(repository), "head": head, "remote": remote}


def build_envs(reference: Path, hparams: Mapping[str, object]) -> tuple[object, object]:
    """Construct exact policy-facing wrappers over one nominal plant/reference."""
    common = {
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
        "body_mass_scale": 1.0,
        "effort_limit_scale": 1.0,
        "reference_path": reference,
        "reference_stride": 1,
        "actor_reference_lookahead_steps": (4, 8, 12),
        "actor_observe_motion_anchor_position": False,
        "actor_observation_noise": False,
        "domain_randomization": False,
        "friction_range": (1.0, 1.0),
        "kp_range": (35.0, 35.0),
        "kd_range": (0.5, 0.5),
        "com_offset_range": (0.0, 0.0, 0.0),
        "reference_reset_noise_scale": 0.0,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
    }
    ppo_env = make_evaluation_env(
        "g1_tracking_rmr_50hz_source_step",
        actor_history_len=1,
        actor_reference_preview_mode="absolute",
        tracking_velocity_kernel="exponential",
        tracking_root_velocity_weight=0.0,
        **common,
    )
    diffsim_env = make_evaluation_env(
        str(hparams["env_variant"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_preview_mode=str(hparams["actor_reference_preview_mode"]),
        tracking_velocity_kernel=str(hparams["tracking_velocity_kernel"]),
        tracking_root_velocity_weight=float(hparams["tracking_root_velocity_weight"]),
        **common,
    )
    return ppo_env, diffsim_env


def flatten_traces(
    traces: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    """Flatten arm/phase trace dictionaries into stable NPZ keys."""
    output: dict[str, np.ndarray] = {
        "metric_names": np.asarray(TRANSITION_METRICS),
        "termination_limits": TERMINATION_LIMITS,
        "phases": np.asarray(PHASES, dtype=np.int64),
    }
    for arm, phase_traces in traces.items():
        for phase, trace in phase_traces.items():
            for name, value in trace.items():
                output[f"{arm}_phase_{phase:03d}_{name}"] = np.asarray(value)
    return output


def plot_comparison(
    output: Path,
    traces: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
) -> None:
    """Render the decision-relevant visitation differences."""
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(PHASES)))
    body_pos_index = TRANSITION_METRICS.index("body_position_error")
    contact_force_index = TRANSITION_METRICS.index("contact_force")
    for color, phase in zip(colors, PHASES, strict=True):
        ppo = traces["ppo"][phase]
        diffsim = traces["diffsim"][phase]
        ppo_x = np.asarray(ppo["phase"])
        diff_x = np.asarray(diffsim["phase"])
        ppo_metrics = np.asarray(ppo["metrics"])
        diff_metrics = np.asarray(diffsim["metrics"])
        axes[0, 0].plot(ppo_x, termination_margin(ppo_metrics), color=color)
        axes[0, 0].plot(
            diff_x, termination_margin(diff_metrics), color=color, linestyle="--"
        )
        axes[0, 1].plot(ppo_x, ppo_metrics[:, body_pos_index], color=color)
        axes[0, 1].plot(
            diff_x, diff_metrics[:, body_pos_index], color=color, linestyle="--"
        )
        axes[1, 0].plot(ppo_x, ppo_metrics[:, contact_force_index], color=color)
        axes[1, 0].plot(
            diff_x, diff_metrics[:, contact_force_index], color=color, linestyle="--"
        )
        axes[1, 1].plot(
            ppo_x,
            np.linalg.norm(np.asarray(ppo["model_action"]), axis=1),
            color=color,
        )
        axes[1, 1].plot(
            diff_x,
            np.linalg.norm(np.asarray(diffsim["model_action"]), axis=1),
            color=color,
            linestyle="--",
        )
        overlap = min(len(ppo_x), len(diff_x))
        axes[2, 0].plot(
            diff_x[:overlap],
            np.linalg.norm(
                np.asarray(ppo["qpos"])[:overlap]
                - np.asarray(diffsim["qpos"])[:overlap],
                axis=1,
            ),
            color=color,
            label=f"start {phase}",
        )
        contact_difference = rows_differ(
            np.asarray(ppo["contact_pairs"])[:overlap],
            np.asarray(diffsim["contact_pairs"])[:overlap],
            name="contact-pair",
        )
        axes[2, 1].step(
            diff_x[:overlap],
            contact_difference.astype(float) + 1.25 * PHASES.index(phase),
            where="post",
            color=color,
        )

    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_title("Smallest hard-termination margin")
    axes[0, 1].set_title("Mean body-position error")
    axes[1, 0].set_title("Root generalized contact force")
    axes[1, 1].set_title("Model-order action norm")
    axes[2, 0].set_title("PPO–DiffSim qpos separation on shared phases")
    axes[2, 1].set_title("Contact-pair divergence by start (offset vertically)")
    axes[2, 0].legend(frameon=False, ncol=3)
    for axis in axes.flat:
        axis.set_xlabel("absolute reference phase")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("margin (>0 safe)")
    axes[0, 1].set_ylabel("m")
    axes[1, 0].set_ylabel("sum |qfrc_constraint[:6]|")
    axes[1, 1].set_ylabel("L2")
    axes[2, 0].set_ylabel("L2")
    axes[2, 1].set_ylabel("binary + start offset")
    figure.suptitle("Corrected-long frozen PPO success vs retained DiffSim E002")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--diffsim-checkpoint", type=Path, required=True)
    parser.add_argument("--diffsim-hparams", type=Path, required=True)
    parser.add_argument("--source-ppo-grid", type=Path, required=True)
    parser.add_argument("--source-diffsim-grid", type=Path, required=True)
    parser.add_argument("--source-e006-backend-reproduction", type=Path, required=True)
    parser.add_argument("--source-e006-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> None:
    configure_jax()
    args = build_parser().parse_args()
    paths = {
        "reference": args.reference_path.resolve(),
        "ppo_checkpoint": args.ppo_checkpoint.resolve(),
        "diffsim_checkpoint": args.diffsim_checkpoint.resolve(),
        "diffsim_hparams": args.diffsim_hparams.resolve(),
        "source_ppo_grid": args.source_ppo_grid.resolve(),
        "source_diffsim_grid": args.source_diffsim_grid.resolve(),
        "source_e006_backend_reproduction": (
            args.source_e006_backend_reproduction.resolve()
        ),
        "source_e006_audit": args.source_e006_audit.resolve(),
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "ppo_checkpoint": PPO_CHECKPOINT_SHA256,
        "diffsim_checkpoint": DIFFSIM_CHECKPOINT_SHA256,
        "diffsim_hparams": DIFFSIM_HPARAMS_SHA256,
        "source_ppo_grid": SOURCE_PPO_GRID_SHA256,
        "source_diffsim_grid": SOURCE_DIFFSIM_GRID_SHA256,
        "source_e006_backend_reproduction": (SOURCE_E006_BACKEND_REPRODUCTION_SHA256),
        "source_e006_audit": SOURCE_E006_AUDIT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    source_ppo_grid = read_json(paths["source_ppo_grid"])
    source_diffsim_grid = read_json(paths["source_diffsim_grid"])
    validate_source_grids(source_ppo_grid, source_diffsim_grid)
    validate_e006_backend_diagnosis(
        read_json(paths["source_e006_backend_reproduction"]),
        read_json(paths["source_e006_audit"]),
    )
    preflight = {
        "protocol": "g1-success-failure-visitation-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "seed": args.seed,
        "phases": list(PHASES),
        "solver_profile": args.solver_profile,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "policy_update_computed": False,
        "policy_retained": False,
    }
    write_json(output_root / "preflight.json", preflight)

    profile = get_solver_profile(args.solver_profile)
    with solver_context(profile):
        ppo_env, diffsim_env = build_envs(paths["reference"], hparams)
        if not (
            np.array_equal(ppo_env.reference.qpos, diffsim_env.reference.qpos)
            and np.array_equal(ppo_env.reference.qvel, diffsim_env.reference.qvel)
            and tuple(ppo_env.controller.joint_names)
            == tuple(diffsim_env.controller.joint_names)
        ):
            raise ValueError("policy environments do not share one plant/reference")
        ppo_policy = load_source_actor_policy(paths["ppo_checkpoint"])
        diffsim_actor, diffsim_params, diffsim_norm = _load_policy(
            diffsim_env, paths["diffsim_checkpoint"], args.seed
        )
        normalizer = Normalizer(diffsim_env.actor_frame_obs_dim)

        def ppo_action(state: object) -> jax.Array:
            observation = select_rmr_policy_observation(ppo_policy, state.obs)
            return apply_trainable_rmr_policy(ppo_policy, observation).astype(
                jnp.float64
            )

        def diffsim_action(state: object) -> jax.Array:
            observation = diffsim_env.normalize_actor_obs(
                normalizer, diffsim_norm, state.obs
            ).astype(jnp.float32)
            return diffsim_actor.apply(diffsim_params, observation).astype(jnp.float64)

        ppo_step = build_compiled_step(ppo_env)
        diffsim_step = build_compiled_step(diffsim_env)
        traces: dict[str, dict[int, dict[str, np.ndarray]]] = {
            "ppo": {},
            "diffsim": {},
        }
        for phase in PHASES:
            remaining = int(ppo_env.reference_transitions) - phase
            traces["ppo"][phase] = rollout_trace(
                ppo_env,
                ppo_action,
                phase=phase,
                seed=args.seed,
                max_steps=remaining,
                step_fn=ppo_step,
            )
            traces["diffsim"][phase] = rollout_trace(
                diffsim_env,
                diffsim_action,
                phase=phase,
                seed=args.seed,
                max_steps=remaining,
                step_fn=diffsim_step,
            )

    summaries = {arm: [] for arm in traces}
    comparisons = []
    selected_boundaries = []
    for phase in PHASES:
        remaining = 271 - phase
        for arm in traces:
            summaries[arm].append(
                summarize_trace(traces[arm][phase], phase=phase, remaining=remaining)
            )
        comparison, selected = compare_traces(
            traces["ppo"][phase], traces["diffsim"][phase], phase=phase
        )
        comparisons.append(comparison)
        selected_boundaries.extend(
            {
                "start_phase": phase,
                "offset": offset,
                "absolute_phase": phase + offset,
                "ppo_trace_key": f"ppo_phase_{phase:03d}",
                "diffsim_trace_key": f"diffsim_phase_{phase:03d}",
            }
            for offset in selected
        )

    classification = classify_frozen_controls(summaries["ppo"], summaries["diffsim"])
    observed_ppo = tuple(classification["ppo_survival"])
    observed_diffsim = tuple(classification["diffsim_survival"])

    trace_path = output_root / "paired_trajectories.npz"
    write_npz(trace_path, flatten_traces(traces))
    selected_path = output_root / "selected_h1_boundaries.json"
    write_json(
        selected_path,
        {
            "protocol": "g1-success-failure-selected-h1-boundaries-v1",
            "selection": "quartiles-plus-first-state-target-support-contact-divergence",
            "count": len(selected_boundaries),
            "boundaries": selected_boundaries,
            "trajectory_sha256": sha256_file(trace_path),
        },
    )
    plot_path = output_root / "visitation_comparison.png"
    plot_comparison(plot_path, traces)
    summary = {
        "protocol": "g1-success-failure-visitation-comparison-v1",
        "valid": True,
        "outcome": classification["outcome"],
        "phases": list(PHASES),
        "ppo": summaries["ppo"],
        "diffsim": summaries["diffsim"],
        "comparisons": comparisons,
        "selected_h1_boundary_count": len(selected_boundaries),
        "fresh_ppo_survival": list(observed_ppo),
        "fresh_diffsim_survival": list(observed_diffsim),
        "ppo_source_grid_reproduced": True,
        "diffsim_source_cpu_survival": list(DIFFSIM_SURVIVAL),
        "diffsim_same_backend_failing_control": True,
        "exact_reset_pairing": True,
        "common_corrected_reference": REFERENCE_SHA256,
        "nominal_unassisted": True,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "paired_trajectories.npz": sha256_file(trace_path),
            "selected_h1_boundaries.json": sha256_file(selected_path),
            "visitation_comparison.png": sha256_file(plot_path),
        },
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-success-failure-visitation-completion-v1",
        "valid": True,
        "outcome": summary["outcome"],
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(output_root / "preflight.json"),
            "paired_trajectories.npz": sha256_file(trace_path),
            "selected_h1_boundaries.json": sha256_file(selected_path),
            "visitation_comparison.png": sha256_file(plot_path),
            "summary.json": sha256_file(summary_path),
        },
    }
    write_json(output_root / "completion.json", completion)
    print(
        json.dumps(
            {
                "outcome": summary["outcome"],
                "ppo_survival": list(observed_ppo),
                "diffsim_survival": list(observed_diffsim),
                "selected_h1_boundary_count": len(selected_boundaries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
