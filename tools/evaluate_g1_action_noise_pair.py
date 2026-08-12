"""Publish a provenance-bound deterministic/RMR-action-noise rollout pair."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.rmr_action_noise import (
    RMR_ACTION_STD,
    RMR_ACTION_STD_JOINT_NAMES,
    validate_action_noise_std,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_tracking import (
    _load_policy,
    _render_pair,
    configure_jax,
    make_evaluation_env,
    remaining_reference_transitions,
    summarize_stability_errors,
)
from tools.prepare_g1_rmr_reference import sha256_file


E008_SELECTED_CHECKPOINT_SHA256 = (
    "2de4af6d78cd5250c87577397c048b06e60c5b8a7b272c0f8966b8bf589b4474"
)
E008_SELECTED_HPARAMS_SHA256 = (
    "e0b78f2185d91e7d2edadff0afb4f470e70d38f1f7716c304cf866380e594dba"
)
E008_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
E008_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
E008_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
DEFAULT_REFERENCE_PATH = Path(
    "/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/"
    "artifacts/E-20260808-000/reference/dance1_subject2_f122_422_50hz.npz"
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "evaluation.npz",
    "evaluation.mp4",
    "contact_sheet.png",
)
RECORD_COLUMNS = (
    "step",
    "phase",
    "reward",
    "done",
    "terminal",
    "anchor_position_error",
    "anchor_orientation_error",
    "body_position_error",
    "body_orientation_error",
    "body_linear_velocity_error",
    "body_angular_velocity_error",
    "transition_phase",
    "termination_anchor_z_error",
    "termination_anchor_xy_error",
    "termination_gravity_z_error",
    "termination_distal_z_error",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def paired_reset(env: Any, *, phase: int, seed: int) -> tuple[Any, Any]:
    """Reset once, then branch the immutable state for the two matched arms."""
    import jax
    import jax.numpy as jnp

    state = env.reset_at_phase(
        jax.random.PRNGKey(seed), jnp.array(0.0), jnp.array(phase)
    )
    return state, state


def action_noise_tape(*, seed: int, steps: int, action_std) -> np.ndarray:
    """Draw caller-owned standard-normal epsilon and scale it per joint."""
    std = np.asarray(action_std, dtype=np.float64)
    if steps < 1 or std.shape != (len(RMR_ACTION_STD_JOINT_NAMES),):
        raise ValueError("action noise tape requires positive steps and shape (29,)")
    if not np.isfinite(std).all() or (std < 0.0).any():
        raise ValueError("action noise standard deviations must be finite and non-negative")
    return epsilon_tape(seed=seed, steps=steps) * std


def epsilon_tape(*, seed: int, steps: int) -> np.ndarray:
    """Create the caller-owned, per-step standard-normal sample tape."""
    if steps < 1:
        raise ValueError("epsilon tape requires positive steps")
    return np.random.default_rng(seed).standard_normal(
        (steps, len(RMR_ACTION_STD_JOINT_NAMES))
    )


def noisy_action(
    action_mean, epsilon, action_std, *, actor_joint_names
) -> np.ndarray:
    """Apply the validated, source-order perturbation directly before stepping."""
    std = validate_action_noise_std(
        action_std,
        action_dim=len(RMR_ACTION_STD_JOINT_NAMES),
        actor_joint_names=actor_joint_names,
    )
    mean = np.asarray(action_mean, dtype=np.float64)
    sample = np.asarray(epsilon, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != std.shape or sample.shape != std.shape:
        raise ValueError("action mean, epsilon, and standard deviation must all have shape (29,)")
    return np.clip(mean + sample * std, -1.0, 1.0)


def _state_fingerprint(state: Any) -> str:
    import jax

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_provenance(
    *,
    checkpoint: Path,
    reference: Path,
    model_path: Path,
    controller_path: Path,
    seed: int,
    solver_profile: str,
) -> dict[str, Any]:
    """Hash every causal input before the first rollout is allowed to begin."""
    paths = (checkpoint, reference, model_path, controller_path)
    if any(not path.is_file() for path in paths):
        raise ValueError("checkpoint, reference, model, and controller must be readable files")
    profile = get_solver_profile(solver_profile)
    return {
        "protocol": "g1-rmr-action-noise-matched-pair-v1",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_hparams_path": str((checkpoint.parent / "hparams.json").resolve()),
        "checkpoint_hparams_sha256": _sha256(checkpoint.parent / "hparams.json"),
        "reference_path": str(reference.resolve()),
        "reference_sha256": _sha256(reference),
        "seed": int(seed),
        "solver_profile": solver_profile,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
        "action_noise_joint_names": list(RMR_ACTION_STD_JOINT_NAMES),
        "rmr_action_std": np.asarray(RMR_ACTION_STD, dtype=np.float32).tolist(),
        "rmr_action_std_dtype": "float32",
        "runtime_assets": {
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "controller_path": str(controller_path.resolve()),
            "controller_sha256": _sha256(controller_path),
        },
    }


def validate_selected_e008(provenance: dict[str, Any]) -> None:
    """Reject an unregistered checkpoint or environment before publishing evidence."""
    expected = {
        "checkpoint_sha256": E008_SELECTED_CHECKPOINT_SHA256,
        "checkpoint_hparams_sha256": E008_SELECTED_HPARAMS_SHA256,
        "reference_sha256": E008_REFERENCE_SHA256,
    }
    actual = {key: provenance.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"selected E-20260812-008 provenance mismatch: {actual}")
    assets = provenance["runtime_assets"]
    if assets["model_sha256"] != E008_MODEL_SHA256 or assets["controller_sha256"] != E008_CONTROLLER_SHA256:
        raise ValueError("selected E-20260812-008 runtime asset provenance mismatch")


def _record(state: Any, *, step: int, phase: int, next_phase: int) -> tuple[float, ...]:
    return (
        float(step), float(phase), float(state.reward), float(state.done),
        float(state.info["terminal"]),
        float(state.metrics["anchor_position_error"]),
        float(state.metrics["anchor_orientation_error"]),
        float(state.metrics["body_position_error"]),
        float(state.metrics["body_orientation_error"]),
        float(state.metrics["body_linear_velocity_error"]),
        float(state.metrics["body_angular_velocity_error"]), float(next_phase),
        float(state.metrics["termination_anchor_z_error"]),
        float(state.metrics["termination_anchor_xy_error"]),
        float(state.metrics["termination_gravity_z_error"]),
        float(state.metrics["termination_distal_z_error"]),
    )


def _write_media_atomic(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    import imageio.v2 as imageio

    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    imageio.mimsave(temporary, frames, fps=fps, quality=8)
    os.replace(temporary, path)


def _write_contact_sheet_atomic(path: Path, frames: list[np.ndarray]) -> None:
    from tools.evaluate_g1_phase_grid import make_contact_sheet

    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    make_contact_sheet(frames, temporary)
    os.replace(temporary, path)


def rollout_arm(
    env: Any,
    *,
    initial_state: Any,
    action_fn: Callable[[Any], Any],
    action_std,
    phase: int,
    seed: int,
    max_steps: int | None,
    profile: Any,
    output_dir: Path,
    render: bool,
) -> dict[str, Any]:
    """Run one arm; its only treatment hook is the pre-step action addition."""
    import jax.numpy as jnp
    import mujoco

    remaining = remaining_reference_transitions(
        env.reference_length, phase, env.reference_stride
    )
    step_limit = remaining if max_steps is None else min(max_steps, remaining)
    std = validate_action_noise_std(
        action_std, action_dim=env.action_dim, actor_joint_names=env.actor_joint_names
    )
    epsilon = epsilon_tape(seed=seed, steps=step_limit)
    noise = epsilon * np.asarray(std, dtype=np.float64)
    state = initial_state
    records: list[tuple[float, ...]] = []
    action_means: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    assistance_exact_zero = True
    reset_fingerprint = _state_fingerprint(initial_state)
    actual_renderer = reference_renderer = actual_data = reference_data = None
    if render:
        actual_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        reference_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        actual_data = mujoco.MjData(env.mj_model)
        reference_data = mujoco.MjData(env.mj_model)
    for step in range(step_limit):
        current_phase = int(state.info["phase"])
        if render and step % 2 == 0:
            frames.append(_render_pair(env, np.asarray(state.data.qpos), np.asarray(state.data.qvel), current_phase, actual_renderer, reference_renderer, actual_data, reference_data))
        assistance_exact_zero &= bool(np.array_equal(np.asarray(state.data.xfrc_applied), np.zeros_like(np.asarray(state.data.xfrc_applied))))
        mean = np.asarray(action_fn(state), dtype=np.float64)
        action = noisy_action(
            mean, epsilon[step], std, actor_joint_names=env.actor_joint_names
        )
        # The noise is injected at the latest possible point: immediately before env.step.
        with solver_context(profile) if profile is not None else nullcontext():
            state = env.step(state, jnp.asarray(action, dtype=jnp.float64))
        next_phase = min(current_phase + env.reference_stride, env.reference_length - 1)
        records.append(_record(state, step=step, phase=current_phase, next_phase=next_phase))
        action_means.append(mean)
        applied_actions.append(action)
        if float(state.done) > 0.5:
            break
    values = np.asarray(records, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("rollout emitted no finite telemetry")
    _write_npz_atomic(output_dir / "evaluation.npz", columns=np.asarray(RECORD_COLUMNS), values=values, action_mean=np.asarray(action_means), epsilon=epsilon[:len(records)], action_noise=noise[:len(records)], action=np.asarray(applied_actions), joint_names=np.asarray(env.actor_joint_names))
    if render:
        if not frames:
            raise ValueError("rendered rollout emitted no frames")
        _write_media_atomic(output_dir / "evaluation.mp4", frames, fps=round(1.0 / (env.dt * 2)))
        _write_contact_sheet_atomic(output_dir / "contact_sheet.png", frames)
    stability = summarize_stability_errors({"anchor_z_error": values[:, 12], "anchor_xy_error": values[:, 13], "gravity_z_error": values[:, 14], "distal_z_error": values[:, 15]})
    true_terminal = bool(np.any(values[:, 4] > 0.5))
    summary = {
        "steps": len(records), "terminal": bool(values[-1, 4] > 0.5),
        "mean_reward": float(np.mean(values[:, 2])), "evaluation_start_phase": phase,
        "remaining_reference_transitions": remaining,
        "completed_reference_suffix": len(records) == remaining and not true_terminal,
        "intermediate_reset_occurred": true_terminal,
        "paired_reset_state_sha256": reset_fingerprint,
        "action_noise_exact_zero": bool(np.array_equal(noise[:len(records)], np.zeros_like(noise[:len(records)]))),
        "assistance_exact_zero": assistance_exact_zero,
        "noise_seed": seed,
        "noise_joint_names": list(env.actor_joint_names),
        **stability,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def build_pair_manifest(*, output_dir: Path, provenance: dict[str, Any], arms: dict[str, dict[str, Any]], require_media: bool = True) -> dict[str, Any]:
    """Fail closed unless both arms have complete, matching evidence."""
    if set(arms) != {"deterministic", "rmr-noisy"}:
        raise ValueError("pair requires deterministic and rmr-noisy arms")
    required = REQUIRED_ARTIFACTS if require_media else REQUIRED_ARTIFACTS[:2]
    for arm in arms:
        for name in required:
            if not (output_dir / arm / name).is_file():
                raise ValueError(f"{arm} missing required artifact: {name}")
    deterministic, noisy = arms["deterministic"], arms["rmr-noisy"]
    if not deterministic.get("action_noise_exact_zero", False):
        raise ValueError("deterministic arm does not have exact-zero action noise")
    if noisy.get("action_noise_exact_zero", True):
        raise ValueError("RMR noisy arm unexpectedly has exact-zero action noise")
    if deterministic.get("paired_reset_state_sha256") != noisy.get("paired_reset_state_sha256") and "paired_reset_state_sha256" in deterministic:
        raise ValueError("arms do not share an identical reset state")
    return {
        "protocol": "g1-rmr-action-noise-matched-pair-v1", "valid": bool(require_media),
        "publication_complete": bool(require_media), "provenance": provenance,
        "arms": arms, "artifact_requirements": list(required),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-render", action="store_true", help="CPU smoke only; produces a non-publishable manifest")
    return parser


def main() -> None:
    configure_jax()
    args = build_parser().parse_args()
    if args.seed != 0 or args.phase != 0:
        raise ValueError("the registered pair fixes phase and noise seed to zero")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    profile = get_solver_profile("g1-4x5")
    env = make_evaluation_env("g1_tracking_rmr_50hz_source_step", solver_iterations=profile.iterations, solver_ls_iterations=profile.ls_iterations, reference_path=args.reference_path, reference_stride=1, actor_history_len=10, actor_reference_lookahead_steps=(4, 8, 12), actor_reference_preview_mode="delta", reference_residual_control=True, reference_residual_scale=0.5)
    provenance = build_provenance(checkpoint=args.checkpoint.resolve(), reference=args.reference_path.resolve(), model_path=Path(env.xml_path), controller_path=Path(env.controller_path), seed=args.seed, solver_profile="g1-4x5")
    validate_selected_e008(provenance)
    actor, actor_params, normalizer_state = _load_policy(env, args.checkpoint.resolve(), args.seed)
    from src.core.data_structures import Normalizer

    normalizer = Normalizer(env.actor_frame_obs_dim)
    def action_fn(state):
        normalized = env.normalize_actor_obs(normalizer, normalizer_state, state.obs).astype(np.float32)
        return actor.apply(actor_params, normalized).astype(np.float64)
    deterministic_initial, noisy_initial = paired_reset(env, phase=args.phase, seed=args.seed)
    output_dir = args.output_dir.resolve()
    arms = {
        "deterministic": rollout_arm(env, initial_state=deterministic_initial, action_fn=action_fn, action_std=np.zeros(env.action_dim, dtype=np.float32), phase=args.phase, seed=args.seed, max_steps=args.max_steps, profile=profile, output_dir=output_dir / "deterministic", render=not args.no_render),
        "rmr-noisy": rollout_arm(env, initial_state=noisy_initial, action_fn=action_fn, action_std=RMR_ACTION_STD, phase=args.phase, seed=args.seed, max_steps=args.max_steps, profile=profile, output_dir=output_dir / "rmr-noisy", render=not args.no_render),
    }
    manifest = build_pair_manifest(output_dir=output_dir, provenance=provenance, arms=arms, require_media=not args.no_render)
    _write_json_atomic(output_dir / "action_noise_pair.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
