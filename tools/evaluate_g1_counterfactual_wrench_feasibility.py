"""Test whether E004's torso wrench is locally reachable through G1 legs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from src.algorithms.shac.counterfactual_wrench_distillation import (
    counterfactual_target_change,
    resolve_leg_action_indices,
    scatter_leg_residual,
)
from src.algorithms.shac.learned_torso_wrench import (
    FrozenControllerWrenchParams,
    normalized_yaw_wrench_to_world,
)
from src.core.data_structures import Normalizer
from src.envs.g1_tracking.centroidal_momentum import mjx_centroidal_momentum
from src.envs.g1_tracking.environment import (
    DEFAULT_CONTROLLER_PATH,
    DEFAULT_MODEL_PATH,
)
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from src.evaluation.g1_torso_wrench_oracle import (
    torso_wrench_parameters_from_environment,
    write_torso_wrench,
)
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.evaluate_g1_tracking import (
    _load_policy,
    make_evaluation_env,
    reset_evaluation_state,
)
from tools.prepare_g1_rmr_reference import sha256_file


FEASIBILITY_PHASES = (0, 25, 50, 75, 100)
FEASIBILITY_THRESHOLD = 0.50
PROTOCOL = "g1-counterfactual-wrench-feasibility-v1"
EXPECTED_TEACHER_SHA256 = (
    "b7fd54a82380e032f91da6e12b7252f2bd42f4a1f5fe6be0f206849282811870"
)
EXPECTED_HPARAMS_SHA256 = (
    "6838465c6b6190a9ab165c82d61b35effd93ceab613d1d18993ea8b3154bffb6"
)
EXPECTED_REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
EXPECTED_TEACHER_TREE_SHA256 = (
    "b3dac1d8fb6fe002c66841b44b4fd856e863d2249965991ab893a0dd1d10f48c"
)
EXPECTED_E026_TREE_SHA256 = (
    "58af1e665c1d323d254607298a960279afc3228d4f74421acd39f244cf6be74f"
)
EXPECTED_WRENCH_TREE_SHA256 = (
    "42adfbd94fe8bb1b0f54b3093f5a61965e3e7955cf741787fa836dda04993eee"
)


def bounded_damped_projection(
    jacobian: np.ndarray,
    target: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    damping: float,
) -> dict[str, Any]:
    """Project one normalized target through a bounded local leg Jacobian."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    desired = np.asarray(target, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if matrix.shape != (12, 12) or desired.shape != (12,):
        raise ValueError("projection requires a 12x12 Jacobian and 12-vector")
    if low.shape != (12,) or high.shape != (12,) or np.any(low > high):
        raise ValueError("projection action bounds are invalid")
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(desired).all()
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or not math.isfinite(damping)
        or damping <= 0.0
    ):
        raise ValueError("projection inputs must be finite")
    augmented_matrix = np.concatenate(
        (matrix, math.sqrt(damping) * np.eye(12)), axis=0
    )
    augmented_target = np.concatenate((desired, np.zeros(12)), axis=0)
    solution = lsq_linear(
        augmented_matrix,
        augmented_target,
        bounds=(low, high),
        tol=1e-12,
        lsmr_tol=1e-12,
        max_iter=500,
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError("bounded projection did not converge")
    correction = solution.x
    achieved = matrix @ correction
    target_norm = float(np.linalg.norm(desired))
    residual_norm = float(np.linalg.norm(achieved - desired))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    bound_mask = np.isclose(correction, low, atol=1e-9) | np.isclose(
        correction, high, atol=1e-9
    )
    return {
        "correction": correction,
        "achieved": achieved,
        "rank": int(np.linalg.matrix_rank(matrix)),
        "singular_values": singular_values,
        "target_norm": target_norm,
        "residual_norm": residual_norm,
        "normalized_residual": residual_norm / max(target_norm, 1e-12),
        "action_rms": float(np.sqrt(np.mean(np.square(correction)))),
        "action_max": float(np.max(np.abs(correction))),
        "bound_fraction": float(np.mean(bound_mask)),
    }


def leg_residual_bounds(base_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the tanh head's correction bounds, independent of parent action."""
    parent = np.asarray(base_action, dtype=np.float64)
    if parent.shape != (29,) or not np.isfinite(parent).all():
        raise ValueError("parent action must be a finite 29-vector")
    return np.full(12, -1.0), np.full(12, 1.0)


def classify_feasibility(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Apply the immutable five-phase necessary-value gate."""
    required = {
        "phase",
        "target_norm",
        "jacobian_rank",
        "normalized_residual",
        "action_rms",
        "action_max",
        "bound_fraction",
        "finite",
    }
    phase_counts = {phase: 0 for phase in FEASIBILITY_PHASES}
    residuals: list[float] = []
    rows_valid = bool(rows)
    for row in rows:
        if set(row) < required:
            rows_valid = False
            continue
        try:
            phase = int(row["phase"])
            numeric = np.asarray(
                [
                    row["target_norm"],
                    row["jacobian_rank"],
                    row["normalized_residual"],
                    row["action_rms"],
                    row["action_max"],
                    row["bound_fraction"],
                ],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            rows_valid = False
            continue
        row_valid = bool(
            row["finite"] is True
            and phase in phase_counts
            and np.isfinite(numeric).all()
            and float(row["target_norm"]) > 0.0
            and int(row["jacobian_rank"]) > 0
            and float(row["normalized_residual"]) >= 0.0
            and 0.0 <= float(row["bound_fraction"]) <= 1.0
        )
        rows_valid = rows_valid and row_valid
        if row_valid:
            phase_counts[phase] += 1
            residuals.append(float(row["normalized_residual"]))
    coverage = all(count > 0 for count in phase_counts.values())
    median = float(np.median(residuals)) if residuals else math.inf
    valid = bool(rows_valid and coverage and residuals)
    feasible = bool(valid and median <= FEASIBILITY_THRESHOLD)
    return {
        "valid": valid,
        "outcome": (
            "leg-counterfactual-feasible"
            if feasible
            else "leg-counterfactual-not-feasible"
        ),
        "threshold": FEASIBILITY_THRESHOLD,
        "phase_counts": {str(key): value for key, value in phase_counts.items()},
        "median_normalized_residual": median if math.isfinite(median) else None,
        "row_count": len(rows),
    }


def publish_feasibility_artifacts(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    payload: dict[str, object],
) -> Path:
    """Publish the raw NPZ before one SHA-bound atomic JSON report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / "counterfactual_wrench_feasibility.npz"
    with tempfile.NamedTemporaryFile(
        dir=directory, suffix=".npz", delete=False
    ) as stream:
        temporary_npz = Path(stream.name)
    try:
        np.savez_compressed(temporary_npz, **arrays)
        os.replace(temporary_npz, npz_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
    report = {
        **payload,
        "protocol": PROTOCOL,
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
    }
    report_path = directory / "counterfactual_wrench_feasibility.json"
    _write_json_atomically(report_path, report)
    validate_feasibility_artifacts(report_path)
    return report_path


def validate_feasibility_artifacts(report_path: Path) -> dict[str, object]:
    """Validate the publishable feasibility pair without trusting filenames."""
    path = Path(report_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("feasibility report is not valid JSON") from error
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("feasibility protocol is invalid")
    if not isinstance(payload.get("valid"), bool):
        raise ValueError("feasibility validity is missing")
    if payload.get("outcome") not in {
        "leg-counterfactual-feasible",
        "leg-counterfactual-not-feasible",
    }:
        raise ValueError("feasibility outcome is invalid")
    if payload.get("phases") != list(FEASIBILITY_PHASES):
        raise ValueError("feasibility phases are invalid")
    if payload.get("threshold") != FEASIBILITY_THRESHOLD:
        raise ValueError("feasibility threshold is invalid")
    npz_name = payload.get("npz_file")
    expected_sha = payload.get("npz_sha256")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise ValueError("feasibility NPZ filename is invalid")
    npz_path = path.with_name(npz_name)
    if (
        not npz_path.is_file()
        or not isinstance(expected_sha, str)
        or sha256_file(npz_path) != expected_sha
    ):
        raise ValueError("feasibility NPZ SHA-256 does not match")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) < {"phase", "jacobian", "target"}:
                raise ValueError("feasibility NPZ arrays are incomplete")
            phase = np.asarray(archive["phase"])
            jacobian = np.asarray(archive["jacobian"])
            target = np.asarray(archive["target"])
    except (OSError, ValueError) as error:
        raise ValueError("feasibility NPZ is invalid") from error
    if (
        phase.ndim != 1
        or jacobian.shape != (len(phase), 12, 12)
        or target.shape != (len(phase), 12)
        or not np.isfinite(jacobian).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("feasibility NPZ shapes or values are invalid")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument(
        "--solver-profile", choices=tuple(sorted(SOLVER_PROFILES)), required=True
    )
    parser.add_argument("--phases", type=int, nargs="+", default=FEASIBILITY_PHASES)
    parser.add_argument("--max-states-per-phase", type=int, default=24)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _state_features(env: object, state: object) -> jax.Array:
    momentum = mjx_centroidal_momentum(
        env.mjx_model,
        state.data,
        env.root_body_id,
        env.nominal_total_mass,
    )
    return jnp.concatenate((state.data.qvel[:6], momentum))


def _validate_preflight(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != args.code_commit or dirty:
        raise ValueError("feasibility execution requires the clean registered commit")
    hparams_path = args.teacher_checkpoint.with_name("hparams.json")
    assets = (
        (args.teacher_checkpoint, EXPECTED_TEACHER_SHA256, "teacher checkpoint"),
        (hparams_path, EXPECTED_HPARAMS_SHA256, "teacher hparams"),
        (args.reference_path, EXPECTED_REFERENCE_SHA256, "reference"),
        (args.model_path, EXPECTED_MODEL_SHA256, "model"),
        (args.controller_path, EXPECTED_CONTROLLER_SHA256, "controller"),
    )
    for path, expected, label in assets:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    if args.model_path.resolve() != Path(DEFAULT_MODEL_PATH).resolve():
        raise ValueError("runtime model path does not match the environment")
    if args.controller_path.resolve() != Path(DEFAULT_CONTROLLER_PATH).resolve():
        raise ValueError("runtime controller path does not match the environment")
    if tuple(args.phases) != FEASIBILITY_PHASES:
        raise ValueError("feasibility phases are immutable")
    if args.max_states_per_phase < 1 or args.seed != 0:
        raise ValueError("feasibility state count and seed are invalid")
    return {
        "code_commit": head,
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "teacher_hparams_sha256": EXPECTED_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
    }


def _load_teacher(args: argparse.Namespace):
    hparams_path = args.teacher_checkpoint.with_name("hparams.json")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    env = make_evaluation_env(
        hparams["env_variant"],
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        reference_path=args.reference_path,
        reference_stride=int(hparams["reference_stride"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            hparams["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=hparams["actor_reference_preview_mode"],
        actor_observe_motion_anchor_position=bool(
            hparams["actor_observe_motion_anchor_position"]
        ),
        tracking_velocity_kernel=hparams["tracking_velocity_kernel"],
        tracking_torso_orientation_weight=float(
            hparams["tracking_torso_orientation_weight"]
        ),
        reference_residual_control=bool(hparams["reference_residual_control"]),
        reference_residual_scale=float(hparams["reference_residual_scale"]),
    )
    actor, params, normalizer_state = _load_policy(
        env, args.teacher_checkpoint, args.seed
    )
    if not isinstance(params, FrozenControllerWrenchParams):
        raise ValueError("teacher checkpoint is not a learned wrench policy")
    tree_hashes = {
        "teacher_tree_sha256": parameter_tree_sha256(params),
        "e026_tree_sha256": parameter_tree_sha256(params.controller),
        "wrench_tree_sha256": parameter_tree_sha256(params.wrench),
    }
    expected = {
        "teacher_tree_sha256": EXPECTED_TEACHER_TREE_SHA256,
        "e026_tree_sha256": EXPECTED_E026_TREE_SHA256,
        "wrench_tree_sha256": EXPECTED_WRENCH_TREE_SHA256,
    }
    if tree_hashes != expected:
        raise ValueError("teacher parameter provenance does not match")
    return env, actor, params, normalizer_state, hparams, tree_hashes


def _collect_rows(
    *,
    env: object,
    actor: object,
    params: FrozenControllerWrenchParams,
    normalizer_state: object,
    profile: object,
    phases: Sequence[int],
    max_states_per_phase: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], np.ndarray]:
    leg_indices = resolve_leg_action_indices(env.actor_joint_names)
    torso_body_id, wrench_parameters = torso_wrench_parameters_from_environment(env)
    normalizer = Normalizer(env.actor_frame_obs_dim)
    raw_rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    lower_bounds: list[np.ndarray] = []
    upper_bounds: list[np.ndarray] = []
    phases_out: list[int] = []
    wrench_rows: list[np.ndarray] = []
    support_rows: list[np.ndarray] = []
    reset_key = jax.random.PRNGKey(seed)

    def teacher_step(teacher_state, teacher_action):
        return env.step(teacher_state, teacher_action)

    def student_change(residual, clean_state, base_action, before_features):
        student_action = base_action + scatter_leg_residual(
            residual, leg_indices, action_dim=29
        ).astype(base_action.dtype)
        next_state = env.step(clean_state, student_action)
        return counterfactual_target_change(
            before_features, _state_features(env, next_state)
        )

    compiled_teacher_step = jax.jit(teacher_step)
    compiled_leg_jacobian = jax.jit(jax.jacfwd(student_change, argnums=0))
    for reset_phase in phases:
        state = reset_evaluation_state(
            env,
            reset_key=jax.random.fold_in(reset_key, reset_phase),
            difficulty=jnp.asarray(0.0, dtype=jnp.float64),
            phase=reset_phase,
            sample_training_reset=False,
            profile=profile,
            compile_reset=True,
        )
        for _ in range(max_states_per_phase):
            clean_state = state.replace(
                data=state.data.replace(
                    xfrc_applied=jnp.zeros_like(state.data.xfrc_applied)
                )
            )
            normalized = env.normalize_actor_obs(
                normalizer, normalizer_state, clean_state.obs
            ).astype(jnp.float32)
            base_action = actor.apply(params, normalized).astype(jnp.float64)
            normalized_wrench = actor.normalized_wrench(
                params, normalized, 1.0
            ).astype(clean_state.data.qpos.dtype)
            world_wrench = normalized_yaw_wrench_to_world(
                normalized_wrench,
                root_quaternion=clean_state.data.qpos[3:7],
                force_cap=wrench_parameters.force_cap,
                torque_cap=wrench_parameters.torque_cap,
                scale=1.0,
            )
            teacher_state = clean_state.replace(
                data=clean_state.data.replace(
                    xfrc_applied=write_torso_wrench(
                        clean_state.data.xfrc_applied,
                        torso_body_id=torso_body_id,
                        world_wrench=world_wrench,
                    )
                )
            )
            scope = nullcontext() if profile is None else solver_context(profile)
            with scope:
                teacher_next = compiled_teacher_step(teacher_state, base_action)
            if bool(np.asarray(teacher_next.done)):
                break
            before_features = _state_features(env, clean_state)
            teacher_change = counterfactual_target_change(
                before_features, _state_features(env, teacher_next)
            )
            support = np.asarray(env.foot_support_signature(clean_state.data))
            if bool(np.any(support)) and float(np.linalg.norm(np.asarray(world_wrench))) > 1e-8:
                scope = nullcontext() if profile is None else solver_context(profile)
                with scope:
                    jacobian = compiled_leg_jacobian(
                        jnp.zeros((12,), dtype=base_action.dtype),
                        clean_state,
                        base_action,
                        before_features,
                    )
                target_np = np.asarray(teacher_change, dtype=np.float64)
                jacobian_np = np.asarray(jacobian, dtype=np.float64)
                if np.isfinite(target_np).all() and np.isfinite(jacobian_np).all():
                    targets.append(target_np)
                    jacobians.append(jacobian_np)
                    lower, upper = leg_residual_bounds(
                        np.asarray(base_action, dtype=np.float64)
                    )
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                    phases_out.append(reset_phase)
                    wrench_rows.append(np.asarray(world_wrench, dtype=np.float64))
                    support_rows.append(support.astype(np.int8))
            state = teacher_next
    if not targets:
        return [], {
            "phase": np.empty((0,), dtype=np.int32),
            "jacobian": np.empty((0, 12, 12)),
            "target": np.empty((0, 12)),
        }, np.empty((12,))
    target_array = np.asarray(targets)
    target_rms = np.sqrt(np.mean(np.square(target_array), axis=0))
    safe_rms = np.maximum(target_rms, 1e-3)
    for phase, jacobian, target, lower, upper in zip(
        phases_out,
        jacobians,
        targets,
        lower_bounds,
        upper_bounds,
        strict=True,
    ):
        projection = bounded_damped_projection(
            jacobian / safe_rms[:, None],
            target / safe_rms,
            lower=lower,
            upper=upper,
            damping=1e-4,
        )
        raw_rows.append(
            {
                "phase": phase,
                "target_norm": projection["target_norm"],
                "jacobian_rank": projection["rank"],
                "normalized_residual": projection["normalized_residual"],
                "action_rms": projection["action_rms"],
                "action_max": projection["action_max"],
                "bound_fraction": projection["bound_fraction"],
                "finite": True,
            }
        )
    arrays = {
        "phase": np.asarray(phases_out, dtype=np.int32),
        "jacobian": np.asarray(jacobians, dtype=np.float64),
        "target": target_array,
        "target_rms": target_rms,
        "lower_bound": np.asarray(lower_bounds, dtype=np.float64),
        "upper_bound": np.asarray(upper_bounds, dtype=np.float64),
        "teacher_wrench": np.asarray(wrench_rows, dtype=np.float64),
        "foot_support": np.asarray(support_rows, dtype=np.int8),
    }
    return raw_rows, arrays, target_rms


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    args = build_parser().parse_args()
    preflight = _validate_preflight(args)
    env, actor, params, normalizer_state, _hparams, tree_hashes = _load_teacher(args)
    profile = get_solver_profile(args.solver_profile)
    rows, arrays, target_rms = _collect_rows(
        env=env,
        actor=actor,
        params=params,
        normalizer_state=normalizer_state,
        profile=profile,
        phases=args.phases,
        max_states_per_phase=args.max_states_per_phase,
        seed=args.seed,
    )
    classification = classify_feasibility(rows)
    payload = {
        **classification,
        **preflight,
        **tree_hashes,
        "phases": list(FEASIBILITY_PHASES),
        "rows": rows,
        "target_rms": np.asarray(target_rms).tolist(),
        "solver_profile": args.solver_profile,
        "max_states_per_phase": args.max_states_per_phase,
        "seed": args.seed,
    }
    report_path = publish_feasibility_artifacts(args.output_dir, arrays, payload)
    print(report_path)
    print(classification["outcome"])


if __name__ == "__main__":
    main()
