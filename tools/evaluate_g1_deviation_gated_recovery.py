"""Frozen three-arm evaluation of deviation-gated G1 recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.algorithms.shac.deviation_gated_recovery import (
    REGISTERED_DEVIATION_GATE,
    deviation_recovery_gate,
)


PHASES = (0, 25, 50, 75, 100)
ARMS = ("parent", "global", "gated")
EXPECTED_SUFFIXES = np.asarray((124, 99, 74, 49, 24), dtype=np.int64)
ROLLOUT_ARRAY_KEYS = frozenset(
    {
        "columns",
        "values",
        "pre_body_position_error",
        "gate",
        "parent_action",
        "raw_residual_action",
        "gated_residual_action",
        "candidate_action",
        "sampled_action",
        "effective_action",
        "qpos",
        "qvel",
    }
)


def build_parser() -> argparse.ArgumentParser:
    """Build the immutable frozen-composition evaluator CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--solver-profile", default="g1-4x5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write strict JSON atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed no-pickle NumPy archive atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _summary_survival(summary: Mapping[str, object], label: str) -> np.ndarray:
    survival = np.asarray(summary.get("survival"), dtype=np.int64)
    if survival.shape != (5,) or np.any(survival < 0):
        raise ValueError(f"{label} survival must contain five nonnegative values")
    return survival


def _tracking_gate(summary: Mapping[str, object]) -> bool:
    mask = np.asarray(
        summary.get("tracking_metric_mask", [True] * 5), dtype=bool
    )
    if mask.shape != (5,):
        raise ValueError("gated tracking metric mask must contain five values")
    for key in (
        "body_position_error_ratio",
        "body_orientation_error_ratio",
    ):
        values = np.asarray(summary.get(key), dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError(f"gated {key} must contain five finite values")
        if np.any(values[mask] > 1.05):
            return False
    return True


def classify_deviation_gate(
    *,
    parent: Mapping[str, object],
    global_arm: Mapping[str, object],
    gated: Mapping[str, object],
    final_body_position_error: np.ndarray,
) -> str:
    """Classify only the preregistered frozen-composition outcomes."""
    parent_survival = _summary_survival(parent, "parent")
    global_survival = _summary_survival(global_arm, "global")
    gated_survival = _summary_survival(gated, "gated")
    tail = np.asarray(final_body_position_error, dtype=np.float64)
    if tail.shape != (10,) or not np.isfinite(tail).all():
        raise ValueError("final body-position-error tail must contain ten values")

    preserved = bool(np.all(gated_survival >= parent_survival))
    improved = bool(np.any(gated_survival > parent_survival))
    metrics_valid = _tracking_gate(gated)
    stable_tail = bool(np.all(np.diff(tail) <= 0.0))
    if (
        preserved
        and improved
        and metrics_valid
        and stable_tail
        and np.array_equal(gated_survival, EXPECTED_SUFFIXES)
    ):
        return "deviation-gating-solves-short-clip"
    if preserved and improved and metrics_valid:
        return "deviation-gating-advances"
    if np.any(global_survival > parent_survival):
        return "useful-correction-not-localizable"
    return "correction-intrinsically-insufficient"


def validate_raw_rollout(
    path: Path, *, arm: str, phase: int
) -> dict[str, object]:
    """Recompute the registered action composition from one raw rollout."""
    if arm not in ARMS or phase not in PHASES:
        raise ValueError("rollout arm or phase is not registered")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ROLLOUT_ARRAY_KEYS:
            raise ValueError("raw rollout array keys do not match the contract")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    values = arrays["values"]
    if values.ndim != 2 or values.shape[1] != 16 or values.shape[0] < 1:
        raise ValueError("raw rollout values have an invalid shape")
    rows = values.shape[0]
    numeric_names = ROLLOUT_ARRAY_KEYS - {"columns"}
    for name in numeric_names:
        value = arrays[name]
        if value.shape[0] != rows or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"raw rollout {name} does not match row count")
        if not np.isfinite(value).all():
            raise ValueError(f"raw rollout {name} is nonfinite")
    columns = tuple(str(value) for value in arrays["columns"])
    if len(columns) != 16 or columns[:2] != ("step", "phase"):
        raise ValueError("raw rollout columns do not match the contract")
    np.testing.assert_array_equal(values[:, 0], np.arange(rows))
    if rows > 1 and np.any(np.diff(values[:, 1]) < 0):
        raise ValueError("raw rollout phase is nonmonotone")

    parent = arrays["parent_action"]
    residual = arrays["raw_residual_action"]
    if parent.ndim != 2 or residual.shape != parent.shape:
        raise ValueError("raw action arrays have invalid shapes")
    expected_gate = np.asarray(
        deviation_recovery_gate(arrays["pre_body_position_error"])
    )
    if not np.allclose(arrays["gate"], expected_gate, atol=1e-7, rtol=0.0):
        raise ValueError("raw rollout gate does not recompute")
    if arm == "parent":
        expected_gated = np.zeros_like(residual)
    elif arm == "global":
        expected_gated = residual
    else:
        expected_gated = expected_gate[:, None] * residual
    expected_candidate = parent + expected_gated
    if not np.allclose(
        arrays["gated_residual_action"], expected_gated, atol=1e-7, rtol=0.0
    ) or not np.allclose(
        arrays["candidate_action"], expected_candidate, atol=1e-7, rtol=0.0
    ):
        raise ValueError("raw rollout action composition does not recompute")
    if not np.array_equal(arrays["sampled_action"], arrays["candidate_action"]):
        raise ValueError("frozen rollout sampled action must equal candidate action")
    return {"valid": True, "rows": rows, "arm": arm, "phase": phase}


def build_completion_manifest(
    path: Path,
    *,
    records: Sequence[Mapping[str, object]],
    selection_path: Path,
    provenance: Mapping[str, object],
) -> None:
    """Hash-bind every raw rollout and selection before publishing completion."""
    bound_records = []
    for record in records:
        record_path = Path(str(record["path"])).resolve()
        bound_records.append(
            {
                "arm": str(record["arm"]),
                "phase": int(record["phase"]),
                "path": str(record_path),
                "sha256": _sha256(record_path),
                "summary_path": str(
                    Path(str(record["summary_path"])).resolve()
                ),
                "summary_sha256": _sha256(
                    Path(str(record["summary_path"])).resolve()
                ),
            }
        )
    selection_path = Path(selection_path).resolve()
    payload = {
        "valid": True,
        "protocol": "g1-deviation-gated-recovery-v1",
        "gate": {
            "lower": REGISTERED_DEVIATION_GATE.lower,
            "upper": REGISTERED_DEVIATION_GATE.upper,
        },
        "provenance": dict(provenance),
        "records": bound_records,
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
        },
    }
    atomic_json(path, payload)


def validate_completion_manifest(path: Path) -> dict[str, object]:
    """Reopen every manifest-bound artifact and reject drift or omission."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("valid") is not True:
        raise ValueError("completion manifest is not valid")
    if payload.get("protocol") != "g1-deviation-gated-recovery-v1":
        raise ValueError("completion protocol does not match")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 15:
        raise ValueError("completion manifest must contain fifteen rollouts")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("completion provenance is missing")
    for label in (
        "checkpoint",
        "hparams",
        "reference",
        "model",
        "controller",
        "video",
        "contact_sheet",
        "preflight",
    ):
        path_key = f"{label}_path"
        hash_key = f"{label}_sha256"
        if path_key in provenance:
            asset_path = Path(str(provenance[path_key]))
            if _sha256(asset_path) != provenance.get(hash_key):
                raise ValueError(f"{label} provenance hash does not match")
    observed = set()
    raw_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    summaries_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for record in records:
        arm = str(record["arm"])
        phase = int(record["phase"])
        key = (arm, phase)
        if key in observed:
            raise ValueError("completion manifest contains duplicate rollout")
        observed.add(key)
        record_path = Path(str(record["path"]))
        if _sha256(record_path) != record.get("sha256"):
            raise ValueError("raw rollout hash does not match")
        raw_report = validate_raw_rollout(record_path, arm=arm, phase=phase)
        summary_path = Path(str(record["summary_path"]))
        if _sha256(summary_path) != record.get("summary_sha256"):
            raise ValueError("rollout summary hash does not match")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("arm") != arm
            or int(summary.get("phase", -1)) != phase
            or int(summary.get("steps", -1)) != int(raw_report["rows"])
        ):
            raise ValueError("rollout summary does not match raw record")
        with np.load(record_path, allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float64)
        expected_summary = {
            "terminal": bool(values[-1, 4] > 0.5),
            "mean_body_position_error": float(np.mean(values[:, 7])),
            "mean_body_orientation_error": float(np.mean(values[:, 8])),
        }
        for name, expected_value in expected_summary.items():
            actual_value = summary.get(name)
            if isinstance(expected_value, bool):
                agrees = actual_value is expected_value
            else:
                agrees = bool(
                    np.isfinite(actual_value)
                    and np.isclose(
                        float(actual_value), expected_value, atol=1e-12, rtol=0.0
                    )
                )
            if not agrees:
                raise ValueError(f"rollout summary {name} disagrees with raw evidence")
        raw_by_key[key] = {"values": values}
        summaries_by_key[key] = summary
    if observed != {(arm, phase) for arm in ARMS for phase in PHASES}:
        raise ValueError("completion manifest rollout grid is incomplete")
    selection = payload.get("selection")
    selection_path = Path(str(selection["path"]))
    if _sha256(selection_path) != selection.get("sha256"):
        raise ValueError("selection hash does not match")
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    survival = {
        arm: [int(summaries_by_key[(arm, phase)]["steps"]) for phase in PHASES]
        for arm in ARMS
    }
    parent_body = np.asarray(
        [
            summaries_by_key[("parent", phase)]["mean_body_position_error"]
            for phase in PHASES
        ],
        dtype=np.float64,
    )
    parent_orientation = np.asarray(
        [
            summaries_by_key[("parent", phase)]["mean_body_orientation_error"]
            for phase in PHASES
        ],
        dtype=np.float64,
    )
    gated_body = np.asarray(
        [
            summaries_by_key[("gated", phase)]["mean_body_position_error"]
            for phase in PHASES
        ],
        dtype=np.float64,
    )
    gated_orientation = np.asarray(
        [
            summaries_by_key[("gated", phase)]["mean_body_orientation_error"]
            for phase in PHASES
        ],
        dtype=np.float64,
    )
    metric_mask = (
        np.asarray(survival["parent"], dtype=np.int64) == EXPECTED_SUFFIXES
    )
    classifier_summaries = {
        "parent": {"survival": survival["parent"]},
        "global": {"survival": survival["global"]},
        "gated": {
            "survival": survival["gated"],
            "body_position_error_ratio": (
                gated_body / np.maximum(parent_body, 1e-12)
            ).tolist(),
            "body_orientation_error_ratio": (
                gated_orientation / np.maximum(parent_orientation, 1e-12)
            ).tolist(),
            "tracking_metric_mask": metric_mask.tolist(),
        },
    }
    gated_phase_zero_values = raw_by_key[("gated", 0)]["values"]
    if gated_phase_zero_values.shape[0] < 10:
        raise ValueError("gated phase-zero evidence is too short for selection")
    expected_outcome = classify_deviation_gate(
        parent=classifier_summaries["parent"],
        global_arm=classifier_summaries["global"],
        gated=classifier_summaries["gated"],
        final_body_position_error=gated_phase_zero_values[-10:, 7],
    )
    if selection_payload.get("outcome") != expected_outcome:
        raise ValueError("selection outcome disagrees with raw evidence")
    recorded_summaries = selection_payload.get("summaries")
    if not isinstance(recorded_summaries, dict):
        raise ValueError("selection summaries are missing")
    for arm in ARMS:
        if list(recorded_summaries.get(arm, {}).get("survival", ())) != survival[arm]:
            raise ValueError("selection survival disagrees with raw evidence")
    return payload


EXPECTED_CHECKPOINT_SHA256 = (
    "4f9a2b49c7368f5323ab81c4c3de4aae208413987ab4858c44bf76872d0f86dd"
)
EXPECTED_HPARAMS_SHA256 = (
    "6b60d0b8ea96fa27d633c6f80f8df82a6c09c848d58b9636fd75759bbda486f7"
)
EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def validate_preflight(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    """Bind the exact E026 composite and current clean evaluator code."""
    repository = args.repository.resolve()
    checkpoint = args.checkpoint.resolve()
    hparams_path = checkpoint.with_name("hparams.json")
    reference_path = args.reference_path.resolve()
    if args.seed != 0 or args.solver_profile != "g1-4x5":
        raise ValueError("evaluation requires seed zero and g1-4x5")
    if args.max_steps is not None and (args.max_steps < 1 or args.render):
        raise ValueError("max-steps is allowed only for a positive no-render smoke")
    if _git_output(repository, "rev-parse", "HEAD") != args.code_commit:
        raise ValueError("code commit does not match repository HEAD")
    if _git_output(repository, "status", "--porcelain"):
        raise ValueError("repository must be clean")
    expected_assets = {
        "checkpoint": (checkpoint, EXPECTED_CHECKPOINT_SHA256),
        "hparams": (hparams_path, EXPECTED_HPARAMS_SHA256),
        "reference": (reference_path, EXPECTED_REFERENCE_SHA256),
    }
    for label, (path, expected) in expected_assets.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label} provenance does not match")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    if not isinstance(hparams, dict):
        raise ValueError("checkpoint hparams must be an object")
    required_hparams = {
        "seed": 0,
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "actor_history_len": 10,
        "actor_reference_lookahead_steps": [4, 8, 12],
        "actor_reference_preview_mode": "delta",
        "actor_residual_preview_adapter": True,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "domain_randomization": False,
        "reference_reset_noise_scale": 0.0,
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
    }
    for key, expected in required_hparams.items():
        if hparams.get(key) != expected:
            raise ValueError(f"E026 hparams drifted at {key}")
    model_path = Path(str(hparams["xml_path"])).resolve()
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH

    controller_path = Path(DEFAULT_CONTROLLER_PATH).resolve()
    if _sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("runtime model provenance does not match")
    if _sha256(controller_path) != EXPECTED_CONTROLLER_SHA256:
        raise ValueError("runtime controller provenance does not match")
    provenance = {
        "code_commit": args.code_commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "hparams_path": str(hparams_path),
        "hparams_sha256": EXPECTED_HPARAMS_SHA256,
        "reference_path": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "controller_path": str(controller_path),
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
        "solver_profile": args.solver_profile,
        "seed": args.seed,
    }
    return hparams, provenance


def _rollout_summary(
    *, arm: str, phase: int, values: np.ndarray, remaining: int
) -> dict[str, object]:
    return {
        "valid": True,
        "arm": arm,
        "phase": phase,
        "steps": int(values.shape[0]),
        "remaining": remaining,
        "terminal": bool(values[-1, 4] > 0.5),
        "completed_suffix": bool(
            values.shape[0] == remaining and not np.any(values[:, 4] > 0.5)
        ),
        "mean_reward": float(np.mean(values[:, 2])),
        "mean_body_position_error": float(np.mean(values[:, 7])),
        "mean_body_orientation_error": float(np.mean(values[:, 8])),
        "max_body_position_error": float(np.max(values[:, 7])),
    }


def _write_video_atomic(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    imageio.mimsave(temporary, frames, fps=fps, quality=8)
    os.replace(temporary, path)


def _build_environment(hparams: Mapping[str, object], reference_path: Path):
    from tools.evaluate_g1_tracking import make_evaluation_env

    return make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        body_mass_scale=float(hparams["mass_range"][0]),
        effort_limit_scale=float(hparams["effort_limit_scale"]),
        reference_path=reference_path,
        reference_stride=int(hparams["reference_stride"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            int(value) for value in hparams["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(hparams["actor_reference_preview_mode"]),
        actor_observe_motion_anchor_position=bool(
            hparams.get("actor_observe_motion_anchor_position", False)
        ),
        tracking_velocity_kernel=str(
            hparams.get("tracking_velocity_kernel", "exponential")
        ),
        actor_observation_noise=False,
        domain_randomization=False,
        friction_range=tuple(float(value) for value in hparams["friction_range"]),
        kp_range=tuple(float(value) for value in hparams["kp_range"]),
        kd_range=tuple(float(value) for value in hparams["kd_range"]),
        com_offset_range=tuple(float(value) for value in hparams["com_offset_range"]),
        reference_reset_noise_scale=0.0,
        reference_residual_control=bool(hparams["reference_residual_control"]),
        reference_residual_scale=float(hparams["reference_residual_scale"]),
    )


def run_evaluation(args: argparse.Namespace) -> Path:
    """Execute the exact three-arm grid and return the completion artifact."""
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        apply_frozen_preview_residual,
    )
    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.evaluate_g1_phase_grid import make_contact_sheet
    from tools.evaluate_g1_tracking import (
        _load_policy,
        _render_pair,
        build_compiled_step,
        prepare_evaluation_action,
        remaining_reference_transitions,
    )

    jax.config.update("jax_enable_x64", True)
    hparams, provenance = validate_preflight(args)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "preflight.json",
        {
            "valid": True,
            "protocol": "g1-deviation-gated-recovery-v1",
            "provenance": provenance,
            "gate": {
                "lower": REGISTERED_DEVIATION_GATE.lower,
                "upper": REGISTERED_DEVIATION_GATE.upper,
            },
        },
    )
    env = _build_environment(hparams, args.reference_path.resolve())
    actor, actor_params, normalizer_state = _load_policy(
        env, args.checkpoint.resolve(), args.seed
    )
    if not isinstance(actor_params, FrozenPreviewResidualParams):
        raise ValueError("E026 checkpoint must contain frozen parent and residual")
    profile = get_solver_profile(args.solver_profile)
    with solver_context(profile):
        compiled_step = build_compiled_step(env)

    columns = np.asarray(
        (
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
    )
    actual_renderer = reference_renderer = actual_data = reference_data = None
    if args.render:
        import mujoco

        actual_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        reference_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        actual_data = mujoco.MjData(env.mj_model)
        reference_data = mujoco.MjData(env.mj_model)
    records = []
    summaries: dict[str, dict[int, dict[str, object]]] = {
        arm: {} for arm in ARMS
    }
    gated_phase_zero_frames: list[np.ndarray] = []
    reset_key = jax.random.PRNGKey(args.seed)
    normalizer = Normalizer(env.actor_frame_obs_dim)
    try:
        for phase in PHASES:
            with solver_context(profile):
                base_state = env.reset_at_phase(
                    reset_key, jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(phase)
                )
            remaining = remaining_reference_transitions(
                env.reference_length, phase, env.reference_stride
            )
            step_limit = remaining if args.max_steps is None else min(args.max_steps, remaining)
            for arm in ARMS:
                state = base_state
                rows = []
                pre_errors = []
                gates = []
                parent_actions = []
                residual_actions = []
                gated_residual_actions = []
                candidate_actions = []
                effective_actions = []
                qpos = []
                qvel = []
                for step in range(step_limit):
                    current_phase = int(state.info["phase"])
                    if args.render and arm == "gated" and phase == 0:
                        gated_phase_zero_frames.append(
                            _render_pair(
                                env,
                                np.asarray(state.data.qpos),
                                np.asarray(state.data.qvel),
                                current_phase,
                                actual_renderer,
                                reference_renderer,
                                actual_data,
                                reference_data,
                            )
                        )
                    normalized = env.normalize_actor_obs(
                        normalizer, normalizer_state, state.obs
                    ).astype(jnp.float32)
                    _, parent_action, residual_action = apply_frozen_preview_residual(
                        actor.parent_actor,
                        actor.residual_actor,
                        actor_params,
                        normalized,
                        history_len=env.actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                    )
                    pre_error = state.metrics["body_position_error"]
                    gate = deviation_recovery_gate(pre_error)
                    if arm == "parent":
                        gated_residual = jnp.zeros_like(residual_action)
                    elif arm == "global":
                        gated_residual = residual_action
                    else:
                        gated_residual = gate * residual_action
                    candidate = parent_action + gated_residual
                    effective = prepare_evaluation_action(
                        candidate.astype(jnp.float64),
                        squash=bool(env.clip_sampled_actor_actions),
                    )
                    with solver_context(profile):
                        state = compiled_step(state, effective)
                    next_phase = min(
                        current_phase + env.reference_stride,
                        env.reference_length - 1,
                    )
                    rows.append(
                        (
                            step,
                            current_phase,
                            float(state.reward),
                            float(state.done),
                            float(state.info["terminal"]),
                            float(state.metrics["anchor_position_error"]),
                            float(state.metrics["anchor_orientation_error"]),
                            float(state.metrics["body_position_error"]),
                            float(state.metrics["body_orientation_error"]),
                            float(state.metrics["body_linear_velocity_error"]),
                            float(state.metrics["body_angular_velocity_error"]),
                            next_phase,
                            float(state.metrics["termination_anchor_z_error"]),
                            float(state.metrics["termination_anchor_xy_error"]),
                            float(state.metrics["termination_gravity_z_error"]),
                            float(state.metrics["termination_distal_z_error"]),
                        )
                    )
                    pre_errors.append(np.asarray(pre_error))
                    gates.append(np.asarray(gate))
                    parent_actions.append(np.asarray(parent_action))
                    residual_actions.append(np.asarray(residual_action))
                    gated_residual_actions.append(np.asarray(gated_residual))
                    candidate_actions.append(np.asarray(candidate))
                    effective_actions.append(np.asarray(effective))
                    qpos.append(np.asarray(state.data.qpos))
                    qvel.append(np.asarray(state.data.qvel))
                    if float(state.done) > 0.5:
                        break
                values = np.asarray(rows, dtype=np.float64)
                arm_dir = output / "arms" / arm
                raw_path = arm_dir / f"phase_{phase:03d}.npz"
                summary_path = raw_path.with_suffix(".json")
                atomic_npz(
                    raw_path,
                    columns=columns,
                    values=values,
                    pre_body_position_error=np.asarray(pre_errors),
                    gate=np.asarray(gates),
                    parent_action=np.asarray(parent_actions),
                    raw_residual_action=np.asarray(residual_actions),
                    gated_residual_action=np.asarray(gated_residual_actions),
                    candidate_action=np.asarray(candidate_actions),
                    sampled_action=np.asarray(candidate_actions),
                    effective_action=np.asarray(effective_actions),
                    qpos=np.asarray(qpos),
                    qvel=np.asarray(qvel),
                )
                validate_raw_rollout(raw_path, arm=arm, phase=phase)
                summary = _rollout_summary(
                    arm=arm, phase=phase, values=values, remaining=remaining
                )
                atomic_json(summary_path, summary)
                summaries[arm][phase] = summary
                records.append(
                    {
                        "arm": arm,
                        "phase": phase,
                        "path": str(raw_path),
                        "summary_path": str(summary_path),
                    }
                )
    finally:
        if actual_renderer is not None:
            actual_renderer.close()
        if reference_renderer is not None:
            reference_renderer.close()

    if not args.render:
        smoke_path = output / "smoke.json"
        atomic_json(
            smoke_path,
            {
                "valid": True,
                "publication_complete": False,
                "reason": "no-render bounded smoke is not scientific evidence",
                "records": len(records),
            },
        )
        return smoke_path

    survival = {
        arm: [int(summaries[arm][phase]["steps"]) for phase in PHASES]
        for arm in ARMS
    }
    parent_body = np.asarray(
        [summaries["parent"][phase]["mean_body_position_error"] for phase in PHASES]
    )
    parent_orientation = np.asarray(
        [summaries["parent"][phase]["mean_body_orientation_error"] for phase in PHASES]
    )
    gated_body = np.asarray(
        [summaries["gated"][phase]["mean_body_position_error"] for phase in PHASES]
    )
    gated_orientation = np.asarray(
        [summaries["gated"][phase]["mean_body_orientation_error"] for phase in PHASES]
    )
    classifier_summary = {
        "parent": {"survival": survival["parent"]},
        "global": {"survival": survival["global"]},
        "gated": {
            "survival": survival["gated"],
            "body_position_error_ratio": (
                gated_body / np.maximum(parent_body, 1e-12)
            ).tolist(),
            "body_orientation_error_ratio": (
                gated_orientation / np.maximum(parent_orientation, 1e-12)
            ).tolist(),
            "tracking_metric_mask": (
                np.asarray(survival["parent"], dtype=np.int64)
                == EXPECTED_SUFFIXES
            ).tolist(),
        },
    }
    gated_phase_zero_raw = Path(
        next(
            str(record["path"])
            for record in records
            if record["arm"] == "gated" and record["phase"] == 0
        )
    )
    with np.load(gated_phase_zero_raw, allow_pickle=False) as archive:
        phase_zero_body_error = np.asarray(archive["values"][:, 7])
    if phase_zero_body_error.shape[0] < 10:
        raise ValueError("gated phase-zero rollout is too short for the tail gate")
    outcome = classify_deviation_gate(
        parent=classifier_summary["parent"],
        global_arm=classifier_summary["global"],
        gated=classifier_summary["gated"],
        final_body_position_error=phase_zero_body_error[-10:],
    )
    selection_path = output / "selection.json"
    atomic_json(
        selection_path,
        {
            "valid": True,
            "outcome": outcome,
            "summaries": classifier_summary,
            "final_body_position_error": phase_zero_body_error[-10:].tolist(),
        },
    )
    if len(gated_phase_zero_frames) < 1:
        raise ValueError("gated phase-zero render produced no frames")
    video_path = output / "gated_phase0.mp4"
    sheet_path = output / "gated_phase0_contact_sheet.png"
    _write_video_atomic(video_path, gated_phase_zero_frames, fps=round(1.0 / env.dt))
    temporary_sheet = sheet_path.with_name(f".{sheet_path.stem}.tmp{sheet_path.suffix}")
    make_contact_sheet(gated_phase_zero_frames, temporary_sheet)
    os.replace(temporary_sheet, sheet_path)
    completion_path = output / "completion.json"
    build_completion_manifest(
        completion_path,
        records=records,
        selection_path=selection_path,
        provenance={
            **provenance,
            "video_path": str(video_path),
            "video_sha256": _sha256(video_path),
            "contact_sheet_path": str(sheet_path),
            "contact_sheet_sha256": _sha256(sheet_path),
            "preflight_path": str(output / "preflight.json"),
            "preflight_sha256": _sha256(output / "preflight.json"),
        },
    )
    validate_completion_manifest(completion_path)
    return completion_path


def main() -> None:
    args = build_parser().parse_args()
    artifact = run_evaluation(args)
    print(artifact)


if __name__ == "__main__":
    main()
