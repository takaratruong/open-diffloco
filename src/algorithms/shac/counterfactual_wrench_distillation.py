"""Leg-only credit from a frozen assisted counterfactual transition."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from src.core.rmr_action_noise import RMR_ACTION_STD_JOINT_NAMES


LEG_ACTION_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

_BLOCK_NAMES = (
    "base_linear",
    "base_angular",
    "centroidal_linear",
    "centroidal_angular",
)


class CounterfactualFeasibility(NamedTuple):
    """Immutable inputs certified by the frozen-teacher discriminator."""

    target_rms: np.ndarray
    teacher_checkpoint_sha256: str
    teacher_tree_sha256: str
    e026_tree_sha256: str
    wrench_tree_sha256: str
    artifact_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_tree_sha256(tree: object) -> str:
    """Hash pytree structure, paths, dtypes, shapes, and exact leaf bytes."""
    digest = hashlib.sha256()
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    digest.update(repr(treedef).encode("utf-8"))
    for path, value in paths_and_leaves:
        digest.update(repr(path).encode("utf-8"))
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(b"array")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_counterfactual_feasibility(
    path: str | Path,
    *,
    expected_sha256: str,
) -> CounterfactualFeasibility:
    """Load one passing, hash-bound feasibility result or fail closed."""
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file() or _sha256_file(artifact_path) != expected_sha256:
        raise ValueError("counterfactual feasibility artifact hash mismatch")
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("counterfactual feasibility artifact is invalid") from error
    if (
        payload.get("protocol") != "g1-counterfactual-wrench-feasibility-v1"
        or payload.get("valid") is not True
        or payload.get("outcome") != "leg-counterfactual-feasible"
        or payload.get("phases") != [0, 25, 50, 75, 100]
        or payload.get("threshold") != 0.5
        or not isinstance(payload.get("phase_counts"), dict)
        or any(payload["phase_counts"].get(str(phase), 0) < 1 for phase in payload["phases"])
        or payload.get("row_count", 0) < 5
        or not np.isfinite(payload.get("median_normalized_residual", np.nan))
        or payload["median_normalized_residual"] > payload["threshold"]
    ):
        raise ValueError("counterfactual feasibility gate did not pass")
    target_rms = np.asarray(payload.get("target_rms"), dtype=np.float64)
    if target_rms.shape != (12,) or not np.all(np.isfinite(target_rms)) or np.any(target_rms <= 0.0):
        raise ValueError("counterfactual target RMS is invalid")
    npz_name = payload.get("npz_file")
    npz_sha = payload.get("npz_sha256")
    npz_path = artifact_path.parent / str(npz_name)
    if (
        not isinstance(npz_name, str)
        or Path(npz_name).name != npz_name
        or not isinstance(npz_sha, str)
        or not npz_path.is_file()
        or _sha256_file(npz_path) != npz_sha
    ):
        raise ValueError("counterfactual feasibility raw evidence is invalid")
    hashes = tuple(
        payload.get(key)
        for key in (
            "teacher_checkpoint_sha256",
            "teacher_tree_sha256",
            "e026_tree_sha256",
            "wrench_tree_sha256",
        )
    )
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ValueError("counterfactual feasibility provenance is invalid")
    return CounterfactualFeasibility(
        target_rms=target_rms,
        teacher_checkpoint_sha256=hashes[0],
        teacher_tree_sha256=hashes[1],
        e026_tree_sha256=hashes[2],
        wrench_tree_sha256=hashes[3],
        artifact_sha256=expected_sha256,
    )


def resolve_counterfactual_wrench_resume_setting(
    resumed_hparams: dict[str, object] | None,
    *,
    requested: bool,
    teacher_sha256: str | None,
    feasibility_sha256: str | None,
    allow_start: bool,
    is_resume: bool,
) -> tuple[bool, bool]:
    """Resolve the single explicit zero-wrench distillation upgrade."""
    if not all(isinstance(value, bool) for value in (requested, allow_start, is_resume)):
        raise ValueError("counterfactual settings must be boolean")
    if requested and (
        not isinstance(teacher_sha256, str)
        or len(teacher_sha256) != 64
        or not isinstance(feasibility_sha256, str)
        or len(feasibility_sha256) != 64
    ):
        raise ValueError("counterfactual source hashes are required")
    if not is_resume:
        if requested:
            raise ValueError("counterfactual distillation requires an E026 resume")
        return False, False
    if resumed_hparams is None:
        raise ValueError("counterfactual resume metadata is required")
    saved = resumed_hparams.get("actor_counterfactual_wrench_distillation", False)
    if not isinstance(saved, bool):
        raise ValueError("saved counterfactual setting is invalid")
    upgrade = bool(requested and not saved)
    if upgrade and not allow_start:
        raise ValueError("counterfactual distillation requires explicit authority")
    if saved != requested and not upgrade:
        raise ValueError("counterfactual setting must match the checkpoint")
    if saved:
        if resumed_hparams.get("actor_counterfactual_wrench_teacher_sha256") != teacher_sha256:
            raise ValueError("counterfactual teacher must match the checkpoint")
        if resumed_hparams.get("actor_counterfactual_wrench_feasibility_sha256") != feasibility_sha256:
            raise ValueError("counterfactual feasibility must match the checkpoint")
    return requested, upgrade


def resolve_leg_action_indices(actor_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Resolve the twelve legs in the exact registered 29-action order."""
    names = tuple(map(str, actor_joint_names))
    if len(names) != 29:
        raise ValueError("leg residual requires 29 canonical actor joints")
    if len(set(names)) != len(names):
        raise ValueError("actor joint names must be unique")
    if names != RMR_ACTION_STD_JOINT_NAMES:
        raise ValueError("actor joints must use the canonical order")
    indices = tuple(names.index(name) for name in LEG_ACTION_NAMES)
    if len(set(indices)) != 12:
        raise ValueError("leg residual indices must be unique")
    return indices


def scatter_leg_residual(
    residual: jax.Array,
    indices: Sequence[int],
    *,
    action_dim: int = 29,
) -> jax.Array:
    """Scatter twelve leg corrections into an otherwise exact-zero action."""
    values = jp.asarray(residual)
    static_indices = tuple(int(index) for index in indices)
    if values.ndim < 1 or values.shape[-1] != 12 or len(static_indices) != 12:
        raise ValueError("leg residual requires exactly twelve actions")
    if (
        action_dim != 29
        or len(set(static_indices)) != 12
        or min(static_indices) < 0
        or max(static_indices) >= action_dim
    ):
        raise ValueError("leg residual indices are invalid")
    values = _finite_or_nan(values, "leg residual values must be finite")
    output = jp.zeros(values.shape[:-1] + (action_dim,), dtype=values.dtype)
    return output.at[..., jp.asarray(static_indices, dtype=jp.int32)].set(values)


def counterfactual_target_change(
    before: jax.Array,
    after: jax.Array,
) -> jax.Array:
    """Return one finite 12-D local dynamics change."""
    before_values = jp.asarray(before)
    after_values = jp.asarray(after, dtype=before_values.dtype)
    if before_values.shape != (12,) or after_values.shape != (12,):
        raise ValueError("counterfactual change requires two 12-vectors")
    change = after_values - before_values
    return _finite_or_nan(change, "counterfactual change must be finite")


def counterfactual_transition_loss(
    student_change: jax.Array,
    teacher_change: jax.Array,
    target_rms: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare student and frozen-teacher local dynamics in four blocks."""
    student = jp.asarray(student_change)
    teacher = jp.asarray(teacher_change, dtype=student.dtype)
    rms = jp.asarray(target_rms, dtype=student.dtype)
    if student.shape != (12,) or teacher.shape != (12,) or rms.shape != (12,):
        raise ValueError("counterfactual loss requires three 12-vectors")
    valid = (
        jp.all(jp.isfinite(student))
        & jp.all(jp.isfinite(teacher))
        & jp.all(jp.isfinite(rms))
        & jp.all(rms >= 0.0)
    )
    safe_rms = jp.maximum(rms, jp.asarray(1e-3, dtype=student.dtype))
    normalized_error = (student - teacher) / safe_rms
    delta = jp.asarray(0.1, dtype=student.dtype)
    element_loss = jp.square(delta) * (
        jp.sqrt(1.0 + jp.square(normalized_error / delta)) - 1.0
    )
    block_losses = jp.mean(element_loss.reshape(4, 3), axis=-1)
    loss = jp.mean(block_losses)
    student_norm = jp.linalg.norm(student)
    teacher_norm = jp.linalg.norm(teacher)
    cosine = jp.vdot(student, teacher) / jp.maximum(
        student_norm * teacher_norm,
        jp.asarray(1e-12, dtype=student.dtype),
    )
    telemetry = {
        f"{name}_loss": block_losses[index]
        for index, name in enumerate(_BLOCK_NAMES)
    }
    telemetry.update(
        cosine=cosine,
        student_rms=jp.sqrt(jp.mean(jp.square(student))),
        teacher_rms=jp.sqrt(jp.mean(jp.square(teacher))),
        normalized_error_rms=jp.sqrt(jp.mean(jp.square(normalized_error))),
        valid=valid.astype(student.dtype),
    )
    failed = jp.asarray(jp.nan, dtype=student.dtype)
    return jp.where(valid, loss, failed), {
        name: jp.where(valid, value, failed) for name, value in telemetry.items()
    }


def _finite_or_nan(values: jax.Array, message: str) -> jax.Array:
    """Raise eagerly and propagate NaN under JIT for fail-closed execution."""
    if not isinstance(values, jax.core.Tracer):
        if not bool(np.all(np.isfinite(np.asarray(values)))):
            raise ValueError(message)
        return values
    valid = jp.all(jp.isfinite(values))
    return jp.where(valid, values, jp.full_like(values, jp.nan))
