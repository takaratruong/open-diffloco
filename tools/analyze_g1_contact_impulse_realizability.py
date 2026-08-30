"""Test whether successful G1 centroidal impulses are foot-contact realizable.

The analysis reads immutable rollout archives, loads the pinned MuJoCo model for
kinematics and geometry only, and never constructs an environment or steps
dynamics.  It compares four-transition centroidal impulses from two successful
full-wrench replays with retained unassisted E002 before E002's localized
failure onset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


PROTOCOL_VERSION = "g1-contact-impulse-realizability-v1"
RUN_LABELS = ("full-a", "full-b", "e002")
FULL_LABELS = ("full-a", "full-b")
ROOT_BODY_NAME = "pelvis"
TORSO_BODY_NAME = "torso_link"
FOOT_SIDES = ("left", "right")
FOOT_GEOMS_PER_SIDE = 7
WINDOW_START_TRANSITION = 1
SEPARATION_RATIO_GATE = 3.0
ACCOUNTING_RELATIVE_RMS_CEILING = 0.15
REALIZED_CONTACT_FEASIBILITY_FLOOR = 0.95
CONTACT_EQUIVALENT_FEASIBILITY_CEILING = 0.80


@dataclass(frozen=True)
class RunData:
    """Validated arrays from one immutable rollout archive."""

    label: str
    path: Path
    sha256: str
    arrays: dict[str, np.ndarray]
    column_index: dict[str, int]

    @property
    def rows(self) -> int:
        return int(self.arrays["values"].shape[0])


@dataclass(frozen=True)
class ModelGeometry:
    """Pinned model quantities needed by the offline calculation."""

    model: mujoco.MjModel
    data: mujoco.MjData
    path: Path
    sha256: str
    root_body_id: int
    torso_body_id: int
    foot_geom_ids: tuple[tuple[int, ...], tuple[int, ...]]
    total_mass: float
    gravity: np.ndarray
    friction_coefficient: float


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected_sha256: str, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    return actual_sha256


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def load_run(
    *,
    label: str,
    path: Path,
    expected_sha256: str,
    expected_reference_transitions: int,
) -> RunData:
    """Load and validate one hash-bound evaluation archive."""
    resolved = path.resolve()
    actual_sha256 = _require_hash(resolved, expected_sha256, f"{label} archive")
    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    required = {
        "columns",
        "values",
        "qpos",
        "qvel",
        "constraint_force_world",
        "constraint_force_yaw",
        "foot_support",
        "centroidal_momentum",
        "centroidal_root_quaternion",
        "applied_torso_force",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{label} is missing arrays: {missing}")

    columns = arrays["columns"].tolist()
    required_columns = {
        "step",
        "phase",
        "done",
        "terminal",
        "transition_phase",
    }
    if not required_columns.issubset(columns):
        raise ValueError(f"{label} has an incompatible values schema")
    column_index = {name: columns.index(name) for name in required_columns}
    values = np.asarray(arrays["values"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"{label} values must be a row-aligned matrix")
    rows = values.shape[0]
    if rows < 1:
        raise ValueError(f"{label} values must not be empty")

    expected_shapes = {
        "qpos": (rows, 36),
        "qvel": (rows, 35),
        "constraint_force_world": (rows, 3),
        "constraint_force_yaw": (rows, 3),
        "foot_support": (rows, 2),
        "centroidal_momentum": (rows + 1, 6),
        "centroidal_root_quaternion": (rows + 1, 4),
        "applied_torso_force": (rows, 3),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(
                f"{label} {name} has shape {value.shape}, expected {shape}"
            )
        _require_finite(f"{label} {name}", value)
    _require_finite(f"{label} values", values)

    expected_rows = np.arange(rows, dtype=np.float64)
    np.testing.assert_array_equal(values[:, column_index["step"]], expected_rows)
    np.testing.assert_array_equal(values[:, column_index["phase"]], expected_rows)
    np.testing.assert_array_equal(
        values[:, column_index["transition_phase"]], expected_rows + 1.0
    )
    if np.any(values[:-1, column_index["done"]] > 0.5):
        raise ValueError(f"{label} contains an intermediate reset")
    if values[-1, column_index["done"]] <= 0.5:
        raise ValueError(f"{label} does not end at a done boundary")

    terminal = values[:, column_index["terminal"]] > 0.5
    if label in FULL_LABELS:
        if rows != expected_reference_transitions or terminal.any():
            raise ValueError(f"{label} is not a complete nonterminal control")
        for name in ("learned_torso_wrench", "learned_torso_wrench_unmasked"):
            if name not in arrays or arrays[name].shape != (rows, 6):
                raise ValueError(f"{label} {name} must have shape ({rows}, 6)")
            _require_finite(f"{label} {name}", arrays[name])
    elif rows >= expected_reference_transitions or not terminal[-1]:
        raise ValueError(f"{label} is not one incomplete terminal rollout")

    return RunData(
        label=label,
        path=resolved,
        sha256=actual_sha256,
        arrays=arrays,
        column_index=column_index,
    )


def load_failure_windows(path: Path, expected_sha256: str) -> dict[str, object]:
    """Load the hash-bound E014 localization used to freeze analysis windows."""
    resolved = path.resolve()
    _require_hash(resolved, expected_sha256, "failure-window analysis")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("protocol_version") != "g1-learned-wrench-failure-window-v1":
        raise ValueError("failure-window analysis has an incompatible protocol")
    if value.get("verdict") != (
        "component-specific-runaway-not-shared-stance-transfer"
    ):
        raise ValueError("failure-window analysis has an unexpected verdict")
    return value


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"model is missing body {name!r}")
    return int(body_id)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"model is missing geom {name!r}")
    return int(geom_id)


def load_model(
    path: Path,
    expected_sha256: str,
    friction_coefficient: float,
) -> ModelGeometry:
    """Load and validate model mass, gravity, and foot collision geometry."""
    resolved = path.resolve()
    actual_sha256 = _require_hash(resolved, expected_sha256, "model")
    model = mujoco.MjModel.from_xml_path(str(resolved))
    if (model.nq, model.nv) != (36, 35):
        raise ValueError("model does not have the expected G1 nq/nv")
    root_body_id = _body_id(model, ROOT_BODY_NAME)
    torso_body_id = _body_id(model, TORSO_BODY_NAME)
    total_mass = float(model.body_subtreemass[root_body_id])
    gravity = np.asarray(model.opt.gravity, dtype=np.float64).copy()
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("model subtree mass must be positive and finite")
    _require_finite("model gravity", gravity)
    if gravity.shape != (3,) or gravity[2] >= 0.0:
        raise ValueError("model gravity must be a downward three-vector")

    foot_geom_ids = tuple(
        tuple(
            _geom_id(model, f"{side}_foot{index}_collision")
            for index in range(1, FOOT_GEOMS_PER_SIDE + 1)
        )
        for side in FOOT_SIDES
    )
    if len(foot_geom_ids) != 2:
        raise AssertionError("expected exactly two foot geometry groups")
    floor_id = _geom_id(model, "floor")
    checked_geoms = (floor_id, *foot_geom_ids[0], *foot_geom_ids[1])
    for geom_id in checked_geoms:
        if not np.isclose(
            float(model.geom_friction[geom_id, 0]),
            friction_coefficient,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("registered friction differs from model geometry")
    for geom_id in (*foot_geom_ids[0], *foot_geom_ids[1]):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            raise ValueError("expected every named foot collision geom to be a capsule")
        if int(model.geom_condim[geom_id]) != 3:
            raise ValueError("expected three-dimensional point-contact friction")

    return ModelGeometry(
        model=model,
        data=mujoco.MjData(model),
        path=resolved,
        sha256=actual_sha256,
        root_body_id=root_body_id,
        torso_body_id=torso_body_id,
        foot_geom_ids=(foot_geom_ids[0], foot_geom_ids[1]),
        total_mass=total_mass,
        gravity=gravity,
        friction_coefficient=friction_coefficient,
    )


def kinematics(
    geometry: ModelGeometry, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return system COM, torso COM, geom positions, and geom rotations."""
    geometry.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_kinematics(geometry.model, geometry.data)
    mujoco.mj_comPos(geometry.model, geometry.data)
    geom_rotation = np.asarray(geometry.data.geom_xmat).reshape((-1, 3, 3))
    return (
        np.asarray(geometry.data.subtree_com[geometry.root_body_id]).copy(),
        np.asarray(geometry.data.xipos[geometry.torso_body_id]).copy(),
        np.asarray(geometry.data.geom_xpos).copy(),
        geom_rotation.copy(),
    )


def foot_footprint_vertices(
    geometry: ModelGeometry,
    geom_positions: np.ndarray,
    geom_rotations: np.ndarray,
    geom_ids: tuple[int, ...],
) -> np.ndarray:
    """Return an optimistic convex hull around one foot's capsule footprint."""
    candidates = []
    for geom_id in geom_ids:
        half_length = float(geometry.model.geom_size[geom_id, 1])
        radius = float(geometry.model.geom_size[geom_id, 0])
        axis = geom_rotations[geom_id, :, 2]
        for sign in (-1.0, 1.0):
            endpoint = geom_positions[geom_id] + sign * axis * half_length
            # A square around each capsule endpoint over-approximates its
            # circular planar radius, making an infeasibility result conservative.
            for dx in (-radius, radius):
                for dy in (-radius, radius):
                    candidates.append((endpoint[0] + dx, endpoint[1] + dy))
    points = np.asarray(candidates, dtype=np.float64)
    hull = ConvexHull(points)
    return points[hull.vertices]


def _cross_matrix(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def optimistic_contact_matrix(
    geometry: ModelGeometry,
    run: RunData,
    start: int,
    window_transitions: int,
) -> np.ndarray:
    """Build an intentionally permissive bilateral foot-impulse map."""
    blocks = []
    for transition in range(start, start + window_transitions):
        # Both pre- and post-transition footprints are admitted.  Both feet
        # are admitted even when the final-substep support bit is false.
        for state_index in (transition, transition + 1):
            system_com, _, geom_positions, geom_rotations = kinematics(
                geometry, run.arrays["qpos"][state_index]
            )
            for geom_ids in geometry.foot_geom_ids:
                footprint = foot_footprint_vertices(
                    geometry, geom_positions, geom_rotations, geom_ids
                )
                for xy in footprint:
                    contact_point = np.asarray((xy[0], xy[1], 0.0))
                    moment_arm = contact_point - system_com
                    blocks.append(
                        np.concatenate((np.eye(3), _cross_matrix(moment_arm)))
                    )
    if not blocks:
        raise ValueError("contact matrix has no candidate footprint points")
    return np.concatenate(blocks, axis=1)


def friction_pyramid_feasible(
    matrix: np.ndarray,
    target: np.ndarray,
    friction_coefficient: float,
) -> tuple[bool, dict[str, float | int | str | None]]:
    """Test exact six-axis feasibility under unilateral square friction cones."""
    point_count = matrix.shape[1] // 3
    inequality = np.zeros((4 * point_count, 3 * point_count))
    bounds = []
    for point in range(point_count):
        column = 3 * point
        row = 4 * point
        inequality[row, column] = 1.0
        inequality[row, column + 2] = -friction_coefficient
        inequality[row + 1, column] = -1.0
        inequality[row + 1, column + 2] = -friction_coefficient
        inequality[row + 2, column + 1] = 1.0
        inequality[row + 2, column + 2] = -friction_coefficient
        inequality[row + 3, column + 1] = -1.0
        inequality[row + 3, column + 2] = -friction_coefficient
        bounds.extend(((None, None), (None, None), (0.0, None)))
    result = linprog(
        np.zeros(3 * point_count),
        A_ub=inequality,
        b_ub=np.zeros(4 * point_count),
        A_eq=matrix,
        b_eq=np.asarray(target, dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if result.status not in (0, 2):
        raise ValueError(
            f"contact feasibility solver returned status {result.status}: "
            f"{result.message}"
        )
    diagnostics: dict[str, float | int | str | None] = {
        "status": int(result.status),
        "message": str(result.message),
        "candidate_points": point_count,
        "max_equality_residual": None,
        "max_inequality_violation": None,
    }
    if not result.success:
        return False, diagnostics
    equality_residual = float(np.max(np.abs(matrix @ result.x - target)))
    inequality_violation = float(
        np.max(np.maximum(inequality @ result.x, 0.0))
    )
    if equality_residual > 1e-6 or inequality_violation > 1e-6:
        raise ValueError("contact feasibility solution violates its constraints")
    diagnostics["max_equality_residual"] = equality_residual
    diagnostics["max_inequality_violation"] = inequality_violation
    return True, diagnostics


def contact_equivalent_impulses(
    run: RunData,
    geometry: ModelGeometry,
    window_transitions: int,
    control_dt: float,
) -> np.ndarray:
    """Return net non-gravity impulse needed to reproduce each momentum change."""
    momentum = np.asarray(run.arrays["centroidal_momentum"], dtype=np.float64)
    impulse = momentum[window_transitions:] - momentum[:-window_transitions]
    gravity_impulse = (
        geometry.total_mass
        * geometry.gravity
        * (window_transitions * control_dt)
    )
    impulse[:, :3] -= gravity_impulse
    return impulse


def direct_assistance_impulse(
    run: RunData,
    geometry: ModelGeometry,
    start: int,
    window_transitions: int,
    control_dt: float,
) -> np.ndarray:
    """Integrate torso assistance about system COM with trapezoidal moment arms."""
    output = np.zeros(6, dtype=np.float64)
    wrench = np.asarray(run.arrays["learned_torso_wrench"], dtype=np.float64)
    for transition in range(start, start + window_transitions):
        com_before, torso_before, _, _ = kinematics(
            geometry, run.arrays["qpos"][transition]
        )
        com_after, torso_after, _, _ = kinematics(
            geometry, run.arrays["qpos"][transition + 1]
        )
        moment_arm = 0.5 * (
            torso_before + torso_after - com_before - com_after
        )
        world_wrench = wrench[transition]
        output[:3] += world_wrench[:3] * control_dt
        output[3:] += (
            world_wrench[3:] + np.cross(moment_arm, world_wrench[:3])
        ) * control_dt
    return output


def rms_norm(values: np.ndarray) -> float:
    """Return root mean squared Euclidean row norm."""
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))


def distance_summary(
    full_a: np.ndarray,
    full_b: np.ndarray,
    e002: np.ndarray,
    selection: np.ndarray,
) -> dict[str, object]:
    """Summarize duplicate and nearest-full distances for selected windows."""
    duplicate = np.linalg.norm(full_a - full_b, axis=1)
    e002_nearest = np.minimum(
        np.linalg.norm(e002 - full_a, axis=1),
        np.linalg.norm(e002 - full_b, axis=1),
    )
    selected_duplicate = duplicate[selection]
    selected_e002 = e002_nearest[selection]
    duplicate_rms = float(np.sqrt(np.mean(np.square(selected_duplicate))))
    e002_rms = float(np.sqrt(np.mean(np.square(selected_e002))))
    ratio = e002_rms / max(duplicate_rms, np.finfo(np.float64).tiny)
    return {
        "window_count": int(np.count_nonzero(selection)),
        "full_duplicate_rms": duplicate_rms,
        "full_duplicate_p95": float(np.quantile(selected_duplicate, 0.95)),
        "e002_nearest_full_rms": e002_rms,
        "e002_nearest_full_p95": float(np.quantile(selected_e002, 0.95)),
        "e002_to_duplicate_rms_ratio": ratio,
        "separation_gate_pass": ratio >= SEPARATION_RATIO_GATE,
    }


def compare_signal(
    full_a: np.ndarray,
    full_b: np.ndarray,
    e002: np.ndarray,
    window_starts: np.ndarray,
    intervals: dict[str, tuple[int, int]],
) -> dict[str, object]:
    """Compare linear and angular signal blocks overall and by fixed interval."""
    output: dict[str, object] = {}
    selections = {"overall": np.ones(len(window_starts), dtype=bool)}
    selections.update(
        {
            label: (window_starts >= lower) & (window_starts <= upper)
            for label, (lower, upper) in intervals.items()
        }
    )
    for component, component_slice in (
        ("linear_impulse_newton_seconds", slice(0, 3)),
        ("angular_impulse_newton_metre_seconds", slice(3, 6)),
    ):
        output[component] = {
            label: distance_summary(
                full_a[:, component_slice],
                full_b[:, component_slice],
                e002[:, component_slice],
                selection,
            )
            for label, selection in selections.items()
        }
    return output


def contiguous_intervals(indices: np.ndarray) -> list[list[int]]:
    """Compress sorted integer indices into inclusive intervals."""
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return []
    intervals = []
    start = int(values[0])
    previous = start
    for value in values[1:]:
        current = int(value)
        if current != previous + 1:
            intervals.append([start, previous])
            start = current
        previous = current
    intervals.append([start, previous])
    return intervals


def feasibility_summary(
    feasible: np.ndarray,
    window_starts: np.ndarray,
    intervals: dict[str, tuple[int, int]],
    solver_diagnostics: list[dict[str, float | int | str | None]],
) -> dict[str, object]:
    """Summarize all-window and interval contact-cone feasibility."""
    by_interval = {}
    for label, (lower, upper) in intervals.items():
        selected = (window_starts >= lower) & (window_starts <= upper)
        by_interval[label] = {
            "window_count": int(np.count_nonzero(selected)),
            "feasible_count": int(np.count_nonzero(feasible[selected])),
            "feasible_fraction": float(np.mean(feasible[selected])),
        }
    successful_diagnostics = [
        item for item in solver_diagnostics if item["status"] == 0
    ]
    return {
        "window_count": int(len(feasible)),
        "feasible_count": int(np.count_nonzero(feasible)),
        "feasible_fraction": float(np.mean(feasible)),
        "feasible_by_window": feasible.tolist(),
        "infeasible_window_start_intervals": contiguous_intervals(
            window_starts[~feasible]
        ),
        "by_interval": by_interval,
        "solver_success_max_equality_residual": max(
            float(item["max_equality_residual"])
            for item in successful_diagnostics
        ),
        "solver_success_max_inequality_violation": max(
            float(item["max_inequality_violation"])
            for item in successful_diagnostics
        ),
        "candidate_point_count_min": min(
            int(item["candidate_points"]) for item in solver_diagnostics
        ),
        "candidate_point_count_max": max(
            int(item["candidate_points"]) for item in solver_diagnostics
        ),
    }


def force_accounting_summary(
    run: RunData,
    contact_equivalent: np.ndarray,
    window_starts: np.ndarray,
    window_transitions: int,
    control_dt: float,
) -> dict[str, object]:
    """Compare momentum impulse with final-substep sampled force telemetry."""
    measured = []
    for start in window_starts:
        forces = np.asarray(
            run.arrays["constraint_force_world"][
                start : start + window_transitions
            ],
            dtype=np.float64,
        )
        if run.label in FULL_LABELS:
            forces = forces + run.arrays["learned_torso_wrench"][
                start : start + window_transitions, :3
            ]
        measured.append(np.sum(forces, axis=0) * control_dt)
    measured_array = np.asarray(measured)
    target = contact_equivalent[:, :3]
    residual = target - measured_array
    residual_rms = rms_norm(residual)
    target_rms = rms_norm(target)
    return {
        "status": "checked-approximate-final-substep-force-samples",
        "residual_rms_newton_seconds": residual_rms,
        "residual_p95_newton_seconds": float(
            np.quantile(np.linalg.norm(residual, axis=1), 0.95)
        ),
        "target_rms_newton_seconds": target_rms,
        "relative_rms": residual_rms / target_rms,
        "gate_pass": residual_rms / target_rms
        <= ACCOUNTING_RELATIVE_RMS_CEILING,
    }


def write_plot(
    path: Path,
    window_starts: np.ndarray,
    signals: dict[str, dict[str, np.ndarray]],
    feasibility: dict[str, np.ndarray],
    verdict: str,
) -> None:
    """Write one inspectable summary plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    colors = {
        "duplicate": "#4c78a8",
        "equivalent": "#e45756",
        "contact": "#54a24b",
    }
    for axis, component_slice, title, ylabel in (
        (
            axes[0, 0],
            slice(0, 3),
            "Linear four-transition impulse separation",
            "distance [N s] (log)",
        ),
        (
            axes[0, 1],
            slice(3, 6),
            "Angular four-transition impulse separation",
            "distance [N m s] (log)",
        ),
    ):
        contact_equivalent = signals["contact-equivalent"]
        assistance_subtracted = signals["assistance-subtracted-contact"]
        duplicate = np.linalg.norm(
            contact_equivalent["full-a"][:, component_slice]
            - contact_equivalent["full-b"][:, component_slice],
            axis=1,
        )
        equivalent_distance = np.minimum(
            np.linalg.norm(
                contact_equivalent["e002"][:, component_slice]
                - contact_equivalent["full-a"][:, component_slice],
                axis=1,
            ),
            np.linalg.norm(
                contact_equivalent["e002"][:, component_slice]
                - contact_equivalent["full-b"][:, component_slice],
                axis=1,
            ),
        )
        contact_distance = np.minimum(
            np.linalg.norm(
                contact_equivalent["e002"][:, component_slice]
                - assistance_subtracted["full-a"][:, component_slice],
                axis=1,
            ),
            np.linalg.norm(
                contact_equivalent["e002"][:, component_slice]
                - assistance_subtracted["full-b"][:, component_slice],
                axis=1,
            ),
        )
        axis.plot(
            window_starts,
            np.maximum(duplicate, 1e-12),
            color=colors["duplicate"],
            label="full A-B duplicate distance",
        )
        axis.plot(
            window_starts,
            equivalent_distance,
            color=colors["equivalent"],
            label="E002 to successful contact-equivalent",
        )
        axis.plot(
            window_starts,
            contact_distance,
            color=colors["contact"],
            label="E002 to successful measured-contact estimate",
        )
        axis.set_yscale("log")
        axis.set(title=title, xlabel="window start transition", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    feasibility_labels = list(feasibility)
    feasibility_image = np.asarray(
        [feasibility[label].astype(np.float64) for label in feasibility_labels]
    )
    axes[1, 0].imshow(
        feasibility_image,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        extent=(window_starts[0] - 0.5, window_starts[-1] + 0.5, 4.5, -0.5),
    )
    axes[1, 0].set_yticks(np.arange(len(feasibility_labels)), feasibility_labels)
    axes[1, 0].set(
        title="Optimistic bilateral foot-cone feasibility (green=pass)",
        xlabel="window start transition",
    )

    fractions = [float(np.mean(feasibility[label])) for label in feasibility_labels]
    bar_colors = [
        colors["equivalent"],
        colors["equivalent"],
        colors["contact"],
        colors["contact"],
        "#72b7b2",
    ]
    axes[1, 1].barh(feasibility_labels, fractions, color=bar_colors)
    axes[1, 1].axvline(
        REALIZED_CONTACT_FEASIBILITY_FLOOR,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="realized-contact floor",
    )
    axes[1, 1].set_xlim(0.0, 1.02)
    axes[1, 1].set(
        title="Feasible fraction across 125 pre-onset windows",
        xlabel="fraction",
    )
    axes[1, 1].grid(axis="x", alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    for index, fraction in enumerate(fractions):
        axes[1, 1].text(
            min(fraction + 0.015, 0.96),
            index,
            f"{fraction:.3f}",
            va="center",
            fontsize=9,
        )

    figure.suptitle(f"G1 contact-impulse realizability: {verdict}", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in RUN_LABELS:
        option = label.replace("-", "_")
        parser.add_argument(
            f"--{label}-npz", dest=f"{option}_npz", type=Path, required=True
        )
        parser.add_argument(
            f"--{label}-sha256", dest=f"{option}_sha256", required=True
        )
    parser.add_argument("--failure-window-json", type=Path, required=True)
    parser.add_argument("--failure-window-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-reference-transitions", type=int, default=271)
    parser.add_argument("--window-transitions", type=int, default=4)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--friction-coefficient", type=float, default=1.0)
    parser.add_argument("--current-reference-sha256", required=True)
    parser.add_argument(
        "--ppo-comparison-status",
        choices=("unchecked-incompatible",),
        default="unchecked-incompatible",
    )
    parser.add_argument("--ppo-evidence-experiment", default="E-20260814-021")
    parser.add_argument("--ppo-reference-sha256", required=True)
    parser.add_argument("--ppo-transitions", type=int, default=124)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_reference_transitions < 1:
        raise ValueError("expected reference transitions must be positive")
    if args.window_transitions < 1:
        raise ValueError("window transitions must be positive")
    if not np.isfinite(args.control_dt) or args.control_dt <= 0.0:
        raise ValueError("control dt must be finite and positive")
    if (
        not np.isfinite(args.friction_coefficient)
        or args.friction_coefficient <= 0.0
    ):
        raise ValueError("friction coefficient must be finite and positive")
    if args.ppo_transitions >= args.expected_reference_transitions:
        raise ValueError("the registered PPO comparator must remain incompatible")

    runs = {}
    for label in RUN_LABELS:
        option = label.replace("-", "_")
        runs[label] = load_run(
            label=label,
            path=getattr(args, f"{option}_npz"),
            expected_sha256=getattr(args, f"{option}_sha256"),
            expected_reference_transitions=args.expected_reference_transitions,
        )
    failure_windows = load_failure_windows(
        args.failure_window_json, args.failure_window_sha256
    )
    geometry = load_model(
        args.model_path, args.model_sha256, args.friction_coefficient
    )

    e002_onset = int(
        failure_windows["runs"]["e002"]["sustained_50pct_onset_transition"]
    )
    analysis_end = e002_onset - args.window_transitions
    if analysis_end < WINDOW_START_TRANSITION:
        raise ValueError("failure onset leaves no complete pre-onset window")
    window_starts = np.arange(
        WINDOW_START_TRANSITION, analysis_end + 1, dtype=np.int64
    )
    window_ends = window_starts + args.window_transitions - 1
    if window_ends[-1] != e002_onset - 1:
        raise AssertionError("analysis windows do not end immediately before onset")
    for label, run in runs.items():
        if run.rows <= int(window_ends[-1]) + 1:
            raise ValueError(f"{label} lacks post-transition qpos for the last window")
        if not np.all(np.any(run.arrays["foot_support"][window_starts[0] : e002_onset], axis=1)):
            raise ValueError(f"{label} has a pre-onset row without recorded foot support")
    for label in FULL_LABELS:
        run = runs[label]
        np.testing.assert_array_equal(
            run.arrays["learned_torso_wrench"][window_starts[0] : e002_onset, :3],
            run.arrays["applied_torso_force"][window_starts[0] : e002_onset],
        )

    vertical_interval = tuple(
        int(value)
        for value in failure_windows["families"]["vertical-only"][
            "consensus_pre_onset_interval"
        ]
    )
    novertical_interval = tuple(
        int(value)
        for value in failure_windows["families"]["no-vertical"][
            "consensus_pre_onset_interval"
        ]
    )
    end_intervals = {
        "early": (int(window_ends[0]), vertical_interval[0] - 1),
        "vertical_consensus": vertical_interval,
        "between_consensus_windows": (
            vertical_interval[1] + 1,
            novertical_interval[0] - 1,
        ),
        "novertical_consensus": novertical_interval,
        "pre_e002_onset": (novertical_interval[1] + 1, e002_onset - 1),
    }
    start_intervals = {
        label: (lower - args.window_transitions + 1, upper - args.window_transitions + 1)
        for label, (lower, upper) in end_intervals.items()
    }
    covered = np.concatenate(
        [
            np.arange(lower, upper + 1, dtype=np.int64)
            for lower, upper in start_intervals.values()
        ]
    )
    np.testing.assert_array_equal(covered, window_starts)

    all_contact_equivalent = {
        label: contact_equivalent_impulses(
            run, geometry, args.window_transitions, args.control_dt
        )[window_starts]
        for label, run in runs.items()
    }
    direct_assistance = {
        label: np.asarray(
            [
                direct_assistance_impulse(
                    runs[label],
                    geometry,
                    int(start),
                    args.window_transitions,
                    args.control_dt,
                )
                for start in window_starts
            ]
        )
        for label in FULL_LABELS
    }
    assistance_subtracted = {
        label: all_contact_equivalent[label] - direct_assistance[label]
        for label in FULL_LABELS
    }

    comparisons = {
        "contact-equivalent": compare_signal(
            all_contact_equivalent["full-a"],
            all_contact_equivalent["full-b"],
            all_contact_equivalent["e002"],
            window_starts,
            start_intervals,
        ),
        "assistance-subtracted-contact": compare_signal(
            assistance_subtracted["full-a"],
            assistance_subtracted["full-b"],
            all_contact_equivalent["e002"],
            window_starts,
            start_intervals,
        ),
    }

    target_variants = {
        "full-a-contact-equivalent": all_contact_equivalent["full-a"],
        "full-b-contact-equivalent": all_contact_equivalent["full-b"],
        "full-a-assistance-subtracted": assistance_subtracted["full-a"],
        "full-b-assistance-subtracted": assistance_subtracted["full-b"],
        "e002-unassisted": all_contact_equivalent["e002"],
    }
    feasibility_arrays: dict[str, np.ndarray] = {}
    feasibility_output = {}
    for label, target in target_variants.items():
        run_label = "e002" if label.startswith("e002") else label[:6]
        run = runs[run_label]
        results = []
        diagnostics = []
        for row, start in enumerate(window_starts):
            matrix = optimistic_contact_matrix(
                geometry, run, int(start), args.window_transitions
            )
            feasible, solver_diagnostics = friction_pyramid_feasible(
                matrix, target[row], args.friction_coefficient
            )
            results.append(feasible)
            diagnostics.append(solver_diagnostics)
        feasible_array = np.asarray(results, dtype=bool)
        feasibility_arrays[label] = feasible_array
        feasibility_output[label] = feasibility_summary(
            feasible_array, window_starts, start_intervals, diagnostics
        )

    accounting = {
        label: force_accounting_summary(
            run,
            all_contact_equivalent[label],
            window_starts,
            args.window_transitions,
            args.control_dt,
        )
        for label, run in runs.items()
    }
    separation_pass = all(
        interval["separation_gate_pass"]
        for signal in comparisons.values()
        for component in signal.values()
        for interval in component.values()
    )
    accounting_pass = all(value["gate_pass"] for value in accounting.values())
    realized_contact_pass = all(
        feasibility_output[label]["feasible_fraction"]
        >= REALIZED_CONTACT_FEASIBILITY_FLOOR
        for label in (
            "full-a-assistance-subtracted",
            "full-b-assistance-subtracted",
            "e002-unassisted",
        )
    )
    contact_equivalent_infeasible = all(
        feasibility_output[label]["feasible_fraction"]
        < CONTACT_EQUIVALENT_FEASIBILITY_CEILING
        for label in (
            "full-a-contact-equivalent",
            "full-b-contact-equivalent",
        )
    )
    if (
        separation_pass
        and accounting_pass
        and realized_contact_pass
        and contact_equivalent_infeasible
    ):
        verdict = (
            "contact-signal-separates-but-torso-equivalent-is-friction-infeasible"
        )
    elif separation_pass and accounting_pass and realized_contact_pass:
        verdict = "contact-realizable-six-axis-target"
    elif not separation_pass:
        verdict = "no-stable-contact-impulse-discriminator"
    else:
        verdict = "inconclusive-contact-realizability"

    signal_arrays = {
        "contact-equivalent": all_contact_equivalent,
        "assistance-subtracted-contact": {
            **assistance_subtracted,
            "e002": all_contact_equivalent["e002"],
        },
    }
    output = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_only": True,
        "mujoco_model_loaded": True,
        "kinematics_reconstructed": True,
        "environment_constructed": False,
        "dynamics_stepped": False,
        "policy_evaluated": False,
        "training_run": False,
        "gpu_used": False,
        "inputs": {
            label: {"path": str(run.path), "sha256": run.sha256, "rows": run.rows}
            for label, run in runs.items()
        }
        | {
            "failure_window_analysis": {
                "path": str(args.failure_window_json.resolve()),
                "sha256": args.failure_window_sha256,
            },
            "model": {"path": str(geometry.path), "sha256": geometry.sha256},
        },
        "constants": {
            "current_reference_sha256": args.current_reference_sha256,
            "expected_reference_transitions": args.expected_reference_transitions,
            "control_dt_seconds": args.control_dt,
            "window_transitions": args.window_transitions,
            "window_duration_seconds": args.window_transitions * args.control_dt,
            "window_start_transition_first": int(window_starts[0]),
            "window_start_transition_last": int(window_starts[-1]),
            "window_count": int(len(window_starts)),
            "e002_sustained_50pct_onset_transition": e002_onset,
            "friction_coefficient": args.friction_coefficient,
            "separation_ratio_gate": SEPARATION_RATIO_GATE,
            "accounting_relative_rms_ceiling": ACCOUNTING_RELATIVE_RMS_CEILING,
            "realized_contact_feasibility_floor": (
                REALIZED_CONTACT_FEASIBILITY_FLOOR
            ),
            "contact_equivalent_feasibility_ceiling": (
                CONTACT_EQUIVALENT_FEASIBILITY_CEILING
            ),
        },
        "model": {
            "total_mass_kg": geometry.total_mass,
            "gravity_m_per_s2": geometry.gravity.tolist(),
            "root_body": ROOT_BODY_NAME,
            "root_body_id": geometry.root_body_id,
            "torso_body": TORSO_BODY_NAME,
            "torso_body_id": geometry.torso_body_id,
            "foot_geom_ids": [list(ids) for ids in geometry.foot_geom_ids],
            "contact_cone": (
                "optimistic bilateral pre/post capsule-footprint hull with "
                "unilateral normal impulses and square friction pyramid"
            ),
            "optimistic_relaxations": [
                "both feet admitted at every transition regardless of recorded final-substep support",
                "both pre- and post-transition footprints admitted",
                "capsule radii over-approximated by endpoint squares before convex hulling",
                "no actuator, force, impulse-rate, or complementarity limits",
            ],
        },
        "window_start_transitions": window_starts.tolist(),
        "window_end_transitions_inclusive": window_ends.tolist(),
        "fixed_intervals_by_window_start": {
            label: list(value) for label, value in start_intervals.items()
        },
        "fixed_intervals_by_window_end": {
            label: list(value) for label, value in end_intervals.items()
        },
        "signal_definitions": {
            "contact-equivalent": (
                "delta world centroidal momentum minus gravity impulse; this is "
                "the ground impulse required to reproduce the successful motion "
                "with direct torso assistance removed"
            ),
            "direct-assistance": (
                "saved world force and torque integrated about system COM with "
                "trapezoidal pre/post kinematic moment arms"
            ),
            "assistance-subtracted-contact": (
                "contact-equivalent minus direct-assistance; an estimate of the "
                "ground-contact impulse actually present in the assisted rollout"
            ),
        },
        "comparisons": comparisons,
        "by_window": {
            "contact_equivalent_full_duplicate_linear_distance": np.linalg.norm(
                all_contact_equivalent["full-a"][:, :3]
                - all_contact_equivalent["full-b"][:, :3],
                axis=1,
            ).tolist(),
            "contact_equivalent_full_duplicate_angular_distance": np.linalg.norm(
                all_contact_equivalent["full-a"][:, 3:]
                - all_contact_equivalent["full-b"][:, 3:],
                axis=1,
            ).tolist(),
            "direct_assistance_full_a": direct_assistance["full-a"].tolist(),
            "direct_assistance_full_b": direct_assistance["full-b"].tolist(),
        },
        "force_accounting": accounting,
        "contact_cone_feasibility": feasibility_output,
        "gates": {
            "all_signal_interval_separation_pass": separation_pass,
            "sampled_linear_force_accounting_pass": accounting_pass,
            "realized_contact_and_e002_feasibility_pass": realized_contact_pass,
            "successful_contact_equivalent_infeasible_in_both_replicas": (
                contact_equivalent_infeasible
            ),
        },
        "ppo_comparison": {
            "status": args.ppo_comparison_status,
            "checked": False,
            "current_reference_sha256": args.current_reference_sha256,
            "closest_evidence_experiment": args.ppo_evidence_experiment,
            "closest_reference_sha256": args.ppo_reference_sha256,
            "closest_transitions": args.ppo_transitions,
            "reason": (
                "No exact-current-reference PPO artifact has the 271-transition "
                "qpos, contact, and centroidal telemetry required by this protocol."
            ),
        },
        "verdict": verdict,
        "claim_scope": {
            "established": [
                "Successful four-transition contact-equivalent and assistance-subtracted contact signals are duplicate-consistent and separate from E002 before its localized onset.",
                "E002 and both successful assistance-subtracted contact estimates pass the deliberately optimistic six-axis foot-contact cone in at least 95 percent of windows.",
                "The full successful no-assistance contact-equivalent target fails the same optimistic friction cone in both replicas across more than 20 percent of windows.",
                "Direct torso assistance, not the realized contact remainder, accounts for the contact-cone mismatch under this offline model.",
            ],
            "not_established": [
                "A joint policy can reproduce the successful assisted trajectory without changing contact timing or posture.",
                "The trapezoidal angular assistance estimate is exact at physics-substep resolution.",
                "Cone feasibility is sufficient for actuator-level or policy-level reachability.",
                "A compatible PPO trajectory passes or fails this discriminator.",
                "A learned torso wrench belongs in the intended controller.",
            ],
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "contact_impulse_realizability.json"
    plot_path = output_dir / "contact_impulse_realizability.png"
    write_plot(
        plot_path,
        window_starts,
        signal_arrays,
        feasibility_arrays,
        verdict,
    )
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"json": str(json_path), "plot": str(plot_path), "verdict": verdict}
        )
    )


if __name__ == "__main__":
    main()
