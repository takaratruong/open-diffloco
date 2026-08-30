"""Project successful G1 net impulses into the pinned optimistic foot cone.

This is an offline follow-up to ``analyze_g1_contact_impulse_realizability``.
It reuses that analysis' immutable inputs, MuJoCo kinematics, candidate contact
points, and friction model.  For each four-transition window it computes a
body-weight-normalized weighted-L1 projection of the successful
contact-equivalent impulse into the contact cone, then bounds every discarded
residual component over the complete optimal set.  It never constructs an
environment, steps dynamics, evaluates a policy, or trains a controller.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import analyze_g1_contact_impulse_realizability as realizability


PROTOCOL_VERSION = "g1-contact-impulse-projection-v1"
AXIS_NAMES = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")
AXIS_UNITS = ("N s", "N s", "N s", "N m s", "N m s", "N m s")
WINDOW_START_TRANSITION = realizability.WINDOW_START_TRANSITION
FULL_LABELS = realizability.FULL_LABELS
RUN_LABELS = realizability.RUN_LABELS
SEPARATION_RATIO_GATE = realizability.SEPARATION_RATIO_GATE
SOLVER_CONSTRAINT_TOLERANCE = 1e-6
ZERO_COST_TOLERANCE = 1e-8
OPTIMAL_SET_COST_TOLERANCE = 1e-8
AMBIGUITY_NORMALIZED_WIDTH_CEILING = 0.05


def friction_constraints(
    point_count: int, friction_coefficient: float
) -> tuple[np.ndarray, list[tuple[float | None, float | None]]]:
    """Return the exact unilateral square-pyramid constraints used by E017."""
    inequality = np.zeros((4 * point_count, 3 * point_count))
    bounds: list[tuple[float | None, float | None]] = []
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
    return inequality, bounds


def solve_projection(
    matrix: np.ndarray,
    target: np.ndarray,
    friction_coefficient: float,
    component_scales: np.ndarray,
) -> dict[str, object]:
    """Solve one weighted-L1 projection and bound its optimal residual set."""
    target = np.asarray(target, dtype=np.float64)
    scales = np.asarray(component_scales, dtype=np.float64)
    if matrix.shape[0] != 6 or target.shape != (6,) or scales.shape != (6,):
        raise ValueError("projection expects a six-axis contact map and target")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("projection inputs must be finite")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("projection component scales must be finite and positive")

    point_count = matrix.shape[1] // 3
    contact_variables = 3 * point_count
    variable_count = contact_variables + 12
    residual_positive = slice(contact_variables, contact_variables + 6)
    residual_negative = slice(contact_variables + 6, variable_count)

    friction_inequality, contact_bounds = friction_constraints(
        point_count, friction_coefficient
    )
    inequality = np.pad(friction_inequality, ((0, 0), (0, 12)))
    inequality_rhs = np.zeros(inequality.shape[0])
    equality = np.concatenate(
        (matrix, np.eye(6, dtype=np.float64), -np.eye(6, dtype=np.float64)),
        axis=1,
    )
    bounds = contact_bounds + [(0.0, None)] * 12
    cost = np.zeros(variable_count)
    cost[residual_positive] = 1.0 / scales
    cost[residual_negative] = 1.0 / scales

    result = linprog(
        cost,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=target,
        bounds=bounds,
        method="highs-ipm",
    )
    if result.status != 0 or not result.success:
        raise ValueError(
            f"contact projection solver returned status {result.status}: "
            f"{result.message}"
        )

    contact_solution = result.x[:contact_variables]
    projected = matrix @ contact_solution
    residual = target - projected
    represented_residual = (
        result.x[residual_positive] - result.x[residual_negative]
    )
    equality_error = float(np.max(np.abs(equality @ result.x - target)))
    inequality_violation = float(
        np.max(np.maximum(inequality @ result.x - inequality_rhs, 0.0))
    )
    representation_error = float(
        np.max(np.abs(residual - represented_residual))
    )
    normalized_l1_cost = float(np.sum(np.abs(residual) / scales))
    objective_error = abs(normalized_l1_cost - float(result.fun))
    if max(equality_error, inequality_violation, representation_error) > SOLVER_CONSTRAINT_TOLERANCE:
        raise ValueError("contact projection solution violates its constraints")
    if objective_error > SOLVER_CONSTRAINT_TOLERANCE:
        raise ValueError("contact projection objective does not match residual")

    # A feasible target has the unique zero residual even though its contact
    # distribution may be non-unique.  Avoid amplifying numerical tolerance in
    # twelve unnecessary range solves for that case.
    if normalized_l1_cost <= ZERO_COST_TOLERANCE:
        projected = target.copy()
        residual = np.zeros(6)
        residual_bounds = np.zeros((6, 2))
    else:
        optimal_set_row = cost.copy()
        range_inequality = np.vstack((inequality, optimal_set_row))
        range_rhs = np.concatenate(
            (inequality_rhs, [float(result.fun) + OPTIMAL_SET_COST_TOLERANCE])
        )
        residual_bounds = np.empty((6, 2), dtype=np.float64)
        for axis in range(6):
            component_objective = np.zeros(variable_count)
            component_objective[contact_variables + axis] = 1.0
            component_objective[contact_variables + 6 + axis] = -1.0
            extrema = []
            for direction in (1.0, -1.0):
                range_result = linprog(
                    direction * component_objective,
                    A_ub=range_inequality,
                    b_ub=range_rhs,
                    A_eq=equality,
                    b_eq=target,
                    bounds=bounds,
                    method="highs-ipm",
                )
                if range_result.status != 0 or not range_result.success:
                    raise ValueError(
                        "optimal-set range solver returned status "
                        f"{range_result.status}: {range_result.message}"
                    )
                range_equality_error = float(
                    np.max(np.abs(equality @ range_result.x - target))
                )
                range_inequality_violation = float(
                    np.max(
                        np.maximum(
                            range_inequality @ range_result.x - range_rhs,
                            0.0,
                        )
                    )
                )
                if max(range_equality_error, range_inequality_violation) > SOLVER_CONSTRAINT_TOLERANCE:
                    raise ValueError("optimal-set range solution violates constraints")
                extrema.append(
                    float(component_objective @ range_result.x)
                )
            residual_bounds[axis] = (extrema[0], extrema[1])
        if np.any(residual < residual_bounds[:, 0] - SOLVER_CONSTRAINT_TOLERANCE) or np.any(
            residual > residual_bounds[:, 1] + SOLVER_CONSTRAINT_TOLERANCE
        ):
            raise ValueError("solver-selected residual lies outside optimal-set bounds")

    return {
        "projected": projected,
        "residual": residual,
        "residual_bounds": residual_bounds,
        "normalized_l1_cost": normalized_l1_cost,
        "candidate_points": point_count,
        "max_equality_residual": equality_error,
        "max_inequality_violation": inequality_violation,
        "max_residual_representation_error": representation_error,
        "objective_error": objective_error,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }


def interval_masks(
    window_starts: np.ndarray, intervals: dict[str, tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Return overall and registered E014 interval selections."""
    masks = {"overall": np.ones(len(window_starts), dtype=bool)}
    masks.update(
        {
            label: (window_starts >= lower) & (window_starts <= upper)
            for label, (lower, upper) in intervals.items()
        }
    )
    return masks


def residual_summary(
    residual: np.ndarray,
    residual_bounds: np.ndarray,
    component_scales: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    """Report discarded impulse by physical axis and fixed interval."""
    output: dict[str, object] = {}
    widths = residual_bounds[:, :, 1] - residual_bounds[:, :, 0]
    for interval, selection in masks.items():
        selected = residual[selection]
        selected_widths = widths[selection]
        square_energy = np.sum(np.square(selected), axis=0)
        total_energy = float(np.sum(square_energy))
        axes = {}
        for axis, (name, unit) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
            absolute = np.abs(selected[:, axis])
            axes[name] = {
                "unit": unit,
                "mean_signed": float(np.mean(selected[:, axis])),
                "rms": float(np.sqrt(np.mean(np.square(selected[:, axis])))),
                "mean_absolute": float(np.mean(absolute)),
                "median_absolute": float(np.median(absolute)),
                "p95_absolute": float(np.quantile(absolute, 0.95)),
                "max_absolute": float(np.max(absolute)),
                "squared_residual_fraction": (
                    float(square_energy[axis] / total_energy)
                    if total_energy > 0.0
                    else 0.0
                ),
                "optimal_set_width_max": float(np.max(selected_widths[:, axis])),
                "optimal_set_width_normalized_max": float(
                    np.max(selected_widths[:, axis]) / component_scales[axis]
                ),
            }
        normalized_l1 = np.sum(np.abs(selected) / component_scales, axis=1)
        output[interval] = {
            "window_count": int(np.count_nonzero(selection)),
            "zero_residual_count": int(
                np.count_nonzero(normalized_l1 <= ZERO_COST_TOLERANCE)
            ),
            "vector_rms_physical_mixed_units": realizability.rms_norm(selected),
            "normalized_l1_mean": float(np.mean(normalized_l1)),
            "normalized_l1_median": float(np.median(normalized_l1)),
            "normalized_l1_p95": float(np.quantile(normalized_l1, 0.95)),
            "normalized_l1_max": float(np.max(normalized_l1)),
            "axes": axes,
        }
    return output


def distance_summary(
    projected_a: np.ndarray,
    projected_b: np.ndarray,
    e002: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    """Measure duplicate consistency and E002 separation of projected targets."""
    output: dict[str, object] = {}
    for component, component_slice in (
        ("linear_impulse_newton_seconds", slice(0, 3)),
        ("angular_impulse_newton_metre_seconds", slice(3, 6)),
    ):
        duplicate = np.linalg.norm(
            projected_a[:, component_slice] - projected_b[:, component_slice],
            axis=1,
        )
        e002_nearest = np.minimum(
            np.linalg.norm(
                e002[:, component_slice] - projected_a[:, component_slice], axis=1
            ),
            np.linalg.norm(
                e002[:, component_slice] - projected_b[:, component_slice], axis=1
            ),
        )
        output[component] = {}
        for interval, selection in masks.items():
            duplicate_rms = float(
                np.sqrt(np.mean(np.square(duplicate[selection])))
            )
            e002_rms = float(
                np.sqrt(np.mean(np.square(e002_nearest[selection])))
            )
            ratio = e002_rms / max(
                duplicate_rms, np.finfo(np.float64).tiny
            )
            output[component][interval] = {
                "window_count": int(np.count_nonzero(selection)),
                "projected_full_duplicate_rms": duplicate_rms,
                "projected_full_duplicate_p95": float(
                    np.quantile(duplicate[selection], 0.95)
                ),
                "e002_nearest_projected_full_rms": e002_rms,
                "e002_nearest_projected_full_p95": float(
                    np.quantile(e002_nearest[selection], 0.95)
                ),
                "e002_to_duplicate_rms_ratio": ratio,
                "separation_gate_pass": ratio >= SEPARATION_RATIO_GATE,
            }
    return output


def assistance_comparison(
    residual: np.ndarray,
    assistance: np.ndarray,
    projected: np.ndarray,
    realized_contact: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    """Compare discarded impulse with the assisted rollout's direct wrench."""
    output = {}
    for interval, selection in masks.items():
        selected_residual = residual[selection]
        selected_assistance = assistance[selection]
        residual_norm = np.linalg.norm(selected_residual, axis=1)
        assistance_norm = np.linalg.norm(selected_assistance, axis=1)
        cosine = np.sum(selected_residual * selected_assistance, axis=1) / np.maximum(
            residual_norm * assistance_norm, np.finfo(np.float64).tiny
        )
        nonzero = (residual_norm > 1e-10) & (assistance_norm > 1e-10)
        output[interval] = {
            "window_count": int(np.count_nonzero(selection)),
            "discarded_residual_rms_physical_mixed_units": realizability.rms_norm(
                selected_residual
            ),
            "direct_assistance_rms_physical_mixed_units": realizability.rms_norm(
                selected_assistance
            ),
            "discarded_to_assistance_rms_ratio": (
                realizability.rms_norm(selected_residual)
                / max(
                    realizability.rms_norm(selected_assistance),
                    np.finfo(np.float64).tiny,
                )
            ),
            "discarded_assistance_cosine_mean_nonzero": (
                float(np.mean(cosine[nonzero])) if np.any(nonzero) else None
            ),
            "discarded_assistance_cosine_median_nonzero": (
                float(np.median(cosine[nonzero])) if np.any(nonzero) else None
            ),
            "projected_to_realized_contact_rms_physical_mixed_units": (
                realizability.rms_norm(
                    projected[selection] - realized_contact[selection]
                )
            ),
        }
    return output


def write_plot(
    path: Path,
    window_starts: np.ndarray,
    residuals: dict[str, np.ndarray],
    projected: dict[str, np.ndarray],
    e002: np.ndarray,
    component_scales: np.ndarray,
    verdict: str,
) -> None:
    """Write one compact inspectable projection summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    colors = {"full-a": "#4c78a8", "full-b": "#f58518"}
    for label in FULL_LABELS:
        normalized = np.abs(residuals[label]) / component_scales
        axes[0, 0].plot(
            window_starts,
            np.sum(normalized, axis=1),
            color=colors[label],
            label=label,
            alpha=0.9,
        )
    axes[0, 0].set(
        title="Discarded weighted-L1 residual",
        xlabel="window start transition",
        ylabel="sum absolute normalized residual",
    )
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    width = 0.36
    axis_index = np.arange(6)
    for offset, label in zip((-width / 2, width / 2), FULL_LABELS):
        rms = np.sqrt(np.mean(np.square(residuals[label]), axis=0))
        axes[0, 1].bar(
            axis_index + offset,
            rms / component_scales,
            width,
            color=colors[label],
            label=label,
        )
    axes[0, 1].set_xticks(axis_index, AXIS_NAMES, rotation=30, ha="right")
    axes[0, 1].set(
        title="Discarded residual RMS by axis",
        ylabel="body-weight normalized RMS",
    )
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    mean_normalized = 0.5 * (
        np.abs(residuals["full-a"]) + np.abs(residuals["full-b"])
    ) / component_scales
    image = axes[1, 0].imshow(
        mean_normalized.T,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=(window_starts[0] - 0.5, window_starts[-1] + 0.5, 5.5, -0.5),
    )
    axes[1, 0].set_yticks(np.arange(6), AXIS_NAMES)
    axes[1, 0].set(
        title="Mean absolute discarded residual (A/B)",
        xlabel="window start transition",
    )
    figure.colorbar(image, ax=axes[1, 0], label="normalized magnitude")

    for component_slice, linestyle, component in (
        (slice(0, 3), "-", "linear"),
        (slice(3, 6), "--", "angular"),
    ):
        duplicate = np.linalg.norm(
            projected["full-a"][:, component_slice]
            - projected["full-b"][:, component_slice],
            axis=1,
        )
        nearest_e002 = np.minimum(
            np.linalg.norm(
                e002[:, component_slice] - projected["full-a"][:, component_slice],
                axis=1,
            ),
            np.linalg.norm(
                e002[:, component_slice] - projected["full-b"][:, component_slice],
                axis=1,
            ),
        )
        axes[1, 1].plot(
            window_starts,
            np.maximum(duplicate, 1e-12),
            linestyle=linestyle,
            color="#54a24b",
            label=f"{component} A-B duplicate",
        )
        axes[1, 1].plot(
            window_starts,
            np.maximum(nearest_e002, 1e-12),
            linestyle=linestyle,
            color="#e45756",
            label=f"{component} E002-nearest",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="Projected-target duplicate consistency",
        xlabel="window start transition",
        ylabel="distance (component units, log)",
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(f"G1 contact-impulse projection: {verdict}", fontsize=15)
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
    parser.add_argument(
        "--angular-normalization-lever-arm", type=float, default=0.3
    )
    parser.add_argument("--current-reference-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_reference_transitions < 1 or args.window_transitions < 1:
        raise ValueError("transition counts must be positive")
    for name, value in (
        ("control dt", args.control_dt),
        ("friction coefficient", args.friction_coefficient),
        ("angular normalization lever arm", args.angular_normalization_lever_arm),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    runs = {}
    for label in RUN_LABELS:
        option = label.replace("-", "_")
        runs[label] = realizability.load_run(
            label=label,
            path=getattr(args, f"{option}_npz"),
            expected_sha256=getattr(args, f"{option}_sha256"),
            expected_reference_transitions=args.expected_reference_transitions,
        )
    failure_windows = realizability.load_failure_windows(
        args.failure_window_json, args.failure_window_sha256
    )
    geometry = realizability.load_model(
        args.model_path, args.model_sha256, args.friction_coefficient
    )

    e002_onset = int(
        failure_windows["runs"]["e002"]["sustained_50pct_onset_transition"]
    )
    analysis_end = e002_onset - args.window_transitions
    window_starts = np.arange(
        WINDOW_START_TRANSITION, analysis_end + 1, dtype=np.int64
    )
    window_ends = window_starts + args.window_transitions - 1
    if len(window_starts) != 125 or int(window_ends[-1]) != e002_onset - 1:
        raise ValueError("registered pre-onset analysis must contain 125 windows")

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
        label: (
            lower - args.window_transitions + 1,
            upper - args.window_transitions + 1,
        )
        for label, (lower, upper) in end_intervals.items()
    }
    masks = interval_masks(window_starts, start_intervals)
    coverage = np.sum(np.stack(list(masks.values())[1:]), axis=0)
    if not np.all(coverage == 1):
        raise ValueError("registered E014 intervals must partition all windows")

    all_contact_equivalent = {
        label: realizability.contact_equivalent_impulses(
            run, geometry, args.window_transitions, args.control_dt
        )[window_starts]
        for label, run in runs.items()
    }
    direct_assistance = {
        label: np.asarray(
            [
                realizability.direct_assistance_impulse(
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
    realized_contact = {
        label: all_contact_equivalent[label] - direct_assistance[label]
        for label in FULL_LABELS
    }

    linear_scale = (
        geometry.total_mass
        * abs(float(geometry.gravity[2]))
        * args.window_transitions
        * args.control_dt
    )
    angular_scale = linear_scale * args.angular_normalization_lever_arm
    component_scales = np.asarray(
        (linear_scale, linear_scale, linear_scale, angular_scale, angular_scale, angular_scale)
    )

    projected: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    residual_bounds: dict[str, np.ndarray] = {}
    costs: dict[str, np.ndarray] = {}
    solver_diagnostics: dict[str, list[dict[str, object]]] = {}
    for label in FULL_LABELS:
        rows = []
        for row, start in enumerate(window_starts):
            matrix = realizability.optimistic_contact_matrix(
                geometry, runs[label], int(start), args.window_transitions
            )
            rows.append(
                solve_projection(
                    matrix,
                    all_contact_equivalent[label][row],
                    args.friction_coefficient,
                    component_scales,
                )
            )
        projected[label] = np.asarray([item["projected"] for item in rows])
        residuals[label] = np.asarray([item["residual"] for item in rows])
        residual_bounds[label] = np.asarray(
            [item["residual_bounds"] for item in rows]
        )
        costs[label] = np.asarray([item["normalized_l1_cost"] for item in rows])
        solver_diagnostics[label] = [
            {
                key: item[key]
                for key in (
                    "candidate_points",
                    "max_equality_residual",
                    "max_inequality_violation",
                    "max_residual_representation_error",
                    "objective_error",
                    "solver_status",
                    "solver_message",
                )
            }
            for item in rows
        ]

    comparisons = distance_summary(
        projected["full-a"],
        projected["full-b"],
        all_contact_equivalent["e002"],
        masks,
    )
    summaries = {
        label: residual_summary(
            residuals[label], residual_bounds[label], component_scales, masks
        )
        for label in FULL_LABELS
    }
    assistance = {
        label: assistance_comparison(
            residuals[label],
            direct_assistance[label],
            projected[label],
            realized_contact[label],
            masks,
        )
        for label in FULL_LABELS
    }

    max_constraint_error = max(
        max(
            float(item[key])
            for label in FULL_LABELS
            for item in solver_diagnostics[label]
        )
        for key in (
            "max_equality_residual",
            "max_inequality_violation",
            "max_residual_representation_error",
            "objective_error",
        )
    )
    separation_pass = all(
        interval["separation_gate_pass"]
        for component in comparisons.values()
        for interval in component.values()
    )
    max_ambiguity_normalized_width = max(
        float(
            np.max(
                (bounds[:, :, 1] - bounds[:, :, 0]) / component_scales
            )
        )
        for bounds in residual_bounds.values()
    )
    ambiguity_pass = (
        max_ambiguity_normalized_width <= AMBIGUITY_NORMALIZED_WIDTH_CEILING
    )
    solver_pass = max_constraint_error <= SOLVER_CONSTRAINT_TOLERANCE
    if solver_pass and separation_pass and ambiguity_pass:
        verdict = "stable-contact-projected-target-with-axis-resolved-discard"
    elif solver_pass and separation_pass:
        verdict = "projected-signal-separates-but-weighted-l1-target-is-ambiguous"
    elif solver_pass:
        verdict = "contact-projected-target-is-not-duplicate-stable"
    else:
        verdict = "invalid-contact-projection-solve"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "contact_impulse_projection.json"
    arrays_path = output_dir / "contact_impulse_projection.npz"
    plot_path = output_dir / "contact_impulse_projection.png"
    np.savez_compressed(
        arrays_path,
        window_start_transitions=window_starts,
        window_end_transitions_inclusive=window_ends,
        component_scales=component_scales,
        contact_equivalent_full_a=all_contact_equivalent["full-a"],
        contact_equivalent_full_b=all_contact_equivalent["full-b"],
        e002_unassisted=all_contact_equivalent["e002"],
        projected_full_a=projected["full-a"],
        projected_full_b=projected["full-b"],
        discarded_residual_full_a=residuals["full-a"],
        discarded_residual_full_b=residuals["full-b"],
        discarded_residual_bounds_full_a=residual_bounds["full-a"],
        discarded_residual_bounds_full_b=residual_bounds["full-b"],
        realized_contact_full_a=realized_contact["full-a"],
        realized_contact_full_b=realized_contact["full-b"],
        direct_assistance_full_a=direct_assistance["full-a"],
        direct_assistance_full_b=direct_assistance["full-b"],
    )
    write_plot(
        plot_path,
        window_starts,
        residuals,
        projected,
        all_contact_equivalent["e002"],
        component_scales,
        verdict,
    )

    output = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": realizability.sha256_file(Path(__file__).resolve()),
        "analysis_only": True,
        "mujoco_model_loaded": True,
        "kinematics_reconstructed": True,
        "environment_constructed": False,
        "dynamics_stepped": False,
        "policy_evaluated": False,
        "training_run": False,
        "gpu_used": False,
        "inputs": {
            label: {
                "path": str(run.path),
                "sha256": run.sha256,
                "rows": run.rows,
            }
            for label, run in runs.items()
        }
        | {
            "failure_window_analysis": {
                "path": str(args.failure_window_json.resolve()),
                "sha256": args.failure_window_sha256,
            },
            "model": {
                "path": str(geometry.path),
                "sha256": geometry.sha256,
            },
        },
        "outputs": {
            "arrays": str(arrays_path),
            "plot": str(plot_path),
        },
        "constants": {
            "current_reference_sha256": args.current_reference_sha256,
            "expected_reference_transitions": args.expected_reference_transitions,
            "control_dt_seconds": args.control_dt,
            "window_transitions": args.window_transitions,
            "window_duration_seconds": args.window_transitions * args.control_dt,
            "window_count": int(len(window_starts)),
            "friction_coefficient": args.friction_coefficient,
            "linear_normalization_body_weight_window_impulse_newton_seconds": linear_scale,
            "angular_normalization_lever_arm_metres": args.angular_normalization_lever_arm,
            "angular_normalization_newton_metre_seconds": angular_scale,
            "component_scales": component_scales.tolist(),
            "separation_ratio_gate": SEPARATION_RATIO_GATE,
            "solver_constraint_tolerance": SOLVER_CONSTRAINT_TOLERANCE,
            "zero_cost_tolerance": ZERO_COST_TOLERANCE,
            "optimal_set_cost_tolerance": OPTIMAL_SET_COST_TOLERANCE,
            "ambiguity_normalized_width_ceiling": AMBIGUITY_NORMALIZED_WIDTH_CEILING,
        },
        "model": {
            "total_mass_kg": geometry.total_mass,
            "gravity_m_per_s2": geometry.gravity.tolist(),
            "contact_cone": (
                "E017 optimistic bilateral pre/post capsule-footprint hull with "
                "unilateral normal impulses and square friction pyramid"
            ),
            "optimistic_relaxations": [
                "both feet admitted at every transition regardless of recorded final-substep support",
                "both pre- and post-transition footprints admitted",
                "capsule radii over-approximated by endpoint squares before convex hulling",
                "no actuator, force, impulse-rate, or complementarity limits",
            ],
        },
        "projection_definition": {
            "objective": (
                "minimize sum_j abs(target_j - contact_cone_impulse_j) / component_scale_j"
            ),
            "solver": "scipy.optimize.linprog(method='highs-ipm')",
            "residual_sign": "target minus projected contact-cone impulse",
            "normalization_rationale": (
                "linear axes use body-weight impulse over the four-transition window; "
                "angular axes use the same impulse times the frozen 0.3 metre prior lever arm"
            ),
            "nonuniqueness_check": (
                "componentwise min/max residual over the complete weighted-L1 optimal set "
                "within the registered normalized-cost tolerance"
            ),
        },
        "window_start_transitions": window_starts.tolist(),
        "window_end_transitions_inclusive": window_ends.tolist(),
        "fixed_intervals_by_window_start": {
            label: list(value) for label, value in start_intervals.items()
        },
        "fixed_intervals_by_window_end": {
            label: list(value) for label, value in end_intervals.items()
        },
        "projected_target_comparisons": comparisons,
        "discarded_residual": summaries,
        "discarded_residual_vs_direct_assistance": assistance,
        "solver": {
            "max_constraint_or_objective_error": max_constraint_error,
            "candidate_point_count_min": min(
                int(item["candidate_points"])
                for label in FULL_LABELS
                for item in solver_diagnostics[label]
            ),
            "candidate_point_count_max": max(
                int(item["candidate_points"])
                for label in FULL_LABELS
                for item in solver_diagnostics[label]
            ),
            "zero_cost_window_count": {
                label: int(np.count_nonzero(costs[label] <= ZERO_COST_TOLERANCE))
                for label in FULL_LABELS
            },
            "max_optimal_set_normalized_component_width": max_ambiguity_normalized_width,
        },
        "gates": {
            "solver_constraints_pass": solver_pass,
            "all_projected_signal_interval_separation_pass": separation_pass,
            "weighted_l1_optimal_set_ambiguity_pass": ambiguity_pass,
        },
        "verdict": verdict,
        "claim_scope": {
            "established": [
                "The solver-selected weighted-L1 contact-cone projection and discarded residual are measured for both successful replicas over all 125 registered windows.",
                "Discarded residuals are reported by physical axis and E014 interval.",
                "The complete weighted-L1 optimal-set residual ambiguity is bounded componentwise rather than hidden behind one solver solution.",
            ],
            "not_established": [
                "The projected impulse is reachable by the G1 actuators or a joint policy.",
                "The optimistic contact cone matches the actual support schedule or substep contact geometry.",
                "A projected target preserves the successful trajectory when direct torso assistance is removed.",
                "A learned torso wrench belongs in the intended controller.",
                "A compatible PPO trajectory passes or fails this projection analysis.",
            ],
        },
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "arrays": str(arrays_path),
                "plot": str(plot_path),
                "verdict": verdict,
            }
        )
    )


if __name__ == "__main__":
    main()
