"""Audit friction margin for E018 targets under recorded G1 foot support.

The analysis is offline and reuses E017's model geometry, kinematics, capsule
footprints, and categorical HiGHS-IPM friction-pyramid solver.  It changes only
the contact candidate filter: each transition admits the left and/or right foot
reported active by the saved post-transition support signature.  It estimates
the minimum square-pyramid friction coefficient for the E018 projected and
unprojected successful targets and checks saved realized-contact/E002 controls
at the pinned model coefficient.  It never constructs an environment, steps
dynamics, evaluates a policy, or trains a controller.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools import analyze_g1_contact_impulse_projection as projection
from tools import analyze_g1_contact_impulse_realizability as realizability


PROTOCOL_VERSION = "g1-contact-friction-margin-v1"
FULL_LABELS = realizability.FULL_LABELS
RUN_LABELS = realizability.RUN_LABELS
TARGET_LABELS = (
    "full-a-unprojected",
    "full-b-unprojected",
    "full-a-projected",
    "full-b-projected",
)
CONTROL_LABELS = (
    "full-a-realized-contact",
    "full-b-realized-contact",
    "e002-unassisted",
)
SOLVER_TOLERANCE = 1e-6


def load_projection_arrays(
    path: Path, expected_sha256: str
) -> tuple[Path, str, dict[str, np.ndarray]]:
    """Load and validate the immutable E018 projected-target artifact."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing projection artifact: {resolved}")
    actual_sha256 = realizability.sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"projection artifact SHA-256 mismatch: {actual_sha256} != "
            f"{expected_sha256}"
        )
    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    expected_shapes = {
        "window_start_transitions": (125,),
        "window_end_transitions_inclusive": (125,),
        "component_scales": (6,),
        "contact_equivalent_full_a": (125, 6),
        "contact_equivalent_full_b": (125, 6),
        "e002_unassisted": (125, 6),
        "projected_full_a": (125, 6),
        "projected_full_b": (125, 6),
        "discarded_residual_full_a": (125, 6),
        "discarded_residual_full_b": (125, 6),
        "discarded_residual_bounds_full_a": (125, 6, 2),
        "discarded_residual_bounds_full_b": (125, 6, 2),
        "realized_contact_full_a": (125, 6),
        "realized_contact_full_b": (125, 6),
        "direct_assistance_full_a": (125, 6),
        "direct_assistance_full_b": (125, 6),
    }
    if set(arrays) != set(expected_shapes):
        raise ValueError("projection artifact array set is incompatible")
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if value.shape != expected_shape:
            raise ValueError(
                f"projection array {name} shape {value.shape} != "
                f"{expected_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"projection array {name} must be finite")
    np.testing.assert_array_equal(
        arrays["window_start_transitions"], np.arange(1, 126)
    )
    np.testing.assert_array_equal(
        arrays["window_end_transitions_inclusive"], np.arange(4, 129)
    )
    for suffix in ("a", "b"):
        np.testing.assert_allclose(
            arrays[f"contact_equivalent_full_{suffix}"],
            arrays[f"projected_full_{suffix}"]
            + arrays[f"discarded_residual_full_{suffix}"],
            rtol=0.0,
            atol=2e-13,
        )
    return resolved, actual_sha256, arrays


def recorded_support_contact_matrix(
    geometry: realizability.ModelGeometry,
    run: realizability.RunData,
    start: int,
    window_transitions: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build E017's map after filtering each transition by saved support."""
    blocks = []
    admitted_side_steps = np.zeros(2, dtype=np.int64)
    for transition in range(start, start + window_transitions):
        support = np.asarray(run.arrays["foot_support"][transition], dtype=bool)
        if support.shape != (2,) or not np.any(support):
            raise ValueError(
                f"{run.label} transition {transition} has no recorded support"
            )
        admitted_side_steps += support.astype(np.int64)
        for state_index in (transition, transition + 1):
            system_com, _, geom_positions, geom_rotations = (
                realizability.kinematics(
                    geometry, run.arrays["qpos"][state_index]
                )
            )
            for side_index, geom_ids in enumerate(geometry.foot_geom_ids):
                if not support[side_index]:
                    continue
                footprint = realizability.foot_footprint_vertices(
                    geometry, geom_positions, geom_rotations, geom_ids
                )
                for xy in footprint:
                    contact_point = np.asarray((xy[0], xy[1], 0.0))
                    moment_arm = contact_point - system_com
                    blocks.append(
                        np.concatenate(
                            (
                                np.eye(3),
                                realizability._cross_matrix(moment_arm),
                            )
                        )
                    )
    if not blocks:
        raise ValueError("recorded-support contact matrix has no candidates")
    matrix = np.concatenate(blocks, axis=1)
    return matrix, {
        "candidate_points": int(matrix.shape[1] // 3),
        "left_supported_transition_count": int(admitted_side_steps[0]),
        "right_supported_transition_count": int(admitted_side_steps[1]),
        "double_support_transition_count": int(
            np.count_nonzero(
                np.all(
                    run.arrays["foot_support"][
                        start : start + window_transitions
                    ],
                    axis=1,
                )
            )
        ),
    }


def minimum_friction(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    model_friction: float,
    maximum_friction: float,
    tolerance: float,
) -> dict[str, object]:
    """Bracket the minimum feasible coefficient with categorical LP solves."""
    solve_count = 0
    maximum_constraint_error = 0.0

    def feasible(coefficient: float) -> bool:
        nonlocal solve_count, maximum_constraint_error
        passed, diagnostics = realizability.friction_pyramid_feasible(
            matrix, target, coefficient
        )
        solve_count += 1
        if passed:
            maximum_constraint_error = max(
                maximum_constraint_error,
                float(diagnostics["max_equality_residual"]),
                float(diagnostics["max_inequality_violation"]),
            )
        return passed

    feasible_at_zero = feasible(0.0)
    if feasible_at_zero:
        return {
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "censored_above_maximum": False,
            "feasible_at_model_friction": True,
            "solve_count": solve_count,
            "max_constraint_error": maximum_constraint_error,
        }

    feasible_at_model = feasible(model_friction)
    if feasible_at_model:
        lower = 0.0
        upper = model_friction
    else:
        feasible_at_maximum = feasible(maximum_friction)
        if not feasible_at_maximum:
            return {
                "lower_bound": maximum_friction,
                "upper_bound": None,
                "censored_above_maximum": True,
                "feasible_at_model_friction": False,
                "solve_count": solve_count,
                "max_constraint_error": maximum_constraint_error,
            }
        lower = model_friction
        upper = maximum_friction

    while upper - lower > tolerance:
        midpoint = 0.5 * (lower + upper)
        if feasible(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "censored_above_maximum": False,
        "feasible_at_model_friction": feasible_at_model,
        "solve_count": solve_count,
        "max_constraint_error": maximum_constraint_error,
    }


def conservative_quantile(
    upper_bounds: np.ndarray, censored: np.ndarray, quantile: float
) -> float | None:
    """Return an order-statistic upper quantile, or None if it is censored."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    order = np.argsort(
        np.where(censored, np.inf, np.asarray(upper_bounds, dtype=np.float64))
    )
    index = max(0, math.ceil(quantile * len(order)) - 1)
    selected = int(order[index])
    return None if bool(censored[selected]) else float(upper_bounds[selected])


def margin_summary(
    results: list[dict[str, object]],
    selection: np.ndarray,
    *,
    model_friction: float,
    margin_friction: float,
    modest_friction: float,
    maximum_friction: float,
) -> dict[str, object]:
    """Summarize friction brackets for one registered window selection."""
    indices = np.flatnonzero(selection)
    lower = np.asarray([results[index]["lower_bound"] for index in indices])
    upper = np.asarray(
        [
            np.nan
            if results[index]["upper_bound"] is None
            else results[index]["upper_bound"]
            for index in indices
        ]
    )
    censored = np.asarray(
        [results[index]["censored_above_maximum"] for index in indices],
        dtype=bool,
    )
    at_model = np.asarray(
        [results[index]["feasible_at_model_friction"] for index in indices],
        dtype=bool,
    )

    def definitely_at_or_below(coefficient: float) -> np.ndarray:
        return (~censored) & (upper <= coefficient)

    finite_upper = upper[~censored]
    return {
        "window_count": int(len(indices)),
        "censored_above_maximum_count": int(np.count_nonzero(censored)),
        "censored_above_maximum_fraction": float(np.mean(censored)),
        "feasible_at_model_friction_count": int(np.count_nonzero(at_model)),
        "feasible_at_model_friction_fraction": float(np.mean(at_model)),
        "definitely_feasible_at_margin_fraction": float(
            np.mean(definitely_at_or_below(margin_friction))
        ),
        "definitely_feasible_at_modest_friction_fraction": float(
            np.mean(definitely_at_or_below(modest_friction))
        ),
        "definitely_feasible_at_maximum_fraction": float(
            np.mean(definitely_at_or_below(maximum_friction))
        ),
        "minimum_friction_upper_bound_p50": conservative_quantile(
            upper, censored, 0.50
        ),
        "minimum_friction_upper_bound_p95": conservative_quantile(
            upper, censored, 0.95
        ),
        "minimum_friction_upper_bound_max": (
            None if np.any(censored) else float(np.max(finite_upper))
        ),
        "minimum_friction_lower_bound_max": float(np.max(lower)),
        "bisection_bracket_width_max": (
            float(np.max(upper[~censored] - lower[~censored]))
            if np.any(~censored)
            else None
        ),
        "model_friction": model_friction,
        "margin_friction": margin_friction,
        "modest_friction": modest_friction,
        "maximum_friction": maximum_friction,
    }


def control_summary(
    feasible: np.ndarray,
    window_starts: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    """Summarize exact model-friction feasibility of one physical control."""
    output = {
        "overall": {
            "window_count": int(len(feasible)),
            "feasible_count": int(np.count_nonzero(feasible)),
            "feasible_fraction": float(np.mean(feasible)),
            "infeasible_window_start_intervals": realizability.contiguous_intervals(
                window_starts[~feasible]
            ),
        }
    }
    for label, selection in masks.items():
        if label == "overall":
            continue
        output[label] = {
            "window_count": int(np.count_nonzero(selection)),
            "feasible_count": int(np.count_nonzero(feasible[selection])),
            "feasible_fraction": float(np.mean(feasible[selection])),
        }
    return output


def write_plot(
    path: Path,
    window_starts: np.ndarray,
    target_results: dict[str, list[dict[str, object]]],
    target_summaries: dict[str, dict[str, object]],
    control_feasibility: dict[str, np.ndarray],
    maximum_friction: float,
    model_friction: float,
    margin_friction: float,
    modest_friction: float,
    verdict: str,
) -> None:
    """Write one inspectable friction-margin summary plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "full-a-unprojected": "#e45756",
        "full-b-unprojected": "#ff9d98",
        "full-a-projected": "#4c78a8",
        "full-b-projected": "#72b7b2",
    }
    styles = {
        "full-a-unprojected": "-",
        "full-b-unprojected": "--",
        "full-a-projected": "-",
        "full-b-projected": "--",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    plot_values = {}
    for label, results in target_results.items():
        plot_values[label] = np.asarray(
            [
                maximum_friction + 0.15
                if item["upper_bound"] is None
                else item["upper_bound"]
                for item in results
            ]
        )
        ordered = np.sort(plot_values[label])
        axes[0, 0].step(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            where="post",
            color=colors[label],
            linestyle=styles[label],
            label=label,
        )
        axes[1, 0].plot(
            window_starts,
            plot_values[label],
            color=colors[label],
            linestyle=styles[label],
            label=label,
            alpha=0.9,
        )
    for axis in (axes[0, 0], axes[1, 0]):
        axis.axvline(model_friction, color="black", linestyle="-", linewidth=1)
        axis.axvline(margin_friction, color="black", linestyle=":", linewidth=1)
        axis.axvline(modest_friction, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
    axes[0, 0].set(
        title="Minimum-friction upper-bound CDF",
        xlabel="square-pyramid friction coefficient",
        ylabel="window fraction",
        xlim=(0.0, maximum_friction + 0.25),
        ylim=(0.0, 1.02),
    )
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(
        title="Minimum-friction upper bound by window",
        xlabel="window start transition",
        ylabel="friction coefficient (max+ means censored)",
        ylim=(0.0, maximum_friction + 0.25),
    )

    thresholds = np.linspace(0.0, modest_friction, 80)
    for label, values in plot_values.items():
        fractions = [float(np.mean(values <= value)) for value in thresholds]
        axes[0, 1].plot(
            thresholds,
            fractions,
            color=colors[label],
            linestyle=styles[label],
            label=label,
        )
    axes[0, 1].axvline(model_friction, color="black", linewidth=1)
    axes[0, 1].axvline(margin_friction, color="black", linestyle=":", linewidth=1)
    axes[0, 1].set(
        title="Conservative feasible fraction versus friction",
        xlabel="friction coefficient",
        ylabel="definitely feasible window fraction",
        ylim=(0.0, 1.02),
    )
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    labels = list(TARGET_LABELS) + list(CONTROL_LABELS)
    fractions = [
        target_summaries[label]["overall"][
            "feasible_at_model_friction_fraction"
        ]
        if label in target_summaries
        else float(np.mean(control_feasibility[label]))
        for label in labels
    ]
    bar_colors = [colors.get(label, "#54a24b") for label in labels]
    axes[1, 1].barh(labels, fractions, color=bar_colors)
    axes[1, 1].axvline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(
        title=f"Exact feasibility at model friction {model_friction:g}",
        xlabel="window fraction",
        xlim=(0.0, 1.02),
    )
    axes[1, 1].grid(axis="x", alpha=0.25)
    for index, fraction in enumerate(fractions):
        axes[1, 1].text(
            min(fraction + 0.015, 0.96), index, f"{fraction:.3f}", va="center"
        )

    figure.suptitle(f"G1 recorded-support friction margin: {verdict}", fontsize=15)
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
    parser.add_argument("--projection-npz", type=Path, required=True)
    parser.add_argument("--projection-sha256", required=True)
    parser.add_argument("--failure-window-json", type=Path, required=True)
    parser.add_argument("--failure-window-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-reference-transitions", type=int, default=271)
    parser.add_argument("--window-transitions", type=int, default=4)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--model-friction", type=float, default=1.0)
    parser.add_argument("--margin-friction", type=float, default=0.8)
    parser.add_argument("--modest-friction", type=float, default=1.25)
    parser.add_argument("--maximum-friction", type=float, default=4.0)
    parser.add_argument("--friction-tolerance", type=float, default=0.005)
    parser.add_argument("--control-feasibility-floor", type=float, default=0.95)
    parser.add_argument("--projected-feasibility-floor", type=float, default=0.95)
    parser.add_argument("--teacher-feasibility-ceiling", type=float, default=0.80)
    parser.add_argument("--current-reference-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_reference_transitions < 1 or args.window_transitions < 1:
        raise ValueError("transition counts must be positive")
    positive_values = {
        "control dt": args.control_dt,
        "model friction": args.model_friction,
        "margin friction": args.margin_friction,
        "modest friction": args.modest_friction,
        "maximum friction": args.maximum_friction,
        "friction tolerance": args.friction_tolerance,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not (
        args.margin_friction
        < args.model_friction
        < args.modest_friction
        < args.maximum_friction
    ):
        raise ValueError("friction thresholds must be strictly ordered")
    for name, value in (
        ("control feasibility floor", args.control_feasibility_floor),
        ("projected feasibility floor", args.projected_feasibility_floor),
        ("teacher feasibility ceiling", args.teacher_feasibility_ceiling),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie strictly between zero and one")

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
    projection_path, projection_sha256, arrays = load_projection_arrays(
        args.projection_npz, args.projection_sha256
    )
    geometry = realizability.load_model(
        args.model_path, args.model_sha256, args.model_friction
    )

    window_starts = np.asarray(arrays["window_start_transitions"], dtype=np.int64)
    window_ends = np.asarray(
        arrays["window_end_transitions_inclusive"], dtype=np.int64
    )
    e002_onset = int(
        failure_windows["runs"]["e002"]["sustained_50pct_onset_transition"]
    )
    if int(window_ends[-1]) != e002_onset - 1:
        raise ValueError("projection windows do not end before the E002 onset")

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
    masks = projection.interval_masks(window_starts, start_intervals)
    coverage = np.sum(np.stack(list(masks.values())[1:]), axis=0)
    if not np.all(coverage == 1):
        raise ValueError("fixed E014 intervals must partition all windows")

    recomputed_contact_equivalent = {
        label: realizability.contact_equivalent_impulses(
            run, geometry, args.window_transitions, args.control_dt
        )[window_starts]
        for label, run in runs.items()
    }
    for label, suffix in (("full-a", "a"), ("full-b", "b")):
        np.testing.assert_allclose(
            recomputed_contact_equivalent[label],
            arrays[f"contact_equivalent_full_{suffix}"],
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        recomputed_contact_equivalent["e002"],
        arrays["e002_unassisted"],
        rtol=0.0,
        atol=1e-12,
    )

    matrices: dict[str, list[np.ndarray]] = {}
    matrix_diagnostics: dict[str, list[dict[str, int]]] = {}
    for label, run in runs.items():
        rows = [
            recorded_support_contact_matrix(
                geometry, run, int(start), args.window_transitions
            )
            for start in window_starts
        ]
        matrices[label] = [row[0] for row in rows]
        matrix_diagnostics[label] = [row[1] for row in rows]

    targets = {
        "full-a-unprojected": arrays["contact_equivalent_full_a"],
        "full-b-unprojected": arrays["contact_equivalent_full_b"],
        "full-a-projected": arrays["projected_full_a"],
        "full-b-projected": arrays["projected_full_b"],
    }
    target_results: dict[str, list[dict[str, object]]] = {}
    for label, target in targets.items():
        run_label = label[:6]
        target_results[label] = [
            minimum_friction(
                matrices[run_label][row],
                target[row],
                model_friction=args.model_friction,
                maximum_friction=args.maximum_friction,
                tolerance=args.friction_tolerance,
            )
            for row in range(len(window_starts))
        ]

    controls = {
        "full-a-realized-contact": arrays["realized_contact_full_a"],
        "full-b-realized-contact": arrays["realized_contact_full_b"],
        "e002-unassisted": arrays["e002_unassisted"],
    }
    control_feasibility = {}
    control_solver_diagnostics: dict[str, list[dict[str, object]]] = {}
    for label, target in controls.items():
        run_label = "e002" if label.startswith("e002") else label[:6]
        rows = [
            realizability.friction_pyramid_feasible(
                matrices[run_label][row], target[row], args.model_friction
            )
            for row in range(len(window_starts))
        ]
        control_feasibility[label] = np.asarray(
            [item[0] for item in rows], dtype=bool
        )
        control_solver_diagnostics[label] = [item[1] for item in rows]

    target_summaries = {
        label: {
            interval: margin_summary(
                results,
                selection,
                model_friction=args.model_friction,
                margin_friction=args.margin_friction,
                modest_friction=args.modest_friction,
                maximum_friction=args.maximum_friction,
            )
            for interval, selection in masks.items()
        }
        for label, results in target_results.items()
    }
    control_summaries = {
        label: control_summary(feasible, window_starts, masks)
        for label, feasible in control_feasibility.items()
    }

    control_pass = all(
        control_summaries[label]["overall"]["feasible_fraction"]
        >= args.control_feasibility_floor
        for label in CONTROL_LABELS
    )
    projected_model_pass = all(
        target_summaries[label]["overall"][
            "feasible_at_model_friction_fraction"
        ]
        >= args.projected_feasibility_floor
        for label in ("full-a-projected", "full-b-projected")
    )
    projected_margin_pass = all(
        (
            target_summaries[label]["overall"][
                "minimum_friction_upper_bound_p95"
            ]
            is not None
            and target_summaries[label]["overall"][
                "minimum_friction_upper_bound_p95"
            ]
            <= args.margin_friction
        )
        for label in ("full-a-projected", "full-b-projected")
    )
    projected_modest_pass = all(
        target_summaries[label]["overall"][
            "definitely_feasible_at_modest_friction_fraction"
        ]
        >= args.projected_feasibility_floor
        for label in ("full-a-projected", "full-b-projected")
    )
    teacher_model_infeasible = all(
        target_summaries[label]["overall"][
            "feasible_at_model_friction_fraction"
        ]
        < args.teacher_feasibility_ceiling
        for label in ("full-a-unprojected", "full-b-unprojected")
    )
    teacher_modest_pass = all(
        target_summaries[label]["overall"][
            "definitely_feasible_at_modest_friction_fraction"
        ]
        >= args.projected_feasibility_floor
        for label in ("full-a-unprojected", "full-b-unprojected")
    )

    max_solver_error = max(
        [
            float(item["max_constraint_error"])
            for results in target_results.values()
            for item in results
        ]
        + [
            max(
                float(item["max_equality_residual"]),
                float(item["max_inequality_violation"]),
            )
            for rows in control_solver_diagnostics.values()
            for item in rows
            if item["status"] == 0
        ]
    )
    solver_pass = max_solver_error <= SOLVER_TOLERANCE

    if not solver_pass:
        verdict = "invalid-friction-margin-solve"
    elif not control_pass:
        verdict = "recorded-support-proxy-invalid"
    elif projected_model_pass and projected_margin_pass:
        verdict = "projected-target-has-recorded-support-friction-margin"
    elif projected_model_pass:
        verdict = "projected-target-feasible-at-model-friction-with-thin-margin"
    elif projected_modest_pass:
        verdict = "projected-target-requires-higher-friction"
    else:
        verdict = "projected-target-not-rescued-by-modest-friction"

    if not control_pass:
        teacher_interpretation = "unchecked-because-recorded-support-proxy-failed"
    elif teacher_modest_pass:
        teacher_interpretation = "modest-higher-friction-could-rescue-saved-teacher"
    else:
        teacher_interpretation = (
            "saved-teacher-needs-more-than-modest-friction-or-support-change"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "contact_friction_margin.json"
    arrays_path = output_dir / "contact_friction_margin.npz"
    plot_path = output_dir / "contact_friction_margin.png"

    npz_values: dict[str, np.ndarray] = {
        "window_start_transitions": window_starts,
        "window_end_transitions_inclusive": window_ends,
    }
    for label, results in target_results.items():
        key = label.replace("-", "_")
        npz_values[f"{key}_friction_lower_bound"] = np.asarray(
            [item["lower_bound"] for item in results]
        )
        npz_values[f"{key}_friction_upper_bound"] = np.asarray(
            [
                np.nan if item["upper_bound"] is None else item["upper_bound"]
                for item in results
            ]
        )
        npz_values[f"{key}_censored_above_maximum"] = np.asarray(
            [item["censored_above_maximum"] for item in results], dtype=bool
        )
        npz_values[f"{key}_feasible_at_model_friction"] = np.asarray(
            [item["feasible_at_model_friction"] for item in results], dtype=bool
        )
    for label, feasible in control_feasibility.items():
        npz_values[f"{label.replace('-', '_')}_feasible_at_model_friction"] = (
            feasible
        )
    np.savez_compressed(arrays_path, **npz_values)
    write_plot(
        plot_path,
        window_starts,
        target_results,
        target_summaries,
        control_feasibility,
        args.maximum_friction,
        args.model_friction,
        args.margin_friction,
        args.modest_friction,
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
            "projection": {
                "path": str(projection_path),
                "sha256": projection_sha256,
            },
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
            "model_friction": args.model_friction,
            "margin_friction": args.margin_friction,
            "modest_friction": args.modest_friction,
            "maximum_friction": args.maximum_friction,
            "friction_tolerance": args.friction_tolerance,
            "control_feasibility_floor": args.control_feasibility_floor,
            "projected_feasibility_floor": args.projected_feasibility_floor,
            "teacher_feasibility_ceiling": args.teacher_feasibility_ceiling,
            "solver_tolerance": SOLVER_TOLERANCE,
        },
        "contact_candidate_definition": {
            "support_source": (
                "saved post-transition threshold-free grouped active-contact "
                "signature in left/right order"
            ),
            "foot_filter": (
                "for each transition admit only sides whose saved support bit is true"
            ),
            "preserved_e017_relaxations": [
                "both pre- and post-transition footprints admitted for each supported side",
                "capsule radii over-approximated by endpoint squares before convex hulling",
                "no actuator, force, impulse-rate, or complementarity limits",
            ],
            "removed_e017_relaxation": (
                "unsupported feet are no longer admitted at every transition"
            ),
            "limitation": (
                "support is sampled after each control transition rather than at every physics substep"
            ),
        },
        "minimum_friction_definition": {
            "solver": "E017 scipy.optimize.linprog(method='highs-ipm')",
            "cone": "unilateral normal impulse and square tangential friction pyramid",
            "search": (
                "categorical feasibility at zero/model/maximum followed by bisection of the enclosing interval"
            ),
            "reported_value": (
                "conservative feasible upper bound; above-maximum cases are right-censored"
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
        "matrix_diagnostics": {
            label: {
                "candidate_point_count_min": min(
                    item["candidate_points"] for item in rows
                ),
                "candidate_point_count_max": max(
                    item["candidate_points"] for item in rows
                ),
                "left_supported_transition_count_total": sum(
                    item["left_supported_transition_count"] for item in rows
                ),
                "right_supported_transition_count_total": sum(
                    item["right_supported_transition_count"] for item in rows
                ),
                "double_support_transition_count_total": sum(
                    item["double_support_transition_count"] for item in rows
                ),
            }
            for label, rows in matrix_diagnostics.items()
        },
        "target_friction_margin": target_summaries,
        "model_friction_controls": control_summaries,
        "solver": {
            "total_minimum_friction_solve_count": int(
                sum(
                    int(item["solve_count"])
                    for results in target_results.values()
                    for item in results
                )
            ),
            "fixed_control_solve_count": int(
                len(window_starts) * len(CONTROL_LABELS)
            ),
            "max_constraint_error": max_solver_error,
        },
        "gates": {
            "categorical_solver_constraints_pass": solver_pass,
            "recorded_support_physical_controls_pass": control_pass,
            "projected_target_model_friction_pass": projected_model_pass,
            "projected_target_p95_margin_pass": projected_margin_pass,
            "projected_target_modest_friction_pass": projected_modest_pass,
            "unprojected_teacher_model_friction_infeasible": teacher_model_infeasible,
            "unprojected_teacher_modest_friction_pass": teacher_modest_pass,
        },
        "teacher_more_friction_interpretation": teacher_interpretation,
        "verdict": verdict,
        "claim_scope": {
            "established": [
                "Minimum-friction brackets are measured for both E018 projected and unprojected successful targets under saved transition-level support.",
                "Saved E002 and successful realized-contact estimates are checked as physical controls at the pinned model friction.",
                "The model-friction, margin-friction, modest-increase, and maximum-friction decisions use thresholds fixed before execution.",
            ],
            "not_established": [
                "Transition-level support equals the full physics-substep contact schedule.",
                "A coefficient above one is available on deployment hardware or should be used in training.",
                "Contact-cone feasibility is sufficient for actuator or joint-policy reachability.",
                "A projected target preserves the successful trajectory without direct torso assistance.",
                "A learned torso wrench belongs in the intended controller.",
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
