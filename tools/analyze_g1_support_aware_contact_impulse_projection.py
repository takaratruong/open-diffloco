"""Project successful G1 impulses into the recorded-support foot cone.

This offline analysis is the support-aware follow-up to E018 and E000.  It
reuses E018's immutable targets, weighted-L1 objective, normalization, and
HiGHS-IPM solver, while replacing only the bilateral contact-candidate set with
E000's saved post-transition active-foot filter.  It independently checks the
projected targets and successful physical controls in that same cone.  It never
constructs an environment, steps dynamics, evaluates a policy, or trains a
controller.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools import analyze_g1_contact_friction_margin as friction_margin
from tools import analyze_g1_contact_impulse_projection as projection
from tools import analyze_g1_contact_impulse_realizability as realizability


PROTOCOL_VERSION = "g1-support-aware-contact-impulse-projection-v1"
FULL_LABELS = realizability.FULL_LABELS
RUN_LABELS = realizability.RUN_LABELS
CONTROL_LABELS = (
    "full-a-realized-contact",
    "full-b-realized-contact",
    "e002-unassisted",
)
SOLVER_TOLERANCE = 1e-6
NESTED_COST_TOLERANCE = 1e-7


def _verify_module_hash(module: object, expected_sha256: str) -> dict[str, str]:
    """Verify one imported local dependency and return its immutable identity."""
    path = Path(str(getattr(module, "__file__"))).resolve()
    actual_sha256 = realizability.sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"module SHA-256 mismatch for {path}: {actual_sha256} != "
            f"{expected_sha256}"
        )
    return {"path": str(path), "sha256": actual_sha256}


def _fixed_intervals(
    failure_windows: dict[str, object],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    window_transitions: int,
    e002_onset: int,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]], dict[str, np.ndarray]]:
    """Reconstruct and validate E014's immutable interval partition."""
    families = failure_windows["families"]
    vertical_interval = tuple(
        int(value)
        for value in families["vertical-only"]["consensus_pre_onset_interval"]
    )
    novertical_interval = tuple(
        int(value)
        for value in families["no-vertical"]["consensus_pre_onset_interval"]
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
            lower - window_transitions + 1,
            upper - window_transitions + 1,
        )
        for label, (lower, upper) in end_intervals.items()
    }
    masks = projection.interval_masks(window_starts, start_intervals)
    coverage = np.sum(np.stack(list(masks.values())[1:]), axis=0)
    if not np.all(coverage == 1):
        raise ValueError("fixed E014 intervals must partition all windows")
    return end_intervals, start_intervals, masks


def _bilateral_comparison(
    support_projected: dict[str, np.ndarray],
    support_costs: dict[str, np.ndarray],
    bilateral_arrays: dict[str, np.ndarray],
    component_scales: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[dict[str, object], float]:
    """Compare the support-aware optimum with E018's bilateral optimum."""
    output: dict[str, object] = {}
    minimum_increment = np.inf
    for label, suffix in (("full-a", "a"), ("full-b", "b")):
        bilateral_projected = bilateral_arrays[f"projected_full_{suffix}"]
        bilateral_residual = bilateral_arrays[f"discarded_residual_full_{suffix}"]
        bilateral_cost = np.sum(
            np.abs(bilateral_residual) / component_scales, axis=1
        )
        increment = support_costs[label] - bilateral_cost
        minimum_increment = min(minimum_increment, float(np.min(increment)))
        correction = support_projected[label] - bilateral_projected
        output[label] = {}
        for interval, selection in masks.items():
            selected_increment = increment[selection]
            selected_correction = correction[selection]
            correction_norm = np.linalg.norm(selected_correction, axis=1)
            output[label][interval] = {
                "window_count": int(np.count_nonzero(selection)),
                "support_weighted_l1_cost_mean": float(
                    np.mean(support_costs[label][selection])
                ),
                "support_weighted_l1_cost_p95": float(
                    np.quantile(support_costs[label][selection], 0.95)
                ),
                "bilateral_weighted_l1_cost_mean": float(
                    np.mean(bilateral_cost[selection])
                ),
                "incremental_weighted_l1_cost_mean": float(
                    np.mean(selected_increment)
                ),
                "incremental_weighted_l1_cost_p95": float(
                    np.quantile(selected_increment, 0.95)
                ),
                "incremental_weighted_l1_cost_min": float(
                    np.min(selected_increment)
                ),
                "unchanged_from_bilateral_count": int(
                    np.count_nonzero(correction_norm[selection] <= 1e-10)
                ),
                "linear_correction_rms_newton_seconds": realizability.rms_norm(
                    selected_correction[:, :3]
                ),
                "linear_correction_p95_newton_seconds": float(
                    np.quantile(
                        np.linalg.norm(selected_correction[:, :3], axis=1), 0.95
                    )
                ),
                "angular_correction_rms_newton_metre_seconds": (
                    realizability.rms_norm(selected_correction[:, 3:])
                ),
                "angular_correction_p95_newton_metre_seconds": float(
                    np.quantile(
                        np.linalg.norm(selected_correction[:, 3:], axis=1), 0.95
                    )
                ),
            }
    return output, float(minimum_increment)


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
    parser.add_argument("--bilateral-projection-npz", type=Path, required=True)
    parser.add_argument("--bilateral-projection-sha256", required=True)
    parser.add_argument("--failure-window-json", type=Path, required=True)
    parser.add_argument("--failure-window-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--projection-module-sha256", required=True)
    parser.add_argument("--friction-margin-module-sha256", required=True)
    parser.add_argument("--realizability-module-sha256", required=True)
    parser.add_argument("--expected-reference-transitions", type=int, default=271)
    parser.add_argument("--window-transitions", type=int, default=4)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--friction-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--angular-normalization-lever-arm", type=float, default=0.3
    )
    parser.add_argument("--control-feasibility-floor", type=float, default=0.95)
    parser.add_argument("--projected-feasibility-floor", type=float, default=1.0)
    parser.add_argument("--current-reference-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    script_sha256 = realizability.sha256_file(script_path)
    if script_sha256 != args.expected_script_sha256:
        raise ValueError(
            f"script SHA-256 mismatch: {script_sha256} != "
            f"{args.expected_script_sha256}"
        )
    dependency_modules = {
        "projection": _verify_module_hash(
            projection, args.projection_module_sha256
        ),
        "friction_margin": _verify_module_hash(
            friction_margin, args.friction_margin_module_sha256
        ),
        "realizability": _verify_module_hash(
            realizability, args.realizability_module_sha256
        ),
    }
    if args.expected_reference_transitions < 1 or args.window_transitions < 1:
        raise ValueError("transition counts must be positive")
    for name, value in (
        ("control dt", args.control_dt),
        ("friction coefficient", args.friction_coefficient),
        (
            "angular normalization lever arm",
            args.angular_normalization_lever_arm,
        ),
        ("control feasibility floor", args.control_feasibility_floor),
        ("projected feasibility floor", args.projected_feasibility_floor),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.control_feasibility_floor > 1.0:
        raise ValueError("control feasibility floor must not exceed one")
    if args.projected_feasibility_floor != 1.0:
        raise ValueError("support-aware projected feasibility floor must equal one")

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
    bilateral_path, bilateral_sha256, bilateral_arrays = (
        friction_margin.load_projection_arrays(
            args.bilateral_projection_npz,
            args.bilateral_projection_sha256,
        )
    )

    e002_onset = int(
        failure_windows["runs"]["e002"]["sustained_50pct_onset_transition"]
    )
    analysis_end = e002_onset - args.window_transitions
    window_starts = np.arange(
        projection.WINDOW_START_TRANSITION, analysis_end + 1, dtype=np.int64
    )
    window_ends = window_starts + args.window_transitions - 1
    if len(window_starts) != 125 or int(window_ends[-1]) != e002_onset - 1:
        raise ValueError("registered pre-onset analysis must contain 125 windows")
    np.testing.assert_array_equal(
        bilateral_arrays["window_start_transitions"], window_starts
    )
    np.testing.assert_array_equal(
        bilateral_arrays["window_end_transitions_inclusive"], window_ends
    )
    end_intervals, start_intervals, masks = _fixed_intervals(
        failure_windows,
        window_starts,
        window_ends,
        args.window_transitions,
        e002_onset,
    )

    recomputed_contact_equivalent = {
        label: realizability.contact_equivalent_impulses(
            run, geometry, args.window_transitions, args.control_dt
        )[window_starts]
        for label, run in runs.items()
    }
    for label, suffix in (("full-a", "a"), ("full-b", "b")):
        np.testing.assert_allclose(
            recomputed_contact_equivalent[label],
            bilateral_arrays[f"contact_equivalent_full_{suffix}"],
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        recomputed_contact_equivalent["e002"],
        bilateral_arrays["e002_unassisted"],
        rtol=0.0,
        atol=1e-12,
    )

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
        label: recomputed_contact_equivalent[label] - direct_assistance[label]
        for label in FULL_LABELS
    }
    for label, suffix in (("full-a", "a"), ("full-b", "b")):
        np.testing.assert_allclose(
            direct_assistance[label],
            bilateral_arrays[f"direct_assistance_full_{suffix}"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            realized_contact[label],
            bilateral_arrays[f"realized_contact_full_{suffix}"],
            rtol=0.0,
            atol=1e-12,
        )

    linear_scale = (
        geometry.total_mass
        * abs(float(geometry.gravity[2]))
        * args.window_transitions
        * args.control_dt
    )
    angular_scale = linear_scale * args.angular_normalization_lever_arm
    component_scales = np.asarray(
        (
            linear_scale,
            linear_scale,
            linear_scale,
            angular_scale,
            angular_scale,
            angular_scale,
        )
    )
    np.testing.assert_allclose(
        component_scales,
        bilateral_arrays["component_scales"],
        rtol=0.0,
        atol=1e-12,
    )

    matrices: dict[str, list[np.ndarray]] = {}
    matrix_diagnostics: dict[str, list[dict[str, int]]] = {}
    support_signatures: dict[str, np.ndarray] = {}
    for label, run in runs.items():
        rows = [
            friction_margin.recorded_support_contact_matrix(
                geometry, run, int(start), args.window_transitions
            )
            for start in window_starts
        ]
        matrices[label] = [row[0] for row in rows]
        matrix_diagnostics[label] = [row[1] for row in rows]
        support_signatures[label] = np.asarray(
            [
                run.arrays["foot_support"][
                    start : start + args.window_transitions
                ]
                for start in window_starts
            ],
            dtype=bool,
        )
        if support_signatures[label].shape != (
            len(window_starts),
            args.window_transitions,
            2,
        ):
            raise ValueError(f"unexpected support signature shape for {label}")

    projected: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    residual_bounds: dict[str, np.ndarray] = {}
    costs: dict[str, np.ndarray] = {}
    projection_diagnostics: dict[str, list[dict[str, object]]] = {}
    for label, suffix in (("full-a", "a"), ("full-b", "b")):
        target = bilateral_arrays[f"contact_equivalent_full_{suffix}"]
        rows = [
            projection.solve_projection(
                matrices[label][row],
                target[row],
                args.friction_coefficient,
                component_scales,
            )
            for row in range(len(window_starts))
        ]
        projected[label] = np.asarray([item["projected"] for item in rows])
        residuals[label] = np.asarray([item["residual"] for item in rows])
        residual_bounds[label] = np.asarray(
            [item["residual_bounds"] for item in rows]
        )
        costs[label] = np.asarray(
            [item["normalized_l1_cost"] for item in rows]
        )
        projection_diagnostics[label] = rows
        np.testing.assert_allclose(
            projected[label] + residuals[label], target, rtol=0.0, atol=2e-13
        )

    projected_feasibility: dict[str, np.ndarray] = {}
    projected_feasibility_diagnostics: dict[str, list[dict[str, object]]] = {}
    for label in FULL_LABELS:
        rows = [
            realizability.friction_pyramid_feasible(
                matrices[label][row],
                projected[label][row],
                args.friction_coefficient,
            )
            for row in range(len(window_starts))
        ]
        projected_feasibility[label] = np.asarray(
            [item[0] for item in rows], dtype=bool
        )
        projected_feasibility_diagnostics[label] = [item[1] for item in rows]

    controls = {
        "full-a-realized-contact": realized_contact["full-a"],
        "full-b-realized-contact": realized_contact["full-b"],
        "e002-unassisted": bilateral_arrays["e002_unassisted"],
    }
    control_feasibility: dict[str, np.ndarray] = {}
    control_diagnostics: dict[str, list[dict[str, object]]] = {}
    for label, target in controls.items():
        run_label = "e002" if label.startswith("e002") else label[:6]
        rows = [
            realizability.friction_pyramid_feasible(
                matrices[run_label][row],
                target[row],
                args.friction_coefficient,
            )
            for row in range(len(window_starts))
        ]
        control_feasibility[label] = np.asarray(
            [item[0] for item in rows], dtype=bool
        )
        control_diagnostics[label] = [item[1] for item in rows]

    projected_summaries = {
        label: friction_margin.control_summary(feasible, window_starts, masks)
        for label, feasible in projected_feasibility.items()
    }
    control_summaries = {
        label: friction_margin.control_summary(feasible, window_starts, masks)
        for label, feasible in control_feasibility.items()
    }
    comparisons = projection.distance_summary(
        projected["full-a"],
        projected["full-b"],
        bilateral_arrays["e002_unassisted"],
        masks,
    )
    residual_summaries = {
        label: projection.residual_summary(
            residuals[label], residual_bounds[label], component_scales, masks
        )
        for label in FULL_LABELS
    }
    assistance = {
        label: projection.assistance_comparison(
            residuals[label],
            direct_assistance[label],
            projected[label],
            realized_contact[label],
            masks,
        )
        for label in FULL_LABELS
    }
    bilateral_comparison, minimum_cost_increment = _bilateral_comparison(
        projected,
        costs,
        bilateral_arrays,
        component_scales,
        masks,
    )

    projection_error_keys = (
        "max_equality_residual",
        "max_inequality_violation",
        "max_residual_representation_error",
        "objective_error",
    )
    projection_max_error = max(
        float(item[key])
        for label in FULL_LABELS
        for item in projection_diagnostics[label]
        for key in projection_error_keys
    )
    categorical_diagnostic_rows = [
        item
        for rows in (
            list(projected_feasibility_diagnostics.values())
            + list(control_diagnostics.values())
        )
        for item in rows
        if item["status"] == 0
    ]
    categorical_max_error = max(
        max(
            float(item["max_equality_residual"]),
            float(item["max_inequality_violation"]),
        )
        for item in categorical_diagnostic_rows
    )
    max_solver_error = max(projection_max_error, categorical_max_error)
    max_ambiguity_normalized_width = max(
        float(
            np.max(
                (bounds[:, :, 1] - bounds[:, :, 0]) / component_scales
            )
        )
        for bounds in residual_bounds.values()
    )

    solver_pass = max_solver_error <= SOLVER_TOLERANCE
    nested_cost_pass = minimum_cost_increment >= -NESTED_COST_TOLERANCE
    controls_pass = all(
        control_summaries[label]["overall"]["feasible_fraction"]
        >= args.control_feasibility_floor
        for label in CONTROL_LABELS
    )
    projected_feasibility_pass = all(
        projected_summaries[label]["overall"]["feasible_fraction"]
        >= args.projected_feasibility_floor
        for label in FULL_LABELS
    )
    separation_pass = all(
        interval["separation_gate_pass"]
        for component in comparisons.values()
        for interval in component.values()
    )
    ambiguity_pass = (
        max_ambiguity_normalized_width
        <= projection.AMBIGUITY_NORMALIZED_WIDTH_CEILING
    )

    if not solver_pass or not nested_cost_pass or not projected_feasibility_pass:
        verdict = "invalid-support-aware-contact-projection"
    elif not controls_pass:
        verdict = "recorded-support-proxy-invalid"
    elif separation_pass and ambiguity_pass:
        verdict = "stable-support-aware-contact-projected-target"
    elif separation_pass:
        verdict = "support-aware-weighted-l1-target-is-ambiguous"
    else:
        verdict = "support-aware-projected-target-is-not-duplicate-stable"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "support_aware_contact_impulse_projection.json"
    arrays_path = output_dir / "support_aware_contact_impulse_projection.npz"
    plot_path = output_dir / "support_aware_contact_impulse_projection.png"
    np.savez_compressed(
        arrays_path,
        window_start_transitions=window_starts,
        window_end_transitions_inclusive=window_ends,
        component_scales=component_scales,
        contact_equivalent_full_a=bilateral_arrays["contact_equivalent_full_a"],
        contact_equivalent_full_b=bilateral_arrays["contact_equivalent_full_b"],
        e002_unassisted=bilateral_arrays["e002_unassisted"],
        bilateral_projected_full_a=bilateral_arrays["projected_full_a"],
        bilateral_projected_full_b=bilateral_arrays["projected_full_b"],
        support_projected_full_a=projected["full-a"],
        support_projected_full_b=projected["full-b"],
        support_discarded_residual_full_a=residuals["full-a"],
        support_discarded_residual_full_b=residuals["full-b"],
        support_discarded_residual_bounds_full_a=residual_bounds["full-a"],
        support_discarded_residual_bounds_full_b=residual_bounds["full-b"],
        support_projection_feasible_full_a=projected_feasibility["full-a"],
        support_projection_feasible_full_b=projected_feasibility["full-b"],
        realized_contact_full_a=realized_contact["full-a"],
        realized_contact_full_b=realized_contact["full-b"],
        direct_assistance_full_a=direct_assistance["full-a"],
        direct_assistance_full_b=direct_assistance["full-b"],
        control_feasible_full_a=control_feasibility["full-a-realized-contact"],
        control_feasible_full_b=control_feasibility["full-b-realized-contact"],
        control_feasible_e002=control_feasibility["e002-unassisted"],
        support_signature_full_a=support_signatures["full-a"],
        support_signature_full_b=support_signatures["full-b"],
        support_signature_e002=support_signatures["e002"],
        candidate_point_count_full_a=np.asarray(
            [row["candidate_points"] for row in matrix_diagnostics["full-a"]]
        ),
        candidate_point_count_full_b=np.asarray(
            [row["candidate_points"] for row in matrix_diagnostics["full-b"]]
        ),
        candidate_point_count_e002=np.asarray(
            [row["candidate_points"] for row in matrix_diagnostics["e002"]]
        ),
    )
    projection.write_plot(
        plot_path,
        window_starts,
        residuals,
        projected,
        bilateral_arrays["e002_unassisted"],
        component_scales,
        verdict,
    )

    output = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script_path),
        "script_sha256": script_sha256,
        "dependency_modules": dependency_modules,
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
            "bilateral_projection": {
                "path": str(bilateral_path),
                "sha256": bilateral_sha256,
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
            "friction_coefficient": args.friction_coefficient,
            "linear_normalization_body_weight_window_impulse_newton_seconds": (
                linear_scale
            ),
            "angular_normalization_lever_arm_metres": (
                args.angular_normalization_lever_arm
            ),
            "angular_normalization_newton_metre_seconds": angular_scale,
            "component_scales": component_scales.tolist(),
            "control_feasibility_floor": args.control_feasibility_floor,
            "projected_feasibility_floor": args.projected_feasibility_floor,
            "separation_ratio_gate": projection.SEPARATION_RATIO_GATE,
            "solver_constraint_tolerance": SOLVER_TOLERANCE,
            "zero_cost_tolerance": projection.ZERO_COST_TOLERANCE,
            "optimal_set_cost_tolerance": (
                projection.OPTIMAL_SET_COST_TOLERANCE
            ),
            "nested_cost_tolerance": NESTED_COST_TOLERANCE,
            "ambiguity_normalized_width_ceiling": (
                projection.AMBIGUITY_NORMALIZED_WIDTH_CEILING
            ),
        },
        "contact_candidate_definition": {
            "support_source": (
                "saved post-transition threshold-free grouped left/right "
                "active-contact bits"
            ),
            "foot_filter": (
                "for each transition admit only sides whose saved support bit "
                "is true"
            ),
            "footprint_time_samples": (
                "both pre- and post-transition footprints for each admitted side"
            ),
            "optimistic_relaxations": [
                "support is sampled after each control transition rather than at every physics substep",
                "both pre- and post-transition footprints are admitted for each supported side",
                "capsule radii are over-approximated by endpoint squares before convex hulling",
                "no actuator, force, impulse-rate, or complementarity limits",
            ],
        },
        "projection_definition": {
            "target": (
                "E018's exact successful unprojected contact-equivalent impulse"
            ),
            "objective": (
                "minimize sum_j abs(target_j - support_contact_cone_impulse_j) "
                "/ component_scale_j"
            ),
            "solver": "scipy.optimize.linprog(method='highs-ipm')",
            "residual_sign": "target minus support-aware projected impulse",
            "nonuniqueness_check": (
                "componentwise min/max residual over the complete weighted-L1 "
                "optimal set within the fixed normalized-cost tolerance"
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
                    row["candidate_points"] for row in rows
                ),
                "candidate_point_count_max": max(
                    row["candidate_points"] for row in rows
                ),
                "left_supported_transition_count_total": sum(
                    row["left_supported_transition_count"] for row in rows
                ),
                "right_supported_transition_count_total": sum(
                    row["right_supported_transition_count"] for row in rows
                ),
                "double_support_transition_count_total": sum(
                    row["double_support_transition_count"] for row in rows
                ),
            }
            for label, rows in matrix_diagnostics.items()
        }
        | {
            "full_a_full_b_support_signature_disagreement_window_count": int(
                np.count_nonzero(
                    np.any(
                        support_signatures["full-a"]
                        != support_signatures["full-b"],
                        axis=(1, 2),
                    )
                )
            )
        },
        "projected_target_recorded_support_feasibility": projected_summaries,
        "model_friction_controls": control_summaries,
        "projected_target_comparisons": comparisons,
        "discarded_residual": residual_summaries,
        "discarded_residual_vs_direct_assistance": assistance,
        "support_vs_bilateral_projection": bilateral_comparison,
        "solver": {
            "projection_primary_solve_count": int(
                len(window_starts) * len(FULL_LABELS)
            ),
            "projection_optimal_set_range_solve_count": int(
                12
                * sum(
                    np.count_nonzero(costs[label] > projection.ZERO_COST_TOLERANCE)
                    for label in FULL_LABELS
                )
            ),
            "categorical_feasibility_solve_count": int(
                len(window_starts) * (len(FULL_LABELS) + len(CONTROL_LABELS))
            ),
            "max_constraint_or_objective_error": max_solver_error,
            "max_optimal_set_normalized_component_width": (
                max_ambiguity_normalized_width
            ),
            "minimum_support_minus_bilateral_weighted_l1_cost": (
                minimum_cost_increment
            ),
        },
        "gates": {
            "solver_constraints_pass": solver_pass,
            "nested_contact_cone_cost_monotonicity_pass": nested_cost_pass,
            "recorded_support_physical_controls_pass": controls_pass,
            "support_projected_target_feasibility_pass": (
                projected_feasibility_pass
            ),
            "all_projected_signal_interval_separation_pass": separation_pass,
            "weighted_l1_optimal_set_ambiguity_pass": ambiguity_pass,
        },
        "verdict": verdict,
        "search_receipt": {
            "hypothesis": (
                "directly projecting the successful target into the saved-support "
                "cone at friction 1.0 yields a stable candidate"
            ),
            "local_sources": [
                "E017 categorical contact-cone solver and geometry",
                "E018 weighted-L1 projection and ambiguity audit",
                "E000 recorded-support matrix and physical controls",
            ],
            "external_candidates": [],
            "search_stop_reason": (
                "the exact solver, objective, immutable targets, support filter, "
                "and controls already exist locally"
            ),
            "selected_seam": (
                "replace E018's bilateral candidate matrix only; preserve every "
                "other projection input and interface"
            ),
            "verdict": "adapt-local-working-route",
        },
        "claim_scope": {
            "established": [
                "The support-aware weighted-L1 projection and discarded residual are measured for both successful replicas over all 125 fixed windows.",
                "Every projected target is independently checked in the same recorded-support cone at model friction 1.0.",
                "Successful realized-contact estimates and E002 are checked as physical controls under the identical support filter.",
                "Duplicate separation, optimal-set ambiguity, and the incremental correction from E018 are reported without simulation.",
            ],
            "not_established": [
                "Transition-level support equals the complete physics-substep contact schedule.",
                "The projected impulse is reachable under actuator, impulse-rate, or complementarity limits.",
                "A joint policy can preserve the successful trajectory without direct torso assistance.",
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
