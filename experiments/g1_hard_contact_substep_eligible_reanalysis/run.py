"""Classify E014's primal-valid hard-contact substep cases artifact-only.

E014 changed only the hard-contact scan length from four 5 ms substeps to one,
but its phase-100 DiffSim smooth objective failed the frozen direct-versus-AD
primal gate.  This successor runs no physics and computes no derivative.  It
freezes eligibility from the primal gate alone, requires the resulting mask to
be exactly the first nine cases, and classifies only those immutable rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

CASE_COUNT = 10
ACTION_DIMENSION = 29
OBJECTIVE_NAMES = ("smooth_reference_state", "e002_h1_reward")
PHASES = (0, 25, 50, 75, 100)
ARM_ORDER = ("ppo", "diffsim")
PRIMAL_RTOL = 1e-10
PRIMAL_ATOL = 1e-12
GRADIENT_RTOL = 5e-5
GRADIENT_ATOL = 1e-9
FINITE_DIFFERENCE_RTOL = 5e-3
FINITE_DIFFERENCE_ATOL = 5e-5
EXPECTED_ELIGIBLE = np.asarray([True] * 9 + [False], dtype=bool)
EXPECTED_CONTACT_COUNTS = np.asarray([2, 2, 3, 3, 2, 2, 2, 2, 3, 3])
SOURCE_HASHES = {
    "source_e014_raw": "641586cc511737612c0bd085978778d96accefc88a19ac1f9df2bcb9f7a58f48",
    "source_e014_report": "9846b95389d6f3c7239d820360ac1c635ffc4c4ba1dd451dce2916889270d287",
    "source_e014_audit": "63f9f2bb00bdc35eac98a3fb0382ee9b57cbdbdff7a279913c22d8b9d7d73622",
    "source_e009_raw": "fae96f7e218517e46d3556d722c17d23ea10813a37adddcead3762787fedba33",
    "source_e009_report": "764cdd3bfc72924130b362b65ef5f787b7b8c15869380d376c9dd26bab284b4e",
    "source_e009_audit": "b78f475efecf6a8c4a1804c691ac69834cad77fcf64af241a4b92d6b2dcac5b8",
}
METADATA_KEYS = (
    "phases",
    "arms",
    "actor_actions",
    "model_actions",
    "position_targets",
    "reset_qpos",
    "reset_qvel",
    "reset_qpos_max_abs_delta",
    "source_contact_exact",
    "direction",
)
PROBE_KEYS = (
    "source_primal",
    "reverse_primal",
    "forward_primal",
    "reverse_jacobian",
    "forward_jacobian",
    "forward_directional",
    "finite_difference_directional",
    "direct_done",
    "direct_terminal",
    "direct_contact_stiffness",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "US" or right.dtype.kind in "US":
        return bool(np.array_equal(left, right))
    return bool(np.array_equal(left, right, equal_nan=True))


def repository_preflight(repository: Path, expected_commit: str) -> dict[str, str]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "@{upstream}"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=repository, text=True
    )
    if head != expected_commit or remote != expected_commit or dirty:
        raise ValueError("artifact reanalysis requires exact clean pushed code")
    return {"repository": str(repository), "head": head, "remote": remote}


def primal_eligibility(
    source: np.ndarray, reverse: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    """Return per-case eligibility using no gradient value or outcome."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (source, reverse, forward)
    )
    if any(value.shape != (CASE_COUNT, len(OBJECTIVE_NAMES)) for value in arrays):
        raise ValueError("primal arrays have the wrong shape")
    source_array, reverse_array, forward_array = arrays
    finite = np.all(
        np.isfinite(source_array)
        & np.isfinite(reverse_array)
        & np.isfinite(forward_array),
        axis=1,
    )
    reverse_close = np.all(
        np.isclose(
            source_array,
            reverse_array,
            rtol=PRIMAL_RTOL,
            atol=PRIMAL_ATOL,
        ),
        axis=1,
    )
    forward_close = np.all(
        np.isclose(
            source_array,
            forward_array,
            rtol=PRIMAL_RTOL,
            atol=PRIMAL_ATOL,
        ),
        axis=1,
    )
    return finite & reverse_close & forward_close


def gradient_agreement(reverse: np.ndarray, forward: np.ndarray) -> np.ndarray:
    reverse_array = np.asarray(reverse, dtype=np.float64)
    forward_array = np.asarray(forward, dtype=np.float64)
    if reverse_array.shape != (CASE_COUNT, ACTION_DIMENSION) or forward_array.shape != (
        CASE_COUNT,
        ACTION_DIMENSION,
    ):
        raise ValueError("gradient arrays have the wrong shape")
    finite = np.all(np.isfinite(reverse_array) & np.isfinite(forward_array), axis=1)
    close = np.all(
        np.isclose(
            reverse_array,
            forward_array,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        ),
        axis=1,
    )
    return finite & close


def finite_difference_agreement(
    forward_directional: np.ndarray, finite_difference: np.ndarray
) -> np.ndarray:
    forward_array = np.asarray(forward_directional, dtype=np.float64)
    finite_difference_array = np.asarray(finite_difference, dtype=np.float64)
    if forward_array.shape != (CASE_COUNT,) or finite_difference_array.shape != (
        CASE_COUNT,
    ):
        raise ValueError("directional arrays have the wrong shape")
    return (
        np.isfinite(forward_array)
        & np.isfinite(finite_difference_array)
        & np.isclose(
            forward_array,
            finite_difference_array,
            rtol=FINITE_DIFFERENCE_RTOL,
            atol=FINITE_DIFFERENCE_ATOL,
        )
    )


def relative_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim == 2:
        difference = np.linalg.norm(left_array - right_array, axis=1)
        denominator = np.maximum(
            np.maximum(
                np.linalg.norm(left_array, axis=1),
                np.linalg.norm(right_array, axis=1),
            ),
            1e-12,
        )
    else:
        difference = np.abs(left_array - right_array)
        denominator = np.maximum(
            np.maximum(np.abs(left_array), np.abs(right_array)), 1e-12
        )
    return difference / denominator


def classify_eligible_substeps(
    *,
    measurement_valid: bool,
    eligible: np.ndarray,
    control_gradient_agreement: np.ndarray,
    treatment_gradient_agreement: np.ndarray,
) -> dict[str, object]:
    eligible_array = np.asarray(eligible, dtype=bool)
    control = np.asarray(control_gradient_agreement, dtype=bool)
    treatment = np.asarray(treatment_gradient_agreement, dtype=bool)
    shapes_valid = all(
        value.shape == (CASE_COUNT,) for value in (eligible_array, control, treatment)
    )
    mask_valid = bool(
        shapes_valid and np.array_equal(eligible_array, EXPECTED_ELIGIBLE)
    )
    eligible_count = int(np.sum(eligible_array)) if shapes_valid else 0
    control_count = int(np.sum(control & eligible_array)) if shapes_valid else 0
    treatment_count = int(np.sum(treatment & eligible_array)) if shapes_valid else 0
    valid = bool(measurement_valid and mask_valid and control_count == 0)
    if not valid:
        outcome = "invalid-measurement"
        interpretable = False
    elif treatment_count == 0:
        outcome = "first-substep-systematic-on-eligible-cases"
        interpretable = True
    elif treatment_count < eligible_count:
        outcome = "first-and-later-substeps-both-contribute-on-eligible-cases"
        interpretable = True
    else:
        outcome = "ad-inconsistency-emerges-after-first-substep-on-eligible-cases"
        interpretable = True
    return {
        "valid": valid,
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "eligible_case_count": eligible_count,
        "excluded_case_count": CASE_COUNT - eligible_count,
        "control_agreement_count": control_count,
        "treatment_agreement_count": treatment_count,
        "rescued_case_count": int(np.sum((~control) & treatment & eligible_array))
        if shapes_valid
        else 0,
    }


def plot_reanalysis(path: Path, report: dict[str, Any]) -> None:
    labels = [
        f"{phase}\n{arm}"
        for phase, arm in zip(report["phases"], report["arms"], strict=True)
    ]
    eligible = np.asarray(report["eligible_mask"], dtype=bool)
    control_gradient = np.asarray(
        report["control_smooth_gradient_agreement"], dtype=float
    )
    treatment_gradient = np.asarray(
        report["treatment_smooth_gradient_agreement"], dtype=float
    )
    reward_matrix = np.asarray(
        [
            report["control_reward_gradient_agreement"],
            report["treatment_reward_gradient_agreement"],
        ],
        dtype=float,
    )
    smooth_matrix = np.asarray([control_gradient, treatment_gradient], dtype=float)
    smooth_matrix[:, ~eligible] = np.nan
    reward_matrix[:, ~eligible] = np.nan
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#9e9e9e")
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    row_labels = ("hard contact: 4 x 5 ms", "hard contact: 1 x 5 ms")
    for axis, matrix, title in (
        (axes[0, 0], smooth_matrix, "smooth reverse vs complete forward AD (primary)"),
        (axes[0, 1], reward_matrix, "exact reward reverse vs complete forward AD"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        axis.set_xticks(range(CASE_COUNT), labels, fontsize=8)
        axis.set_yticks(range(2), row_labels)
        axis.set_title(title)
        for row in range(2):
            for case in range(CASE_COUNT):
                text = (
                    "EXCLUDED"
                    if not eligible[case]
                    else "PASS"
                    if matrix[row, case]
                    else "FAIL"
                )
                axis.text(case, row, text, ha="center", va="center", fontsize=7)
        figure.colorbar(image, ax=axis, ticks=(0, 1))

    for axis, control_key, treatment_key, title, threshold in (
        (
            axes[1, 0],
            "control_smooth_reverse_forward_relative_error",
            "treatment_smooth_reverse_forward_relative_error",
            "smooth reverse/forward relative error",
            GRADIENT_RTOL,
        ),
        (
            axes[1, 1],
            "control_smooth_finite_difference_relative_error",
            "treatment_smooth_finite_difference_relative_error",
            "smooth forward/fixed-FD relative error (secondary)",
            FINITE_DIFFERENCE_RTOL,
        ),
    ):
        for key, marker, condition in (
            (control_key, "o", "control"),
            (treatment_key, "s", "treatment"),
        ):
            values = np.asarray(report[key], dtype=np.float64).copy()
            values[~eligible] = np.nan
            axis.plot(
                range(CASE_COUNT),
                np.maximum(values, 1e-16),
                marker=marker,
                label=condition,
            )
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.axvspan(8.5, 9.5, color="#9e9e9e", alpha=0.3)
        axis.set_xticks(range(CASE_COUNT), labels, fontsize=8)
        axis.set_yscale("log")
        axis.set_ylabel("relative error")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Artifact-only E014 reanalysis: nine primal-valid cases")
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    os.replace(temporary, path)


def _run(args: argparse.Namespace) -> int:
    paths = {
        "source_e014_raw": args.source_e014_raw.resolve(),
        "source_e014_report": args.source_e014_report.resolve(),
        "source_e014_audit": args.source_e014_audit.resolve(),
        "source_e009_raw": args.source_e009_raw.resolve(),
        "source_e009_report": args.source_e009_report.resolve(),
        "source_e009_audit": args.source_e009_audit.resolve(),
    }
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if observed_hashes != SOURCE_HASHES:
        raise ValueError("immutable source hashes changed")
    e014_report = read_json(paths["source_e014_report"])
    e014_audit = read_json(paths["source_e014_audit"])
    e009_report = read_json(paths["source_e009_report"])
    e009_audit = read_json(paths["source_e009_audit"])
    if (
        e014_report.get("outcome") != "invalid-measurement"
        or e014_report.get("scientifically_interpretable") is not False
        or e014_audit.get("valid") is not True
        or e014_audit.get("checks_passed") != 22
        or e014_audit.get("checks_total") != 22
        or e014_audit.get("outcome") != "invalid-measurement"
        or e014_audit.get("invalid_reason") != "phase100-diffsim-smooth-primal-mismatch"
        or e009_report.get("outcome") != "both-actions-have-smooth-derivative-failures"
        or e009_audit.get("valid") is not True
        or e009_audit.get("checks_passed") != 23
        or e009_audit.get("checks_total") != 23
    ):
        raise ValueError("source classification or audit contract changed")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-hard-contact-substep-eligible-reanalysis-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": SOURCE_HASHES,
        "seed": args.seed,
        "eligibility": (
            "finite direct/reverse/forward primals for both objectives under "
            "rtol 1e-10 and atol 1e-12; no gradient value enters selection"
        ),
        "expected_eligible_mask": EXPECTED_ELIGIBLE.tolist(),
        "primary_gate": "smooth reverse versus complete coordinate-forward AD",
        "simulator_step_computed": False,
        "derivative_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    preflight_path = output_root / "preflight.json"
    write_json(preflight_path, preflight)

    e014_raw = load_npz(paths["source_e014_raw"])
    e009_raw = load_npz(paths["source_e009_raw"])
    control_copy_exact = all(
        arrays_equal(e014_raw[f"control_{name}"], value)
        for name, value in e009_raw.items()
    )
    treatment_inputs_exact = all(
        arrays_equal(e014_raw[f"treatment_{name}"], e009_raw[name])
        for name in METADATA_KEYS
    )
    treatment_repeat_exact = all(
        arrays_equal(
            e014_raw[f"treatment_first_{name}"],
            e014_raw[f"treatment_second_{name}"],
        )
        for name in PROBE_KEYS
    )
    phases = np.asarray(e014_raw["treatment_phases"], dtype=np.int64)
    arms = np.asarray(e014_raw["treatment_arms"])
    expected_phases = np.repeat(np.asarray(PHASES), len(ARM_ORDER))
    expected_arms = np.tile(np.asarray(ARM_ORDER), len(PHASES))
    contact_counts = np.asarray(
        e014_raw["treatment_reset_contact_case_counts"], dtype=np.int64
    )
    case_contract_valid = bool(
        np.array_equal(phases, expected_phases)
        and np.array_equal(arms, expected_arms)
        and np.array_equal(contact_counts, EXPECTED_CONTACT_COUNTS)
        and np.all(e014_raw["treatment_source_contact_exact"])
        and np.all(e014_raw["treatment_first_direct_done"] == 0.0)
        and np.all(e014_raw["treatment_first_direct_terminal"] == 0.0)
        and np.all(np.isfinite(e014_raw["treatment_first_direct_contact_stiffness"]))
    )
    source_primal = np.asarray(e014_raw["treatment_first_source_primal"])
    reverse_primal = np.asarray(e014_raw["treatment_first_reverse_primal"])
    forward_primal = np.asarray(e014_raw["treatment_first_forward_primal"])
    eligible = primal_eligibility(source_primal, reverse_primal, forward_primal)
    control_reverse = np.asarray(e014_raw["control_first_reverse_jacobian"])
    control_forward = np.asarray(e014_raw["control_first_forward_jacobian"])
    treatment_reverse = np.asarray(e014_raw["treatment_first_reverse_jacobian"])
    treatment_forward = np.asarray(e014_raw["treatment_first_forward_jacobian"])
    all_jacobians_finite = bool(
        np.isfinite(control_reverse).all()
        and np.isfinite(control_forward).all()
        and np.isfinite(treatment_reverse).all()
        and np.isfinite(treatment_forward).all()
    )
    objective_results: dict[str, dict[str, np.ndarray]] = {}
    for objective_index, objective in enumerate(OBJECTIVE_NAMES):
        control_gradient = gradient_agreement(
            control_reverse[:, objective_index], control_forward[:, objective_index]
        )
        treatment_gradient = gradient_agreement(
            treatment_reverse[:, objective_index], treatment_forward[:, objective_index]
        )
        control_fd = finite_difference_agreement(
            e014_raw["control_first_forward_directional"][:, objective_index],
            e014_raw["control_first_finite_difference_directional"][:, objective_index],
        )
        treatment_fd = finite_difference_agreement(
            e014_raw["treatment_first_forward_directional"][:, objective_index],
            e014_raw["treatment_first_finite_difference_directional"][
                :, objective_index
            ],
        )
        objective_results[objective] = {
            "control_gradient": control_gradient,
            "treatment_gradient": treatment_gradient,
            "control_fd": control_fd,
            "treatment_fd": treatment_fd,
            "control_gradient_error": relative_error(
                control_reverse[:, objective_index], control_forward[:, objective_index]
            ),
            "treatment_gradient_error": relative_error(
                treatment_reverse[:, objective_index],
                treatment_forward[:, objective_index],
            ),
            "control_fd_error": relative_error(
                e014_raw["control_first_forward_directional"][:, objective_index],
                e014_raw["control_first_finite_difference_directional"][
                    :, objective_index
                ],
            ),
            "treatment_fd_error": relative_error(
                e014_raw["treatment_first_forward_directional"][:, objective_index],
                e014_raw["treatment_first_finite_difference_directional"][
                    :, objective_index
                ],
            ),
        }
    smooth = objective_results["smooth_reference_state"]
    reward = objective_results["e002_h1_reward"]
    measurement_valid = bool(
        control_copy_exact
        and treatment_inputs_exact
        and treatment_repeat_exact
        and case_contract_valid
        and all_jacobians_finite
        and np.array_equal(eligible, EXPECTED_ELIGIBLE)
    )
    classification = classify_eligible_substeps(
        measurement_valid=measurement_valid,
        eligible=eligible,
        control_gradient_agreement=smooth["control_gradient"],
        treatment_gradient_agreement=smooth["treatment_gradient"],
    )
    eligible_indices = np.flatnonzero(eligible)
    excluded_indices = np.flatnonzero(~eligible)
    failing_indices = np.flatnonzero(eligible & ~smooth["treatment_gradient"])
    rescued_indices = np.flatnonzero(
        eligible & ~smooth["control_gradient"] & smooth["treatment_gradient"]
    )
    raw_output = {
        "phases": phases,
        "arms": arms,
        "eligible_mask": eligible,
        "eligible_indices": eligible_indices,
        "excluded_indices": excluded_indices,
        "failing_treatment_indices": failing_indices,
        "rescued_treatment_indices": rescued_indices,
        "source_primal": source_primal,
        "reverse_primal": reverse_primal,
        "forward_primal": forward_primal,
        "contact_counts": contact_counts,
    }
    for objective, values in objective_results.items():
        for name, value in values.items():
            raw_output[f"{objective}_{name}"] = value
    raw_path = output_root / "eligible_substep_reanalysis.npz"
    write_npz(raw_path, raw_output)
    report = {
        "protocol": "g1-hard-contact-substep-eligible-reanalysis-report-v1",
        **classification,
        "code_commit": args.code_commit,
        "source_run_id": "E-20260904-014/20260904T201953Z",
        "control_run_id": "E-20260904-009/20260904T171353Z",
        "source_hashes": SOURCE_HASHES,
        "phases": phases.tolist(),
        "arms": arms.tolist(),
        "eligible_mask": eligible.tolist(),
        "eligible_indices": eligible_indices.tolist(),
        "excluded_indices": excluded_indices.tolist(),
        "excluded_cases": [
            {
                "index": int(index),
                "phase": int(phases[index]),
                "arm": str(arms[index]),
                "reason": "direct/reverse/forward primal mismatch",
            }
            for index in excluded_indices
        ],
        "failing_treatment_indices": failing_indices.tolist(),
        "failing_treatment_cases": [
            {"index": int(index), "phase": int(phases[index]), "arm": str(arms[index])}
            for index in failing_indices
        ],
        "rescued_treatment_indices": rescued_indices.tolist(),
        "control_copy_exact": control_copy_exact,
        "treatment_inputs_exact": treatment_inputs_exact,
        "treatment_repeat_exact": treatment_repeat_exact,
        "case_contract_valid": case_contract_valid,
        "all_jacobians_finite": all_jacobians_finite,
        "control_smooth_gradient_agreement": smooth["control_gradient"].tolist(),
        "treatment_smooth_gradient_agreement": smooth["treatment_gradient"].tolist(),
        "control_reward_gradient_agreement": reward["control_gradient"].tolist(),
        "treatment_reward_gradient_agreement": reward["treatment_gradient"].tolist(),
        "control_smooth_finite_difference_agreement": smooth["control_fd"].tolist(),
        "treatment_smooth_finite_difference_agreement": smooth["treatment_fd"].tolist(),
        "control_smooth_reverse_forward_relative_error": smooth[
            "control_gradient_error"
        ].tolist(),
        "treatment_smooth_reverse_forward_relative_error": smooth[
            "treatment_gradient_error"
        ].tolist(),
        "control_smooth_finite_difference_relative_error": smooth[
            "control_fd_error"
        ].tolist(),
        "treatment_smooth_finite_difference_relative_error": smooth[
            "treatment_fd_error"
        ].tolist(),
        "eligible_reward_control_agreement_count": int(
            np.sum(eligible & reward["control_gradient"])
        ),
        "eligible_reward_treatment_agreement_count": int(
            np.sum(eligible & reward["treatment_gradient"])
        ),
        "eligible_smooth_control_fixed_fd_count": int(
            np.sum(eligible & smooth["control_fd"])
        ),
        "eligible_smooth_treatment_fixed_fd_count": int(
            np.sum(eligible & smooth["treatment_fd"])
        ),
        "primal_tolerances": {"rtol": PRIMAL_RTOL, "atol": PRIMAL_ATOL},
        "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL},
        "finite_difference_tolerances": {
            "rtol": FINITE_DIFFERENCE_RTOL,
            "atol": FINITE_DIFFERENCE_ATOL,
        },
        "raw_npz_sha256": sha256_file(raw_path),
        "simulator_step_computed": False,
        "derivative_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "eligible_substep_reanalysis.png"
    plot_reanalysis(plot_path, report)
    summary = {
        "protocol": "g1-hard-contact-substep-eligible-reanalysis-summary-v1",
        **classification,
        "eligible_indices": eligible_indices.tolist(),
        "excluded_indices": excluded_indices.tolist(),
        "failing_treatment_indices": failing_indices.tolist(),
        "rescued_treatment_indices": rescued_indices.tolist(),
        "eligible_reward_control_agreement_count": report[
            "eligible_reward_control_agreement_count"
        ],
        "eligible_reward_treatment_agreement_count": report[
            "eligible_reward_treatment_agreement_count"
        ],
        "eligible_smooth_control_fixed_fd_count": report[
            "eligible_smooth_control_fixed_fd_count"
        ],
        "eligible_smooth_treatment_fixed_fd_count": report[
            "eligible_smooth_treatment_fixed_fd_count"
        ],
        "raw_npz_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "plot_sha256": sha256_file(plot_path),
        "simulator_step_computed": False,
        "derivative_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-hard-contact-substep-eligible-reanalysis-completion-v1",
        "valid": classification["scientifically_interpretable"],
        "outcome": classification["outcome"],
        "source_raw_reused": True,
        "simulator_step_computed": False,
        "derivative_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "eligible_substep_reanalysis.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "eligible_substep_reanalysis.png": sha256_file(plot_path),
            "summary.json": sha256_file(summary_path),
        },
    }
    write_json(output_root / "completion.json", completion)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if completion["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-e014-raw", type=Path, required=True)
    parser.add_argument("--source-e014-report", type=Path, required=True)
    parser.add_argument("--source-e014-audit", type=Path, required=True)
    parser.add_argument("--source-e009-raw", type=Path, required=True)
    parser.add_argument("--source-e009-report", type=Path, required=True)
    parser.add_argument("--source-e009-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> None:
    raise SystemExit(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
