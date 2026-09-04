"""Correct and classify E016's immutable action-interpolation artifacts.

E016 acquired a complete repeatable grid but rejected it through a host-side
NumPy axis-order bug in the final nonselected-row gate.  This successor runs no
physics and computes no derivative.  It independently corrects that reducer,
freezes primal eligibility, and classifies only the already persisted grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

CASE_COUNT = 10
ACTION_DIMENSION = 29
OBJECTIVE_COUNT = 2
PHASE_CASES = ((4, 5), (6, 7))
SELECTED_PHASES = (50, 75)
ALPHAS = np.linspace(0.0, 1.0, 9, dtype=np.float64)
PRIMAL_RTOL = 1e-10
PRIMAL_ATOL = 1e-12
GRADIENT_RTOL = 5e-5
GRADIENT_ATOL = 1e-9
FINITE_DIFFERENCE_RTOL = 5e-3
FINITE_DIFFERENCE_ATOL = 5e-5
SOURCE_HASHES = {
    "source_e016_raw": "3c660bf47e1e574993c572e5dc7b4bc3d0eb9ac2c48879311fade92c14917685",
    "source_e016_report": "a65dc6eaf4cb398b90cc6104d46cd1161bbdd3a363c02d9477ea32c033e8ee8c",
    "source_e016_audit": "d6bc3de772e9da2546b11fea4a0bdbcbf730f70f91e5245632f795aeb7bd22ee",
}
PROBE_NAMES = (
    "direct_contact_stiffness",
    "direct_done",
    "direct_terminal",
    "finite_difference_directional",
    "forward_directional",
    "forward_jacobian",
    "forward_primal",
    "reverse_jacobian",
    "reverse_primal",
    "source_primal",
)
INPUT_NAMES = (
    "phases",
    "arms",
    "actor_actions",
    "model_actions",
    "position_targets",
    "reset_qpos",
    "reset_qvel",
    "reset_qpos_max_abs_delta",
    "source_contact_exact",
)
DERIVED_NAMES = (
    "selected_source_primal",
    "selected_reverse_primal",
    "selected_forward_primal",
    "selected_reverse_jacobian",
    "selected_forward_jacobian",
    "selected_forward_directional",
    "selected_finite_difference_directional",
    "selected_done",
    "selected_terminal",
    "selected_contact_stiffness",
    "primal_agreement_by_objective",
    "primal_valid",
    "gradient_agreement_by_objective",
    "finite_difference_agreement_by_objective",
    "reverse_forward_relative_error",
    "finite_difference_relative_error",
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


def expected_source_keys() -> set[str]:
    keys = {f"input_{name}" for name in INPUT_NAMES}
    keys.update(
        {
            "direction",
            "alphas",
            "phase_cases",
            "selected_phases",
            "reset_contact_signatures",
            "reset_contact_counts",
            "candidate_position_targets",
            "candidate_actions",
        }
    )
    keys.update(
        f"initial_pd_{name}"
        for name in (
            "raw_torque",
            "clipped_torque",
            "clipped",
            "effort_margin",
            "effort_utilization",
        )
    )
    keys.update(
        f"baseline_{invocation}_{name}"
        for invocation in ("first", "second")
        for name in PROBE_NAMES
    )
    keys.update(
        f"{invocation}_{name}"
        for invocation in ("first", "second")
        for name in PROBE_NAMES
    )
    keys.update(DERIVED_NAMES)
    return keys


def correct_nonselected_outputs(
    raw: dict[str, np.ndarray], *, invocation: str = "first"
) -> tuple[bool, dict[str, bool]]:
    """Compare with alpha-major then case-major indexing, unlike E016's bug."""

    results: dict[str, bool] = {}
    for name in PROBE_NAMES:
        values = np.asarray(raw[f"{invocation}_{name}"])
        baseline = np.asarray(raw[f"baseline_{invocation}_{name}"])
        if values.shape[:3] != (len(PHASE_CASES), ALPHAS.size, CASE_COUNT):
            results[name] = False
            continue
        exact = True
        for slot, (_, selected_index) in enumerate(PHASE_CASES):
            mask = np.arange(CASE_COUNT) != selected_index
            selected = values[slot][:, mask]
            expected = np.broadcast_to(baseline[mask], selected.shape)
            exact = exact and arrays_equal(selected, expected)
        results[name] = exact
    return all(results.values()), results


def selected_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[:3] != (len(PHASE_CASES), ALPHAS.size, CASE_COUNT):
        raise ValueError("interpolation array has the wrong leading shape")
    return np.stack(
        [array[slot, :, selected] for slot, (_, selected) in enumerate(PHASE_CASES)]
    )


def primal_eligibility(
    source: np.ndarray, reverse: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    """Freeze eligibility from primals only, before any gradient is inspected."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (source, reverse, forward)
    )
    expected_shape = (len(PHASE_CASES), ALPHAS.size, OBJECTIVE_COUNT)
    if any(value.shape != expected_shape for value in arrays):
        raise ValueError("selected primal arrays have the wrong shape")
    source_array, reverse_array, forward_array = arrays
    finite = np.all(
        np.isfinite(source_array)
        & np.isfinite(reverse_array)
        & np.isfinite(forward_array),
        axis=-1,
    )
    reverse_close = np.all(
        np.isclose(source_array, reverse_array, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL),
        axis=-1,
    )
    forward_close = np.all(
        np.isclose(source_array, forward_array, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL),
        axis=-1,
    )
    return finite & reverse_close & forward_close


def gradient_agreement(reverse: np.ndarray, forward: np.ndarray) -> np.ndarray:
    reverse_array = np.asarray(reverse, dtype=np.float64)
    forward_array = np.asarray(forward, dtype=np.float64)
    expected_shape = (
        len(PHASE_CASES),
        ALPHAS.size,
        OBJECTIVE_COUNT,
        ACTION_DIMENSION,
    )
    if reverse_array.shape != expected_shape or forward_array.shape != expected_shape:
        raise ValueError("selected Jacobians have the wrong shape")
    return np.all(
        np.isfinite(reverse_array)
        & np.isfinite(forward_array)
        & np.isclose(
            reverse_array,
            forward_array,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        ),
        axis=-1,
    )


def finite_difference_agreement(
    forward_directional: np.ndarray, finite_difference: np.ndarray
) -> np.ndarray:
    forward_array = np.asarray(forward_directional, dtype=np.float64)
    finite_difference_array = np.asarray(finite_difference, dtype=np.float64)
    expected_shape = (len(PHASE_CASES), ALPHAS.size, OBJECTIVE_COUNT)
    if (
        forward_array.shape != expected_shape
        or finite_difference_array.shape != expected_shape
    ):
        raise ValueError("selected directional arrays have the wrong shape")
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


def relative_error(
    left: np.ndarray, right: np.ndarray, *, axis: int | None
) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    difference = (
        np.linalg.norm(left_array - right_array, axis=axis)
        if axis is not None
        else np.abs(left_array - right_array)
    )
    left_norm = (
        np.linalg.norm(left_array, axis=axis)
        if axis is not None
        else np.abs(left_array)
    )
    right_norm = (
        np.linalg.norm(right_array, axis=axis)
        if axis is not None
        else np.abs(right_array)
    )
    return difference / np.maximum(np.maximum(left_norm, right_norm), 1e-12)


def transition_records(mask: np.ndarray, alphas: np.ndarray) -> list[dict[str, object]]:
    mask_array = np.asarray(mask, dtype=bool)
    alpha_array = np.asarray(alphas, dtype=np.float64)
    if mask_array.shape != (ALPHAS.size,) or alpha_array.shape != (ALPHAS.size,):
        raise ValueError("transition inputs have the wrong shape")
    indices = np.flatnonzero(mask_array[:-1] != mask_array[1:])
    return [
        {
            "left_alpha": float(alpha_array[index]),
            "right_alpha": float(alpha_array[index + 1]),
            "from": "pass" if mask_array[index] else "fail",
            "to": "pass" if mask_array[index + 1] else "fail",
        }
        for index in indices
    ]


def classify_action_regimes(
    *,
    measurement_valid: bool,
    primal_valid: np.ndarray,
    smooth_agreement: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, object]:
    primal = np.asarray(primal_valid, dtype=bool)
    smooth = np.asarray(smooth_agreement, dtype=bool)
    alpha_array = np.asarray(alphas, dtype=np.float64)
    expected_shape = (len(PHASE_CASES), ALPHAS.size)
    shapes_valid = (
        primal.shape == expected_shape
        and smooth.shape == expected_shape
        and alpha_array.shape == (ALPHAS.size,)
    )
    alpha_valid = bool(shapes_valid and np.array_equal(alpha_array, ALPHAS))
    endpoint_valid = bool(
        shapes_valid
        and np.all(primal[:, (0, -1)])
        and np.all(smooth[:, 0])
        and not np.any(smooth[:, -1])
    )
    valid = bool(measurement_valid and alpha_valid and endpoint_valid)
    if not valid:
        return {
            "valid": False,
            "scientifically_interpretable": False,
            "outcome": "invalid-measurement",
            "transition_records": [],
            "transition_counts": [],
            "pass_to_fail_counts": [],
            "recovery_counts": [],
        }
    if not np.all(primal):
        outcome = "transform-primal-boundary-along-action-segment"
    else:
        records = [transition_records(row, alpha_array) for row in smooth]
        pass_to_fail_counts = [
            sum(item["from"] == "pass" and item["to"] == "fail" for item in row)
            for row in records
        ]
        recovery_counts = [
            sum(item["from"] == "fail" and item["to"] == "pass" for item in row)
            for row in records
        ]
        single = all(count == 1 for count in pass_to_fail_counts) and not any(
            recovery_counts
        )
        outcome = (
            "single-ad-transition-bracketed-in-both-phases"
            if single
            else "multiple-ad-regimes-along-action-segment"
        )
    records = [transition_records(row, alpha_array) for row in smooth]
    pass_to_fail_counts = [
        sum(item["from"] == "pass" and item["to"] == "fail" for item in row)
        for row in records
    ]
    recovery_counts = [
        sum(item["from"] == "fail" and item["to"] == "pass" for item in row)
        for row in records
    ]
    return {
        "valid": True,
        "scientifically_interpretable": True,
        "outcome": outcome,
        "transition_records": records,
        "transition_counts": [len(row) for row in records],
        "pass_to_fail_counts": pass_to_fail_counts,
        "recovery_counts": recovery_counts,
    }


def plot_reanalysis(path: Path, report: dict[str, Any]) -> None:
    alphas = np.asarray(report["alphas"], dtype=np.float64)
    smooth = np.asarray(report["smooth_gradient_agreement"], dtype=float)
    smooth_error = np.asarray(report["smooth_reverse_forward_relative_error"])
    contact = np.asarray(report["contact_stiffness"])
    effort = np.asarray(report["maximum_effort_utilization_by_alpha"])
    labels = [f"{value:g}" for value in alphas]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    image = axes[0, 0].imshow(smooth, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[0, 0].set_xticks(range(ALPHAS.size), labels)
    axes[0, 0].set_yticks(range(2), ["phase 50", "phase 75"])
    axes[0, 0].set_xlabel("alpha: PPO → DiffSim")
    axes[0, 0].set_title("smooth reverse vs complete forward AD (primary)")
    for row in range(2):
        for column in range(ALPHAS.size):
            axes[0, 0].text(
                column,
                row,
                "PASS" if smooth[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[0, 0], ticks=(0, 1))

    for row, phase in enumerate(SELECTED_PHASES):
        axes[0, 1].plot(
            alphas,
            np.maximum(smooth_error[row, :, 0], 1e-16),
            marker="o",
            label=f"phase {phase}",
        )
    axes[0, 1].axhline(GRADIENT_RTOL, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("smooth reverse/forward relative error")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    for row, phase in enumerate(SELECTED_PHASES):
        axes[1, 0].plot(alphas, contact[row], marker="o", label=f"phase {phase}")
    axes[1, 0].set_title("first-solve contact stiffness scalar")
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("contact stiffness")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    for row, phase in enumerate(SELECTED_PHASES):
        axes[1, 1].plot(alphas, effort[row], marker="o", label=f"phase {phase}")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylim(0.0, max(1.05, float(np.max(effort)) * 1.1))
    axes[1, 1].set_title("maximum initial PD effort utilization")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("fraction of effort limit")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle(
        "E017 artifact-only correction: multiple AD regimes along both action segments"
    )
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    os.replace(temporary, path)


def _run(args: argparse.Namespace) -> int:
    paths = {
        "source_e016_raw": args.source_e016_raw.resolve(),
        "source_e016_report": args.source_e016_report.resolve(),
        "source_e016_audit": args.source_e016_audit.resolve(),
    }
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if observed_hashes != SOURCE_HASHES:
        raise ValueError("immutable E016 source hashes changed")
    source_report = read_json(paths["source_e016_report"])
    source_audit = read_json(paths["source_e016_audit"])
    report_true_fields = (
        "all_interpolated_primals_valid",
        "all_selected_primals_finite",
        "all_transitions_nonterminal",
        "candidate_actions_exact",
        "case_identity_exact",
        "endpoint_contract_valid",
        "endpoint_outputs_match_paired_e014_rows",
        "hard_contact_reset_exact",
        "input_match_to_e014",
        "no_initial_effort_clipping",
        "substep_only",
        "sweep_executed",
        "sweep_repeat_exact",
    )
    source_contract_valid = bool(
        source_report.get("protocol")
        == "g1-hard-contact-action-interpolation-classification-v1"
        and source_report.get("valid") is False
        and source_report.get("scientifically_interpretable") is False
        and source_report.get("outcome") == "invalid-measurement"
        and source_report.get("nonselected_outputs_match_baseline") is False
        and all(source_report.get(name) is True for name in report_true_fields)
        and source_report.get("computed_baseline_probe_invocations") == 2
        and source_report.get("computed_interpolation_probe_invocations") == 36
        and source_report.get("raw_npz_sha256") == SOURCE_HASHES["source_e016_raw"]
        and source_audit.get("valid") is True
        and source_audit.get("checks_passed") == 22
        and source_audit.get("checks_total") == 22
        and source_audit.get("outcome") == "invalid-measurement"
        and source_audit.get("invalid_reason") == "nonselected-output-axis-order"
        and source_audit.get("correct_nonselected_outputs_exact") is True
        and source_audit.get("producer_buggy_nonselected_gate") is False
        and source_audit.get("all_interpolated_primals_valid") is True
    )
    if not source_contract_valid:
        raise ValueError("E016 failure or independent-audit contract changed")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-hard-contact-action-interpolation-reanalysis-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": SOURCE_HASHES,
        "seed": args.seed,
        "source_contract_valid": source_contract_valid,
        "eligibility": (
            "finite direct/reverse/forward primals for both objectives under "
            "rtol 1e-10 and atol 1e-12; frozen before gradient inspection"
        ),
        "corrected_reducer": "values[slot][:, nonselected_case_mask]",
        "simulator_step_computed": False,
        "derivative_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    preflight_path = output_root / "preflight.json"
    write_json(preflight_path, preflight)

    raw = load_npz(paths["source_e016_raw"])
    keyset_exact = set(raw) == expected_source_keys() and all(
        value.dtype.kind != "O" for value in raw.values()
    )
    alphas_exact = arrays_equal(raw["alphas"], ALPHAS)
    pairs_exact = arrays_equal(raw["phase_cases"], np.asarray(PHASE_CASES))
    phases_exact = arrays_equal(raw["selected_phases"], np.asarray(SELECTED_PHASES))
    baseline_repeat_exact = all(
        arrays_equal(raw[f"baseline_first_{name}"], raw[f"baseline_second_{name}"])
        for name in PROBE_NAMES
    )
    sweep_repeat_exact = all(
        arrays_equal(raw[f"first_{name}"], raw[f"second_{name}"])
        for name in PROBE_NAMES
    )
    first_nonselected_exact, first_nonselected_by_output = correct_nonselected_outputs(
        raw, invocation="first"
    )
    second_nonselected_exact, second_nonselected_by_output = (
        correct_nonselected_outputs(raw, invocation="second")
    )

    base_actions = raw["input_actor_actions"]
    expected_actions = np.stack(
        [
            np.stack(
                [
                    np.vstack(
                        (
                            base_actions[:selected],
                            (1.0 - alpha) * base_actions[ppo]
                            + alpha * base_actions[selected],
                            base_actions[selected + 1 :],
                        )
                    )
                    for alpha in ALPHAS
                ]
            )
            for ppo, selected in PHASE_CASES
        ]
    )
    candidate_actions_exact = arrays_equal(raw["candidate_actions"], expected_actions)
    endpoints_exact = all(
        arrays_equal(
            raw[f"first_{name}"][slot, 0, selected],
            raw[f"baseline_first_{name}"][ppo],
        )
        and arrays_equal(
            raw[f"first_{name}"][slot, -1, selected],
            raw[f"baseline_first_{name}"][selected],
        )
        for name in PROBE_NAMES
        for slot, (ppo, selected) in enumerate(PHASE_CASES)
    )

    source_primal = selected_rows(raw["first_source_primal"])
    reverse_primal = selected_rows(raw["first_reverse_primal"])
    forward_primal = selected_rows(raw["first_forward_primal"])
    primal_valid = primal_eligibility(source_primal, reverse_primal, forward_primal)

    reverse_jacobian = selected_rows(raw["first_reverse_jacobian"])
    forward_jacobian = selected_rows(raw["first_forward_jacobian"])
    forward_directional = selected_rows(raw["first_forward_directional"])
    finite_difference = selected_rows(raw["first_finite_difference_directional"])
    gradient_by_objective = gradient_agreement(reverse_jacobian, forward_jacobian)
    fd_by_objective = finite_difference_agreement(
        forward_directional, finite_difference
    )
    reverse_forward_error = relative_error(reverse_jacobian, forward_jacobian, axis=-1)
    finite_difference_error = relative_error(
        forward_directional, finite_difference, axis=None
    )
    derived_exact = bool(
        arrays_equal(raw["selected_source_primal"], source_primal)
        and arrays_equal(raw["selected_reverse_primal"], reverse_primal)
        and arrays_equal(raw["selected_forward_primal"], forward_primal)
        and arrays_equal(raw["selected_reverse_jacobian"], reverse_jacobian)
        and arrays_equal(raw["selected_forward_jacobian"], forward_jacobian)
        and arrays_equal(raw["selected_forward_directional"], forward_directional)
        and arrays_equal(
            raw["selected_finite_difference_directional"], finite_difference
        )
        and arrays_equal(raw["primal_valid"], primal_valid)
        and arrays_equal(raw["gradient_agreement_by_objective"], gradient_by_objective)
        and arrays_equal(
            raw["finite_difference_agreement_by_objective"], fd_by_objective
        )
        and arrays_equal(raw["reverse_forward_relative_error"], reverse_forward_error)
        and arrays_equal(
            raw["finite_difference_relative_error"], finite_difference_error
        )
    )
    contact_stiffness = selected_rows(raw["first_direct_contact_stiffness"])
    selected_done = selected_rows(raw["first_direct_done"])
    selected_terminal = selected_rows(raw["first_direct_terminal"])
    maximum_effort_by_alpha = np.max(raw["initial_pd_effort_utilization"], axis=-1)
    physical_contract_valid = bool(
        np.all(raw["reset_contact_counts"] > 0)
        and np.all(np.isfinite(contact_stiffness))
        and np.all(selected_done == 0.0)
        and np.all(selected_terminal == 0.0)
        and not np.any(raw["initial_pd_clipped"])
        and np.max(maximum_effort_by_alpha) < 1.0
    )
    measurement_valid = bool(
        keyset_exact
        and alphas_exact
        and pairs_exact
        and phases_exact
        and baseline_repeat_exact
        and sweep_repeat_exact
        and first_nonselected_exact
        and second_nonselected_exact
        and candidate_actions_exact
        and endpoints_exact
        and derived_exact
        and physical_contract_valid
    )
    smooth_agreement = gradient_by_objective[..., 0]
    classification = classify_action_regimes(
        measurement_valid=measurement_valid,
        primal_valid=primal_valid,
        smooth_agreement=smooth_agreement,
        alphas=raw["alphas"],
    )
    raw_output = {
        "alphas": raw["alphas"],
        "selected_phases": raw["selected_phases"],
        "primal_valid": primal_valid,
        "smooth_gradient_agreement": smooth_agreement,
        "reward_gradient_agreement": gradient_by_objective[..., 1],
        "smooth_finite_difference_agreement": fd_by_objective[..., 0],
        "reward_finite_difference_agreement": fd_by_objective[..., 1],
        "smooth_reverse_forward_relative_error": reverse_forward_error[..., 0],
        "reward_reverse_forward_relative_error": reverse_forward_error[..., 1],
        "smooth_finite_difference_relative_error": finite_difference_error[..., 0],
        "reward_finite_difference_relative_error": finite_difference_error[..., 1],
        "contact_stiffness": contact_stiffness,
        "maximum_effort_utilization_by_alpha": maximum_effort_by_alpha,
    }
    raw_path = output_root / "action_interpolation_reanalysis.npz"
    write_npz(raw_path, raw_output)
    report = {
        "protocol": "g1-hard-contact-action-interpolation-reanalysis-report-v1",
        **classification,
        "code_commit": args.code_commit,
        "source_run_id": "E-20260904-016/20260904T214632Z",
        "source_hashes": SOURCE_HASHES,
        "source_contract_valid": source_contract_valid,
        "source_producer_outcome": source_report["outcome"],
        "source_invalid_reason": source_audit["invalid_reason"],
        "keyset_exact": keyset_exact,
        "alphas": raw["alphas"].tolist(),
        "selected_phases": raw["selected_phases"].tolist(),
        "phase_cases": raw["phase_cases"].tolist(),
        "baseline_repeat_exact": baseline_repeat_exact,
        "sweep_repeat_exact": sweep_repeat_exact,
        "first_nonselected_outputs_exact": first_nonselected_exact,
        "second_nonselected_outputs_exact": second_nonselected_exact,
        "first_nonselected_by_output": first_nonselected_by_output,
        "second_nonselected_by_output": second_nonselected_by_output,
        "candidate_actions_exact": candidate_actions_exact,
        "endpoints_exact": endpoints_exact,
        "derived_arrays_exact": derived_exact,
        "physical_contract_valid": physical_contract_valid,
        "all_interpolated_primals_valid": bool(np.all(primal_valid)),
        "primal_valid": primal_valid.tolist(),
        "smooth_gradient_agreement": smooth_agreement.tolist(),
        "reward_gradient_agreement": gradient_by_objective[..., 1].tolist(),
        "smooth_finite_difference_agreement": fd_by_objective[..., 0].tolist(),
        "reward_finite_difference_agreement": fd_by_objective[..., 1].tolist(),
        "smooth_reverse_forward_relative_error": reverse_forward_error.tolist(),
        "finite_difference_relative_error": finite_difference_error.tolist(),
        "contact_stiffness": contact_stiffness.tolist(),
        "maximum_effort_utilization_by_alpha": maximum_effort_by_alpha.tolist(),
        "maximum_initial_effort_utilization": float(np.max(maximum_effort_by_alpha)),
        "minimum_initial_effort_margin": float(np.min(raw["initial_pd_effort_margin"])),
        "smooth_ad_agreement_count": int(np.sum(smooth_agreement)),
        "reward_ad_agreement_count": int(np.sum(gradient_by_objective[..., 1])),
        "smooth_fixed_fd_agreement_count": int(np.sum(fd_by_objective[..., 0])),
        "reward_fixed_fd_agreement_count": int(np.sum(fd_by_objective[..., 1])),
        "primal_tolerances": {"rtol": PRIMAL_RTOL, "atol": PRIMAL_ATOL},
        "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL},
        "finite_difference_tolerances": {
            "rtol": FINITE_DIFFERENCE_RTOL,
            "atol": FINITE_DIFFERENCE_ATOL,
        },
        "raw_npz_sha256": sha256_file(raw_path),
        "source_simulator_artifact_reused": True,
        "source_derivative_artifact_reused": True,
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
    plot_path = output_root / "action_interpolation_reanalysis.png"
    plot_reanalysis(plot_path, report)
    summary = {
        "protocol": "g1-hard-contact-action-interpolation-reanalysis-summary-v1",
        **classification,
        "selected_phases": list(SELECTED_PHASES),
        "alphas": ALPHAS.tolist(),
        "primal_valid": primal_valid.tolist(),
        "smooth_gradient_agreement": smooth_agreement.tolist(),
        "smooth_ad_agreement_count": report["smooth_ad_agreement_count"],
        "reward_ad_agreement_count": report["reward_ad_agreement_count"],
        "smooth_fixed_fd_agreement_count": report["smooth_fixed_fd_agreement_count"],
        "maximum_initial_effort_utilization": report[
            "maximum_initial_effort_utilization"
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
        "protocol": "g1-hard-contact-action-interpolation-reanalysis-completion-v1",
        "valid": classification["scientifically_interpretable"],
        "outcome": classification["outcome"],
        "source_raw_reused": True,
        "source_simulator_artifact_reused": True,
        "source_derivative_artifact_reused": True,
        "simulator_step_computed": False,
        "derivative_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "action_interpolation_reanalysis.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "action_interpolation_reanalysis.png": sha256_file(plot_path),
            "summary.json": sha256_file(summary_path),
        },
    }
    write_json(output_root / "completion.json", completion)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if completion["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-e016-raw", type=Path, required=True)
    parser.add_argument("--source-e016-report", type=Path, required=True)
    parser.add_argument("--source-e016-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> None:
    raise SystemExit(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
