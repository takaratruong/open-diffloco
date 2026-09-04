"""Locate G1 hard-contact AD boundaries between paired PPO and DiffSim actions.

E015 established that the phase-50 and phase-75 DiffSim actions disagree across
reverse and complete coordinate-forward AD after the first 5 ms physical solve,
while their paired PPO actions pass from identical reset states.  This runner
rebuilds E014's exact complete ten-case one-substep probe once, requires two
baseline calls to match both persisted E014 invocations bit-for-bit, and then
changes only one DiffSim row at a time along a frozen PPO-to-DiffSim action
segment.  Every call retains the original full batch and compiled graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from experiments.g1_hard_contact_substep_derivative_discriminator.run import (
    set_one_physics_substep,
)
from experiments.g1_reset_action_derivative_discriminator.run import (
    ACTION_DIMENSION,
    CASE_COUNT,
    FINITE_DIFFERENCE_ATOL,
    FINITE_DIFFERENCE_RTOL,
    GRADIENT_ATOL,
    GRADIENT_RTOL,
    OBJECTIVE_NAMES,
    PRIMAL_ATOL,
    PRIMAL_RTOL,
    _arrays_exact,
    _build_compiled_probe,
    _load_source_arrays,
    _prepare_cases,
    _validate_e008_audit,
    _write_npz,
    build_common_probe_env,
)
from experiments.g1_reset_contact_derivative_discriminator.run import _load_npz
from experiments.g1_success_failure_visitation.run import (
    read_json,
    repository_preflight,
    sha256_file,
    validate_diffsim_hparams,
    write_json,
)
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_tracking_shac import configure_jax

REFERENCE_SHA256 = "f47d13b431d85a273eba6022f5a28bd55cae7c788112baf0778ab159914a039c"
DIFFSIM_HPARAMS_SHA256 = (
    "79927f89ef75cf0a6fbfd5c92746a59db587c00319db780dcad702f0c3bbd5eb"
)
SOURCE_TRAJECTORY_SHA256 = (
    "dc4199fa5383e7caf31c89bb56c7d261af6561ce237d48e8e217276827dbc89b"
)
SOURCE_E008_AUDIT_SHA256 = (
    "9859cc5a0d5a91311238d122eb2876f40571843351e6341322abdbf35e6edd56"
)
SOURCE_E014_RAW_SHA256 = (
    "641586cc511737612c0bd085978778d96accefc88a19ac1f9df2bcb9f7a58f48"
)
SOURCE_E014_REPORT_SHA256 = (
    "9846b95389d6f3c7239d820360ac1c635ffc4c4ba1dd451dce2916889270d287"
)
SOURCE_E014_AUDIT_SHA256 = (
    "63f9f2bb00bdc35eac98a3fb0382ee9b57cbdbdff7a279913c22d8b9d7d73622"
)
ALPHAS = np.linspace(0.0, 1.0, 9, dtype=np.float64)
PHASE_CASES = ((4, 5), (6, 7))
SELECTED_PHASES = (50, 75)
PROBE_OUTPUT_NAMES = (
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


def _extract_e014_input(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = {f"treatment_{name}" for name in (*INPUT_NAMES, "direction")}
    if not keys.issubset(raw):
        raise ValueError("E014 treatment input is incomplete")
    return {
        name: np.asarray(raw[f"treatment_{name}"])
        for name in (*INPUT_NAMES, "direction")
    }


def _extract_e014_probe_output(
    raw: Mapping[str, np.ndarray], invocation: str
) -> dict[str, np.ndarray]:
    if invocation not in {"first", "second"}:
        raise ValueError("E014 invocation must be first or second")
    prefix = f"treatment_{invocation}_"
    keys = {f"{prefix}{name}" for name in PROBE_OUTPUT_NAMES}
    if not keys.issubset(raw):
        raise ValueError(f"complete E014 {invocation} probe output is missing")
    return {name: np.asarray(raw[f"{prefix}{name}"]) for name in PROBE_OUTPUT_NAMES}


def baseline_replay_gate(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    expected_first: Mapping[str, np.ndarray],
    expected_second: Mapping[str, np.ndarray],
) -> dict[str, bool]:
    """Require the new executable to reproduce both complete E014 outputs."""
    expected_names = set(PROBE_OUTPUT_NAMES)
    output_names_exact = all(
        set(values) == expected_names
        for values in (first, second, expected_first, expected_second)
    )
    repeat_exact = output_names_exact and _arrays_exact(first, second)
    first_matches = output_names_exact and _arrays_exact(first, expected_first)
    second_matches = output_names_exact and _arrays_exact(second, expected_second)
    return {
        "output_names_exact": bool(output_names_exact),
        "repeat_exact": bool(repeat_exact),
        "first_matches_e014": bool(first_matches),
        "second_matches_e014": bool(second_matches),
        "valid": bool(
            output_names_exact and repeat_exact and first_matches and second_matches
        ),
    }


def interpolate_case_actions(
    actions: object,
    *,
    ppo_index: int,
    diffsim_index: int,
    alpha: float,
):
    """Replace one DiffSim row by a frozen PPO-to-DiffSim interpolation."""
    if (ppo_index, diffsim_index) not in PHASE_CASES:
        raise ValueError("indices must name a registered paired PPO/DiffSim case")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    action_array = jnp.asarray(actions)
    if action_array.shape != (CASE_COUNT, ACTION_DIMENSION):
        raise ValueError("complete action batch has the wrong shape")
    interpolated = (1.0 - alpha_value) * action_array[
        ppo_index
    ] + alpha_value * action_array[diffsim_index]
    return action_array.at[diffsim_index].set(interpolated)


def _invoke_full_probe(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    states: object,
    actions: object,
) -> dict[str, np.ndarray]:
    device_result = compiled_probe(states, actions)
    jax.block_until_ready(device_result)
    result = {name: np.asarray(value) for name, value in device_result.items()}
    if set(result) != set(PROBE_OUTPUT_NAMES):
        raise ValueError("exact E014 probe output names changed")
    return result


def execute_interpolation_sweeps(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    *,
    states: object,
    actions: object,
    alphas: np.ndarray,
    phase_cases: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray | bool]:
    """Run each complete action interpolation twice through one compiled probe."""
    alpha_array = np.asarray(alphas, dtype=np.float64)
    if (
        alpha_array.ndim != 1
        or alpha_array.size < 2
        or not np.all(np.isfinite(alpha_array))
        or not np.all(np.diff(alpha_array) > 0.0)
        or alpha_array[0] != 0.0
        or alpha_array[-1] != 1.0
    ):
        raise ValueError("alpha grid must be finite, increasing, and include 0 and 1")
    pairs = tuple(phase_cases)
    if not pairs or any(pair not in PHASE_CASES for pair in pairs):
        raise ValueError("phase cases must use registered PPO/DiffSim pairs")
    repeats: list[dict[str, np.ndarray]] = []
    candidate_actions: list[np.ndarray] = []
    for repeat in range(2):
        outputs: dict[str, list[list[np.ndarray]]] = {
            name: [] for name in PROBE_OUTPUT_NAMES
        }
        repeat_actions: list[list[np.ndarray]] = []
        for ppo_index, diffsim_index in pairs:
            phase_outputs = {name: [] for name in PROBE_OUTPUT_NAMES}
            phase_actions = []
            for alpha in alpha_array:
                candidate = interpolate_case_actions(
                    actions,
                    ppo_index=ppo_index,
                    diffsim_index=diffsim_index,
                    alpha=float(alpha),
                )
                result = _invoke_full_probe(compiled_probe, states, candidate)
                phase_actions.append(np.asarray(candidate))
                for name in PROBE_OUTPUT_NAMES:
                    phase_outputs[name].append(result[name])
            repeat_actions.append(phase_actions)
            for name in PROBE_OUTPUT_NAMES:
                outputs[name].append(phase_outputs[name])
        repeats.append({name: np.asarray(values) for name, values in outputs.items()})
        candidate_actions.append(np.asarray(repeat_actions))
        if repeat == 1 and not np.array_equal(
            candidate_actions[0], candidate_actions[1]
        ):
            raise ValueError("interpolated action construction did not repeat exactly")
    repeat_exact = _arrays_exact(repeats[0], repeats[1])
    return {
        **{f"first_{name}": value for name, value in repeats[0].items()},
        **{f"second_{name}": value for name, value in repeats[1].items()},
        "candidate_actions": candidate_actions[0],
        "repeat_exact": repeat_exact,
    }


def initial_pd_diagnostics(
    *,
    kp: np.ndarray,
    kd: np.ndarray,
    effort_limit: np.ndarray,
    position_target: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the exact manual PD expression before the first MJX solve."""
    kp_array = np.asarray(kp, dtype=np.float64)
    kd_array = np.asarray(kd, dtype=np.float64)
    limit_array = np.asarray(effort_limit, dtype=np.float64)
    target_array = np.asarray(position_target, dtype=np.float64)
    qpos_array = np.asarray(joint_position, dtype=np.float64)
    qvel_array = np.asarray(joint_velocity, dtype=np.float64)
    if (
        kp_array.ndim != 1
        or kp_array.shape != kd_array.shape
        or kp_array.shape != limit_array.shape
    ):
        raise ValueError("PD controller arrays must share one joint dimension")
    if (
        target_array.shape != qpos_array.shape
        or target_array.shape != qvel_array.shape
        or target_array.shape[-1:] != kp_array.shape
    ):
        raise ValueError("PD state and target arrays have incompatible shapes")
    raw_torque = kp_array * (target_array - qpos_array) - kd_array * qvel_array
    clipped_torque = np.clip(raw_torque, -limit_array, limit_array)
    effort_margin = limit_array - np.abs(raw_torque)
    return {
        "raw_torque": raw_torque,
        "clipped_torque": clipped_torque,
        "clipped": np.abs(raw_torque) >= limit_array,
        "effort_margin": effort_margin,
        "effort_utilization": np.abs(raw_torque) / limit_array,
    }


def _selected_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[:3] != (len(PHASE_CASES), ALPHAS.size, CASE_COUNT):
        raise ValueError("interpolation output has the wrong phase/alpha/case shape")
    return np.stack(
        [
            array[slot, :, diffsim_index]
            for slot, (_, diffsim_index) in enumerate(PHASE_CASES)
        ]
    )


def _relative_error(left: np.ndarray, right: np.ndarray, *, axis: int) -> np.ndarray:
    difference = np.linalg.norm(left - right, axis=axis)
    denominator = np.maximum(
        np.maximum(np.linalg.norm(left, axis=axis), np.linalg.norm(right, axis=axis)),
        1e-12,
    )
    return difference / denominator


def _primal_agreement(
    source: np.ndarray, reverse: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    finite = np.isfinite(source) & np.isfinite(reverse) & np.isfinite(forward)
    return (
        finite
        & np.isclose(source, reverse, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL)
        & np.isclose(source, forward, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL)
    )


def _gradient_agreement(reverse: np.ndarray, forward: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(reverse) & np.isfinite(forward), axis=-1) & np.all(
        np.isclose(reverse, forward, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), axis=-1
    )


def _finite_difference_agreement(
    forward_directional: np.ndarray, finite_difference: np.ndarray
) -> np.ndarray:
    return (
        np.isfinite(forward_directional)
        & np.isfinite(finite_difference)
        & np.isclose(
            forward_directional,
            finite_difference,
            rtol=FINITE_DIFFERENCE_RTOL,
            atol=FINITE_DIFFERENCE_ATOL,
        )
    )


def classify_action_interpolation(
    *,
    measurement_valid: bool,
    alphas: np.ndarray,
    primal_valid: np.ndarray,
    gradient_agreement: np.ndarray,
) -> dict[str, object]:
    """Classify the topology of the smooth-AD pass mask on both action segments."""
    alpha_array = np.asarray(alphas, dtype=np.float64)
    primal = np.asarray(primal_valid, dtype=bool)
    gradient = np.asarray(gradient_agreement, dtype=bool)
    shape_valid = (
        alpha_array.ndim == 1
        and alpha_array.size >= 2
        and primal.shape == (len(PHASE_CASES), alpha_array.size)
        and gradient.shape == primal.shape
        and np.all(np.isfinite(alpha_array))
        and alpha_array[0] == 0.0
        and alpha_array[-1] == 1.0
    )
    endpoint_valid = bool(
        shape_valid
        and np.all(primal[:, [0, -1]])
        and np.all(gradient[:, 0])
        and not np.any(gradient[:, -1])
    )
    valid = bool(measurement_valid and endpoint_valid)
    if not valid:
        outcome = "invalid-measurement"
        interpretable = False
    elif not np.all(primal):
        outcome = "transform-primal-boundary-along-action-segment"
        interpretable = True
    else:
        prefix_patterns = [
            bool(np.all(np.diff(row.astype(np.int8)) <= 0)) for row in gradient
        ]
        if all(prefix_patterns):
            outcome = "single-ad-transition-bracketed-in-both-phases"
        else:
            outcome = "multiple-ad-regimes-along-action-segment"
        interpretable = True
    brackets: list[dict[str, float] | None] = []
    if shape_valid:
        for row in gradient:
            transitions = np.flatnonzero(row[:-1] & ~row[1:])
            if transitions.size == 1:
                index = int(transitions[0])
                brackets.append(
                    {
                        "pass_alpha": float(alpha_array[index]),
                        "fail_alpha": float(alpha_array[index + 1]),
                    }
                )
            else:
                brackets.append(None)
    return {
        "protocol": "g1-hard-contact-action-interpolation-classification-v1",
        "valid": valid,
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "endpoint_contract_valid": endpoint_valid,
        "all_interpolated_primals_valid": bool(shape_valid and np.all(primal)),
        "transition_brackets": brackets,
    }


def _all_nonselected_outputs_match_baseline(
    sweep: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray]
) -> bool:
    for name in PROBE_OUTPUT_NAMES:
        values = np.asarray(sweep[f"first_{name}"])
        expected = np.asarray(baseline[name])
        if values.shape[:3] != (len(PHASE_CASES), ALPHAS.size, CASE_COUNT):
            return False
        for slot, (_, diffsim_index) in enumerate(PHASE_CASES):
            mask = np.arange(CASE_COUNT) != diffsim_index
            tiled = np.broadcast_to(expected[mask], values[slot, :, mask].shape)
            if not np.array_equal(values[slot, :, mask], tiled, equal_nan=True):
                return False
    return True


def _endpoint_outputs_match_pairs(
    sweep: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray]
) -> bool:
    for name in PROBE_OUTPUT_NAMES:
        values = np.asarray(sweep[f"first_{name}"])
        expected = np.asarray(baseline[name])
        for slot, (ppo_index, diffsim_index) in enumerate(PHASE_CASES):
            if not np.array_equal(
                values[slot, 0, diffsim_index], expected[ppo_index], equal_nan=True
            ):
                return False
            if not np.array_equal(
                values[slot, -1, diffsim_index], expected[diffsim_index], equal_nan=True
            ):
                return False
    return True


def _validate_e014_sources(
    report: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    expected_audit = {
        "protocol": "g1-hard-contact-substep-derivative-independent-audit-v1",
        "valid": True,
        "scientifically_interpretable": False,
        "outcome": "invalid-measurement",
        "checks_passed": 22,
        "checks_total": 22,
        "control_gradient_agreement_count": 0,
        "treatment_gradient_agreement_count": 7,
        "simulator_step_computed": True,
        "derivative_computed": True,
        "policy_retained": False,
    }
    mismatches = {
        key: (audit.get(key), expected)
        for key, expected in expected_audit.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"E014 audit contract changed: {mismatches}")
    if (
        report.get("protocol") != "g1-hard-contact-substep-derivative-classification-v1"
        or report.get("valid") is not False
        or report.get("outcome") != "invalid-measurement"
        or report.get("raw_npz_sha256") != SOURCE_E014_RAW_SHA256
        or report.get("treatment_gradient_agreement_count") != 7
    ):
        raise ValueError("E014 report contract changed")


def _finite_or_none(values: np.ndarray) -> object:
    array = np.asarray(values)
    if array.ndim == 0:
        return float(array) if np.isfinite(array) else None
    return [_finite_or_none(value) for value in array]


def _plot_result(
    path: Path,
    *,
    primal_valid: np.ndarray,
    smooth_agreement: np.ndarray,
    smooth_error: np.ndarray,
    contact_stiffness: np.ndarray,
    torque_utilization: np.ndarray,
) -> None:
    status = np.where(primal_valid, smooth_agreement.astype(np.int8), -1)
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    cmap = ListedColormap(("#9ca3af", "#dc2626", "#16a34a"))
    norm = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), cmap.N)
    axes[0, 0].imshow(status, cmap=cmap, norm=norm, aspect="auto")
    axes[0, 0].set_xticks(range(ALPHAS.size), [f"{value:g}" for value in ALPHAS])
    axes[0, 0].set_yticks(range(len(SELECTED_PHASES)), SELECTED_PHASES)
    axes[0, 0].set_xlabel("alpha: PPO (0) to DiffSim (1)")
    axes[0, 0].set_ylabel("reset phase")
    axes[0, 0].set_title("Smooth reverse vs complete-forward AD")
    for row in range(status.shape[0]):
        for column in range(status.shape[1]):
            label = (
                "INVALID"
                if status[row, column] < 0
                else ("PASS" if status[row, column] else "FAIL")
            )
            axes[0, 0].text(column, row, label, ha="center", va="center", fontsize=7)

    for slot, phase in enumerate(SELECTED_PHASES):
        axes[0, 1].plot(
            ALPHAS,
            np.maximum(smooth_error[slot], 1e-16),
            marker="o",
            label=f"phase {phase}",
        )
    axes[0, 1].axhline(GRADIENT_RTOL, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].set_title("Smooth reverse/forward disagreement")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    for slot, phase in enumerate(SELECTED_PHASES):
        axes[1, 0].plot(
            ALPHAS, contact_stiffness[slot], marker="o", label=f"phase {phase}"
        )
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("transition contact stiffness")
    axes[1, 0].set_title("Existing exact-probe contact diagnostic")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    for slot, phase in enumerate(SELECTED_PHASES):
        axes[1, 1].plot(
            ALPHAS, torque_utilization[slot], marker="o", label=f"phase {phase}"
        )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("maximum |initial PD torque| / effort limit")
    axes[1, 1].set_title("Initial effort utilization (no clipping below 1)")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    figure.suptitle("G1 exact one-substep PPO-to-DiffSim action interpolation")
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    os.replace(temporary, path)


def _plot_invalid(path: Path, gate: Mapping[str, bool]) -> None:
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.axis("off")
    lines = ["Exact E014 baseline replay failed; interpolation was not interpreted."]
    lines.extend(f"{name}: {value}" for name, value in gate.items())
    axis.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace")
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    os.replace(temporary, path)


def _run(args: argparse.Namespace) -> int:
    paths = {
        "reference": args.reference_path.resolve(),
        "diffsim_hparams": args.diffsim_hparams.resolve(),
        "source_trajectories": args.source_trajectories.resolve(),
        "source_e008_audit": args.source_e008_audit.resolve(),
        "source_e014_raw": args.source_e014_raw.resolve(),
        "source_e014_report": args.source_e014_report.resolve(),
        "source_e014_audit": args.source_e014_audit.resolve(),
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "diffsim_hparams": DIFFSIM_HPARAMS_SHA256,
        "source_trajectories": SOURCE_TRAJECTORY_SHA256,
        "source_e008_audit": SOURCE_E008_AUDIT_SHA256,
        "source_e014_raw": SOURCE_E014_RAW_SHA256,
        "source_e014_report": SOURCE_E014_REPORT_SHA256,
        "source_e014_audit": SOURCE_E014_AUDIT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")
    if not jax.config.x64_enabled:
        raise ValueError("hard-contact action interpolation requires JAX x64")

    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    _validate_e008_audit(read_json(paths["source_e008_audit"]))
    e014_raw = _load_npz(paths["source_e014_raw"])
    e014_input = _extract_e014_input(e014_raw)
    expected_first = _extract_e014_probe_output(e014_raw, "first")
    expected_second = _extract_e014_probe_output(e014_raw, "second")
    _validate_e014_sources(
        read_json(paths["source_e014_report"]), read_json(paths["source_e014_audit"])
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-hard-contact-action-interpolation-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "seed": args.seed,
        "selected_phases": list(SELECTED_PHASES),
        "phase_cases": [list(pair) for pair in PHASE_CASES],
        "alphas": ALPHAS.tolist(),
        "case_count": CASE_COUNT,
        "action_dimension": ACTION_DIMENSION,
        "objectives": list(OBJECTIVE_NAMES),
        "probe_output_names": list(PROBE_OUTPUT_NAMES),
        "primary_gate": "smooth reverse versus complete coordinate-forward AD",
        "finite_difference_role": "secondary descriptive diagnostic",
        "solver_profile": args.solver_profile,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": True,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    preflight_path = output_root / "preflight.json"
    write_json(preflight_path, preflight)

    source_arrays = _load_source_arrays(paths["source_trajectories"])
    with solver_context(get_solver_profile(args.solver_profile)):
        env = build_common_probe_env(paths["reference"], hparams)
        original_n_frames = int(env.n_frames)
        original_dt = float(env.dt)
        states, actions, metadata = _prepare_cases(env, source_arrays, seed=args.seed)
        reset_contact_signatures = np.asarray(
            jax.vmap(env.contact_pair_signature)(states.data), dtype=bool
        )
        set_one_physics_substep(env)
        treated_n_frames = int(env.n_frames)
        direction = jnp.asarray(e014_input["direction"], dtype=jnp.float64)
        compiled_probe = _build_compiled_probe(env, direction)
        baseline_first = _invoke_full_probe(compiled_probe, states, actions)
        baseline_second = _invoke_full_probe(compiled_probe, states, actions)
        baseline_gate = baseline_replay_gate(
            baseline_first, baseline_second, expected_first, expected_second
        )
        sweeps = None
        if baseline_gate["valid"]:
            sweeps = execute_interpolation_sweeps(
                compiled_probe,
                states=states,
                actions=actions,
                alphas=ALPHAS,
                phase_cases=PHASE_CASES,
            )

    input_match = all(
        np.array_equal(np.asarray(metadata[name]), e014_input[name])
        for name in INPUT_NAMES
    ) and np.array_equal(np.asarray(direction), e014_input["direction"])
    case_identity = all(
        int(metadata["phases"][diffsim]) == phase
        and metadata["arms"][ppo] == "ppo"
        and metadata["arms"][diffsim] == "diffsim"
        and np.array_equal(metadata["reset_qpos"][ppo], metadata["reset_qpos"][diffsim])
        and np.array_equal(metadata["reset_qvel"][ppo], metadata["reset_qvel"][diffsim])
        for phase, (ppo, diffsim) in zip(SELECTED_PHASES, PHASE_CASES, strict=True)
    )
    reset_contact_counts = np.sum(
        reset_contact_signatures.reshape(CASE_COUNT, -1), axis=1
    )
    hard_contact_exact = bool(
        np.all(reset_contact_counts > 0)
        and all(
            np.array_equal(
                reset_contact_signatures[ppo], reset_contact_signatures[diffsim]
            )
            for ppo, diffsim in PHASE_CASES
        )
    )
    substep_only = bool(
        original_n_frames == 4
        and treated_n_frames == 1
        and float(env.dt) == original_dt
    )

    action_array = np.asarray(actions, dtype=np.float64)
    target_array = np.asarray(metadata["position_targets"], dtype=np.float64)
    candidate_targets = np.stack(
        [
            np.stack(
                [
                    (1.0 - alpha) * target_array[ppo] + alpha * target_array[diffsim]
                    for alpha in ALPHAS
                ]
            )
            for ppo, diffsim in PHASE_CASES
        ]
    )
    candidate_qpos = np.stack(
        [
            np.broadcast_to(
                metadata["reset_qpos"][diffsim, 7:], candidate_targets[slot].shape
            )
            for slot, (_, diffsim) in enumerate(PHASE_CASES)
        ]
    )
    candidate_qvel = np.stack(
        [
            np.broadcast_to(
                metadata["reset_qvel"][diffsim, 6:], candidate_targets[slot].shape
            )
            for slot, (_, diffsim) in enumerate(PHASE_CASES)
        ]
    )
    pd = initial_pd_diagnostics(
        kp=np.asarray(env.kp),
        kd=np.asarray(env.kd),
        effort_limit=np.asarray(env.effort_limit),
        position_target=candidate_targets,
        joint_position=candidate_qpos,
        joint_velocity=candidate_qvel,
    )
    no_initial_effort_clipping = bool(not np.any(pd["clipped"]))

    raw_arrays: dict[str, np.ndarray] = {
        **{f"input_{name}": np.asarray(value) for name, value in metadata.items()},
        "direction": np.asarray(direction),
        "alphas": ALPHAS,
        "phase_cases": np.asarray(PHASE_CASES, dtype=np.int64),
        "selected_phases": np.asarray(SELECTED_PHASES, dtype=np.int64),
        "reset_contact_signatures": reset_contact_signatures,
        "reset_contact_counts": reset_contact_counts,
        "candidate_position_targets": candidate_targets,
        **{f"initial_pd_{name}": value for name, value in pd.items()},
        **{f"baseline_first_{name}": value for name, value in baseline_first.items()},
        **{f"baseline_second_{name}": value for name, value in baseline_second.items()},
    }

    measurement_valid = False
    classification = classify_action_interpolation(
        measurement_valid=False,
        alphas=ALPHAS,
        primal_valid=np.zeros((len(PHASE_CASES), ALPHAS.size), dtype=bool),
        gradient_agreement=np.zeros((len(PHASE_CASES), ALPHAS.size), dtype=bool),
    )
    derived: dict[str, np.ndarray] = {}
    nonselected_exact = False
    endpoints_exact = False
    sweep_repeat_exact = False
    all_transitions_nonterminal = False
    all_selected_primals_finite = False
    if sweeps is not None:
        raw_arrays.update(
            {
                name: np.asarray(value)
                for name, value in sweeps.items()
                if name != "repeat_exact"
            }
        )
        sweep_repeat_exact = bool(sweeps["repeat_exact"])
        nonselected_exact = _all_nonselected_outputs_match_baseline(
            sweeps, baseline_first
        )
        endpoints_exact = _endpoint_outputs_match_pairs(sweeps, baseline_first)
        source_primal = _selected_rows(sweeps["first_source_primal"])
        reverse_primal = _selected_rows(sweeps["first_reverse_primal"])
        forward_primal = _selected_rows(sweeps["first_forward_primal"])
        reverse_jacobian = _selected_rows(sweeps["first_reverse_jacobian"])
        forward_jacobian = _selected_rows(sweeps["first_forward_jacobian"])
        forward_directional = _selected_rows(sweeps["first_forward_directional"])
        finite_difference = _selected_rows(
            sweeps["first_finite_difference_directional"]
        )
        done = _selected_rows(sweeps["first_direct_done"])
        terminal = _selected_rows(sweeps["first_direct_terminal"])
        contact_stiffness = _selected_rows(sweeps["first_direct_contact_stiffness"])
        primal_by_objective = _primal_agreement(
            source_primal, reverse_primal, forward_primal
        )
        primal_valid = np.all(primal_by_objective, axis=-1)
        gradient_by_objective = _gradient_agreement(reverse_jacobian, forward_jacobian)
        fd_by_objective = _finite_difference_agreement(
            forward_directional, finite_difference
        )
        reverse_forward_error = _relative_error(
            reverse_jacobian, forward_jacobian, axis=-1
        )
        fd_error = np.abs(forward_directional - finite_difference) / np.maximum(
            np.maximum(np.abs(forward_directional), np.abs(finite_difference)), 1e-12
        )
        all_transitions_nonterminal = bool(
            np.all(done == 0.0) and np.all(terminal == 0.0)
        )
        all_selected_primals_finite = bool(
            np.all(np.isfinite(source_primal))
            and np.all(np.isfinite(reverse_primal))
            and np.all(np.isfinite(forward_primal))
        )
        candidate_actions_exact = bool(
            np.array_equal(
                sweeps["candidate_actions"][:, :, np.asarray([5, 7])],
                np.stack(
                    [
                        np.stack(
                            [
                                np.asarray(
                                    interpolate_case_actions(
                                        action_array,
                                        ppo_index=ppo,
                                        diffsim_index=diffsim,
                                        alpha=float(alpha),
                                    )
                                )[[5, 7]]
                                for alpha in ALPHAS
                            ]
                        )
                        for ppo, diffsim in PHASE_CASES
                    ]
                ),
            )
        )
        measurement_valid = bool(
            baseline_gate["valid"]
            and input_match
            and case_identity
            and hard_contact_exact
            and substep_only
            and sweep_repeat_exact
            and nonselected_exact
            and endpoints_exact
            and candidate_actions_exact
            and all_selected_primals_finite
            and all_transitions_nonterminal
            and no_initial_effort_clipping
        )
        classification = classify_action_interpolation(
            measurement_valid=measurement_valid,
            alphas=ALPHAS,
            primal_valid=primal_valid,
            gradient_agreement=gradient_by_objective[..., 0],
        )
        derived = {
            "selected_source_primal": source_primal,
            "selected_reverse_primal": reverse_primal,
            "selected_forward_primal": forward_primal,
            "selected_reverse_jacobian": reverse_jacobian,
            "selected_forward_jacobian": forward_jacobian,
            "selected_forward_directional": forward_directional,
            "selected_finite_difference_directional": finite_difference,
            "selected_done": done,
            "selected_terminal": terminal,
            "selected_contact_stiffness": contact_stiffness,
            "primal_agreement_by_objective": primal_by_objective,
            "primal_valid": primal_valid,
            "gradient_agreement_by_objective": gradient_by_objective,
            "finite_difference_agreement_by_objective": fd_by_objective,
            "reverse_forward_relative_error": reverse_forward_error,
            "finite_difference_relative_error": fd_error,
        }
        raw_arrays.update(derived)
    else:
        candidate_actions_exact = False
        primal_valid = np.zeros((len(PHASE_CASES), ALPHAS.size), dtype=bool)
        gradient_by_objective = np.zeros(
            (len(PHASE_CASES), ALPHAS.size, len(OBJECTIVE_NAMES)), dtype=bool
        )
        fd_by_objective = gradient_by_objective.copy()
        reverse_forward_error = np.full(gradient_by_objective.shape, np.nan)
        fd_error = reverse_forward_error.copy()
        contact_stiffness = np.full((len(PHASE_CASES), ALPHAS.size), np.nan)

    raw_path = output_root / "action_interpolation_derivatives.npz"
    _write_npz(raw_path, raw_arrays)
    report = {
        **classification,
        "code_commit": args.code_commit,
        "source": "E-20260904-014/20260904T201953Z exact one-substep graph",
        "selected_phases": list(SELECTED_PHASES),
        "phase_cases": [list(pair) for pair in PHASE_CASES],
        "alphas": ALPHAS.tolist(),
        "baseline_replay_gate": baseline_gate,
        "input_match_to_e014": input_match,
        "case_identity_exact": case_identity,
        "hard_contact_reset_exact": hard_contact_exact,
        "reset_contact_counts": reset_contact_counts.tolist(),
        "substep_only": substep_only,
        "sweep_executed": sweeps is not None,
        "sweep_repeat_exact": sweep_repeat_exact,
        "nonselected_outputs_match_baseline": nonselected_exact,
        "endpoint_outputs_match_paired_e014_rows": endpoints_exact,
        "candidate_actions_exact": candidate_actions_exact,
        "all_selected_primals_finite": all_selected_primals_finite,
        "all_transitions_nonterminal": all_transitions_nonterminal,
        "no_initial_effort_clipping": no_initial_effort_clipping,
        "maximum_initial_effort_utilization": float(np.max(pd["effort_utilization"])),
        "minimum_initial_effort_margin": float(np.min(pd["effort_margin"])),
        "primal_valid": primal_valid.tolist(),
        "smooth_gradient_agreement": gradient_by_objective[..., 0].tolist(),
        "reward_gradient_agreement": gradient_by_objective[..., 1].tolist(),
        "smooth_finite_difference_agreement": fd_by_objective[..., 0].tolist(),
        "reward_finite_difference_agreement": fd_by_objective[..., 1].tolist(),
        "smooth_reverse_forward_relative_error": _finite_or_none(
            reverse_forward_error[..., 0]
        ),
        "reward_reverse_forward_relative_error": _finite_or_none(
            reverse_forward_error[..., 1]
        ),
        "smooth_finite_difference_relative_error": _finite_or_none(fd_error[..., 0]),
        "reward_finite_difference_relative_error": _finite_or_none(fd_error[..., 1]),
        "transition_contact_stiffness": _finite_or_none(contact_stiffness),
        "computed_baseline_probe_invocations": 2,
        "computed_interpolation_probe_invocations": (
            int(2 * len(PHASE_CASES) * ALPHAS.size) if sweeps is not None else 0
        ),
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": True,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "source_e014_raw_sha256": SOURCE_E014_RAW_SHA256,
        "raw_npz_sha256": sha256_file(raw_path),
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "action_interpolation_derivatives.png"
    if sweeps is None:
        _plot_invalid(plot_path, baseline_gate)
    else:
        _plot_result(
            plot_path,
            primal_valid=primal_valid,
            smooth_agreement=gradient_by_objective[..., 0],
            smooth_error=reverse_forward_error[..., 0],
            contact_stiffness=contact_stiffness,
            torque_utilization=np.max(pd["effort_utilization"], axis=-1),
        )
    summary = {
        **classification,
        "selected_phases": list(SELECTED_PHASES),
        "alphas": ALPHAS.tolist(),
        "transition_brackets": classification["transition_brackets"],
        "primal_valid": primal_valid.tolist(),
        "smooth_gradient_agreement": gradient_by_objective[..., 0].tolist(),
        "no_initial_effort_clipping": no_initial_effort_clipping,
        "maximum_initial_effort_utilization": float(np.max(pd["effort_utilization"])),
        "baseline_replay_gate": baseline_gate,
        "sweep_repeat_exact": sweep_repeat_exact,
        "raw_npz_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "plot_sha256": sha256_file(plot_path),
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-hard-contact-action-interpolation-completion-v1",
        "valid": bool(classification["scientifically_interpretable"]),
        "outcome": classification["outcome"],
        "computed_baseline_probe_invocations": 2,
        "computed_interpolation_probe_invocations": (
            int(2 * len(PHASE_CASES) * ALPHAS.size) if sweeps is not None else 0
        ),
        "computed_alpha_count": int(ALPHAS.size) if sweeps is not None else 0,
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": True,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "action_interpolation_derivatives.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "action_interpolation_derivatives.png": sha256_file(plot_path),
            "summary.json": sha256_file(summary_path),
        },
    }
    write_json(output_root / "completion.json", completion)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    jax.clear_caches()
    return 0 if completion["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--diffsim-hparams", type=Path, required=True)
    parser.add_argument("--source-trajectories", type=Path, required=True)
    parser.add_argument("--source-e008-audit", type=Path, required=True)
    parser.add_argument("--source-e014-raw", type=Path, required=True)
    parser.add_argument("--source-e014-report", type=Path, required=True)
    parser.add_argument("--source-e014-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def main() -> None:
    configure_jax()
    raise SystemExit(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
