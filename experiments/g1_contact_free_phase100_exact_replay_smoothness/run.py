"""Replay E011's exact compiled graph for the phase-100 smoothness check.

E012 was invalid because putting its base and perturbations into a new batched
callable changed the unperturbed scalar.  This successor builds E011's exact
full ten-case compiled probe once.  It first requires two complete baseline
outputs to match E011's persisted arrays bit-for-bit.  Only after that gate
passes does it change case nine's action in the otherwise unchanged ten-case
batch and read that case's source primal on a frozen two-sided epsilon grid.
The probe necessarily computes its original derivative payload on every call,
but no perturbed derivative is interpreted and no policy is evaluated.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import mujoco

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from experiments.g1_contact_free_phase100_smoothness.run import (
    CASE_ARM,
    CASE_INDEX,
    CASE_PHASE,
    EPSILONS,
    FINE_EPSILONS,
    SOURCE_E011_AUDIT_SHA256,
    SOURCE_E011_RAW_SHA256,
    SOURCE_E011_REPORT_SHA256,
    STABILITY_ATOL,
    STABILITY_RTOL,
    _extract_e011_input,
    _inputs_exact,
    _load_npz,
    _plot_sweep,
    _validate_e011_sources,
    classify_slope_limit,
    directional_slopes,
)
from experiments.g1_reset_action_derivative_discriminator.run import (
    ACTION_DIMENSION,
    CASE_COUNT,
    FINITE_DIFFERENCE_ATOL,
    FINITE_DIFFERENCE_RTOL,
    OBJECTIVE_NAMES,
    PRIMAL_ATOL,
    PRIMAL_RTOL,
    _arrays_exact,
    _build_compiled_probe,
    _load_source_arrays,
    _prepare_cases,
    _write_npz,
    build_common_probe_env,
)
from experiments.g1_reset_contact_derivative_discriminator.run import (
    disable_contact_dynamics,
)
from experiments.g1_success_failure_visitation.run import (
    read_json,
    repository_preflight,
    sha256_file,
    validate_diffsim_hparams,
    write_json,
)
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.run_g1_tracking_shac import configure_jax


REFERENCE_SHA256 = "f47d13b431d85a273eba6022f5a28bd55cae7c788112baf0778ab159914a039c"
DIFFSIM_HPARAMS_SHA256 = (
    "79927f89ef75cf0a6fbfd5c92746a59db587c00319db780dcad702f0c3bbd5eb"
)
SOURCE_TRAJECTORY_SHA256 = (
    "dc4199fa5383e7caf31c89bb56c7d261af6561ce237d48e8e217276827dbc89b"
)
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


def _extract_e011_probe_output(
    raw: Mapping[str, np.ndarray], invocation: str
) -> dict[str, np.ndarray]:
    """Extract one complete persisted E011 treatment invocation."""

    if invocation not in {"first", "second"}:
        raise ValueError("E011 invocation must be first or second")
    prefix = f"treatment_{invocation}_"
    keys = {f"{prefix}{name}" for name in PROBE_OUTPUT_NAMES}
    if not keys.issubset(raw):
        raise ValueError(f"complete E011 {invocation} probe output is missing")
    return {name: np.asarray(raw[f"{prefix}{name}"]) for name in PROBE_OUTPUT_NAMES}


def baseline_replay_gate(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    expected_first: Mapping[str, np.ndarray],
    expected_second: Mapping[str, np.ndarray],
) -> dict[str, bool]:
    """Require the new executable to reproduce both complete E011 outputs."""

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
        "first_matches_e011": bool(first_matches),
        "second_matches_e011": bool(second_matches),
        "valid": bool(
            output_names_exact and repeat_exact and first_matches and second_matches
        ),
    }


def perturb_case_actions(
    actions: object,
    direction: object,
    *,
    epsilon: float,
    sign: int,
):
    """Change only E011 case nine in the complete ten-case action batch."""

    if sign not in {-1, 1}:
        raise ValueError("perturbation sign must be -1 or 1")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    action_array = jnp.asarray(actions)
    direction_array = jnp.asarray(direction, dtype=action_array.dtype)
    if action_array.shape != (CASE_COUNT, ACTION_DIMENSION):
        raise ValueError("complete action batch has the wrong shape")
    if direction_array.shape != (ACTION_DIMENSION,):
        raise ValueError("action direction has the wrong shape")
    return action_array.at[CASE_INDEX].set(
        action_array[CASE_INDEX] + sign * epsilon_value * direction_array
    )


def _invoke_full_probe(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    states: object,
    actions: object,
) -> dict[str, np.ndarray]:
    device_result = compiled_probe(states, actions)
    jax.block_until_ready(device_result)
    return {name: np.asarray(value) for name, value in device_result.items()}


def _invoke_case_primal(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    states: object,
    actions: object,
) -> tuple[np.ndarray, float, float]:
    device_result = compiled_probe(states, actions)
    jax.block_until_ready(device_result)
    required = {"source_primal", "direct_done", "direct_terminal"}
    if not required.issubset(device_result):
        raise ValueError("exact probe lacks required transition outputs")
    source_primal = np.asarray(device_result["source_primal"], dtype=np.float64)
    done = np.asarray(device_result["direct_done"], dtype=np.float64)
    terminal = np.asarray(device_result["direct_terminal"], dtype=np.float64)
    if source_primal.shape != (CASE_COUNT, len(OBJECTIVE_NAMES)):
        raise ValueError("exact probe source primal has the wrong shape")
    if done.shape != (CASE_COUNT,) or terminal.shape != (CASE_COUNT,):
        raise ValueError("exact probe terminal arrays have the wrong shape")
    return (
        source_primal[CASE_INDEX],
        float(done[CASE_INDEX]),
        float(terminal[CASE_INDEX]),
    )


def execute_perturbation_sweeps(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    *,
    states: object,
    actions: object,
    direction: object,
    epsilons: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    """Run the fixed signed grid twice through one already-compiled probe."""

    epsilon_array = np.asarray(epsilons, dtype=np.float64)
    if (
        epsilon_array.ndim != 1
        or epsilon_array.size == 0
        or not np.all(np.isfinite(epsilon_array))
        or not np.all(epsilon_array > 0.0)
    ):
        raise ValueError("epsilon grid is invalid")
    repeats: list[dict[str, np.ndarray]] = []
    for _ in range(2):
        plus_primal = []
        minus_primal = []
        plus_done = []
        minus_done = []
        plus_terminal = []
        minus_terminal = []
        for epsilon in epsilon_array:
            plus = perturb_case_actions(
                actions, direction, epsilon=float(epsilon), sign=1
            )
            minus = perturb_case_actions(
                actions, direction, epsilon=float(epsilon), sign=-1
            )
            primal, done, terminal = _invoke_case_primal(compiled_probe, states, plus)
            plus_primal.append(primal)
            plus_done.append(done)
            plus_terminal.append(terminal)
            primal, done, terminal = _invoke_case_primal(compiled_probe, states, minus)
            minus_primal.append(primal)
            minus_done.append(done)
            minus_terminal.append(terminal)
        repeats.append(
            {
                "plus_primal": np.asarray(plus_primal, dtype=np.float64),
                "minus_primal": np.asarray(minus_primal, dtype=np.float64),
                "plus_done": np.asarray(plus_done, dtype=np.float64),
                "minus_done": np.asarray(minus_done, dtype=np.float64),
                "plus_terminal": np.asarray(plus_terminal, dtype=np.float64),
                "minus_terminal": np.asarray(minus_terminal, dtype=np.float64),
            }
        )
    repeat_exact = _arrays_exact(repeats[0], repeats[1])
    return {
        **{f"first_{name}": value for name, value in repeats[0].items()},
        **{f"second_{name}": value for name, value in repeats[1].items()},
        "repeat_exact": repeat_exact,
    }


def _invalid_classification(ad_slope: float) -> dict[str, object]:
    return {
        "protocol": "g1-contact-free-slope-limit-classification-v1",
        "valid": False,
        "scientifically_interpretable": False,
        "outcome": "invalid-measurement",
        "selected_side": None,
        "ad_slope": float(ad_slope),
        "fine_epsilons": FINE_EPSILONS.tolist(),
        "fine_positive_stable": False,
        "fine_negative_stable": False,
        "fine_central_stable": False,
        "fine_positive_matches_ad": False,
        "fine_negative_matches_ad": False,
        "fine_central_matches_ad": False,
        "fine_sides_separated": False,
        "fine_positive_median": None,
        "fine_negative_median": None,
        "fine_central_median": None,
    }


def _case_source_primal_or_nan(result: Mapping[str, object]) -> np.ndarray:
    """Return case nine's two primals or NaNs for a malformed invalid replay."""

    values = np.asarray(
        result.get("source_primal", np.empty((0,), dtype=np.float64)),
        dtype=np.float64,
    )
    if values.shape != (CASE_COUNT, len(OBJECTIVE_NAMES)):
        return np.full((len(OBJECTIVE_NAMES),), np.nan, dtype=np.float64)
    return values[CASE_INDEX]


def _finite_scalar_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _finite_or_none(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _plot_invalid_baseline(path: Path, gate: Mapping[str, bool]) -> None:
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.axis("off")
    lines = ["Exact E011 baseline replay failed; perturbation sweep not executed."]
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
        "source_e011_raw": args.source_e011_raw.resolve(),
        "source_e011_report": args.source_e011_report.resolve(),
        "source_e011_audit": args.source_e011_audit.resolve(),
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "diffsim_hparams": DIFFSIM_HPARAMS_SHA256,
        "source_trajectories": SOURCE_TRAJECTORY_SHA256,
        "source_e011_raw": SOURCE_E011_RAW_SHA256,
        "source_e011_report": SOURCE_E011_REPORT_SHA256,
        "source_e011_audit": SOURCE_E011_AUDIT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")
    if not jax.config.x64_enabled:
        raise ValueError("exact E011 replay requires JAX x64")

    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    source_raw = _load_npz(paths["source_e011_raw"])
    source = _extract_e011_input(source_raw)
    expected_first = _extract_e011_probe_output(source_raw, "first")
    expected_second = _extract_e011_probe_output(source_raw, "second")
    _validate_e011_sources(
        read_json(paths["source_e011_report"]),
        read_json(paths["source_e011_audit"]),
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-contact-free-phase100-exact-replay-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "seed": args.seed,
        "case_index": CASE_INDEX,
        "phase": CASE_PHASE,
        "arm": CASE_ARM,
        "case_count": CASE_COUNT,
        "action_dimension": ACTION_DIMENSION,
        "objectives": list(OBJECTIVE_NAMES),
        "probe_output_names": list(PROBE_OUTPUT_NAMES),
        "epsilons": EPSILONS.tolist(),
        "fine_epsilons": FINE_EPSILONS.tolist(),
        "finite_difference_tolerances": {
            "rtol": FINITE_DIFFERENCE_RTOL,
            "atol": FINITE_DIFFERENCE_ATOL,
        },
        "stability_tolerances": {"rtol": STABILITY_RTOL, "atol": STABILITY_ATOL},
        "solver_profile": args.solver_profile,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed_as_part_of_exact_probe": True,
        "perturbed_derivatives_interpreted": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    preflight_path = output_root / "preflight.json"
    write_json(preflight_path, preflight)

    source_arrays = _load_source_arrays(paths["source_trajectories"])
    with solver_context(get_solver_profile(args.solver_profile)):
        env = build_common_probe_env(paths["reference"], hparams)
        original_flags = int(env.mjx_model.opt.disableflags)
        disable_contact_dynamics(env)
        treated_flags = int(env.mjx_model.opt.disableflags)
        states, actions, metadata = _prepare_cases(env, source_arrays, seed=args.seed)
        direction = jnp.asarray(source["direction"], dtype=jnp.float64)
        reset_contact_signatures = np.asarray(
            jax.vmap(env.contact_pair_signature)(states.data), dtype=bool
        )
        compiled_probe = _build_compiled_probe(env, direction)
        baseline_first = _invoke_full_probe(compiled_probe, states, actions)
        baseline_second = _invoke_full_probe(compiled_probe, states, actions)
        baseline_gate = baseline_replay_gate(
            baseline_first,
            baseline_second,
            expected_first,
            expected_second,
        )
        sweeps = None
        if baseline_gate["valid"]:
            sweeps = execute_perturbation_sweeps(
                compiled_probe,
                states=states,
                actions=actions,
                direction=direction,
                epsilons=EPSILONS,
            )

    contact_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    contact_flag_only = bool(
        original_flags & contact_bit == 0
        and treated_flags == original_flags | contact_bit
        and int(env.mj_model.opt.disableflags) == treated_flags
    )
    contact_absent = bool(contact_flag_only and not np.any(reset_contact_signatures))
    input_match = _inputs_exact(source, metadata)
    case_identity = bool(
        int(metadata["phases"][CASE_INDEX]) == CASE_PHASE
        and metadata["arms"][CASE_INDEX] == CASE_ARM
    )
    action_array = np.asarray(actions, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    perturbed_case_actions = np.concatenate(
        (
            action_array[CASE_INDEX][None, :]
            + EPSILONS[:, None] * direction_array[None, :],
            action_array[CASE_INDEX][None, :]
            - EPSILONS[:, None] * direction_array[None, :],
        )
    )
    actions_in_bounds = bool(np.max(np.abs(perturbed_case_actions)) < 1.0)
    baseline_primal = _case_source_primal_or_nan(baseline_first)
    source_primal = np.asarray(source["source_primal"], dtype=np.float64)[CASE_INDEX]
    source_ad_slope = np.asarray(source["ad_slope"], dtype=np.float64)[CASE_INDEX]

    raw_arrays: dict[str, np.ndarray] = {
        **{f"input_{name}": value for name, value in metadata.items()},
        "direction": direction_array,
        "reset_contact_signatures": reset_contact_signatures,
        "epsilons": EPSILONS,
        "fine_epsilons": FINE_EPSILONS,
        **{f"baseline_first_{name}": value for name, value in baseline_first.items()},
        **{f"baseline_second_{name}": value for name, value in baseline_second.items()},
    }

    objective_reports: dict[str, object] = {}
    classifications: list[dict[str, object]] = []
    sweep_repeat_exact = False
    all_values_finite = False
    all_transitions_nonterminal = False
    measurement_valid = False
    if sweeps is not None:
        sweep_repeat_exact = bool(sweeps["repeat_exact"])
        plus = np.asarray(sweeps["first_plus_primal"], dtype=np.float64)
        minus = np.asarray(sweeps["first_minus_primal"], dtype=np.float64)
        all_values = np.concatenate((baseline_primal[None, :], plus, minus))
        all_values_finite = bool(np.all(np.isfinite(all_values)))
        terminal_keys = (
            "first_plus_done",
            "first_minus_done",
            "first_plus_terminal",
            "first_minus_terminal",
            "second_plus_done",
            "second_minus_done",
            "second_plus_terminal",
            "second_minus_terminal",
        )
        all_transitions_nonterminal = all(
            bool(np.all(np.asarray(sweeps[name]) == 0.0)) for name in terminal_keys
        )
        measurement_valid = bool(
            baseline_gate["valid"]
            and input_match
            and case_identity
            and contact_absent
            and actions_in_bounds
            and sweep_repeat_exact
            and all_values_finite
            and all_transitions_nonterminal
            and np.all(
                np.isclose(
                    baseline_primal,
                    source_primal,
                    rtol=PRIMAL_RTOL,
                    atol=PRIMAL_ATOL,
                )
            )
        )
        raw_arrays.update(
            {
                name: np.asarray(value)
                for name, value in sweeps.items()
                if name != "repeat_exact"
            }
        )
        for index, objective in enumerate(OBJECTIVE_NAMES):
            slopes = directional_slopes(
                float(baseline_primal[index]), plus[:, index], minus[:, index], EPSILONS
            )
            classification = (
                classify_slope_limit(
                    measurement_valid=True,
                    ad_slope=float(source_ad_slope[index]),
                    **slopes,
                )
                if measurement_valid
                else _invalid_classification(float(source_ad_slope[index]))
            )
            classifications.append(classification)
            objective_reports[objective] = {
                **classification,
                "base": _finite_scalar_or_none(baseline_primal[index]),
                "source_primal": float(source_primal[index]),
                "base_matches_e011_exactly": bool(
                    np.array_equal(baseline_primal[index], source_primal[index])
                ),
                "plus_values": _finite_or_none(plus[:, index]),
                "minus_values": _finite_or_none(minus[:, index]),
                **{
                    f"{name}_slope": _finite_or_none(values)
                    for name, values in slopes.items()
                },
            }
            raw_arrays.update(
                {f"{objective}_{name}_slope": values for name, values in slopes.items()}
            )
    else:
        classifications = [
            _invalid_classification(float(value)) for value in source_ad_slope
        ]
        objective_reports = {
            objective: {
                **classifications[index],
                "base": _finite_scalar_or_none(baseline_primal[index]),
                "source_primal": float(source_primal[index]),
                "base_matches_e011_exactly": bool(
                    np.array_equal(baseline_primal[index], source_primal[index])
                ),
                "plus_values": [],
                "minus_values": [],
                "positive_slope": [],
                "negative_slope": [],
                "central_slope": [],
            }
            for index, objective in enumerate(OBJECTIVE_NAMES)
        }

    raw_path = output_root / "exact_replay_smoothness.npz"
    _write_npz(raw_path, raw_arrays)
    primary = classifications[0]
    report = {
        "protocol": "g1-contact-free-phase100-exact-replay-report-v1",
        "valid": measurement_valid,
        "scientifically_interpretable": primary["scientifically_interpretable"],
        "outcome": primary["outcome"],
        "code_commit": args.code_commit,
        "source": "E-20260904-011/20260904T181644Z complete treatment graph, case 9",
        "case_index": CASE_INDEX,
        "phase": CASE_PHASE,
        "arm": CASE_ARM,
        "epsilons": EPSILONS.tolist(),
        "fine_epsilons": FINE_EPSILONS.tolist(),
        "objectives": objective_reports,
        "baseline_replay_gate": baseline_gate,
        "input_match_to_e011": input_match,
        "case_identity_exact": case_identity,
        "original_disableflags": original_flags,
        "treated_disableflags": treated_flags,
        "contact_disable_bit": contact_bit,
        "contact_flag_only": contact_flag_only,
        "reset_active_contact_count": int(np.sum(reset_contact_signatures)),
        "all_perturbed_actions_strictly_in_bounds": actions_in_bounds,
        "maximum_absolute_perturbed_action": float(
            np.max(np.abs(perturbed_case_actions))
        ),
        "sweep_executed": sweeps is not None,
        "sweep_repeat_exact": sweep_repeat_exact,
        "all_objective_values_finite": all_values_finite,
        "all_transitions_nonterminal": all_transitions_nonterminal,
        "computed_baseline_probe_invocations": 2,
        "computed_perturbation_probe_invocations": (
            int(4 * EPSILONS.size) if sweeps is not None else 0
        ),
        "derivative_computed_as_part_of_exact_probe": True,
        "perturbed_derivatives_interpreted": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "source_e011_raw_sha256": SOURCE_E011_RAW_SHA256,
        "raw_npz_sha256": sha256_file(raw_path),
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "exact_replay_smoothness.png"
    if sweeps is None:
        _plot_invalid_baseline(plot_path, baseline_gate)
    else:
        _plot_sweep(plot_path, report)
    summary = {
        "protocol": "g1-contact-free-phase100-exact-replay-summary-v1",
        "valid": measurement_valid,
        "scientifically_interpretable": primary["scientifically_interpretable"],
        "outcome": primary["outcome"],
        "phase": CASE_PHASE,
        "arm": CASE_ARM,
        "baseline_replay_gate": baseline_gate,
        "sweep_executed": sweeps is not None,
        "smooth_classification": primary,
        "reward_classification": classifications[1],
        "raw_npz_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "plot_sha256": sha256_file(plot_path),
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-contact-free-phase100-exact-replay-completion-v1",
        "valid": bool(primary["scientifically_interpretable"]),
        "outcome": primary["outcome"],
        "baseline_replay_gate_passed": baseline_gate["valid"],
        "computed_baseline_probe_invocations": 2,
        "computed_perturbation_probe_invocations": (
            int(4 * EPSILONS.size) if sweeps is not None else 0
        ),
        "computed_epsilon_count": int(EPSILONS.size) if sweeps is not None else 0,
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed_as_part_of_exact_probe": True,
        "perturbed_derivatives_interpreted": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "exact_replay_smoothness.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "exact_replay_smoothness.png": sha256_file(plot_path),
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
    parser.add_argument("--source-e011-raw", type=Path, required=True)
    parser.add_argument("--source-e011-report", type=Path, required=True)
    parser.add_argument("--source-e011-audit", type=Path, required=True)
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
