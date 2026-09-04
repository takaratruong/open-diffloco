"""Resolve E011's contact-free phase-100 DiffSim finite-difference outlier.

This diagnostic reuses E011's audited action directional derivative and changes
no policy or optimizer state.  It reconstructs the same contact-disabled state
and action, evaluates a frozen logarithmic grid of positive and negative action
perturbations, and reports one-sided and central slopes without selecting an
epsilon after observing the result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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

from experiments.g1_reset_action_derivative_discriminator.run import (
    FINITE_DIFFERENCE_ATOL,
    FINITE_DIFFERENCE_RTOL,
    OBJECTIVE_NAMES,
    PRIMAL_ATOL,
    PRIMAL_RTOL,
    _arrays_exact,
    _load_source_arrays,
    _prepare_cases,
    _write_npz,
    build_common_probe_env,
    smooth_reference_state_loss,
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
SOURCE_E011_RAW_SHA256 = (
    "168d63ff45e61a422944f2018942f2a164993117e8489c9819f5ee9773649052"
)
SOURCE_E011_REPORT_SHA256 = (
    "00c4e99e76164e985f7f70636da0c84de442c61bf365cb90aef4f72c8a3aa641"
)
SOURCE_E011_AUDIT_SHA256 = (
    "8bc8a6c2ee35370ea544651561fb83aa905fe687254a520c476926efcbb838f0"
)
CASE_INDEX = 9
CASE_PHASE = 100
CASE_ARM = "diffsim"
EPSILONS = np.asarray(
    [
        1e-2,
        3e-3,
        1e-3,
        3e-4,
        1e-4,
        3e-5,
        1e-5,
        3e-6,
        1e-6,
        3e-7,
        1e-7,
    ],
    dtype=np.float64,
)
FINE_EPSILONS = EPSILONS[-3:].copy()
STABILITY_RTOL = 2e-2
STABILITY_ATOL = 1e-4


def directional_slopes(
    base: float,
    plus: np.ndarray,
    minus: np.ndarray,
    epsilons: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return positive, negative-side, and central slopes in +direction."""

    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    epsilons = np.asarray(epsilons, dtype=np.float64)
    if (
        plus.shape != epsilons.shape
        or minus.shape != epsilons.shape
        or epsilons.ndim != 1
        or not np.all(np.isfinite(epsilons))
        or not np.all(epsilons > 0.0)
    ):
        raise ValueError("finite-difference values or epsilon grid are invalid")
    return {
        "positive": (plus - base) / epsilons,
        "negative": (base - minus) / epsilons,
        "central": (plus - minus) / (2.0 * epsilons),
    }


def _fine_stable(values: np.ndarray) -> tuple[bool, float]:
    fine = np.asarray(values, dtype=np.float64)[-FINE_EPSILONS.size :]
    median = float(np.median(fine))
    stable = bool(
        np.all(np.isfinite(fine))
        and np.all(
            np.isclose(
                fine,
                median,
                rtol=STABILITY_RTOL,
                atol=STABILITY_ATOL,
            )
        )
    )
    return stable, median


def _fine_matches_ad(values: np.ndarray, ad_slope: float) -> bool:
    fine = np.asarray(values, dtype=np.float64)[-FINE_EPSILONS.size :]
    return bool(
        math.isfinite(ad_slope)
        and np.all(np.isfinite(fine))
        and np.all(
            np.isclose(
                fine,
                ad_slope,
                rtol=FINITE_DIFFERENCE_RTOL,
                atol=FINITE_DIFFERENCE_ATOL,
            )
        )
    )


def classify_slope_limit(
    *,
    measurement_valid: bool,
    ad_slope: float,
    positive: np.ndarray,
    negative: np.ndarray,
    central: np.ndarray,
) -> dict[str, object]:
    """Classify a frozen fine-epsilon window without choosing a scale."""

    arrays = {
        "positive": np.asarray(positive, dtype=np.float64),
        "negative": np.asarray(negative, dtype=np.float64),
        "central": np.asarray(central, dtype=np.float64),
    }
    if any(values.shape != EPSILONS.shape for values in arrays.values()):
        raise ValueError("slope arrays do not match the frozen epsilon grid")
    stable_and_median = {name: _fine_stable(values) for name, values in arrays.items()}
    stable = {name: value[0] for name, value in stable_and_median.items()}
    median = {name: value[1] for name, value in stable_and_median.items()}
    matches = {
        name: _fine_matches_ad(values, ad_slope) for name, values in arrays.items()
    }
    sides_separated = bool(
        not np.isclose(
            median["positive"],
            median["negative"],
            rtol=FINITE_DIFFERENCE_RTOL,
            atol=FINITE_DIFFERENCE_ATOL,
        )
    )
    selected_side = None
    if not measurement_valid:
        outcome = "invalid-measurement"
        interpretable = False
    elif matches["positive"] and matches["negative"] and matches["central"]:
        outcome = "fine-window-ad-consistent"
        interpretable = True
    elif stable["positive"] and stable["negative"] and sides_separated:
        if matches["positive"] != matches["negative"]:
            outcome = "ad-selects-one-sided-branch"
            selected_side = "positive" if matches["positive"] else "negative"
        else:
            outcome = "one-sided-directional-kink"
        interpretable = True
    elif (
        stable["positive"]
        and stable["negative"]
        and stable["central"]
        and not sides_separated
    ):
        outcome = "stable-fd-limit-disagrees-with-ad"
        interpretable = True
    else:
        outcome = "no-stable-fd-limit"
        interpretable = True
    return {
        "protocol": "g1-contact-free-slope-limit-classification-v1",
        "valid": bool(measurement_valid),
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "selected_side": selected_side,
        "ad_slope": float(ad_slope),
        "fine_epsilons": FINE_EPSILONS.tolist(),
        "fine_positive_stable": stable["positive"],
        "fine_negative_stable": stable["negative"],
        "fine_central_stable": stable["central"],
        "fine_positive_matches_ad": matches["positive"],
        "fine_negative_matches_ad": matches["negative"],
        "fine_central_matches_ad": matches["central"],
        "fine_sides_separated": sides_separated,
        "fine_positive_median": median["positive"],
        "fine_negative_median": median["negative"],
        "fine_central_median": median["central"],
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_e011_sources(
    report: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    expected_audit = {
        "protocol": "g1-reset-contact-derivative-independent-audit-v1",
        "experiment": "E-20260904-011",
        "valid": True,
        "measurement_valid": True,
        "scientifically_interpretable": True,
        "outcome": "contact-removes-ad-disagreement-finite-difference-unresolved",
        "checks_passed": 21,
        "checks_total": 21,
        "contact_absent": True,
        "control_smooth_pass_count": 0,
        "treatment_smooth_pass_count": 9,
        "treatment_gradient_agreement_count": 10,
        "treatment_finite_difference_agreement_count": 9,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    mismatches = {
        name: (audit.get(name), expected)
        for name, expected in expected_audit.items()
        if audit.get(name) != expected
    }
    treatment = report.get("treatment", {})
    smooth = treatment.get("smooth_reference_state", {})
    reward = treatment.get("e002_h1_reward", {})
    if mismatches:
        raise ValueError(f"E011 audit contract changed: {mismatches}")
    if (
        report.get("protocol") != "g1-reset-contact-derivative-classification-v1"
        or report.get("outcome")
        != "contact-removes-ad-disagreement-finite-difference-unresolved"
        or report.get("raw_npz_sha256") != SOURCE_E011_RAW_SHA256
        or smooth.get("case_pass") != [True] * CASE_INDEX + [False]
        or reward.get("case_pass") != [True] * CASE_INDEX + [False]
    ):
        raise ValueError("E011 persisted report contract changed")


def _extract_e011_input(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    shared = (
        "phases",
        "arms",
        "actor_actions",
        "model_actions",
        "position_targets",
        "reset_qpos",
        "reset_qvel",
        "reset_qpos_max_abs_delta",
    )
    required = {
        *(f"treatment_{name}" for name in shared),
        "treatment_direction",
        "treatment_first_source_primal",
        "treatment_first_forward_directional",
        "treatment_second_source_primal",
        "treatment_second_forward_directional",
    }
    if not required.issubset(raw):
        raise ValueError("E011 raw archive lacks required treatment arrays")
    return {
        **{name: np.asarray(raw[f"treatment_{name}"]) for name in shared},
        "direction": np.asarray(raw["treatment_direction"], dtype=np.float64),
        "source_primal": np.asarray(
            raw["treatment_first_source_primal"], dtype=np.float64
        ),
        "ad_slope": np.asarray(
            raw["treatment_first_forward_directional"], dtype=np.float64
        ),
        "second_source_primal": np.asarray(
            raw["treatment_second_source_primal"], dtype=np.float64
        ),
        "second_ad_slope": np.asarray(
            raw["treatment_second_forward_directional"], dtype=np.float64
        ),
    }


def _inputs_exact(
    source: Mapping[str, np.ndarray],
    metadata: Mapping[str, np.ndarray],
) -> bool:
    shared = (
        "phases",
        "arms",
        "actor_actions",
        "model_actions",
        "position_targets",
        "reset_qpos",
        "reset_qvel",
        "reset_qpos_max_abs_delta",
    )
    return all(
        np.array_equal(source[name], metadata[name]) for name in shared
    ) and np.array_equal(source["source_primal"], source["second_source_primal"])


def _build_compiled_sweep(env: object, direction: jax.Array, epsilons: jax.Array):
    def objectives(state, action):
        next_state = env.step(state, action)
        next_phase = jnp.minimum(
            state.info["phase"] + env.reference_stride,
            env.reference_length - 1,
        )
        smooth = smooth_reference_state_loss(
            next_state.data.qpos,
            next_state.data.qvel,
            env.qpos_reference[next_phase],
            env.qvel_reference[next_phase],
        )
        return (
            jnp.stack((smooth, -next_state.reward)),
            next_state.done,
            next_state.info["terminal"],
        )

    def sweep(state, action):
        base, base_done, base_terminal = objectives(state, action)

        def at_epsilon(epsilon):
            plus, plus_done, plus_terminal = objectives(
                state, action + epsilon * direction
            )
            minus, minus_done, minus_terminal = objectives(
                state, action - epsilon * direction
            )
            return (
                plus,
                minus,
                plus_done,
                minus_done,
                plus_terminal,
                minus_terminal,
            )

        (
            plus,
            minus,
            plus_done,
            minus_done,
            plus_terminal,
            minus_terminal,
        ) = jax.vmap(at_epsilon)(epsilons)
        return {
            "base": base,
            "plus": plus,
            "minus": minus,
            "base_done": base_done,
            "base_terminal": base_terminal,
            "plus_done": plus_done,
            "minus_done": minus_done,
            "plus_terminal": plus_terminal,
            "minus_terminal": minus_terminal,
        }

    return jax.jit(sweep)


def _plot_sweep(path: Path, report: Mapping[str, object]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    epsilons = np.asarray(report["epsilons"], dtype=np.float64)
    for row, objective in enumerate(OBJECTIVE_NAMES):
        result = report["objectives"][objective]
        ad_slope = float(result["ad_slope"])
        for name, marker in (
            ("positive", "o"),
            ("negative", "s"),
            ("central", "^"),
        ):
            values = np.asarray(result[f"{name}_slope"], dtype=np.float64)
            axes[row, 0].plot(epsilons, values, marker=marker, label=name)
            axes[row, 1].plot(
                epsilons,
                np.maximum(np.abs(values - ad_slope), 1e-16),
                marker=marker,
                label=name,
            )
        axes[row, 0].axhline(ad_slope, color="black", linestyle="--", label="AD")
        for column in range(2):
            axes[row, column].axvline(1e-5, color="gray", linestyle=":")
            axes[row, column].set_xscale("log")
            axes[row, column].invert_xaxis()
            axes[row, column].grid(alpha=0.25)
            axes[row, column].set_xlabel("epsilon")
        axes[row, 0].set_ylabel("directional slope")
        axes[row, 0].set_title(f"{objective}: one-sided and central slopes")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylabel("absolute error from E011 AD")
        axes[row, 1].set_title(f"{objective}: error from AD")
        axes[row, 0].legend()
        axes[row, 1].legend()
    figure.suptitle("Contact-free phase-100 DiffSim fixed-grid directional smoothness")
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
        raise ValueError("phase-100 smoothness diagnostic requires JAX x64")
    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    source_raw = _load_npz(paths["source_e011_raw"])
    source = _extract_e011_input(source_raw)
    _validate_e011_sources(
        read_json(paths["source_e011_report"]),
        read_json(paths["source_e011_audit"]),
    )
    if not np.array_equal(source["ad_slope"], source["second_ad_slope"]):
        raise ValueError("E011 repeated AD slopes changed")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-contact-free-phase100-smoothness-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "seed": args.seed,
        "case_index": CASE_INDEX,
        "phase": CASE_PHASE,
        "arm": CASE_ARM,
        "objectives": list(OBJECTIVE_NAMES),
        "epsilons": EPSILONS.tolist(),
        "fine_epsilons": FINE_EPSILONS.tolist(),
        "finite_difference_tolerances": {
            "rtol": FINITE_DIFFERENCE_RTOL,
            "atol": FINITE_DIFFERENCE_ATOL,
        },
        "stability_tolerances": {
            "rtol": STABILITY_RTOL,
            "atol": STABILITY_ATOL,
        },
        "solver_profile": args.solver_profile,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "new_ad_derivative_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    preflight_path = output_root / "preflight.json"
    write_json(preflight_path, preflight)

    arrays = _load_source_arrays(paths["source_trajectories"])
    with solver_context(get_solver_profile(args.solver_profile)):
        env = build_common_probe_env(paths["reference"], hparams)
        original_flags = int(env.mjx_model.opt.disableflags)
        disable_contact_dynamics(env)
        treated_flags = int(env.mjx_model.opt.disableflags)
        states, actions, metadata = _prepare_cases(env, arrays, seed=args.seed)
        input_match = _inputs_exact(source, metadata)
        state = jax.tree_util.tree_map(lambda value: value[CASE_INDEX], states)
        action = actions[CASE_INDEX]
        reset_contacts = np.asarray(env.contact_pair_signature(state.data), dtype=bool)
        direction = jnp.asarray(source["direction"], dtype=jnp.float64)
        compiled = _build_compiled_sweep(
            env, direction, jnp.asarray(EPSILONS, dtype=jnp.float64)
        )
        first_device = compiled(state, action)
        jax.block_until_ready(first_device)
        second_device = compiled(state, action)
        jax.block_until_ready(second_device)

    first = {name: np.asarray(value) for name, value in first_device.items()}
    second = {name: np.asarray(value) for name, value in second_device.items()}
    repeat_exact = _arrays_exact(first, second)
    contact_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    contact_flag_only = bool(
        original_flags & contact_bit == 0
        and treated_flags == original_flags | contact_bit
        and int(env.mj_model.opt.disableflags) == treated_flags
    )
    contact_absent = bool(contact_flag_only and not np.any(reset_contacts))
    action_array = np.asarray(action, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    perturbed = np.concatenate(
        (
            action_array[None, :] + EPSILONS[:, None] * direction_array[None, :],
            action_array[None, :] - EPSILONS[:, None] * direction_array[None, :],
        ),
        axis=0,
    )
    actions_in_bounds = bool(np.max(np.abs(perturbed)) < 1.0)
    base = np.asarray(first["base"], dtype=np.float64)
    plus = np.asarray(first["plus"], dtype=np.float64)
    minus = np.asarray(first["minus"], dtype=np.float64)
    source_primal = np.asarray(source["source_primal"], dtype=np.float64)[CASE_INDEX]
    source_ad_slope = np.asarray(source["ad_slope"], dtype=np.float64)[CASE_INDEX]
    base_close = np.isclose(base, source_primal, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL)
    transition_values = np.concatenate((base.reshape(1, -1), plus, minus), axis=0)
    done_terminal_arrays = [
        first[name]
        for name in (
            "base_done",
            "base_terminal",
            "plus_done",
            "minus_done",
            "plus_terminal",
            "minus_terminal",
        )
    ]
    all_nonterminal = bool(
        all(
            np.all(np.asarray(values, dtype=np.float64) == 0.0)
            for values in done_terminal_arrays
        )
    )
    measurement_valid = bool(
        input_match
        and repeat_exact
        and contact_absent
        and actions_in_bounds
        and np.all(np.isfinite(transition_values))
        and np.all(base_close)
        and all_nonterminal
        and np.array_equal(metadata["phases"][CASE_INDEX], np.asarray(CASE_PHASE))
        and metadata["arms"][CASE_INDEX] == CASE_ARM
    )

    objective_reports: dict[str, object] = {}
    classifications = []
    raw_arrays: dict[str, np.ndarray] = {
        "epsilons": EPSILONS,
        "fine_epsilons": FINE_EPSILONS,
        "action": action_array,
        "direction": direction_array,
        "reset_qpos": np.asarray(state.data.qpos, dtype=np.float64),
        "reset_qvel": np.asarray(state.data.qvel, dtype=np.float64),
        "source_primal": source_primal,
        "source_ad_slope": source_ad_slope,
        **{f"first_{name}": value for name, value in first.items()},
        **{f"second_{name}": value for name, value in second.items()},
    }
    for index, objective in enumerate(OBJECTIVE_NAMES):
        slopes = directional_slopes(
            float(base[index]), plus[:, index], minus[:, index], EPSILONS
        )
        classification = classify_slope_limit(
            measurement_valid=measurement_valid,
            ad_slope=float(source_ad_slope[index]),
            **slopes,
        )
        classifications.append(classification)
        objective_reports[objective] = {
            **classification,
            "base": float(base[index]),
            "source_primal": float(source_primal[index]),
            "base_matches_e011": bool(base_close[index]),
            "plus_values": plus[:, index].tolist(),
            "minus_values": minus[:, index].tolist(),
            **{f"{name}_slope": values.tolist() for name, values in slopes.items()},
        }
        raw_arrays.update(
            {f"{objective}_{name}_slope": values for name, values in slopes.items()}
        )

    raw_path = output_root / "smoothness_grid.npz"
    _write_npz(raw_path, raw_arrays)
    primary = classifications[0]
    report = {
        "protocol": "g1-contact-free-phase100-smoothness-report-v1",
        "valid": measurement_valid,
        "scientifically_interpretable": primary["scientifically_interpretable"],
        "outcome": primary["outcome"],
        "code_commit": args.code_commit,
        "source": "E-20260904-011/20260904T181644Z case 9",
        "case_index": CASE_INDEX,
        "phase": CASE_PHASE,
        "arm": CASE_ARM,
        "epsilons": EPSILONS.tolist(),
        "fine_epsilons": FINE_EPSILONS.tolist(),
        "objectives": objective_reports,
        "input_match_to_e011": input_match,
        "repeat_exact": repeat_exact,
        "original_disableflags": original_flags,
        "treated_disableflags": treated_flags,
        "contact_disable_bit": contact_bit,
        "contact_flag_only": contact_flag_only,
        "reset_active_contact_count": int(np.sum(reset_contacts)),
        "all_perturbed_actions_strictly_in_bounds": actions_in_bounds,
        "maximum_absolute_perturbed_action": float(np.max(np.abs(perturbed))),
        "all_objective_values_finite": bool(np.all(np.isfinite(transition_values))),
        "all_transitions_nonterminal": all_nonterminal,
        "source_e011_raw_sha256": SOURCE_E011_RAW_SHA256,
        "raw_npz_sha256": sha256_file(raw_path),
        "policy_evaluation_computed": False,
        "new_ad_derivative_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "smoothness_grid.png"
    _plot_sweep(plot_path, report)
    summary = {
        "protocol": "g1-contact-free-phase100-smoothness-summary-v1",
        "valid": measurement_valid,
        "scientifically_interpretable": primary["scientifically_interpretable"],
        "outcome": primary["outcome"],
        "phase": CASE_PHASE,
        "arm": CASE_ARM,
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
        "protocol": "g1-contact-free-phase100-smoothness-completion-v1",
        "valid": primary["scientifically_interpretable"],
        "outcome": primary["outcome"],
        "computed_sweep_invocations": 2,
        "computed_epsilon_count": int(EPSILONS.size),
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "new_ad_derivative_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "smoothness_grid.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "smoothness_grid.png": sha256_file(plot_path),
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
