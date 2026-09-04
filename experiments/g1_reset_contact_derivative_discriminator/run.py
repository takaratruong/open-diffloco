"""Test whether disabling contact rescues G1 reset-state action derivatives.

The independently audited E009 contact-enabled arrays are the immutable
control.  This runner changes only MJX's ``mjDSBL_CONTACT`` flag, reconstructs
the same ten reset/action cases, and repeats E009's complete reverse,
coordinate-forward, and fixed-direction finite-difference measurement.  No
policy is evaluated and no optimizer state is created.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
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
    ACTION_DIMENSION,
    ARM_ORDER,
    CASE_COUNT,
    FINITE_DIFFERENCE_ATOL,
    FINITE_DIFFERENCE_RTOL,
    GRADIENT_ATOL,
    GRADIENT_RTOL,
    OBJECTIVE_NAMES,
    PHASES,
    PRIMAL_ATOL,
    PRIMAL_RTOL,
    RESET_QUATERNION_ATOL,
    _arrays_exact,
    _build_compiled_probe,
    _load_source_arrays,
    _objective_report,
    _prepare_cases,
    _validate_e008_audit,
    _write_npz,
    build_common_probe_env,
)
from experiments.g1_success_failure_visitation.run import (
    read_json,
    repository_preflight,
    sha256_file,
    validate_diffsim_hparams,
    write_json,
)
from src.algorithms.shac.algorithm import (
    H1_ACTION_DIRECTION_SEED,
    H1_ACTION_FINITE_DIFFERENCE_EPSILON,
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
SOURCE_E008_AUDIT_SHA256 = (
    "9859cc5a0d5a91311238d122eb2876f40571843351e6341322abdbf35e6edd56"
)
SOURCE_E009_RAW_SHA256 = (
    "fae96f7e218517e46d3556d722c17d23ea10813a37adddcead3762787fedba33"
)
SOURCE_E009_REPORT_SHA256 = (
    "764cdd3bfc72924130b362b65ef5f787b7b8c15869380d376c9dd26bab284b4e"
)
SOURCE_E009_AUDIT_SHA256 = (
    "b78f475efecf6a8c4a1804c691ac69834cad77fcf64af241a4b92d6b2dcac5b8"
)
CONTROL_LABEL = "contact-enabled E009"
TREATMENT_LABEL = "contact-disabled"


def disable_contact_dynamics(env: object):
    """Set only MuJoCo's contact-disable flag on CPU and MJX models."""

    contact_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    cpu_flags = int(env.mj_model.opt.disableflags)
    mjx_flags = int(env.mjx_model.opt.disableflags)
    if cpu_flags != mjx_flags:
        raise ValueError("CPU and MJX disable flags differ before treatment")
    if cpu_flags & contact_bit:
        raise ValueError("contact is already disabled before treatment")
    treated_flags = cpu_flags | contact_bit
    env.mj_model.opt.disableflags = treated_flags
    env.mjx_model = env.mjx_model.replace(
        opt=env.mjx_model.opt.replace(disableflags=treated_flags)
    )
    if int(env.mjx_model.opt.disableflags) != treated_flags:
        raise ValueError("MJX contact-disable treatment did not persist")
    return env


def _case_flags(report: Mapping[str, object], name: str = "case_pass") -> list[bool]:
    values = report.get(name)
    if (
        not isinstance(values, list)
        or len(values) != CASE_COUNT
        or any(type(value) is not bool for value in values)
    ):
        raise ValueError(f"objective {name} vector is invalid")
    return values


def classify_contact_discriminator(
    *,
    measurement_valid: bool,
    contact_absent: bool,
    control_smooth_report: Mapping[str, object],
    treatment_smooth_report: Mapping[str, object],
) -> dict[str, object]:
    """Classify whether contact-disabled dynamics rescue smooth derivatives."""

    control = _case_flags(control_smooth_report)
    treatment = _case_flags(treatment_smooth_report)
    control_gradient = _case_flags(control_smooth_report, "gradient_agreement")
    treatment_gradient = _case_flags(treatment_smooth_report, "gradient_agreement")
    control_fd = _case_flags(control_smooth_report, "finite_difference_agreement")
    treatment_fd = _case_flags(treatment_smooth_report, "finite_difference_agreement")
    control_count = int(sum(control))
    treatment_count = int(sum(treatment))
    control_gradient_count = int(sum(control_gradient))
    treatment_gradient_count = int(sum(treatment_gradient))
    control_fd_count = int(sum(control_fd))
    treatment_fd_count = int(sum(treatment_fd))
    if not measurement_valid or not contact_absent:
        outcome = "invalid-measurement"
        interpretable = False
    elif treatment_count == CASE_COUNT and control_count < CASE_COUNT:
        outcome = "contact-dynamics-cause-reset-derivative-failure"
        interpretable = True
    elif treatment_gradient_count == CASE_COUNT and control_gradient_count < CASE_COUNT:
        outcome = "contact-removes-ad-disagreement-finite-difference-unresolved"
        interpretable = True
    elif (
        treatment_count > control_count
        or treatment_gradient_count > control_gradient_count
        or treatment_fd_count > control_fd_count
    ):
        outcome = "contact-dynamics-contribute-to-reset-derivative-failure"
        interpretable = True
    else:
        outcome = "contact-removal-does-not-rescue-reset-derivatives"
        interpretable = True
    return {
        "protocol": "g1-reset-contact-derivative-classification-v1",
        "valid": bool(measurement_valid and contact_absent),
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "control_smooth_pass_count": control_count,
        "treatment_smooth_pass_count": treatment_count,
        "control_gradient_agreement_count": control_gradient_count,
        "treatment_gradient_agreement_count": treatment_gradient_count,
        "control_finite_difference_agreement_count": control_fd_count,
        "treatment_finite_difference_agreement_count": treatment_fd_count,
        "smooth_cases_rescued": int(
            sum(new and not old for old, new in zip(control, treatment, strict=True))
        ),
        "contact_absent": bool(contact_absent),
        "policy_evaluation_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }


def _validate_e009_sources(
    report: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    expected_audit = {
        "protocol": "g1-reset-action-derivative-independent-audit-v1",
        "experiment": "E-20260904-009",
        "valid": True,
        "scientifically_interpretable": True,
        "outcome": "both-actions-have-smooth-derivative-failures",
        "checks_passed": 23,
        "checks_total": 23,
        "smooth_pass_count": 0,
        "reward_pass_count": 0,
        "source_contact_exact_count": CASE_COUNT,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
    }
    mismatches = {
        key: (audit.get(key), value)
        for key, value in expected_audit.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"E009 audit contract changed: {mismatches}")
    if (
        report.get("protocol") != "g1-reset-action-derivative-report-v1"
        or report.get("outcome") != "both-actions-have-smooth-derivative-failures"
        or report.get("raw_npz_sha256") != SOURCE_E009_RAW_SHA256
        or _case_flags(report.get("smooth_reference_state", {})) != [False] * CASE_COUNT
    ):
        raise ValueError("E009 control report contract changed")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _strict_input_match(
    control: Mapping[str, np.ndarray],
    metadata: Mapping[str, np.ndarray],
    direction: np.ndarray,
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
        name in control
        and name in metadata
        and np.array_equal(control[name], metadata[name])
        for name in shared
    ) and np.array_equal(control.get("direction"), direction)


def _finite_range(values: Sequence[object]) -> dict[str, float | None]:
    array = np.asarray(
        [np.nan if value is None else value for value in values], dtype=np.float64
    )
    finite = array[np.isfinite(array)]
    return {
        "minimum": float(np.min(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def _plot_comparison(path: Path, report: Mapping[str, object]) -> None:
    phases = np.asarray(report["phases"], dtype=np.int64)
    arms = np.asarray(report["arms"])
    labels = [f"{phase}\n{arm}" for phase, arm in zip(phases, arms, strict=True)]
    control = report["control"]
    treatment = report["treatment"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    for column, objective in enumerate(OBJECTIVE_NAMES):
        matrix = np.asarray(
            [control[objective]["case_pass"], treatment[objective]["case_pass"]],
            dtype=np.int64,
        )
        image = axes[0, column].imshow(
            matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto"
        )
        axes[0, column].set_xticks(range(CASE_COUNT), labels, fontsize=8)
        axes[0, column].set_yticks(range(2), (CONTROL_LABEL, TREATMENT_LABEL))
        axes[0, column].set_title(f"{objective}: complete derivative gate")
        for row in range(2):
            for case in range(CASE_COUNT):
                axes[0, column].text(
                    case,
                    row,
                    "PASS" if matrix[row, case] else "FAIL",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axes[0, column], ticks=(0, 1))

    for axis, metric, title, threshold in (
        (
            axes[1, 0],
            "reverse_forward_relative_error",
            "smooth reverse vs coordinate-forward",
            GRADIENT_RTOL,
        ),
        (
            axes[1, 1],
            "finite_difference_relative_error",
            "smooth forward vs central finite difference",
            FINITE_DIFFERENCE_RTOL,
        ),
    ):
        for condition, marker in (("control", "o"), ("treatment", "s")):
            values = np.asarray(
                report[condition]["smooth_reference_state"][metric],
                dtype=np.float64,
            )
            axis.plot(
                range(CASE_COUNT),
                np.maximum(values, 1e-16),
                marker=marker,
                label=condition,
            )
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(range(CASE_COUNT), labels, fontsize=8)
        axis.set_yscale("log")
        axis.set_ylabel("relative error")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        "G1 reset-state derivative validity: audited contact control vs contact disabled"
    )
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
        "source_e009_raw": args.source_e009_raw.resolve(),
        "source_e009_report": args.source_e009_report.resolve(),
        "source_e009_audit": args.source_e009_audit.resolve(),
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "diffsim_hparams": DIFFSIM_HPARAMS_SHA256,
        "source_trajectories": SOURCE_TRAJECTORY_SHA256,
        "source_e008_audit": SOURCE_E008_AUDIT_SHA256,
        "source_e009_raw": SOURCE_E009_RAW_SHA256,
        "source_e009_report": SOURCE_E009_REPORT_SHA256,
        "source_e009_audit": SOURCE_E009_AUDIT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")

    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    _validate_e008_audit(read_json(paths["source_e008_audit"]))
    if not jax.config.x64_enabled:
        raise ValueError("contact derivative discriminator requires JAX x64")
    source_arrays = _load_source_arrays(paths["source_trajectories"])
    control_raw = _load_npz(paths["source_e009_raw"])
    control_report = read_json(paths["source_e009_report"])
    control_audit = read_json(paths["source_e009_audit"])
    _validate_e009_sources(control_report, control_audit)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    contact_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    preflight = {
        "protocol": "g1-reset-contact-derivative-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "seed": args.seed,
        "phases": list(PHASES),
        "arms": list(ARM_ORDER),
        "case_count": CASE_COUNT,
        "action_dimension": ACTION_DIMENSION,
        "objectives": list(OBJECTIVE_NAMES),
        "direction_seed": H1_ACTION_DIRECTION_SEED,
        "finite_difference_epsilon": H1_ACTION_FINITE_DIFFERENCE_EPSILON,
        "contact_disable_bit": contact_bit,
        "control": "immutable independently audited E-20260904-009 arrays",
        "treatment": "set only mjDSBL_CONTACT on CPU and MJX model options",
        "reset_quaternion_atol": RESET_QUATERNION_ATOL,
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

    with solver_context(get_solver_profile(args.solver_profile)):
        env = build_common_probe_env(paths["reference"], hparams)
        original_flags = int(env.mjx_model.opt.disableflags)
        disable_contact_dynamics(env)
        treated_flags = int(env.mjx_model.opt.disableflags)
        states, actions, metadata = _prepare_cases(env, source_arrays, seed=args.seed)
        reset_contact_signatures = np.asarray(
            jax.vmap(env.contact_pair_signature)(states.data), dtype=bool
        )
        direction = jax.random.rademacher(
            jax.random.PRNGKey(H1_ACTION_DIRECTION_SEED),
            (ACTION_DIMENSION,),
            dtype=jnp.float64,
        ) / jnp.sqrt(jnp.asarray(ACTION_DIMENSION, dtype=jnp.float64))
        direction_array = np.asarray(direction, dtype=np.float64)
        compiled_probe = _build_compiled_probe(env, direction)
        first_device = compiled_probe(states, actions)
        jax.block_until_ready(first_device)
        second_device = compiled_probe(states, actions)
        jax.block_until_ready(second_device)

    first = {name: np.asarray(value) for name, value in first_device.items()}
    second = {name: np.asarray(value) for name, value in second_device.items()}
    repeat_exact = _arrays_exact(first, second)
    input_match = _strict_input_match(control_raw, metadata, direction_array)
    contact_flag_only = bool(
        original_flags & contact_bit == 0
        and treated_flags == original_flags | contact_bit
        and int(env.mj_model.opt.disableflags) == treated_flags
    )
    contact_absent = bool(contact_flag_only and not np.any(reset_contact_signatures))
    direct_done = np.asarray(first["direct_done"], dtype=np.float64)
    direct_terminal = np.asarray(first["direct_terminal"], dtype=np.float64)
    arms = metadata["arms"].tolist()
    smooth_report = _objective_report(first, second, objective_index=0, arms=arms)
    reward_report = _objective_report(first, second, objective_index=1, arms=arms)
    measurement_valid = bool(
        input_match
        and repeat_exact
        and contact_flag_only
        and np.all(direct_done == 0.0)
        and np.all(direct_terminal == 0.0)
        and all(
            objective[name] == [True] * CASE_COUNT
            for objective in (smooth_report, reward_report)
            for name in (
                "source_finite",
                "reverse_primal_close",
                "forward_primal_close",
            )
        )
    )
    classification = classify_contact_discriminator(
        measurement_valid=measurement_valid,
        contact_absent=contact_absent,
        control_smooth_report=control_report["smooth_reference_state"],
        treatment_smooth_report=smooth_report,
    )

    raw_arrays = {
        **{f"control_{name}": value for name, value in control_raw.items()},
        **{f"treatment_{name}": value for name, value in metadata.items()},
        "treatment_direction": direction_array,
        "treatment_reset_contact_signatures": reset_contact_signatures,
        **{f"treatment_first_{name}": value for name, value in first.items()},
        **{f"treatment_second_{name}": value for name, value in second.items()},
    }
    raw_path = output_root / "contact_derivative_discriminator.npz"
    _write_npz(raw_path, raw_arrays)
    report = {
        "protocol": "g1-reset-contact-derivative-report-v1",
        **classification,
        "code_commit": args.code_commit,
        "phases": metadata["phases"].tolist(),
        "arms": arms,
        "case_count": CASE_COUNT,
        "action_dimension": ACTION_DIMENSION,
        "objectives": list(OBJECTIVE_NAMES),
        "control_source": "E-20260904-009/20260904T171353Z",
        "control": {name: control_report[name] for name in OBJECTIVE_NAMES},
        "treatment": {
            "smooth_reference_state": smooth_report,
            "e002_h1_reward": reward_report,
        },
        "input_match_to_control": input_match,
        "repeat_exact": repeat_exact,
        "original_disableflags": original_flags,
        "treated_disableflags": treated_flags,
        "contact_disable_bit": contact_bit,
        "contact_flag_only": contact_flag_only,
        "reset_active_contact_count": int(np.sum(reset_contact_signatures)),
        "reset_contact_signature_shape": list(reset_contact_signatures.shape),
        "all_direct_done_false": bool(np.all(direct_done == 0.0)),
        "all_direct_terminal_false": bool(np.all(direct_terminal == 0.0)),
        "direction_seed": H1_ACTION_DIRECTION_SEED,
        "direction": direction_array.tolist(),
        "direction_norm": float(np.linalg.norm(direction_array)),
        "finite_difference_epsilon": H1_ACTION_FINITE_DIFFERENCE_EPSILON,
        "primal_tolerances": {"rtol": PRIMAL_RTOL, "atol": PRIMAL_ATOL},
        "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL},
        "finite_difference_tolerances": {
            "rtol": FINITE_DIFFERENCE_RTOL,
            "atol": FINITE_DIFFERENCE_ATOL,
        },
        "smooth_error_ranges": {
            "control_reverse_forward": _finite_range(
                control_report["smooth_reference_state"][
                    "reverse_forward_relative_error"
                ]
            ),
            "treatment_reverse_forward": _finite_range(
                smooth_report["reverse_forward_relative_error"]
            ),
            "control_finite_difference": _finite_range(
                control_report["smooth_reference_state"][
                    "finite_difference_relative_error"
                ]
            ),
            "treatment_finite_difference": _finite_range(
                smooth_report["finite_difference_relative_error"]
            ),
        },
        "source_e009_raw_sha256": SOURCE_E009_RAW_SHA256,
        "raw_npz_sha256": sha256_file(raw_path),
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "contact_derivative_comparison.png"
    _plot_comparison(plot_path, report)
    summary = {
        "protocol": "g1-reset-contact-derivative-summary-v1",
        **classification,
        "input_match_to_control": input_match,
        "repeat_exact": repeat_exact,
        "phases": list(PHASES),
        "case_count": CASE_COUNT,
        "reset_active_contact_count": int(np.sum(reset_contact_signatures)),
        "control_smooth_case_pass": control_report["smooth_reference_state"][
            "case_pass"
        ],
        "treatment_smooth_case_pass": smooth_report["case_pass"],
        "treatment_reward_case_pass": reward_report["case_pass"],
        "smooth_error_ranges": report["smooth_error_ranges"],
        "raw_npz_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "plot_sha256": sha256_file(plot_path),
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-reset-contact-derivative-completion-v1",
        "valid": classification["scientifically_interpretable"],
        "outcome": classification["outcome"],
        "computed_treatment_probe_invocations": 2,
        "control_probe_recomputed": False,
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": True,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "contact_derivative_discriminator.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "contact_derivative_comparison.png": sha256_file(plot_path),
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
    parser.add_argument("--source-e009-raw", type=Path, required=True)
    parser.add_argument("--source-e009-report", type=Path, required=True)
    parser.add_argument("--source-e009-audit", type=Path, required=True)
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
