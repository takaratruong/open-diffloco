"""Discriminate which final MJX constraint active set tracks E017's AD regimes.

E017 established an immutable nine-alpha pass/fail mask at phase 50 and 75.
This runner executes only the corresponding direct one-substep transitions,
reconstructs MuJoCo's final constraint context, and asks whether changes in its
active rows exactly coincide with every pass/fail transition.  It does not
differentiate, evaluate a policy, update an optimizer, or retain a checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mujoco.mjx._src import solver as mjx_solver
from mujoco.mjx._src import support as mjx_support

import numpy as np

from experiments.g1_hard_contact_action_interpolation_derivatives.run import (
    INPUT_NAMES,
    PROBE_OUTPUT_NAMES,
    initial_pd_diagnostics,
    interpolate_case_actions,
)
from experiments.g1_hard_contact_substep_derivative_discriminator.run import (
    set_one_physics_substep,
)
from experiments.g1_reset_action_derivative_discriminator.run import (
    CASE_COUNT,
    _load_source_arrays,
    _prepare_cases,
    _write_npz,
    build_common_probe_env,
    smooth_reference_state_loss,
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

ALPHAS = np.linspace(0.0, 1.0, 9, dtype=np.float64)
PHASE_CASES = ((4, 5), (6, 7))
SELECTED_PHASES = (50, 75)

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
SOURCE_E016_RAW_SHA256 = (
    "3c660bf47e1e574993c572e5dc7b4bc3d0eb9ac2c48879311fade92c14917685"
)
SOURCE_E016_REPORT_SHA256 = (
    "a65dc6eaf4cb398b90cc6104d46cd1161bbdd3a363c02d9477ea32c033e8ee8c"
)
SOURCE_E016_AUDIT_SHA256 = (
    "d6bc3de772e9da2546b11fea4a0bdbcbf730f70f91e5245632f795aeb7bd22ee"
)
SOURCE_E017_RAW_SHA256 = (
    "71fcd19de96d704d548d4ed92e5093a88c9c167b668751e106ab91c53108576f"
)
SOURCE_E017_REPORT_SHA256 = (
    "7e33d5d159e3ba45a2054cf1bfb803c5b65c153e409a4ced03e0e922e0e7d56e"
)
SOURCE_E017_AUDIT_SHA256 = (
    "7e3018d3f60b531d35ee834130b518523c02d977ed0e588a5ac4506175f7a65e"
)

DIRECT_REPLAY_NAMES = (
    "source_primal",
    "direct_done",
    "direct_terminal",
    "direct_contact_stiffness",
)
TELEMETRY_NAMES = (
    *DIRECT_REPLAY_NAMES,
    "direct_reward",
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "qfrc_constraint",
    "context_qfrc_constraint",
    "context_Jaref",
    "stored_efc_force",
    "context_efc_force",
    "context_active",
    "efc_type",
    "efc_pos",
    "efc_margin",
    "efc_frictionloss",
    "efc_D",
    "efc_aref",
    "solver_niter",
    "constraint_counts",
    "contact_dist",
    "contact_pos",
    "contact_frame",
    "contact_includemargin",
    "contact_friction",
    "contact_dim",
    "contact_geom",
    "contact_efc_address",
    "contact_force",
)

FRICTIONLOSS_TYPES = frozenset((1, 2))
LIMIT_TYPES = frozenset((3, 4))
CONTACT_TYPES = frozenset((5, 6, 7))


def _arrays_exact(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> bool:
    return set(first) == set(second) and all(
        np.array_equal(
            np.asarray(first[name]), np.asarray(second[name]), equal_nan=True
        )
        for name in first
    )


def _invoke_telemetry_probe(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    states: object,
    actions: object,
) -> dict[str, np.ndarray]:
    device_result = compiled_probe(states, actions)
    jax.tree_util.tree_map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        device_result,
    )
    result = {name: np.asarray(value) for name, value in device_result.items()}
    if set(result) != set(TELEMETRY_NAMES):
        raise ValueError("constraint telemetry output names changed")
    return result


def execute_telemetry_sweeps(
    compiled_probe: Callable[[object, object], Mapping[str, object]],
    *,
    states: object,
    actions: object,
    alphas: np.ndarray,
    phase_cases: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray | bool]:
    """Run each complete action interpolation twice through one direct probe."""

    alpha_array = np.asarray(alphas, dtype=np.float64)
    pairs = tuple(phase_cases)
    if (
        alpha_array.ndim != 1
        or alpha_array.size < 2
        or not np.all(np.isfinite(alpha_array))
        or not np.all(np.diff(alpha_array) > 0.0)
        or alpha_array[0] != 0.0
        or alpha_array[-1] != 1.0
    ):
        raise ValueError("alpha grid must be finite, increasing, and span [0, 1]")
    if not pairs or any(pair not in PHASE_CASES for pair in pairs):
        raise ValueError("phase cases must use registered PPO/DiffSim pairs")

    repeats: list[dict[str, np.ndarray]] = []
    repeated_actions: list[np.ndarray] = []
    for _ in range(2):
        outputs = {name: [] for name in TELEMETRY_NAMES}
        all_actions = []
        for ppo_index, diffsim_index in pairs:
            phase_outputs = {name: [] for name in TELEMETRY_NAMES}
            phase_actions = []
            for alpha in alpha_array:
                candidate = interpolate_case_actions(
                    actions,
                    ppo_index=ppo_index,
                    diffsim_index=diffsim_index,
                    alpha=float(alpha),
                )
                result = _invoke_telemetry_probe(compiled_probe, states, candidate)
                phase_actions.append(np.asarray(candidate))
                for name in TELEMETRY_NAMES:
                    phase_outputs[name].append(result[name])
            all_actions.append(phase_actions)
            for name in TELEMETRY_NAMES:
                outputs[name].append(phase_outputs[name])
        repeats.append({name: np.asarray(value) for name, value in outputs.items()})
        repeated_actions.append(np.asarray(all_actions))
    actions_repeat = np.array_equal(repeated_actions[0], repeated_actions[1])
    return {
        **{f"first_{name}": value for name, value in repeats[0].items()},
        **{f"second_{name}": value for name, value in repeats[1].items()},
        "candidate_actions": repeated_actions[0],
        "repeat_exact": bool(actions_repeat and _arrays_exact(repeats[0], repeats[1])),
    }


def direct_replay_gate(
    actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> dict[str, object]:
    """Require all four legacy direct outputs to match the E016 graph exactly."""

    by_output = {}
    for name in DIRECT_REPLAY_NAMES:
        by_output[name] = bool(
            name in actual
            and name in expected
            and np.array_equal(
                np.asarray(actual[name]), np.asarray(expected[name]), equal_nan=True
            )
        )
    return {"by_output": by_output, "valid": bool(all(by_output.values()))}


def _all_nonselected_telemetry_match_baseline(
    sweep: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray]
) -> bool:
    for name in TELEMETRY_NAMES:
        values = np.asarray(sweep[f"first_{name}"])
        expected = np.asarray(baseline[name])
        if values.shape[:3] != (len(PHASE_CASES), ALPHAS.size, CASE_COUNT):
            return False
        for slot, (_, diffsim_index) in enumerate(PHASE_CASES):
            mask = np.arange(CASE_COUNT) != diffsim_index
            selected = values[slot][:, mask]
            tiled = np.broadcast_to(expected[mask], selected.shape)
            if not np.array_equal(selected, tiled, equal_nan=True):
                return False
    return True


def _endpoint_telemetry_matches_baseline(
    sweep: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray]
) -> bool:
    for name in TELEMETRY_NAMES:
        values = np.asarray(sweep[f"first_{name}"])
        expected = np.asarray(baseline[name])
        for slot, (ppo_index, diffsim_index) in enumerate(PHASE_CASES):
            if not np.array_equal(
                values[slot, 0, diffsim_index], expected[ppo_index], equal_nan=True
            ):
                return False
            if not np.array_equal(
                values[slot, -1, diffsim_index],
                expected[diffsim_index],
                equal_nan=True,
            ):
                return False
    return True


def _selected_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[:3] != (len(PHASE_CASES), ALPHAS.size, CASE_COUNT):
        raise ValueError("telemetry array has the wrong phase/alpha/case shape")
    return np.stack(
        [
            array[slot, :, diffsim_index]
            for slot, (_, diffsim_index) in enumerate(PHASE_CASES)
        ]
    )


def _type_mask(efc_type: np.ndarray, values: frozenset[int]) -> np.ndarray:
    return np.isin(efc_type, np.asarray(sorted(values), dtype=np.int64))


def summarize_active_set_transitions(
    active: np.ndarray, efc_type: np.ndarray
) -> dict[str, np.ndarray]:
    """Summarize adjacent final-active changes by MuJoCo constraint family."""

    active_array = np.asarray(active, dtype=bool)
    type_array = np.asarray(efc_type, dtype=np.int64)
    if (
        active_array.ndim != 3
        or active_array.shape[:2] != (len(PHASE_CASES), ALPHAS.size)
        or type_array.shape != (active_array.shape[-1],)
    ):
        raise ValueError("active set or constraint-type shape is invalid")

    row_change = active_array[:, 1:] != active_array[:, :-1]
    friction = _type_mask(type_array, FRICTIONLOSS_TYPES)
    contact = _type_mask(type_array, CONTACT_TYPES)
    limit = _type_mask(type_array, LIMIT_TYPES)
    other = ~(friction | contact | limit)

    def any_change(mask: np.ndarray) -> np.ndarray:
        return np.any(row_change[..., mask], axis=-1)

    return {
        "row_change": row_change,
        "any_change": np.any(row_change, axis=-1),
        "friction_change": any_change(friction),
        "contact_change": any_change(contact),
        "limit_change": any_change(limit),
        "other_change": any_change(other),
    }


def classify_active_set_discriminator(
    *,
    measurement_valid: bool,
    smooth_agreement: np.ndarray,
    active: np.ndarray,
    efc_type: np.ndarray,
) -> dict[str, object]:
    """Classify whether final constraint activity exactly tracks E017 transitions."""

    smooth = np.asarray(smooth_agreement, dtype=bool)
    expected_shape = (len(PHASE_CASES), ALPHAS.size)
    try:
        transitions = summarize_active_set_transitions(active, efc_type)
        shape_valid = smooth.shape == expected_shape
    except (TypeError, ValueError):
        transitions = None
        shape_valid = False

    valid = bool(measurement_valid and shape_valid and transitions is not None)
    mask_transition = smooth[:, 1:] != smooth[:, :-1] if shape_valid else None
    matches = bool(valid and np.array_equal(transitions["any_change"], mask_transition))

    changed_categories: list[str] = []
    if transitions is not None:
        for label, key in (
            ("frictionloss", "friction_change"),
            ("contact", "contact_change"),
            ("limit", "limit_change"),
            ("other", "other_change"),
        ):
            if np.any(transitions[key]):
                changed_categories.append(label)
        changed_categories.sort()

    if not valid:
        outcome = "invalid-measurement"
        interpretable = False
    elif not matches:
        outcome = "final-active-set-does-not-track-ad-regimes"
        interpretable = True
    elif changed_categories == ["frictionloss"]:
        outcome = "frictionloss-active-set-exactly-tracks-ad-regimes"
        interpretable = True
    elif changed_categories == ["contact"]:
        outcome = "contact-active-set-exactly-tracks-ad-regimes"
        interpretable = True
    elif len(changed_categories) == 1:
        outcome = "other-active-set-exactly-tracks-ad-regimes"
        interpretable = True
    else:
        outcome = "coupled-active-set-exactly-tracks-ad-regimes"
        interpretable = True

    return {
        "protocol": "g1-hard-contact-constraint-active-set-classification-v1",
        "valid": valid,
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "active_change_matches_mask_transition": matches,
        "changed_constraint_categories": changed_categories,
    }


def _validate_source_contracts(
    *,
    e016_report: Mapping[str, object],
    e016_audit: Mapping[str, object],
    e017_report: Mapping[str, object],
    e017_audit: Mapping[str, object],
) -> None:
    expected_e016_report = {
        "protocol": "g1-hard-contact-action-interpolation-classification-v1",
        "valid": False,
        "scientifically_interpretable": False,
        "outcome": "invalid-measurement",
        "sweep_repeat_exact": True,
        "nonselected_outputs_match_baseline": False,
        "all_interpolated_primals_valid": True,
        "computed_interpolation_probe_invocations": 36,
        "raw_npz_sha256": SOURCE_E016_RAW_SHA256,
    }
    expected_e016_audit = {
        "protocol": "g1-hard-contact-action-interpolation-independent-audit-v1",
        "valid": True,
        "outcome": "invalid-measurement",
        "invalid_reason": "nonselected-output-axis-order",
        "checks_passed": 22,
        "checks_total": 22,
        "correct_nonselected_outputs_exact": True,
        "descriptive_corrected_outcome": "multiple-ad-regimes-along-action-segment",
    }
    expected_e017_report = {
        "protocol": "g1-hard-contact-action-interpolation-reanalysis-report-v1",
        "valid": True,
        "scientifically_interpretable": True,
        "outcome": "multiple-ad-regimes-along-action-segment",
        "all_interpolated_primals_valid": True,
        "smooth_ad_agreement_count": 10,
        "raw_npz_sha256": SOURCE_E017_RAW_SHA256,
        "simulator_step_computed": False,
        "derivative_computed": False,
    }
    expected_e017_audit = {
        "protocol": (
            "g1-hard-contact-action-interpolation-reanalysis-independent-audit-v1"
        ),
        "valid": True,
        "scientifically_interpretable": True,
        "outcome": "multiple-ad-regimes-along-action-segment",
        "checks_passed": 19,
        "checks_total": 19,
        "smooth_ad_agreement_count": 10,
        "maximum_output_recomputation_delta": 0.0,
        "simulator_step_computed": False,
        "derivative_computed": False,
    }
    contracts = (
        ("E016 report", e016_report, expected_e016_report),
        ("E016 audit", e016_audit, expected_e016_audit),
        ("E017 report", e017_report, expected_e017_report),
        ("E017 audit", e017_audit, expected_e017_audit),
    )
    for label, actual, expected in contracts:
        mismatches = {
            name: (actual.get(name), value)
            for name, value in expected.items()
            if actual.get(name) != value
        }
        if mismatches:
            raise ValueError(f"{label} contract changed: {mismatches}")


def _build_compiled_telemetry_probe(env: object):
    """Build one direct-step graph that exposes the final constraint context."""

    def case_probe(state, action):
        direct_next = env.step(state, action)
        data = direct_next.data
        model = env._get_randomized_model(state.info)
        context = mjx_solver.Context.create(model, data, grad=False)
        next_phase = jnp.minimum(
            state.info["phase"] + env.reference_stride,
            env.reference_length - 1,
        )
        smooth = smooth_reference_state_loss(
            data.qpos,
            data.qvel,
            env.qpos_reference[next_phase],
            env.qvel_reference[next_phase],
        )
        contact = data._impl.contact
        contact_force = jnp.zeros((data._impl.ncon, 6), dtype=data.qpos.dtype)
        for dimension in sorted(set(np.asarray(contact.dim).tolist())):
            forces, contact_ids = mjx_support.contact_force_dim(model, data, dimension)
            contact_force = contact_force.at[contact_ids].set(forces)
        return {
            "source_primal": jnp.stack((smooth, -direct_next.reward)),
            "direct_done": direct_next.done,
            "direct_terminal": direct_next.info["terminal"],
            "direct_contact_stiffness": direct_next.info[
                "transition_contact_stiffness"
            ],
            "direct_reward": direct_next.reward,
            "qpos": data.qpos,
            "qvel": data.qvel,
            "qacc": data.qacc,
            "qacc_smooth": data.qacc_smooth,
            "qacc_warmstart": data.qacc_warmstart,
            "qfrc_constraint": data.qfrc_constraint,
            "context_qfrc_constraint": context.qfrc_constraint,
            "context_Jaref": context.Jaref,
            "stored_efc_force": data._impl.efc_force,
            "context_efc_force": context.efc_force,
            "context_active": context.active,
            "efc_type": jnp.asarray(data._impl.efc_type, dtype=jnp.int32),
            "efc_pos": data._impl.efc_pos,
            "efc_margin": data._impl.efc_margin,
            "efc_frictionloss": data._impl.efc_frictionloss,
            "efc_D": data._impl.efc_D,
            "efc_aref": data._impl.efc_aref,
            "solver_niter": data._impl.solver_niter,
            "constraint_counts": jnp.asarray(
                (
                    data._impl.ne,
                    data._impl.nf,
                    data._impl.nl,
                    data._impl.nefc,
                    data._impl.ncon,
                ),
                dtype=jnp.int32,
            ),
            "contact_dist": contact.dist,
            "contact_pos": contact.pos,
            "contact_frame": contact.frame,
            "contact_includemargin": contact.includemargin,
            "contact_friction": contact.friction,
            "contact_dim": jnp.asarray(contact.dim, dtype=jnp.int32),
            "contact_geom": contact.geom,
            "contact_efc_address": jnp.asarray(contact.efc_address, dtype=jnp.int32),
            "contact_force": contact_force,
        }

    return jax.jit(jax.vmap(case_probe))


def reconstruct_pyramidal_active_set(
    *,
    jaref: np.ndarray,
    efc_type: np.ndarray,
    efc_frictionloss: np.ndarray,
    efc_d: np.ndarray,
) -> np.ndarray:
    """Recompute MuJoCo 3.9's pyramidal final-active predicate on the host."""

    jaref_array = np.asarray(jaref, dtype=np.float64)
    type_array = np.asarray(efc_type, dtype=np.int64)
    frictionloss = np.asarray(efc_frictionloss, dtype=np.float64)
    d_array = np.asarray(efc_d, dtype=np.float64)
    if (
        jaref_array.shape != frictionloss.shape
        or jaref_array.shape != d_array.shape
        or type_array.shape != (jaref_array.shape[-1],)
    ):
        raise ValueError("constraint arrays have incompatible shapes")
    active = jaref_array < 0.0
    equality = type_array == 0
    friction = _type_mask(type_array, FRICTIONLOSS_TYPES)
    active[..., equality] = True
    active[..., friction] = True
    denominator = d_array + (d_array == 0.0) * mjx_solver.mujoco.mjMINVAL
    half_width = frictionloss / denominator
    linear = ((jaref_array <= -half_width) | (jaref_array >= half_width)) & (
        frictionloss > 0.0
    )
    active[..., friction] &= ~linear[..., friction]
    return active


def _category_name(constraint_type: int) -> str:
    if constraint_type in FRICTIONLOSS_TYPES:
        return "frictionloss"
    if constraint_type in CONTACT_TYPES:
        return "contact"
    if constraint_type in LIMIT_TYPES:
        return "limit"
    return "other"


def _transition_details(
    *,
    smooth_agreement: np.ndarray,
    active: np.ndarray,
    efc_type: np.ndarray,
    jaref: np.ndarray,
    efc_force: np.ndarray,
) -> list[dict[str, object]]:
    details = []
    for phase_slot, phase in enumerate(SELECTED_PHASES):
        for boundary in range(ALPHAS.size - 1):
            changed = np.flatnonzero(
                active[phase_slot, boundary] != active[phase_slot, boundary + 1]
            )
            rows = []
            for row in changed:
                rows.append(
                    {
                        "row": int(row),
                        "type": int(efc_type[row]),
                        "category": _category_name(int(efc_type[row])),
                        "active_left": bool(active[phase_slot, boundary, row]),
                        "active_right": bool(active[phase_slot, boundary + 1, row]),
                        "jaref_left": float(jaref[phase_slot, boundary, row]),
                        "jaref_right": float(jaref[phase_slot, boundary + 1, row]),
                        "force_left": float(efc_force[phase_slot, boundary, row]),
                        "force_right": float(efc_force[phase_slot, boundary + 1, row]),
                    }
                )
            details.append(
                {
                    "phase": phase,
                    "left_alpha": float(ALPHAS[boundary]),
                    "right_alpha": float(ALPHAS[boundary + 1]),
                    "ad_regime_changed": bool(
                        smooth_agreement[phase_slot, boundary]
                        != smooth_agreement[phase_slot, boundary + 1]
                    ),
                    "active_set_changed": bool(changed.size),
                    "changed_rows": rows,
                }
            )
    return details


def _finite_or_none(values: np.ndarray) -> object:
    array = np.asarray(values)
    if array.ndim == 0:
        if array.dtype.kind == "b":
            return bool(array)
        if array.dtype.kind in "iu":
            return int(array)
        return float(array) if np.isfinite(array) else None
    return [_finite_or_none(value) for value in array]


def _active_counts(active: np.ndarray, efc_type: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "frictionloss": _type_mask(efc_type, FRICTIONLOSS_TYPES),
        "contact": _type_mask(efc_type, CONTACT_TYPES),
        "limit": _type_mask(efc_type, LIMIT_TYPES),
    }
    masks["other"] = ~np.logical_or.reduce(tuple(masks.values()))
    return {
        name: np.sum(active[..., mask], axis=-1, dtype=np.int64)
        for name, mask in masks.items()
    }


def _plot_result(
    path: Path,
    *,
    smooth_agreement: np.ndarray,
    transitions: Mapping[str, np.ndarray],
    active_counts: Mapping[str, np.ndarray],
    jaref: np.ndarray,
    efc_type: np.ndarray,
    contact_force: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(smooth_agreement.astype(np.int8), vmin=0, vmax=1, cmap="RdYlGn")
    axes[0, 0].set_xticks(range(ALPHAS.size), [f"{value:g}" for value in ALPHAS])
    axes[0, 0].set_yticks(range(2), SELECTED_PHASES)
    axes[0, 0].set_xlabel("alpha: PPO (0) to DiffSim (1)")
    axes[0, 0].set_ylabel("reset phase")
    axes[0, 0].set_title("Frozen E017 smooth-AD agreement")
    for row in range(2):
        for column in range(ALPHAS.size):
            axes[0, 0].text(
                column,
                row,
                "PASS" if smooth_agreement[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=7,
            )

    ad_change = smooth_agreement[:, 1:] != smooth_agreement[:, :-1]
    active_change = transitions["any_change"]
    comparison = ad_change.astype(np.int8) * 2 + active_change.astype(np.int8)
    axes[0, 1].imshow(comparison, vmin=0, vmax=3, cmap="viridis", aspect="auto")
    labels = {0: "neither", 1: "active only", 2: "AD only", 3: "both"}
    axes[0, 1].set_xticks(
        range(ALPHAS.size - 1),
        [f"{ALPHAS[index]:g}→{ALPHAS[index + 1]:g}" for index in range(8)],
        rotation=35,
        ha="right",
    )
    axes[0, 1].set_yticks(range(2), SELECTED_PHASES)
    axes[0, 1].set_title("Adjacent AD vs final-active transitions")
    for row in range(2):
        for column in range(ALPHAS.size - 1):
            axes[0, 1].text(
                column,
                row,
                labels[int(comparison[row, column])],
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    styles = {"frictionloss": "-", "contact": "--", "limit": ":", "other": "-."}
    for phase_slot, phase in enumerate(SELECTED_PHASES):
        for category, counts in active_counts.items():
            axes[1, 0].plot(
                ALPHAS,
                counts[phase_slot],
                linestyle=styles[category],
                marker="o",
                label=f"phase {phase} {category}",
            )
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("active constraint rows")
    axes[1, 0].set_title("Final active-row count by constraint family")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=7, ncol=2)

    contact_mask = _type_mask(efc_type, CONTACT_TYPES)
    friction_mask = _type_mask(efc_type, FRICTIONLOSS_TYPES)
    for phase_slot, phase in enumerate(SELECTED_PHASES):
        if np.any(contact_mask):
            axes[1, 1].plot(
                ALPHAS,
                np.min(np.abs(jaref[phase_slot, :, contact_mask]), axis=-1),
                marker="o",
                label=f"phase {phase} min |contact Jaref|",
            )
        if np.any(friction_mask):
            axes[1, 1].plot(
                ALPHAS,
                np.min(np.abs(jaref[phase_slot, :, friction_mask]), axis=-1),
                marker="s",
                label=f"phase {phase} min |friction Jaref|",
            )
        axes[1, 1].plot(
            ALPHAS,
            np.linalg.norm(contact_force[phase_slot], axis=(-2, -1)),
            linestyle=":",
            label=f"phase {phase} contact-force norm",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_title("Constraint-boundary and decoded-contact diagnostics")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)
    figure.suptitle("G1 first-solve final constraint active-set discriminator")
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    os.replace(temporary, path)


def _plot_invalid(path: Path, gates: Mapping[str, object]) -> None:
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.axis("off")
    lines = ["Constraint active-set measurement invalid; no scientific outcome."]
    lines.extend(f"{name}: {value}" for name, value in gates.items())
    axis.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
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
        "source_e016_raw": args.source_e016_raw.resolve(),
        "source_e016_report": args.source_e016_report.resolve(),
        "source_e016_audit": args.source_e016_audit.resolve(),
        "source_e017_raw": args.source_e017_raw.resolve(),
        "source_e017_report": args.source_e017_report.resolve(),
        "source_e017_audit": args.source_e017_audit.resolve(),
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "diffsim_hparams": DIFFSIM_HPARAMS_SHA256,
        "source_trajectories": SOURCE_TRAJECTORY_SHA256,
        "source_e008_audit": SOURCE_E008_AUDIT_SHA256,
        "source_e016_raw": SOURCE_E016_RAW_SHA256,
        "source_e016_report": SOURCE_E016_REPORT_SHA256,
        "source_e016_audit": SOURCE_E016_AUDIT_SHA256,
        "source_e017_raw": SOURCE_E017_RAW_SHA256,
        "source_e017_report": SOURCE_E017_REPORT_SHA256,
        "source_e017_audit": SOURCE_E017_AUDIT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")
    if not jax.config.x64_enabled:
        raise ValueError("constraint active-set discriminator requires JAX x64")

    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    e016_raw = _load_npz(paths["source_e016_raw"])
    e017_raw = _load_npz(paths["source_e017_raw"])
    e016_report = read_json(paths["source_e016_report"])
    e016_audit = read_json(paths["source_e016_audit"])
    e017_report = read_json(paths["source_e017_report"])
    e017_audit = read_json(paths["source_e017_audit"])
    _validate_source_contracts(
        e016_report=e016_report,
        e016_audit=e016_audit,
        e017_report=e017_report,
        e017_audit=e017_audit,
    )
    smooth_agreement = np.asarray(e017_raw["smooth_gradient_agreement"], dtype=bool)
    source_primal_valid = np.asarray(e017_raw["primal_valid"], dtype=bool)
    e017_contract_exact = bool(
        np.array_equal(e017_raw["alphas"], ALPHAS)
        and np.array_equal(e017_raw["selected_phases"], SELECTED_PHASES)
        and np.all(source_primal_valid)
        and smooth_agreement.tolist() == e017_report["smooth_gradient_agreement"]
        and smooth_agreement.tolist() == e017_audit["smooth_gradient_agreement"]
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-hard-contact-constraint-active-set-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "seed": args.seed,
        "selected_phases": list(SELECTED_PHASES),
        "phase_cases": [list(pair) for pair in PHASE_CASES],
        "alphas": ALPHAS.tolist(),
        "telemetry_names": list(TELEMETRY_NAMES),
        "source_ad_mask": smooth_agreement.tolist(),
        "solver_profile": args.solver_profile,
        "mujoco_constraint_context": "mujoco-3.9.0 Context.create grad=False",
        "primary_gate": "final active-set changes versus E017 AD-mask changes",
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": False,
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
        compiled_probe = _build_compiled_telemetry_probe(env)
        baseline_first = _invoke_telemetry_probe(compiled_probe, states, actions)
        baseline_second = _invoke_telemetry_probe(compiled_probe, states, actions)
        sweeps = execute_telemetry_sweeps(
            compiled_probe,
            states=states,
            actions=actions,
            alphas=ALPHAS,
            phase_cases=PHASE_CASES,
        )

    input_match = all(
        np.array_equal(np.asarray(metadata[name]), e016_raw[f"input_{name}"])
        for name in INPUT_NAMES
    )
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
        and np.array_equal(
            reset_contact_signatures, e016_raw["reset_contact_signatures"]
        )
    )
    substep_only = bool(
        original_n_frames == 4
        and treated_n_frames == 1
        and float(env.dt) == original_dt
    )
    baseline_repeat_exact = _arrays_exact(baseline_first, baseline_second)
    sweep_repeat_exact = bool(sweeps["repeat_exact"])
    nonselected_exact = _all_nonselected_telemetry_match_baseline(
        sweeps, baseline_first
    )
    endpoints_exact = _endpoint_telemetry_matches_baseline(sweeps, baseline_first)
    candidate_actions_exact = bool(
        np.array_equal(sweeps["candidate_actions"], e016_raw["candidate_actions"])
    )
    actual_direct = {
        name: np.asarray(sweeps[f"first_{name}"]) for name in DIRECT_REPLAY_NAMES
    }
    expected_direct = {
        name: np.asarray(e016_raw[f"first_{name}"]) for name in DIRECT_REPLAY_NAMES
    }
    direct_gate = direct_replay_gate(actual_direct, expected_direct)

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
            np.broadcast_to(metadata["reset_qpos"][diffsim, 7:], target.shape)
            for target, (_, diffsim) in zip(candidate_targets, PHASE_CASES, strict=True)
        ]
    )
    candidate_qvel = np.stack(
        [
            np.broadcast_to(metadata["reset_qvel"][diffsim, 6:], target.shape)
            for target, (_, diffsim) in zip(candidate_targets, PHASE_CASES, strict=True)
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
    pd_exact = bool(
        np.array_equal(candidate_targets, e016_raw["candidate_position_targets"])
        and all(
            np.array_equal(value, e016_raw[f"initial_pd_{name}"])
            for name, value in pd.items()
        )
    )
    no_initial_effort_clipping = bool(not np.any(pd["clipped"]))

    selected = {
        name: _selected_rows(sweeps[f"first_{name}"]) for name in TELEMETRY_NAMES
    }
    efc_type = np.asarray(selected["efc_type"][0, 0], dtype=np.int64)
    static_rows = bool(
        np.all(selected["efc_type"] == efc_type)
        and np.all(selected["constraint_counts"] == selected["constraint_counts"][0, 0])
        and np.all(selected["contact_dim"] == selected["contact_dim"][0, 0])
        and np.all(
            selected["contact_efc_address"] == selected["contact_efc_address"][0, 0]
        )
    )
    context_force_exact = bool(
        np.array_equal(
            selected["context_efc_force"],
            selected["stored_efc_force"],
            equal_nan=True,
        )
        and np.array_equal(
            selected["context_qfrc_constraint"],
            selected["qfrc_constraint"],
            equal_nan=True,
        )
    )
    reconstructed_active = reconstruct_pyramidal_active_set(
        jaref=selected["context_Jaref"],
        efc_type=efc_type,
        efc_frictionloss=selected["efc_frictionloss"],
        efc_d=selected["efc_D"],
    )
    active = np.asarray(selected["context_active"], dtype=bool)
    active_reconstruction_exact = bool(np.array_equal(active, reconstructed_active))
    objective_identity_exact = bool(
        np.array_equal(
            selected["source_primal"][..., 1],
            -selected["direct_reward"],
            equal_nan=True,
        )
    )
    all_nonterminal = bool(
        np.all(selected["direct_done"] == 0.0)
        and np.all(selected["direct_terminal"] == 0.0)
    )
    finite_names = (
        "source_primal",
        "qpos",
        "qvel",
        "qacc",
        "qacc_smooth",
        "qacc_warmstart",
        "qfrc_constraint",
        "context_qfrc_constraint",
        "context_Jaref",
        "stored_efc_force",
        "context_efc_force",
        "efc_pos",
        "efc_margin",
        "efc_frictionloss",
        "efc_D",
        "efc_aref",
        "contact_dist",
        "contact_pos",
        "contact_frame",
        "contact_friction",
        "contact_force",
    )
    all_finite = bool(all(np.all(np.isfinite(selected[name])) for name in finite_names))
    source_probe_names_present = bool(
        all(f"first_{name}" in e016_raw for name in PROBE_OUTPUT_NAMES)
    )
    measurement_gates = {
        "e017_contract_exact": e017_contract_exact,
        "source_probe_names_present": source_probe_names_present,
        "input_match_to_e016": input_match,
        "case_identity_exact": case_identity,
        "hard_contact_reset_exact": hard_contact_exact,
        "one_substep_only": substep_only,
        "baseline_telemetry_repeat_exact": baseline_repeat_exact,
        "sweep_repeat_exact": sweep_repeat_exact,
        "nonselected_telemetry_exact": nonselected_exact,
        "endpoint_telemetry_exact": endpoints_exact,
        "candidate_actions_exact": candidate_actions_exact,
        "direct_e016_replay_exact": bool(direct_gate["valid"]),
        "pd_diagnostics_exact": pd_exact,
        "no_initial_effort_clipping": no_initial_effort_clipping,
        "static_constraint_row_layout": static_rows,
        "context_force_exact": context_force_exact,
        "active_reconstruction_exact": active_reconstruction_exact,
        "objective_identity_exact": objective_identity_exact,
        "all_transitions_nonterminal": all_nonterminal,
        "all_telemetry_finite": all_finite,
    }
    measurement_valid = bool(all(measurement_gates.values()))
    classification = classify_active_set_discriminator(
        measurement_valid=measurement_valid,
        smooth_agreement=smooth_agreement,
        active=active,
        efc_type=efc_type,
    )
    transitions = summarize_active_set_transitions(active, efc_type)
    counts = _active_counts(active, efc_type)
    details = _transition_details(
        smooth_agreement=smooth_agreement,
        active=active,
        efc_type=efc_type,
        jaref=selected["context_Jaref"],
        efc_force=selected["stored_efc_force"],
    )
    contact_included = selected["contact_dist"] < selected["contact_includemargin"]
    contact_identity_change = np.any(
        selected["contact_geom"][:, 1:] != selected["contact_geom"][:, :-1],
        axis=(-2, -1),
    )
    contact_inclusion_change = np.any(
        contact_included[:, 1:] != contact_included[:, :-1], axis=-1
    )

    raw_arrays = {
        **{f"input_{name}": np.asarray(value) for name, value in metadata.items()},
        "alphas": ALPHAS,
        "selected_phases": np.asarray(SELECTED_PHASES, dtype=np.int64),
        "phase_cases": np.asarray(PHASE_CASES, dtype=np.int64),
        "source_smooth_gradient_agreement": smooth_agreement,
        "source_primal_valid": source_primal_valid,
        "reset_contact_signatures": reset_contact_signatures,
        "reset_contact_counts": reset_contact_counts,
        "candidate_actions": np.asarray(sweeps["candidate_actions"]),
        "candidate_position_targets": candidate_targets,
        **{f"initial_pd_{name}": value for name, value in pd.items()},
        **{f"baseline_first_{name}": value for name, value in baseline_first.items()},
        **{f"baseline_second_{name}": value for name, value in baseline_second.items()},
        **{
            name: np.asarray(value)
            for name, value in sweeps.items()
            if name != "repeat_exact"
        },
        **{f"selected_{name}": value for name, value in selected.items()},
        "reconstructed_active": reconstructed_active,
        **{f"transition_{name}": value for name, value in transitions.items()},
        **{f"active_count_{name}": value for name, value in counts.items()},
        "contact_included": contact_included,
        "contact_identity_change": contact_identity_change,
        "contact_inclusion_change": contact_inclusion_change,
    }
    raw_path = output_root / "constraint_active_set_discriminator.npz"
    _write_npz(raw_path, raw_arrays)
    report = {
        **classification,
        "code_commit": args.code_commit,
        "source_run": "E-20260904-017/20260904T222809Z",
        "selected_phases": list(SELECTED_PHASES),
        "phase_cases": [list(pair) for pair in PHASE_CASES],
        "alphas": ALPHAS.tolist(),
        "source_smooth_gradient_agreement": smooth_agreement.tolist(),
        "measurement_gates": measurement_gates,
        "direct_replay_gate": direct_gate,
        "constraint_type_counts": {
            "equality": int(np.sum(efc_type == 0)),
            "frictionloss": int(np.sum(_type_mask(efc_type, FRICTIONLOSS_TYPES))),
            "limit": int(np.sum(_type_mask(efc_type, LIMIT_TYPES))),
            "contact": int(np.sum(_type_mask(efc_type, CONTACT_TYPES))),
        },
        "constraint_counts_ne_nf_nl_nefc_ncon": selected["constraint_counts"][
            0, 0
        ].tolist(),
        "active_counts": {name: value.tolist() for name, value in counts.items()},
        "ad_transition_mask": (
            smooth_agreement[:, 1:] != smooth_agreement[:, :-1]
        ).tolist(),
        "active_transition_mask": transitions["any_change"].tolist(),
        "friction_transition_mask": transitions["friction_change"].tolist(),
        "contact_transition_mask": transitions["contact_change"].tolist(),
        "limit_transition_mask": transitions["limit_change"].tolist(),
        "other_transition_mask": transitions["other_change"].tolist(),
        "contact_identity_change": contact_identity_change.tolist(),
        "contact_inclusion_change": contact_inclusion_change.tolist(),
        "transition_details": details,
        "maximum_initial_effort_utilization": float(np.max(pd["effort_utilization"])),
        "minimum_initial_effort_margin": float(np.min(pd["effort_margin"])),
        "computed_baseline_probe_invocations": 2,
        "computed_interpolation_probe_invocations": 36,
        "computed_full_batch_direct_steps": 380,
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "source_hashes": expected_hashes,
        "raw_npz_sha256": sha256_file(raw_path),
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "constraint_active_set_discriminator.png"
    if classification["scientifically_interpretable"]:
        _plot_result(
            plot_path,
            smooth_agreement=smooth_agreement,
            transitions=transitions,
            active_counts=counts,
            jaref=selected["context_Jaref"],
            efc_type=efc_type,
            contact_force=selected["contact_force"],
        )
    else:
        _plot_invalid(plot_path, measurement_gates)
    summary = {
        **classification,
        "selected_phases": list(SELECTED_PHASES),
        "alphas": ALPHAS.tolist(),
        "ad_transition_mask": report["ad_transition_mask"],
        "active_transition_mask": report["active_transition_mask"],
        "changed_constraint_categories": classification[
            "changed_constraint_categories"
        ],
        "measurement_gates": measurement_gates,
        "raw_npz_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "plot_sha256": sha256_file(plot_path),
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-hard-contact-constraint-active-set-completion-v1",
        "valid": bool(classification["scientifically_interpretable"]),
        "outcome": classification["outcome"],
        "computed_baseline_probe_invocations": 2,
        "computed_interpolation_probe_invocations": 36,
        "computed_full_batch_direct_steps": 380,
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "constraint_active_set_discriminator.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "constraint_active_set_discriminator.png": sha256_file(plot_path),
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
    parser.add_argument("--source-e016-raw", type=Path, required=True)
    parser.add_argument("--source-e016-report", type=Path, required=True)
    parser.add_argument("--source-e016-audit", type=Path, required=True)
    parser.add_argument("--source-e017-raw", type=Path, required=True)
    parser.add_argument("--source-e017-report", type=Path, required=True)
    parser.add_argument("--source-e017-audit", type=Path, required=True)
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
