"""Compare PPO-success and DiffSim-failure action derivatives at equal resets.

The probe reuses the ten frozen actions captured by E007 at five bit-exact
reset states.  It differentiates two H1 scalars for every state/action pair:
a smooth reference-state loss that isolates the MJX transition, and the exact
retained-E002 one-step reward loss.  Direct reverse mode, all-coordinate
forward mode, and one deterministic central finite-difference direction are
evaluated twice.  No policy is evaluated and no optimizer state is created.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from experiments.g1_success_failure_visitation.run import (
    PHASES,
    read_json,
    repository_preflight,
    sha256_file,
    validate_diffsim_hparams,
    write_json,
)
from src.algorithms.shac.algorithm import (
    H1_ACTION_DIRECTION_SEED,
    H1_ACTION_FINITE_DIFFERENCE_EPSILON,
    compute_action_derivative_pair,
)
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_tracking import make_evaluation_env
from tools.run_g1_tracking_shac import configure_jax


REFERENCE_SHA256 = "f47d13b431d85a273eba6022f5a28bd55cae7c788112baf0778ab159914a039c"
DIFFSIM_HPARAMS_SHA256 = (
    "79927f89ef75cf0a6fbfd5c92746a59db587c00319db780dcad702f0c3bbd5eb"
)
SOURCE_TRAJECTORY_SHA256 = (
    "dc4199fa5383e7caf31c89bb56c7d261af6561ce237d48e8e217276827dbc89b"
)
SOURCE_TRAJECTORY_BASENAME = "paired_trajectories.npz"
SOURCE_E008_AUDIT_SHA256 = (
    "9859cc5a0d5a91311238d122eb2876f40571843351e6341322abdbf35e6edd56"
)
OBJECTIVE_NAMES = ("smooth_reference_state", "e002_h1_reward")
ARM_ORDER = ("ppo", "diffsim")
ACTION_DIMENSION = 29
CASE_COUNT = len(PHASES) * len(ARM_ORDER)
PRIMAL_RTOL = 1e-10
PRIMAL_ATOL = 1e-12
GRADIENT_RTOL = 5e-5
GRADIENT_ATOL = 1e-9
FINITE_DIFFERENCE_RTOL = 5e-3
FINITE_DIFFERENCE_ATOL = 5e-5
RESET_QUATERNION_ATOL = 5e-15


def smooth_reference_state_loss(
    qpos: jax.Array,
    qvel: jax.Array,
    reference_qpos: jax.Array,
    reference_qvel: jax.Array,
) -> jax.Array:
    """Return a smooth, dimension-balanced reference-state tracking loss.

    Position scales are 0.3 m for root translation, 0.4 for quaternion
    components, and 1 rad for joints. Velocity scales are 1 m/s for root
    translation and pi rad/s for root angular and joint velocities. This loss
    deliberately avoids norms, arccos, clipping, termination, and reward
    regularizers so its derivative isolates the one-step physical state map.
    """

    qpos = jnp.asarray(qpos)
    qvel = jnp.asarray(qvel)
    reference_qpos = jnp.asarray(reference_qpos, dtype=qpos.dtype)
    reference_qvel = jnp.asarray(reference_qvel, dtype=qvel.dtype)
    if (
        qpos.ndim != 1
        or qvel.ndim != 1
        or qpos.shape != reference_qpos.shape
        or qvel.shape != reference_qvel.shape
        or qpos.shape[0] < 7
        or qvel.shape[0] < 6
    ):
        raise ValueError("reference-state loss inputs have invalid shapes")
    qpos_scale = jnp.concatenate(
        (
            jnp.full((3,), 0.3, dtype=qpos.dtype),
            jnp.full((4,), 0.4, dtype=qpos.dtype),
            jnp.ones((qpos.shape[0] - 7,), dtype=qpos.dtype),
        )
    )
    qvel_scale = jnp.concatenate(
        (
            jnp.ones((3,), dtype=qvel.dtype),
            jnp.full((qvel.shape[0] - 3,), jnp.pi, dtype=qvel.dtype),
        )
    )
    qpos_error = (qpos - reference_qpos) / qpos_scale
    qvel_error = (qvel - reference_qvel) / qvel_scale
    return 0.5 * (jnp.mean(jnp.square(qpos_error)) + jnp.mean(jnp.square(qvel_error)))


def compute_two_objective_derivatives(
    objectives: Sequence[Callable[[jax.Array], jax.Array]],
    action: jax.Array,
    *,
    direction: jax.Array,
    finite_difference_epsilon: float,
) -> dict[str, jax.Array]:
    """Apply the proven scalar derivative seam independently to two outputs."""

    if len(objectives) != len(OBJECTIVE_NAMES):
        raise ValueError("probe requires exactly two structurally separate objectives")
    per_objective = []
    for scalar_objective in objectives:
        probe = jnp.asarray(scalar_objective(action))
        if probe.shape != ():
            raise ValueError("each paired objective must return one scalar")
        result = compute_action_derivative_pair(
            scalar_objective,
            action,
            direction=direction,
            finite_difference_epsilon=finite_difference_epsilon,
        )
        per_objective.append({**result, "source_primal": scalar_objective(action)})
    stacked = {
        name: jnp.stack([result[name] for result in per_objective])
        for name in per_objective[0]
    }
    return {
        **{
            name: value
            for name, value in stacked.items()
            if name not in {"reverse_gradient", "forward_gradient"}
        },
        "reverse_jacobian": stacked["reverse_gradient"],
        "forward_jacobian": stacked["forward_gradient"],
    }


def actor_action_from_model_action(
    model_action: np.ndarray, model_to_actor_permutation: np.ndarray
) -> np.ndarray:
    """Invert the source-order environment action permutation."""

    action = np.asarray(model_action)
    permutation = np.asarray(model_to_actor_permutation)
    if (
        action.ndim != 1
        or permutation.shape != action.shape
        or permutation.dtype.kind not in "iu"
        or not np.array_equal(np.sort(permutation), np.arange(action.size))
    ):
        raise ValueError("model action or model-to-actor permutation is invalid")
    return action[permutation]


def validate_reconstructed_reset_qpos(
    reconstructed: np.ndarray, captured: np.ndarray
) -> float:
    """Require exact non-quaternion qpos and bounded normalization roundoff."""

    reconstructed = np.asarray(reconstructed)
    captured = np.asarray(captured)
    if (
        reconstructed.ndim != 1
        or reconstructed.shape != captured.shape
        or reconstructed.shape[0] < 7
    ):
        raise ValueError("reconstructed and captured qpos shapes are invalid")
    non_quaternion = np.concatenate(
        (np.arange(3), np.arange(7, reconstructed.shape[0]))
    )
    if not np.array_equal(reconstructed[non_quaternion], captured[non_quaternion]):
        raise ValueError("non-quaternion reset qpos changed")
    maximum_delta = float(np.max(np.abs(reconstructed[3:7] - captured[3:7])))
    if not math.isfinite(maximum_delta) or maximum_delta > RESET_QUATERNION_ATOL:
        raise ValueError(
            "root-quaternion reconstruction exceeds the fixed roundoff bound: "
            f"{maximum_delta} > {RESET_QUATERNION_ATOL}"
        )
    return maximum_delta


def build_common_probe_env(reference: str | Path, hparams: Mapping[str, object]):
    """Build only the exact retained-E002 environment used by E007."""

    return make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=4,
        solver_ls_iterations=5,
        body_mass_scale=1.0,
        effort_limit_scale=1.0,
        reference_path=reference,
        reference_stride=1,
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=(4, 8, 12),
        actor_reference_preview_mode=str(hparams["actor_reference_preview_mode"]),
        actor_observe_motion_anchor_position=False,
        tracking_velocity_kernel=str(hparams["tracking_velocity_kernel"]),
        tracking_root_velocity_weight=float(hparams["tracking_root_velocity_weight"]),
        actor_observation_noise=False,
        domain_randomization=False,
        friction_range=(1.0, 1.0),
        kp_range=(35.0, 35.0),
        kd_range=(0.5, 0.5),
        com_offset_range=(0.0, 0.0, 0.0),
        reference_reset_noise_scale=0.0,
        reference_residual_control=True,
        reference_residual_scale=1.0,
    )


def _json_safe_numeric(values: np.ndarray) -> object:
    array = np.asarray(values)
    if array.ndim == 0:
        if array.dtype.kind == "b":
            return bool(array)
        if array.dtype.kind in "iu":
            return int(array)
        value = float(array)
        return value if math.isfinite(value) else None
    return [_json_safe_numeric(value) for value in array]


def _relative_error(left: np.ndarray, right: np.ndarray, *, axis=None) -> np.ndarray:
    difference = (
        np.linalg.norm(left - right, axis=axis)
        if axis is not None
        else np.abs(left - right)
    )
    left_norm = np.linalg.norm(left, axis=axis) if axis is not None else np.abs(left)
    right_norm = np.linalg.norm(right, axis=axis) if axis is not None else np.abs(right)
    return difference / np.maximum(np.maximum(left_norm, right_norm), 1e-12)


def _objective_report(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    objective_index: int,
    arms: Sequence[str],
) -> dict[str, object]:
    derivative_keys = (
        "source_primal",
        "reverse_primal",
        "forward_primal",
        "reverse_jacobian",
        "forward_jacobian",
        "forward_directional",
        "finite_difference_directional",
    )
    repeat_exact = all(
        np.array_equal(
            np.asarray(first[name])[:, objective_index],
            np.asarray(second[name])[:, objective_index],
            equal_nan=True,
        )
        for name in derivative_keys
    )
    source = np.asarray(first["source_primal"], dtype=np.float64)[:, objective_index]
    reverse_primal = np.asarray(first["reverse_primal"], dtype=np.float64)[
        :, objective_index
    ]
    forward_primal = np.asarray(first["forward_primal"], dtype=np.float64)[
        :, objective_index
    ]
    reverse = np.asarray(first["reverse_jacobian"], dtype=np.float64)[
        :, objective_index
    ]
    forward = np.asarray(first["forward_jacobian"], dtype=np.float64)[
        :, objective_index
    ]
    forward_directional = np.asarray(first["forward_directional"], dtype=np.float64)[
        :, objective_index
    ]
    finite_difference = np.asarray(
        first["finite_difference_directional"], dtype=np.float64
    )[:, objective_index]
    if (
        source.shape != (CASE_COUNT,)
        or reverse.shape != (CASE_COUNT, ACTION_DIMENSION)
        or forward.shape != reverse.shape
        or forward_directional.shape != source.shape
        or finite_difference.shape != source.shape
    ):
        raise ValueError("paired derivative result shapes are invalid")

    source_finite = np.isfinite(source)
    reverse_primal_close = np.isclose(
        source, reverse_primal, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL
    )
    forward_primal_close = np.isclose(
        source, forward_primal, rtol=PRIMAL_RTOL, atol=PRIMAL_ATOL
    )
    reverse_finite = np.all(np.isfinite(reverse), axis=1)
    forward_finite = np.all(np.isfinite(forward), axis=1)
    jointly_finite = reverse_finite & forward_finite
    gradient_agreement = jointly_finite & np.all(
        np.isclose(
            reverse,
            forward,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
            equal_nan=False,
        ),
        axis=1,
    )
    finite_difference_finite = np.isfinite(forward_directional) & np.isfinite(
        finite_difference
    )
    finite_difference_agreement = finite_difference_finite & np.isclose(
        forward_directional,
        finite_difference,
        rtol=FINITE_DIFFERENCE_RTOL,
        atol=FINITE_DIFFERENCE_ATOL,
    )
    case_pass = (
        source_finite
        & reverse_primal_close
        & forward_primal_close
        & reverse_finite
        & forward_finite
        & gradient_agreement
        & finite_difference_agreement
    )
    return {
        "objective": OBJECTIVE_NAMES[objective_index],
        "case_count": CASE_COUNT,
        "repeat_exact": repeat_exact,
        "case_pass": case_pass.tolist(),
        "pass_count": int(np.sum(case_pass)),
        "ppo_pass_count": int(
            np.sum(
                [
                    passed
                    for passed, arm in zip(case_pass, arms, strict=True)
                    if arm == "ppo"
                ]
            )
        ),
        "diffsim_pass_count": int(
            np.sum(
                [
                    passed
                    for passed, arm in zip(case_pass, arms, strict=True)
                    if arm == "diffsim"
                ]
            )
        ),
        "source_finite": source_finite.tolist(),
        "reverse_primal_close": reverse_primal_close.tolist(),
        "forward_primal_close": forward_primal_close.tolist(),
        "reverse_finite": reverse_finite.tolist(),
        "forward_finite": forward_finite.tolist(),
        "gradient_agreement": gradient_agreement.tolist(),
        "finite_difference_finite": finite_difference_finite.tolist(),
        "finite_difference_agreement": finite_difference_agreement.tolist(),
        "source_primal": _json_safe_numeric(source),
        "reverse_primal": _json_safe_numeric(reverse_primal),
        "forward_primal": _json_safe_numeric(forward_primal),
        "reverse_gradient": _json_safe_numeric(reverse),
        "forward_gradient": _json_safe_numeric(forward),
        "reverse_forward_relative_error": _json_safe_numeric(
            _relative_error(reverse, forward, axis=1)
        ),
        "reverse_gradient_norm": _json_safe_numeric(np.linalg.norm(reverse, axis=1)),
        "forward_gradient_norm": _json_safe_numeric(np.linalg.norm(forward, axis=1)),
        "forward_directional": _json_safe_numeric(forward_directional),
        "finite_difference_directional": _json_safe_numeric(finite_difference),
        "finite_difference_relative_error": _json_safe_numeric(
            _relative_error(forward_directional, finite_difference)
        ),
    }


def classify_derivative_cases(
    *,
    arms: Sequence[str],
    measurement_valid: bool,
    smooth_report: Mapping[str, object],
    reward_report: Mapping[str, object],
) -> dict[str, object]:
    """Classify whether derivative validity depends on action or reward seam."""

    if list(arms) != list(ARM_ORDER) * len(PHASES):
        raise ValueError("paired action arms are not in canonical phase-major order")

    def flags(report: Mapping[str, object]) -> list[bool]:
        values = report.get("case_pass")
        if (
            not isinstance(values, list)
            or len(values) != CASE_COUNT
            or any(type(value) is not bool for value in values)
        ):
            raise ValueError("objective case-pass vector is invalid")
        return values

    smooth = flags(smooth_report)
    reward = flags(reward_report)
    ppo_smooth = [
        value for value, arm in zip(smooth, arms, strict=True) if arm == "ppo"
    ]
    diffsim_smooth = [
        value for value, arm in zip(smooth, arms, strict=True) if arm == "diffsim"
    ]
    if not measurement_valid:
        outcome = "invalid-measurement"
        interpretable = False
    elif all(smooth) and all(reward):
        outcome = "reset-boundary-derivatives-valid"
        interpretable = True
    elif all(smooth):
        outcome = "smooth-physics-valid-reward-derivative-failure"
        interpretable = True
    elif all(ppo_smooth) and not all(diffsim_smooth):
        outcome = "diffsim-action-only-smooth-derivative-failure"
        interpretable = True
    elif all(diffsim_smooth) and not all(ppo_smooth):
        outcome = "ppo-action-only-smooth-derivative-failure"
        interpretable = True
    else:
        outcome = "both-actions-have-smooth-derivative-failures"
        interpretable = True
    return {
        "protocol": "g1-reset-action-derivative-classification-v1",
        "valid": bool(measurement_valid),
        "scientifically_interpretable": interpretable,
        "outcome": outcome,
        "smooth_pass_count": int(sum(smooth)),
        "smooth_ppo_pass_count": int(sum(ppo_smooth)),
        "smooth_diffsim_pass_count": int(sum(diffsim_smooth)),
        "reward_pass_count": int(sum(reward)),
        "policy_evaluation_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
    }


def _validate_e008_audit(audit: Mapping[str, object]) -> None:
    required = {
        "protocol": "g1-success-failure-visitation-reanalysis-independent-audit-v1",
        "valid": True,
        "outcome": "paired-success-failure-visitation-captured",
        "checks_passed": 22,
        "checks_total": 22,
        "source_trajectory_sha256": SOURCE_TRAJECTORY_SHA256,
        "corrected_boundary_count": 34,
        "ppo_survival": [271, 246, 221, 196, 171],
        "diffsim_survival": [124, 135, 81, 92, 79],
        "first_contact_divergence_offsets": [6, 2, 1, 1, 14],
        "policy_retained": False,
        "policy_evaluation_computed": False,
        "simulator_step_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
    }
    mismatches = {
        name: (audit.get(name), expected)
        for name, expected in required.items()
        if audit.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"E008 audit contract changed: {mismatches}")


def _load_source_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {"phases", "metric_names", "termination_limits"}
    for phase in PHASES:
        for arm in ARM_ORDER:
            prefix = f"{arm}_phase_{phase:03d}_"
            required.update(
                {
                    f"{prefix}phase",
                    f"{prefix}qpos",
                    f"{prefix}qvel",
                    f"{prefix}model_action",
                    f"{prefix}position_target",
                    f"{prefix}last_action",
                    f"{prefix}contact_pairs",
                }
            )
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("source trajectory archive lacks reset-action arrays")
        arrays = {name: np.asarray(archive[name]) for name in required}
    if not np.array_equal(arrays["phases"], np.asarray(PHASES)):
        raise ValueError("source phase grid changed")
    return arrays


def _stack_states(states: Sequence[object]) -> object:
    return jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *states,
    )


def _prepare_cases(
    env: object,
    arrays: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[object, jax.Array, dict[str, np.ndarray]]:
    states = []
    actor_actions = []
    model_actions = []
    position_targets = []
    reset_qpos = []
    reset_qvel = []
    reset_qpos_max_abs_delta = []
    source_contact_exact = []
    phases = []
    arms = []
    model_to_actor = np.asarray(env.model_to_actor_permutation, dtype=np.int64)
    actor_to_model = np.asarray(env.actor_to_model_permutation, dtype=np.int64)
    if not np.array_equal(np.argsort(actor_to_model), model_to_actor):
        raise ValueError("environment action permutations are not inverse")

    for phase in PHASES:
        state = env.reset_at_phase(
            jax.random.PRNGKey(seed),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(phase, dtype=jnp.int32),
        )
        state_qpos = np.asarray(state.data.qpos)
        state_qvel = np.asarray(state.data.qvel)
        state_contacts = np.asarray(env.contact_pair_signature(state.data))
        if int(state.info["phase"]) != phase:
            raise ValueError("common environment reset phase changed")
        for arm in ARM_ORDER:
            prefix = f"{arm}_phase_{phase:03d}_"
            source_phase = np.asarray(arrays[f"{prefix}phase"])
            source_qpos = np.asarray(arrays[f"{prefix}qpos"])
            source_qvel = np.asarray(arrays[f"{prefix}qvel"])
            source_model_action = np.asarray(arrays[f"{prefix}model_action"])[0]
            source_target = np.asarray(arrays[f"{prefix}position_target"])[0]
            source_contacts = np.asarray(arrays[f"{prefix}contact_pairs"])[0]
            if source_qpos.ndim == 2:
                try:
                    qpos_max_abs_delta = validate_reconstructed_reset_qpos(
                        state_qpos, source_qpos[0]
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{arm} phase {phase} reset qpos changed: {error}"
                    ) from error
            else:
                qpos_max_abs_delta = math.inf
            source_checks = {
                "phase-rank": source_phase.ndim == 1,
                "phase-value": source_phase.ndim == 1 and source_phase[0] == phase,
                "qpos-rank": source_qpos.ndim == 2,
                "qvel-rank": source_qvel.ndim == 2,
                "qpos-reconstructed": math.isfinite(qpos_max_abs_delta),
                "qvel-exact": source_qvel.ndim == 2
                and np.array_equal(state_qvel, source_qvel[0]),
                "action-shape": source_model_action.shape == (ACTION_DIMENSION,),
                "target-shape": source_target.shape == (ACTION_DIMENSION,),
                "action-finite": np.isfinite(source_model_action).all(),
                "target-finite": np.isfinite(source_target).all(),
            }
            failed_source_checks = [
                name for name, passed in source_checks.items() if not passed
            ]
            if failed_source_checks:
                raise ValueError(
                    f"{arm} phase {phase} reset/action source changed: "
                    f"{failed_source_checks}"
                )
            actor_action = actor_action_from_model_action(
                source_model_action, model_to_actor
            )
            if not np.array_equal(actor_action[actor_to_model], source_model_action):
                raise ValueError("source model action does not round-trip")
            computed_target = np.asarray(
                env.position_target(state, jnp.asarray(actor_action, dtype=jnp.float64))
            )
            if not np.array_equal(computed_target, source_target):
                raise ValueError(f"{arm} phase {phase} position target changed")
            states.append(state)
            actor_actions.append(actor_action)
            model_actions.append(source_model_action)
            position_targets.append(source_target)
            reset_qpos.append(state_qpos)
            reset_qvel.append(state_qvel)
            reset_qpos_max_abs_delta.append(qpos_max_abs_delta)
            source_contact_exact.append(np.array_equal(state_contacts, source_contacts))
            phases.append(phase)
            arms.append(arm)
    metadata = {
        "phases": np.asarray(phases, dtype=np.int64),
        "arms": np.asarray(arms),
        "actor_actions": np.asarray(actor_actions, dtype=np.float64),
        "model_actions": np.asarray(model_actions, dtype=np.float64),
        "position_targets": np.asarray(position_targets, dtype=np.float64),
        "reset_qpos": np.asarray(reset_qpos, dtype=np.float64),
        "reset_qvel": np.asarray(reset_qvel, dtype=np.float64),
        "reset_qpos_max_abs_delta": np.asarray(
            reset_qpos_max_abs_delta, dtype=np.float64
        ),
        "source_contact_exact": np.asarray(source_contact_exact, dtype=bool),
    }
    return (
        _stack_states(states),
        jnp.asarray(metadata["actor_actions"], dtype=jnp.float64),
        metadata,
    )


def _build_compiled_probe(env: object, direction: jax.Array):
    """Build one shared compiled callable for both deterministic invocations."""

    def smooth_objective(state, action):
        next_state = env.step(state, action)
        next_phase = jnp.minimum(
            state.info["phase"] + env.reference_stride,
            env.reference_length - 1,
        )
        return smooth_reference_state_loss(
            next_state.data.qpos,
            next_state.data.qvel,
            env.qpos_reference[next_phase],
            env.qvel_reference[next_phase],
        )

    def reward_objective(state, action):
        return -env.step(state, action).reward

    def case_probe(state, action):
        result = compute_two_objective_derivatives(
            (
                lambda candidate: smooth_objective(state, candidate),
                lambda candidate: reward_objective(state, candidate),
            ),
            action,
            direction=direction,
            finite_difference_epsilon=H1_ACTION_FINITE_DIFFERENCE_EPSILON,
        )
        direct_next = env.step(state, action)
        return {
            **result,
            "direct_done": direct_next.done,
            "direct_terminal": direct_next.info["terminal"],
            "direct_contact_stiffness": direct_next.info[
                "transition_contact_stiffness"
            ],
        }

    return jax.jit(jax.vmap(case_probe))


def _arrays_exact(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> bool:
    return set(first) == set(second) and all(
        np.array_equal(
            np.asarray(first[name]), np.asarray(second[name]), equal_nan=True
        )
        for name in first
    )


def _plot_report(path: Path, report: Mapping[str, object]) -> None:
    phases = np.asarray(report["phases"], dtype=np.int64)
    arms = np.asarray(report["arms"])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, objective_name in enumerate(OBJECTIVE_NAMES):
        objective = report[objective_name]
        passes = (
            np.asarray(objective["case_pass"], dtype=bool)
            .reshape(len(PHASES), len(ARM_ORDER))
            .T
        )
        image = axes[0, column].imshow(
            passes.astype(np.int64), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto"
        )
        axes[0, column].set_xticks(range(len(PHASES)), PHASES)
        axes[0, column].set_yticks(range(len(ARM_ORDER)), ARM_ORDER)
        axes[0, column].set_xlabel("exact reset phase")
        axes[0, column].set_title(f"{objective_name}: complete derivative gate")
        for row in range(len(ARM_ORDER)):
            for col in range(len(PHASES)):
                axes[0, column].text(
                    col,
                    row,
                    "PASS" if passes[row, col] else "FAIL",
                    ha="center",
                    va="center",
                    color="black",
                )
        figure.colorbar(image, ax=axes[0, column], ticks=(0, 1))

        fd_error = np.asarray(
            [
                np.nan if value is None else value
                for value in objective["finite_difference_relative_error"]
            ],
            dtype=np.float64,
        )
        for arm, marker in zip(ARM_ORDER, ("o", "s"), strict=True):
            selected = arms == arm
            axes[1, column].plot(
                phases[selected],
                np.maximum(fd_error[selected], 1e-16),
                marker=marker,
                label=arm,
            )
        axes[1, column].axhline(
            FINITE_DIFFERENCE_RTOL, color="black", linestyle="--", linewidth=1
        )
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("exact reset phase")
        axes[1, column].set_ylabel("forward vs central-FD relative error")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend()
    figure.suptitle(
        "Frozen PPO-success vs DiffSim-failure H1 action derivatives at equal reset states"
    )
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _run(args: argparse.Namespace) -> int:
    paths = {
        "reference": args.reference_path.resolve(),
        "diffsim_hparams": args.diffsim_hparams.resolve(),
        "source_trajectories": args.source_trajectories.resolve(),
        "source_e008_audit": args.source_e008_audit.resolve(),
    }
    expected_hashes = {
        "reference": REFERENCE_SHA256,
        "diffsim_hparams": DIFFSIM_HPARAMS_SHA256,
        "source_trajectories": SOURCE_TRAJECTORY_SHA256,
        "source_e008_audit": SOURCE_E008_AUDIT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")
    if paths["source_trajectories"].name != SOURCE_TRAJECTORY_BASENAME:
        raise ValueError("source trajectory basename changed")

    hparams = read_json(paths["diffsim_hparams"])
    validate_diffsim_hparams(hparams)
    _validate_e008_audit(read_json(paths["source_e008_audit"]))
    source_arrays = _load_source_arrays(paths["source_trajectories"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    preflight = {
        "protocol": "g1-reset-action-derivative-preflight-v1",
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
        states, actions, metadata = _prepare_cases(env, source_arrays, seed=args.seed)
        direction = jax.random.rademacher(
            jax.random.PRNGKey(H1_ACTION_DIRECTION_SEED),
            (ACTION_DIMENSION,),
            dtype=jnp.float64,
        ) / jnp.sqrt(jnp.asarray(ACTION_DIMENSION, dtype=jnp.float64))
        compiled_probe = _build_compiled_probe(env, direction)
        first_device = compiled_probe(states, actions)
        jax.block_until_ready(first_device)
        second_device = compiled_probe(states, actions)
        jax.block_until_ready(second_device)

    first = {name: np.asarray(value) for name, value in first_device.items()}
    second = {name: np.asarray(value) for name, value in second_device.items()}
    repeat_exact = _arrays_exact(first, second)
    direct_done = np.asarray(first["direct_done"], dtype=np.float64)
    direct_terminal = np.asarray(first["direct_terminal"], dtype=np.float64)
    arms = metadata["arms"].tolist()
    smooth_report = _objective_report(first, second, objective_index=0, arms=arms)
    reward_report = _objective_report(first, second, objective_index=1, arms=arms)
    measurement_valid = bool(
        repeat_exact
        and np.all(direct_done == 0.0)
        and np.all(direct_terminal == 0.0)
        and all(
            smooth_report[name] == [True] * CASE_COUNT
            for name in (
                "source_finite",
                "reverse_primal_close",
                "forward_primal_close",
            )
        )
        and all(
            reward_report[name] == [True] * CASE_COUNT
            for name in (
                "source_finite",
                "reverse_primal_close",
                "forward_primal_close",
            )
        )
    )
    classification = classify_derivative_cases(
        arms=arms,
        measurement_valid=measurement_valid,
        smooth_report=smooth_report,
        reward_report=reward_report,
    )
    direction_array = np.asarray(direction, dtype=np.float64)
    raw_arrays = {
        **metadata,
        "direction": direction_array,
        **{f"first_{name}": value for name, value in first.items()},
        **{f"second_{name}": value for name, value in second.items()},
    }
    raw_path = output_root / "reset_action_derivatives.npz"
    _write_npz(raw_path, raw_arrays)
    report = {
        "protocol": "g1-reset-action-derivative-report-v1",
        **classification,
        "code_commit": args.code_commit,
        "phases": metadata["phases"].tolist(),
        "arms": arms,
        "case_count": CASE_COUNT,
        "action_dimension": ACTION_DIMENSION,
        "objectives": list(OBJECTIVE_NAMES),
        "repeat_exact": repeat_exact,
        "reset_qpos_max_abs_delta": metadata["reset_qpos_max_abs_delta"].tolist(),
        "maximum_reset_qpos_abs_delta": float(
            np.max(metadata["reset_qpos_max_abs_delta"])
        ),
        "source_contact_exact": metadata["source_contact_exact"].tolist(),
        "all_direct_done_false": bool(np.all(direct_done == 0.0)),
        "all_direct_terminal_false": bool(np.all(direct_terminal == 0.0)),
        "direction_seed": H1_ACTION_DIRECTION_SEED,
        "direction": direction_array.tolist(),
        "direction_norm": float(np.linalg.norm(direction_array)),
        "finite_difference_epsilon": H1_ACTION_FINITE_DIFFERENCE_EPSILON,
        "primal_tolerances": {"rtol": PRIMAL_RTOL, "atol": PRIMAL_ATOL},
        "gradient_tolerances": {
            "rtol": GRADIENT_RTOL,
            "atol": GRADIENT_ATOL,
        },
        "finite_difference_tolerances": {
            "rtol": FINITE_DIFFERENCE_RTOL,
            "atol": FINITE_DIFFERENCE_ATOL,
        },
        "smooth_reference_state": smooth_report,
        "e002_h1_reward": reward_report,
        "source_trajectory_sha256": SOURCE_TRAJECTORY_SHA256,
        "raw_npz_sha256": sha256_file(raw_path),
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    plot_path = output_root / "derivative_comparison.png"
    _plot_report(plot_path, report)
    summary = {
        "protocol": "g1-reset-action-derivative-summary-v1",
        **classification,
        "repeat_exact": repeat_exact,
        "phases": list(PHASES),
        "case_count": CASE_COUNT,
        "maximum_reset_qpos_abs_delta": float(
            np.max(metadata["reset_qpos_max_abs_delta"])
        ),
        "source_contact_exact_count": int(np.sum(metadata["source_contact_exact"])),
        "smooth_case_pass": smooth_report["case_pass"],
        "reward_case_pass": reward_report["case_pass"],
        "raw_npz_sha256": sha256_file(raw_path),
        "report_sha256": sha256_file(report_path),
        "plot_sha256": sha256_file(plot_path),
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-reset-action-derivative-completion-v1",
        "valid": classification["scientifically_interpretable"],
        "outcome": classification["outcome"],
        "computed_probe_invocations": 2,
        "policy_evaluation_computed": False,
        "simulator_step_computed": True,
        "derivative_computed": True,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "reset_action_derivatives.npz": sha256_file(raw_path),
            "report.json": sha256_file(report_path),
            "derivative_comparison.png": sha256_file(plot_path),
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
