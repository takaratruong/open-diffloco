"""One-case runtime for the bounded E012 first-action derivative audit."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.g1_gradient_audit import FirstActionObjective
from src.algorithms.shac.g1_tail_contact_derivative_audit import (
    DerivativeComparison,
    canonical_forward_scalar_gradient,
    compare_derivative_case,
    seeded_unbounded_unit_direction,
)

_ACTION_DIMENSION = 29
_HORIZON = 48
_FORWARD_SWEEPS = 3
_FINITE_DIFFERENCE_EPSILON = 0.001


@dataclass(frozen=True)
class CaseEvaluation:
    """Complete numerical and timing evidence for one E012 fragment."""

    shard_seed: int
    phase_bin: int
    direction_seed: int
    finite_difference_epsilon: float
    nominal_action: np.ndarray
    nominal_objective: np.ndarray
    reverse_primal: np.ndarray
    reverse_gradient: np.ndarray
    reverse_duration_seconds: float
    forward_primals: np.ndarray
    forward_gradients: np.ndarray
    forward_durations_seconds: np.ndarray
    direction: np.ndarray
    positive_action: np.ndarray
    negative_action: np.ndarray
    probe_objectives: np.ndarray
    probe_durations_seconds: np.ndarray
    directional_finite_difference: np.ndarray
    positive_dones: np.ndarray
    negative_dones: np.ndarray
    positive_done_exact: bool
    negative_done_exact: bool
    positive_support_exact: bool
    negative_support_exact: bool
    probes_preserve_done_and_support: bool
    comparison: DerivativeComparison
    execution_valid: bool
    case_outcome: str


def _host_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _exact_array(left: Any, right: Any) -> bool:
    left_array = np.ascontiguousarray(_host_array(left))
    right_array = np.ascontiguousarray(_host_array(right))
    return bool(
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes() == right_array.tobytes()
    )


def _case_outcome(comparison: DerivativeComparison) -> str:
    if not comparison.forward_valid:
        return "forward-contact-derivative-invalid"
    if not comparison.reverse_parity_valid:
        return "forward-rescues-reverse-tail-adjoint"
    return "reverse-and-forward-valid"


def _validate_nominal(diagnostic: FirstActionObjective) -> tuple[np.ndarray, ...]:
    if not isinstance(diagnostic, FirstActionObjective):
        raise TypeError("diagnostic must be a FirstActionObjective")
    action = _host_array(diagnostic.nominal_first_action)
    objective = _host_array(diagnostic.nominal_objective)
    dones = _host_array(diagnostic.nominal_trajectory.dones)
    actions = _host_array(diagnostic.nominal_trajectory.actions)
    if action.shape != (_ACTION_DIMENSION,):
        raise ValueError("nominal action must have shape (29,)")
    if action.dtype != np.dtype(np.float64):
        raise TypeError("nominal action must have float64 dtype")
    if objective.shape != () or not np.isfinite(objective):
        raise ValueError("nominal objective must be a finite scalar")
    if dones.shape != (_HORIZON,):
        raise ValueError("nominal dones must have shape (48,)")
    if actions.shape != (_HORIZON, _ACTION_DIMENSION):
        raise ValueError("nominal trajectory actions must have shape (48, 29)")
    if not np.isfinite(action).all() or not np.isfinite(actions).all():
        raise ValueError("nominal action support must be finite")
    return action, objective, dones, actions


def evaluate_case(
    diagnostic: FirstActionObjective,
    *,
    shard_seed: int,
    phase_bin: int,
    direction_seed: int,
) -> CaseEvaluation:
    """Evaluate one first-action derivative without constructing an update."""

    nominal_action, nominal_objective, nominal_dones, nominal_actions = (
        _validate_nominal(diagnostic)
    )
    if type(shard_seed) is not int or shard_seed not in (0, 1, 2, 3):
        raise ValueError("shard seed must be one of 0, 1, 2, 3")
    if type(phase_bin) is not int or not 0 <= phase_bin < 5:
        raise ValueError("phase bin must be an integer in range [0, 5)")
    expected_direction_seed = 12001 + shard_seed * 5 + phase_bin
    if (
        type(direction_seed) is not int
        or direction_seed != expected_direction_seed
    ):
        raise ValueError(
            "direction_seed must equal canonical direction seed "
            f"{expected_direction_seed}"
        )

    action = jp.asarray(diagnostic.nominal_first_action)
    reverse_started = time.perf_counter()
    reverse_primal, reverse_gradient = jax.value_and_grad(
        diagnostic.objective
    )(action)
    jax.block_until_ready((reverse_primal, reverse_gradient))
    reverse_duration = time.perf_counter() - reverse_started
    if not _exact_array(reverse_primal, nominal_objective):
        raise ValueError("reverse derivative primal differs from nominal objective")

    def directional(current_action: jax.Array, tangent: jax.Array):
        return jax.jvp(
            diagnostic.objective,
            (current_action,),
            (tangent,),
        )

    forward_primals = []
    forward_gradients = []
    forward_durations = []
    for _ in range(_FORWARD_SWEEPS):
        started = time.perf_counter()
        primal, gradient = canonical_forward_scalar_gradient(directional, action)
        jax.block_until_ready((primal, gradient))
        forward_durations.append(time.perf_counter() - started)
        if not _exact_array(primal, nominal_objective):
            raise ValueError("forward derivative primal differs from nominal objective")
        forward_primals.append(_host_array(primal))
        forward_gradients.append(_host_array(gradient))
    forward_primals_array = np.stack(forward_primals)
    forward_gradients_array = np.stack(forward_gradients)

    direction = seeded_unbounded_unit_direction(action, seed=direction_seed)
    epsilon = jp.asarray(_FINITE_DIFFERENCE_EPSILON, dtype=action.dtype)
    positive_action = action + epsilon * jp.asarray(direction)
    negative_action = action - epsilon * jp.asarray(direction)
    probe_objectives = []
    probe_durations = []
    probe_trajectories = []
    for probe_action in (positive_action, negative_action):
        started = time.perf_counter()
        probe_objective = diagnostic.objective(probe_action)
        probe_trajectory, _ = diagnostic.rollout(probe_action)
        jax.block_until_ready((probe_objective, probe_trajectory))
        probe_durations.append(time.perf_counter() - started)
        probe_objectives.append(_host_array(probe_objective))
        probe_trajectories.append(probe_trajectory)
    probe_objectives_array = np.stack(probe_objectives)
    directional_fd = (probe_objectives_array[0] - probe_objectives_array[1]) / (
        2.0 * _FINITE_DIFFERENCE_EPSILON
    )

    probe_dones = [_host_array(value.dones) for value in probe_trajectories]
    probe_actions = [_host_array(value.actions) for value in probe_trajectories]
    done_exact = [
        _exact_array(value, nominal_dones) for value in probe_dones
    ]
    nominal_support = np.isfinite(nominal_actions)
    support_exact = [
        bool(
            value.shape == nominal_actions.shape
            and np.array_equal(np.isfinite(value), nominal_support)
        )
        for value in probe_actions
    ]
    probes_preserve = bool(all(done_exact) and all(support_exact))

    comparison = compare_derivative_case(
        shard_seed=shard_seed,
        phase_bin=phase_bin,
        forward_gradients=forward_gradients_array,
        reverse_gradient=_host_array(reverse_gradient),
        finite_difference_direction=direction,
        directional_finite_difference=directional_fd,
    )
    if not probes_preserve:
        comparison = replace(
            comparison,
            forward_fd_valid=False,
            forward_valid=False,
        )

    return CaseEvaluation(
        shard_seed=shard_seed,
        phase_bin=phase_bin,
        direction_seed=int(direction_seed),
        finite_difference_epsilon=_FINITE_DIFFERENCE_EPSILON,
        nominal_action=nominal_action,
        nominal_objective=nominal_objective,
        reverse_primal=_host_array(reverse_primal),
        reverse_gradient=_host_array(reverse_gradient),
        reverse_duration_seconds=reverse_duration,
        forward_primals=forward_primals_array,
        forward_gradients=forward_gradients_array,
        forward_durations_seconds=np.asarray(forward_durations),
        direction=direction,
        positive_action=_host_array(positive_action),
        negative_action=_host_array(negative_action),
        probe_objectives=probe_objectives_array,
        probe_durations_seconds=np.asarray(probe_durations),
        directional_finite_difference=np.asarray(directional_fd),
        positive_dones=probe_dones[0],
        negative_dones=probe_dones[1],
        positive_done_exact=done_exact[0],
        negative_done_exact=done_exact[1],
        positive_support_exact=support_exact[0],
        negative_support_exact=support_exact[1],
        probes_preserve_done_and_support=probes_preserve,
        comparison=comparison,
        execution_valid=True,
        case_outcome=_case_outcome(comparison),
    )
