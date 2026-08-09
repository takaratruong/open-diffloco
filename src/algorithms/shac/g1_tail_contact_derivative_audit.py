"""Pure selection and derivative gates for the bounded E012 tail audit."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

_SHARD_SEEDS = (0, 1, 2, 3)
_PHASE_BIN_COUNT = 5
_POPULATION = 64
_ACTION_DIMENSION = 29
_FD_RELATIVE_TOLERANCE = 0.05
_FD_SMALL_MAGNITUDE = 1e-5
_FD_ABSOLUTE_TOLERANCE = 1e-6
_FORWARD_REPEAT_MAXIMUM_ABSOLUTE_ERROR = 1e-6
_REVERSE_FORWARD_MINIMUM_COSINE = 0.999
_REVERSE_FORWARD_RELATIVE_L2_TOLERANCE = 0.01


@dataclass(frozen=True)
class TailFragment:
    """Identity of one shard-local, phase-local highest-loss fragment."""

    shard_seed: int
    phase_bin: int
    environment_index: int
    initial_phase: int
    loss: float


@dataclass(frozen=True)
class DerivativeComparison:
    """Pure validity receipt for one selected first-action derivative."""

    shard_seed: int
    phase_bin: int
    forward_finite: bool
    forward_repeat_maximum_absolute_error: float | None
    forward_repeat_valid: bool
    forward_directional_derivative: float | None
    directional_finite_difference: float | None
    forward_fd_absolute_error: float | None
    forward_fd_relative_error: float | None
    forward_fd_valid: bool
    forward_valid: bool
    reverse_finite: bool
    reverse_forward_cosine: float | None
    reverse_forward_relative_l2: float | None
    reverse_parity_valid: bool


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


def _require_seed_bin(shard_seed: int, phase_bin: int) -> None:
    if type(shard_seed) is not int or shard_seed not in _SHARD_SEEDS:
        raise ValueError("shard seed must be one of 0, 1, 2, 3")
    if type(phase_bin) is not int or not 0 <= phase_bin < _PHASE_BIN_COUNT:
        raise ValueError("phase bin must be an integer in range [0, 5)")


def _require_exact_shard_keys(values: Mapping[int, Any], *, label: str) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    keys = tuple(values.keys())
    if any(type(key) is not int for key in keys) or set(keys) != set(_SHARD_SEEDS):
        raise ValueError(f"{label} shard seeds must be exactly 0, 1, 2, 3")


def select_rank_zero_fragments(
    losses_by_shard: Mapping[int, Any],
    initial_phases_by_shard: Mapping[int, Any],
) -> tuple[TailFragment, ...]:
    """Select one stable highest-loss fragment per frozen shard and phase bin."""

    _require_exact_shard_keys(losses_by_shard, label="losses")
    _require_exact_shard_keys(initial_phases_by_shard, label="initial phases")
    selected = []
    for seed in _SHARD_SEEDS:
        losses = _host_array(losses_by_shard[seed])
        phases = _host_array(initial_phases_by_shard[seed])
        if losses.shape != (_POPULATION,) or phases.shape != (_POPULATION,):
            raise ValueError("loss and initial-phase arrays must have shape (64,)")
        if not np.issubdtype(losses.dtype, np.floating):
            raise TypeError("losses must have floating dtype")
        if not np.isfinite(losses).all():
            raise ValueError("losses must be finite")
        if not np.issubdtype(phases.dtype, np.integer):
            raise TypeError("initial phases must have integer dtype")
        if not np.all((phases >= 0) & (phases < 500)):
            raise ValueError("initial phases must be in range [0, 500)")
        bins = phases // 100
        for phase_bin in range(_PHASE_BIN_COUNT):
            indices = np.flatnonzero(bins == phase_bin)
            if indices.size == 0:
                raise ValueError("every shard phase bin must be nonempty")
            local_losses = losses[indices]
            environment_index = int(indices[int(np.argmax(local_losses))])
            selected.append(
                TailFragment(
                    shard_seed=seed,
                    phase_bin=phase_bin,
                    environment_index=environment_index,
                    initial_phase=int(phases[environment_index]),
                    loss=float(losses[environment_index]),
                )
            )
    return tuple(selected)


def canonical_forward_scalar_gradient(
    directional_function: Callable[..., tuple[Any, Any]],
    action: Any,
    *function_args: Any,
) -> tuple[Any, jax.Array]:
    """Assemble a scalar gradient from all 29 canonical forward JVPs."""

    action_array = _host_array(action)
    if action_array.shape != (_ACTION_DIMENSION,):
        raise ValueError("action must have shape (29,)")
    if not np.issubdtype(action_array.dtype, np.floating):
        raise TypeError("action must have floating dtype")
    if not np.isfinite(action_array).all():
        raise ValueError("action must be finite")
    primal = None
    derivatives = []
    basis = np.eye(_ACTION_DIMENSION, dtype=action_array.dtype)
    for tangent in basis:
        current_primal, current_derivative = directional_function(
            action, jp.asarray(tangent), *function_args
        )
        current_primal_array = _host_array(current_primal)
        current_derivative_array = _host_array(current_derivative)
        if current_primal_array.shape != () or current_derivative_array.shape != ():
            raise ValueError("forward JVP primal and derivative must be scalars")
        if primal is None:
            if not np.isfinite(current_primal_array):
                raise ValueError("forward JVP shared primal must be finite")
            primal = current_primal
        elif not _exact_array(current_primal, primal):
            raise ValueError("forward JVP calls do not share an exact shared primal")
        derivatives.append(jp.asarray(current_derivative))
    return primal, jp.stack(derivatives)


def seeded_unbounded_unit_direction(action: Any, *, seed: int) -> np.ndarray:
    """Return a deterministic unit direction for finite unbounded actions."""

    action_array = _host_array(action)
    if action_array.shape != (_ACTION_DIMENSION,):
        raise ValueError("action must have shape (29,)")
    if not np.issubdtype(action_array.dtype, np.floating):
        raise TypeError("action must have floating dtype")
    if not np.isfinite(action_array).all():
        raise ValueError("action must be finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    direction64 = jax.random.normal(
        jax.random.PRNGKey(int(seed)),
        (_ACTION_DIMENSION,),
        dtype=jp.float64,
    )
    if direction64.dtype != jp.float64:
        raise ValueError("seeded direction requires JAX float64 mode")
    direction64 = direction64 / jp.linalg.norm(direction64)
    direction = np.asarray(
        jax.device_get(direction64), dtype=action_array.dtype
    )
    if not np.isfinite(direction).all():
        raise ValueError("seeded direction must be finite")
    return direction


def compare_derivative_case(
    *,
    shard_seed: int,
    phase_bin: int,
    forward_gradients: Any,
    reverse_gradient: Any,
    finite_difference_direction: Any,
    directional_finite_difference: Any,
) -> DerivativeComparison:
    """Apply the frozen forward, finite-difference, and reverse parity gates."""

    _require_seed_bin(shard_seed, phase_bin)
    forward = _host_array(forward_gradients)
    reverse = _host_array(reverse_gradient)
    direction = _host_array(finite_difference_direction)
    if forward.shape != (3, 29):
        raise ValueError("forward gradients must have shape (3, 29)")
    if reverse.shape != (29,) or direction.shape != (29,):
        raise ValueError("reverse gradient and direction must have shape (29,)")
    if not all(np.issubdtype(value.dtype, np.floating) for value in (forward, reverse, direction)):
        raise TypeError("derivative arrays must have floating dtype")
    if not np.isfinite(direction).all() or not np.isclose(
        np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError("finite-difference direction must be finite and unit norm")
    try:
        directional_fd = float(directional_finite_difference)
    except (TypeError, ValueError) as error:
        raise TypeError("directional finite difference must be a scalar") from error

    forward_finite = bool(np.isfinite(forward).all())
    forward_repeat_maximum_absolute_error = None
    forward_repeat_valid = False
    directional_ad = None
    absolute_error = None
    relative_error = None
    forward_fd_valid = False
    if forward_finite:
        forward64 = forward.astype(np.float64)
        forward_repeat_maximum_absolute_error = float(
            np.max(np.ptp(forward64, axis=0))
        )
        forward_repeat_valid = bool(
            forward_repeat_maximum_absolute_error
            <= _FORWARD_REPEAT_MAXIMUM_ABSOLUTE_ERROR
        )
        directional_ad = float(np.dot(forward64[0], direction))
        if np.isfinite(directional_ad) and np.isfinite(directional_fd):
            absolute_error = abs(directional_ad - directional_fd)
            relative_error = absolute_error / max(abs(directional_fd), 1e-8)
            if abs(directional_fd) < _FD_SMALL_MAGNITUDE:
                forward_fd_valid = absolute_error <= _FD_ABSOLUTE_TOLERANCE
            else:
                forward_fd_valid = relative_error <= _FD_RELATIVE_TOLERANCE
    forward_valid = bool(
        forward_finite and forward_repeat_valid and forward_fd_valid
    )

    reverse_finite = bool(np.isfinite(reverse).all())
    cosine = None
    relative_l2 = None
    reverse_parity_valid = False
    if reverse_finite and np.isfinite(forward[0]).all():
        forward64 = forward[0].astype(np.float64)
        reverse64 = reverse.astype(np.float64)
        forward_norm = float(np.linalg.norm(forward64))
        reverse_norm = float(np.linalg.norm(reverse64))
        if forward_norm == 0.0 or reverse_norm == 0.0:
            cosine = 1.0 if _exact_array(forward[0], reverse) else 0.0
        else:
            cosine = float(
                np.dot(reverse64, forward64) / (reverse_norm * forward_norm)
            )
        relative_l2 = float(
            np.linalg.norm(reverse64 - forward64) / max(forward_norm, 1e-12)
        )
        reverse_parity_valid = bool(
            cosine >= _REVERSE_FORWARD_MINIMUM_COSINE
            and relative_l2 <= _REVERSE_FORWARD_RELATIVE_L2_TOLERANCE
        )

    return DerivativeComparison(
        shard_seed=shard_seed,
        phase_bin=phase_bin,
        forward_finite=forward_finite,
        forward_repeat_maximum_absolute_error=(
            forward_repeat_maximum_absolute_error
        ),
        forward_repeat_valid=forward_repeat_valid,
        forward_directional_derivative=directional_ad,
        directional_finite_difference=(
            directional_fd if np.isfinite(directional_fd) else None
        ),
        forward_fd_absolute_error=absolute_error,
        forward_fd_relative_error=relative_error,
        forward_fd_valid=forward_fd_valid,
        forward_valid=forward_valid,
        reverse_finite=reverse_finite,
        reverse_forward_cosine=cosine,
        reverse_forward_relative_l2=relative_l2,
        reverse_parity_valid=reverse_parity_valid,
    )


def classify_derivative_cases(
    cases: Sequence[DerivativeComparison], *, execution_valid: bool = True
) -> str:
    """Select the preregistered E012 outcome for 20 complete comparisons."""

    expected = {(seed, phase_bin) for seed in _SHARD_SEEDS for phase_bin in range(5)}
    if execution_valid is not True or not isinstance(cases, Sequence):
        return "invalid-execution"
    if len(cases) != len(expected) or any(
        not isinstance(case, DerivativeComparison) for case in cases
    ):
        return "invalid-execution"
    identities = {(case.shard_seed, case.phase_bin) for case in cases}
    if identities != expected:
        return "invalid-execution"
    if any(not case.forward_valid for case in cases):
        return "forward-contact-derivative-invalid"
    if any(not case.reverse_parity_valid for case in cases):
        return "forward-rescues-reverse-tail-adjoint"
    return "reverse-and-forward-valid"
