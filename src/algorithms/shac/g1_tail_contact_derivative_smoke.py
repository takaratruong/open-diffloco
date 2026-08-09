"""Compiled runtime for the one canonical G1 tail-contact smoke case.

This module deliberately exposes no orchestration or output publication.  It
only compiles and executes the shard-0, bin-0 diagnostic needed to decide
whether a full forward-mode SHAC experiment is warranted.
"""

from __future__ import annotations

import dataclasses
import resource
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.g1_gradient_audit import FirstActionObjective
from src.algorithms.shac.g1_tail_contact_derivative_audit import (
    DerivativeComparison,
    compare_derivative_case,
    seeded_unbounded_unit_direction,
)

_ACTION_DIMENSION = 29
_HORIZON = 48
_FORWARD_SWEEPS = 3
_FINITE_DIFFERENCE_EPSILON = 0.001
_CANONICAL_SHARD_SEED = 0
_CANONICAL_PHASE_BIN = 0
_CANONICAL_DIRECTION_SEED = 12001
_MINIMUM_DEVICE_HEADROOM_BYTES = 2 * 1024**3
_MAXIMUM_PROJECTED_TWENTY_CASE_SECONDS = 3600.0
_FORBIDDEN_EXPERIMENT = "E-20260809-012"
_PUBLICATION_BUDGET_SECONDS = 5.0


@dataclass(frozen=True)
class CompiledCaseSmoke:
    """Reusable, separately-compiled kernels for the only admissible case."""

    diagnostic: FirstActionObjective
    shard_seed: int
    phase_bin: int
    direction_seed: int
    nominal_action: np.ndarray
    nominal_objective: np.ndarray
    reverse_kernel: Callable[[jax.Array], tuple[jax.Array, jax.Array]]
    directional_jvp_kernel: Callable[
        [jax.Array, jax.Array], tuple[jax.Array, jax.Array]
    ]
    reverse_compile_duration_seconds: float
    forward_compile_duration_seconds: float


@dataclass(frozen=True)
class CaseSmokeReceipt:
    """Complete numerical and timing evidence from one compiled smoke run."""

    shard_seed: int
    phase_bin: int
    direction_seed: int
    finite_difference_epsilon: float
    nominal_action: np.ndarray
    nominal_objective: np.ndarray
    reverse_compile_duration_seconds: float
    forward_compile_duration_seconds: float
    reverse_primal: np.ndarray
    reverse_gradient: np.ndarray
    reverse_cached_duration_seconds: float
    forward_primals: np.ndarray
    forward_gradients: np.ndarray
    forward_cached_sweep_durations_seconds: np.ndarray
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


def _require_canonical_case(
    *, shard_seed: int, phase_bin: int, direction_seed: int
) -> None:
    if (
        type(shard_seed) is not int
        or type(phase_bin) is not int
        or type(direction_seed) is not int
        or (shard_seed, phase_bin, direction_seed)
        != (
            _CANONICAL_SHARD_SEED,
            _CANONICAL_PHASE_BIN,
            _CANONICAL_DIRECTION_SEED,
        )
    ):
        raise ValueError(
            "the compiled smoke permits only the canonical case: "
            "shard_seed=0, phase_bin=0, direction_seed=12001"
        )


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
    if not np.issubdtype(objective.dtype, np.floating):
        raise TypeError("nominal objective must have floating dtype")
    if objective.shape != () or not np.isfinite(objective):
        raise ValueError("nominal objective must be a finite scalar")
    if dones.shape != (_HORIZON,):
        raise ValueError("nominal dones must have shape (48,)")
    if dones.dtype != np.dtype(np.bool_):
        raise TypeError("nominal dones must have bool dtype")
    if actions.shape != (_HORIZON, _ACTION_DIMENSION):
        raise ValueError("nominal trajectory actions must have shape (48, 29)")
    if not np.issubdtype(actions.dtype, np.floating):
        raise TypeError("nominal trajectory actions must have floating dtype")
    if not np.isfinite(action).all() or not np.isfinite(actions).all():
        raise ValueError("nominal action support must be finite")
    return action, objective, dones, actions


def compile_case_kernels(
    diagnostic: FirstActionObjective,
    *,
    shard_seed: int = _CANONICAL_SHARD_SEED,
    phase_bin: int = _CANONICAL_PHASE_BIN,
    direction_seed: int = _CANONICAL_DIRECTION_SEED,
) -> CompiledCaseSmoke:
    """Compile one reverse and one directional-JVP kernel for shard 0/bin 0."""

    _require_canonical_case(
        shard_seed=shard_seed,
        phase_bin=phase_bin,
        direction_seed=direction_seed,
    )
    nominal_action, nominal_objective, _, _ = _validate_nominal(diagnostic)
    action = jp.asarray(diagnostic.nominal_first_action)

    reverse_jitted = jax.jit(jax.value_and_grad(diagnostic.objective))
    reverse_started = time.perf_counter()
    reverse_kernel = reverse_jitted.lower(action).compile()
    reverse_compile_duration = time.perf_counter() - reverse_started

    def directional_jvp(
        current_action: jax.Array, tangent: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.jvp(diagnostic.objective, (current_action,), (tangent,))

    directional_jvp_jitted = jax.jit(directional_jvp)
    first_tangent = jp.asarray(np.eye(_ACTION_DIMENSION, dtype=action.dtype)[0])
    forward_started = time.perf_counter()
    directional_jvp_kernel = directional_jvp_jitted.lower(
        action, first_tangent
    ).compile()
    forward_compile_duration = time.perf_counter() - forward_started

    return CompiledCaseSmoke(
        diagnostic=diagnostic,
        shard_seed=shard_seed,
        phase_bin=phase_bin,
        direction_seed=direction_seed,
        nominal_action=nominal_action,
        nominal_objective=nominal_objective,
        reverse_kernel=reverse_kernel,
        directional_jvp_kernel=directional_jvp_kernel,
        reverse_compile_duration_seconds=reverse_compile_duration,
        forward_compile_duration_seconds=forward_compile_duration,
    )


def run_compiled_case_smoke(compiled: CompiledCaseSmoke) -> CaseSmokeReceipt:
    """Run the cached canonical kernels, sequential JVP sweeps, and probes."""

    if not isinstance(compiled, CompiledCaseSmoke):
        raise TypeError("compiled must be a CompiledCaseSmoke")
    _require_canonical_case(
        shard_seed=compiled.shard_seed,
        phase_bin=compiled.phase_bin,
        direction_seed=compiled.direction_seed,
    )
    _, _, nominal_dones, nominal_actions = _validate_nominal(compiled.diagnostic)
    action = jp.asarray(compiled.diagnostic.nominal_first_action)

    reverse_started = time.perf_counter()
    reverse_primal, reverse_gradient = compiled.reverse_kernel(action)
    jax.block_until_ready((reverse_primal, reverse_gradient))
    reverse_cached_duration = time.perf_counter() - reverse_started
    if not _exact_array(reverse_primal, compiled.nominal_objective):
        raise ValueError("reverse derivative primal differs from nominal objective")

    basis = np.eye(_ACTION_DIMENSION, dtype=action.dtype)
    forward_primals = []
    forward_gradients = []
    forward_durations = []
    for _ in range(_FORWARD_SWEEPS):
        started = time.perf_counter()
        primals = []
        derivatives = []
        for tangent in basis:
            primal, derivative = compiled.directional_jvp_kernel(
                action, jp.asarray(tangent)
            )
            jax.block_until_ready((primal, derivative))
            primals.append(primal)
            derivatives.append(derivative)
        forward_durations.append(time.perf_counter() - started)
        if any(
            not _exact_array(primal, compiled.nominal_objective)
            for primal in primals
        ):
            raise ValueError("forward derivative primal differs from nominal objective")
        forward_primals.append(_host_array(primals[0]))
        forward_gradients.append(
            np.stack([_host_array(derivative) for derivative in derivatives])
        )
    forward_primals_array = np.stack(forward_primals)
    forward_gradients_array = np.stack(forward_gradients)

    direction = seeded_unbounded_unit_direction(
        action, seed=compiled.direction_seed
    )
    epsilon = jp.asarray(_FINITE_DIFFERENCE_EPSILON, dtype=action.dtype)
    positive_action = action + epsilon * jp.asarray(direction)
    negative_action = action - epsilon * jp.asarray(direction)
    probe_objectives = []
    probe_durations = []
    probe_trajectories = []
    for probe_action in (positive_action, negative_action):
        started = time.perf_counter()
        probe_objective = compiled.diagnostic.objective(probe_action)
        probe_trajectory, _ = compiled.diagnostic.rollout(probe_action)
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
    done_exact = [_exact_array(value, nominal_dones) for value in probe_dones]
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
        shard_seed=compiled.shard_seed,
        phase_bin=compiled.phase_bin,
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

    return CaseSmokeReceipt(
        shard_seed=compiled.shard_seed,
        phase_bin=compiled.phase_bin,
        direction_seed=compiled.direction_seed,
        finite_difference_epsilon=_FINITE_DIFFERENCE_EPSILON,
        nominal_action=compiled.nominal_action,
        nominal_objective=compiled.nominal_objective,
        reverse_compile_duration_seconds=compiled.reverse_compile_duration_seconds,
        forward_compile_duration_seconds=compiled.forward_compile_duration_seconds,
        reverse_primal=_host_array(reverse_primal),
        reverse_gradient=_host_array(reverse_gradient),
        reverse_cached_duration_seconds=reverse_cached_duration,
        forward_primals=forward_primals_array,
        forward_gradients=forward_gradients_array,
        forward_cached_sweep_durations_seconds=np.asarray(forward_durations),
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


def _validate_operational_output(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    for candidate in (output_dir, *output_dir.parents):
        if (
            candidate.name == _FORBIDDEN_EXPERIMENT
            and candidate.parent.name == "runs"
        ):
            raise ValueError(
                "the operational smoke must not write within runs/E-20260809-012"
            )
    receipt_path = output_dir / "smoke_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("smoke receipt already exists; refusing to overwrite it")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("smoke output directory must be empty")
    return receipt_path


def _select_bin_zero(losses: Any, phases: Any) -> dict[str, Any]:
    losses_array = np.asarray(jax.device_get(losses))
    phases_array = np.asarray(jax.device_get(phases))
    if losses_array.shape != (64,) or phases_array.shape != (64,):
        raise ValueError("authoritative shard-0 losses and phases must have shape (64,)")
    if not np.issubdtype(losses_array.dtype, np.floating):
        raise TypeError("authoritative shard-0 losses must have floating dtype")
    if not np.isfinite(losses_array).all():
        raise ValueError("authoritative shard-0 losses must be finite")
    if not np.issubdtype(phases_array.dtype, np.integer):
        raise TypeError("authoritative shard-0 phases must have integer dtype")
    eligible = np.flatnonzero((phases_array >= 0) & (phases_array < 100))
    if eligible.size == 0:
        raise ValueError("authoritative shard 0 has no phase-bin-0 fragment")
    # np.argmax is stable: an exact loss tie selects the lower environment index.
    environment_index = int(eligible[int(np.argmax(losses_array[eligible]))])
    return {
        "shard_seed": _CANONICAL_SHARD_SEED,
        "phase_bin": _CANONICAL_PHASE_BIN,
        "environment_index": environment_index,
        "initial_phase": int(phases_array[environment_index]),
        "loss": float(losses_array[environment_index]),
    }


def _default_memory_snapshot() -> dict[str, Any]:
    """Return best-effort allocator telemetry without making it a dependency."""

    try:
        device = jax.devices()[0]
        stats = device.memory_stats()
    except (AttributeError, IndexError, RuntimeError, TypeError) as error:
        return {"available": False, "reason": type(error).__name__}
    if not isinstance(stats, Mapping):
        return {"available": False, "reason": "allocator-stats-unavailable"}
    normalized = {}
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
        value = stats.get(key)
        if isinstance(value, (int, np.integer)):
            normalized[key] = int(value)
    available = "peak_bytes_in_use" in normalized and "bytes_limit" in normalized
    return {
        "available": available,
        "device": str(device),
        **normalized,
        **({} if available else {"reason": "required-counters-unavailable"}),
    }


def _default_source_hashes(e011_run_dir: Path) -> dict[str, str]:
    from tools.audit_g1_shac_gradient_quality import sha256_file

    relative_paths = (
        Path("experiment.yaml"),
        Path("run.json"),
        Path("seed-1/evidence/manifest.json"),
        Path("seed-1/evidence/outcome.json"),
        Path("seed-1/evidence/validity.json"),
        Path("seed-1/evidence/failure_weight_receipts.json"),
        Path("seed-1/evidence/estimator_receipts.json"),
    )
    return {
        path.as_posix(): sha256_file(Path(e011_run_dir) / path)
        for path in relative_paths
    }


def _memory_gate(snapshots: list[dict[str, Any]]) -> tuple[bool, int | None]:
    available = [snapshot for snapshot in snapshots if snapshot.get("available")]
    if not available:
        # Completion without an allocator/OOM failure is the documented fallback.
        return True, None
    headroom = min(
        int(snapshot["bytes_limit"]) - int(snapshot["peak_bytes_in_use"])
        for snapshot in available
    )
    return headroom >= _MINIMUM_DEVICE_HEADROOM_BYTES, headroom


def _telemetry_json(
    value: Any,
    *,
    path: str = "numerical",
    nonfinite_counts: dict[str, int] | None = None,
) -> tuple[Any, dict[str, int]]:
    """Encode nonfinite diagnostic values as null without losing their location."""

    counts = nonfinite_counts if nonfinite_counts is not None else {}
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif hasattr(value, "_asdict"):
        value = value._asdict()
    if isinstance(value, Mapping):
        return (
            {
                str(key): _telemetry_json(
                    child,
                    path=f"{path}.{key}",
                    nonfinite_counts=counts,
                )[0]
                for key, child in value.items()
            },
            counts,
        )
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(jax.device_get(value))
        return _telemetry_json(
            array.item() if array.ndim == 0 else array.tolist(),
            path=path,
            nonfinite_counts=counts,
        )
    if isinstance(value, (list, tuple)):
        return (
            [
                _telemetry_json(
                    child,
                    path=f"{path}[{index}]",
                    nonfinite_counts=counts,
                )[0]
                for index, child in enumerate(value)
            ],
            counts,
        )
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            counts[path] = counts.get(path, 0) + 1
            return None, counts
        return float(value), counts
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value), counts
    if isinstance(value, (bool, str)) or value is None:
        return value, counts
    raise TypeError(f"telemetry value at {path} is not JSON serializable: {type(value)!r}")


def run_one_case_smoke(
    contract: Any,
    e011_run_dir: Path,
    *,
    load_source_receipts_impl: Callable[[Path], Any] | None = None,
    prepare_e064_execution_impl: Callable[[Any], Any] | None = None,
    make_action_noise_impl: Callable[[int], Any] | None = None,
    compile_case_kernels_impl: Callable[[Any], CompiledCaseSmoke] | None = None,
    run_compiled_case_smoke_impl: Callable[[CompiledCaseSmoke], Any] | None = None,
    memory_snapshot_impl: Callable[[], dict[str, Any]] | None = None,
    source_hashes_impl: Callable[[Path], Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Replay E011 shard 0 and publish the only permitted operational smoke."""

    from src.algorithms.shac.g1_gradient_audit_runner import to_finite_json
    from tools.audit_g1_shac_gradient_quality import write_json_atomically

    total_started = time.perf_counter()
    receipt_path = _validate_operational_output(Path(contract.output_dir))
    memory_snapshot = memory_snapshot_impl or _default_memory_snapshot
    memory_snapshots = [{"boundary": "start", **memory_snapshot()}]

    if load_source_receipts_impl is None:
        from src.algorithms.shac.g1_tail_contact_derivative_source import (
            load_e011_source_receipts,
        )

        load_source_receipts_impl = load_e011_source_receipts
    source_started = time.perf_counter()
    source = load_source_receipts_impl(Path(e011_run_dir))
    source_load_duration = time.perf_counter() - source_started
    hashes = (source_hashes_impl or _default_source_hashes)(Path(e011_run_dir))

    if prepare_e064_execution_impl is None:
        from src.algorithms.shac.g1_gradient_audit_execution import (
            _prepare_e064_execution,
        )

        prepare_e064_execution_impl = _prepare_e064_execution
    preparation_started = time.perf_counter()
    prepared = prepare_e064_execution_impl(contract)
    preparation_duration = time.perf_counter() - preparation_started
    memory_snapshots.append({"boundary": "after-e064-preparation", **memory_snapshot()})

    if make_action_noise_impl is None:
        from src.algorithms.shac.g1_gradient_audit_execution import (
            make_frozen_action_noise,
        )

        make_action_noise_impl = make_frozen_action_noise
    action_noise = make_action_noise_impl(_CANONICAL_SHARD_SEED)
    shard_started = time.perf_counter()
    with prepared.gradient_solver_context():
        evidence = prepared.estimate_shard(_CANONICAL_SHARD_SEED)
    jax.block_until_ready(
        (evidence.result.losses, evidence.result.trajectory.initial_phase)
    )
    shard_duration = time.perf_counter() - shard_started
    memory_snapshots.append({"boundary": "after-shard-zero", **memory_snapshot()})

    estimator_receipts_json_valid = True
    try:
        actual_pathwise = to_finite_json(dict(evidence.pathwise_receipt))
        actual_score = to_finite_json(dict(evidence.score_receipt))
    except (TypeError, ValueError):
        actual_pathwise = None
        actual_score = None
        estimator_receipts_json_valid = False
    expected_estimators = to_finite_json(
        dict(source.estimator_receipts["per_shard"]["0"]["pathwise"])
    )
    estimator_receipts_exact = bool(
        estimator_receipts_json_valid
        and actual_pathwise == actual_score == expected_estimators
    )

    source_losses = np.asarray(source.losses_by_shard[0], dtype=np.float64)
    source_phases = np.asarray(source.initial_phases_by_shard[0])
    actual_losses = np.asarray(jax.device_get(evidence.result.losses))
    actual_phases = np.asarray(jax.device_get(evidence.result.trajectory.initial_phase))
    losses_exact = bool(np.array_equal(actual_losses, source_losses))
    phases_exact = bool(np.array_equal(actual_phases, source_phases))
    source_case = _select_bin_zero(source_losses, source_phases)
    try:
        actual_case = _select_bin_zero(actual_losses, actual_phases)
    except (TypeError, ValueError):
        actual_case = None
    selection_exact = bool(actual_case is not None and source_case == actual_case)
    authoritative_binding = {
        "estimator_receipts_finite_json": estimator_receipts_json_valid,
        "estimator_receipts_exact": estimator_receipts_exact,
        "losses_exact": losses_exact,
        "initial_phases_exact": phases_exact,
        "selected_case_exact": selection_exact,
        "all_exact": bool(
            estimator_receipts_exact
            and losses_exact
            and phases_exact
            and selection_exact
        ),
    }
    if not authoritative_binding["all_exact"]:
        replay_mismatches = [
            name for name, exact in authoritative_binding.items() if exact is not True
        ]
        memory_valid, minimum_headroom = _memory_gate(memory_snapshots)
        elapsed_before_publication = time.perf_counter() - total_started
        elapsed_upper_bound = elapsed_before_publication + _PUBLICATION_BUDGET_SECONDS
        decision_gates = {
            "authoritative_binding": False,
            "forward_valid": False,
            "probes_preserve_done_and_support": False,
            "execution_valid": False,
            "projection_within_one_hour": False,
            "memory_headroom_valid": memory_valid,
        }
        receipt = to_finite_json(
            {
                "schema": "g1-tail-contact-one-case-smoke-v1",
                "decision": "abandon-forward-shac-mechanism",
                "classification_reason": "authoritative-replay-mismatch",
                "replay_mismatches": replay_mismatches,
                "decision_gates": decision_gates,
                "authoritative_binding": authoritative_binding,
                "selected_case": actual_case,
                "source_selected_case": source_case,
                "source": {
                    "e011_run_dir": str(Path(e011_run_dir).resolve()),
                    "e011_evidence_dir": str(source.evidence_dir),
                    "e011_verdict": source.outcome["verdict"],
                    "file_sha256": dict(hashes),
                    "checkpoint": str(getattr(contract, "checkpoint", "")),
                    "checkpoint_sha256": getattr(
                        contract, "checkpoint_sha256", ""
                    ),
                    "reference": str(getattr(contract, "reference", "")),
                    "reference_sha256": getattr(
                        contract, "reference_sha256", ""
                    ),
                },
                "runtime_provenance": dict(
                    getattr(prepared, "runtime_provenance", {})
                ),
                "external_inputs": dict(getattr(prepared, "external_inputs", {})),
                "numerical": None,
                "numerical_nonfinite_counts": {},
                "timings_seconds": {
                    "source_receipt_load": source_load_duration,
                    "e064_preparation": preparation_duration,
                    "shard_estimator": shard_duration,
                    "first_action_objective_preparation": None,
                    "reverse_compile": None,
                    "forward_compile": None,
                    "reverse_cached": None,
                    "forward_cached_sweeps": None,
                    "probes": None,
                    "cached_case": None,
                    "total_smoke_elapsed_before_publication": elapsed_before_publication,
                    "publication_budget": _PUBLICATION_BUDGET_SECONDS,
                    "total_smoke_elapsed_upper_bound": elapsed_upper_bound,
                    "projected_twenty_case": None,
                    "projection_formula": None,
                },
                "resources": {
                    "host_ru_maxrss": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    ),
                    "host_ru_maxrss_units": "KiB-on-Linux",
                    "device_memory_snapshots": memory_snapshots,
                    "minimum_headroom_bytes": minimum_headroom,
                    "headroom_rule_bytes": _MINIMUM_DEVICE_HEADROOM_BYTES,
                    "unavailable_allocator_rule": "valid only when the live process completes without allocator/OOM failure",
                },
                "fixed_thresholds": {
                    "finite_difference_epsilon": _FINITE_DIFFERENCE_EPSILON,
                    "forward_repeat_maximum_absolute_error": 1e-6,
                    "projected_twenty_case_maximum_seconds": _MAXIMUM_PROJECTED_TWENTY_CASE_SECONDS,
                    "minimum_device_headroom_bytes": _MINIMUM_DEVICE_HEADROOM_BYTES,
                },
                "reverse_mode_role": "telemetry-only",
                "derivatives_executed": False,
                "no_training_update": True,
                "e012_artifacts_written": False,
            }
        )
        write_json_atomically(receipt_path, receipt)
        return receipt

    compile_impl = compile_case_kernels_impl or compile_case_kernels
    run_impl = run_compiled_case_smoke_impl or run_compiled_case_smoke
    with prepared.gradient_solver_context():
        objective_started = time.perf_counter()
        diagnostic = prepared.prepare_first_action_objective(
            action_noise,
            actual_case["environment_index"],
            expected_shared_trajectory=evidence.result.trajectory,
        )
        jax.block_until_ready(
            (diagnostic.nominal_first_action, diagnostic.nominal_objective)
        )
        objective_duration = time.perf_counter() - objective_started
        compiled = compile_impl(diagnostic)
        numerical = run_impl(compiled)
    memory_snapshots.append({"boundary": "after-derivatives", **memory_snapshot()})

    cached_case_duration = float(
        shard_duration
        + objective_duration
        + numerical.reverse_cached_duration_seconds
        + np.sum(numerical.forward_cached_sweep_durations_seconds)
        + np.sum(numerical.probe_durations_seconds)
    )
    numerical_json, numerical_nonfinite_counts = _telemetry_json(numerical)
    elapsed_before_publication = time.perf_counter() - total_started
    elapsed_upper_bound = elapsed_before_publication + _PUBLICATION_BUDGET_SECONDS
    projected_twenty_case_seconds = float(
        elapsed_upper_bound + 19 * cached_case_duration
    )
    memory_valid, minimum_headroom = _memory_gate(memory_snapshots)
    projection_valid = (
        projected_twenty_case_seconds <= _MAXIMUM_PROJECTED_TWENTY_CASE_SECONDS
    )
    decision_gates = {
        "authoritative_binding": authoritative_binding["all_exact"],
        "forward_valid": bool(numerical.comparison.forward_valid),
        "probes_preserve_done_and_support": bool(
            numerical.probes_preserve_done_and_support
        ),
        "execution_valid": bool(numerical.execution_valid),
        "projection_within_one_hour": projection_valid,
        "memory_headroom_valid": memory_valid,
    }
    decision = (
        "authorize-forward-shac-method"
        if all(decision_gates.values())
        else "abandon-forward-shac-mechanism"
    )
    receipt = to_finite_json(
        {
            "schema": "g1-tail-contact-one-case-smoke-v1",
            "decision": decision,
            "decision_gates": decision_gates,
            "authoritative_binding": authoritative_binding,
            "selected_case": actual_case,
            "source_selected_case": source_case,
            "source": {
                "e011_run_dir": str(Path(e011_run_dir).resolve()),
                "e011_evidence_dir": str(source.evidence_dir),
                "e011_verdict": source.outcome["verdict"],
                "file_sha256": dict(hashes),
                "checkpoint": str(getattr(contract, "checkpoint", "")),
                "checkpoint_sha256": getattr(contract, "checkpoint_sha256", ""),
                "reference": str(getattr(contract, "reference", "")),
                "reference_sha256": getattr(contract, "reference_sha256", ""),
            },
            "runtime_provenance": dict(prepared.runtime_provenance),
            "external_inputs": dict(prepared.external_inputs),
            "numerical": numerical_json,
            "numerical_nonfinite_counts": numerical_nonfinite_counts,
            "timings_seconds": {
                "source_receipt_load": source_load_duration,
                "e064_preparation": preparation_duration,
                "shard_estimator": shard_duration,
                "first_action_objective_preparation": objective_duration,
                "reverse_compile": numerical.reverse_compile_duration_seconds,
                "forward_compile": numerical.forward_compile_duration_seconds,
                "reverse_cached": numerical.reverse_cached_duration_seconds,
                "forward_cached_sweeps": numerical.forward_cached_sweep_durations_seconds,
                "probes": numerical.probe_durations_seconds,
                "cached_case": cached_case_duration,
                "total_smoke_elapsed_before_publication": elapsed_before_publication,
                "publication_budget": _PUBLICATION_BUDGET_SECONDS,
                "total_smoke_elapsed_upper_bound": elapsed_upper_bound,
                "projected_twenty_case": projected_twenty_case_seconds,
                "projection_formula": "total_smoke_elapsed_upper_bound + 19 * cached_case",
            },
            "resources": {
                "host_ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "host_ru_maxrss_units": "KiB-on-Linux",
                "device_memory_snapshots": memory_snapshots,
                "minimum_headroom_bytes": minimum_headroom,
                "headroom_rule_bytes": _MINIMUM_DEVICE_HEADROOM_BYTES,
                "unavailable_allocator_rule": "valid only when the live process completes without allocator/OOM failure",
            },
            "fixed_thresholds": {
                "finite_difference_epsilon": _FINITE_DIFFERENCE_EPSILON,
                "forward_repeat_maximum_absolute_error": 1e-6,
                "projected_twenty_case_maximum_seconds": _MAXIMUM_PROJECTED_TWENTY_CASE_SECONDS,
                "minimum_device_headroom_bytes": _MINIMUM_DEVICE_HEADROOM_BYTES,
            },
            "reverse_mode_role": "telemetry-only",
            "derivatives_executed": True,
            "no_training_update": True,
            "e012_artifacts_written": False,
        }
    )
    write_json_atomically(receipt_path, receipt)
    return receipt
