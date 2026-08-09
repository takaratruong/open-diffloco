"""Execution and publication seam for the frozen failure-weighted G1 audit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.g1_failure_weighted_audit import (
    aggregate_failure_weighted_shards,
    build_failure_weighted_candidates,
    classify_failure_weighted_outcome,
    validate_failure_weighted_aggregation,
)
from src.algorithms.shac.g1_gradient_audit_execution import (
    _ACTION_DIMENSION,
    _SOLVER_TRACE_CONTEXT,
    _array_sha256,
    _baseline_competence_receipt,
    _candidate_delta_is_finite_nonzero,
    _prepare_e064_execution,
    _tree_is_finite_nonzero,
    _validate_execution_contract,
    _validate_ordinary_row,
    make_frozen_action_noise,
    replace_actor_parameters,
    summarize_stochastic_rollout,
)
from src.algorithms.shac.g1_gradient_audit_runner import to_finite_json
from src.algorithms.shac.gradient_audit import apply_functional_actor_step

_CANDIDATE_LABELS = ("baseline", "uniform", "tail")
_REUSED_RENDER_LABELS = {
    "baseline": "baseline",
    "uniform": "pathwise",
    "tail": "score",
}
_ALGORITHMIC_VALIDITY_KEYS = {
    "analytic_gaussian_sign",
    "detachment",
    "done_boundary_return",
    "ppo_ratio_one",
    "pytree_order",
    "smooth_toy_convergence",
}
_RECOMPUTATION_BOOLEAN_KEYS = {
    "weight_receipts_exact",
    "uniform_reproduction_exact",
    "tail_reproduction_exact",
    "stability_evidence_exact",
}
_HOST_RECOMPUTATION_KEYS = {
    "tolerance_units",
    "tolerance_model",
    "uniform_maximum_absolute_error",
    "tail_maximum_absolute_error",
}
_HOST_RECOMPUTATION_TOLERANCE_UNITS = 256
_HOST_RECOMPUTATION_TOLERANCE_MODEL = (
    "absolute_error <= tolerance_units * dtype_epsilon * max(1, abs(expected))"
)


def _tree_signature(tree: Any) -> list[dict[str, Any]]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(np.shape(leaf)),
            "dtype": str(np.asarray(jax.device_get(leaf)).dtype),
        }
        for path, leaf in leaves
    ]


def _assert_exact_tree(actual: Any, expected: Any, *, label: str) -> None:
    actual_leaves, actual_definition = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_definition = jax.tree_util.tree_flatten(expected)
    if actual_definition != expected_definition:
        raise ValueError(f"{label} PyTree structure differs")
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array = np.asarray(jax.device_get(actual_leaf))
        expected_array = np.asarray(jax.device_get(expected_leaf))
        if (
            actual_array.shape != expected_array.shape
            or actual_array.dtype != expected_array.dtype
            or not np.array_equal(actual_array, expected_array)
        ):
            raise ValueError(f"{label} does not reconstruct exactly")


def _gradient_hash_receipt(
    results: tuple[Any, ...], aggregation: Any
) -> dict[str, Any]:
    from src.algorithms.shac.g1_gradient_audit import stable_pytree_sha256

    return {
        "per_shard": [
            {
                "seed": seed,
                "per_environment_clipped": stable_pytree_sha256(
                    result.pathwise_effective_gradients
                ),
                "uniform": stable_pytree_sha256(aggregation.uniform_shard_means[seed]),
                "tail": stable_pytree_sha256(aggregation.tail_shard_means[seed]),
            }
            for seed, result in enumerate(results)
        ],
        "aggregate": {
            "uniform": stable_pytree_sha256(aggregation.uniform_mean),
            "tail": stable_pytree_sha256(aggregation.tail_mean),
        },
    }


def _validate_aggregation(results: tuple[Any, ...], aggregation: Any) -> dict[str, Any]:
    receipt = to_finite_json(
        validate_failure_weighted_aggregation(results, aggregation)
    )
    if not isinstance(receipt, dict):
        raise TypeError("authoritative recomputation receipt must be a mapping")
    required = _RECOMPUTATION_BOOLEAN_KEYS | {"independent_host_recomputation"}
    if missing := sorted(required - set(receipt)):
        raise ValueError(
            f"authoritative recomputation receipt is missing keys: {missing}"
        )
    if any(receipt[key] is not True for key in _RECOMPUTATION_BOOLEAN_KEYS):
        raise ValueError("authoritative recomputation checks did not all pass")
    host_receipt = receipt["independent_host_recomputation"]
    if not isinstance(host_receipt, dict):
        raise TypeError("authoritative host recomputation receipt must be a mapping")
    if missing := sorted(_HOST_RECOMPUTATION_KEYS - set(host_receipt)):
        raise ValueError(
            f"authoritative host recomputation receipt is missing keys: {missing}"
        )
    if (
        isinstance(host_receipt["tolerance_units"], bool)
        or host_receipt["tolerance_units"] != _HOST_RECOMPUTATION_TOLERANCE_UNITS
        or host_receipt["tolerance_model"] != _HOST_RECOMPUTATION_TOLERANCE_MODEL
    ):
        raise ValueError("authoritative host recomputation used a non-frozen tolerance")
    for key in (
        "uniform_maximum_absolute_error",
        "tail_maximum_absolute_error",
    ):
        value = host_receipt[key]
        if isinstance(value, bool) or float(value) < 0.0:
            raise ValueError(f"authoritative recomputation metric {key} is invalid")
    return receipt


def _candidate_reconstruction(
    *,
    actor_apply: Any,
    actor_params: Any,
    aggregation: Any,
    candidates: Any,
    target_rms: float,
) -> dict[str, Any]:
    from src.algorithms.shac.g1_gradient_audit import stable_pytree_sha256

    baseline_signature = _tree_signature(actor_params)
    hashes = {"baseline": stable_pytree_sha256(actor_params)}
    for label, gradient in (
        ("uniform", aggregation.uniform_mean),
        ("tail", aggregation.tail_mean),
    ):
        if _tree_signature(gradient) != baseline_signature:
            raise ValueError(
                f"{label} aggregate dtype/tree differs from actor parameters"
            )
        direction = jax.tree_util.tree_map(lambda leaf: -leaf, gradient)
        reconstructed, summary = apply_functional_actor_step(
            actor_apply,
            actor_params,
            direction,
            aggregation.normalized_observations,
            target_rms=target_rms,
        )
        candidate = getattr(candidates, label)
        _assert_exact_tree(candidate, reconstructed, label=f"{label} candidate")
        _assert_exact_tree(
            to_finite_json(summary),
            candidates.functional_steps[label],
            label=f"{label} functional step summary",
        )
        if _tree_signature(candidate) != baseline_signature:
            raise ValueError(f"{label} candidate dtype/tree differs from baseline")
        hashes[label] = stable_pytree_sha256(candidate)
    return {
        "exact": True,
        "dtype_tree_exact": True,
        "actor_parameter_signature": baseline_signature,
        "actor_parameter_hashes": hashes,
    }


def _stochastic_evaluation(
    *, contract: Any, prepared: Any, candidate_parameters: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    per_seed = []
    for seed in contract.held_out_seeds:
        action_noise = make_frozen_action_noise(seed)
        summaries = {}
        for label in _CANDIDATE_LABELS:
            with prepared.gradient_solver_context():
                trajectory = prepared.stochastic_rollout(
                    candidate_parameters[label], action_noise
                )
            if hasattr(trajectory, "trajectory"):
                trajectory = trajectory.trajectory
            summaries[label] = summarize_stochastic_rollout(
                trajectory,
                gamma=contract.gamma,
                expected_population=contract.population,
                expected_horizon=contract.horizon,
                expected_action_dimension=_ACTION_DIMENSION,
                expected_observation_dimension=154,
            )
        per_seed.append(
            {
                "seed": int(seed),
                "action_noise_sha256": _array_sha256(action_noise),
                "candidates": summaries,
            }
        )
    complete = bool(
        len(per_seed) == 4
        and all(
            summary["trajectory_complete"] is True
            for seed_row in per_seed
            for summary in seed_row["candidates"].values()
        )
    )
    return (
        to_finite_json(
            {
                "held_out_seeds": contract.held_out_seeds,
                "common_random_numbers": True,
                "finite_complete": complete,
                "per_seed": per_seed,
            }
        ),
        complete,
    )


def _ordinary_evaluation(
    *, contract: Any, prepared: Any, candidate_parameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[Any, Any], dict[str, Any]]:
    ordinary_seed = 0
    ordinary_rows = {}
    ordinary_arrays = {}
    candidates = {}
    for label in _CANDIDATE_LABELS:
        summaries = []
        for phase in contract.phases:
            raw_row = prepared.phase_rollout(
                candidate_parameters[label],
                prepared.normalizer_state,
                seed=ordinary_seed,
                phase=phase,
            )
            summary, arrays = _validate_ordinary_row(raw_row)
            ordinary_rows[(ordinary_seed, label, phase)] = dict(raw_row)
            for field, array in arrays.items():
                ordinary_arrays[
                    f"seed_{ordinary_seed}/{label}/phase_{phase}/{field}"
                ] = array
            summaries.append(summary)
        candidates[label] = summaries
    evaluation = to_finite_json(
        {
            "mode": "single-deterministic-five-phase-grid",
            "seed": ordinary_seed,
            "phases": contract.phases,
            "per_seed": [{"seed": ordinary_seed, "candidates": candidates}],
        }
    )
    baseline = {
        **_baseline_competence_receipt(
            phases=contract.phases,
            actual_survival=[int(row["survival"]) for row in candidates["baseline"]],
        ),
        "seed": ordinary_seed,
    }
    return evaluation, ordinary_arrays, ordinary_rows, baseline


def _render_candidate_videos(
    *, prepared: Any, rows: dict[Any, Any], output_dir: Path
) -> dict[str, Path]:
    if prepared.render_phase_zero is None:
        raise ValueError("required phase-zero renderer is not configured")
    aliased_rows = {
        (seed, _REUSED_RENDER_LABELS[label], phase): row
        for (seed, label, phase), row in rows.items()
    }
    rendered = {
        str(label): Path(path)
        for label, path in prepared.render_phase_zero(
            rows=aliased_rows,
            output_dir=output_dir,
        ).items()
    }
    if set(rendered) != {"baseline", "pathwise", "score"}:
        raise ValueError("phase-zero renderer must return all three reused labels")
    published = {}
    reverse = {value: key for key, value in _REUSED_RENDER_LABELS.items()}
    for old_label, old_path in rendered.items():
        if not old_path.is_file():
            raise ValueError("phase-zero renderer returned a missing video")
        label = reverse[old_label]
        new_path = output_dir / "videos" / f"{label}_phase0.mp4"
        if old_path != new_path:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, new_path)
        published[label] = new_path
    return published


def run_audit(contract: Any) -> dict[str, Any]:
    """Execute one frozen failure-weighted audit and publish its manifest last."""
    from tools.audit_g1_shac_gradient_quality import (
        sha256_file,
        write_json_atomically,
        write_npz_atomically,
        write_pickle_atomically,
    )

    jax.config.update("jax_enable_x64", True)
    _validate_execution_contract(contract)
    output_dir = Path(contract.output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("audit manifest already exists; refusing to overwrite it")
    prepared = _prepare_e064_execution(contract)
    if set(prepared.algorithmic_validity) != _ALGORITHMIC_VALIDITY_KEYS or not all(
        value is True for value in prepared.algorithmic_validity.values()
    ):
        raise ValueError("preserved E008 algorithmic validity checks did not all pass")

    estimator_receipts = {}
    shard_results = []
    for seed in contract.shard_seeds:
        with prepared.gradient_solver_context():
            evidence = prepared.estimate_shard(seed)
        pathwise = to_finite_json(dict(evidence.pathwise_receipt))
        score = to_finite_json(dict(evidence.score_receipt))
        if pathwise != score:
            raise ValueError(f"shard {seed} estimator identity receipts differ")
        estimator_receipts[str(seed)] = {"pathwise": pathwise, "score": score}
        shard_results.append(evidence.result)
    shard_results = tuple(shard_results)

    aggregation = aggregate_failure_weighted_shards(shard_results)
    recomputation = _validate_aggregation(shard_results, aggregation)
    gradient_hashes = _gradient_hash_receipt(shard_results, aggregation)
    candidates = build_failure_weighted_candidates(
        actor_apply=prepared.actor_apply,
        actor_params=prepared.checkpoint_state.actor_params,
        uniform_gradient=aggregation.uniform_mean,
        tail_gradient=aggregation.tail_mean,
        normalized_observations=aggregation.normalized_observations,
        target_rms=contract.functional_rms,
    )
    candidate_reconstruction = _candidate_reconstruction(
        actor_apply=prepared.actor_apply,
        actor_params=prepared.checkpoint_state.actor_params,
        aggregation=aggregation,
        candidates=candidates,
        target_rms=contract.functional_rms,
    )
    candidate_parameters = {
        label: getattr(candidates, label) for label in _CANDIDATE_LABELS
    }

    heldout, heldout_complete = _stochastic_evaluation(
        contract=contract,
        prepared=prepared,
        candidate_parameters=candidate_parameters,
    )
    ordinary, ordinary_arrays, ordinary_rows, baseline = _ordinary_evaluation(
        contract=contract,
        prepared=prepared,
        candidate_parameters=candidate_parameters,
    )
    functional_steps = {
        **dict(candidates.functional_steps),
        "candidate_reconstruction": candidate_reconstruction,
    }
    validity = {
        "frozen_hashes": True,
        "weight_receipts_exact": recomputation["weight_receipts_exact"] is True,
        "uniform_reproduction_exact": (
            recomputation["uniform_reproduction_exact"] is True
        ),
        "tail_reproduction_exact": recomputation["tail_reproduction_exact"] is True,
        "stability_evidence_exact": (recomputation["stability_evidence_exact"] is True),
        "aggregate_gradients_finite_nonzero": (
            _tree_is_finite_nonzero(aggregation.uniform_mean)
            and _tree_is_finite_nonzero(aggregation.tail_mean)
        ),
        "candidate_trees_finite_nonzero": all(
            _candidate_delta_is_finite_nonzero(
                candidates.baseline, candidate_parameters[label]
            )
            for label in ("uniform", "tail")
        ),
        "functional_steps_valid": bool(
            candidate_reconstruction["exact"]
            and candidate_reconstruction["dtype_tree_exact"]
        ),
        "rollouts_fresh_replay_free_complete_finite": bool(
            heldout_complete
            and baseline["competence_floor_pass"]
            and all(
                row["complete"] is True and row["replay_free"] is True
                for candidate_rows in ordinary["per_seed"][0]["candidates"].values()
                for row in candidate_rows
            )
        ),
    }
    outcome = classify_failure_weighted_outcome(
        geometry=aggregation.geometry,
        evaluation=ordinary,
        functional_steps=functional_steps,
        validity=validity,
    )

    documents = {
        "failure_weight_receipts.json": {
            "weighting": aggregation.weighting_receipts,
            "independent_recomputation": recomputation,
            "gradient_hashes": gradient_hashes,
        },
        "gradient_geometry.json": aggregation.geometry,
        "estimator_receipts.json": {
            "shared_rollout_identity": True,
            "algorithmic_validity": prepared.algorithmic_validity,
            "per_shard": estimator_receipts,
        },
        "functional_step_receipt.json": functional_steps,
        "heldout_stochastic.json": heldout,
        "ordinary_phase_grid.json": ordinary,
        "ordinary_baseline_reproduction.json": baseline,
        "solver_trace_context.json": _SOLVER_TRACE_CONTEXT,
        "outcome.json": outcome,
        "validity.json": validity,
    }
    artifact_paths = []
    for label in _CANDIDATE_LABELS:
        path = output_dir / f"{label}_candidate.pkl"
        write_pickle_atomically(
            path,
            replace_actor_parameters(
                prepared.checkpoint_state, candidate_parameters[label]
            ),
        )
        artifact_paths.append(path)
    for relative, document in documents.items():
        path = output_dir / relative
        write_json_atomically(path, to_finite_json(document))
        artifact_paths.append(path)
    arrays_path = output_dir / "ordinary_phase_grid_arrays.npz"
    write_npz_atomically(arrays_path, ordinary_arrays)
    artifact_paths.append(arrays_path)

    videos = _render_candidate_videos(
        prepared=prepared,
        rows=ordinary_rows,
        output_dir=output_dir,
    )
    artifact_paths.extend(videos.values())
    video_receipt = {
        "status": "produced",
        "seed": 0,
        "phase": 0,
        "paths": {
            label: str(path.resolve().relative_to(output_dir.resolve()))
            for label, path in videos.items()
        },
    }
    artifacts = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(artifact_paths)
    }
    manifest = to_finite_json(
        {
            "schema_version": "g1-shac-failure-weighted-audit/v1",
            "status": "complete",
            "contract": {
                "checkpoint": str(contract.checkpoint),
                "checkpoint_sha256": contract.checkpoint_sha256,
                "reference": str(contract.reference),
                "reference_sha256": contract.reference_sha256,
                "hparams_path": str(contract.hparams_path),
                "hparams_file_sha256": sha256_file(contract.hparams_path),
                "shard_seeds": contract.shard_seeds,
                "held_out_seeds": contract.held_out_seeds,
                "phases": contract.phases,
                "population": contract.population,
                "horizon": contract.horizon,
                "sigma": contract.sigma,
                "gamma": contract.gamma,
                "per_env_clip": contract.per_env_clip,
                "functional_rms": contract.functional_rms,
                "solver_iterations": contract.solver_iterations,
                "solver_ls_iterations": contract.solver_ls_iterations,
            },
            "validated_e064_contract": prepared.validated_contract,
            "external_inputs": prepared.external_inputs,
            "runtime_provenance": prepared.runtime_provenance,
            "solver_trace_context": _SOLVER_TRACE_CONTEXT,
            "thresholds": {
                "uniform_tail_mixture": 0.5,
                "worst_quartile_divisor": 4,
                "stability_noninferiority_tolerance": 0.05,
                "minimum_return_improvement": 0.001,
                "minimum_survival_improvement": 5.0,
                "maximum_phase_survival_loss_fraction": 0.10,
                "minimum_material_phase_wins": 2,
                "functional_rms_relative_tolerance": 5e-5,
            },
            "decision_rules": {
                "ordinary_transfer_evidence": (
                    "one deterministic seed-0 grid; held-out stochastic fragments "
                    "are descriptive"
                ),
                "stability_precedence": (
                    "pairwise mean/minimum and leave-one-out minimum before transfer"
                ),
                "candidate_output_separation": "descriptive only",
            },
            "canonical_metrics": {
                "return": "mean ordinary reward over active steps",
                "survival": "active steps",
            },
            "ordinary_baseline_reproduction": baseline,
            "heldout_stochastic_finite_complete": heldout_complete,
            "validity": validity,
            "outcome": outcome,
            "videos": video_receipt,
            "artifacts": artifacts,
        }
    )
    write_json_atomically(output_dir / "manifest.json", manifest)
    return manifest
