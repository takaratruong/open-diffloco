"""Compare E002 and its rejected residual on one identical saved rollout batch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.algorithm import train
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualOptState,
    FrozenControllerResidualParams,
    frozen_controller_residual_depth,
)
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.run_g1_dual_scale_root_position import (
    REFERENCE_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_support_aware_impulse_continuation import (
    SUPPORT_TARGET_SHA256,
    build_support_aware_kwargs,
    validate_target_artifact,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_966_080
EFFECTIVE_NUM_ENVS = 512
UNROLL_LENGTH = 24
END_STEP = START_STEP + EFFECTIVE_NUM_ENVS * UNROLL_LENGTH
FIXED_BATCH_IMPROVEMENT_FLOOR = 0.01
CANDIDATE_CHECKPOINT_SHA256 = (
    "6a00c82ffedda63c6206a045996f444211d9176376972e408660bf5cb2d8c71e"
)
CANDIDATE_HPARAMS_SHA256 = (
    "cd3ef866014fe12732c156942626da9b699e5bb9a3d05afe480353e305e009d8"
)
DETERMINISTIC_SELECTION_SHA256 = (
    "e2228422ac2d9b60b5d5516b283d492467d5971d6139a4e0240dc10abd15ce4c"
)
GROUP_SIZES = {"phase": 5, "support": 4, "terminal": 4}
GROUP_LABELS = {
    "phase": ["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"],
    "support": ["none", "left", "right", "double"],
    "terminal": ["survived-H24", "early", "middle", "late"],
}
GLOBAL_GRADIENT_KEYS = (
    "actor_grad_population_mean_norm",
    "actor_grad_population_rms_norm",
    "actor_grad_population_variance_trace",
    "actor_grad_population_cancellation_ratio",
    "actor_grad_population_noise_scale",
    "actor_grad_population_esnr",
)


def build_fixed_batch_audit_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    target_path: str | Path,
) -> dict[str, Any]:
    """Resume the depth-two support objective for one rejected diagnostic update."""
    kwargs = build_support_aware_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        target_path,
    )
    kwargs.update(
        total_steps=END_STEP,
        checkpoint_steps=(END_STEP,),
        diagnose=True,
    )
    return kwargs


def zero_newest_residual_output_head(
    params: FrozenControllerResidualParams,
) -> FrozenControllerResidualParams:
    """Restore exact parent behavior while preserving the newest hidden layer."""
    if (
        not isinstance(params, FrozenControllerResidualParams)
        or frozen_controller_residual_depth(params) != 2
        or not isinstance(params.adapter, dict)
    ):
        raise ValueError("fixed-batch control requires depth-two residual params")
    modules = params.adapter.get("params")
    if not isinstance(modules, dict) or set(modules) != {"Dense_0", "Dense_1"}:
        raise ValueError("newest residual adapter layout is invalid")
    output = modules["Dense_1"]
    if not isinstance(output, dict) or set(output) != {"kernel", "bias"}:
        raise ValueError("newest residual output head layout is invalid")
    zero_output = {
        "kernel": jnp.zeros_like(output["kernel"]),
        "bias": jnp.zeros_like(output["bias"]),
    }
    adapter = {
        **params.adapter,
        "params": {**modules, "Dense_1": zero_output},
    }
    return params._replace(adapter=adapter)


def _finite_loss(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite nonnegative loss")
    return float(value)


def classify_fixed_batch_distribution(
    *,
    source_primary_loss: float,
    source_heldout_loss: float,
    candidate_primary_loss: float,
    candidate_heldout_loss: float,
    deterministic: Mapping[str, object],
) -> dict[str, object]:
    """Distinguish objective-distribution reversal from local update failure."""
    fixed_values = {
        "primary": (
            _finite_loss(source_primary_loss, "source primary"),
            _finite_loss(candidate_primary_loss, "candidate primary"),
        ),
        "heldout": (
            _finite_loss(source_heldout_loss, "source heldout"),
            _finite_loss(candidate_heldout_loss, "candidate heldout"),
        ),
    }
    deterministic_values = {
        "primary": (
            _finite_loss(
                deterministic.get("source_primary_loss"),
                "deterministic source primary",
            ),
            _finite_loss(
                deterministic.get("candidate_primary_loss"),
                "deterministic candidate primary",
            ),
        ),
        "heldout": (
            _finite_loss(
                deterministic.get("source_heldout_loss"),
                "deterministic source heldout",
            ),
            _finite_loss(
                deterministic.get("candidate_heldout_loss"),
                "deterministic candidate heldout",
            ),
        ),
    }
    improvements = {
        name: (source - candidate) / max(source, 1e-12)
        for name, (source, candidate) in fixed_values.items()
    }
    fixed_improves = {
        name: value >= FIXED_BATCH_IMPROVEMENT_FLOOR
        for name, value in improvements.items()
    }
    deterministic_reverses = {
        name: candidate > source
        for name, (source, candidate) in deterministic_values.items()
    }
    both_improve = all(fixed_improves.values())
    both_reverse = all(deterministic_reverses.values())
    if both_improve and both_reverse:
        classification = (
            "training-distribution-improves-deterministic-reverses"
        )
    elif both_improve:
        classification = (
            "training-distribution-improves-without-deterministic-reversal"
        )
    elif any(fixed_improves.values()):
        classification = "replica-specific-objective-shift"
    else:
        classification = "fixed-batch-objective-not-improved"
    return {
        "protocol": "g1-fixed-batch-distribution-classification-v1",
        "classification": classification,
        "relative_improvement_floor": FIXED_BATCH_IMPROVEMENT_FLOOR,
        "fixed_batch_relative_improvements": improvements,
        "fixed_batch_replica_improves": fixed_improves,
        "fixed_batch_both_improve": both_improve,
        "deterministic_replica_reverses": deterministic_reverses,
        "deterministic_both_reverse": both_reverse,
    }


def _finite_tree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(tree)
    )


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=3e-4, abs_tol=3e-7)


def _single_row(path: Path) -> dict[str, object]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{path.name} must contain exactly one row")
    return rows[0]


def _gradient_group_summary(
    row: Mapping[str, object], *, group: str
) -> dict[str, object]:
    size = GROUP_SIZES[group]
    prefix = f"actor_grad_{group}"
    arrays: dict[str, list[object]] = {}
    for suffix in (
        "bin_counts",
        "bin_mean_norms",
        "bin_rms_norms",
        "bin_variance_traces",
        "bin_cancellation_ratios",
        "bin_noise_scales",
        "bin_esnr",
    ):
        array = np.asarray(row.get(f"{prefix}_{suffix}"), dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{group} gradient {suffix} is invalid")
        arrays[suffix] = array.tolist()
    counts = np.asarray(arrays["bin_counts"])
    if (
        not np.array_equal(counts, np.rint(counts))
        or np.any(counts < 0)
        or int(np.sum(counts)) != EFFECTIVE_NUM_ENVS
    ):
        raise ValueError(f"{group} gradient counts do not cover the population")
    for suffix in (
        "bin_mean_norms",
        "bin_rms_norms",
        "bin_variance_traces",
        "bin_cancellation_ratios",
        "bin_noise_scales",
        "bin_esnr",
    ):
        if np.any(np.asarray(arrays[suffix]) < 0.0):
            raise ValueError(f"{group} gradient {suffix} is negative")
    cosine = np.asarray(
        row.get(f"{prefix}_bin_cosine_matrix"), dtype=np.float64
    )
    if cosine.shape != (size, size) or not np.isfinite(cosine).all():
        raise ValueError(f"{group} gradient cosine matrix is invalid")
    scalars = {
        suffix: _finite_loss(row.get(f"{prefix}_{suffix}"), f"{group} {suffix}")
        for suffix in (
            "within_variance_trace",
            "between_variance_trace",
            "total_variance_trace",
            "within_variance_fraction",
            "between_variance_fraction",
        )
    }
    global_variance = _finite_loss(
        row.get("actor_grad_population_variance_trace"),
        "population variance",
    )
    if not (
        _close(
            scalars["within_variance_trace"]
            + scalars["between_variance_trace"],
            scalars["total_variance_trace"],
        )
        and _close(scalars["total_variance_trace"], global_variance)
        and _close(
            scalars["within_variance_fraction"]
            + scalars["between_variance_fraction"],
            1.0,
        )
    ):
        raise ValueError(f"{group} gradient variance decomposition does not close")
    return {
        "labels": GROUP_LABELS[group],
        **arrays,
        "bin_cosine_matrix": cosine.tolist(),
        **scalars,
    }


def _load_arm_row(run_directory: Path) -> dict[str, object]:
    checkpoint = _single_row(run_directory / "checkpoint_phase_metrics.json")
    diagnostic = _single_row(run_directory / "diag_log.json")
    if checkpoint.get("step") != END_STEP or diagnostic.get("step") != END_STEP:
        raise ValueError("fixed-batch telemetry step is invalid")
    duplicated = ["actor_cagrad_valid", *GLOBAL_GRADIENT_KEYS]
    duplicated.extend(
        key
        for key in checkpoint
        if key.startswith("actor_grad_phase_")
        or key.startswith("actor_grad_support_")
        or key.startswith("actor_grad_terminal_")
    )
    for key in duplicated:
        if checkpoint.get(key) != diagnostic.get(key):
            raise ValueError(f"checkpoint and diagnostic {key} do not match")
    row = dict(checkpoint)
    row["actor_grad_finite_fraction"] = diagnostic.get(
        "actor_grad_finite_fraction"
    )
    return row


def stable_path_leaf_tree_sha256(tree: object) -> str:
    """Hash tree leaf identity without unstable custom-node auxiliary reprs."""
    digest = hashlib.sha256()
    digest.update(
        f"{type(tree).__module__}.{type(tree).__qualname__}".encode("utf-8")
    )
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    digest.update(str(len(paths_and_leaves)).encode("ascii"))
    for path, value in paths_and_leaves:
        digest.update(repr(path).encode("utf-8"))
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_lineage_hashes(state: object) -> dict[str, str]:
    return {
        "key": stable_path_leaf_tree_sha256(state.key),
        "env_state": stable_path_leaf_tree_sha256(state.env_state),
        "critic_params": stable_path_leaf_tree_sha256(state.critic_params),
        "target_critic_params": stable_path_leaf_tree_sha256(
            state.target_critic_params
        ),
        "normalizer": stable_path_leaf_tree_sha256(state.normalizer),
        "actor_optimizer": stable_path_leaf_tree_sha256(state.actor_opt),
        "critic_optimizer": stable_path_leaf_tree_sha256(state.critic_opt),
    }


def _write_pickle_atomically(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        pickle.dump(value, stream)
    os.replace(temporary, path)


def create_common_state_control(
    *, candidate_checkpoint: Path, output_root: Path
) -> tuple[Path, dict[str, object]]:
    """Create a hash-bound candidate-state checkpoint with only its head zeroed."""
    with candidate_checkpoint.open("rb") as stream:
        candidate = pickle.load(stream)
    if (
        int(candidate.step) != START_STEP
        or not isinstance(candidate.actor_params, FrozenControllerResidualParams)
        or not isinstance(candidate.actor_opt, FrozenControllerResidualOptState)
        or frozen_controller_residual_depth(candidate.actor_params) != 2
        or not _finite_tree(candidate)
    ):
        raise ValueError("candidate checkpoint is not a finite depth-two state")
    candidate_actor_hash = parameter_tree_sha256(candidate.actor_params)
    candidate_parent_hash = parameter_tree_sha256(candidate.actor_params.parent)
    common_hashes = _state_lineage_hashes(candidate)
    control_params = zero_newest_residual_output_head(candidate.actor_params)
    control = candidate.replace(actor_params=control_params)
    if _state_lineage_hashes(control) != common_hashes:
        raise ValueError("control construction changed common saved state")
    output = output_root / "common_state_control" / candidate_checkpoint.name
    _write_pickle_atomically(output, control)
    hparams_source = candidate_checkpoint.with_name("hparams.json")
    hparams = json.loads(hparams_source.read_text(encoding="utf-8"))
    hparams_output = output.with_name("hparams.json")
    _write_json_atomically(hparams_output, hparams)
    report = {
        "protocol": "g1-common-state-zero-head-control-v1",
        "valid": True,
        "source_step": START_STEP,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "candidate_hparams_sha256": sha256_file(hparams_source),
        "candidate_actor_tree_sha256": candidate_actor_hash,
        "candidate_parent_tree_sha256": candidate_parent_hash,
        "control_checkpoint": str(output),
        "control_checkpoint_sha256": sha256_file(output),
        "control_hparams_sha256": sha256_file(hparams_output),
        "control_actor_tree_sha256": parameter_tree_sha256(control_params),
        "control_parent_tree_sha256": parameter_tree_sha256(
            control_params.parent
        ),
        "common_state_tree_sha256": common_hashes,
        "newest_output_head_zero": all(
            bool(np.all(np.asarray(value) == 0.0))
            for value in control_params.adapter["params"]["Dense_1"].values()
        ),
        "optimizer_snapshot_shared": True,
        "optimizer_update_retained": False,
    }
    if (
        report["control_hparams_sha256"] != report["candidate_hparams_sha256"]
        or report["control_parent_tree_sha256"] != candidate_parent_hash
        or report["newest_output_head_zero"] is not True
    ):
        raise ValueError("common-state control receipt is invalid")
    del candidate, control
    gc.collect()
    return output, report


def validate_arm_artifacts(
    run_directory: Path,
    *,
    initial_checkpoint: Path,
    candidate_checkpoint: Path,
    target_path: Path,
) -> dict[str, object]:
    """Validate one pre-update fixed-batch capture and reject its update."""
    hparams_path = run_directory / "hparams.json"
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    required = {
        "total_steps": END_STEP,
        "checkpoint_steps": [END_STEP],
        "effective_num_envs": EFFECTIVE_NUM_ENVS,
        "num_envs": 256,
        "gradient_accumulation_steps": 2,
        "unroll_length": UNROLL_LENGTH,
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "actor_per_env_grad_clip": 1.0,
        "actor_bootstrap_scale": 0.0,
        "actor_frozen_controller_residual_depth": 2,
        "actor_support_aware_impulse": True,
        "actor_support_aware_impulse_path": str(target_path.resolve()),
        "actor_support_aware_impulse_sha256": SUPPORT_TARGET_SHA256,
        "ahac": False,
        "torso_wrench_assistance": False,
        "domain_randomization": False,
    }
    if any(hparams.get(key) != value for key, value in required.items()):
        raise ValueError("fixed-batch arm hparams violate the registered contract")
    row = _load_arm_row(run_directory)
    if (
        row.get("actor_cagrad_valid") is not True
        or row.get("actor_preview_valid") is not True
        or row.get("actor_grad_finite_fraction") != 1.0
        or row.get("actor_preview_frozen_parameter_drift_max_abs") != 0.0
        or row.get("actor_preview_frozen_moment_drift_max_abs") != 0.0
        or row.get("actor_preview_normalizer_drift_max_abs") != 0.0
    ):
        raise ValueError("fixed-batch arm gradient or frozen-lineage telemetry failed")
    mean_norm = _finite_loss(
        row.get("actor_grad_population_mean_norm"), "population mean norm"
    )
    rms_norm = _finite_loss(
        row.get("actor_grad_population_rms_norm"), "population RMS norm"
    )
    variance = _finite_loss(
        row.get("actor_grad_population_variance_trace"), "population variance"
    )
    if mean_norm <= 0.0 or rms_norm <= 0.0 or not _close(
        variance, max(rms_norm * rms_norm - mean_norm * mean_norm, 0.0)
    ):
        raise ValueError("fixed-batch arm population moments do not close")
    groups = {
        group: _gradient_group_summary(row, group=group)
        for group in GROUP_SIZES
    }
    if groups["phase"]["bin_counts"] != row.get("actor_cagrad_bin_counts"):
        raise ValueError("phase decomposition and CAGrad counts differ")
    primary_loss = _finite_loss(
        row.get("actor_support_aware_impulse_loss"), "primary support loss"
    )
    heldout_loss = _finite_loss(
        row.get("actor_support_aware_impulse_heldout_loss"),
        "heldout support loss",
    )
    window_count = row.get("actor_support_aware_impulse_valid_window_count")
    if (
        isinstance(window_count, bool)
        or not isinstance(window_count, int)
        or window_count <= 0
    ):
        raise ValueError("fixed-batch support window coverage is invalid")
    checkpoint = run_directory / f"checkpoint_step_{END_STEP}.pkl"
    with candidate_checkpoint.open("rb") as stream:
        candidate_source = pickle.load(stream)
    frozen_hashes = {
        "parent": parameter_tree_sha256(candidate_source.actor_params.parent),
        "parent_optimizer": parameter_tree_sha256(
            candidate_source.actor_opt.parent_optimizer_state
        ),
        "normalizer": parameter_tree_sha256(candidate_source.normalizer),
    }
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != END_STEP
        or not _finite_tree(state)
        or parameter_tree_sha256(state.actor_params.parent)
        != frozen_hashes["parent"]
        or parameter_tree_sha256(state.actor_opt.parent_optimizer_state)
        != frozen_hashes["parent_optimizer"]
        or parameter_tree_sha256(state.normalizer) != frozen_hashes["normalizer"]
    ):
        raise ValueError("fixed-batch diagnostic checkpoint violates lineage")
    return {
        "protocol": "g1-fixed-batch-arm-validation-v1",
        "valid": True,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "captured_update_step": END_STEP,
        "primary_support_loss": primary_loss,
        "heldout_support_loss": heldout_loss,
        "valid_support_window_count": window_count,
        "gradient_population": {
            key: row[key] for key in GLOBAL_GRADIENT_KEYS
        },
        "gradient_groups": groups,
        "diagnostic_checkpoint": str(checkpoint),
        "diagnostic_checkpoint_sha256": sha256_file(checkpoint),
        "hparams_sha256": sha256_file(hparams_path),
        "checkpoint_metrics_sha256": sha256_file(
            run_directory / "checkpoint_phase_metrics.json"
        ),
        "diagnostic_log_sha256": sha256_file(run_directory / "diag_log.json"),
        "optimizer_update_retained": False,
        "retained_policy": None,
    }


def _run_arm(
    *,
    label: str,
    resume_from: Path,
    profile_name: str,
    reference_path: Path,
    target_path: Path,
    seed: int,
    output_root: Path,
) -> Path:
    arm_root = output_root / label
    arm_root.mkdir(parents=True, exist_ok=False)
    kwargs = build_fixed_batch_audit_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        target_path,
    )
    previous = Path.cwd()
    try:
        os.chdir(arm_root)
        with solver_context(get_solver_profile(profile_name)):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous)
    return (arm_root / relative_save_dir).resolve()


def _deterministic_metrics(selection: Mapping[str, object]) -> dict[str, object]:
    source = selection.get("source_target_metrics")
    candidate = selection.get("candidate_target_metrics")
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        raise ValueError("deterministic selection target metrics are missing")
    output: dict[str, object] = {}
    for replica in ("primary", "heldout"):
        source_replica = source.get(replica)
        candidate_replica = candidate.get(replica)
        if not isinstance(source_replica, dict) or not isinstance(
            candidate_replica, dict
        ):
            raise ValueError("deterministic selection replica is invalid")
        output[f"source_{replica}_loss"] = _finite_loss(
            source_replica.get("loss"), f"deterministic source {replica}"
        )
        output[f"candidate_{replica}_loss"] = _finite_loss(
            candidate_replica.get("loss"),
            f"deterministic candidate {replica}",
        )
        output[f"source_{replica}_window_count"] = source_replica.get(
            "valid_window_count"
        )
        output[f"candidate_{replica}_window_count"] = candidate_replica.get(
            "valid_window_count"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--retained-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--support-target", type=Path, required=True)
    parser.add_argument("--deterministic-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("fixed-batch distribution audit seed must equal zero")
    repository = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    retained = args.retained_checkpoint.resolve()
    candidate = args.candidate_checkpoint.resolve()
    reference = args.reference_path.resolve()
    target = args.support_target.resolve()
    selection_path = args.deterministic_selection.resolve()
    preflight = validate_preflight(
        repository=repository,
        checkpoint=retained,
        reference=reference,
        code_commit=args.code_commit,
    )
    if (
        sha256_file(candidate) != CANDIDATE_CHECKPOINT_SHA256
        or sha256_file(candidate.with_name("hparams.json"))
        != CANDIDATE_HPARAMS_SHA256
        or sha256_file(selection_path) != DETERMINISTIC_SELECTION_SHA256
        or sha256_file(target) != SUPPORT_TARGET_SHA256
        or sha256_file(reference) != REFERENCE_SHA256
    ):
        raise ValueError("fixed-batch audit immutable input SHA-256 mismatch")
    target_report = validate_target_artifact(target)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selection.get("policy_retained") is not False
        or selection.get("source_survival") != [136, 144, 84, 90, 79]
        or selection.get("candidate_survival") != [98, 113, 78, 106, 95]
    ):
        raise ValueError("deterministic selection boundary is invalid")
    with retained.open("rb") as stream:
        retained_state = pickle.load(stream)
    with candidate.open("rb") as stream:
        candidate_state = pickle.load(stream)
    if (
        int(candidate_state.step) != START_STEP
        or frozen_controller_residual_depth(candidate_state.actor_params) != 2
        or parameter_tree_sha256(candidate_state.actor_params.parent)
        != parameter_tree_sha256(retained_state.actor_params)
        or parameter_tree_sha256(
            candidate_state.actor_opt.parent_optimizer_state
        )
        != parameter_tree_sha256(retained_state.actor_opt)
        or parameter_tree_sha256(candidate_state.normalizer)
        != parameter_tree_sha256(retained_state.normalizer)
    ):
        raise ValueError("candidate does not have exact retained E002 lineage")
    del retained_state, candidate_state
    gc.collect()
    control, control_report = create_common_state_control(
        candidate_checkpoint=candidate,
        output_root=output_root,
    )
    control_report_path = output_root / "common_state_control.json"
    _write_json_atomically(control_report_path, control_report)
    preflight.update(
        protocol="g1-fixed-batch-distribution-preflight-v1",
        retained_checkpoint=str(retained),
        candidate_checkpoint=str(candidate),
        candidate_checkpoint_sha256=CANDIDATE_CHECKPOINT_SHA256,
        candidate_hparams_sha256=CANDIDATE_HPARAMS_SHA256,
        deterministic_selection=str(selection_path),
        deterministic_selection_sha256=DETERMINISTIC_SELECTION_SHA256,
        support_target=target_report,
        common_state_control_sha256=sha256_file(control),
        start_step=START_STEP,
        end_step=END_STEP,
        diagnostic_updates_per_arm=1,
        optimizer_update_retained=False,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    configure_jax()
    arm_directories = {}
    arm_validations = {}
    for label, initial in (
        ("exact_e002_behavior_control", control),
        ("learned_residual_candidate", candidate),
    ):
        run_directory = _run_arm(
            label=label,
            resume_from=initial,
            profile_name=args.solver_profile,
            reference_path=reference,
            target_path=target,
            seed=args.seed,
            output_root=output_root,
        )
        validation = validate_arm_artifacts(
            run_directory,
            initial_checkpoint=initial,
            candidate_checkpoint=candidate,
            target_path=target,
        )
        validation_path = output_root / label / "validation.json"
        _write_json_atomically(validation_path, validation)
        arm_directories[label] = str(run_directory)
        arm_validations[label] = validation

    source_arm = arm_validations["exact_e002_behavior_control"]
    candidate_arm = arm_validations["learned_residual_candidate"]
    deterministic = _deterministic_metrics(selection)
    classification = classify_fixed_batch_distribution(
        source_primary_loss=source_arm["primary_support_loss"],
        source_heldout_loss=source_arm["heldout_support_loss"],
        candidate_primary_loss=candidate_arm["primary_support_loss"],
        candidate_heldout_loss=candidate_arm["heldout_support_loss"],
        deterministic=deterministic,
    )
    result = {
        "protocol": "g1-fixed-batch-distribution-audit-v1",
        "valid": True,
        **classification,
        "fixed_batch": {
            "source_primary_loss": source_arm["primary_support_loss"],
            "source_heldout_loss": source_arm["heldout_support_loss"],
            "source_valid_window_count": source_arm[
                "valid_support_window_count"
            ],
            "candidate_primary_loss": candidate_arm[
                "primary_support_loss"
            ],
            "candidate_heldout_loss": candidate_arm[
                "heldout_support_loss"
            ],
            "candidate_valid_window_count": candidate_arm[
                "valid_support_window_count"
            ],
        },
        "deterministic_phase_zero": deterministic,
        "gradient_population": {
            "source": source_arm["gradient_population"],
            "candidate": candidate_arm["gradient_population"],
        },
        "gradient_groups": {
            group: {
                "source": source_arm["gradient_groups"][group],
                "candidate": candidate_arm["gradient_groups"][group],
            }
            for group in GROUP_SIZES
        },
        "arm_directories": arm_directories,
        "optimizer_update_retained": False,
        "retained_policy": None,
    }
    result_path = output_root / "paired_distribution.json"
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": "g1-fixed-batch-distribution-completion-v1",
        "valid": True,
        "classification": result["classification"],
        "optimizer_update_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "common_state_control.json": sha256_file(control_report_path),
            "paired_distribution.json": sha256_file(result_path),
            **{
                f"{label}/validation.json": sha256_file(
                    output_root / label / "validation.json"
                )
                for label in arm_validations
            },
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps({"classification": result["classification"], **arm_directories}))


if __name__ == "__main__":
    main()
