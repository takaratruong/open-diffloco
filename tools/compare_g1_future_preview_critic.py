"""Compare current-only and future-preview carried-return critics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.data_structures import Normalizer
from src.core.networks import Critic
from src.algorithms.shac.residual_preview_adapter import FrozenPreviewResidualParams
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.consolidate_g1_carried_return_critic import (
    CONSOLIDATION_STEPS,
    load_frozen_dataset,
)
from tools.evaluate_g1_flax_phase_grid import ACTOR_HISTORY_LEN, LOOKAHEAD_STEPS
from tools.evaluate_g1_terminal_value_calibration import (
    _sha256,
    _write_json_atomic,
    calibration_metrics,
    runtime_asset_provenance,
    validate_code_provenance,
    validate_runtime_contract,
)
from tools.evaluate_g1_tracking import configure_jax, make_evaluation_env
from tools.refit_g1_carried_return_critic import (
    CRITIC_LR,
    _atomic_pickle,
    _h12_records,
    _predict,
    capture_trajectories,
)


CONFIRMATION_PHASES = (15, 115, 215, 315, 415)


def row_phases(
    start_phase: int,
    count: int,
    *,
    reference_length: int,
    reference_stride: int,
) -> np.ndarray:
    """Reconstruct the pre-step reference phase for carried dataset rows."""
    if start_phase < 0 or count < 1 or reference_length < 1 or reference_stride < 1:
        raise ValueError("phase reconstruction arguments are invalid")
    phases = start_phase + np.arange(count, dtype=np.int64) * reference_stride
    return np.minimum(phases, reference_length - 1).astype(np.int32)


def future_preview_rows(
    env,
    *,
    start_phase: int,
    count: int,
    actor_normalizer_state,
) -> np.ndarray:
    """Build normalized future-preview rows in exact carried-row order."""
    phases = row_phases(
        start_phase,
        count,
        reference_length=int(env.reference_length),
        reference_stride=int(env.reference_stride),
    )
    raw = jax.vmap(env._future_reference_command)(jnp.asarray(phases))
    future_dim = int(env.actor_future_reference_dim)
    if raw.shape != (count, future_dim):
        raise ValueError("future-reference command shape does not match environment")
    if actor_normalizer_state.mean.shape[0] < future_dim:
        raise ValueError("actor normalizer is narrower than future preview")
    suffix_state = actor_normalizer_state.replace(
        mean=actor_normalizer_state.mean[-future_dim:],
        var=actor_normalizer_state.var[-future_dim:],
    )
    normalized = Normalizer(future_dim).normalize(suffix_state, raw)
    result = np.asarray(normalized, dtype=np.float32)
    if result.shape != (count, future_dim) or not np.all(np.isfinite(result)):
        raise ValueError("normalized future-reference rows are malformed")
    return result


def _replace_first_kernel(params: Any, kernel: jax.Array) -> Any:
    copied = copy.deepcopy(params)
    try:
        copied["params"]["Dense_0"]["kernel"] = kernel
    except (KeyError, TypeError) as error:
        raise ValueError("critic parameters must expose params/Dense_0/kernel") from error
    return copied


def migrate_critic_input(
    params: Any,
    opt_state: Any,
    *,
    extra_dim: int,
    optimizer,
) -> tuple[Any, Any]:
    """Append zero critic input rows and matching Adam moment rows."""
    if extra_dim < 1:
        raise ValueError("extra critic input dimension must be positive")
    try:
        old_kernel = params["params"]["Dense_0"]["kernel"]
    except (KeyError, TypeError) as error:
        raise ValueError("critic parameters must expose params/Dense_0/kernel") from error
    if old_kernel.ndim != 2:
        raise ValueError("critic input kernel must be a matrix")
    new_kernel = jnp.concatenate(
        (
            old_kernel,
            jnp.zeros((extra_dim, old_kernel.shape[1]), dtype=old_kernel.dtype),
        ),
        axis=0,
    )
    new_params = _replace_first_kernel(params, new_kernel)
    template = optimizer.init(new_params)
    old_shape = old_kernel.shape
    new_shape = new_kernel.shape
    expanded = 0

    def migrate(old, new):
        nonlocal expanded
        old_array = jnp.asarray(old)
        new_array = jnp.asarray(new)
        if old_array.shape == new_array.shape:
            return old
        if old_array.shape == old_shape and new_array.shape == new_shape:
            expanded += 1
            return jnp.concatenate(
                (
                    old_array,
                    jnp.zeros(
                        (extra_dim, old_array.shape[1]), dtype=old_array.dtype
                    ),
                ),
                axis=0,
            )
        raise ValueError(
            "critic optimizer has unsupported shape mismatch: "
            f"{old_array.shape} -> {new_array.shape}"
        )

    migrated_opt = jax.tree_util.tree_map(migrate, opt_state, template)
    if expanded != 2:
        raise ValueError("critic optimizer migration must expand Adam mu and nu")
    return new_params, migrated_opt


def build_paired_dataset(env, checkpoint_state, raw_dataset):
    """Normalize current critic rows and append phase-matched future preview."""
    critic_normalizer = Normalizer(int(env.critic_obs_dim))
    paired = {}
    for phase, rows in raw_dataset.items():
        raw_observations = jnp.asarray(rows["critic_observations"])
        control = np.asarray(
            critic_normalizer.normalize(
                checkpoint_state.critic_normalizer, raw_observations
            ),
            dtype=np.float32,
        )
        future = future_preview_rows(
            env,
            start_phase=int(phase),
            count=control.shape[0],
            actor_normalizer_state=checkpoint_state.normalizer,
        )
        preview = np.concatenate((control, future), axis=-1)
        if (
            control.shape != (control.shape[0], int(env.critic_obs_dim))
            or preview.shape[-1]
            != int(env.critic_obs_dim) + int(env.actor_future_reference_dim)
            or not np.all(np.isfinite(control))
            or not np.all(np.isfinite(preview))
        ):
            raise ValueError("paired critic dataset contains malformed observations")
        paired[int(phase)] = {
            **rows,
            "control_observations": control,
            "preview_observations": preview,
        }
    return paired


def fit_critic_arm(
    critic,
    params,
    opt_state,
    observations,
    returns,
    *,
    steps: int,
    optimizer=None,
):
    """Fit one critic arm for an exact full-dataset update count."""
    if steps < 1:
        raise ValueError("critic fit steps must be positive")
    obs = jnp.asarray(observations, dtype=jnp.float32)
    targets = jnp.asarray(returns, dtype=jnp.float32)
    if (
        obs.ndim != 2
        or targets.shape != (obs.shape[0],)
        or obs.shape[0] < 1
        or not bool(jnp.all(jnp.isfinite(obs)))
        or not bool(jnp.all(jnp.isfinite(targets)))
    ):
        raise ValueError("critic fit arrays are malformed or nonfinite")
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(5e-4)
        )

    @jax.jit
    def update(current_params, current_opt):
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean(
                jnp.square(critic.apply(p, obs).reshape(-1) - targets)
            )
        )(current_params)
        updates, next_opt = optimizer.update(grads, current_opt)
        return optax.apply_updates(current_params, updates), next_opt, loss

    loss = None
    for _ in range(steps):
        params, opt_state, loss = update(params, opt_state)
    final_loss = float(loss)
    if not np.isfinite(final_loss):
        raise ValueError("critic fit produced nonfinite loss")
    return params, opt_state, final_loss


def concatenate_paired(
    paired_dataset,
    phases: tuple[int, ...],
    *,
    observation_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate one paired observation arm in explicit phase order."""
    if observation_key not in {"control_observations", "preview_observations"}:
        raise ValueError("unknown paired critic observation key")
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("paired phases must be nonempty and unique")
    try:
        observations = np.concatenate(
            [paired_dataset[phase][observation_key] for phase in phases]
        )
        returns = np.concatenate(
            [paired_dataset[phase]["returns"] for phase in phases]
        )
    except KeyError as error:
        raise ValueError("paired dataset is missing a required phase or field") from error
    if observations.shape[0] != returns.size:
        raise ValueError("paired observation and return counts differ")
    return observations, returns


def validate_initial_equivalence(
    control_predictions,
    preview_predictions,
    *,
    tolerance: float,
) -> float:
    """Fail closed unless zero-row migration preserves initial predictions."""
    control = np.asarray(control_predictions)
    preview = np.asarray(preview_predictions)
    if control.shape != preview.shape or control.size < 1:
        raise ValueError("initial prediction arrays differ in shape")
    if not np.all(np.isfinite(control)) or not np.all(np.isfinite(preview)):
        raise ValueError("initial prediction arrays must be finite")
    drift = float(np.max(np.abs(control - preview)))
    if drift > tolerance:
        raise ValueError(
            f"initial predictions drift {drift} exceeds tolerance {tolerance}"
        )
    return drift


def write_confirmation_artifact(
    path: Path,
    confirmation,
    *,
    phases: tuple[int, ...],
) -> None:
    """Atomically retain the complete shared raw confirmation trajectories."""
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("confirmation artifact phases must be nonempty and unique")
    payload = {}
    for phase in phases:
        try:
            rows = confirmation[phase]
            observations = np.asarray(rows["critic_observations"])
            rewards = np.asarray(rows["rewards"])
            returns = np.asarray(rows["returns"])
        except KeyError as error:
            raise ValueError("confirmation artifact is missing required rows") from error
        if (
            observations.ndim != 2
            or rewards.shape != (observations.shape[0],)
            or returns.shape != (observations.shape[0],)
            or not np.all(np.isfinite(observations))
            or not np.all(np.isfinite(rewards))
            or not np.all(np.isfinite(returns))
        ):
            raise ValueError("confirmation artifact rows are malformed or nonfinite")
        payload[f"phase_{phase}_critic_observations"] = observations
        payload[f"phase_{phase}_rewards"] = rewards
        payload[f"phase_{phase}_returns"] = returns
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def future_preview_advances(
    original_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    preview_metrics: dict[str, float],
    baseline_h12: list[dict[str, float | int]],
    preview_h12: list[dict[str, float | int]],
) -> bool:
    """Apply the preregistered absolute and paired representation gates."""
    if [row["phase"] for row in baseline_h12] != [
        row["phase"] for row in preview_h12
    ]:
        raise ValueError("paired H12 phase grids differ")
    values = tuple(original_metrics.values()) + tuple(baseline_metrics.values()) + tuple(
        preview_metrics.values()
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("critic metrics must be finite")
    return bool(
        preview_metrics["rank_correlation"] >= 0.8
        and preview_metrics["nrmse"] <= 0.25
        and preview_metrics["rank_correlation"]
        > baseline_metrics["rank_correlation"]
        and preview_metrics["nrmse"] < baseline_metrics["nrmse"]
        and preview_metrics["rank_correlation"]
        > original_metrics["rank_correlation"]
        and preview_metrics["nrmse"] < original_metrics["nrmse"]
        and all(float(row["relative_error"]) <= 0.25 for row in preview_h12)
        and all(
            float(treatment["relative_error"])
            <= float(control["relative_error"])
            for control, treatment in zip(baseline_h12, preview_h12)
        )
    )


def _predictions_by_phase(
    predictions: np.ndarray,
    paired_dataset,
    phases: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Split concatenated predictions by the paired trajectory row counts."""
    result = {}
    offset = 0
    for phase in phases:
        count = int(np.asarray(paired_dataset[phase]["returns"]).size)
        result[phase] = predictions[offset : offset + count]
        offset += count
    if offset != predictions.size:
        raise ValueError("prediction count does not match phase trajectories")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-hparams-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--solver-profile", default="g1-4x5")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    runtime_contract = validate_runtime_contract(
        solver_profile=args.solver_profile, seed=args.seed
    )
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    code_provenance = validate_code_provenance(
        expected_commit=args.code_commit,
        actual_commit=actual_commit,
        dirty=dirty,
    )
    checkpoint_path = args.checkpoint.resolve()
    hparams_path = args.hparams.resolve()
    dataset_path = args.dataset.resolve()
    reference_path = args.reference_path.resolve()
    for path, expected, label in (
        (checkpoint_path, args.expected_checkpoint_sha256, "checkpoint"),
        (hparams_path, args.expected_hparams_sha256, "hparams"),
        (dataset_path, args.expected_dataset_sha256, "dataset"),
        (reference_path, args.expected_reference_sha256, "reference"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    with checkpoint_path.open("rb") as stream:
        checkpoint_state = pickle.load(stream)
    if not isinstance(checkpoint_state.actor_params, FrozenPreviewResidualParams):
        raise ValueError("checkpoint is not a frozen residual actor")

    profile = get_solver_profile(args.solver_profile)
    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_source_step",
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference_path,
        reference_stride=1,
        actor_history_len=ACTOR_HISTORY_LEN,
        actor_reference_lookahead_steps=LOOKAHEAD_STEPS,
        actor_reference_preview_mode="delta",
        reference_residual_control=True,
        reference_residual_scale=0.5,
    )
    provenance = {
        **code_provenance,
        **runtime_contract,
        **runtime_asset_provenance(env),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "hparams_path": str(hparams_path),
        "hparams_sha256": _sha256(hparams_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
    }

    raw_dataset = load_frozen_dataset(dataset_path)
    paired_dataset = build_paired_dataset(env, checkpoint_state, raw_dataset)
    fit_phases = tuple(raw_dataset)
    control_obs, fit_returns = concatenate_paired(
        paired_dataset,
        fit_phases,
        observation_key="control_observations",
    )
    preview_obs, preview_returns = concatenate_paired(
        paired_dataset,
        fit_phases,
        observation_key="preview_observations",
    )
    np.testing.assert_array_equal(fit_returns, preview_returns)

    critic = Critic()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(CRITIC_LR)
    )
    preview_initial_params, preview_initial_opt = migrate_critic_input(
        checkpoint_state.critic_params,
        checkpoint_state.critic_opt,
        extra_dim=int(env.actor_future_reference_dim),
        optimizer=optimizer,
    )
    initial_drift = validate_initial_equivalence(
        _predict(critic, checkpoint_state.critic_params, jnp.asarray(control_obs)),
        _predict(critic, preview_initial_params, jnp.asarray(preview_obs)),
        tolerance=1e-6,
    )
    initial_count = int(np.asarray(checkpoint_state.critic_opt[1][0].count))
    baseline_params, baseline_opt, baseline_fit_loss = fit_critic_arm(
        critic,
        checkpoint_state.critic_params,
        checkpoint_state.critic_opt,
        control_obs,
        fit_returns,
        steps=CONSOLIDATION_STEPS,
        optimizer=optimizer,
    )
    preview_params, preview_opt, preview_fit_loss = fit_critic_arm(
        critic,
        preview_initial_params,
        preview_initial_opt,
        preview_obs,
        fit_returns,
        steps=CONSOLIDATION_STEPS,
        optimizer=optimizer,
    )
    expected_count = initial_count + CONSOLIDATION_STEPS
    if (
        int(np.asarray(baseline_opt[1][0].count)) != expected_count
        or int(np.asarray(preview_opt[1][0].count)) != expected_count
    ):
        raise ValueError("paired critic arms did not execute the exact update budget")

    with solver_context(profile):
        confirmation = capture_trajectories(
            env,
            checkpoint_state,
            seed=args.seed,
            phases=CONFIRMATION_PHASES,
        )
    paired_confirmation = build_paired_dataset(
        env, checkpoint_state, confirmation
    )
    confirmation_control, confirmation_returns = concatenate_paired(
        paired_confirmation,
        CONFIRMATION_PHASES,
        observation_key="control_observations",
    )
    confirmation_preview, preview_confirmation_returns = concatenate_paired(
        paired_confirmation,
        CONFIRMATION_PHASES,
        observation_key="preview_observations",
    )
    np.testing.assert_array_equal(
        confirmation_returns, preview_confirmation_returns
    )

    original_predictions = _predict(
        critic,
        checkpoint_state.target_critic_params,
        jnp.asarray(confirmation_control),
    )
    baseline_predictions = _predict(
        critic, baseline_params, jnp.asarray(confirmation_control)
    )
    preview_predictions = _predict(
        critic, preview_params, jnp.asarray(confirmation_preview)
    )
    original_metrics = calibration_metrics(
        original_predictions, confirmation_returns
    )
    baseline_metrics = calibration_metrics(
        baseline_predictions, confirmation_returns
    )
    preview_metrics = calibration_metrics(
        preview_predictions, confirmation_returns
    )
    original_h12 = _h12_records(
        paired_confirmation,
        CONFIRMATION_PHASES,
        _predictions_by_phase(
            original_predictions, paired_confirmation, CONFIRMATION_PHASES
        ),
    )
    baseline_h12 = _h12_records(
        paired_confirmation,
        CONFIRMATION_PHASES,
        _predictions_by_phase(
            baseline_predictions, paired_confirmation, CONFIRMATION_PHASES
        ),
    )
    preview_h12 = _h12_records(
        paired_confirmation,
        CONFIRMATION_PHASES,
        _predictions_by_phase(
            preview_predictions, paired_confirmation, CONFIRMATION_PHASES
        ),
    )
    success = future_preview_advances(
        original_metrics,
        baseline_metrics,
        preview_metrics,
        baseline_h12,
        preview_h12,
    )

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    confirmation_path = output / "confirmation_trajectories.npz"
    write_confirmation_artifact(
        confirmation_path,
        confirmation,
        phases=CONFIRMATION_PHASES,
    )
    confirmation_sha256 = _sha256(confirmation_path)
    artifact_path = output / "future_preview_critic.pkl"
    artifact_sha256 = None
    if success:
        future_dim = int(env.actor_future_reference_dim)
        augmented_normalizer = checkpoint_state.critic_normalizer.replace(
            mean=jnp.concatenate(
                (
                    checkpoint_state.critic_normalizer.mean,
                    checkpoint_state.normalizer.mean[-future_dim:],
                )
            ),
            var=jnp.concatenate(
                (
                    checkpoint_state.critic_normalizer.var,
                    checkpoint_state.normalizer.var[-future_dim:],
                )
            ),
        )
        _atomic_pickle(
            artifact_path,
            {
                "protocol": "g1-future-preview-critic-discriminator-v1",
                "critic_params": preview_params,
                "critic_opt": preview_opt,
                "critic_normalizer": augmented_normalizer,
                "base_critic_obs_dim": int(env.critic_obs_dim),
                "future_reference_dim": future_dim,
                "lookahead_steps": LOOKAHEAD_STEPS,
                "preview_mode": "delta",
                "source_checkpoint_sha256": _sha256(checkpoint_path),
                "source_dataset_sha256": _sha256(dataset_path),
            },
        )
        artifact_sha256 = _sha256(artifact_path)
    report = {
        "protocol": "g1-future-preview-critic-discriminator-v1",
        "provenance": provenance,
        "fit_phases": list(fit_phases),
        "fit_rows": int(fit_returns.size),
        "fit_steps_per_arm": CONSOLIDATION_STEPS,
        "critic_lr": CRITIC_LR,
        "base_critic_obs_dim": int(env.critic_obs_dim),
        "future_reference_dim": int(env.actor_future_reference_dim),
        "preview_critic_obs_dim": int(
            env.critic_obs_dim + env.actor_future_reference_dim
        ),
        "initial_prediction_max_abs_drift": initial_drift,
        "initial_optimizer_count": initial_count,
        "final_optimizer_count": expected_count,
        "baseline_fit_loss": baseline_fit_loss,
        "preview_fit_loss": preview_fit_loss,
        "confirmation_phases": list(CONFIRMATION_PHASES),
        "confirmation_survival": {
            str(phase): int(confirmation[phase]["returns"].size)
            for phase in CONFIRMATION_PHASES
        },
        "confirmation_artifact": {
            "path": confirmation_path.name,
            "sha256": confirmation_sha256,
        },
        "original_metrics": original_metrics,
        "baseline_metrics": baseline_metrics,
        "preview_metrics": preview_metrics,
        "original_h12_records": original_h12,
        "baseline_h12_records": baseline_h12,
        "preview_h12_records": preview_h12,
        "outcome": (
            "future-preview-critic-advances"
            if success
            else "future-preview-critic-insufficient"
        ),
        "artifact_sha256": artifact_sha256,
    }
    _write_json_atomic(output / "future_preview_critic_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
