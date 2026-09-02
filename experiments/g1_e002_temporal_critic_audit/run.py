"""Test a capacity-controlled temporal value residual on E010's frozen data."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.g1_e002_critic_representation_audit.run import (
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    CRITIC_LR,
    CRITIC_OBS_DIM,
    EVALUATION_INTERVAL,
    FIT_STEPS,
    MAX_NRMSE,
    METRIC_GROUPS,
    MIN_RANK_CORRELATION,
    START_STEP,
    _atomic_npz,
    _validate_runtime,
    representation_adequate,
    summarize_representation_metrics,
    validation_candidate_key,
)
from src.core.networks import Critic
from tools.run_g1_dual_scale_root_position import (
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_HPARAMS_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


SOURCE_DATASET_SHA256 = (
    "5468c73d586f6d7b13ebe282eb2e0e3f6b3de73752d5b22a3166332ece95c3dd"
)
SOURCE_AUDIT_SHA256 = "69591232c064911791a84a58dcc335029ecfe71afb083e2f5fa9554040c41089"
SOURCE_COMPLETION_SHA256 = (
    "13d6233c6bbb700a871e70791d21bf38818a5e9425137380af9414b0d99a1778"
)
SOURCE_PROTOCOL = "g1-e002-critic-representation-raw-history-h0-audit-v1"
SOURCE_CONTROL = "current_plus_actor_latent"
TEMPORAL_HIDDEN_DIM = 64
TEMPORAL_INIT_SEED = 20_260_902
TEMPORAL_PARAMETER_COUNT = 75_585
COMBINED_PARAMETER_COUNT = 388_674
REQUIRED_BOUNDARIES = ("aggregate", "h0", "h24")
EXPECTED_ROWS = 38_384
EXPECTED_FIT_ROWS = 5_844
EXPECTED_VALIDATION_ROWS = 7_959
EXPECTED_TEST_ROWS = 8_001


class TemporalHistoryEncoder(nn.Module):
    """Encode oldest-to-newest actor frames with one shared GRU."""

    frame_dim: int = ACTOR_FRAME_DIM
    history_len: int = ACTOR_HISTORY_LEN
    hidden_dim: int = TEMPORAL_HIDDEN_DIM

    @nn.compact
    def __call__(self, histories):
        if histories.ndim != 3 or histories.shape[-2:] != (
            self.history_len,
            self.frame_dim,
        ):
            raise ValueError("temporal history shape is invalid")
        carry, _ = nn.RNN(
            nn.GRUCell(self.hidden_dim),
            return_carry=True,
        )(histories)
        return carry


class TemporalValueResidual(nn.Module):
    """Map full actor history to an exactly zero-initialized value residual."""

    frame_dim: int = ACTOR_FRAME_DIM
    history_len: int = ACTOR_HISTORY_LEN
    hidden_dim: int = TEMPORAL_HIDDEN_DIM

    @nn.compact
    def __call__(self, histories):
        encoded = TemporalHistoryEncoder(
            frame_dim=self.frame_dim,
            history_len=self.history_len,
            hidden_dim=self.hidden_dim,
            name="history_encoder",
        )(histories)
        return nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="residual_head",
        )(encoded)


def parameter_count(params: Any) -> int:
    """Count numeric parameters in a pytree."""

    return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(params)))


def optimizer_count(opt_state: Any) -> int:
    """Read the Adam count from the registered clip-plus-Adam chain."""

    try:
        return int(np.asarray(opt_state[1][0].count))
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("temporal critic optimizer count is unavailable") from error


def graft_temporal_optimizer(critic_opt: Any, combined_opt: Any) -> Any:
    """Preserve critic Adam history and add exact-zero temporal moments."""

    try:
        old_adam = critic_opt[1][0]
        new_adam = combined_opt[1][0]
        new_mu = new_adam.mu
        new_nu = new_adam.nu
        if set(new_mu) != {"critic", "temporal"} or set(new_nu) != {
            "critic",
            "temporal",
        }:
            raise ValueError("combined Adam tree is malformed")
        if (
            jax.tree.structure(old_adam.mu) != jax.tree.structure(new_mu["critic"])
            or jax.tree.structure(old_adam.nu) != jax.tree.structure(new_nu["critic"])
            or int(np.asarray(new_adam.count)) != 0
        ):
            raise ValueError("critic Adam tree is incompatible")
        if any(
            np.count_nonzero(np.asarray(value))
            for tree in (new_mu["temporal"], new_nu["temporal"])
            for value in jax.tree.leaves(tree)
        ):
            raise ValueError("new temporal Adam moments are not zero")
        grafted_adam = new_adam._replace(
            count=old_adam.count,
            mu={"critic": old_adam.mu, "temporal": new_mu["temporal"]},
            nu={"critic": old_adam.nu, "temporal": new_nu["temporal"]},
        )
        grafted = (combined_opt[0], (grafted_adam, combined_opt[1][1]))
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("temporal optimizer graft is incompatible") from error
    if optimizer_count(grafted) != optimizer_count(critic_opt):
        raise ValueError("temporal optimizer graft changed the Adam count")
    return grafted


def validate_temporal_dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    critic_dim: int = CRITIC_OBS_DIM,
    history_len: int = ACTOR_HISTORY_LEN,
    frame_dim: int = ACTOR_FRAME_DIM,
) -> dict[str, np.ndarray]:
    """Validate and reshape E010 rows without changing history order."""

    required = (
        "normalized_current_critic_obs",
        "normalized_actor_history",
        "realized_return",
        "arm",
        "environment_id",
        "time_index",
        "split_code",
        "fit_row_selected",
    )
    try:
        source = {name: np.asarray(arrays[name]) for name in required}
    except KeyError as error:
        raise ValueError("temporal dataset is missing a required array") from error
    current = source["normalized_current_critic_obs"]
    flat_history = source["normalized_actor_history"]
    targets = source["realized_return"]
    rows = int(targets.size)
    vectors = (
        source["arm"],
        source["environment_id"],
        source["time_index"],
        source["split_code"],
        source["fit_row_selected"],
    )
    if (
        rows < 1
        or targets.shape != (rows,)
        or current.shape != (rows, critic_dim)
        or flat_history.shape != (rows, history_len * frame_dim)
        or any(value.shape != (rows,) for value in vectors)
        or not np.isfinite(current).all()
        or not np.isfinite(flat_history).all()
        or not np.isfinite(targets).all()
        or not set(np.unique(source["arm"])).issubset({0, 1})
        or set(np.unique(source["split_code"])) != {0, 1, 2}
    ):
        raise ValueError("temporal dataset arrays are malformed")
    fit_mask = np.asarray(source["fit_row_selected"], dtype=bool)
    validation_mask = source["split_code"] == 1
    test_mask = source["split_code"] == 2
    expected_fit = (source["split_code"] == 0) & (source["time_index"] % 4 == 0)
    if not np.array_equal(fit_mask, expected_fit):
        raise ValueError("temporal dataset fit mask drifted")
    for environment_id in np.unique(source["environment_id"]):
        environment_codes = np.unique(
            source["split_code"][source["environment_id"] == environment_id]
        )
        if environment_codes.size != 1:
            raise ValueError("temporal dataset leaks an environment across splits")
    return {
        "current": np.asarray(current, dtype=np.float32),
        "history": np.asarray(flat_history, dtype=np.float32).reshape(
            rows,
            history_len,
            frame_dim,
        ),
        "return": np.asarray(targets, dtype=np.float64),
        "arm": np.asarray(source["arm"], dtype=np.int8),
        "environment_id": np.asarray(source["environment_id"], dtype=np.int32),
        "time_index": np.asarray(source["time_index"], dtype=np.int32),
        "fit_mask": fit_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "split_code": np.asarray(source["split_code"], dtype=np.int8),
    }


def _predict_temporal(
    critic: Critic,
    temporal: TemporalValueResidual,
    params: Any,
    current: np.ndarray,
    history: np.ndarray,
) -> np.ndarray:
    @jax.jit
    def predict(candidate, current_values, history_values):
        base = critic.apply(candidate["critic"], current_values).reshape(-1)
        residual = temporal.apply(candidate["temporal"], history_values).reshape(-1)
        return base + residual

    return np.asarray(
        predict(
            params,
            jnp.asarray(current, dtype=jnp.float32),
            jnp.asarray(history, dtype=jnp.float32),
        ),
        dtype=np.float64,
    )


def fit_temporal_critic_with_validation(
    critic: Critic,
    temporal: TemporalValueResidual,
    params: Any,
    opt_state: Any,
    current: np.ndarray,
    history: np.ndarray,
    realized_returns: np.ndarray,
    arm: np.ndarray,
    time_index: np.ndarray,
    *,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    steps: int,
    evaluation_interval: int,
    optimizer,
    boundary_indices: Mapping[str, int],
) -> tuple[Any, dict[str, Any]]:
    """Fit one temporal arm and select snapshots only on validation rows."""

    current_values = np.asarray(current, dtype=np.float32)
    histories = np.asarray(history, dtype=np.float32)
    targets = np.asarray(realized_returns, dtype=np.float32)
    arms = np.asarray(arm)
    times = np.asarray(time_index)
    fitting = np.asarray(fit_mask, dtype=bool)
    validating = np.asarray(validation_mask, dtype=bool)
    if (
        current_values.ndim != 2
        or histories.ndim != 3
        or histories.shape[0] != current_values.shape[0]
        or targets.shape != (current_values.shape[0],)
        or arms.shape != targets.shape
        or times.shape != targets.shape
        or fitting.shape != targets.shape
        or validating.shape != targets.shape
        or np.any(fitting & validating)
        or np.sum(fitting) < 2
        or np.sum(validating) < 2
        or steps < 1
        or evaluation_interval < 1
        or steps % evaluation_interval != 0
        or not np.isfinite(current_values).all()
        or not np.isfinite(histories).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("temporal critic fit contract is invalid")
    fit_current = jnp.asarray(current_values[fitting])
    fit_history = jnp.asarray(histories[fitting])
    fit_returns = jnp.asarray(targets[fitting])
    validation_current = jnp.asarray(current_values[validating])
    validation_history = jnp.asarray(histories[validating])
    validation_targets = targets[validating]
    validation_arms = arms[validating]
    validation_times = times[validating]
    boundaries = dict(boundary_indices)
    required_boundaries = ("aggregate", *boundaries)

    @jax.jit
    def update(current_params, current_opt_state):
        def loss(candidate):
            base = critic.apply(candidate["critic"], fit_current).reshape(-1)
            residual = temporal.apply(candidate["temporal"], fit_history).reshape(-1)
            return jnp.mean(jnp.square(base + residual - fit_returns))

        value, gradients = jax.value_and_grad(loss)(current_params)
        updates, next_opt_state = optimizer.update(gradients, current_opt_state)
        return optax.apply_updates(current_params, updates), next_opt_state, value

    @jax.jit
    def predict(candidate, current_batch, history_batch):
        base = critic.apply(candidate["critic"], current_batch).reshape(-1)
        residual = temporal.apply(candidate["temporal"], history_batch).reshape(-1)
        return base + residual

    def candidate(step: int, loss: float | None) -> dict[str, Any]:
        predictions = np.asarray(
            predict(params, validation_current, validation_history),
            dtype=np.float64,
        )
        metrics = summarize_representation_metrics(
            predictions,
            validation_targets,
            validation_arms,
            validation_times,
            np.ones(validation_targets.shape, dtype=bool),
            boundary_indices=boundaries,
        )
        return {"step": step, "fit_loss": loss, "metrics": metrics}

    initial_count = optimizer_count(opt_state)
    first = candidate(0, None)
    candidates = [first]
    best_key = validation_candidate_key(
        first["metrics"],
        step=0,
        required_boundaries=required_boundaries,
    )
    best_params = params
    selected = first
    last_loss = None
    for step in range(evaluation_interval, steps + 1, evaluation_interval):
        for _ in range(evaluation_interval):
            params, opt_state, loss = update(params, opt_state)
        last_loss = float(loss)
        row = candidate(step, last_loss)
        candidates.append(row)
        key = validation_candidate_key(
            row["metrics"],
            step=step,
            required_boundaries=required_boundaries,
        )
        if key > best_key:
            best_key = key
            best_params = params
            selected = row
    final_count = optimizer_count(opt_state)
    if final_count != initial_count + steps or not math.isfinite(float(last_loss)):
        raise ValueError("temporal critic fit did not execute its exact budget")
    return best_params, {
        "fit_rows": int(np.sum(fitting)),
        "validation_rows": int(np.sum(validating)),
        "executed_steps": steps,
        "evaluation_interval": evaluation_interval,
        "initial_optimizer_count": initial_count,
        "final_optimizer_count": final_count,
        "final_fit_loss": float(last_loss),
        "selected": selected,
        "candidates": candidates,
    }


def _quality(metrics: Mapping[str, Any]) -> tuple[float, float]:
    rows = [
        metrics[group][boundary]
        for group in METRIC_GROUPS
        for boundary in REQUIRED_BOUNDARIES
    ]
    errors = [float(row["nrmse"]) for row in rows]
    ranks = [float(row["rank_correlation"]) for row in rows]
    if not all(math.isfinite(value) for value in (*errors, *ranks)):
        raise ValueError("temporal classification metrics are nonfinite")
    return -max(errors), min(ranks)


def classify_temporal_representation(
    temporal_metrics: Mapping[str, Any],
    control_metrics: Mapping[str, Any],
) -> str:
    """Apply the frozen scalar gate, then compare with the final-latent control."""

    if representation_adequate(
        temporal_metrics,
        required_boundaries=REQUIRED_BOUNDARIES,
    ):
        return "temporal-representation-adequate"
    if _quality(temporal_metrics) > _quality(control_metrics):
        return "temporal-improves-but-insufficient"
    return "temporal-no-gain"


def _metric_max_abs_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    maximum = 0.0
    for group in METRIC_GROUPS:
        for boundary in REQUIRED_BOUNDARIES:
            for field in (
                "count",
                "pearson",
                "rank_correlation",
                "rmse",
                "nrmse",
                "bias",
            ):
                maximum = max(
                    maximum,
                    abs(
                        float(left[group][boundary][field])
                        - float(right[group][boundary][field])
                    ),
                )
    return maximum


def _load_source_bundle(
    dataset_path: Path,
    audit_path: Path,
    completion_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sha256_file(dataset_path) != SOURCE_DATASET_SHA256:
        raise ValueError("E010 dataset SHA-256 mismatch")
    if sha256_file(audit_path) != SOURCE_AUDIT_SHA256:
        raise ValueError("E010 audit SHA-256 mismatch")
    if sha256_file(completion_path) != SOURCE_COMPLETION_SHA256:
        raise ValueError("E010 completion SHA-256 mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        audit.get("protocol") != SOURCE_PROTOCOL
        or audit.get("valid") is not True
        or audit.get("policy_retained") is not False
        or audit.get("critic_retained") is not False
        or audit.get("dataset_sha256") != SOURCE_DATASET_SHA256
        or completion.get("valid") is not True
        or completion.get("artifacts", {}).get("critic_representation_dataset.npz")
        != SOURCE_DATASET_SHA256
        or completion.get("artifacts", {}).get("critic_representation_audit.json")
        != SOURCE_AUDIT_SHA256
    ):
        raise ValueError("E010 source bundle contract drifted")
    required = (
        "normalized_current_critic_obs",
        "normalized_actor_history",
        "realized_return",
        "arm",
        "environment_id",
        "time_index",
        "split_code",
        "fit_row_selected",
        "selected_current_plus_actor_latent_prediction",
    )
    with np.load(dataset_path, allow_pickle=False) as archive:
        try:
            arrays = {name: np.asarray(archive[name]) for name in required}
        except KeyError as error:
            raise ValueError("E010 dataset lacks a temporal input") from error
    return arrays, audit


def collect_temporal_audit(
    *,
    checkpoint: Path,
    dataset_path: Path,
    audit_path: Path,
    completion_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit and evaluate one parameter-matched temporal residual critic."""

    arrays, source_audit = _load_source_bundle(
        dataset_path,
        audit_path,
        completion_path,
    )
    dataset = validate_temporal_dataset(arrays)
    if (
        dataset["return"].size != EXPECTED_ROWS
        or int(np.sum(dataset["fit_mask"])) != EXPECTED_FIT_ROWS
        or int(np.sum(dataset["validation_mask"])) != EXPECTED_VALIDATION_ROWS
        or int(np.sum(dataset["test_mask"])) != EXPECTED_TEST_ROWS
    ):
        raise ValueError("E010 temporal row counts drifted")
    source_control = source_audit["selected_final_test"][SOURCE_CONTROL]
    source_control_recomputed = summarize_representation_metrics(
        arrays["selected_current_plus_actor_latent_prediction"],
        dataset["return"],
        dataset["arm"],
        dataset["time_index"],
        dataset["test_mask"],
        boundary_indices={"h0": 0, "h24": 24},
    )
    source_control_delta = _metric_max_abs_delta(
        source_control_recomputed,
        source_control,
    )
    if source_control_delta != 0.0:
        raise ValueError("E010 final-latent control does not reproduce exactly")

    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if int(state.step) != START_STEP:
        raise ValueError("retained E002 step drifted")
    critic = Critic()
    temporal = TemporalValueResidual()
    temporal_params = temporal.init(
        jax.random.PRNGKey(TEMPORAL_INIT_SEED),
        jnp.asarray(dataset["history"][:1]),
    )
    if parameter_count(temporal_params) != TEMPORAL_PARAMETER_COUNT:
        raise ValueError("temporal residual parameter count drifted")
    params = {"critic": state.critic_params, "temporal": temporal_params}
    if parameter_count(params) != COMBINED_PARAMETER_COUNT:
        raise ValueError("combined temporal critic parameter count drifted")
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(CRITIC_LR))
    opt_state = graft_temporal_optimizer(
        state.critic_opt,
        optimizer.init(params),
    )
    baseline = np.asarray(
        jax.jit(lambda p, x: critic.apply(p, x).reshape(-1))(
            state.critic_params,
            jnp.asarray(dataset["current"]),
        ),
        dtype=np.float64,
    )
    initial = _predict_temporal(
        critic,
        temporal,
        params,
        dataset["current"],
        dataset["history"],
    )
    initial_drift = float(np.max(np.abs(initial - baseline)))
    if initial_drift > 1e-6:
        raise ValueError("temporal residual changed the initial E002 value")

    selected_params, fit_report = fit_temporal_critic_with_validation(
        critic,
        temporal,
        params,
        opt_state,
        dataset["current"],
        dataset["history"],
        dataset["return"],
        dataset["arm"],
        dataset["time_index"],
        fit_mask=dataset["fit_mask"],
        validation_mask=dataset["validation_mask"],
        steps=FIT_STEPS,
        evaluation_interval=EVALUATION_INTERVAL,
        optimizer=optimizer,
        boundary_indices={"h0": 0, "h24": 24},
    )
    selected_prediction = _predict_temporal(
        critic,
        temporal,
        selected_params,
        dataset["current"],
        dataset["history"],
    )
    final_test = summarize_representation_metrics(
        selected_prediction,
        dataset["return"],
        dataset["arm"],
        dataset["time_index"],
        dataset["test_mask"],
        boundary_indices={"h0": 0, "h24": 24},
    )
    classification = classify_temporal_representation(final_test, source_control)
    arrays_out = {
        "selected_temporal_prediction": selected_prediction,
    }
    result = {
        "protocol": "g1-e002-temporal-residual-critic-audit-v1",
        "valid": True,
        "classification": classification,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_hparams_sha256": SOURCE_HPARAMS_SHA256,
        "source_dataset_sha256": SOURCE_DATASET_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "source_completion_sha256": SOURCE_COMPLETION_SHA256,
        "source_control": {
            "name": SOURCE_CONTROL,
            "final_test": source_control,
            "recompute_max_abs_delta": source_control_delta,
        },
        "dataset": {
            "row_count": EXPECTED_ROWS,
            "fit_rows": EXPECTED_FIT_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "test_rows": EXPECTED_TEST_ROWS,
            "history_shape": [EXPECTED_ROWS, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM],
            "history_order": "oldest-to-newest",
            "paired_environment_split": True,
            "source_raw_history_statistics": source_audit["dataset"][
                "raw_history_statistics"
            ],
        },
        "model": {
            "architecture": "e002-current-critic-plus-zero-head-flax-gru-residual",
            "temporal_hidden_dim": TEMPORAL_HIDDEN_DIM,
            "temporal_init_seed": TEMPORAL_INIT_SEED,
            "temporal_parameter_count": TEMPORAL_PARAMETER_COUNT,
            "combined_parameter_count": COMBINED_PARAMETER_COUNT,
            "final_latent_control_parameter_count": source_audit["representations"][
                SOURCE_CONTROL
            ]["parameter_count"],
            "history_parameter_sharing": True,
            "initial_prediction_max_abs_drift": initial_drift,
            "critic_adam_count_preserved": optimizer_count(state.critic_opt),
            "temporal_adam_moments_initialized_zero": True,
        },
        "fit_budget": {
            "optimizer_steps": FIT_STEPS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "learning_rate": CRITIC_LR,
            "gradient_clip_global_norm": 1.0,
            "selection_source": "validation-only",
            "final_test_evaluations": 1,
        },
        "validation_selection": fit_report,
        "selected_final_test": final_test,
        "selection_thresholds": {
            "minimum_rank_correlation": MIN_RANK_CORRELATION,
            "maximum_nrmse": MAX_NRMSE,
            "required_groups": list(METRIC_GROUPS),
            "required_boundaries": list(REQUIRED_BOUNDARIES),
            "required_gate_count": len(METRIC_GROUPS) * len(REQUIRED_BOUNDARIES),
            "all_required_gates_must_pass": True,
        },
        "actor_optimizer_updates": 0,
        "critic_optimizer_updates": FIT_STEPS,
        "temporal_optimizer_updates": FIT_STEPS,
        "environment_steps_retained": 0,
        "policy_retained": False,
        "critic_retained": False,
        "retained_policy": None,
        "retained_critic": None,
        "interpretation_boundary": (
            "This offline paired audit tests scalar return prediction by one "
            "capacity-controlled temporal residual. It does not create an "
            "independent critic pair, test critic state gradients, update the "
            "actor, run AHAC, or establish locomotion improvement."
        ),
    }
    return arrays_out, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-completion", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("E002 temporal critic seed must equal zero")
    configure_jax()
    runtime = _validate_runtime()
    repository = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    dataset = args.dataset.resolve()
    source_audit = args.source_audit.resolve()
    source_completion = args.source_completion.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=args.code_commit,
    )
    _load_source_bundle(dataset, source_audit, source_completion)
    preflight.update(
        protocol="g1-e002-temporal-residual-critic-preflight-v1",
        runtime=runtime,
        source_dataset_path=str(dataset),
        source_dataset_sha256=SOURCE_DATASET_SHA256,
        source_audit_path=str(source_audit),
        source_audit_sha256=SOURCE_AUDIT_SHA256,
        source_completion_path=str(source_completion),
        source_completion_sha256=SOURCE_COMPLETION_SHA256,
        temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
        temporal_init_seed=TEMPORAL_INIT_SEED,
        temporal_parameter_count=TEMPORAL_PARAMETER_COUNT,
        combined_parameter_count=COMBINED_PARAMETER_COUNT,
        actor_optimizer_updates=0,
        critic_optimizer_updates=FIT_STEPS,
        environment_steps_retained=0,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    arrays, result = collect_temporal_audit(
        checkpoint=checkpoint,
        dataset_path=dataset,
        audit_path=source_audit,
        completion_path=source_completion,
    )
    predictions_path = output_root / "temporal_critic_predictions.npz"
    result_path = output_root / "temporal_critic_audit.json"
    _atomic_npz(predictions_path, arrays)
    result["predictions_sha256"] = sha256_file(predictions_path)
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": "g1-e002-temporal-residual-critic-completion-v1",
        "valid": True,
        "classification": result["classification"],
        "actor_optimizer_updates": 0,
        "critic_optimizer_updates": FIT_STEPS,
        "temporal_optimizer_updates": FIT_STEPS,
        "environment_steps_retained": 0,
        "policy_retained": False,
        "critic_retained": False,
        "retained_policy": None,
        "retained_critic": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "temporal_critic_predictions.npz": sha256_file(predictions_path),
            "temporal_critic_audit.json": sha256_file(result_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(output_root)


if __name__ == "__main__":
    main()
