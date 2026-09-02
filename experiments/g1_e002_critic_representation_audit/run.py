"""Compare current-only and policy-context-conditioned E002 critics."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.g1_e002_critic_calibration_audit.run import (
    BOUNDARY_INDEX,
    EFFECTIVE_NUM_ENVS,
    GAMMA,
    START_STEP,
    _atomic_npz,
    _distribution,
    _make_environment,
    _validate_runtime,
    first_terminal_returns,
)
from experiments.g1_e002_policy_memory_alias_audit.run import (
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    replace_with_repeated_current_history,
)
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
)
from src.core.data_structures import Normalizer
from src.core.networks import Actor, Critic
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.compare_g1_future_preview_critic import (
    migrate_critic_input,
    validate_initial_equivalence,
)
from tools.evaluate_g1_flax_phase_grid import (
    load_checkpoint_environment_contract,
)
from tools.evaluate_g1_terminal_value_calibration import calibration_metrics
from tools.evaluate_g1_tracking import _load_policy
from tools.run_g1_dual_scale_root_position import (
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_HPARAMS_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically

MIN_RANK_CORRELATION = 0.8
MAX_NRMSE = 0.25
FIT_ROW_STRIDE = 4
FIT_STEPS = 1_640
EVALUATION_INTERVAL = 20
CRITIC_LR = 5e-4
SPLIT_SEED = 20_260_902
ACTION_DIM = 29
ACTOR_LATENT_DIM = 128
ACTOR_EARLY_LATENT_DIM = 512
E005_ALIAS_TRACE_SHA256 = (
    "0064b6bdcf59c1852b9102a4be5f222798d7996fe7e03f50787bdac697551ec0"
)
REPRESENTATION_NAMES = (
    "current_only",
    "current_plus_action",
    "current_plus_actor_latent",
)
EARLY_LATENT_REPRESENTATION = "current_plus_actor_early_latent"
RAW_HISTORY_REPRESENTATION = "current_plus_actor_history"
METRIC_GROUPS = ("combined", "carried", "repeated_current")
METRIC_BOUNDARIES = ("aggregate", "h24")


class ActorHistoryLatent(nn.Module):
    """Expose the final hidden activation of the frozen base actor MLP."""

    hidden: Sequence[int] = (512, 256, 128)
    output_index: int = -1

    @nn.compact
    def __call__(self, observations):
        value = observations
        activations = []
        for width in self.hidden:
            value = nn.Dense(width)(value)
            value = nn.LayerNorm()(value)
            value = nn.elu(value)
            activations.append(value)
        index = self.output_index
        if index < 0:
            index += len(activations)
        if index < 0 or index >= len(activations):
            raise ValueError("actor history latent output index is invalid")
        return activations[index]


def validate_representation_options(
    *,
    include_h0: bool,
    include_early_latent: bool,
    include_raw_history: bool,
) -> None:
    """Fail closed on experimental option combinations."""

    if (include_early_latent or include_raw_history) and not include_h0:
        raise ValueError("actor history representation requires H0 gates")
    if include_early_latent and include_raw_history:
        raise ValueError("early latent and raw history are mutually exclusive")


def reconstruct_base_actor_action(params: Mapping[str, Any], latent, *, squash: bool):
    """Apply the existing base actor output head to an extracted latent."""

    try:
        modules = params["params"]
        dense_names = sorted(
            (name for name in modules if name.startswith("Dense_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        output = modules[dense_names[-1]]
        kernel = output["kernel"]
        bias = output["bias"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("base actor output head is malformed") from error
    if latent.shape[-1] != kernel.shape[0]:
        raise ValueError("base actor latent does not match output head")
    logits = jnp.asarray(latent) @ kernel + bias
    return jnp.tanh(logits) if squash else logits


def unwrap_base_actor(
    actor_params: Any,
) -> tuple[FrozenPreviewResidualParams, int]:
    """Return E002's preview-residual base and controller-residual depth."""

    current = actor_params
    depth = 0
    while isinstance(current, FrozenControllerResidualParams):
        current = current.parent
        depth += 1
    if not isinstance(current, FrozenPreviewResidualParams):
        raise TypeError("actor does not have a preview-residual base")
    return current, depth


def e005_scan_carrier(
    source: Mapping[str, np.ndarray],
    *,
    horizon: int,
    population: int,
    action_dim: int,
):
    """Build an unused deterministic scan carrier from E005's retained shape."""

    if horizon < 1 or population < 1 or action_dim < 1:
        raise ValueError("E005 scan carrier dimensions are invalid")
    expected = (horizon, population)
    try:
        shapes = {
            name: np.asarray(source[name]).shape
            for name in ("carried_reward", "repeated_current_reward")
        }
    except KeyError as error:
        raise ValueError("E005 trace is missing a rollout shape") from error
    if any(shape != expected for shape in shapes.values()):
        raise ValueError("E005 retained rollout shape drifted")
    return jnp.zeros((horizon, population, action_dim), dtype=jnp.float64)


def environment_group_splits(population: int, *, seed: int) -> dict[str, np.ndarray]:
    """Split paired environment identities once for fit/validation/final test."""

    if population < 5 or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("environment split arguments are invalid")
    permutation = np.random.default_rng(seed).permutation(population)
    fit_count = math.floor(0.60 * population)
    validation_count = math.floor(0.20 * population)
    return {
        "fit": permutation[:fit_count],
        "validation": permutation[fit_count : fit_count + validation_count],
        "test": permutation[fit_count + validation_count :],
    }


def fit_row_mask(
    environment_id: np.ndarray,
    time_index: np.ndarray,
    alive: np.ndarray,
    *,
    fit_environment_ids: np.ndarray,
    stride: int = FIT_ROW_STRIDE,
) -> np.ndarray:
    """Select alive fit-group rows on the registered temporal stride."""

    environments = np.asarray(environment_id)
    times = np.asarray(time_index)
    active = np.asarray(alive, dtype=bool)
    fit_ids = np.asarray(fit_environment_ids)
    if (
        environments.ndim != 1
        or times.shape != environments.shape
        or active.shape != environments.shape
        or fit_ids.ndim != 1
        or fit_ids.size < 1
        or stride < 1
    ):
        raise ValueError("fit row mask inputs are invalid")
    return active & np.isin(environments, fit_ids) & (times % stride == 0)


def normalize_extra_features(
    values: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Standardize action/latent columns using fit rows only."""

    features = np.asarray(values, dtype=np.float64)
    selected = np.asarray(fit_mask, dtype=bool)
    if (
        features.ndim != 2
        or selected.shape != (features.shape[0],)
        or np.sum(selected) < 2
        or not np.isfinite(features).all()
    ):
        raise ValueError("extra feature normalization inputs are invalid")
    mean = np.mean(features[selected], axis=0)
    std = np.std(features[selected], axis=0)
    safe_std = np.maximum(std, 1e-6)
    normalized = (features - mean) / safe_std
    if not np.isfinite(normalized).all():
        raise ValueError("extra feature normalization is nonfinite")
    return normalized.astype(np.float32), {
        "mean": mean,
        "std": safe_std,
        "minimum_raw_std": np.asarray(float(np.min(std))),
    }


def summarize_representation_metrics(
    predictions: np.ndarray,
    realized_returns: np.ndarray,
    arm: np.ndarray,
    time_index: np.ndarray,
    selected: np.ndarray,
    *,
    boundary_indices: Mapping[str, int] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate aggregate and registered boundary calibration by arm."""

    values = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(realized_returns, dtype=np.float64)
    arms = np.asarray(arm)
    times = np.asarray(time_index)
    base_mask = np.asarray(selected, dtype=bool)
    if (
        values.ndim != 1
        or targets.shape != values.shape
        or arms.shape != values.shape
        or times.shape != values.shape
        or base_mask.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
        or not set(np.unique(arms)).issubset({0, 1})
    ):
        raise ValueError("representation prediction arrays are invalid")
    group_masks = {
        "combined": base_mask,
        "carried": base_mask & (arms == 0),
        "repeated_current": base_mask & (arms == 1),
    }
    if boundary_indices is None:
        boundary_indices = {"h24": BOUNDARY_INDEX}
    else:
        boundary_indices = dict(boundary_indices)
    if (
        not boundary_indices
        or "aggregate" in boundary_indices
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for name, index in boundary_indices.items()
        )
    ):
        raise ValueError("representation boundary indices are invalid")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for name in METRIC_GROUPS:
        aggregate = group_masks[name]
        boundary_masks = {
            boundary: aggregate & (times == index)
            for boundary, index in boundary_indices.items()
        }
        if np.sum(aggregate) < 2 or any(
            np.sum(mask) < 2 for mask in boundary_masks.values()
        ):
            raise ValueError("representation metric group is too small")
        result[name] = {
            "aggregate": calibration_metrics(values[aggregate], targets[aggregate]),
            **{
                boundary: calibration_metrics(values[mask], targets[mask])
                for boundary, mask in boundary_masks.items()
            },
        }
    return result


def _optimizer_count(opt_state: Any) -> int:
    try:
        return int(np.asarray(opt_state[1][0].count))
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("critic optimizer count is unavailable") from error


def fit_critic_with_validation(
    critic,
    params: Any,
    opt_state: Any,
    observations: np.ndarray,
    realized_returns: np.ndarray,
    arm: np.ndarray,
    time_index: np.ndarray,
    *,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    steps: int,
    evaluation_interval: int,
    optimizer,
    boundary_indices: Mapping[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit one arm and select snapshots exclusively on validation rows."""

    features = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(realized_returns, dtype=np.float32)
    arms = np.asarray(arm)
    times = np.asarray(time_index)
    fitting = np.asarray(fit_mask, dtype=bool)
    validating = np.asarray(validation_mask, dtype=bool)
    if (
        features.ndim != 2
        or targets.shape != (features.shape[0],)
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
        or not np.isfinite(features).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("critic fit contract is invalid")
    fit_observations = jnp.asarray(features[fitting])
    fit_returns = jnp.asarray(targets[fitting])
    validation_observations = jnp.asarray(features[validating])
    validation_targets = targets[validating]
    validation_arms = arms[validating]
    validation_times = times[validating]
    if boundary_indices is None:
        boundary_indices = {"h24": BOUNDARY_INDEX}
    else:
        boundary_indices = dict(boundary_indices)
    required_boundaries = ("aggregate", *boundary_indices)

    @jax.jit
    def update(current_params, current_opt_state):
        loss, gradients = jax.value_and_grad(
            lambda candidate: jnp.mean(
                jnp.square(
                    critic.apply(candidate, fit_observations).reshape(-1) - fit_returns
                )
            )
        )(current_params)
        updates, next_opt_state = optimizer.update(gradients, current_opt_state)
        return (
            optax.apply_updates(current_params, updates),
            next_opt_state,
            loss,
        )

    @jax.jit
    def predict(current_params, values):
        return critic.apply(current_params, values).reshape(-1)

    def candidate(step: int, loss: float | None) -> dict[str, Any]:
        predictions = np.asarray(
            predict(params, validation_observations), dtype=np.float64
        )
        metrics = summarize_representation_metrics(
            predictions,
            validation_targets,
            validation_arms,
            validation_times,
            np.ones(validation_targets.shape, dtype=bool),
            boundary_indices=boundary_indices,
        )
        return {"step": step, "fit_loss": loss, "metrics": metrics}

    initial_count = _optimizer_count(opt_state)
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
    final_count = _optimizer_count(opt_state)
    if final_count != initial_count + steps or not math.isfinite(float(last_loss)):
        raise ValueError("critic fit did not execute its exact finite budget")
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


def _metric_values(
    metrics: Mapping[str, Any],
    *,
    required_boundaries: Sequence[str] = METRIC_BOUNDARIES,
) -> tuple[list[float], list[float]]:
    boundaries = tuple(required_boundaries)
    if (
        not boundaries
        or len(set(boundaries)) != len(boundaries)
        or boundaries[0] != "aggregate"
    ):
        raise ValueError("representation metric boundaries are invalid")
    if set(metrics) != set(METRIC_GROUPS):
        raise ValueError("representation metrics are incomplete")
    ranks: list[float] = []
    errors: list[float] = []
    try:
        for group in METRIC_GROUPS:
            group_metrics = metrics[group]
            if set(group_metrics) != set(boundaries):
                raise ValueError("representation metrics are incomplete")
            for boundary in boundaries:
                record = group_metrics[boundary]
                rank = float(record["rank_correlation"])
                nrmse = float(record["nrmse"])
                if not math.isfinite(rank) or not math.isfinite(nrmse):
                    raise ValueError("representation metrics are nonfinite")
                ranks.append(rank)
                errors.append(nrmse)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("representation metrics are invalid") from error
    return ranks, errors


def representation_adequate(
    metrics: Mapping[str, Any],
    *,
    required_boundaries: Sequence[str] = METRIC_BOUNDARIES,
) -> bool:
    """Require every registered population/boundary held-out gate to pass."""

    try:
        ranks, errors = _metric_values(
            metrics,
            required_boundaries=required_boundaries,
        )
    except ValueError as error:
        raise ValueError("representation metrics are invalid") from error
    return bool(min(ranks) >= MIN_RANK_CORRELATION and max(errors) <= MAX_NRMSE)


def validation_candidate_key(
    metrics: Mapping[str, Any],
    *,
    step: int,
    required_boundaries: Sequence[str] = METRIC_BOUNDARIES,
) -> tuple[bool, float, float, int]:
    """Select only on validation: pass, worst NRMSE, worst rank, earlier."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("validation candidate step is invalid")
    ranks, errors = _metric_values(
        metrics,
        required_boundaries=required_boundaries,
    )
    return (
        representation_adequate(
            metrics,
            required_boundaries=required_boundaries,
        ),
        -max(errors),
        min(ranks),
        -step,
    )


def _quality(
    metrics: Mapping[str, Any],
    *,
    required_boundaries: Sequence[str] = METRIC_BOUNDARIES,
) -> tuple[float, float]:
    ranks, errors = _metric_values(
        metrics,
        required_boundaries=required_boundaries,
    )
    return (-max(errors), min(ranks))


def classify_representations(
    final_test: Mapping[str, Mapping[str, Any]],
    *,
    required_boundaries: Sequence[str] = METRIC_BOUNDARIES,
    current_only_first: bool = False,
) -> str:
    """Classify the frozen final-test metrics without selecting on them."""

    if set(final_test) != set(REPRESENTATION_NAMES):
        raise ValueError("representation test arms are incomplete")
    adequate = {
        name: representation_adequate(
            final_test[name],
            required_boundaries=required_boundaries,
        )
        for name in REPRESENTATION_NAMES
    }
    if current_only_first and adequate["current_only"]:
        return "current-only-refit-adequate"
    if adequate["current_plus_actor_latent"]:
        return "actor-latent-representation-adequate"
    if adequate["current_plus_action"]:
        return "policy-action-representation-adequate"
    if adequate["current_only"]:
        return "current-only-refit-adequate"
    baseline_quality = _quality(
        final_test["current_only"],
        required_boundaries=required_boundaries,
    )
    if (
        max(
            _quality(
                final_test["current_plus_action"],
                required_boundaries=required_boundaries,
            ),
            _quality(
                final_test["current_plus_actor_latent"],
                required_boundaries=required_boundaries,
            ),
        )
        > baseline_quality
    ):
        return "augmented-representation-improves-but-insufficient"
    return "tested-representations-insufficient"


def classify_early_latent_representations(
    final_test: Mapping[str, Mapping[str, Any]],
    *,
    required_boundaries: Sequence[str],
) -> str:
    """Classify the richer frozen-history arm with simplest-first precedence."""

    names = (*REPRESENTATION_NAMES, EARLY_LATENT_REPRESENTATION)
    if set(final_test) != set(names):
        raise ValueError("early-latent representation test arms are incomplete")
    adequate = {
        name: representation_adequate(
            final_test[name],
            required_boundaries=required_boundaries,
        )
        for name in names
    }
    priority = (
        ("current_only", "current-only-refit-adequate"),
        (
            EARLY_LATENT_REPRESENTATION,
            "actor-early-latent-representation-adequate",
        ),
        (
            "current_plus_actor_latent",
            "actor-latent-representation-adequate",
        ),
        ("current_plus_action", "policy-action-representation-adequate"),
    )
    for name, label in priority:
        if adequate[name]:
            return label
    baseline_quality = _quality(
        final_test["current_only"],
        required_boundaries=required_boundaries,
    )
    if (
        max(
            _quality(
                final_test[name],
                required_boundaries=required_boundaries,
            )
            for name in names
            if name != "current_only"
        )
        > baseline_quality
    ):
        return "augmented-representation-improves-but-insufficient"
    return "tested-representations-insufficient"


def classify_raw_history_representations(
    final_test: Mapping[str, Mapping[str, Any]],
    *,
    required_boundaries: Sequence[str],
) -> str:
    """Classify the full frozen actor-input arm with smallest-first precedence."""

    names = (*REPRESENTATION_NAMES, RAW_HISTORY_REPRESENTATION)
    if set(final_test) != set(names):
        raise ValueError("raw-history representation test arms are incomplete")
    adequate = {
        name: representation_adequate(
            final_test[name],
            required_boundaries=required_boundaries,
        )
        for name in names
    }
    priority = (
        ("current_only", "current-only-refit-adequate"),
        ("current_plus_action", "policy-action-representation-adequate"),
        (
            "current_plus_actor_latent",
            "actor-latent-representation-adequate",
        ),
        (RAW_HISTORY_REPRESENTATION, "actor-history-representation-adequate"),
    )
    for name, label in priority:
        if adequate[name]:
            return label
    baseline_quality = _quality(
        final_test["current_only"],
        required_boundaries=required_boundaries,
    )
    if (
        max(
            _quality(
                final_test[name],
                required_boundaries=required_boundaries,
            )
            for name in names
            if name != "current_only"
        )
        > baseline_quality
    ):
        return "augmented-representation-improves-but-insufficient"
    return "tested-representations-insufficient"


def _predict(critic, params: Any, observations: np.ndarray) -> np.ndarray:
    values = jnp.asarray(observations, dtype=jnp.float32)
    return np.asarray(
        jax.jit(lambda current: critic.apply(current, values).reshape(-1))(params),
        dtype=np.float64,
    )


def _parameter_count(params: Any) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(params)))


def _derive_rollout(
    trace: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if (
        not bool(np.all(trace["finite"]))
        or float(np.max(trace["xfrc_max"])) != 0.0
        or not all(
            np.isfinite(np.asarray(trace[name])).all()
            for name in (
                "online_value",
                "target_value",
                "reward",
                "action",
                "normalized_critic_obs",
                "actor_latent",
            )
        )
    ):
        raise ValueError("representation rollout is nonfinite or assisted")
    done = np.asarray(trace["done"], dtype=bool)
    realized, alive = first_terminal_returns(
        np.asarray(trace["reward"]), done, gamma=GAMMA
    )
    if not np.all(np.any(done, axis=0)):
        raise ValueError("representation rollout did not reach first done")
    first_done = np.argmax(done, axis=0) + 1
    columns = np.arange(done.shape[1])
    first_terminal = np.asarray(trace["terminal"], dtype=bool)[first_done - 1, columns]
    derived = {
        "alive": alive,
        "realized_return": realized,
        "first_done": first_done,
        "first_done_terminal": first_terminal,
    }
    summary = {
        "alive_rows": int(np.sum(alive)),
        "survival": _distribution(first_done),
        "return_from_start": _distribution(realized[0]),
        "natural_terminal_count": int(np.sum(first_terminal)),
        "truncation_count": int(np.sum(~first_terminal)),
        "post_first_done_rewards_masked": True,
    }
    return derived, summary


def _e005_parity(
    traces: Mapping[str, Mapping[str, np.ndarray]],
    derived: Mapping[str, Mapping[str, np.ndarray]],
    source: Mapping[str, np.ndarray],
) -> dict[str, object]:
    core_fields = (
        "online_value",
        "target_value",
        "reward",
        "done",
        "terminal",
        "phase_before",
        "xfrc_max",
        "finite",
    )
    derived_fields = (
        "alive",
        "realized_return",
        "first_done",
        "first_done_terminal",
    )
    parity: dict[str, object] = {}
    for arm in ("carried", "repeated_current"):
        for field in core_fields:
            parity[f"{arm}_{field}_exact"] = bool(
                np.array_equal(
                    np.asarray(traces[arm][field]),
                    np.asarray(source[f"{arm}_{field}"]),
                )
            )
        for field in derived_fields:
            parity[f"{arm}_{field}_exact"] = bool(
                np.array_equal(
                    np.asarray(derived[arm][field]),
                    np.asarray(source[f"{arm}_{field}"]),
                )
            )
        parity[f"{arm}_start_action_exact"] = bool(
            np.array_equal(
                np.asarray(traces[arm]["action"])[0],
                np.asarray(source[f"initial_{arm}_action"]),
            )
        )
    parity["all_exact"] = all(bool(value) for value in parity.values())
    return parity


def _phase_split_summary(
    start_phase: np.ndarray, splits: Mapping[str, np.ndarray]
) -> dict[str, dict[str, int]]:
    phases = np.asarray(start_phase)
    result = {}
    for name, ids in splits.items():
        selected = phases[np.asarray(ids)]
        result[name] = {
            "environment_count": int(selected.size),
            "minimum_phase": int(np.min(selected)),
            "maximum_phase": int(np.max(selected)),
            "unique_phase_count": int(np.unique(selected).size),
        }
    return result


def collect_representation_audit(
    *,
    checkpoint: Path,
    reference: Path,
    alias_trace: Path,
    include_h0: bool = False,
    include_early_latent: bool = False,
    include_raw_history: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Capture paired policy contexts and fit matched critic arms."""

    validate_representation_options(
        include_h0=include_h0,
        include_early_latent=include_early_latent,
        include_raw_history=include_raw_history,
    )
    representation_names = (
        (*REPRESENTATION_NAMES, EARLY_LATENT_REPRESENTATION)
        if include_early_latent
        else (*REPRESENTATION_NAMES, RAW_HISTORY_REPRESENTATION)
        if include_raw_history
        else REPRESENTATION_NAMES
    )
    boundary_indices = (
        {"h0": 0, "h24": BOUNDARY_INDEX} if include_h0 else {"h24": BOUNDARY_INDEX}
    )
    required_boundaries = ("aggregate", *boundary_indices)
    hparams = json.loads(
        checkpoint.with_name("hparams.json").read_text(encoding="utf-8")
    )
    contract = load_checkpoint_environment_contract(checkpoint)
    if (
        int(hparams["effective_num_envs"]) != EFFECTIVE_NUM_ENVS
        or int(hparams["unroll_length"]) != BOUNDARY_INDEX
        or int(hparams["actor_history_len"]) != ACTOR_HISTORY_LEN
        or hparams.get("ahac") is not False
        or float(hparams.get("actor_bootstrap_scale", -1.0)) != 0.0
        or contract["env_variant"] != hparams["env_variant"]
        or float(hparams["zero_difficulty_frac"]) != 1.0
        or bool(hparams["actor_observation_noise"])
        or bool(hparams["terrain"])
        or bool(hparams["domain_randomization"])
        or bool(hparams["torso_wrench_assistance"])
        or bool(hparams["actor_learned_torso_wrench"])
    ):
        raise ValueError("retained E002 boundary does not match representation")
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != START_STEP
        or np.asarray(state.env_state.obs).shape != (EFFECTIVE_NUM_ENVS, ACTOR_OBS_DIM)
        or np.asarray(state.env_state.info["actor_obs_history"]).shape
        != (EFFECTIVE_NUM_ENVS, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)
    ):
        raise ValueError("retained E002 state does not match representation")

    environment = _make_environment(hparams, reference)
    if (
        int(environment.actor_frame_obs_dim) != ACTOR_FRAME_DIM
        or int(environment.actor_obs_dim) != ACTOR_OBS_DIM
        or int(environment.critic_obs_dim) != CRITIC_OBS_DIM
        or int(environment.action_dim) != ACTION_DIM
    ):
        raise ValueError("representation observation dimensions drifted")
    horizon = int(environment.max_episode_length)
    actor, actor_params, actor_normalizer_state = _load_policy(
        environment, checkpoint, 0
    )
    base_params, controller_residual_depth = unwrap_base_actor(actor_params)
    modules = base_params.parent["params"]
    dense_names = sorted(
        (name for name in modules if name.startswith("Dense_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    hidden = tuple(int(modules[name]["kernel"].shape[-1]) for name in dense_names[:-1])
    if (
        hidden != (512, 256, ACTOR_LATENT_DIM)
        or len(dense_names) != 4
        or controller_residual_depth != 1
    ):
        raise ValueError("E002 base actor latent contract drifted")
    base_squash = bool(
        getattr(
            environment,
            "squash_actor_mean",
            getattr(environment, "squash_actor_actions", True),
        )
    )
    base_actor = Actor(
        ACTION_DIM,
        hidden=hidden,
        squash=base_squash,
        layer_norm=True,
        zero_output=False,
    )
    latent_encoder = ActorHistoryLatent(hidden=hidden)
    early_latent_encoder = ActorHistoryLatent(hidden=hidden, output_index=0)
    actor_normalizer = Normalizer(ACTOR_FRAME_DIM)
    critic_normalizer = Normalizer(CRITIC_OBS_DIM)
    critic = Critic()
    repeated_state = replace_with_repeated_current_history(state.env_state)

    with np.load(alias_trace, allow_pickle=False) as archive:
        source = {name: np.asarray(archive[name]) for name in archive.files}
    scan_noise = e005_scan_carrier(
        source,
        horizon=horizon,
        population=EFFECTIVE_NUM_ENVS,
        action_dim=ACTION_DIM,
    )

    def make_rollout():
        def rollout(initial_state, noise):
            def step(env_state, epsilon):
                del epsilon
                critic_obs = jax.vmap(environment._get_critic_obs)(
                    env_state.data, env_state.info
                )
                normalized_critic_obs = critic_normalizer.normalize(
                    state.critic_normalizer, critic_obs
                ).astype(jnp.float32)
                online_value = critic.apply(
                    state.critic_params, normalized_critic_obs
                ).squeeze(-1)
                target_value = critic.apply(
                    state.target_critic_params, normalized_critic_obs
                ).squeeze(-1)
                rng_pairs = jax.vmap(lambda key: jax.random.split(key, 2))(
                    env_state.info["rng"]
                )
                obs_rng = rng_pairs[:, 0]
                env_rng = rng_pairs[:, 1]
                env_state = env_state.replace(info={**env_state.info, "rng": env_rng})
                actor_obs = jax.vmap(environment._apply_obs_noise)(
                    env_state.obs, obs_rng
                )
                normalized_actor_obs = environment.normalize_actor_obs(
                    actor_normalizer,
                    actor_normalizer_state,
                    actor_obs,
                ).astype(jnp.float32)
                action = jax.vmap(
                    lambda observation: actor.apply(actor_params, observation)
                )(normalized_actor_obs).astype(jnp.float64)
                if environment.clip_sampled_actor_actions:
                    action = jnp.clip(action, -1.0, 1.0)
                actor_latent = latent_encoder.apply(
                    base_params.parent, normalized_actor_obs
                )
                actor_early_latent = (
                    early_latent_encoder.apply(
                        base_params.parent,
                        normalized_actor_obs,
                    )
                    if include_early_latent
                    else None
                )
                next_state = jax.vmap(environment.step)(env_state, action)
                finite_values = (
                    online_value,
                    target_value,
                    next_state.reward,
                    action,
                    actor_latent,
                    normalized_critic_obs,
                    next_state.data.qpos,
                    next_state.data.qvel,
                )
                if actor_early_latent is not None:
                    finite_values = (*finite_values, actor_early_latent)
                if include_raw_history:
                    finite_values = (*finite_values, normalized_actor_obs)
                output = {
                    "online_value": online_value,
                    "target_value": target_value,
                    "reward": next_state.reward,
                    "done": next_state.done,
                    "terminal": next_state.info["terminal"],
                    "phase_before": env_state.info["phase"],
                    "xfrc_max": jnp.max(jnp.abs(env_state.data.xfrc_applied)),
                    "finite": jnp.all(
                        jnp.stack(
                            [jnp.all(jnp.isfinite(value)) for value in finite_values]
                        )
                    ),
                    "action": action,
                    "normalized_critic_obs": normalized_critic_obs,
                    "actor_latent": actor_latent,
                }
                if actor_early_latent is not None:
                    output["actor_early_latent"] = actor_early_latent
                if include_raw_history:
                    output["normalized_actor_history"] = normalized_actor_obs
                return next_state, output

            return jax.lax.scan(step, initial_state, noise)

        return rollout

    profile = get_solver_profile(str(hparams["solver_profile"]))
    with solver_context(profile):
        compiled_carried = jax.jit(make_rollout())
        compiled_repeated = jax.jit(make_rollout())
        _, carried_device = compiled_carried(state.env_state, scan_noise)
        _, repeated_device = compiled_repeated(repeated_state, scan_noise)
    traces = {
        "carried": jax.tree.map(np.asarray, carried_device),
        "repeated_current": jax.tree.map(np.asarray, repeated_device),
    }
    derived: dict[str, dict[str, np.ndarray]] = {}
    rollout_summaries: dict[str, dict[str, object]] = {}
    for name, trace in traces.items():
        derived[name], rollout_summaries[name] = _derive_rollout(trace)
    parity = _e005_parity(traces, derived, source)
    if not bool(parity["all_exact"]):
        raise ValueError("feature rollouts do not exactly reproduce E005")

    normalized_start = environment.normalize_actor_obs(
        actor_normalizer,
        actor_normalizer_state,
        state.env_state.obs,
    ).astype(jnp.float32)
    start_latent = latent_encoder.apply(base_params.parent, normalized_start)
    reconstructed_base_action = reconstruct_base_actor_action(
        base_params.parent, start_latent, squash=base_squash
    )
    direct_base_action = base_actor.apply(base_params.parent, normalized_start)
    base_action_max_abs_delta = float(
        np.max(
            np.abs(
                np.asarray(reconstructed_base_action) - np.asarray(direct_base_action)
            )
        )
    )
    start_latent_max_abs_delta = float(
        np.max(
            np.abs(
                np.asarray(start_latent)
                - np.asarray(traces["carried"]["actor_latent"])[0]
            )
        )
    )
    start_early_latent_max_abs_delta = None
    if include_early_latent:
        start_early_latent = early_latent_encoder.apply(
            base_params.parent,
            normalized_start,
        )
        start_early_latent_max_abs_delta = float(
            np.max(
                np.abs(
                    np.asarray(start_early_latent)
                    - np.asarray(traces["carried"]["actor_early_latent"])[0]
                )
            )
        )
    start_raw_history_max_abs_delta = None
    if include_raw_history:
        start_raw_history_max_abs_delta = float(
            np.max(
                np.abs(
                    np.asarray(normalized_start)
                    - np.asarray(traces["carried"]["normalized_actor_history"])[0]
                )
            )
        )
    if (
        base_action_max_abs_delta > 1e-6
        or start_latent_max_abs_delta > 1e-6
        or (
            start_early_latent_max_abs_delta is not None
            and start_early_latent_max_abs_delta > 1e-6
        )
        or (
            start_raw_history_max_abs_delta is not None
            and start_raw_history_max_abs_delta > 1e-6
        )
    ):
        raise ValueError("actor representation does not reproduce the frozen base")

    time_grid = np.broadcast_to(
        np.arange(horizon, dtype=np.int32)[:, None],
        (horizon, EFFECTIVE_NUM_ENVS),
    )
    environment_grid = np.broadcast_to(
        np.arange(EFFECTIVE_NUM_ENVS, dtype=np.int32)[None, :],
        (horizon, EFFECTIVE_NUM_ENVS),
    )
    dataset_parts: dict[str, list[np.ndarray]] = {
        "current": [],
        "action": [],
        "latent": [],
        "return": [],
        "arm": [],
        "environment_id": [],
        "time_index": [],
        "phase": [],
    }
    if include_early_latent:
        dataset_parts["early_latent"] = []
    if include_raw_history:
        dataset_parts["raw_history"] = []
    for arm_id, name in enumerate(("carried", "repeated_current")):
        alive = np.asarray(derived[name]["alive"], dtype=bool)
        dataset_parts["current"].append(
            np.asarray(traces[name]["normalized_critic_obs"])[alive]
        )
        dataset_parts["action"].append(
            np.asarray(traces[name]["action"], dtype=np.float32)[alive]
        )
        dataset_parts["latent"].append(
            np.asarray(traces[name]["actor_latent"], dtype=np.float32)[alive]
        )
        if include_early_latent:
            dataset_parts["early_latent"].append(
                np.asarray(
                    traces[name]["actor_early_latent"],
                    dtype=np.float32,
                )[alive]
            )
        if include_raw_history:
            dataset_parts["raw_history"].append(
                np.asarray(
                    traces[name]["normalized_actor_history"],
                    dtype=np.float32,
                )[alive]
            )
        dataset_parts["return"].append(
            np.asarray(derived[name]["realized_return"])[alive]
        )
        count = int(np.sum(alive))
        dataset_parts["arm"].append(np.full(count, arm_id, dtype=np.int8))
        dataset_parts["environment_id"].append(environment_grid[alive])
        dataset_parts["time_index"].append(time_grid[alive])
        dataset_parts["phase"].append(np.asarray(traces[name]["phase_before"])[alive])
    dataset = {
        key: np.concatenate(parts, axis=0) for key, parts in dataset_parts.items()
    }
    row_count = int(dataset["return"].size)
    if (
        dataset["current"].shape != (row_count, CRITIC_OBS_DIM)
        or dataset["action"].shape != (row_count, ACTION_DIM)
        or dataset["latent"].shape != (row_count, ACTOR_LATENT_DIM)
        or (
            include_early_latent
            and dataset["early_latent"].shape != (row_count, ACTOR_EARLY_LATENT_DIM)
        )
        or (
            include_raw_history
            and dataset["raw_history"].shape != (row_count, ACTOR_OBS_DIM)
        )
        or not all(np.isfinite(value).all() for value in dataset.values())
    ):
        raise ValueError("representation dataset is malformed")

    splits = environment_group_splits(EFFECTIVE_NUM_ENVS, seed=SPLIT_SEED)
    all_alive = np.ones(row_count, dtype=bool)
    fit_mask = fit_row_mask(
        dataset["environment_id"],
        dataset["time_index"],
        all_alive,
        fit_environment_ids=splits["fit"],
    )
    validation_mask = np.isin(dataset["environment_id"], splits["validation"])
    test_mask = np.isin(dataset["environment_id"], splits["test"])
    fit_group_mask = np.isin(dataset["environment_id"], splits["fit"])
    if (
        np.any(fit_group_mask & validation_mask)
        or np.any(fit_group_mask & test_mask)
        or np.any(validation_mask & test_mask)
        or not np.all(fit_group_mask | validation_mask | test_mask)
        or np.sum(fit_mask & (dataset["time_index"] == BOUNDARY_INDEX)) < 2
        or np.sum(validation_mask & (dataset["time_index"] == BOUNDARY_INDEX)) < 4
        or np.sum(test_mask & (dataset["time_index"] == BOUNDARY_INDEX)) < 4
    ):
        raise ValueError("representation group split is invalid")
    normalized_action, action_statistics = normalize_extra_features(
        dataset["action"], fit_mask
    )
    normalized_latent, latent_statistics = normalize_extra_features(
        dataset["latent"], fit_mask
    )
    normalized_early_latent = None
    early_latent_statistics = None
    if include_early_latent:
        normalized_early_latent, early_latent_statistics = normalize_extra_features(
            dataset["early_latent"],
            fit_mask,
        )
    raw_history_statistics = None
    if include_raw_history:
        raw_history_std = np.std(
            np.asarray(dataset["raw_history"], dtype=np.float64)[fit_mask],
            axis=0,
        )
        raw_history_statistics = {
            "minimum_normalized_column_std": float(np.min(raw_history_std)),
            "maximum_normalized_column_std": float(np.max(raw_history_std)),
            "nonconstant_columns": int(np.sum(raw_history_std > 1e-6)),
        }
    representations = {
        "current_only": np.asarray(dataset["current"], dtype=np.float32),
        "current_plus_action": np.concatenate(
            (dataset["current"], normalized_action), axis=-1
        ).astype(np.float32),
        "current_plus_actor_latent": np.concatenate(
            (dataset["current"], normalized_latent), axis=-1
        ).astype(np.float32),
    }
    if normalized_early_latent is not None:
        representations[EARLY_LATENT_REPRESENTATION] = np.concatenate(
            (dataset["current"], normalized_early_latent),
            axis=-1,
        ).astype(np.float32)
    if include_raw_history:
        representations[RAW_HISTORY_REPRESENTATION] = np.concatenate(
            (dataset["current"], dataset["raw_history"]),
            axis=-1,
        ).astype(np.float32)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(CRITIC_LR))
    action_params, action_opt = migrate_critic_input(
        state.critic_params,
        state.critic_opt,
        extra_dim=ACTION_DIM,
        optimizer=optimizer,
    )
    latent_params, latent_opt = migrate_critic_input(
        state.critic_params,
        state.critic_opt,
        extra_dim=ACTOR_LATENT_DIM,
        optimizer=optimizer,
    )
    early_latent_params = None
    early_latent_opt = None
    if include_early_latent:
        early_latent_params, early_latent_opt = migrate_critic_input(
            state.critic_params,
            state.critic_opt,
            extra_dim=ACTOR_EARLY_LATENT_DIM,
            optimizer=optimizer,
        )
    raw_history_params = None
    raw_history_opt = None
    if include_raw_history:
        raw_history_params, raw_history_opt = migrate_critic_input(
            state.critic_params,
            state.critic_opt,
            extra_dim=ACTOR_OBS_DIM,
            optimizer=optimizer,
        )
    initial_baseline = _predict(
        critic, state.critic_params, representations["current_only"]
    )
    migration_drift = {
        "current_plus_action": validate_initial_equivalence(
            initial_baseline,
            _predict(
                critic,
                action_params,
                representations["current_plus_action"],
            ),
            tolerance=1e-6,
        ),
        "current_plus_actor_latent": validate_initial_equivalence(
            initial_baseline,
            _predict(
                critic,
                latent_params,
                representations["current_plus_actor_latent"],
            ),
            tolerance=1e-6,
        ),
    }
    if early_latent_params is not None:
        migration_drift[EARLY_LATENT_REPRESENTATION] = validate_initial_equivalence(
            initial_baseline,
            _predict(
                critic,
                early_latent_params,
                representations[EARLY_LATENT_REPRESENTATION],
            ),
            tolerance=1e-6,
        )
    if raw_history_params is not None:
        migration_drift[RAW_HISTORY_REPRESENTATION] = validate_initial_equivalence(
            initial_baseline,
            _predict(
                critic,
                raw_history_params,
                representations[RAW_HISTORY_REPRESENTATION],
            ),
            tolerance=1e-6,
        )
    starting = {
        "current_only": (state.critic_params, state.critic_opt),
        "current_plus_action": (action_params, action_opt),
        "current_plus_actor_latent": (latent_params, latent_opt),
    }
    if early_latent_params is not None and early_latent_opt is not None:
        starting[EARLY_LATENT_REPRESENTATION] = (
            early_latent_params,
            early_latent_opt,
        )
    if raw_history_params is not None and raw_history_opt is not None:
        starting[RAW_HISTORY_REPRESENTATION] = (
            raw_history_params,
            raw_history_opt,
        )
    selected_params: dict[str, Any] = {}
    fit_reports: dict[str, dict[str, Any]] = {}
    for name in representation_names:
        selected_params[name], fit_reports[name] = fit_critic_with_validation(
            critic,
            starting[name][0],
            starting[name][1],
            representations[name],
            dataset["return"],
            dataset["arm"],
            dataset["time_index"],
            fit_mask=fit_mask,
            validation_mask=validation_mask,
            steps=FIT_STEPS,
            evaluation_interval=EVALUATION_INTERVAL,
            optimizer=optimizer,
            boundary_indices=boundary_indices,
        )

    predictions = {
        name: _predict(critic, selected_params[name], representations[name])
        for name in representation_names
    }
    final_test = {
        name: summarize_representation_metrics(
            predictions[name],
            dataset["return"],
            dataset["arm"],
            dataset["time_index"],
            test_mask,
            boundary_indices=boundary_indices,
        )
        for name in representation_names
    }
    original_test = summarize_representation_metrics(
        initial_baseline,
        dataset["return"],
        dataset["arm"],
        dataset["time_index"],
        test_mask,
        boundary_indices=boundary_indices,
    )
    if include_early_latent:
        classification = classify_early_latent_representations(
            final_test,
            required_boundaries=required_boundaries,
        )
    elif include_raw_history:
        classification = classify_raw_history_representations(
            final_test,
            required_boundaries=required_boundaries,
        )
    else:
        classification = classify_representations(
            final_test,
            required_boundaries=required_boundaries,
            current_only_first=include_h0,
        )

    split_code = np.full(row_count, -1, dtype=np.int8)
    split_code[fit_group_mask] = 0
    split_code[validation_mask] = 1
    split_code[test_mask] = 2
    artifact_arrays = {
        "normalized_current_critic_obs": representations["current_only"],
        "executed_action": np.asarray(dataset["action"], dtype=np.float32),
        "actor_history_latent": np.asarray(dataset["latent"], dtype=np.float32),
        "realized_return": np.asarray(dataset["return"], dtype=np.float64),
        "arm": np.asarray(dataset["arm"], dtype=np.int8),
        "environment_id": np.asarray(dataset["environment_id"], dtype=np.int32),
        "time_index": np.asarray(dataset["time_index"], dtype=np.int32),
        "phase": np.asarray(dataset["phase"], dtype=np.int32),
        "split_code": split_code,
        "fit_row_selected": fit_mask,
        "action_fit_mean": np.asarray(action_statistics["mean"]),
        "action_fit_std": np.asarray(action_statistics["std"]),
        "latent_fit_mean": np.asarray(latent_statistics["mean"]),
        "latent_fit_std": np.asarray(latent_statistics["std"]),
        **{
            f"selected_{name}_prediction": np.asarray(value)
            for name, value in predictions.items()
        },
    }
    if include_early_latent:
        if early_latent_statistics is None:
            raise ValueError("early latent normalization statistics are unavailable")
        artifact_arrays.update(
            actor_history_early_latent=np.asarray(
                dataset["early_latent"],
                dtype=np.float32,
            ),
            early_latent_fit_mean=np.asarray(early_latent_statistics["mean"]),
            early_latent_fit_std=np.asarray(early_latent_statistics["std"]),
        )
    if include_raw_history:
        artifact_arrays["normalized_actor_history"] = np.asarray(
            dataset["raw_history"],
            dtype=np.float32,
        )
    result = {
        "protocol": (
            "g1-e002-critic-representation-early-latent-h0-audit-v1"
            if include_early_latent
            else "g1-e002-critic-representation-raw-history-h0-audit-v1"
            if include_raw_history
            else "g1-e002-critic-representation-h0-audit-v1"
            if include_h0
            else "g1-e002-critic-representation-audit-v1"
        ),
        "include_h0": include_h0,
        "include_early_latent": include_early_latent,
        "include_raw_history": include_raw_history,
        "valid": True,
        "classification": classification,
        "actor_optimizer_updates": 0,
        "critic_optimizer_updates_per_arm": FIT_STEPS,
        "critic_optimizer_updates_total": FIT_STEPS * len(representation_names),
        "environment_steps_retained": 0,
        "policy_retained": False,
        "critic_retained": False,
        "retained_policy": None,
        "retained_critic": None,
        "source_step": START_STEP,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_hparams_sha256": SOURCE_HPARAMS_SHA256,
        "source_actor_bootstrap_scale": float(hparams["actor_bootstrap_scale"]),
        "population_size": EFFECTIVE_NUM_ENVS,
        "rollout_horizon": horizon,
        "gamma": GAMMA,
        "e005_trace_sha256": E005_ALIAS_TRACE_SHA256,
        "e005_rollout_parity": parity,
        "rollouts": rollout_summaries,
        "actor_latent_contract": {
            "controller_residual_depth": controller_residual_depth,
            "base_hidden": list(hidden),
            "latent_dim": ACTOR_LATENT_DIM,
            "early_latent_dim": (
                ACTOR_EARLY_LATENT_DIM if include_early_latent else None
            ),
            "raw_history_dim": ACTOR_OBS_DIM if include_raw_history else None,
            "base_action_max_abs_reconstruction_delta": (base_action_max_abs_delta),
            "start_latent_max_abs_rollout_delta": (start_latent_max_abs_delta),
            "start_early_latent_max_abs_rollout_delta": (
                start_early_latent_max_abs_delta
            ),
            "start_raw_history_max_abs_rollout_delta": (
                start_raw_history_max_abs_delta
            ),
            "history_sensitive_base": True,
            "current_frame_residual_adapters": 2,
        },
        "dataset": {
            "alive_row_count": row_count,
            "arm_row_counts": {
                "carried": int(np.sum(dataset["arm"] == 0)),
                "repeated_current": int(np.sum(dataset["arm"] == 1)),
            },
            "fit_stride": FIT_ROW_STRIDE,
            "fit_rows": int(np.sum(fit_mask)),
            "validation_rows": int(np.sum(validation_mask)),
            "test_rows": int(np.sum(test_mask)),
            "split_seed": SPLIT_SEED,
            "split_phase_coverage": _phase_split_summary(
                np.asarray(source["start_phase"]), splits
            ),
            "split_leakage": False,
            "paired_environment_assignment": True,
            "action_minimum_raw_std": float(action_statistics["minimum_raw_std"]),
            "latent_minimum_raw_std": float(latent_statistics["minimum_raw_std"]),
            "early_latent_minimum_raw_std": (
                float(early_latent_statistics["minimum_raw_std"])
                if early_latent_statistics is not None
                else None
            ),
            "raw_history_statistics": raw_history_statistics,
        },
        "representations": {
            "current_only": {
                "dimension": CRITIC_OBS_DIM,
                "parameter_count": _parameter_count(state.critic_params),
            },
            "current_plus_action": {
                "dimension": CRITIC_OBS_DIM + ACTION_DIM,
                "parameter_count": _parameter_count(action_params),
            },
            "current_plus_actor_latent": {
                "dimension": CRITIC_OBS_DIM + ACTOR_LATENT_DIM,
                "parameter_count": _parameter_count(latent_params),
            },
        },
        "migration_initial_prediction_max_abs_drift": migration_drift,
        "fit_budget": {
            "steps_per_arm": FIT_STEPS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "learning_rate": CRITIC_LR,
            "gradient_clip_global_norm": 1.0,
            "selection_source": "validation-only",
            "final_test_evaluations_per_arm": 1,
        },
        "validation_selection": fit_reports,
        "original_online_final_test": original_test,
        "selected_final_test": final_test,
        "selection_thresholds": {
            "minimum_rank_correlation": MIN_RANK_CORRELATION,
            "maximum_nrmse": MAX_NRMSE,
            "required_groups": list(METRIC_GROUPS),
            "required_boundaries": list(required_boundaries),
            "required_gate_count": len(METRIC_GROUPS) * len(required_boundaries),
            "all_required_gates_must_pass": True,
            "classification_priority": (
                [
                    "current_only",
                    EARLY_LATENT_REPRESENTATION,
                    "current_plus_actor_latent",
                    "current_plus_action",
                ]
                if include_early_latent
                else [
                    "current_only",
                    "current_plus_action",
                    "current_plus_actor_latent",
                    RAW_HISTORY_REPRESENTATION,
                ]
                if include_raw_history
                else [
                    "current_only",
                    "current_plus_actor_latent",
                    "current_plus_action",
                ]
                if include_h0
                else [
                    "current_plus_actor_latent",
                    "current_plus_action",
                    "current_only",
                ]
            ),
        },
        "interpretation_boundary": (
            "This critic-only paired audit tests held-out scalar return "
            "prediction capacity after continuity-preserving input migration. "
            "It does not measure critic state-gradient alignment, create an "
            "independent second head, retain a fitted critic, update the actor, "
            "or establish AHAC behavior. The repeated-current arm remains a "
            "controlled counterfactual rather than a natural collision rate."
        ),
    }
    if early_latent_params is not None:
        result["representations"][EARLY_LATENT_REPRESENTATION] = {
            "dimension": CRITIC_OBS_DIM + ACTOR_EARLY_LATENT_DIM,
            "parameter_count": _parameter_count(early_latent_params),
        }
    if raw_history_params is not None:
        result["representations"][RAW_HISTORY_REPRESENTATION] = {
            "dimension": CRITIC_OBS_DIM + ACTOR_OBS_DIM,
            "parameter_count": _parameter_count(raw_history_params),
        }
    return artifact_arrays, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--alias-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-h0",
        action="store_true",
        help="add the exact common-input H0 boundary to selection and final gates",
    )
    parser.add_argument(
        "--include-early-latent",
        action="store_true",
        help="add the frozen actor's first 512-wide history activation",
    )
    parser.add_argument(
        "--include-raw-history",
        action="store_true",
        help="add the actor-normalized full ten-frame history input",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("E002 critic-representation seed must equal zero")
    validate_representation_options(
        include_h0=args.include_h0,
        include_early_latent=args.include_early_latent,
        include_raw_history=args.include_raw_history,
    )
    configure_jax()
    runtime = _validate_runtime()
    repository = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    alias_trace = args.alias_trace.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=args.code_commit,
    )
    if not alias_trace.is_file() or sha256_file(alias_trace) != E005_ALIAS_TRACE_SHA256:
        raise ValueError("E005 alias trace SHA-256 mismatch")
    preflight.update(
        protocol=(
            "g1-e002-critic-representation-early-latent-h0-preflight-v1"
            if args.include_early_latent
            else "g1-e002-critic-representation-raw-history-h0-preflight-v1"
            if args.include_raw_history
            else "g1-e002-critic-representation-h0-preflight-v1"
            if args.include_h0
            else "g1-e002-critic-representation-preflight-v1"
        ),
        runtime=runtime,
        alias_trace_path=str(alias_trace),
        alias_trace_sha256=E005_ALIAS_TRACE_SHA256,
        actor_optimizer_updates=0,
        critic_optimizer_updates_per_arm=FIT_STEPS,
        environment_steps_retained=0,
        include_h0=args.include_h0,
        include_early_latent=args.include_early_latent,
        include_raw_history=args.include_raw_history,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    arrays, result = collect_representation_audit(
        checkpoint=checkpoint,
        reference=reference,
        alias_trace=alias_trace,
        include_h0=args.include_h0,
        include_early_latent=args.include_early_latent,
        include_raw_history=args.include_raw_history,
    )
    dataset_path = output_root / "critic_representation_dataset.npz"
    result_path = output_root / "critic_representation_audit.json"
    _atomic_npz(dataset_path, arrays)
    result["dataset_sha256"] = sha256_file(dataset_path)
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": (
            "g1-e002-critic-representation-early-latent-h0-completion-v1"
            if args.include_early_latent
            else "g1-e002-critic-representation-raw-history-h0-completion-v1"
            if args.include_raw_history
            else "g1-e002-critic-representation-h0-completion-v1"
            if args.include_h0
            else "g1-e002-critic-representation-completion-v1"
        ),
        "valid": True,
        "classification": result["classification"],
        "actor_optimizer_updates": 0,
        "critic_optimizer_updates_per_arm": FIT_STEPS,
        "environment_steps_retained": 0,
        "policy_retained": False,
        "critic_retained": False,
        "retained_policy": None,
        "retained_critic": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "critic_representation_dataset.npz": sha256_file(dataset_path),
            "critic_representation_audit.json": sha256_file(result_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(output_root)


if __name__ == "__main__":
    main()
