"""Refit the frozen E012 critic on longer carried terminal returns."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
)
from src.core.data_structures import Normalizer
from src.core.networks import Actor, Critic
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_flax_phase_grid import ACTOR_HISTORY_LEN, LOOKAHEAD_STEPS
from tools.evaluate_g1_terminal_value_calibration import (
    GAMMA,
    _sha256,
    _write_json_atomic,
    calibration_metrics,
    discounted_terminal_returns,
    runtime_asset_provenance,
    validate_code_provenance,
    validate_runtime_contract,
)
from tools.evaluate_g1_tracking import (
    configure_jax,
    make_evaluation_env,
    scale_policy_action,
)


FIT_STEPS = 2_000
EVAL_INTERVAL = 20
CRITIC_LR = 5e-4


def phase_splits() -> dict[str, tuple[int, ...]]:
    """Return immutable disjoint fit, validation, and final-test phases."""
    return {
        "fit": tuple(range(10, 400, 20)),
        "validation": (20, 120, 220, 320, 380),
        "test": (0, 100, 200, 300, 400),
    }


def trajectory_rows(
    critic_observations,
    rewards,
    *,
    terminals,
    gamma: float,
) -> dict[str, np.ndarray]:
    """Extract the prefix through the first natural terminal."""
    observations = np.asarray(critic_observations)
    reward_array = np.asarray(rewards, dtype=np.float64)
    terminal_array = np.asarray(terminals, dtype=bool)
    if (
        observations.ndim != 2
        or reward_array.ndim != 1
        or terminal_array.ndim != 1
        or observations.shape[0] != reward_array.size
        or reward_array.size != terminal_array.size
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(reward_array))
    ):
        raise ValueError("trajectory arrays are malformed or nonfinite")
    terminal_indices = np.flatnonzero(terminal_array)
    if terminal_indices.size != 1:
        raise ValueError("trajectory must contain exactly one natural terminal")
    end = int(terminal_indices[0]) + 1
    if np.any(terminal_array[: end - 1]) or np.any(terminal_array[end:]):
        raise ValueError("trajectory contains an invalid terminal sequence")
    return {
        "critic_observations": observations[:end],
        "rewards": reward_array[:end],
        "returns": discounted_terminal_returns(
            reward_array[:end], gamma=gamma
        ),
    }


def calibration_candidate_key(
    metrics: dict[str, float], *, step: int
) -> tuple[bool, float, float, int]:
    """Select by passing validation, lower NRMSE, rank, then earlier step."""
    rank = float(metrics["rank_correlation"])
    nrmse = float(metrics["nrmse"])
    if not np.isfinite(rank) or not np.isfinite(nrmse) or step < 0:
        raise ValueError("candidate metrics must be finite and step nonnegative")
    return (rank >= 0.8 and nrmse <= 0.25, -nrmse, rank, -step)


def _tree_max_abs_difference(left: Any, right: Any) -> float:
    if left is right:
        return 0.0
    left_structure = jax.tree_util.tree_structure(left)
    if left_structure != jax.tree_util.tree_structure(right):
        return float("inf")
    maxima = []
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    ):
        lhs_array = np.asarray(lhs)
        rhs_array = np.asarray(rhs)
        if lhs_array.shape != rhs_array.shape:
            return float("inf")
        if lhs_array.size:
            if np.issubdtype(lhs_array.dtype, np.number) and np.issubdtype(
                rhs_array.dtype, np.number
            ):
                maxima.append(float(np.max(np.abs(lhs_array - rhs_array))))
            else:
                maxima.append(
                    0.0
                    if np.array_equal(lhs_array, rhs_array)
                    else float("inf")
                )
    return max(maxima, default=0.0)


def noncritic_state_drift(original: Any, candidate: Any) -> dict[str, float | bool]:
    """Require every TrainState field outside the critic triplet to be exact."""
    excluded = {"critic_params", "target_critic_params", "critic_opt"}
    field_names = tuple(getattr(original, "__dataclass_fields__", {}))
    if not field_names:
        raise ValueError("state must expose dataclass fields")
    all_drift = {
        name: _tree_max_abs_difference(
            getattr(original, name), getattr(candidate, name)
        )
        for name in field_names
        if name not in excluded
    }
    selected = {
        name: all_drift[name]
        for name in ("actor_params", "actor_opt", "normalizer", "env_state")
    }
    return {**selected, "valid": all(value == 0.0 for value in all_drift.values())}


def replace_critic_state(
    state: Any,
    *,
    critic_params: Any,
    critic_opt: Any,
    replace_fn: Callable[..., Any] | None = None,
) -> Any:
    """Replace only online/target critic parameters and critic optimizer."""
    if replace_fn is None:
        def replace_fn(value, **changes):
            return value.replace(**changes)
    return replace_fn(
        state,
        critic_params=critic_params,
        target_critic_params=critic_params,
        critic_opt=critic_opt,
    )


def _concatenate_trajectories(
    trajectories: dict[int, dict[str, np.ndarray]], phases: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(
            [trajectories[phase]["critic_observations"] for phase in phases]
        ),
        np.concatenate([trajectories[phase]["returns"] for phase in phases]),
    )


def _predict(critic: Critic, params: Any, observations: jax.Array) -> np.ndarray:
    return np.asarray(
        jax.jit(lambda p, x: critic.apply(p, x).reshape(-1))(
            params, observations
        )
    )


def _h12_records(
    trajectories: dict[int, dict[str, np.ndarray]],
    phases: tuple[int, ...],
    predictions: dict[int, np.ndarray],
) -> list[dict[str, float | int]]:
    records = []
    for phase in phases:
        target = float(trajectories[phase]["returns"][12])
        value = float(predictions[phase][12])
        records.append(
            {
                "phase": phase,
                "value": value,
                "realized_return": target,
                "relative_error": abs(value - target) / max(abs(target), 1e-12),
            }
        )
    return records


def capture_trajectories(env, checkpoint_state: Any, *, seed: int) -> dict[int, dict]:
    """Capture every frozen split trajectory in one vmapped MJX execution."""
    splits = phase_splits()
    phases = tuple((*splits["fit"], *splits["validation"], *splits["test"]))
    phase_array = jnp.asarray(phases, dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(phases))
    states = jax.jit(jax.vmap(env.reset_at_phase))(
        keys, jnp.zeros(len(phases), dtype=jnp.float64), phase_array
    )
    actor = Actor(
        env.action_dim,
        hidden=(512, 256, 128),
        squash=getattr(env, "squash_actor_actions", True),
        layer_norm=True,
        zero_output=False,
    )
    residual = PreviewResidualAdapter(action_dim=env.action_dim, hidden_dim=256)
    actor_normalizer = Normalizer(env.actor_frame_obs_dim)

    @jax.jit
    def batched_step(current_states):
        normalized = jax.vmap(
            lambda obs: env.normalize_actor_obs(
                actor_normalizer, checkpoint_state.normalizer, obs
            )
        )(current_states.obs).astype(jnp.float32)
        action, _, _ = apply_frozen_preview_residual(
            actor,
            residual,
            checkpoint_state.actor_params,
            normalized,
            history_len=ACTOR_HISTORY_LEN,
            treatment_frame_dim=env.actor_frame_obs_dim,
        )
        action = scale_policy_action(action, 1.0).astype(jnp.float64)
        critic_obs = jax.vmap(env._get_critic_obs)(
            current_states.data, current_states.info
        )
        state_finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(leaf))
                    for leaf in jax.tree_util.tree_leaves(current_states.data)
                ]
            )
        )
        next_states = jax.vmap(env.step)(current_states, action)
        return (
            next_states,
            critic_obs,
            action,
            state_finite,
            jnp.all(current_states.data.xfrc_applied == 0),
        )

    active = np.ones(len(phases), dtype=bool)
    observations: list[list[np.ndarray]] = [[] for _ in phases]
    rewards: list[list[float]] = [[] for _ in phases]
    terminals: list[list[bool]] = [[] for _ in phases]
    max_steps = np.asarray([int(env.reference_transitions) - p for p in phases])
    for elapsed in range(int(np.max(max_steps))):
        next_states, critic_obs, action, state_finite, wrench_zero = batched_step(
            states
        )
        critic_obs_np = np.asarray(critic_obs)
        action_np = np.asarray(action)
        reward_np = np.asarray(next_states.reward)
        done_np = np.asarray(next_states.done) > 0.5
        terminal_np = np.asarray(next_states.info["terminal"]) > 0.5
        if (
            not bool(state_finite)
            or not bool(wrench_zero)
            or not np.all(np.isfinite(critic_obs_np))
            or not np.all(np.isfinite(action_np))
            or not np.all(np.isfinite(reward_np))
        ):
            raise ValueError("carried capture contains nonfinite or nonzero-wrench data")
        for index in np.flatnonzero(active):
            observations[index].append(critic_obs_np[index])
            rewards[index].append(float(reward_np[index]))
            terminals[index].append(bool(terminal_np[index]))
            if done_np[index]:
                if not terminal_np[index]:
                    raise ValueError("carried capture ended by truncation")
                active[index] = False
            elif elapsed + 1 >= max_steps[index]:
                raise ValueError("carried capture did not naturally terminate")
        states = next_states
        if not np.any(active):
            break
    if np.any(active):
        raise ValueError("not every carried trajectory terminated")
    return {
        phase: trajectory_rows(
            np.asarray(observations[index]),
            np.asarray(rewards[index]),
            terminals=np.asarray(terminals[index]),
            gamma=GAMMA,
        )
        for index, phase in enumerate(phases)
    }


def fit_critic(
    checkpoint_state: Any,
    critic: Critic,
    normalized_trajectories: dict[int, dict[str, np.ndarray]],
) -> tuple[Any, Any, dict[str, object]]:
    """Continue only critic Adam and select exclusively on validation phases."""
    splits = phase_splits()
    fit_obs, fit_returns = _concatenate_trajectories(
        normalized_trajectories, splits["fit"]
    )
    validation_obs, validation_returns = _concatenate_trajectories(
        normalized_trajectories, splits["validation"]
    )
    fit_obs_jax = jnp.asarray(fit_obs, dtype=jnp.float32)
    fit_returns_jax = jnp.asarray(fit_returns, dtype=jnp.float32)
    validation_obs_jax = jnp.asarray(validation_obs, dtype=jnp.float32)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(CRITIC_LR)
    )

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean(
                jnp.square(
                    critic.apply(p, fit_obs_jax).reshape(-1) - fit_returns_jax
                )
            )
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, loss

    params = checkpoint_state.critic_params
    opt_state = checkpoint_state.critic_opt
    original_target_metrics = calibration_metrics(
        _predict(critic, checkpoint_state.target_critic_params, validation_obs_jax),
        validation_returns,
    )
    original_current_metrics = calibration_metrics(
        _predict(critic, params, validation_obs_jax), validation_returns
    )
    candidates = [
        {
            "step": 0,
            "fit_loss": None,
            **original_current_metrics,
        }
    ]
    best = (
        calibration_candidate_key(original_current_metrics, step=0),
        params,
        opt_state,
        candidates[0],
    )
    for step in range(EVAL_INTERVAL, FIT_STEPS + 1, EVAL_INTERVAL):
        loss = None
        for _ in range(EVAL_INTERVAL):
            params, opt_state, loss = update(params, opt_state)
        predictions = _predict(critic, params, validation_obs_jax)
        metrics = calibration_metrics(predictions, validation_returns)
        row = {"step": step, "fit_loss": float(loss), **metrics}
        candidates.append(row)
        key = calibration_candidate_key(metrics, step=step)
        if key > best[0]:
            best = (key, params, opt_state, row)
    return best[1], best[2], {
        "original_target_validation": original_target_metrics,
        "original_current_validation": original_current_metrics,
        "candidates": candidates,
        "selected": best[3],
        "selected_source": (
            "existing-current-critic"
            if best[3]["step"] == 0
            else "critic-adam-refit"
        ),
    }


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-hparams-sha256", required=True)
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
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
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
    reference_path = args.reference_path.resolve()
    if _sha256(checkpoint_path) != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    if _sha256(reference_path) != args.expected_reference_sha256:
        raise ValueError("reference SHA-256 mismatch")
    source_hparams = checkpoint_path.parent / "hparams.json"
    if _sha256(source_hparams) != args.expected_hparams_sha256:
        raise ValueError("checkpoint sibling hparams SHA-256 mismatch")
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
        "hparams_path": str(source_hparams),
        "hparams_sha256": _sha256(source_hparams),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
    }
    with solver_context(profile):
        trajectories = capture_trajectories(env, checkpoint_state, seed=args.seed)
    critic = Critic()
    critic_normalizer = Normalizer(env.critic_obs_dim)
    normalized = {
        phase: {
            **rows,
            "critic_observations": np.asarray(
                critic_normalizer.normalize(
                    checkpoint_state.critic_normalizer,
                    jnp.asarray(rows["critic_observations"]),
                ),
                dtype=np.float32,
            ),
        }
        for phase, rows in trajectories.items()
    }
    selected_params, selected_opt, fit_report = fit_critic(
        checkpoint_state, critic, normalized
    )
    splits = phase_splits()
    split_metrics = {}
    split_predictions = {}
    for split, phases in splits.items():
        obs, returns = _concatenate_trajectories(normalized, phases)
        predictions = _predict(critic, selected_params, jnp.asarray(obs))
        split_metrics[split] = calibration_metrics(predictions, returns)
        offset = 0
        split_predictions[split] = {}
        for phase in phases:
            count = normalized[phase]["returns"].size
            split_predictions[split][phase] = predictions[offset : offset + count]
            offset += count
    test_obs, test_returns = _concatenate_trajectories(
        normalized, splits["test"]
    )
    original_test_predictions = {
        "target": _predict(
            critic,
            checkpoint_state.target_critic_params,
            jnp.asarray(test_obs),
        ),
        "current": _predict(
            critic,
            checkpoint_state.critic_params,
            jnp.asarray(test_obs),
        ),
    }
    original_test_metrics = {
        name: calibration_metrics(predictions, test_returns)
        for name, predictions in original_test_predictions.items()
    }
    original_target_predictions_by_phase = {}
    offset = 0
    for phase in splits["test"]:
        count = normalized[phase]["returns"].size
        original_target_predictions_by_phase[phase] = (
            original_test_predictions["target"][offset : offset + count]
        )
        offset += count
    original_target_h12 = _h12_records(
        normalized,
        splits["test"],
        original_target_predictions_by_phase,
    )
    h12 = _h12_records(
        normalized, splits["test"], split_predictions["test"]
    )
    original_target_validation = fit_report["original_target_validation"]
    improves_parent = (
        split_metrics["validation"]["rank_correlation"]
        > original_target_validation["rank_correlation"]
        and split_metrics["validation"]["nrmse"]
        < original_target_validation["nrmse"]
        and split_metrics["test"]["rank_correlation"]
        > original_test_metrics["target"]["rank_correlation"]
        and split_metrics["test"]["nrmse"]
        < original_test_metrics["target"]["nrmse"]
        and all(
            selected["relative_error"] < original["relative_error"]
            for selected, original in zip(h12, original_target_h12)
        )
    )
    success = (
        split_metrics["validation"]["rank_correlation"] >= 0.8
        and split_metrics["validation"]["nrmse"] <= 0.25
        and split_metrics["test"]["rank_correlation"] >= 0.8
        and split_metrics["test"]["nrmse"] <= 0.25
        and all(row["relative_error"] <= 0.25 for row in h12)
        and improves_parent
    )
    candidate_state = replace_critic_state(
        checkpoint_state,
        critic_params=selected_params,
        critic_opt=selected_opt,
    )
    drift = noncritic_state_drift(checkpoint_state, candidate_state)
    if not drift["valid"]:
        raise ValueError("critic refit changed a non-critic TrainState leaf")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_path = output / "carried_return_dataset.npz"
    dataset_payload = {}
    for phase, rows in trajectories.items():
        for key, value in rows.items():
            dataset_payload[f"phase_{phase}_{key}"] = value
    dataset_temporary = dataset_path.with_name(f".{dataset_path.name}.tmp")
    with dataset_temporary.open("wb") as stream:
        np.savez_compressed(stream, **dataset_payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(dataset_temporary, dataset_path)
    checkpoint_output = output / "critic_refit.pkl"
    if success:
        _atomic_pickle(checkpoint_output, candidate_state)
        hparams_output = output / "hparams.json"
        hparams_temporary = output / ".hparams.json.tmp"
        source_hparams_payload = json.loads(
            source_hparams.read_text(encoding="utf-8")
        )
        source_hparams_payload.update(
            {
                "carried_return_critic_refit": True,
                "carried_return_critic_refit_source_checkpoint_sha256": (
                    _sha256(checkpoint_path)
                ),
                "carried_return_critic_refit_steps": int(
                    fit_report["selected"]["step"]
                ),
                "carried_return_critic_refit_lr": CRITIC_LR,
                "carried_return_critic_refit_protocol": (
                    "g1-carried-return-critic-refit-v1"
                ),
            }
        )
        hparams_temporary.write_text(
            json.dumps(
                source_hparams_payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(hparams_temporary, hparams_output)
    report = {
        "protocol": "g1-carried-return-critic-refit-v1",
        "provenance": provenance,
        "phase_splits": {key: list(value) for key, value in splits.items()},
        "fit_steps": FIT_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "critic_lr": CRITIC_LR,
        "survival": {
            str(p): int(rows["returns"].size)
            for p, rows in trajectories.items()
        },
        "fit_report": fit_report,
        "original_test_metrics": original_test_metrics,
        "original_target_test_h12_records": original_target_h12,
        "selected_split_metrics": split_metrics,
        "test_h12_records": h12,
        "improves_original_target": improves_parent,
        "noncritic_state_drift": drift,
        "outcome": "critic-refit-calibrated" if success else "critic-refit-insufficient",
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_sha256": _sha256(checkpoint_output) if success else None,
    }
    _write_json_atomic(output / "critic_refit_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
