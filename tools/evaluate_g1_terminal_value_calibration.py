"""Evaluate target-critic calibration against realized carried returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
)
from src.core.data_structures import Normalizer
from src.core.networks import Actor, Critic
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_flax_phase_grid import (
    ACTOR_HISTORY_LEN,
    DEFAULT_PHASES,
    LOOKAHEAD_STEPS,
)
from tools.evaluate_g1_tracking import (
    configure_jax,
    make_evaluation_env,
    scale_policy_action,
)
from tools.run_g1_root_recovery_continuation import (
    EXPECTED_CONTROLLER_SHA256,
    EXPECTED_MODEL_SHA256,
)


GAMMA = 0.99


def _finite_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def discounted_terminal_returns(rewards, *, gamma: float) -> np.ndarray:
    """Return realized discounted sums with no post-terminal bootstrap."""
    reward_array = _finite_vector(rewards, name="rewards")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    returns = np.empty_like(reward_array)
    tail = 0.0
    for index in range(reward_array.size - 1, -1, -1):
        tail = float(reward_array[index]) + gamma * tail
        returns[index] = tail
    return returns


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def calibration_metrics(values, realized_returns) -> dict[str, float]:
    """Compute scalar target-critic calibration diagnostics."""
    value_array = _finite_vector(values, name="values")
    return_array = _finite_vector(realized_returns, name="realized_returns")
    if value_array.shape != return_array.shape:
        raise ValueError("values and realized_returns must have equal length")
    errors = value_array - return_array
    return_scale = max(float(np.mean(np.abs(return_array))), 1e-12)
    return {
        "count": int(value_array.size),
        "pearson": _correlation(value_array, return_array),
        "rank_correlation": _correlation(
            _ordinal_ranks(value_array), _ordinal_ranks(return_array)
        ),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "nrmse": float(np.sqrt(np.mean(np.square(errors))) / return_scale),
        "bias": float(np.mean(errors)),
    }


def h12_boundary_records(traces: list[dict]) -> list[dict[str, float | int]]:
    """Extract critic and realized return after exactly 12 transitions."""
    records: list[dict[str, float | int]] = []
    for trace in traces:
        values = _finite_vector(trace["values"], name="trace values")
        returns = _finite_vector(
            trace["realized_returns"], name="trace realized_returns"
        )
        if values.shape != returns.shape or values.size <= 12:
            raise ValueError("every trace must contain an H12 boundary")
        value = float(values[12])
        realized = float(returns[12])
        records.append(
            {
                "phase": int(trace["phase"]),
                "transition": 12,
                "value": value,
                "realized_return": realized,
                "relative_error": abs(value - realized)
                / max(abs(realized), 1e-12),
            }
        )
    return records


def classify_calibration(
    aggregate: dict[str, float], h12_records: list[dict]
) -> str:
    """Apply the preregistered scalar-calibration decision rule."""
    if len(h12_records) != 5:
        raise ValueError("classification requires five H12 records")
    adequate = (
        float(aggregate["rank_correlation"]) >= 0.8
        and float(aggregate["nrmse"]) <= 0.25
        and all(float(row["relative_error"]) <= 0.25 for row in h12_records)
    )
    return (
        "terminal-value-calibration-adequate"
        if adequate
        else "terminal-value-miscalibrated"
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_code_provenance(
    *, expected_commit: str, actual_commit: str, dirty: bool
) -> dict[str, object]:
    """Require the evaluator to run from its registered clean commit."""
    if len(expected_commit) != 40 or actual_commit != expected_commit or dirty:
        raise ValueError("code checkout does not match clean registered commit")
    return {"code_commit": actual_commit, "dirty": False}


def validate_runtime_contract(
    *, solver_profile: str, seed: int
) -> dict[str, object]:
    """Fail closed on the registered deterministic evaluation settings."""
    if solver_profile != "g1-4x5" or seed != 0:
        raise ValueError("diagnostic requires solver g1-4x5 and seed zero")
    return {"solver_profile": solver_profile, "seed": seed}


def validate_step_arrays(
    *,
    physical_arrays,
    observation,
    action,
    critic_observation,
    wrench,
) -> None:
    """Require finite rollout state/action/observations and exact-zero wrench."""
    named_arrays = {
        "observation": observation,
        "action": action,
        "critic_observation": critic_observation,
        "wrench": wrench,
    }
    for index, value in enumerate(physical_arrays):
        named_arrays[f"physical_state[{index}]"] = value
    for name, value in named_arrays.items():
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains a nonfinite value")
    wrench_array = np.asarray(wrench)
    if not np.array_equal(wrench_array, np.zeros_like(wrench_array)):
        raise ValueError("terminal-value diagnostic requires exact-zero wrench")


def runtime_asset_provenance(env) -> dict[str, str]:
    """Hash the actual XML and controller consumed by the environment."""
    model_path = Path(env.xml_path).resolve()
    controller_path = Path(env.controller_path).resolve()
    model_sha256 = _sha256(model_path)
    controller_sha256 = _sha256(controller_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError("runtime model SHA-256 mismatch")
    if controller_sha256 != EXPECTED_CONTROLLER_SHA256:
        raise ValueError("runtime controller SHA-256 mismatch")
    return {
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "controller_path": str(controller_path),
        "controller_sha256": controller_sha256,
    }


def _critic_value(
    env,
    critic: Critic,
    target_critic_params,
    critic_normalizer: Normalizer,
    critic_normalizer_state,
    critic_obs,
) -> float:
    normalized = critic_normalizer.normalize(
        critic_normalizer_state, critic_obs
    ).astype(jnp.float32)
    value = critic.apply(target_critic_params, normalized)
    return float(jnp.asarray(value).reshape(-1)[0])


def evaluate_phase(
    env,
    *,
    phase: int,
    seed: int,
    action_fn,
    critic: Critic,
    target_critic_params,
    critic_normalizer: Normalizer,
    critic_normalizer_state,
) -> dict[str, object]:
    """Collect one replay-free carried trajectory and realized returns."""
    state = env.reset_at_phase(
        jax.random.PRNGKey(seed), jnp.array(0.0), jnp.array(phase)
    )
    values: list[float] = []
    rewards: list[float] = []
    terminal = False
    max_wrench = 0.0
    wrench_maxima: list[float] = []
    for _ in range(int(env.reference_transitions) - phase):
        action = action_fn(state)
        critic_obs = env._get_critic_obs(state.data, state.info)
        validate_step_arrays(
            physical_arrays=jax.tree_util.tree_leaves(state.data),
            observation=state.obs,
            action=action,
            critic_observation=critic_obs,
            wrench=state.data.xfrc_applied,
        )
        values.append(
            _critic_value(
                env,
                critic,
                target_critic_params,
                critic_normalizer,
                critic_normalizer_state,
                critic_obs,
            )
        )
        wrench_maxima.append(
            float(jnp.max(jnp.abs(state.data.xfrc_applied)))
        )
        state = env.step(state, action)
        rewards.append(float(state.reward))
        terminal = bool(float(state.info["terminal"]) > 0.5)
        if float(state.done) > 0.5:
            break
    if not terminal:
        raise ValueError(f"phase {phase} did not reach a natural terminal")
    final_critic_obs = env._get_critic_obs(state.data, state.info)
    validate_step_arrays(
        physical_arrays=jax.tree_util.tree_leaves(state.data),
        observation=state.obs,
        action=np.zeros(env.action_dim, dtype=np.float64),
        critic_observation=final_critic_obs,
        wrench=state.data.xfrc_applied,
    )
    max_wrench = max(wrench_maxima, default=0.0)
    realized_returns = discounted_terminal_returns(rewards, gamma=GAMMA)
    metrics = calibration_metrics(values, realized_returns)
    return {
        "phase": int(phase),
        "steps": len(rewards),
        "terminal": True,
        "max_abs_xfrc_applied": max_wrench,
        "max_abs_xfrc_applied_by_step": wrench_maxima,
        "values": [float(value) for value in values],
        "rewards": rewards,
        "realized_returns": [float(value) for value in realized_returns],
        "metrics": metrics,
    }


def build_payload(
    traces: list[dict], *, provenance: dict[str, object]
) -> dict[str, object]:
    """Build the manifest-last calibration artifact and outcome."""
    if [trace.get("phase") for trace in traces] != list(DEFAULT_PHASES):
        raise ValueError("calibration requires the registered phase ordering")
    values = [value for trace in traces for value in trace["values"]]
    returns = [
        value for trace in traces for value in trace["realized_returns"]
    ]
    aggregate = calibration_metrics(values, returns)
    boundaries = h12_boundary_records(traces)
    return {
        "protocol": "g1-terminal-value-calibration-v1",
        "provenance": provenance,
        "gamma": GAMMA,
        "traces": traces,
        "aggregate_metrics": aggregate,
        "h12_boundary_records": boundaries,
        "outcome": classify_calibration(aggregate, boundaries),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default="g1-4x5",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    runtime_contract = validate_runtime_contract(
        solver_profile=args.solver_profile, seed=args.seed
    )
    checkpoint_path = args.checkpoint.resolve()
    reference_path = args.reference_path.resolve()
    for path in (checkpoint_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    reference_sha256 = _sha256(reference_path)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    if reference_sha256 != args.expected_reference_sha256:
        raise ValueError("reference SHA-256 mismatch")
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
    with checkpoint_path.open("rb") as stream:
        checkpoint_state = pickle.load(stream)
    actor_params = checkpoint_state.actor_params
    if not isinstance(actor_params, FrozenPreviewResidualParams):
        raise ValueError("checkpoint is not a frozen residual preview actor")
    actor = Actor(
        env.action_dim,
        hidden=(512, 256, 128),
        squash=getattr(env, "squash_actor_actions", True),
        layer_norm=True,
        zero_output=False,
    )
    residual_actor = PreviewResidualAdapter(
        action_dim=env.action_dim, hidden_dim=256
    )
    actor_normalizer = Normalizer(env.actor_frame_obs_dim)
    critic_normalizer = Normalizer(env.critic_obs_dim)
    critic = Critic()

    def action(state):
        normalized = env.normalize_actor_obs(
            actor_normalizer, checkpoint_state.normalizer, state.obs
        ).astype(jnp.float32)
        candidate, _, _ = apply_frozen_preview_residual(
            actor,
            residual_actor,
            actor_params,
            normalized,
            history_len=ACTOR_HISTORY_LEN,
            treatment_frame_dim=env.actor_frame_obs_dim,
        )
        return scale_policy_action(candidate, 1.0).astype(jnp.float64)

    provenance = {
        **code_provenance,
        **runtime_asset_provenance(env),
        **runtime_contract,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
        "phases": list(DEFAULT_PHASES),
        "actor_reference_preview_mode": "delta",
        "actor_residual_preview_hidden": 256,
    }
    traces = []
    with solver_context(profile):
        for phase in DEFAULT_PHASES:
            traces.append(
                evaluate_phase(
                    env,
                    phase=phase,
                    seed=args.seed,
                    action_fn=action,
                    critic=critic,
                    target_critic_params=checkpoint_state.target_critic_params,
                    critic_normalizer=critic_normalizer,
                    critic_normalizer_state=checkpoint_state.critic_normalizer,
                )
            )
    payload = build_payload(traces, provenance=provenance)
    _write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "aggregate_metrics": payload["aggregate_metrics"],
                "h12_boundary_records": payload["h12_boundary_records"],
                "survival": [trace["steps"] for trace in traces],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
