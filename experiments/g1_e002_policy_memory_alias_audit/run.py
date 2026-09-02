"""Audit omitted policy memory on the exact retained E002 carried state bank.

The control arm uses E002's carried ten-frame actor history.  The treatment
repeats only the environment-derived current actor frame ten times while
preserving the physical state, phase, last action, RNG, and every other info
field.  The retained deterministic actor is then rolled out from both states in
separately compiled graphs.  No parameter, optimizer, normalizer, environment
state, or policy is retained.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
from src.core.data_structures import Normalizer
from src.core.networks import Critic
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_flax_phase_grid import (
    load_checkpoint_environment_contract,
)
from tools.evaluate_g1_tracking import _load_policy
from tools.run_g1_dual_scale_root_position import (
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_HPARAMS_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


ACTOR_HISTORY_LEN = 10
ACTOR_FRAME_DIM = 328
ACTOR_OBS_DIM = ACTOR_HISTORY_LEN * ACTOR_FRAME_DIM
CRITIC_OBS_DIM = 286
MIN_ACTION_RELATIVE_RMS = 0.02
MIN_AFFECTED_ENV_FRACTION = 0.10
PER_ENV_ACTION_RMS_FLOOR = 0.01
MIN_RETURN_NRMSE = 0.10
MIN_SURVIVAL_MAE = 2.0
START_EQUALITY_TOLERANCE = 1e-6
E004_CALIBRATION_TRACE_SHA256 = (
    "86860b9b3ad45b11ba88bfb9ddaae0a361957220ee06ded1661d07bc0019dbe7"
)


def replace_with_repeated_current_history(env_state):
    """Change only flattened actor history and its canonical info copy."""

    history = env_state.info.get("actor_obs_history")
    shape = getattr(history, "shape", None)
    obs_shape = getattr(env_state.obs, "shape", None)
    if (
        shape is None
        or len(shape) != 3
        or shape[0] < 1
        or shape[1] < 2
        or shape[2] < 1
        or obs_shape != (shape[0], shape[1] * shape[2])
    ):
        raise ValueError("flattened actor observation does not match history")
    repeated = jnp.repeat(jnp.asarray(history)[:, -1:, :], shape[1], axis=1)
    return env_state.replace(
        obs=repeated.reshape(shape[0], shape[1] * shape[2]),
        info={**env_state.info, "actor_obs_history": repeated},
    )


def paired_divergence_metrics(
    carried: np.ndarray, repeated: np.ndarray
) -> dict[str, float]:
    """Summarize paired scalar outcomes using the carried arm's scale."""

    baseline = np.asarray(carried, dtype=np.float64)
    treatment = np.asarray(repeated, dtype=np.float64)
    if (
        baseline.ndim != 1
        or treatment.shape != baseline.shape
        or baseline.size < 1
        or not np.isfinite(baseline).all()
        or not np.isfinite(treatment).all()
    ):
        raise ValueError("paired divergence arrays are invalid")
    difference = treatment - baseline
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    baseline_scale = float(np.mean(np.abs(baseline)))
    return {
        "mean_carried": float(np.mean(baseline)),
        "mean_repeated": float(np.mean(treatment)),
        "mean_repeated_minus_carried": float(np.mean(difference)),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "carried_mean_absolute_scale": baseline_scale,
        "carried_mean_absolute_normalized_rmse": (rmse / max(baseline_scale, 1e-12)),
        "pearson_correlation": (
            float(np.corrcoef(baseline, treatment)[0, 1])
            if np.std(baseline) > 0.0 and np.std(treatment) > 0.0
            else 1.0
            if np.array_equal(baseline, treatment)
            else 0.0
        ),
        "exact_equal_fraction": float(np.mean(baseline == treatment)),
    }


def policy_action_metrics(
    carried: np.ndarray, repeated: np.ndarray
) -> dict[str, object]:
    """Measure paired differences between the two executed start actions."""

    baseline = np.asarray(carried, dtype=np.float64)
    treatment = np.asarray(repeated, dtype=np.float64)
    if (
        baseline.ndim != 2
        or treatment.shape != baseline.shape
        or baseline.size < 1
        or not np.isfinite(baseline).all()
        or not np.isfinite(treatment).all()
    ):
        raise ValueError("paired policy actions are invalid")
    difference = treatment - baseline
    per_environment_rms = np.sqrt(np.mean(np.square(difference), axis=-1))
    global_difference_rms = float(np.sqrt(np.mean(np.square(difference))))
    carried_action_rms = float(np.sqrt(np.mean(np.square(baseline))))
    return {
        "global_difference_rms": global_difference_rms,
        "carried_action_rms": carried_action_rms,
        "relative_rms": global_difference_rms / max(carried_action_rms, 1e-12),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "exact_equal_fraction": float(np.mean(baseline == treatment)),
        "fraction_env_rms_at_least_0p01": float(
            np.mean(per_environment_rms >= PER_ENV_ACTION_RMS_FLOOR)
        ),
        "per_environment_rms": _distribution(per_environment_rms),
    }


def classify_policy_memory_alias(
    *,
    action_relative_rms: float,
    action_env_fraction: float,
    return_nrmse: float,
    survival_mae: float,
) -> str:
    """Classify the preregistered materiality gates after invariants pass."""

    values = (
        action_relative_rms,
        action_env_fraction,
        return_nrmse,
        survival_mae,
    )
    if (
        not all(math.isfinite(float(value)) for value in values)
        or action_relative_rms < 0.0
        or not 0.0 <= action_env_fraction <= 1.0
        or return_nrmse < 0.0
        or survival_mae < 0.0
    ):
        raise ValueError("alias metrics are invalid")
    action_material = (
        action_relative_rms >= MIN_ACTION_RELATIVE_RMS
        or action_env_fraction >= MIN_AFFECTED_ENV_FRACTION
    )
    outcome_material = (
        return_nrmse >= MIN_RETURN_NRMSE or survival_mae >= MIN_SURVIVAL_MAE
    )
    if action_material and outcome_material:
        return "policy-memory-alias-material"
    if action_material:
        return "policy-memory-action-sensitive-return-immaterial"
    return "policy-memory-immaterial"


def _tree_max_abs_delta(first: Any, second: Any) -> float:
    first_leaves, first_tree = jax.tree.flatten(first)
    second_leaves, second_tree = jax.tree.flatten(second)
    if first_tree != second_tree or len(first_leaves) != len(second_leaves):
        return math.inf
    maximum = 0.0
    for left, right in zip(first_leaves, second_leaves, strict=True):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if (
            left_array.shape != right_array.shape
            or left_array.dtype != right_array.dtype
        ):
            return math.inf
        if np.array_equal(left_array, right_array):
            continue
        if left_array.dtype.kind in "biufc" and right_array.dtype.kind in "biufc":
            delta = np.abs(
                left_array.astype(np.float64) - right_array.astype(np.float64)
            )
            if not np.isfinite(delta).all():
                return math.inf
            maximum = max(maximum, float(np.max(delta)))
        else:
            return math.inf
    return maximum


def _start_probe(
    environment,
    actor,
    actor_params,
    actor_normalizer,
    actor_normalizer_state,
    critic,
    critic_params,
    target_critic_params,
    critic_normalizer,
    critic_normalizer_state,
    env_state,
) -> dict[str, jax.Array]:
    current_frame = jax.vmap(environment._get_actor_obs)(env_state.data, env_state.info)
    critic_obs = jax.vmap(environment._get_critic_obs)(env_state.data, env_state.info)
    normalized_actor_obs = environment.normalize_actor_obs(
        actor_normalizer,
        actor_normalizer_state,
        env_state.obs,
    ).astype(jnp.float32)
    action = jax.vmap(lambda observation: actor.apply(actor_params, observation))(
        normalized_actor_obs
    ).astype(jnp.float64)
    if environment.clip_sampled_actor_actions:
        action = jnp.clip(action, -1.0, 1.0)
    normalized_critic_obs = critic_normalizer.normalize(
        critic_normalizer_state, critic_obs
    ).astype(jnp.float32)
    return {
        "current_frame": current_frame,
        "critic_obs": critic_obs,
        "online_value": critic.apply(critic_params, normalized_critic_obs).squeeze(-1),
        "target_value": critic.apply(
            target_critic_params, normalized_critic_obs
        ).squeeze(-1),
        "action": action,
    }


def _rollout_summary(
    trace: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if (
        not bool(np.all(trace["finite"]))
        or float(np.max(trace["xfrc_max"])) != 0.0
        or not np.isfinite(trace["reward"]).all()
    ):
        raise ValueError("rollout is nonfinite or assisted")
    done = np.asarray(trace["done"], dtype=bool)
    realized, alive = first_terminal_returns(trace["reward"], done, gamma=GAMMA)
    if not np.all(np.any(done, axis=0)):
        raise ValueError("rollout did not reach first done for every state")
    first_done = np.argmax(done, axis=0) + 1
    columns = np.arange(done.shape[1])
    first_terminal = np.asarray(trace["terminal"], dtype=bool)[first_done - 1, columns]
    summary = {
        "survival": _distribution(first_done),
        "natural_terminal_count": int(np.sum(first_terminal)),
        "truncation_count": int(np.sum(~first_terminal)),
        "reward": _distribution(np.asarray(trace["reward"])[alive]),
        "return_from_start": _distribution(realized[0]),
        "post_first_done_rewards_masked": True,
    }
    derived = {
        "alive": alive,
        "realized_return": realized,
        "first_done": first_done,
        "first_done_terminal": first_terminal,
    }
    return summary, derived


def _exact_e004_parity(
    carried_trace: Mapping[str, np.ndarray],
    carried_derived: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
) -> dict[str, object]:
    fields = (
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
    for name in fields:
        expected = np.asarray(source[f"deterministic_{name}"])
        actual = np.asarray(carried_trace[name])
        parity[f"{name}_exact"] = bool(np.array_equal(actual, expected))
    for name in derived_fields:
        expected = np.asarray(source[f"deterministic_{name}"])
        actual = np.asarray(carried_derived[name])
        parity[f"{name}_exact"] = bool(np.array_equal(actual, expected))
    parity["start_phase_exact"] = bool(
        np.array_equal(
            np.asarray(carried_trace["phase_before"])[0],
            np.asarray(source["start_phase"]),
        )
    )
    parity["all_exact"] = all(bool(value) for value in parity.values())
    return parity


def collect_policy_memory_alias(
    *, checkpoint: Path, reference: Path, calibration_trace: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run the exact carried and history-only-counterfactual policy pair."""

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
        raise ValueError("retained E002 boundary does not match alias audit")
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != START_STEP
        or np.asarray(state.env_state.obs).shape != (EFFECTIVE_NUM_ENVS, ACTOR_OBS_DIM)
        or np.asarray(state.env_state.info["actor_obs_history"]).shape
        != (EFFECTIVE_NUM_ENVS, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)
    ):
        raise ValueError("retained E002 checkpoint state is invalid")

    environment = _make_environment(hparams, reference)
    if (
        int(environment.actor_history_len) != ACTOR_HISTORY_LEN
        or int(environment.actor_frame_obs_dim) != ACTOR_FRAME_DIM
        or int(environment.actor_obs_dim) != ACTOR_OBS_DIM
        or int(environment.critic_obs_dim) != CRITIC_OBS_DIM
    ):
        raise ValueError("environment observation dimensions drifted")
    horizon = int(environment.max_episode_length)
    actor, actor_params, actor_normalizer_state = _load_policy(
        environment, checkpoint, 0
    )
    actor_normalizer = Normalizer(ACTOR_FRAME_DIM)
    critic_normalizer = Normalizer(CRITIC_OBS_DIM)
    critic = Critic()
    repeated_state = replace_with_repeated_current_history(state.env_state)

    probe = jax.jit(
        lambda env_state: _start_probe(
            environment,
            actor,
            actor_params,
            actor_normalizer,
            actor_normalizer_state,
            critic,
            state.critic_params,
            state.target_critic_params,
            critic_normalizer,
            state.critic_normalizer,
            env_state,
        )
    )
    carried_probe = jax.tree.map(np.asarray, probe(state.env_state))
    repeated_probe = jax.tree.map(np.asarray, probe(repeated_state))
    carried_history = np.asarray(state.env_state.info["actor_obs_history"])
    repeated_history = np.asarray(repeated_state.info["actor_obs_history"])
    current_frame_exact = bool(
        np.array_equal(carried_history[:, -1], carried_probe["current_frame"])
        and np.array_equal(
            carried_probe["current_frame"], repeated_probe["current_frame"]
        )
        and np.array_equal(repeated_history[:, -1], repeated_probe["current_frame"])
    )
    critic_obs_exact = bool(
        np.array_equal(carried_probe["critic_obs"], repeated_probe["critic_obs"])
    )
    online_value_max_abs_delta = float(
        np.max(np.abs(carried_probe["online_value"] - repeated_probe["online_value"]))
    )
    target_value_max_abs_delta = float(
        np.max(np.abs(carried_probe["target_value"] - repeated_probe["target_value"]))
    )
    older_difference = carried_history[:, :-1] - repeated_history[:, :-1]
    older_different_fraction = float(
        np.mean(np.any(older_difference != 0.0, axis=(1, 2)))
    )
    nonhistory_info_delta = _tree_max_abs_delta(
        {
            key: value
            for key, value in state.env_state.info.items()
            if key != "actor_obs_history"
        },
        {
            key: value
            for key, value in repeated_state.info.items()
            if key != "actor_obs_history"
        },
    )
    if (
        not current_frame_exact
        or not critic_obs_exact
        or online_value_max_abs_delta > START_EQUALITY_TOLERANCE
        or target_value_max_abs_delta > START_EQUALITY_TOLERANCE
        or older_different_fraction <= 0.0
        or nonhistory_info_delta != 0.0
    ):
        raise ValueError("history-only start-state invariants failed")

    with np.load(calibration_trace, allow_pickle=False) as source_archive:
        source = {
            name: np.asarray(source_archive[name]) for name in source_archive.files
        }
    action_noise = np.asarray(source["action_noise"])
    if action_noise.shape != (
        EFFECTIVE_NUM_ENVS,
        horizon,
        environment.action_dim,
    ):
        raise ValueError("E004 prospective noise tape shape drifted")
    scan_noise = jnp.asarray(np.swapaxes(action_noise, 0, 1))

    def prepare_action(env_state, epsilon):
        del epsilon
        rng_pairs = jax.vmap(lambda key: jax.random.split(key, 2))(
            env_state.info["rng"]
        )
        obs_rng = rng_pairs[:, 0]
        env_rng = rng_pairs[:, 1]
        env_state = env_state.replace(info={**env_state.info, "rng": env_rng})
        actor_obs = jax.vmap(environment._apply_obs_noise)(env_state.obs, obs_rng)
        normalized = environment.normalize_actor_obs(
            actor_normalizer,
            actor_normalizer_state,
            actor_obs,
        ).astype(jnp.float32)
        action = jax.vmap(lambda observation: actor.apply(actor_params, observation))(
            normalized
        ).astype(jnp.float64)
        if environment.clip_sampled_actor_actions:
            action = jnp.clip(action, -1.0, 1.0)
        return env_state, action

    def make_rollout():
        def rollout(initial_state, noise):
            def step(env_state, epsilon):
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
                env_state, action = prepare_action(env_state, epsilon)
                next_state = jax.vmap(environment.step)(env_state, action)
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
                            [
                                jnp.all(jnp.isfinite(value))
                                for value in (
                                    online_value,
                                    target_value,
                                    next_state.reward,
                                    action,
                                    next_state.data.qpos,
                                    next_state.data.qvel,
                                )
                            ]
                        )
                    ),
                }
                return next_state, output

            return jax.lax.scan(step, initial_state, noise)

        return rollout

    profile = get_solver_profile(str(hparams["solver_profile"]))
    with solver_context(profile):
        compiled_carried = jax.jit(make_rollout())
        compiled_repeated = jax.jit(make_rollout())
        _, carried_device_trace = compiled_carried(state.env_state, scan_noise)
        _, repeated_device_trace = compiled_repeated(repeated_state, scan_noise)
    traces = {
        "carried": jax.tree.map(np.asarray, carried_device_trace),
        "repeated_current": jax.tree.map(np.asarray, repeated_device_trace),
    }
    summaries: dict[str, dict[str, object]] = {}
    derived: dict[str, dict[str, np.ndarray]] = {}
    for name, trace in traces.items():
        summaries[name], derived[name] = _rollout_summary(trace)

    e004_parity = _exact_e004_parity(traces["carried"], derived["carried"], source)
    if not bool(e004_parity["all_exact"]):
        raise ValueError("carried control does not exactly reproduce E004")
    if (
        not np.array_equal(np.asarray(state.key), source["checkpoint_key"])
        or not np.array_equal(
            np.asarray(state.env_state.info["phase"]), source["start_phase"]
        )
        or not np.array_equal(
            carried_probe["online_value"], traces["carried"]["online_value"][0]
        )
        or not np.array_equal(
            carried_probe["target_value"], traces["carried"]["target_value"][0]
        )
    ):
        raise ValueError("checkpoint/start-probe parity failed")

    action_metrics = policy_action_metrics(
        carried_probe["action"], repeated_probe["action"]
    )
    return_metrics = paired_divergence_metrics(
        derived["carried"]["realized_return"][0],
        derived["repeated_current"]["realized_return"][0],
    )
    survival_metrics = paired_divergence_metrics(
        derived["carried"]["first_done"].astype(np.float64),
        derived["repeated_current"]["first_done"].astype(np.float64),
    )
    classification = classify_policy_memory_alias(
        action_relative_rms=float(action_metrics["relative_rms"]),
        action_env_fraction=float(action_metrics["fraction_env_rms_at_least_0p01"]),
        return_nrmse=float(return_metrics["carried_mean_absolute_normalized_rmse"]),
        survival_mae=float(survival_metrics["mean_absolute_difference"]),
    )

    arrays: dict[str, np.ndarray] = {
        "checkpoint_key": np.asarray(state.key),
        "start_phase": np.asarray(state.env_state.info["phase"]),
        "initial_carried_history": carried_history,
        "initial_repeated_current_history": repeated_history,
        "initial_current_frame": np.asarray(carried_probe["current_frame"]),
        "initial_critic_obs": np.asarray(carried_probe["critic_obs"]),
        "initial_carried_action": np.asarray(carried_probe["action"]),
        "initial_repeated_current_action": np.asarray(repeated_probe["action"]),
        "initial_online_value": np.asarray(carried_probe["online_value"]),
        "initial_target_value": np.asarray(carried_probe["target_value"]),
    }
    for name, trace in traces.items():
        for field, value in trace.items():
            arrays[f"{name}_{field}"] = np.asarray(value)
        for field, value in derived[name].items():
            arrays[f"{name}_{field}"] = np.asarray(value)

    result = {
        "protocol": "g1-e002-policy-memory-alias-audit-v1",
        "valid": True,
        "classification": classification,
        "optimizer_updates": 0,
        "environment_steps_retained": 0,
        "policy_retained": False,
        "retained_policy": None,
        "source_step": START_STEP,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_hparams_sha256": SOURCE_HPARAMS_SHA256,
        "source_actor_bootstrap_scale": float(hparams["actor_bootstrap_scale"]),
        "population_size": EFFECTIVE_NUM_ENVS,
        "rollout_horizon": horizon,
        "gamma": GAMMA,
        "history_treatment": {
            "actor_history_len": ACTOR_HISTORY_LEN,
            "actor_frame_dim": ACTOR_FRAME_DIM,
            "critic_obs_dim": CRITIC_OBS_DIM,
            "changed_fields": ["obs", "info.actor_obs_history"],
            "preserved_last_act": True,
            "preserved_full_nonhistory_info": True,
            "full_environment_reset": False,
            "older_history_different_env_fraction": older_different_fraction,
            "older_history_difference_rms": float(
                np.sqrt(np.mean(np.square(older_difference)))
            ),
        },
        "start_invariants": {
            "current_frame_exact": current_frame_exact,
            "critic_obs_exact": critic_obs_exact,
            "online_value_max_abs_delta": online_value_max_abs_delta,
            "target_value_max_abs_delta": target_value_max_abs_delta,
            "nonhistory_info_max_abs_delta": nonhistory_info_delta,
        },
        "e004_carried_control_parity": e004_parity,
        "start_action": action_metrics,
        "complete_return": return_metrics,
        "survival": survival_metrics,
        "arms": summaries,
        "selection_thresholds": {
            "minimum_action_relative_rms": MIN_ACTION_RELATIVE_RMS,
            "minimum_affected_env_fraction": MIN_AFFECTED_ENV_FRACTION,
            "per_env_action_rms_floor": PER_ENV_ACTION_RMS_FLOOR,
            "minimum_return_nrmse": MIN_RETURN_NRMSE,
            "minimum_survival_mae_transitions": MIN_SURVIVAL_MAE,
            "action_gate_combination": "relative-rms OR affected-env-fraction",
            "outcome_gate_combination": "return-nrmse OR survival-mae",
        },
        "interpretation_boundary": (
            "This controlled counterfactual tests whether omitted older actor "
            "history can change the retained policy and its complete first-done "
            "outcome while the start critic input/value stays fixed. It does not "
            "show that repeated history is a better policy input, estimate the "
            "frequency of identical physical states with distinct histories, "
            "train a history critic, or establish AHAC improvement."
        ),
    }
    return arrays, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("E002 policy-memory alias seed must equal zero")
    configure_jax()
    runtime = _validate_runtime()
    repository = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    calibration_trace = args.calibration_trace.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=args.code_commit,
    )
    if (
        not calibration_trace.is_file()
        or sha256_file(calibration_trace) != E004_CALIBRATION_TRACE_SHA256
    ):
        raise ValueError("E004 calibration trace SHA-256 mismatch")
    preflight.update(
        protocol="g1-e002-policy-memory-alias-preflight-v1",
        runtime=runtime,
        calibration_trace_path=str(calibration_trace),
        calibration_trace_sha256=E004_CALIBRATION_TRACE_SHA256,
        optimizer_updates=0,
        environment_steps_retained=0,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    arrays, result = collect_policy_memory_alias(
        checkpoint=checkpoint,
        reference=reference,
        calibration_trace=calibration_trace,
    )
    trace_path = output_root / "policy_memory_alias_trace.npz"
    result_path = output_root / "policy_memory_alias_audit.json"
    _atomic_npz(trace_path, arrays)
    result["trace_sha256"] = sha256_file(trace_path)
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": "g1-e002-policy-memory-alias-completion-v1",
        "valid": True,
        "classification": result["classification"],
        "optimizer_updates": 0,
        "environment_steps_retained": 0,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "policy_memory_alias_trace.npz": sha256_file(trace_path),
            "policy_memory_alias_audit.json": sha256_file(result_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(output_root)


if __name__ == "__main__":
    main()
