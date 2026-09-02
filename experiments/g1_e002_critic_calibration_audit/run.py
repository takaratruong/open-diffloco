"""Compare retained E002 critics with complete first-terminal returns.

The exact carried effective-512 state bank is rolled out twice in separately
compiled graphs: once with the deterministic retained actor and once with the
checkpoint's current training action-noise distribution.  Rewards after the
first done transition are excluded even though the environment auto-resets.
No parameter, optimizer, normalizer, or environment state is retained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.core.data_structures import Normalizer
from src.core.networks import Critic
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_flax_phase_grid import (
    load_checkpoint_environment_contract,
)
from tools.evaluate_g1_terminal_value_calibration import calibration_metrics
from tools.evaluate_g1_tracking import (
    _load_policy,
    make_evaluation_env,
    training_action_noise_at_step,
)
from tools.run_g1_dual_scale_root_position import (
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_HPARAMS_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically

START_STEP = 1_867_776
EFFECTIVE_NUM_ENVS = 512
GAMMA = 0.99
BOUNDARY_INDEX = 24
MIN_RANK_CORRELATION = 0.8
MAX_NRMSE = 0.25
MIGRATION_VALUES_SHA256 = (
    "b295cde07f29b363e0dec31690eeae597eef6313c45e50e54ce75969b1548d0c"
)
DETERMINISTIC_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops"


def first_terminal_returns(
    rewards: np.ndarray, dones: np.ndarray, *, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-state discounted sums ending at each column's first done."""

    reward = np.asarray(rewards, dtype=np.float64)
    done = np.asarray(dones, dtype=bool)
    if (
        reward.ndim != 2
        or reward.shape != done.shape
        or reward.size < 1
        or not np.isfinite(reward).all()
        or not math.isfinite(gamma)
        or not 0.0 <= gamma <= 1.0
    ):
        raise ValueError("reward/done trace is invalid")
    alive = np.ones(done.shape, dtype=bool)
    for index in range(1, done.shape[0]):
        alive[index] = alive[index - 1] & ~done[index - 1]
    realized = np.zeros_like(reward)
    tail = np.zeros((reward.shape[1],), dtype=np.float64)
    for index in range(reward.shape[0] - 1, -1, -1):
        tail = np.where(
            alive[index], reward[index] + gamma * tail, 0.0
        )
        realized[index] = tail
    return realized, alive


def _predictor_status(
    aggregate: Mapping[str, float], boundary: Mapping[str, float]
) -> str:
    predictive = (
        float(aggregate["rank_correlation"]) >= MIN_RANK_CORRELATION
        and float(boundary["rank_correlation"]) >= MIN_RANK_CORRELATION
    )
    calibrated = (
        predictive
        and float(aggregate["nrmse"]) <= MAX_NRMSE
        and float(boundary["nrmse"]) <= MAX_NRMSE
    )
    if calibrated:
        return "calibrated"
    if predictive:
        return "rank-predictive-miscalibrated"
    return "not-predictive"


def summarize_calibration_mode(
    online_values: np.ndarray,
    target_values: np.ndarray,
    realized_returns: np.ndarray,
    alive: np.ndarray,
    *,
    boundary_index: int = BOUNDARY_INDEX,
) -> dict[str, object]:
    """Summarize scalar calibration over all alive rows and the H24 boundary."""

    online = np.asarray(online_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    realized = np.asarray(realized_returns, dtype=np.float64)
    active = np.asarray(alive, dtype=bool)
    if (
        online.ndim != 2
        or target.shape != online.shape
        or realized.shape != online.shape
        or active.shape != online.shape
        or not all(
            np.isfinite(value).all() for value in (online, target, realized)
        )
        or boundary_index < 0
        or boundary_index >= online.shape[0]
        or np.sum(active) < 2
        or np.sum(active[boundary_index]) < 2
    ):
        raise ValueError("calibration mode arrays are invalid")

    def metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        return calibration_metrics(values[mask], realized[mask])

    aggregate_online = metrics(online, active)
    aggregate_target = metrics(target, active)
    boundary_mask = np.zeros_like(active)
    boundary_mask[boundary_index] = active[boundary_index]
    boundary_online = metrics(online, boundary_mask)
    boundary_target = metrics(target, boundary_mask)
    return {
        "protocol": "g1-e002-critic-calibration-mode-v1",
        "online_status": _predictor_status(
            aggregate_online, boundary_online
        ),
        "aggregate": {
            "online": aggregate_online,
            "legacy_delayed_target": aggregate_target,
        },
        "h24_boundary": {
            "transition": boundary_index,
            "active_count": int(np.sum(active[boundary_index])),
            "online": boundary_online,
            "legacy_delayed_target": boundary_target,
        },
    }


def classify_calibration_modes(
    modes: Mapping[str, Mapping[str, object]],
) -> str:
    """Require the online critic to pass under both registered policies."""

    if set(modes) != {"deterministic", "training_noise"}:
        raise ValueError("both registered calibration modes are required")
    statuses = [str(modes[name]["online_status"]) for name in sorted(modes)]
    if statuses == ["calibrated", "calibrated"]:
        return "online-critic-calibrated-for-ahac-bootstrap"
    if statuses == [
        "rank-predictive-miscalibrated",
        "rank-predictive-miscalibrated",
    ]:
        return "online-critic-rank-predictive-but-miscalibrated"
    return "online-critic-not-predictive"


def execute_separately_compiled_rollouts(
    deterministic_rollout,
    training_noise_rollout,
    initial_state,
    action_noise,
    *,
    compile_fn=jax.jit,
):
    """Keep the two policy distributions in distinct producing XLA graphs."""

    compiled_deterministic = compile_fn(deterministic_rollout)
    compiled_training_noise = compile_fn(training_noise_rollout)
    return (
        compiled_deterministic(initial_state, action_noise),
        compiled_training_noise(initial_state, action_noise),
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("distribution must be finite and nonempty")
    quantiles = np.percentile(array, (0, 10, 25, 50, 75, 90, 100))
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "max"),
            quantiles,
            strict=True,
        )
    } | {"mean": float(np.mean(array)), "std": float(np.std(array))}


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _validate_runtime() -> dict[str, object]:
    if DETERMINISTIC_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise ValueError("calibration audit requires deterministic XLA")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("calibration audit requires exactly one visible GPU")
    if not bool(jax.config.jax_enable_x64):
        raise ValueError("calibration audit requires JAX float64 mode")
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        "jax_enable_x64": True,
        "xla_deterministic_reductions": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _make_environment(hparams: Mapping[str, object], reference: Path):
    return make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        body_mass_scale=float(hparams["mass_range"][0]),
        effort_limit_scale=float(hparams["effort_limit_scale"]),
        reference_path=reference,
        reference_stride=int(hparams["reference_stride"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            int(value) for value in hparams["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(hparams["actor_reference_preview_mode"]),
        actor_observe_motion_anchor_position=bool(
            hparams.get("actor_observe_motion_anchor_position", False)
        ),
        tracking_velocity_kernel=str(hparams["tracking_velocity_kernel"]),
        tracking_anchor_position_kernel=str(
            hparams.get("tracking_anchor_position_kernel") or "exponential"
        ),
        tracking_torso_orientation_weight=float(
            hparams["tracking_torso_orientation_weight"]
        ),
        tracking_root_velocity_weight=float(hparams["tracking_root_velocity_weight"]),
        actor_observation_noise=bool(hparams["actor_observation_noise"]),
        domain_randomization=bool(hparams["domain_randomization"]),
        friction_range=tuple(float(value) for value in hparams["friction_range"]),
        kp_range=tuple(float(value) for value in hparams["kp_range"]),
        kd_range=tuple(float(value) for value in hparams["kd_range"]),
        com_offset_range=tuple(
            float(value) for value in hparams["com_offset_range"]
        ),
        reference_reset_noise_scale=float(hparams["reference_reset_noise_scale"]),
        reference_residual_control=bool(hparams["reference_residual_control"]),
        reference_residual_scale=float(hparams["reference_residual_scale"]),
        contact_stiffness_metric="root_generalized",
    )


def collect_calibration_traces(
    *, checkpoint: Path, reference: Path, migration_values: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Execute both complete first-done policy distributions from E002."""

    hparams = json.loads(
        checkpoint.with_name("hparams.json").read_text(encoding="utf-8")
    )
    contract = load_checkpoint_environment_contract(checkpoint)
    if (
        int(hparams["effective_num_envs"]) != EFFECTIVE_NUM_ENVS
        or int(hparams["unroll_length"]) != BOUNDARY_INDEX
        or hparams.get("ahac") is not False
        or float(hparams.get("actor_bootstrap_scale", -1.0)) != 0.0
        or contract["env_variant"] != hparams["env_variant"]
        or float(hparams["zero_difficulty_frac"]) != 1.0
        or bool(hparams["terrain"])
        or bool(hparams["torso_wrench_assistance"])
        or bool(hparams["actor_learned_torso_wrench"])
    ):
        raise ValueError("retained E002 boundary does not match calibration")
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != START_STEP
        or np.asarray(state.env_state.obs).shape[0] != EFFECTIVE_NUM_ENVS
    ):
        raise ValueError("retained E002 checkpoint state is invalid")

    environment = _make_environment(hparams, reference)
    horizon = int(environment.max_episode_length)
    if horizon < BOUNDARY_INDEX + 1:
        raise ValueError("environment episode horizon is too short")
    actor, actor_params, actor_normalizer_state = _load_policy(
        environment, checkpoint, 0
    )
    actor_normalizer = Normalizer(environment.actor_frame_obs_dim)
    critic_normalizer = Normalizer(environment.critic_obs_dim)
    critic = Critic()
    current_noise_std = jnp.asarray(
        training_action_noise_at_step(
            hparams, START_STEP, action_dim=environment.action_dim
        ),
        dtype=jnp.float64,
    )
    split_keys = jax.random.split(state.key, 6)
    action_noise = jax.random.normal(
        split_keys[1],
        (EFFECTIVE_NUM_ENVS, horizon, environment.action_dim),
        dtype=jnp.float64,
    )
    scan_noise = jnp.swapaxes(action_noise, 0, 1)

    def prepare_action(env_state, epsilon, *, add_noise: bool):
        rng_pairs = jax.vmap(lambda key: jax.random.split(key, 2))(
            env_state.info["rng"]
        )
        obs_rng = rng_pairs[:, 0]
        env_rng = rng_pairs[:, 1]
        env_state = env_state.replace(info={**env_state.info, "rng": env_rng})
        actor_obs = jax.vmap(environment._apply_obs_noise)(
            env_state.obs, obs_rng
        )
        normalized = environment.normalize_actor_obs(
            actor_normalizer,
            actor_normalizer_state,
            actor_obs,
        ).astype(jnp.float32)
        action = jax.vmap(
            lambda observation: actor.apply(actor_params, observation)
        )(normalized).astype(jnp.float64)
        if add_noise:
            action = action + epsilon * current_noise_std
        if environment.clip_sampled_actor_actions:
            action = jnp.clip(action, -1.0, 1.0)
        return env_state, action

    def make_rollout(*, add_noise: bool):
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
                env_state, action = prepare_action(
                    env_state, epsilon, add_noise=add_noise
                )
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

    deterministic_rollout = make_rollout(add_noise=False)
    training_noise_rollout = make_rollout(add_noise=True)
    profile = get_solver_profile(str(hparams["solver_profile"]))
    with solver_context(profile):
        (_, deterministic), (_, noisy) = execute_separately_compiled_rollouts(
            deterministic_rollout,
            training_noise_rollout,
            state.env_state,
            scan_noise,
        )
    traces = {
        "deterministic": jax.tree.map(np.asarray, deterministic),
        "training_noise": jax.tree.map(np.asarray, noisy),
    }
    arrays: dict[str, np.ndarray] = {
        "action_noise": np.asarray(action_noise),
        "action_noise_std": np.asarray(current_noise_std),
        "start_phase": np.asarray(state.env_state.info["phase"]),
        "checkpoint_key": np.asarray(state.key),
    }
    summaries: dict[str, dict[str, object]] = {}
    with np.load(migration_values, allow_pickle=False) as source_values:
        source_online = np.asarray(source_values["online_values"])
        source_target = np.asarray(
            source_values["legacy_delayed_target_values"]
        )
        source_phase = np.asarray(source_values["phase"])
    for name, trace in traces.items():
        if (
            not bool(np.all(trace["finite"]))
            or float(np.max(trace["xfrc_max"])) != 0.0
            or not np.isfinite(trace["reward"]).all()
        ):
            raise ValueError(f"{name} rollout is nonfinite or assisted")
        done = np.asarray(trace["done"], dtype=bool)
        realized, alive = first_terminal_returns(
            trace["reward"], done, gamma=GAMMA
        )
        completed = np.any(done, axis=0)
        if not np.all(completed):
            raise ValueError(f"{name} did not reach first done for every state")
        first_done = np.argmax(done, axis=0) + 1
        first_done_rows = first_done - 1
        columns = np.arange(EFFECTIVE_NUM_ENVS)
        first_terminal = np.asarray(trace["terminal"], dtype=bool)[
            first_done_rows, columns
        ]
        summaries[name] = summarize_calibration_mode(
            trace["online_value"],
            trace["target_value"],
            realized,
            alive,
        )
        summaries[name].update(
            survival=_distribution(first_done),
            natural_terminal_count=int(np.sum(first_terminal)),
            truncation_count=int(np.sum(~first_terminal)),
            reward=_distribution(trace["reward"][alive]),
            return_from_start=_distribution(realized[0]),
            post_first_done_rewards_masked=True,
        )
        for field, value in trace.items():
            arrays[f"{name}_{field}"] = np.asarray(value)
        arrays[f"{name}_alive"] = alive
        arrays[f"{name}_realized_return"] = realized
        arrays[f"{name}_first_done"] = first_done
        arrays[f"{name}_first_done_terminal"] = first_terminal

    start_parity = {
        "phase_exact": bool(
            np.array_equal(arrays["start_phase"], source_phase)
        ),
        "deterministic_online_max_abs_delta": float(
            np.max(
                np.abs(
                    traces["deterministic"]["online_value"][0]
                    - source_online
                )
            )
        ),
        "deterministic_target_max_abs_delta": float(
            np.max(
                np.abs(
                    traces["deterministic"]["target_value"][0]
                    - source_target
                )
            )
        ),
        "mode_online_max_abs_delta": float(
            np.max(
                np.abs(
                    traces["deterministic"]["online_value"][0]
                    - traces["training_noise"]["online_value"][0]
                )
            )
        ),
        "mode_target_max_abs_delta": float(
            np.max(
                np.abs(
                    traces["deterministic"]["target_value"][0]
                    - traces["training_noise"]["target_value"][0]
                )
            )
        ),
    }
    if not start_parity["phase_exact"] or any(
        float(value) > 1e-6
        for key, value in start_parity.items()
        if key != "phase_exact"
    ):
        raise ValueError("calibration start values do not close to E003")
    result = {
        "protocol": "g1-e002-carried-critic-calibration-audit-v1",
        "valid": True,
        "classification": classify_calibration_modes(summaries),
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
        "boundary_index": BOUNDARY_INDEX,
        "selection_thresholds": {
            "minimum_rank_correlation": MIN_RANK_CORRELATION,
            "maximum_nrmse": MAX_NRMSE,
            "must_pass_both_modes": True,
        },
        "start_value_parity": start_parity,
        "modes": summaries,
        "phase_population": {
            "minimum": int(np.min(arrays["start_phase"])),
            "maximum": int(np.max(arrays["start_phase"])),
            "unique_count": int(np.unique(arrays["start_phase"]).size),
        },
        "interpretation_boundary": (
            "Scalar carried-return calibration determines whether the retained "
            "online critic can enter an AHAC bootstrap gate. It does not assess "
            "critic state-gradient alignment, train a second head, or improve "
            "closed-loop behavior."
        ),
    }
    return arrays, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--migration-values", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("E002 critic calibration seed must equal zero")
    configure_jax()
    runtime = _validate_runtime()
    repository = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    migration_values = args.migration_values.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=args.code_commit,
    )
    if (
        not migration_values.is_file()
        or sha256_file(migration_values) != MIGRATION_VALUES_SHA256
    ):
        raise ValueError("E003 migration values SHA-256 mismatch")
    preflight.update(
        protocol="g1-e002-carried-critic-calibration-preflight-v1",
        runtime=runtime,
        migration_values_path=str(migration_values),
        migration_values_sha256=MIGRATION_VALUES_SHA256,
        optimizer_updates=0,
        environment_steps_retained=0,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    arrays, result = collect_calibration_traces(
        checkpoint=checkpoint,
        reference=reference,
        migration_values=migration_values,
    )
    trace_path = output_root / "critic_calibration_trace.npz"
    result_path = output_root / "critic_calibration_audit.json"
    _atomic_npz(trace_path, arrays)
    result["trace_sha256"] = sha256_file(trace_path)
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": "g1-e002-carried-critic-calibration-completion-v1",
        "valid": True,
        "classification": result["classification"],
        "optimizer_updates": 0,
        "environment_steps_retained": 0,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "critic_calibration_trace.npz": sha256_file(trace_path),
            "critic_calibration_audit.json": sha256_file(result_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(output_root)


if __name__ == "__main__":
    main()
