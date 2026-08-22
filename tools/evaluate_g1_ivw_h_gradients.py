"""Evaluate phase-local IVW-H gradient reliability for G1 SHAC walking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np


PROTOCOL = "g1-ivw-h-gradient-v1"
POPULATION = 120
PHASES = np.asarray((0, 25, 50, 75, 100), dtype=np.int32)
REPLICAS_PER_PHASE = 24
HORIZON = 24
ACTION_DIM = 29
SOLVERS = ("g1-4x5", "diagnostic-10x20")
ACTORS = ("fresh", "e023")
ESTIMATORS = ("ordinary", "score", "ivw_h")
TAPE_SEEDS = (913_024, 913_025)
EXPECTED_INPUT_SHA256 = {
    "checkpoint": "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f",
    "hparams": "a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650",
    "reference": "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca",
}
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
REGISTERED_OUTCOMES = frozenset(
    {"ivw-h-robust", "ivw-h-neutral", "ivw-h-destructive", "invalid-execution"}
)


def build_fixed_phase_population(seed: int) -> dict[str, np.ndarray]:
    """Return the exact E005 starts and two distinct fixed H24 noise tapes."""

    if seed != 0:
        raise ValueError("IVW-H gradient seed must be zero")
    phase = np.repeat(PHASES, REPLICAS_PER_PHASE)
    noise = np.stack(
        [
            np.random.default_rng(tape_seed).standard_normal(
                (POPULATION, HORIZON, ACTION_DIM), dtype=np.float32
            )
            for tape_seed in TAPE_SEEDS
        ]
    )
    return {"phase": phase, "noise": noise}


def push_action_gradients_to_policy(
    actor_apply: Callable[[Any, Any], Any],
    parameters: Any,
    observations: Any,
    action_gradients: Any,
) -> Any:
    """Push stopped action-node gradients through actor means only."""

    import jax
    import jax.numpy as jnp

    observations = jax.lax.stop_gradient(jnp.asarray(observations))
    action_gradients = jax.lax.stop_gradient(jnp.asarray(action_gradients))

    def surrogate(candidate_parameters):
        means = actor_apply(candidate_parameters, observations)
        if means.shape != action_gradients.shape:
            raise ValueError("actor means and action gradients must share shape")
        return jnp.sum(means * action_gradients)

    return jax.grad(surrogate)(parameters)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_actor_row(row: object) -> bool:
    if not isinstance(row, Mapping):
        return False
    scalars = (
        "pathwise_vjp_cosine_min",
        "pathwise_vjp_norm_ratio_min",
        "pathwise_vjp_norm_ratio_max",
        "finite_phase_count_min",
        "ordinary_mean_solver_cosine",
        "ivw_h_mean_solver_cosine",
        "ordinary_mean_tape_cosine",
        "ivw_h_mean_tape_cosine",
        "retained_pathwise_cosine",
        "retained_pathwise_norm_ratio",
    )
    vectors = (
        "ordinary_phase_solver_cosine",
        "ivw_h_phase_solver_cosine",
        "ordinary_phase_tape_cosine",
        "ivw_h_phase_tape_cosine",
    )
    if any(not _finite_number(row.get(name)) for name in scalars):
        return False
    for name in vectors:
        value = row.get(name)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != len(PHASES)
            or any(not _finite_number(item) for item in value)
        ):
            return False
    return (
        float(row["pathwise_vjp_cosine_min"]) >= 0.999
        and 0.999 <= float(row["pathwise_vjp_norm_ratio_min"])
        and float(row["pathwise_vjp_norm_ratio_max"]) <= 1.001
        and int(row["finite_phase_count_min"]) >= 16
    )


def classify_ivw_h_gradient_audit(evidence: Mapping[str, object]) -> str:
    """Apply invalid/destructive/robust/neutral outcome precedence."""

    actors = evidence.get("actors")
    if evidence.get("valid") is not True or not isinstance(actors, Mapping):
        return "invalid-execution"
    rows: list[Mapping[str, object]] = []
    for actor in ACTORS:
        row = actors.get(actor)
        if not _valid_actor_row(row):
            return "invalid-execution"
        assert isinstance(row, Mapping)
        rows.append(row)

    destructive = any(
        float(row["retained_pathwise_cosine"]) < 0.5
        or not 0.25 <= float(row["retained_pathwise_norm_ratio"]) <= 4.0
        or any(
            float(fused) < float(ordinary) - 0.05
            for fused, ordinary in zip(
                row["ivw_h_phase_solver_cosine"],
                row["ordinary_phase_solver_cosine"],
            )
        )
        or any(
            float(fused) < float(ordinary) - 0.05
            for fused, ordinary in zip(
                row["ivw_h_phase_tape_cosine"],
                row["ordinary_phase_tape_cosine"],
            )
        )
        for row in rows
    )
    if destructive:
        return "ivw-h-destructive"

    robust = all(
        float(row["ivw_h_mean_solver_cosine"])
        >= float(row["ordinary_mean_solver_cosine"]) + 0.05
        and float(row["ivw_h_mean_tape_cosine"])
        >= float(row["ordinary_mean_tape_cosine"]) + 0.05
        for row in rows
    )
    return "ivw-h-robust" if robust else "ivw-h-neutral"


def validate_completion(path: Path) -> dict[str, object]:
    """Reopen completion and hash-validate every bound artifact."""

    from tools.prepare_g1_rmr_reference import sha256_file

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("valid") is not True
        or payload.get("protocol") != PROTOCOL
        or payload.get("outcome") not in REGISTERED_OUTCOMES
    ):
        raise ValueError("completion contract is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("completion artifacts are missing")
    for name, expected in artifacts.items():
        artifact = path.parent / str(name)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"artifact hash mismatch for {name}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _tree_vector(tree: Any) -> np.ndarray:
    import jax

    leaves = [np.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        raise ValueError("gradient tree must be nonempty")
    return np.concatenate(leaves).astype(np.float64, copy=False)


def _tree_matrix(tree: Any) -> np.ndarray:
    import jax

    leaves = [
        np.asarray(leaf).reshape(np.asarray(leaf).shape[0], -1)
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    if not leaves:
        raise ValueError("gradient tree must be nonempty")
    return np.concatenate(leaves, axis=1).astype(np.float64, copy=False)


def _vector_cosine(left: Any, right: Any) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("gradient vectors must be finite")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("gradient vectors must be nonzero")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _validate_clean_source(repository: Path, code_commit: str) -> dict[str, str]:
    if len(code_commit) != 40:
        raise ValueError("code commit must be a full SHA-1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tools"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != code_commit or dirty:
        raise ValueError("IVW-H evaluator requires exact clean source")
    return {
        "code_commit": head,
        "dirty_patch_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _concat_trees(chunks: list[Any]) -> Any:
    import jax

    return jax.tree_util.tree_map(
        lambda *values: np.concatenate(values, axis=0), *chunks
    )


def _action_standard_deviation(
    value: object, *, action_dim: int = ACTION_DIM
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full((action_dim,), float(array), dtype=np.float64)
    if array.shape != (action_dim,) or not np.isfinite(array).all():
        raise ValueError("action standard deviation must be finite with shape (29,)")
    if np.any(array <= 0.0):
        raise ValueError("action standard deviation must be positive")
    return array


def _parity_statistics(direct: Any, pushed: Any) -> dict[str, object]:
    direct_matrix = _tree_matrix(direct)
    pushed_matrix = _tree_matrix(pushed)
    if direct_matrix.shape != pushed_matrix.shape:
        raise ValueError("direct and pushed gradient matrices differ")
    direct_norm = np.linalg.norm(direct_matrix, axis=1)
    pushed_norm = np.linalg.norm(pushed_matrix, axis=1)
    finite = (
        np.isfinite(direct_matrix).all(axis=1)
        & np.isfinite(pushed_matrix).all(axis=1)
        & (direct_norm > 0.0)
        & (pushed_norm > 0.0)
    )
    cosine = np.full(direct_norm.shape, np.nan, dtype=np.float64)
    ratio = np.full(direct_norm.shape, np.nan, dtype=np.float64)
    cosine[finite] = np.sum(
        direct_matrix[finite] * pushed_matrix[finite], axis=1
    ) / (direct_norm[finite] * pushed_norm[finite])
    ratio[finite] = pushed_norm[finite] / direct_norm[finite]
    return {"finite": finite, "cosine": cosine, "norm_ratio": ratio}


def _capture_one_population(
    *,
    env: Any,
    actor: Any,
    parameters: Any,
    normalizer: Any,
    normalizer_state: Any,
    states: Any,
    epsilon: np.ndarray,
    phases: np.ndarray,
    sigma: np.ndarray,
    gamma: float,
    chunk_size: int,
    gradient_env: Any | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.gradients import per_env_gradient_statistics
    from src.algorithms.shac.contact_compliance import backward_from_compliant
    from src.algorithms.shac.ivw_h import (
        discounted_reward_to_go,
        fuse_action_gradients,
        gaussian_mean_score_gradients,
        leave_one_out_phase_advantages,
        phase_step_action_ivw,
    )

    count = int(len(phases))
    sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)

    def loss(candidate_parameters, initial_state, epsilon_i, delta):
        def rollout_step(state, values):
            epsilon_t, delta_t = values
            _obs_key, env_key = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_key})
            normalized_obs = env.normalize_actor_obs(
                normalizer, normalizer_state, state.obs
            ).astype(jnp.float32)
            mean = actor.apply(candidate_parameters, normalized_obs).astype(
                jnp.float64
            )
            sampled_action = mean + epsilon_t.astype(jnp.float64) * sigma_jax
            effective_action = sampled_action + delta_t
            hard_next_state = env.step(state, effective_action)
            next_state = hard_next_state
            if gradient_env is not None:
                compliant_next_state = gradient_env.step(state, effective_action)
                next_state = backward_from_compliant(
                    hard_next_state, compliant_next_state
                )
            return next_state, {
                "reward": next_state.reward,
                "done": next_state.done,
                "terminal": next_state.info["terminal"],
                "normalized_obs": normalized_obs,
                "mean": mean,
                "sampled_action": sampled_action,
                "qpos": next_state.data.qpos,
                "qvel": next_state.data.qvel,
            }

        _, trajectory = jax.lax.scan(
            rollout_step,
            initial_state,
            (epsilon_i, delta),
            length=HORIZON,
        )

        def accumulate(carry, values):
            total, running, discount = carry
            reward, done = values
            running = running + discount * reward
            total = total + jnp.where(done, running, 0.0)
            return (
                total,
                jnp.where(done, 0.0, running),
                jnp.where(done, 1.0, discount * gamma),
            ), None

        (total, running, _), _ = jax.lax.scan(
            accumulate,
            (jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0)),
            (trajectory["reward"], trajectory["done"]),
        )
        return -(total + running) / HORIZON, trajectory

    capture_fn = jax.jit(
        jax.vmap(
            jax.value_and_grad(loss, argnums=(0, 3), has_aux=True),
            in_axes=(None, 0, 0, 0),
        )
    )

    direct_chunks: list[Any] = []
    action_gradient_chunks: list[np.ndarray] = []
    auxiliary_chunks: list[Any] = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        state_chunk = jax.tree_util.tree_map(lambda value: value[start:stop], states)
        delta = jnp.zeros((stop - start, HORIZON, ACTION_DIM), dtype=jnp.float64)
        (losses_aux, gradients) = capture_fn(
            parameters,
            state_chunk,
            jnp.asarray(epsilon[start:stop]),
            delta,
        )
        losses, auxiliary = losses_aux
        direct_gradient, action_gradient = gradients
        direct_chunks.append(jax.device_get(direct_gradient))
        action_gradient_chunks.append(np.asarray(jax.device_get(action_gradient)))
        auxiliary_chunks.append(
            jax.device_get({**auxiliary, "loss": losses})
        )

    direct_gradients = _concat_trees(direct_chunks)
    pathwise_action = np.concatenate(action_gradient_chunks, axis=0)
    auxiliary = _concat_trees(auxiliary_chunks)
    observations = np.asarray(auxiliary["normalized_obs"])
    means = np.asarray(auxiliary["mean"])
    sampled_actions = np.asarray(auxiliary["sampled_action"])
    rewards = np.asarray(auxiliary["reward"])
    dones = np.asarray(auxiliary["done"], dtype=np.bool_)

    reward_to_go = np.asarray(
        discounted_reward_to_go(rewards, dones, gamma=gamma)
    )
    advantages = np.asarray(
        leave_one_out_phase_advantages(reward_to_go, phases)
    )
    score_action = np.asarray(
        gaussian_mean_score_gradients(
            means,
            sampled_actions,
            advantages,
            sigma,
            horizon=HORIZON,
        )
    )
    pathwise_weight = np.asarray(
        phase_step_action_ivw(score_action, pathwise_action, phases)
    )
    fused_action = np.asarray(
        fuse_action_gradients(pathwise_action, score_action, pathwise_weight)
    )

    def push_batch(action_gradient: np.ndarray) -> Any:
        push_fn = jax.jit(
            jax.vmap(
                lambda obs_i, gradient_i: push_action_gradients_to_policy(
                    actor.apply,
                    parameters,
                    obs_i,
                    gradient_i,
                ),
                in_axes=(0, 0),
            )
        )
        chunks = []
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            chunks.append(
                jax.device_get(
                    push_fn(
                        jnp.asarray(observations[start:stop]),
                        jnp.asarray(action_gradient[start:stop]),
                    )
                )
            )
        return _concat_trees(chunks)

    pushed_pathwise = push_batch(pathwise_action)
    score_gradients = push_batch(score_action)
    fused_gradients = push_batch(fused_action)
    parity = _parity_statistics(direct_gradients, pushed_pathwise)

    gradients = {
        "ordinary": direct_gradients,
        "score": score_gradients,
        "ivw_h": fused_gradients,
    }
    finite_by_estimator = {
        name: np.asarray(
            per_env_gradient_statistics(value)["finite_by_env"], dtype=np.bool_
        )
        for name, value in gradients.items()
    }
    finite_phase_counts = {
        name: [
            int(np.sum(finite[phases == phase]))
            for phase in PHASES
            if np.any(phases == phase)
        ]
        for name, finite in finite_by_estimator.items()
    }
    return {
        "gradients": gradients,
        "finite_phase_counts": finite_phase_counts,
        "parity": parity,
        "auxiliary": auxiliary,
        "pathwise_action_gradient": pathwise_action,
        "score_action_gradient": score_action,
        "ivw_h_action_gradient": fused_action,
        "pathwise_weight": pathwise_weight,
        "reward_to_go": reward_to_go,
        "advantage": advantages,
        "means": means,
        "sampled_actions": sampled_actions,
    }


def run_gradient_capture(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    output_directory: Path,
    repository: Path,
    code_commit: str,
    seed: int,
    smoke: bool = False,
    smoke_phase: int = 25,
    smoke_replicas: int = 8,
) -> dict[str, object]:
    """Run the registered fresh/E023 paired-solver, paired-tape audit."""

    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.gradients import per_env_gradient_statistics
    from src.algorithms.shac.objective_direction_audit import aggregate_audit_direction
    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.build_g1_e023_carried_reset_bank import validate_e023_hparams
    from tools.evaluate_g1_tracking import _load_policy
    from tools.prepare_g1_rmr_reference import sha256_file
    from tools.run_g1_action_sequence_recovery_oracle import _build_environment
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets

    if seed != 0:
        raise ValueError("IVW-H gradient seed must be zero")
    if smoke_replicas < 2 or smoke_replicas > REPLICAS_PER_PHASE:
        raise ValueError("smoke replicas must be in [2, 24]")
    source = _validate_clean_source(repository.resolve(), code_commit)
    paths = {
        "checkpoint": checkpoint_path.resolve(),
        "hparams": hparams_path.resolve(),
        "reference": reference_path.resolve(),
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("IVW-H inputs do not match E023")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    if hparams.get("clip_sampled_actor_actions") is not False:
        raise ValueError("IVW-H requires the exact unclipped E023 action contract")
    if float(hparams.get("actor_bootstrap_scale", math.nan)) != 0.0:
        raise ValueError("IVW-H requires the exact zero-bootstrap E023 objective")
    runtime = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    if (
        runtime["model_sha256"] != EXPECTED_MODEL_SHA256
        or runtime["controller_sha256"] != EXPECTED_CONTROLLER_SHA256
    ):
        raise ValueError("IVW-H runtime assets do not match E023")
    input_hashes.update(
        model=runtime["model_sha256"], controller=runtime["controller_sha256"]
    )

    population = build_fixed_phase_population(seed)
    phases = population["phase"]
    noise = population["noise"]
    if smoke:
        selected = np.flatnonzero(phases == smoke_phase)[:smoke_replicas]
        if selected.size != smoke_replicas:
            raise ValueError("smoke phase does not contain enough replicas")
        phases = phases[selected]
        noise = noise[:, selected]
    count = int(len(phases))
    keys = jax.random.split(jax.random.PRNGKey(seed), count)

    captures: dict[tuple[str, str, int], dict[str, Any]] = {}
    initial_arrays: dict[str, np.ndarray] | None = None
    actor_parameters: dict[str, Any] = {}
    solver_names = SOLVERS[:1] if smoke else SOLVERS
    actor_names = ("e023",) if smoke else ACTORS
    tape_indices = (0,) if smoke else (0, 1)

    for solver_name in solver_names:
        profile = get_solver_profile(solver_name)
        solver_hparams = {
            **hparams,
            "solver_iterations": profile.iterations,
            "solver_ls_iterations": profile.ls_iterations,
        }
        env = _build_environment(solver_hparams, reference_path)
        with solver_context(profile):
            states = jax.jit(jax.vmap(env.reset_at_phase))(
                keys,
                jnp.zeros((count,), dtype=jnp.float64),
                jnp.asarray(phases, dtype=jnp.int32),
            )
        current_initial = {
            "phase": np.asarray(states.info["phase"], dtype=np.int32),
            "qpos": np.asarray(states.data.qpos),
            "qvel": np.asarray(states.data.qvel),
            "history": np.asarray(states.info["actor_obs_history"]),
        }
        if initial_arrays is None:
            initial_arrays = current_initial
        else:
            for name, value in current_initial.items():
                if not np.array_equal(value, initial_arrays[name]):
                    raise ValueError("solver initial states are not bit-identical")

        e023_actor, e023_params, e023_norm = _load_policy(env, checkpoint_path, seed)
        fresh_actor, fresh_params, fresh_norm = _load_policy(
            env,
            None,
            seed,
            actor_hidden=tuple(hparams["actor_hidden"]),
            actor_layer_norm=bool(hparams["actor_layer_norm"]),
            actor_zero_output=bool(hparams["actor_zero_output"]),
            training_initialization=True,
        )
        actors = {"fresh": fresh_actor, "e023": e023_actor}
        params_by_actor = {"fresh": fresh_params, "e023": e023_params}
        norms_by_actor = {"fresh": fresh_norm, "e023": e023_norm}
        if not actor_parameters:
            actor_parameters = params_by_actor
        else:
            for actor_name in ACTORS:
                if not np.array_equal(
                    _tree_vector(params_by_actor[actor_name]),
                    _tree_vector(actor_parameters[actor_name]),
                ):
                    raise ValueError("actor parameters differ across solvers")

        normalizer = Normalizer(env.actor_frame_obs_dim)
        sigma_by_actor = {
            "fresh": _action_standard_deviation(hparams["action_noise_std_start"]),
            "e023": _action_standard_deviation(hparams["action_noise_std_end"]),
        }
        for actor_name in actor_names:
            for tape_index in tape_indices:
                with solver_context(profile):
                    capture = _capture_one_population(
                        env=env,
                        actor=actors[actor_name],
                        parameters=params_by_actor[actor_name],
                        normalizer=normalizer,
                        normalizer_state=norms_by_actor[actor_name],
                        states=states,
                        epsilon=noise[tape_index],
                        phases=phases,
                        sigma=sigma_by_actor[actor_name],
                        gamma=float(hparams["gamma"]),
                        chunk_size=(smoke_replicas if smoke else REPLICAS_PER_PHASE),
                    )
                captures[(actor_name, solver_name, tape_index)] = capture

    assert initial_arrays is not None
    if smoke:
        capture = captures[("e023", solver_names[0], 0)]
        finite_counts = capture["finite_phase_counts"]
        parity = capture["parity"]
        finite_parity = np.asarray(parity["finite"], dtype=np.bool_)
        report = {
            "valid": bool(
                all(min(counts) >= 2 for counts in finite_counts.values())
                and np.sum(finite_parity) >= 2
                and np.nanmin(parity["cosine"]) >= 0.999
                and np.nanmin(parity["norm_ratio"]) >= 0.999
                and np.nanmax(parity["norm_ratio"]) <= 1.001
            ),
            "scientific": False,
            "protocol": f"{PROTOCOL}-smoke",
            "phase": smoke_phase,
            "replicas": smoke_replicas,
            "finite_phase_counts": finite_counts,
            "pathwise_vjp_cosine_min": float(np.nanmin(parity["cosine"])),
            "pathwise_vjp_norm_ratio_min": float(
                np.nanmin(parity["norm_ratio"])
            ),
            "pathwise_vjp_norm_ratio_max": float(
                np.nanmax(parity["norm_ratio"])
            ),
            "pathwise_weight_mean": float(np.mean(capture["pathwise_weight"])),
            **source,
        }
        if not report["valid"]:
            raise ValueError(f"IVW-H smoke failed: {report}")
        _atomic_json(output_directory.resolve() / "smoke_summary.json", report)
        return report

    arrays: dict[str, np.ndarray] = {
        "phase": np.asarray(phases, dtype=np.int32),
        "noise": np.asarray(noise, dtype=np.float32),
        "initial_qpos": initial_arrays["qpos"],
        "initial_qvel": initial_arrays["qvel"],
        "initial_actor_obs_history": initial_arrays["history"],
    }
    aggregate_vectors: dict[tuple[str, str, int, str], np.ndarray] = {}
    task_vectors: dict[tuple[str, str, int, str], np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}

    for key, capture in captures.items():
        actor_name, solver_name, tape_index = key
        prefix = f"{actor_name}_{solver_name}_tape{tape_index}".replace("-", "_")
        parity = capture["parity"]
        arrays[f"{prefix}_pathwise_vjp_cosine"] = np.asarray(parity["cosine"])
        arrays[f"{prefix}_pathwise_vjp_norm_ratio"] = np.asarray(
            parity["norm_ratio"]
        )
        for name in (
            "pathwise_action_gradient",
            "score_action_gradient",
            "ivw_h_action_gradient",
            "pathwise_weight",
            "reward_to_go",
            "advantage",
            "means",
            "sampled_actions",
        ):
            arrays[f"{prefix}_{name}"] = np.asarray(capture[name])
        for estimator, gradients in capture["gradients"].items():
            aggregated = aggregate_audit_direction(
                gradients,
                phases,
                phase_count=125,
                clip_norm=1.0,
                alpha=0.5,
                iterations=32,
            )
            aggregate = _tree_vector(aggregated.combined_gradient)
            tasks = _tree_matrix(aggregated.task_gradients)
            aggregate_vectors[(*key, estimator)] = aggregate
            task_vectors[(*key, estimator)] = tasks
            arrays[f"{prefix}_{estimator}_combined"] = aggregate
            arrays[f"{prefix}_{estimator}_task"] = tasks
            stats = per_env_gradient_statistics(gradients)
            arrays[f"{prefix}_{estimator}_env_norm"] = np.asarray(
                stats["raw_norm_by_env"]
            )

    for actor_name in ACTORS:
        def aggregate_cosines(estimator: str):
            solver = [
                _vector_cosine(
                    aggregate_vectors[(actor_name, SOLVERS[0], tape, estimator)],
                    aggregate_vectors[(actor_name, SOLVERS[1], tape, estimator)],
                )
                for tape in (0, 1)
            ]
            tape = [
                _vector_cosine(
                    aggregate_vectors[(actor_name, solver_name, 0, estimator)],
                    aggregate_vectors[(actor_name, solver_name, 1, estimator)],
                )
                for solver_name in SOLVERS
            ]
            phase_solver = np.mean(
                [
                    [
                        _vector_cosine(
                            task_vectors[(actor_name, SOLVERS[0], tape_id, estimator)][phase_id],
                            task_vectors[(actor_name, SOLVERS[1], tape_id, estimator)][phase_id],
                        )
                        for phase_id in range(5)
                    ]
                    for tape_id in (0, 1)
                ],
                axis=0,
            )
            phase_tape = np.mean(
                [
                    [
                        _vector_cosine(
                            task_vectors[(actor_name, solver_name, 0, estimator)][phase_id],
                            task_vectors[(actor_name, solver_name, 1, estimator)][phase_id],
                        )
                        for phase_id in range(5)
                    ]
                    for solver_name in SOLVERS
                ],
                axis=0,
            )
            return solver, tape, phase_solver, phase_tape

        ordinary_solver, ordinary_tape, ordinary_phase_solver, ordinary_phase_tape = (
            aggregate_cosines("ordinary")
        )
        ivw_solver, ivw_tape, ivw_phase_solver, ivw_phase_tape = aggregate_cosines(
            "ivw_h"
        )
        parity_rows = [
            captures[(actor_name, solver_name, tape)]["parity"]
            for solver_name in SOLVERS
            for tape in (0, 1)
        ]
        parity_cosines = np.concatenate(
            [np.asarray(row["cosine"])[np.asarray(row["finite"])] for row in parity_rows]
        )
        parity_ratios = np.concatenate(
            [np.asarray(row["norm_ratio"])[np.asarray(row["finite"])] for row in parity_rows]
        )
        finite_counts = [
            count
            for solver_name in SOLVERS
            for tape in (0, 1)
            for counts in captures[(actor_name, solver_name, tape)][
                "finite_phase_counts"
            ].values()
            for count in counts
        ]
        ordinary_nominal = aggregate_vectors[
            (actor_name, SOLVERS[0], 0, "ordinary")
        ]
        ivw_nominal = aggregate_vectors[(actor_name, SOLVERS[0], 0, "ivw_h")]
        summaries[actor_name] = {
            "pathwise_vjp_cosine_min": float(np.min(parity_cosines)),
            "pathwise_vjp_norm_ratio_min": float(np.min(parity_ratios)),
            "pathwise_vjp_norm_ratio_max": float(np.max(parity_ratios)),
            "finite_phase_count_min": int(min(finite_counts)),
            "ordinary_solver_cosine": ordinary_solver,
            "ivw_h_solver_cosine": ivw_solver,
            "ordinary_tape_cosine": ordinary_tape,
            "ivw_h_tape_cosine": ivw_tape,
            "ordinary_mean_solver_cosine": float(np.mean(ordinary_solver)),
            "ivw_h_mean_solver_cosine": float(np.mean(ivw_solver)),
            "ordinary_mean_tape_cosine": float(np.mean(ordinary_tape)),
            "ivw_h_mean_tape_cosine": float(np.mean(ivw_tape)),
            "ordinary_phase_solver_cosine": ordinary_phase_solver.tolist(),
            "ivw_h_phase_solver_cosine": ivw_phase_solver.tolist(),
            "ordinary_phase_tape_cosine": ordinary_phase_tape.tolist(),
            "ivw_h_phase_tape_cosine": ivw_phase_tape.tolist(),
            "retained_pathwise_cosine": _vector_cosine(
                ordinary_nominal, ivw_nominal
            ),
            "retained_pathwise_norm_ratio": float(
                np.linalg.norm(ivw_nominal) / np.linalg.norm(ordinary_nominal)
            ),
            "pathwise_weight_mean": float(
                np.mean(
                    [
                        captures[(actor_name, solver_name, tape)][
                            "pathwise_weight"
                        ]
                        for solver_name in SOLVERS
                        for tape in (0, 1)
                    ]
                )
            ),
        }

    evidence = {"valid": True, "actors": summaries}
    outcome = classify_ivw_h_gradient_audit(evidence)
    if outcome == "invalid-execution":
        raise ValueError(f"IVW-H evidence failed validity gates: {summaries}")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    preflight_path = output_directory / "preflight.json"
    evidence_path = output_directory / "gradient_evidence.npz"
    summary_path = output_directory / "gradient_summary.json"
    plot_path = output_directory / "gradient_reliability.png"
    _atomic_json(
        preflight_path,
        {
            "valid": True,
            "protocol": PROTOCOL,
            **source,
            "input_sha256": input_hashes,
            "solvers": list(SOLVERS),
            "tape_seeds": list(TAPE_SEEDS),
            "seed": seed,
        },
    )
    _atomic_npz(evidence_path, arrays)
    _atomic_json(
        summary_path,
        {
            **evidence,
            "protocol": PROTOCOL,
            "outcome": outcome,
            "gradient_evidence_sha256": sha256_file(evidence_path),
            "input_sha256": input_hashes,
            **source,
        },
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, actor_name in zip(axes, ACTORS, strict=True):
        row = summaries[actor_name]
        axis.bar(
            ["solver/RP", "solver/IVW-H", "tape/RP", "tape/IVW-H"],
            [
                row["ordinary_mean_solver_cosine"],
                row["ivw_h_mean_solver_cosine"],
                row["ordinary_mean_tape_cosine"],
                row["ivw_h_mean_tape_cosine"],
            ],
            color=["#777777", "#2a78c5", "#999999", "#39a96b"],
        )
        axis.set_ylim(-1.0, 1.0)
        axis.set_title(actor_name)
        axis.set_ylabel("gradient cosine")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    temporary_plot = plot_path.with_name(f".{plot_path.name}.tmp.png")
    figure.savefig(temporary_plot, dpi=160)
    plt.close(figure)
    os.replace(temporary_plot, plot_path)
    artifacts = {
        path.name: sha256_file(path)
        for path in (preflight_path, evidence_path, summary_path, plot_path)
    }
    completion_path = output_directory / "completion.json"
    _atomic_json(
        completion_path,
        {
            "valid": True,
            "protocol": PROTOCOL,
            "outcome": outcome,
            "artifacts": artifacts,
        },
    )
    validate_completion(completion_path)
    return json.loads(completion_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-phase", type=int, default=25)
    parser.add_argument("--smoke-replicas", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_gradient_capture(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        reference_path=args.reference_path,
        output_directory=args.output_directory,
        repository=args.repository,
        code_commit=args.code_commit,
        seed=args.seed,
        smoke=args.smoke,
        smoke_phase=args.smoke_phase,
        smoke_replicas=args.smoke_replicas,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
