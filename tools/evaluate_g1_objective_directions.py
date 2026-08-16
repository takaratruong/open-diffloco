"""Audit frozen E023 SHAC objective directions on one fixed G1 population."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any, Mapping

import numpy as np

from tools.prepare_g1_rmr_reference import sha256_file


POPULATION = 512
CARRIED_POPULATION = 128
EXACT_POPULATION = 384
BANK_ROWS = 120
ACTION_DIM = 29
MAX_HORIZON = 48
DIRECTIONS = ("h24_a", "h24_b", "h48_a", "bootstrap_a")
EXPECTED_INPUT_SHA256 = {
    "checkpoint": "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f",
    "hparams": "a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650",
    "reference": "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db",
    "bank": "d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5",
    "expert": "373fd6528d135dac65b38c35728800da693780558a03bb0cca6a412e314f7bd2",
}
EXPECTED_MODEL_SHA256 = "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
EXPECTED_CONTROLLER_SHA256 = "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
PROTOCOL = "g1-objective-direction-audit-v1"


def build_fixed_population_indices(seed: int) -> dict[str, np.ndarray]:
    """Return the immutable 128 carried plus 384 exact-reference population."""
    if seed != 0:
        raise ValueError("objective-direction population seed must be zero")
    rng = np.random.default_rng(seed)
    repeats = rng.choice(BANK_ROWS, size=CARRIED_POPULATION - BANK_ROWS, replace=False)
    carried = np.concatenate((np.arange(BANK_ROWS), repeats)).astype(np.int32)
    exact = np.floor(
        np.linspace(0, 499, EXACT_POPULATION, endpoint=False)
    ).astype(np.int32)
    return {
        "source_kind": np.concatenate(
            (
                np.ones(CARRIED_POPULATION, dtype=np.int8),
                np.zeros(EXACT_POPULATION, dtype=np.int8),
            )
        ),
        "source_index": np.concatenate((carried, exact)),
    }


def build_fixed_noise_tapes(seed: int) -> dict[str, np.ndarray]:
    """Generate two immutable standard-normal epsilon tapes."""
    if seed != 0:
        raise ValueError("objective-direction noise seed must be zero")
    rng_a = np.random.default_rng(41_024 + seed)
    rng_b = np.random.default_rng(82_048 + seed)
    shape = (POPULATION, MAX_HORIZON, ACTION_DIM)
    return {
        "a": rng_a.standard_normal(shape, dtype=np.float32),
        "b": rng_b.standard_normal(shape, dtype=np.float32),
    }


def validate_common_noise_prefix(prefix: np.ndarray, full: np.ndarray) -> bool:
    """Require a bit-exact H24 prefix of the H48 tape."""
    prefix = np.asarray(prefix)
    full = np.asarray(full)
    if (
        prefix.shape != (POPULATION, 24, ACTION_DIM)
        or full.shape != (POPULATION, MAX_HORIZON, ACTION_DIM)
        or prefix.dtype != np.float32
        or full.dtype != np.float32
        or not np.array_equal(prefix, full[:, :24])
    ):
        raise ValueError("H24 noise is not the exact H48 common prefix")
    return True


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


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite numeric evidence")
    return array


def validate_gradient_artifacts(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Validate the complete fixed-population gradient evidence."""
    required = {
        "source_kind",
        "source_index",
        "phase",
        "rng_key",
        "noise_tape_a",
        "noise_tape_b",
        "h24_tape_env_cosine",
        "h24_h48_env_cosine",
    }
    for direction in DIRECTIONS:
        required.update(
            {
                f"{direction}_combined",
                f"{direction}_task",
                f"{direction}_counts",
                f"{direction}_cosine",
                f"{direction}_weights",
                f"{direction}_env_norm",
            }
        )
    if set(arrays) != required:
        missing = sorted(required - set(arrays))
        extra = sorted(set(arrays) - required)
        raise ValueError(f"gradient artifact schema differs: missing={missing}, extra={extra}")
    population = int(np.asarray(arrays["source_kind"]).shape[0])
    if population != POPULATION:
        raise ValueError("scientific gradient evidence requires 512 states")
    if np.asarray(arrays["source_kind"]).shape != (POPULATION,):
        raise ValueError("source kind shape does not match")
    if np.asarray(arrays["source_index"]).shape != (POPULATION,):
        raise ValueError("source index shape does not match")
    if np.asarray(arrays["phase"]).shape != (POPULATION,):
        raise ValueError("phase shape does not match")
    if np.asarray(arrays["rng_key"]).shape != (POPULATION, 2):
        raise ValueError("RNG key shape does not match")
    if np.asarray(arrays["rng_key"]).dtype != np.uint32:
        raise ValueError("RNG keys must be uint32")
    validate_common_noise_prefix(
        np.asarray(arrays["noise_tape_a"])[:, :24],
        np.asarray(arrays["noise_tape_a"]),
    )
    if np.array_equal(arrays["noise_tape_a"], arrays["noise_tape_b"]):
        raise ValueError("noise tapes must be independent")
    vector_width = None
    direction_norms: dict[str, float] = {}
    for direction in DIRECTIONS:
        combined = _require_finite(
            f"{direction} combined", arrays[f"{direction}_combined"]
        )
        task = _require_finite(f"{direction} task", arrays[f"{direction}_task"])
        counts = _require_finite(
            f"{direction} counts", arrays[f"{direction}_counts"]
        )
        cosine = _require_finite(
            f"{direction} cosine", arrays[f"{direction}_cosine"]
        )
        weights = _require_finite(
            f"{direction} weights", arrays[f"{direction}_weights"]
        )
        env_norm = _require_finite(
            f"{direction} env norm", arrays[f"{direction}_env_norm"]
        )
        if combined.ndim != 1 or combined.size == 0:
            raise ValueError("combined gradient vector must be nonempty")
        vector_width = combined.size if vector_width is None else vector_width
        if combined.size != vector_width or task.shape != (5, vector_width):
            raise ValueError("gradient vector widths do not agree")
        if counts.shape != (5,) or np.any(counts <= 0):
            raise ValueError("every direction must occupy all five phase bins")
        if cosine.shape != (5, 5) or weights.shape != (5,):
            raise ValueError("CAGrad diagnostic shapes do not match")
        if env_norm.shape != (POPULATION,) or np.any(env_norm < 0):
            raise ValueError("per-environment gradient norms do not match")
        norm = float(np.linalg.norm(combined))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("combined gradient direction must be nonzero")
        direction_norms[direction] = norm
    for name in ("h24_tape_env_cosine", "h24_h48_env_cosine"):
        values = _require_finite(name, arrays[name])
        if values.shape != (POPULATION,) or np.any(np.abs(values) > 1.000001):
            raise ValueError("per-environment cosine evidence does not match")
    return {
        "valid": True,
        "protocol": PROTOCOL,
        "population": POPULATION,
        "carried_population": int(np.sum(np.asarray(arrays["source_kind"]) == 1)),
        "exact_population": int(np.sum(np.asarray(arrays["source_kind"]) == 0)),
        "direction_norms": direction_norms,
        "h24_tape_env_cosine_mean": float(np.mean(arrays["h24_tape_env_cosine"])),
        "h24_h48_env_cosine_mean": float(np.mean(arrays["h24_h48_env_cosine"])),
    }


def publish_gradient_artifacts(
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    provenance: Mapping[str, object],
    scientific: bool,
) -> dict[str, object]:
    """Publish hash-bound evidence, writing the completion marker last."""
    validation = validate_gradient_artifacts(arrays)
    code_commit = provenance.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise ValueError("a full code commit is required")
    input_hashes = provenance.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("input SHA-256 provenance is required")
    output_directory = output_directory.resolve()
    artifact = output_directory / "gradient_evidence.npz"
    summary_path = output_directory / "gradient_summary.json"
    completion_path = output_directory / "completion.json"
    _atomic_npz(artifact, arrays)
    summary = {
        **validation,
        "scientific": bool(scientific),
        "code_commit": code_commit,
        "input_sha256": dict(input_hashes),
        "gradient_artifact_path": str(artifact),
        "gradient_artifact_sha256": sha256_file(artifact),
    }
    _atomic_json(summary_path, summary)
    completion = {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }
    _atomic_json(completion_path, completion)
    return completion


def validate_completion_manifest(path: Path) -> dict[str, object]:
    """Reopen a completed publication and independently verify its hashes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        raise ValueError("completion manifest is invalid")
    artifact = Path(str(payload.get("gradient_artifact_path", "")))
    summary = Path(str(payload.get("summary_path", "")))
    if (
        not artifact.is_file()
        or sha256_file(artifact) != payload.get("gradient_artifact_sha256")
        or not summary.is_file()
        or sha256_file(summary) != payload.get("summary_sha256")
    ):
        raise ValueError("completion artifact SHA-256 does not match")
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    validate_gradient_artifacts(arrays)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    if summary_payload.get("gradient_artifact_sha256") != payload.get(
        "gradient_artifact_sha256"
    ):
        raise ValueError("completion summary does not match")
    return payload


def build_preflight(
    paths: Mapping[str, Path], *, code_commit: str, repository: Path
) -> dict[str, object]:
    """Bind the exact frozen inputs, runtime assets, and clean source commit."""
    if set(paths) != set(EXPECTED_INPUT_SHA256):
        raise ValueError("objective-direction input names are not exact")
    hashes = {name: sha256_file(path.resolve()) for name, path in paths.items()}
    if hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("objective-direction input SHA-256 does not match")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tools"],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout
    if head != code_commit or len(code_commit) != 40 or dirty:
        raise ValueError("objective-direction source provenance does not match")
    return {
        "valid": True,
        "protocol": f"{PROTOCOL}-preflight",
        "code_commit": code_commit,
        "input_sha256": hashes,
    }


def _tree_matrix(tree: Any) -> np.ndarray:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("gradient tree is empty")
    rows = int(np.asarray(leaves[0]).shape[0])
    return np.concatenate(
        [np.asarray(leaf).reshape(rows, -1) for leaf in leaves], axis=1
    )


def _tree_vector(tree: Any) -> np.ndarray:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    return np.concatenate([np.asarray(leaf).reshape(-1) for leaf in leaves])


def _env_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=1)
    denom = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)


def run_gradient_capture(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    bank_path: Path,
    expert_path: Path,
    output_directory: Path,
    seed: int,
    code_commit: str,
    repository: Path,
) -> dict[str, object]:
    """Capture four training-effective adapter directions on one frozen batch."""
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.algorithm import squeeze_value_head
    from src.algorithms.shac.objective_direction_audit import (
        aggregate_audit_direction,
    )
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
        transplant_zero_head_recovery_features,
    )
    from src.core.data_structures import Normalizer
    from src.core.networks import Critic
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.build_g1_e023_carried_reset_bank import validate_e023_hparams
    from tools.evaluate_g1_e038_recovery_transfer import (
        _load_all_bank_rows,
        validate_parent_checkpoint,
    )
    from tools.evaluate_g1_tracking import _load_policy
    from tools.run_g1_action_sequence_recovery_oracle import _build_environment
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets

    if seed != 0:
        raise ValueError("objective-direction audit seed must be zero")
    input_paths = {
        "checkpoint": checkpoint_path.resolve(),
        "hparams": hparams_path.resolve(),
        "reference": reference_path.resolve(),
        "bank": bank_path.resolve(),
        "expert": expert_path.resolve(),
    }
    preflight = build_preflight(
        input_paths, code_commit=code_commit, repository=repository.resolve()
    )
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    runtime = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    if (
        runtime["model_sha256"] != EXPECTED_MODEL_SHA256
        or runtime["controller_sha256"] != EXPECTED_CONTROLLER_SHA256
    ):
        raise ValueError("objective-direction runtime assets do not match")
    preflight["input_sha256"].update(
        model=runtime["model_sha256"], controller=runtime["controller_sha256"]
    )
    env = _build_environment(hparams, reference_path)
    profile = get_solver_profile("g1-4x5")
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    validate_parent_checkpoint(checkpoint)
    actor, parent_params, actor_norm_state = _load_policy(env, checkpoint_path, seed)
    actor_normalizer = Normalizer(env.actor_frame_obs_dim)
    critic_normalizer = Normalizer(env.critic_obs_dim)
    critic_layers = checkpoint.target_critic_params["params"]
    dense_names = sorted(
        (name for name in critic_layers if name.startswith("Dense_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    critic_hidden = tuple(
        int(critic_layers[name]["kernel"].shape[-1]) for name in dense_names[:-1]
    )
    critic = Critic(hidden=critic_hidden)
    residual = PreviewResidualAdapter(action_dim=env.action_dim, hidden_dim=256)
    template = residual.init(
        jax.random.PRNGKey(13),
        jnp.zeros(env.actor_frame_obs_dim, dtype=jnp.float32),
    )
    with expert_path.open("rb") as stream:
        expert_params = pickle.load(stream)
    adapter_params, transfer = transplant_zero_head_recovery_features(
        template, expert_params
    )
    if transfer.get("output_head_zero") is not True:
        raise ValueError("objective-direction adapter head is not exactly zero")

    population = build_fixed_population_indices(seed)
    rows = _load_all_bank_rows(bank_path)
    carried_indices = population["source_index"][:CARRIED_POPULATION]
    exact_phases = population["source_index"][CARRIED_POPULATION:]
    keys = jax.random.split(jax.random.PRNGKey(seed), POPULATION)

    def carried_state(qpos, qvel, phase, last_act, history, rng):
        randomization = env._nominal_randomization()
        data = env._data_from_state(qpos=qpos, qvel=qvel, randomization=randomization)
        return env._initial_state_from_data(
            data=data,
            rng=rng,
            difficulty=jnp.asarray(0.0),
            phase=phase,
            randomization=randomization,
            last_act=last_act,
            actor_obs_history=history,
        )

    with solver_context(profile):
        carried_states = jax.jit(jax.vmap(carried_state))(
            jnp.asarray(rows["qpos"][carried_indices]),
            jnp.asarray(rows["qvel"][carried_indices]),
            jnp.asarray(rows["phase"][carried_indices], dtype=jnp.int32),
            jnp.asarray(rows["last_act"][carried_indices]),
            jnp.asarray(rows["actor_obs_history"][carried_indices]),
            keys[:CARRIED_POPULATION],
        )
        exact_states = jax.jit(jax.vmap(env.reset_at_phase))(
            keys[CARRIED_POPULATION:],
            jnp.zeros(EXACT_POPULATION, dtype=jnp.float64),
            jnp.asarray(exact_phases, dtype=jnp.int32),
        )
    states = jax.tree_util.tree_map(
        lambda left, right: jnp.concatenate((left, right), axis=0),
        carried_states,
        exact_states,
    )
    phases = np.concatenate(
        (np.asarray(rows["phase"][carried_indices], dtype=np.int32), exact_phases)
    )
    composite = FrozenPreviewResidualParams(parent_params, adapter_params)
    normalized = jax.vmap(
        lambda obs: env.normalize_actor_obs(actor_normalizer, actor_norm_state, obs)
    )(states.obs).astype(jnp.float32)
    candidate_action, parent_action, correction = apply_frozen_preview_residual(
        actor,
        residual,
        composite,
        normalized,
        history_len=env.actor_history_len,
        treatment_frame_dim=env.actor_frame_obs_dim,
    )
    if not np.array_equal(np.asarray(candidate_action), np.asarray(parent_action)) or not np.array_equal(
        np.asarray(correction), np.zeros_like(np.asarray(correction))
    ):
        raise ValueError("zero-head adapter does not preserve E023 actions")

    noise_std = jnp.asarray(hparams["action_noise_std_end"], dtype=jnp.float64)
    if noise_std.shape != (ACTION_DIM,) or not bool(jnp.all(jnp.isfinite(noise_std))):
        raise ValueError("E023 final RMR action-noise vector does not match")
    gamma = float(hparams["gamma"])
    clip_actions = bool(hparams["clip_sampled_actor_actions"])

    def loss(adapter, initial_state, noise, bootstrap_scale, *, horizon):
        def rollout_step(state, epsilon):
            _obs_key, env_key = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_key})
            obs_norm = env.normalize_actor_obs(
                actor_normalizer, actor_norm_state, state.obs
            ).astype(jnp.float32)
            action, _, _ = apply_frozen_preview_residual(
                actor,
                residual,
                FrozenPreviewResidualParams(parent_params, adapter),
                obs_norm,
                history_len=env.actor_history_len,
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            noisy = action.astype(jnp.float64) + epsilon.astype(jnp.float64) * noise_std
            if clip_actions:
                noisy = jnp.clip(noisy, -1.0, 1.0)
            next_state = env.step(state, noisy)
            return next_state, (
                next_state.reward,
                next_state.done,
                next_state.info["terminal"],
                next_state.info["bootstrap_critic_obs"],
            )

        final_state, trajectory = jax.lax.scan(
            rollout_step, initial_state, noise, length=horizon
        )
        reward, done, terminal, bootstrap_critic_obs = trajectory
        bootstrap_normalized = critic_normalizer.normalize(
            checkpoint.critic_normalizer, bootstrap_critic_obs
        ).astype(jnp.float32)
        bootstrap_v = squeeze_value_head(
            critic.apply(checkpoint.target_critic_params, bootstrap_normalized)
        )

        def accumulate(carry, values):
            total, running, discount = carry
            step_reward, step_done, step_terminal, value_next = values
            next_discount = discount * gamma
            running = running + discount * step_reward
            truncation = bootstrap_scale * (1.0 - step_terminal) * next_discount * value_next
            total = total + jnp.where(step_done, running + truncation, 0.0)
            return (
                total,
                jnp.where(step_done, 0.0, running),
                jnp.where(step_done, 1.0, next_discount),
            ), None

        (total, running, discount), _ = jax.lax.scan(
            accumulate,
            (jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0)),
            (reward, done, terminal, bootstrap_v),
        )
        final_obs = critic_normalizer.normalize(
            checkpoint.critic_normalizer,
            env._get_critic_obs(final_state.data, final_state.info),
        ).astype(jnp.float32)
        final_v = squeeze_value_head(
            critic.apply(checkpoint.target_critic_params, final_obs)
        )
        final_bootstrap = jnp.where(
            done[-1], 0.0, bootstrap_scale * discount * final_v
        )
        return -(total + running + final_bootstrap) / horizon

    tapes = build_fixed_noise_tapes(seed)

    def capture(noise: np.ndarray, *, horizon: int, bootstrap_scale: float):
        gradient_fn = jax.jit(
            jax.vmap(
                jax.grad(
                    lambda adapter, state, epsilon: loss(
                        adapter, state, epsilon, bootstrap_scale, horizon=horizon
                    )
                ),
                in_axes=(None, 0, 0),
            )
        )
        chunks = []
        with solver_context(profile):
            for start in range(0, POPULATION, 256):
                stop = start + 256
                state_chunk = jax.tree_util.tree_map(lambda value: value[start:stop], states)
                chunk = gradient_fn(
                    adapter_params,
                    state_chunk,
                    jnp.asarray(noise[start:stop, :horizon]),
                )
                chunks.append(jax.device_get(chunk))
        return jax.tree_util.tree_map(
            lambda first, second: np.concatenate((first, second), axis=0),
            chunks[0],
            chunks[1],
        )

    gradient_trees = {
        "h24_a": capture(tapes["a"], horizon=24, bootstrap_scale=0.0),
        "h24_b": capture(tapes["b"], horizon=24, bootstrap_scale=0.0),
        "h48_a": capture(tapes["a"], horizon=48, bootstrap_scale=0.0),
    }
    h24_bootstrapped = capture(tapes["a"], horizon=24, bootstrap_scale=1.0)
    gradient_trees["bootstrap_a"] = jax.tree_util.tree_map(
        lambda full, immediate: full - immediate,
        h24_bootstrapped,
        gradient_trees["h24_a"],
    )
    matrices = {
        name: _tree_matrix(tree).astype(np.float32, copy=False)
        for name, tree in gradient_trees.items()
    }
    arrays: dict[str, np.ndarray] = {
        "source_kind": population["source_kind"],
        "source_index": population["source_index"],
        "phase": phases,
        "rng_key": np.asarray(keys, dtype=np.uint32),
        "noise_tape_a": tapes["a"],
        "noise_tape_b": tapes["b"],
        "h24_tape_env_cosine": _env_cosine(matrices["h24_a"], matrices["h24_b"]),
        "h24_h48_env_cosine": _env_cosine(matrices["h24_a"], matrices["h48_a"]),
    }
    for name, tree in gradient_trees.items():
        aggregated = aggregate_audit_direction(
            tree,
            phases,
            phase_count=env.reference_length,
            clip_norm=1.0,
            alpha=0.5,
            iterations=32,
        )
        arrays.update(
            {
                f"{name}_combined": _tree_vector(aggregated.combined_gradient),
                f"{name}_task": _tree_matrix(aggregated.task_gradients),
                f"{name}_counts": np.asarray(aggregated.env_counts, dtype=np.int32),
                f"{name}_cosine": np.asarray(aggregated.cosine_matrix),
                f"{name}_weights": np.asarray(aggregated.weights),
                f"{name}_env_norm": np.linalg.norm(matrices[name], axis=1),
            }
        )
    return publish_gradient_artifacts(
        output_directory,
        arrays,
        provenance=preflight,
        scientific=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    from tools.run_g1_tracking_shac import configure_jax

    args = build_parser().parse_args()
    configure_jax()
    manifest = run_gradient_capture(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        reference_path=args.reference_path,
        bank_path=args.source_bank,
        expert_path=args.expert_checkpoint,
        output_directory=args.output_directory,
        seed=args.seed,
        code_commit=args.code_commit,
        repository=args.repository,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
