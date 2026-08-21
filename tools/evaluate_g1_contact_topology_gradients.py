"""Audit threshold-free contact-topology gradient truncation for G1 SHAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from tools.prepare_g1_rmr_reference import sha256_file


PROTOCOL = "g1-contact-topology-gradient-v1"
POPULATION = 120
PHASES = np.asarray((0, 25, 50, 75, 100), dtype=np.int32)
REPLICAS_PER_PHASE = 24
HORIZON = 24
ACTION_DIM = 29
SOLVERS = ("g1-4x5", "diagnostic-10x20")
MODES = ("ordinary", "contact_truncated")
ACTORS = ("fresh", "e023")
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
    {
        "contact-truncation-robust",
        "contact-truncation-neutral",
        "contact-truncation-destructive",
        "invalid-execution",
    }
)


def build_fixed_phase_population(seed: int) -> dict[str, np.ndarray]:
    """Return the registered five-phase starts and one fixed H24 noise tape."""

    if seed != 0:
        raise ValueError("contact topology gradient seed must be zero")
    phase = np.repeat(PHASES, REPLICAS_PER_PHASE)
    rng = np.random.default_rng(913_024)
    noise = rng.standard_normal(
        (POPULATION, HORIZON, ACTION_DIM), dtype=np.float32
    )
    return {"phase": phase, "noise": noise}


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("gradient vectors must be finite")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("gradient vectors must be nonzero")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def compare_solver_gradients(
    *,
    ordinary_4x5: np.ndarray,
    ordinary_10x20: np.ndarray,
    truncated_4x5: np.ndarray,
    truncated_10x20: np.ndarray,
) -> dict[str, float]:
    """Compare solver angular error before and after truncation."""

    ordinary_cosine = _vector_cosine(ordinary_4x5, ordinary_10x20)
    truncated_cosine = _vector_cosine(truncated_4x5, truncated_10x20)
    ordinary_error = 1.0 - ordinary_cosine
    truncated_error = 1.0 - truncated_cosine
    reduction = (
        (ordinary_error - truncated_error) / ordinary_error
        if ordinary_error > 1e-12
        else 0.0
    )
    return {
        "ordinary_solver_cosine": ordinary_cosine,
        "truncated_solver_cosine": truncated_cosine,
        "ordinary_solver_angular_error": ordinary_error,
        "truncated_solver_angular_error": truncated_error,
        "angular_error_reduction": float(reduction),
    }


def classify_contact_topology_gradient_audit(
    evidence: Mapping[str, object],
) -> str:
    """Apply the preregistered invalid/destructive/robust/neutral order."""

    actors = evidence.get("actors")
    if evidence.get("valid") is not True or not isinstance(actors, Mapping):
        return "invalid-execution"
    rows = []
    for actor in ACTORS:
        row = actors.get(actor)
        if not isinstance(row, Mapping):
            return "invalid-execution"
        required = (
            "ordinary_solver_cosine",
            "truncated_solver_cosine",
            "ordinary_truncated_cosine",
            "truncated_to_ordinary_norm_ratio",
            "phase_solver_cosine_delta",
            "event_bins",
            "event_count",
            "finite",
        )
        if any(key not in row for key in required) or row["finite"] is not True:
            return "invalid-execution"
        numeric = np.asarray(
            [
                row["ordinary_solver_cosine"],
                row["truncated_solver_cosine"],
                row["ordinary_truncated_cosine"],
                row["truncated_to_ordinary_norm_ratio"],
                *row["phase_solver_cosine_delta"],
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or len(row["phase_solver_cosine_delta"]) != 5
            or int(row["event_bins"]) < 3
            or int(row["event_count"]) < 24
        ):
            return "invalid-execution"
        rows.append(row)

    destructive = any(
        float(row["ordinary_truncated_cosine"]) < 0.8
        or not 0.25
        <= float(row["truncated_to_ordinary_norm_ratio"])
        <= 4.0
        or float(row["truncated_solver_cosine"])
        < float(row["ordinary_solver_cosine"]) - 0.02
        for row in rows
    )
    if destructive:
        return "contact-truncation-destructive"

    robust = all(
        (
            (1.0 - float(row["truncated_solver_cosine"]))
            <= 0.8 * (1.0 - float(row["ordinary_solver_cosine"]))
            and min(map(float, row["phase_solver_cosine_delta"])) >= -0.02
        )
        for row in rows
    )
    return "contact-truncation-robust" if robust else "contact-truncation-neutral"


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


def validate_completion(path: Path) -> dict[str, object]:
    """Reopen a completion record and hash-validate every bound artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("valid") is not True or payload.get("protocol") != PROTOCOL:
        raise ValueError("completion contract is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("completion artifacts are missing")
    for name, expected in artifacts.items():
        artifact = path.parent / str(name)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"artifact hash mismatch for {name}")
    return payload


def validate_forward_identity(
    ordinary: Mapping[str, np.ndarray],
    truncated: Mapping[str, np.ndarray],
    *,
    label: str,
) -> bool:
    """Require exact primal identity and localize the first mismatch."""

    if set(ordinary) != set(truncated):
        raise ValueError(f"{label} forward schemas differ")
    for name in sorted(ordinary):
        left = np.asarray(ordinary[name])
        right = np.asarray(truncated[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{label} forward {name} shape/dtype differs")
        if np.array_equal(left, right):
            continue
        if np.issubdtype(left.dtype, np.number):
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            mismatch = np.argwhere(left != right)
            first = tuple(int(index) for index in mismatch[0])
            detail = (
                f"first_index={first}, left={left[first]!r}, "
                f"right={right[first]!r}, "
                f"max_abs={float(np.nanmax(difference)):.17g}"
            )
        else:
            detail = f"different_count={int(np.sum(left != right))}"
        raise ValueError(f"{label} forward {name} differs: {detail}")
    return True


def _tree_vector(tree: Any) -> np.ndarray:
    leaves = [np.asarray(leaf).reshape(-1) for leaf in __import__("jax").tree_util.tree_leaves(tree)]
    return np.concatenate(leaves).astype(np.float64, copy=False)


def _tree_matrix(tree: Any) -> np.ndarray:
    leaves = [np.asarray(leaf).reshape(np.asarray(leaf).shape[0], -1) for leaf in __import__("jax").tree_util.tree_leaves(tree)]
    return np.concatenate(leaves, axis=1).astype(np.float64, copy=False)


def _validate_clean_source(repository: Path, code_commit: str) -> dict[str, str]:
    if len(code_commit) != 40:
        raise ValueError("code commit must be a full SHA-1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tools"],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout
    if head != code_commit or dirty:
        raise ValueError("contact topology evaluator requires exact clean source")
    return {"code_commit": head, "dirty_patch_sha256": hashlib.sha256(b"").hexdigest()}


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
) -> dict[str, object]:
    """Capture the registered ordinary/truncated gradients under two solvers."""

    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.contact_truncation import contact_gradient_barrier
    from src.algorithms.shac.objective_direction_audit import aggregate_audit_direction
    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.build_g1_e023_carried_reset_bank import validate_e023_hparams
    from tools.evaluate_g1_tracking import _load_policy
    from tools.run_g1_action_sequence_recovery_oracle import _build_environment
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets

    if seed != 0:
        raise ValueError("contact topology gradient seed must be zero")
    source = _validate_clean_source(repository.resolve(), code_commit)
    paths = {
        "checkpoint": checkpoint_path.resolve(),
        "hparams": hparams_path.resolve(),
        "reference": reference_path.resolve(),
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("contact topology inputs do not match E023")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    runtime = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    if (
        runtime["model_sha256"] != EXPECTED_MODEL_SHA256
        or runtime["controller_sha256"] != EXPECTED_CONTROLLER_SHA256
    ):
        raise ValueError("contact topology runtime assets do not match")
    input_hashes.update(
        model=runtime["model_sha256"], controller=runtime["controller_sha256"]
    )
    population = build_fixed_phase_population(seed)
    phases = population["phase"]
    noise = population["noise"]
    if smoke:
        matches = np.flatnonzero(phases == smoke_phase)
        if matches.size == 0:
            raise ValueError("smoke phase is not registered")
        selected = matches[:1]
        phases = phases[selected]
        noise = noise[selected]
    count = int(phases.shape[0])
    keys = jax.random.split(jax.random.PRNGKey(seed), count)

    captures: dict[tuple[str, str, str], dict[str, Any]] = {}
    initial_arrays: dict[str, np.ndarray] | None = None
    actor_parameters: dict[str, Any] = {}

    for solver_name in SOLVERS if not smoke else SOLVERS[:1]:
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
            for name in current_initial:
                if not np.array_equal(current_initial[name], initial_arrays[name]):
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
        gamma = float(hparams["gamma"])
        noise_std_by_actor = {
            "fresh": jnp.asarray(hparams["action_noise_std_start"], dtype=jnp.float64),
            "e023": jnp.asarray(hparams["action_noise_std_end"], dtype=jnp.float64),
        }

        for actor_name in ACTORS if not smoke else ("e023",):
            actor = actors[actor_name]
            params = params_by_actor[actor_name]
            norm_state = norms_by_actor[actor_name]
            noise_std = noise_std_by_actor[actor_name]

            def loss(parameters, initial_state, epsilon, *, truncate):
                def rollout_step(state, epsilon_t):
                    _obs_key, env_key = jax.random.split(state.info["rng"])
                    state = state.replace(info={**state.info, "rng": env_key})
                    normalized = env.normalize_actor_obs(
                        normalizer, norm_state, state.obs
                    ).astype(jnp.float32)
                    action = actor.apply(parameters, normalized).astype(jnp.float64)
                    noisy_action = action + epsilon_t.astype(jnp.float64) * noise_std
                    candidate = env.step(state, noisy_action)
                    event = jax.lax.stop_gradient(
                        candidate.info["transition_contact_topology_event"]
                    )
                    next_state = contact_gradient_barrier(
                        candidate, event, enabled=truncate
                    )
                    return next_state, {
                        "reward": next_state.reward,
                        "done": candidate.done,
                        "terminal": candidate.info["terminal"],
                        "event": event,
                        "qpos": candidate.data.qpos,
                        "qvel": candidate.data.qvel,
                        "action": noisy_action,
                    }

                _, trajectory = jax.lax.scan(
                    rollout_step, initial_state, epsilon, length=HORIZON
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

            value_grad = jax.jit(
                jax.vmap(
                    jax.value_and_grad(
                        lambda parameters, state, epsilon, truncate: loss(
                            parameters, state, epsilon, truncate=truncate
                        ),
                        has_aux=True,
                    ),
                    in_axes=(None, 0, 0, None),
                )
            )
            for mode in MODES:
                truncate = jnp.asarray(mode == "contact_truncated")
                gradient_chunks = []
                auxiliary_chunks = []
                with solver_context(profile):
                    for start in range(0, count, REPLICAS_PER_PHASE):
                        stop = min(start + REPLICAS_PER_PHASE, count)
                        state_chunk = jax.tree_util.tree_map(
                            lambda value: value[start:stop], states
                        )
                        (losses, auxiliary), gradient = value_grad(
                            params,
                            state_chunk,
                            jnp.asarray(noise[start:stop]),
                            truncate,
                        )
                        gradient_chunks.append(jax.device_get(gradient))
                        auxiliary_chunks.append(
                            jax.device_get({**auxiliary, "loss": losses})
                        )
                gradients = jax.tree_util.tree_map(
                    lambda *values: np.concatenate(values, axis=0),
                    *gradient_chunks,
                )
                auxiliary = jax.tree_util.tree_map(
                    lambda *values: np.concatenate(values, axis=0),
                    *auxiliary_chunks,
                )
                captures[(actor_name, solver_name, mode)] = {
                    "gradients": gradients,
                    "auxiliary": auxiliary,
                }

    assert initial_arrays is not None
    if smoke:
        ordinary = captures[("e023", SOLVERS[0], "ordinary")]
        truncated = captures[("e023", SOLVERS[0], "contact_truncated")]
        validate_forward_identity(
            ordinary["auxiliary"],
            truncated["auxiliary"],
            label="smoke",
        )
        norms = {
            mode: float(
                np.linalg.norm(
                    _tree_vector(captures[("e023", SOLVERS[0], mode)]["gradients"])
                )
            )
            for mode in MODES
        }
        event_count = int(np.sum(ordinary["auxiliary"]["event"]))
        if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
            raise ValueError("smoke gradients are not finite and nonzero")
        report = {
            "valid": True,
            "scientific": False,
            "protocol": f"{PROTOCOL}-smoke",
            "phase": smoke_phase,
            "event_count": event_count,
            "gradient_norms": norms,
            **source,
        }
        _atomic_json(output_directory / "smoke_summary.json", report)
        return report

    arrays: dict[str, np.ndarray] = {
        "phase": np.asarray(phases, dtype=np.int32),
        "noise": np.asarray(noise, dtype=np.float32),
        "initial_qpos": initial_arrays["qpos"],
        "initial_qvel": initial_arrays["qvel"],
        "initial_actor_obs_history": initial_arrays["history"],
    }
    summaries: dict[str, dict[str, object]] = {}
    aggregate_vectors: dict[tuple[str, str, str], np.ndarray] = {}
    task_vectors: dict[tuple[str, str, str], np.ndarray] = {}
    for actor_name in ACTORS:
        for solver_name in SOLVERS:
            ordinary_aux = captures[(actor_name, solver_name, "ordinary")]["auxiliary"]
            truncated_aux = captures[(actor_name, solver_name, "contact_truncated")]["auxiliary"]
            validate_forward_identity(
                ordinary_aux,
                truncated_aux,
                label=f"{actor_name}/{solver_name}",
            )
            for mode in MODES:
                capture = captures[(actor_name, solver_name, mode)]
                aggregated = aggregate_audit_direction(
                    capture["gradients"],
                    phases,
                    phase_count=125,
                    clip_norm=1.0,
                    alpha=0.5,
                    iterations=32,
                )
                key = (actor_name, solver_name, mode)
                combined = _tree_vector(aggregated.combined_gradient)
                task = _tree_matrix(aggregated.task_gradients)
                env_matrix = _tree_matrix(capture["gradients"])
                aggregate_vectors[key] = combined
                task_vectors[key] = task
                prefix = f"{actor_name}_{solver_name}_{mode}".replace("-", "_")
                arrays[f"{prefix}_combined"] = combined
                arrays[f"{prefix}_task"] = task
                arrays[f"{prefix}_env_norm"] = np.linalg.norm(env_matrix, axis=1)
                arrays[f"{prefix}_event"] = np.asarray(
                    capture["auxiliary"]["event"], dtype=np.bool_
                )

        comparison = compare_solver_gradients(
            ordinary_4x5=aggregate_vectors[(actor_name, SOLVERS[0], MODES[0])],
            ordinary_10x20=aggregate_vectors[(actor_name, SOLVERS[1], MODES[0])],
            truncated_4x5=aggregate_vectors[(actor_name, SOLVERS[0], MODES[1])],
            truncated_10x20=aggregate_vectors[(actor_name, SOLVERS[1], MODES[1])],
        )
        phase_ordinary = np.asarray(
            [
                _vector_cosine(
                    task_vectors[(actor_name, SOLVERS[0], MODES[0])][index],
                    task_vectors[(actor_name, SOLVERS[1], MODES[0])][index],
                )
                for index in range(5)
            ]
        )
        phase_truncated = np.asarray(
            [
                _vector_cosine(
                    task_vectors[(actor_name, SOLVERS[0], MODES[1])][index],
                    task_vectors[(actor_name, SOLVERS[1], MODES[1])][index],
                )
                for index in range(5)
            ]
        )
        ordinary_nominal = aggregate_vectors[(actor_name, SOLVERS[0], MODES[0])]
        truncated_nominal = aggregate_vectors[(actor_name, SOLVERS[0], MODES[1])]
        event_mask = captures[(actor_name, SOLVERS[0], MODES[0])]["auxiliary"]["event"]
        event_by_env = np.any(event_mask, axis=1)
        event_bins = sum(
            bool(np.any(event_by_env[phases == phase])) for phase in PHASES
        )
        summaries[actor_name] = {
            **comparison,
            "ordinary_truncated_cosine": _vector_cosine(
                ordinary_nominal, truncated_nominal
            ),
            "truncated_to_ordinary_norm_ratio": float(
                np.linalg.norm(truncated_nominal) / np.linalg.norm(ordinary_nominal)
            ),
            "phase_ordinary_solver_cosine": phase_ordinary.tolist(),
            "phase_truncated_solver_cosine": phase_truncated.tolist(),
            "phase_solver_cosine_delta": (
                phase_truncated - phase_ordinary
            ).tolist(),
            "event_bins": int(event_bins),
            "event_count": int(np.sum(event_mask)),
            "finite": True,
        }

    evidence = {"valid": True, "actors": summaries}
    outcome = classify_contact_topology_gradient_audit(evidence)
    if outcome not in REGISTERED_OUTCOMES:
        raise ValueError("unregistered contact topology outcome")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    preflight = {
        "valid": True,
        "protocol": PROTOCOL,
        **source,
        "input_sha256": input_hashes,
        "solvers": list(SOLVERS),
        "seed": seed,
    }
    preflight_path = output_directory / "preflight.json"
    evidence_path = output_directory / "gradient_evidence.npz"
    summary_path = output_directory / "gradient_summary.json"
    plot_path = output_directory / "gradient_cosines.png"
    _atomic_json(preflight_path, preflight)
    _atomic_npz(evidence_path, arrays)
    summary = {
        **evidence,
        "protocol": PROTOCOL,
        "outcome": outcome,
        "gradient_evidence_sha256": sha256_file(evidence_path),
        "input_sha256": input_hashes,
        **source,
    }
    _atomic_json(summary_path, summary)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, actor_name in zip(axes, ACTORS):
        row = summaries[actor_name]
        axis.bar(
            ["ordinary", "contact-truncated"],
            [row["ordinary_solver_cosine"], row["truncated_solver_cosine"]],
            color=["#777777", "#2a78c5"],
        )
        axis.set_ylim(-1.0, 1.0)
        axis.set_title(actor_name)
        axis.set_ylabel("4x5 vs 10x20 gradient cosine")
        axis.grid(axis="y", alpha=0.25)
    temporary_plot = plot_path.with_name(f".{plot_path.name}.tmp.png")
    figure.savefig(temporary_plot, dpi=160)
    plt.close(figure)
    os.replace(temporary_plot, plot_path)
    artifacts = {
        path.name: sha256_file(path)
        for path in (preflight_path, evidence_path, summary_path, plot_path)
    }
    completion = {
        "valid": True,
        "protocol": PROTOCOL,
        "outcome": outcome,
        "artifacts": artifacts,
    }
    completion_path = output_directory / "completion.json"
    _atomic_json(completion_path, completion)
    validate_completion(completion_path)
    return completion


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
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
