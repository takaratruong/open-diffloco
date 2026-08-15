"""Distill the successful E034 recovery tape into a small closed-loop expert."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algorithms.shac.residual_preview_adapter import (
    PreviewResidualAdapter,
    current_treatment_frame,
)
from src.core.data_structures import Normalizer
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.build_g1_e023_carried_reset_bank import validate_code_commit
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_action_sequence_recovery_oracle import (
    HORIZON,
    SOURCE_ROWS,
    _build_environment,
    _load_failure_rows,
)
from tools.build_g1_e034_recovery_teacher_dataset import E034_SURVIVAL
from tools.run_g1_progressive_recovery_expert import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
    EXPECTED_SOURCE_BANK_SHA256,
)
from tools.evaluate_g1_tracking import _load_policy


PROTOCOL = "g1-e034-state-conditioned-recovery-expert-v1"
EXPECTED_DATASET_SHA256 = (
    "203effe85e34794a76ebd344018e928f224d9cb8c9cedca9e2c4108f62343ad2"
)
BASELINE_SURVIVAL = tuple(range(28, 4, -1))
TRAINING_UPDATES = 2_000
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256


def _zero_seed(value: str) -> int:
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("recovery expert seed must be exactly zero")
    return seed


def select_teacher_rows(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Flatten exactly the non-contiguous thirteen successful E034 rows."""
    expected_mask = np.asarray(E034_SURVIVAL) == HORIZON
    required_shapes = {
        "actor_obs": (SOURCE_ROWS, HORIZON, 3280),
        "parent_action": (SOURCE_ROWS, HORIZON, 29),
        "correction": (SOURCE_ROWS, HORIZON, 29),
        "effective_action": (SOURCE_ROWS, HORIZON, 29),
        "success_mask": (SOURCE_ROWS,),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"teacher {name} shape does not match")
    if not np.array_equal(np.asarray(arrays["success_mask"]), expected_mask):
        raise ValueError("teacher success mask does not match E034")
    selected: dict[str, np.ndarray] = {}
    for name in ("actor_obs", "parent_action", "correction", "effective_action"):
        value = np.asarray(arrays[name])[expected_mask]
        if not np.isfinite(value).all():
            raise ValueError(f"teacher {name} must be finite")
        selected[name] = value.reshape((-1, value.shape[-1]))
    if selected["actor_obs"].shape[0] != 416:
        raise ValueError("teacher selection must contain exactly 416 transitions")
    return selected


def imitation_loss(
    predicted_correction: jax.Array,
    parent_action: jax.Array,
    teacher_correction: jax.Array,
    teacher_effective_action: jax.Array,
) -> jax.Array:
    """Fit the bounded correction and its actual post-boundary action."""
    correction_error = jnp.mean(
        jnp.square(predicted_correction - teacher_correction)
    )
    predicted_effective = jnp.clip(parent_action + predicted_correction, -1.0, 1.0)
    effective_error = jnp.mean(
        jnp.square(predicted_effective - teacher_effective_action)
    )
    return correction_error + effective_error


def classify_recovery_expert(
    *,
    baseline_survival: list[int],
    candidate_survival: list[int],
    teacher_success_mask: np.ndarray,
    execution_valid: bool,
) -> str:
    """Classify closed-loop reproduction without generalization claims."""
    if not execution_valid:
        return "invalid-execution"
    if (
        len(baseline_survival) != SOURCE_ROWS
        or len(candidate_survival) != SOURCE_ROWS
        or np.asarray(teacher_success_mask).shape != (SOURCE_ROWS,)
    ):
        raise ValueError("recovery expert survival evidence is malformed")
    baseline = np.asarray(baseline_survival)
    candidate = np.asarray(candidate_survival)
    mask = np.asarray(teacher_success_mask, dtype=bool)
    reproduced = int(np.sum(candidate[mask] >= HORIZON))
    if reproduced >= 10 and np.all(candidate >= baseline):
        return "state-conditioned-recovery-reproduced"
    if int(np.sum(candidate >= HORIZON)) > int(np.sum(baseline >= HORIZON)):
        return "state-conditioned-recovery-partial"
    return "state-conditioned-recovery-insufficient"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
    os.replace(temporary, path)


def _survival(terminals: np.ndarray) -> list[int]:
    result = []
    for row in terminals:
        indices = np.flatnonzero(row)
        result.append(int(indices[0]) if indices.size else HORIZON)
    return result


def fit_expert(
    *,
    expert: PreviewResidualAdapter,
    initial_params,
    frames: jax.Array,
    parent_actions: jax.Array,
    teacher_corrections: jax.Array,
    teacher_effective_actions: jax.Array,
):
    """Run the single registered full-batch imitation fit."""
    optimizer = optax.adam(LEARNING_RATE)
    optimizer_state = optimizer.init(initial_params)

    @jax.jit
    def update(params, state):
        def loss_fn(candidate):
            prediction = expert.apply(candidate, frames)
            return imitation_loss(
                prediction,
                parent_actions,
                teacher_corrections,
                teacher_effective_actions,
            )

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, next_state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), next_state, loss, gradients

    params = initial_params
    best_params = initial_params
    best_loss = math.inf
    curve: list[dict[str, float | int]] = []
    for update_index in range(1, TRAINING_UPDATES + 1):
        params, optimizer_state, loss, gradients = update(params, optimizer_state)
        loss_value = float(loss)
        gradient_norm = float(optax.global_norm(gradients))
        if not math.isfinite(loss_value) or not math.isfinite(gradient_norm):
            raise RuntimeError("recovery expert fit became nonfinite")
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
        if update_index == 1 or update_index % 100 == 0:
            record = {
                "update": update_index,
                "loss": loss_value,
                "gradient_norm": gradient_norm,
            }
            curve.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return best_params, curve, best_loss


def run_experiment(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    bank_path: Path,
    dataset_path: Path,
    output_directory: Path,
    seed: int,
    code_commit: str,
) -> dict[str, object]:
    """Fit the expert and evaluate it closed loop on all registered starts."""
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as archive:
        dataset = {name: archive[name] for name in archive.files}
    teacher = select_teacher_rows(dataset)
    success_mask = np.asarray(dataset["success_mask"], dtype=bool)

    env = _build_environment(hparams, reference_path)
    actor, actor_params, normalizer_state = _load_policy(env, checkpoint_path, seed)
    normalizer = Normalizer(env.actor_frame_obs_dim)
    normalized_history = env.normalize_actor_obs(
        normalizer, normalizer_state, jnp.asarray(teacher["actor_obs"])
    ).astype(jnp.float32)
    frames = current_treatment_frame(
        normalized_history,
        history_len=env.actor_history_len,
        treatment_frame_dim=env.actor_frame_obs_dim,
    )
    recomputed_parent = actor.apply(actor_params, normalized_history).astype(jnp.float32)
    parent_difference = float(
        jnp.max(jnp.abs(recomputed_parent - jnp.asarray(teacher["parent_action"])))
    )
    if not math.isfinite(parent_difference) or parent_difference > 2e-5:
        raise ValueError("teacher parent action does not reproduce frozen policy")

    expert = PreviewResidualAdapter(action_dim=env.action_dim, hidden_dim=HIDDEN_DIM)
    initial_params = expert.init(jax.random.PRNGKey(seed), frames[:1])
    expert_params, curve, best_loss = fit_expert(
        expert=expert,
        initial_params=initial_params,
        frames=frames,
        parent_actions=jnp.asarray(teacher["parent_action"], dtype=jnp.float32),
        teacher_corrections=jnp.asarray(teacher["correction"], dtype=jnp.float32),
        teacher_effective_actions=jnp.asarray(
            teacher["effective_action"], dtype=jnp.float32
        ),
    )
    fitted_correction = expert.apply(expert_params, frames)
    fitted_effective = jnp.clip(recomputed_parent + fitted_correction, -1.0, 1.0)
    fitted_correction_mse = float(
        jnp.mean(
            jnp.square(fitted_correction - jnp.asarray(teacher["correction"]))
        )
    )
    fitted_effective_mse = float(
        jnp.mean(
            jnp.square(fitted_effective - jnp.asarray(teacher["effective_action"]))
        )
    )

    rows = _load_failure_rows(bank_path)

    def make_state(qpos, qvel, phase, last_act, history, rng):
        randomization = env._nominal_randomization()
        data = env._data_from_state(
            qpos=qpos, qvel=qvel, randomization=randomization
        )
        return env._initial_state_from_data(
            data=data,
            rng=rng,
            difficulty=jnp.asarray(0.0),
            phase=phase,
            randomization=randomization,
            last_act=last_act,
            actor_obs_history=history,
        )

    profile = get_solver_profile("g1-4x5")
    keys = jax.random.split(jax.random.PRNGKey(seed), SOURCE_ROWS)
    with solver_context(profile):
        initial_states = jax.vmap(make_state)(
            jnp.asarray(rows["qpos"]),
            jnp.asarray(rows["qvel"]),
            jnp.asarray(rows["phase"], dtype=jnp.int32),
            jnp.asarray(rows["last_act"]),
            jnp.asarray(rows["actor_obs_history"]),
            keys,
        )
    thresholds = jnp.asarray([0.25, 1.3, 0.8, 0.4], dtype=jnp.float64)

    def rollout_one(initial_state):
        def step(carry, _):
            state, alive = carry
            normalized = env.normalize_actor_obs(
                normalizer, normalizer_state, state.obs
            ).astype(jnp.float32)
            frame = current_treatment_frame(
                normalized,
                history_len=env.actor_history_len,
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            parent = actor.apply(actor_params, normalized).astype(jnp.float64)
            correction = expert.apply(expert_params, frame).astype(jnp.float64)
            raw_action = parent + correction
            next_state = env.step(state, raw_action)
            terminal = next_state.info["terminal"] > 0.5
            errors = jnp.stack(
                [
                    next_state.metrics["termination_anchor_z_error"],
                    next_state.metrics["termination_anchor_xy_error"],
                    next_state.metrics["termination_gravity_z_error"],
                    next_state.metrics["termination_distal_z_error"],
                ]
            ) / thresholds
            output = (
                state.obs,
                state.info["phase"],
                parent,
                correction,
                raw_action,
                jnp.clip(raw_action, -1.0, 1.0),
                alive,
                terminal,
                next_state.reward,
                errors,
            )
            return (next_state, alive & ~terminal), output

        return jax.lax.scan(step, (initial_state, jnp.asarray(True)), None, HORIZON)[1]

    with solver_context(profile):
        rollout = jax.jit(jax.vmap(rollout_one))(initial_states)
    names = (
        "actor_obs",
        "phase",
        "parent_action",
        "correction",
        "raw_action",
        "effective_action",
        "alive",
        "terminal",
        "reward",
        "normalized_termination_errors",
    )
    evidence = {name: np.asarray(value) for name, value in zip(names, rollout)}
    candidate_survival = _survival(evidence["terminal"])
    execution_valid = all(np.isfinite(value).all() for value in evidence.values())
    outcome = classify_recovery_expert(
        baseline_survival=list(BASELINE_SURVIVAL),
        candidate_survival=candidate_survival,
        teacher_success_mask=success_mask,
        execution_valid=execution_valid,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = output_directory / "closed_loop_rollout.npz"
    checkpoint_output = output_directory / "recovery_expert.pkl"
    _write_npz(evidence_path, evidence)
    _write_pickle(checkpoint_output, expert_params)
    summary = {
        "valid": bool(execution_valid),
        "protocol": PROTOCOL,
        "outcome": outcome,
        "code_commit": code_commit,
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "training_updates": TRAINING_UPDATES,
        "learning_rate": LEARNING_RATE,
        "teacher_rows": 416,
        "teacher_success_rows": np.flatnonzero(success_mask).tolist(),
        "parent_action_max_abs_difference": parent_difference,
        "best_imitation_loss": best_loss,
        "fitted_correction_mse": fitted_correction_mse,
        "fitted_effective_action_mse": fitted_effective_mse,
        "baseline_survival": list(BASELINE_SURVIVAL),
        "candidate_survival": candidate_survival,
        "candidate_full_horizon_count": int(
            sum(value >= HORIZON for value in candidate_survival)
        ),
        "teacher_successes_reproduced": int(
            np.sum(np.asarray(candidate_survival)[success_mask] >= HORIZON)
        ),
        "candidate_correction_rms": float(
            np.sqrt(np.mean(np.square(evidence["correction"])))
        ),
        "candidate_correction_max_abs": float(
            np.max(np.abs(evidence["correction"]))
        ),
        "candidate_action_clip_fraction": float(
            np.mean(np.abs(evidence["raw_action"]) > 1.0)
        ),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "source_bank_sha256": EXPECTED_SOURCE_BANK_SHA256,
        "teacher_dataset_sha256": EXPECTED_DATASET_SHA256,
        "expert_checkpoint_path": str(checkpoint_output),
        "expert_checkpoint_sha256": sha256_file(checkpoint_output),
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "curve": curve,
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=_zero_seed, default=0)
    return parser


def main() -> None:
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets
    from tools.run_g1_tracking_shac import configure_jax

    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    code_commit = validate_code_commit(repository, args.code_commit)
    for path, expected, label in (
        (args.checkpoint, EXPECTED_RESUME_SHA256, "checkpoint"),
        (args.hparams, EXPECTED_RESUME_HPARAMS_SHA256, "hparams"),
        (args.reference_path, EXPECTED_LAFAN_REFERENCE_SHA256, "reference"),
        (args.source_bank, EXPECTED_SOURCE_BANK_SHA256, "source bank"),
        (args.teacher_dataset, EXPECTED_DATASET_SHA256, "teacher dataset"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"recovery expert {label} SHA-256 does not match")
    hparams = json.loads(args.hparams.read_text(encoding="utf-8"))
    validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    configure_jax()
    summary = run_experiment(
        checkpoint_path=args.checkpoint.resolve(),
        hparams_path=args.hparams.resolve(),
        reference_path=args.reference_path.resolve(),
        bank_path=args.source_bank.resolve(),
        dataset_path=args.teacher_dataset.resolve(),
        output_directory=args.output_directory.resolve(),
        seed=args.seed,
        code_commit=code_commit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
