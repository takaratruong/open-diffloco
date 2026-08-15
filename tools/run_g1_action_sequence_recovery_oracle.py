"""Optimize one bounded phase-indexed action tape over E023 failure states."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.data_structures import Normalizer
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.build_g1_e023_carried_reset_bank import validate_code_commit
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_progressive_recovery_expert import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
    EXPECTED_SOURCE_BANK_SHA256,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.evaluate_g1_tracking import (
    _load_policy,
    make_evaluation_env,
)


PROTOCOL = "g1-e023-action-sequence-recovery-oracle-v1"
HORIZON = 32
UPDATES = 64
CORRECTION_BOUND = 0.5
LEARNING_RATE = 0.03
PER_START_GRADIENT_CLIP = 1.0
SOURCE_PHASE = 0
SOURCE_ROWS = 24


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def phase_tape_correction(
    tape: jax.Array, phases: jax.Array, *, phase_min: int
) -> jax.Array:
    """Index a phase tape with exact zeros before and after its support."""
    values = jnp.asarray(tape)
    phase_values = jnp.asarray(phases)
    indices = phase_values - phase_min
    valid = (indices >= 0) & (indices < values.shape[0])
    safe_indices = jnp.clip(indices, 0, values.shape[0] - 1)
    selected = values[safe_indices]
    return jnp.where(valid[..., None], selected, jnp.zeros_like(selected))


def recovery_oracle_outcome(
    *,
    baseline_survival: list[int],
    candidate_survival: list[int],
    horizon: int,
    execution_valid: bool,
) -> str:
    """Classify physical recoverability without making a policy claim."""
    if not execution_valid:
        return "invalid-execution"
    if (
        len(baseline_survival) != len(candidate_survival)
        or len(candidate_survival) == 0
        or horizon < 1
    ):
        raise ValueError("oracle survival evidence is malformed")
    if all(value >= horizon for value in candidate_survival):
        return "action-sequence-recoverable"
    return "action-sequence-insufficient"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_failure_rows(bank_path: Path) -> dict[str, np.ndarray]:
    with np.load(bank_path, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    selected = np.asarray(source["source_start_phase"]) == SOURCE_PHASE
    if int(np.sum(selected)) != SOURCE_ROWS:
        raise ValueError("oracle requires exactly 24 phase-zero failure rows")
    rows = {
        name: np.asarray(values)[selected]
        for name, values in source.items()
        if np.asarray(values).shape[:1] == (120,)
    }
    if (
        rows["qpos"].shape != (SOURCE_ROWS, 36)
        or rows["qvel"].shape != (SOURCE_ROWS, 35)
        or rows["last_act"].shape != (SOURCE_ROWS, 29)
        or rows["actor_obs_history"].shape != (SOURCE_ROWS, 10, 328)
        or not np.array_equal(rows["phase"], np.arange(87, 111))
    ):
        raise ValueError("oracle failure-row layout does not match E027")
    return rows


def _build_environment(hparams: dict[str, object], reference_path: Path):
    return make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        reference_path=reference_path,
        reference_stride=int(hparams["reference_stride"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            hparams["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(
            hparams["actor_reference_preview_mode"]
        ),
        actor_observation_noise=False,
        domain_randomization=False,
        reference_reset_noise_scale=0.0,
        reference_residual_control=bool(
            hparams["reference_residual_control"]
        ),
        reference_residual_scale=float(hparams["reference_residual_scale"]),
    )


def run_oracle(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    bank_path: Path,
    output_directory: Path,
    seed: int,
    independent_tapes: bool = False,
    worst_margin_objective: bool = False,
    updates: int = UPDATES,
) -> dict[str, object]:
    """Optimize and evaluate one shared phase-indexed action correction tape."""
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    rows = _load_failure_rows(bank_path)
    env = _build_environment(hparams, reference_path)
    actor, actor_params, normalizer_state = _load_policy(
        env, checkpoint_path, seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)
    profile = get_solver_profile("g1-4x5")

    def make_state(qpos, qvel, phase, last_act, history, rng):
        randomization = env._nominal_randomization()
        data = env._data_from_state(
            qpos=qpos,
            qvel=qvel,
            randomization=randomization,
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

    phase_min = int(np.min(rows["phase"]))
    tape_length = int(np.max(rows["phase"])) + HORIZON - phase_min
    thresholds = jnp.asarray(
        [0.25, 1.3, 0.8, 0.4], dtype=jnp.float64
    )

    def parent_action(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        return actor.apply(actor_params, normalized).astype(jnp.float64)

    def single_loss(tape_logits, initial_state):
        correction_tape = CORRECTION_BOUND * jnp.tanh(tape_logits)

        def step(carry, step_index):
            state, alive = carry
            correction = (
                correction_tape[step_index]
                if independent_tapes
                else phase_tape_correction(
                    correction_tape,
                    state.info["phase"],
                    phase_min=phase_min,
                )
            ).astype(jnp.float64)
            next_state = env.step(state, parent_action(state) + correction)
            errors = jnp.stack(
                [
                    next_state.metrics["termination_anchor_z_error"],
                    next_state.metrics["termination_anchor_xy_error"],
                    next_state.metrics["termination_gravity_z_error"],
                    next_state.metrics["termination_distal_z_error"],
                ]
            )
            normalized_errors = errors / thresholds
            margin = jnp.sum(
                jax.nn.softplus(12.0 * (normalized_errors - 0.65)) / 12.0
            )
            step_loss = (
                -next_state.reward
                + 0.5 * margin
                + 0.01 * jnp.mean(jnp.square(correction))
            )
            terminal = next_state.info["terminal"] > 0.5
            return (next_state, alive & ~terminal), (
                jnp.where(alive, step_loss, 0.0),
                terminal,
                correction,
                next_state.reward,
                normalized_errors,
                alive,
            )

        (_, _), (
            losses,
            terminals,
            corrections,
            rewards,
            normalized_errors,
            active,
        ) = jax.lax.scan(
            step,
            (initial_state, jnp.asarray(True)),
            jnp.arange(HORIZON),
        )
        if worst_margin_objective:
            masked_errors = jnp.where(
                active[:, None], normalized_errors, -10.0
            )
            objective = (
                jax.nn.logsumexp(12.0 * masked_errors) / 12.0
                + 0.01 * jnp.mean(jnp.square(corrections))
            )
        else:
            objective = jnp.sum(losses)
        return objective, (
            terminals,
            corrections,
            rewards,
            normalized_errors,
        )

    value_and_gradient = jax.vmap(
        jax.value_and_grad(single_loss, has_aux=True),
        in_axes=(0, 0) if independent_tapes else (None, 0),
    )
    optimizer = optax.adam(LEARNING_RATE)
    tape_shape = (
        (SOURCE_ROWS, HORIZON, env.action_dim)
        if independent_tapes
        else (tape_length, env.action_dim)
    )
    tape_logits = jnp.zeros(tape_shape, dtype=jnp.float64)
    optimizer_state = optimizer.init(tape_logits)

    @jax.jit
    def update(tape, state):
        (loss_aux, gradients) = value_and_gradient(tape, initial_states)
        losses, _aux = loss_aux
        flat = gradients.reshape(SOURCE_ROWS, -1)
        norms = jnp.linalg.norm(flat, axis=1)
        scale = jnp.minimum(1.0, PER_START_GRADIENT_CLIP / (norms + 1e-12))
        clipped = gradients * scale[:, None, None]
        gradient = clipped if independent_tapes else jnp.mean(clipped, axis=0)
        updates, next_state = optimizer.update(gradient, state, tape)
        next_tape = optax.apply_updates(tape, updates)
        return next_tape, next_state, jnp.mean(losses), norms, gradient

    curve = []
    for update_index in range(1, updates + 1):
        tape_logits, optimizer_state, loss, gradient_norms, gradient = update(
            tape_logits, optimizer_state
        )
        jax.block_until_ready(tape_logits)
        record = {
            "update": update_index,
            "loss": float(loss),
            "gradient_norm": float(jnp.linalg.norm(gradient)),
            "per_start_gradient_norm_median": float(jnp.median(gradient_norms)),
            "per_start_gradient_norm_max": float(jnp.max(gradient_norms)),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise RuntimeError("action-sequence oracle became nonfinite")
        curve.append(record)
        if update_index == 1 or update_index % 8 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)

    final_tape = CORRECTION_BOUND * jnp.tanh(tape_logits)
    zero_tape = jnp.zeros_like(tape_logits)
    evaluate_population = jax.vmap(
        single_loss,
        in_axes=(0, 0) if independent_tapes else (None, 0),
    )

    @jax.jit
    def evaluate(tape):
        _losses, auxiliary = evaluate_population(tape, initial_states)
        return auxiliary

    (
        baseline_terminals,
        baseline_corrections,
        baseline_rewards,
        baseline_normalized_errors,
    ) = evaluate(zero_tape)
    (
        candidate_terminals,
        candidate_corrections,
        candidate_rewards,
        candidate_normalized_errors,
    ) = evaluate(tape_logits)

    def survival(terminals: np.ndarray) -> list[int]:
        output = []
        for row in terminals:
            indices = np.flatnonzero(row)
            output.append(int(indices[0]) if indices.size else HORIZON)
        return output

    baseline_survival = survival(np.asarray(baseline_terminals))
    candidate_survival = survival(np.asarray(candidate_terminals))
    execution_valid = (
        np.isfinite(np.asarray(final_tape)).all()
        and np.isfinite(np.asarray(candidate_rewards)).all()
        and float(jnp.max(jnp.abs(final_tape))) <= CORRECTION_BOUND + 1e-12
    )
    outcome = recovery_oracle_outcome(
        baseline_survival=baseline_survival,
        candidate_survival=candidate_survival,
        horizon=HORIZON,
        execution_valid=bool(execution_valid),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = output_directory / "action_sequence_oracle.npz"
    evidence_arrays = {
        "phase": np.asarray(rows["phase"]),
        "correction_tape": np.asarray(final_tape),
        "baseline_terminal": np.asarray(baseline_terminals),
        "candidate_terminal": np.asarray(candidate_terminals),
        "baseline_correction": np.asarray(baseline_corrections),
        "candidate_correction": np.asarray(candidate_corrections),
        "baseline_reward": np.asarray(baseline_rewards),
        "candidate_reward": np.asarray(candidate_rewards),
        "baseline_normalized_termination_errors": np.asarray(
            baseline_normalized_errors
        ),
        "candidate_normalized_termination_errors": np.asarray(
            candidate_normalized_errors
        ),
    }
    if independent_tapes:
        evidence_arrays["tape_step"] = np.arange(HORIZON)
    else:
        evidence_arrays["tape_phase"] = np.arange(
            phase_min, phase_min + tape_length
        )
    _write_npz(evidence_path, **evidence_arrays)
    summary = {
        "valid": bool(execution_valid),
        "protocol": (
            "g1-e023-worst-margin-action-sequence-recovery-oracle-v1"
            if worst_margin_objective
            else
            "g1-e023-independent-action-sequence-recovery-oracle-v1"
            if independent_tapes
            else PROTOCOL
        ),
        "outcome": outcome,
        "horizon": HORIZON,
        "updates": updates,
        "correction_bound": CORRECTION_BOUND,
        "learning_rate": LEARNING_RATE,
        "phase_min": phase_min,
        "phase_max": phase_min + tape_length - 1,
        "independent_tapes": independent_tapes,
        "worst_margin_objective": worst_margin_objective,
        "baseline_survival": baseline_survival,
        "candidate_survival": candidate_survival,
        "candidate_full_horizon_count": int(
            sum(value >= HORIZON for value in candidate_survival)
        ),
        "correction_rms": float(jnp.sqrt(jnp.mean(jnp.square(final_tape)))),
        "correction_max_abs": float(jnp.max(jnp.abs(final_tape))),
        "checkpoint_sha256": EXPECTED_RESUME_SHA256,
        "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
        "source_bank_sha256": EXPECTED_SOURCE_BANK_SHA256,
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
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--independent-tapes", action="store_true")
    parser.add_argument("--worst-margin-objective", action="store_true")
    parser.add_argument("--updates", type=_positive_int, default=UPDATES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    validate_code_commit(repository, args.code_commit)
    for path, expected, label in (
        (args.checkpoint, EXPECTED_RESUME_SHA256, "checkpoint"),
        (args.hparams, EXPECTED_RESUME_HPARAMS_SHA256, "hparams"),
        (args.reference_path, EXPECTED_LAFAN_REFERENCE_SHA256, "reference"),
        (args.source_bank, EXPECTED_SOURCE_BANK_SHA256, "source bank"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"oracle {label} SHA-256 does not match")
    configure_jax()
    with solver_context(get_solver_profile("g1-4x5")):
        summary = run_oracle(
            checkpoint_path=args.checkpoint.resolve(),
            hparams_path=args.hparams.resolve(),
            reference_path=args.reference_path.resolve(),
            bank_path=args.source_bank.resolve(),
            output_directory=args.output_directory.resolve(),
            seed=args.seed,
            independent_tapes=args.independent_tapes,
            worst_margin_objective=args.worst_margin_objective,
            updates=args.updates,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
