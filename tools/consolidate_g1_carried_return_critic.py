"""Consolidate E021's fixed critic recipe and gate unseen phases."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algorithms.shac.residual_preview_adapter import FrozenPreviewResidualParams
from src.core.data_structures import Normalizer
from src.core.networks import Critic
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.evaluate_g1_flax_phase_grid import ACTOR_HISTORY_LEN, LOOKAHEAD_STEPS
from tools.evaluate_g1_terminal_value_calibration import (
    _sha256,
    _write_json_atomic,
    calibration_metrics,
    runtime_asset_provenance,
    validate_code_provenance,
    validate_runtime_contract,
)
from tools.evaluate_g1_tracking import configure_jax, make_evaluation_env
from tools.refit_g1_carried_return_critic import (
    CRITIC_LR,
    _atomic_pickle,
    _h12_records,
    _predict,
    capture_trajectories,
    noncritic_state_drift,
    phase_splits,
    replace_critic_state,
)


CONSOLIDATION_STEPS = 1_640
CONFIRMATION_PHASES = (5, 105, 205, 305, 405)


def load_frozen_dataset(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load exactly E021's complete 30-phase carried dataset."""
    expected = tuple(
        (*phase_splits()["fit"], *phase_splits()["validation"], *phase_splits()["test"])
    )
    with np.load(path.resolve(), allow_pickle=False) as archive:
        dataset = {}
        expected_keys = set()
        for phase in expected:
            rows = {}
            for field in ("critic_observations", "rewards", "returns"):
                key = f"phase_{phase}_{field}"
                expected_keys.add(key)
                rows[field] = np.asarray(archive[key])
            count = rows["returns"].size
            if (
                rows["critic_observations"].ndim != 2
                or rows["critic_observations"].shape[0] != count
                or rows["rewards"].shape != (count,)
                or not all(np.all(np.isfinite(value)) for value in rows.values())
            ):
                raise ValueError("frozen dataset contains malformed rows")
            dataset[phase] = rows
        if set(archive.files) != expected_keys:
            raise ValueError("frozen dataset key set does not match E021")
    return dataset


def fit_fixed_recipe(
    checkpoint_state,
    critic: Critic,
    normalized_observations: np.ndarray,
    returns: np.ndarray,
):
    """Run the validation-selected E021 recipe once on all observed phases."""
    observations = jnp.asarray(normalized_observations, dtype=jnp.float32)
    targets = jnp.asarray(returns, dtype=jnp.float32)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(CRITIC_LR)
    )

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean(
                jnp.square(critic.apply(p, observations).reshape(-1) - targets)
            )
        )(params)
        updates, new_opt = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt, loss

    params = checkpoint_state.critic_params
    opt_state = checkpoint_state.critic_opt
    loss = None
    for _ in range(CONSOLIDATION_STEPS):
        params, opt_state, loss = update(params, opt_state)
    return params, opt_state, float(loss)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-hparams-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
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
    hparams_path = args.hparams.resolve()
    dataset_path = args.dataset.resolve()
    reference_path = args.reference_path.resolve()
    for path, expected, label in (
        (checkpoint_path, args.expected_checkpoint_sha256, "checkpoint"),
        (hparams_path, args.expected_hparams_sha256, "hparams"),
        (dataset_path, args.expected_dataset_sha256, "dataset"),
        (reference_path, args.expected_reference_sha256, "reference"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
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
        "hparams_path": str(hparams_path),
        "hparams_sha256": _sha256(hparams_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
    }
    raw_dataset = load_frozen_dataset(dataset_path)
    critic_normalizer = Normalizer(env.critic_obs_dim)
    normalized_dataset = {
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
        for phase, rows in raw_dataset.items()
    }
    all_observations = np.concatenate(
        [rows["critic_observations"] for rows in normalized_dataset.values()]
    )
    all_returns = np.concatenate(
        [rows["returns"] for rows in normalized_dataset.values()]
    )
    critic = Critic()
    selected_params, selected_opt, fit_loss = fit_fixed_recipe(
        checkpoint_state, critic, all_observations, all_returns
    )
    with solver_context(profile):
        confirmation = capture_trajectories(
            env,
            checkpoint_state,
            seed=args.seed,
            phases=CONFIRMATION_PHASES,
        )
    normalized_confirmation = {
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
        for phase, rows in confirmation.items()
    }
    confirmation_observations = np.concatenate(
        [normalized_confirmation[p]["critic_observations"] for p in CONFIRMATION_PHASES]
    )
    confirmation_returns = np.concatenate(
        [normalized_confirmation[p]["returns"] for p in CONFIRMATION_PHASES]
    )
    original_predictions = _predict(
        critic,
        checkpoint_state.target_critic_params,
        jnp.asarray(confirmation_observations),
    )
    selected_predictions = _predict(
        critic, selected_params, jnp.asarray(confirmation_observations)
    )
    original_metrics = calibration_metrics(
        original_predictions, confirmation_returns
    )
    selected_metrics = calibration_metrics(
        selected_predictions, confirmation_returns
    )
    selected_by_phase = {}
    original_by_phase = {}
    offset = 0
    for phase in CONFIRMATION_PHASES:
        count = normalized_confirmation[phase]["returns"].size
        selected_by_phase[phase] = selected_predictions[offset : offset + count]
        original_by_phase[phase] = original_predictions[offset : offset + count]
        offset += count
    selected_h12 = _h12_records(
        normalized_confirmation, CONFIRMATION_PHASES, selected_by_phase
    )
    original_h12 = _h12_records(
        normalized_confirmation, CONFIRMATION_PHASES, original_by_phase
    )
    improves = (
        selected_metrics["rank_correlation"] > original_metrics["rank_correlation"]
        and selected_metrics["nrmse"] < original_metrics["nrmse"]
        and all(
            selected["relative_error"] < original["relative_error"]
            for selected, original in zip(selected_h12, original_h12)
        )
    )
    success = (
        selected_metrics["rank_correlation"] >= 0.8
        and selected_metrics["nrmse"] <= 0.25
        and all(row["relative_error"] <= 0.25 for row in selected_h12)
        and improves
    )
    candidate_state = replace_critic_state(
        checkpoint_state,
        critic_params=selected_params,
        critic_opt=selected_opt,
    )
    drift = noncritic_state_drift(checkpoint_state, candidate_state)
    if not drift["valid"]:
        raise ValueError("consolidation changed non-critic TrainState")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_output = output / "critic_consolidated.pkl"
    if success:
        _atomic_pickle(checkpoint_output, candidate_state)
        source_hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
        source_hparams.update(
            {
                "carried_return_critic_consolidation": True,
                "carried_return_critic_consolidation_steps": CONSOLIDATION_STEPS,
                "carried_return_critic_consolidation_lr": CRITIC_LR,
                "carried_return_critic_consolidation_dataset_sha256": (
                    _sha256(dataset_path)
                ),
            }
        )
        temporary = output / ".hparams.json.tmp"
        temporary.write_text(
            json.dumps(source_hparams, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output / "hparams.json")
    report = {
        "protocol": "g1-carried-return-critic-consolidation-v1",
        "provenance": provenance,
        "consolidation_steps": CONSOLIDATION_STEPS,
        "critic_lr": CRITIC_LR,
        "fit_loss": fit_loss,
        "confirmation_phases": list(CONFIRMATION_PHASES),
        "confirmation_survival": {
            str(p): int(confirmation[p]["returns"].size)
            for p in CONFIRMATION_PHASES
        },
        "original_metrics": original_metrics,
        "selected_metrics": selected_metrics,
        "original_h12_records": original_h12,
        "selected_h12_records": selected_h12,
        "improves_original_target": improves,
        "noncritic_state_drift": drift,
        "outcome": (
            "critic-consolidation-calibrated"
            if success
            else "critic-consolidation-insufficient"
        ),
        "checkpoint_sha256": _sha256(checkpoint_output) if success else None,
    }
    _write_json_atomic(output / "critic_consolidation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
