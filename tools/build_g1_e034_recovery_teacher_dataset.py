"""Build immutable state/action supervision from the E034 recovery oracle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path

import numpy as np

from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_progressive_recovery_expert import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
    EXPECTED_SOURCE_BANK_SHA256,
)


E034_SURVIVAL = (
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    30,
    32,
    26,
    21,
    20,
    18,
    16,
    15,
    14,
    12,
    11,
    9,
)
PROTOCOL = "g1-e034-recovery-teacher-dataset-v1"
EXPECTED_ORACLE_SHA256 = (
    "a6b9fed0df1efd00a5fc72d3389e753ed3d6c6ad2b1e6f342e047ae386bc6b55"
)

_SHAPES = {
    "actor_obs": (24, 32, 3280),
    "phase": (24, 32),
    "parent_action": (24, 32, 29),
    "correction": (24, 32, 29),
    "raw_action": (24, 32, 29),
    "effective_action": (24, 32, 29),
    "alive": (24, 32),
    "terminal": (24, 32),
    "replay_alive": (24, 32),
    "replay_terminal": (24, 32),
    "reward": (24, 32),
    "normalized_termination_errors": (24, 32, 4),
    "success_mask": (24,),
}


def _survival(terminals: np.ndarray) -> list[int]:
    output = []
    for row in terminals:
        indices = np.flatnonzero(row)
        output.append(int(indices[0]) if indices.size else 32)
    return output


def _alive_from_survival(survival: list[int] | tuple[int, ...]) -> np.ndarray:
    alive = np.ones((24, 32), dtype=bool)
    for row, survived in enumerate(survival):
        if survived < 32:
            alive[row, survived + 1 :] = False
    return alive


def validate_teacher_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Fail closed unless arrays exactly reproduce the E034 replay contract."""
    for name, shape in _SHAPES.items():
        if name not in arrays:
            raise ValueError(f"teacher dataset is missing {name}")
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"teacher dataset {name} shape does not match")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"teacher dataset {name} must be finite")

    terminal = np.asarray(arrays["terminal"], dtype=bool)
    alive = np.asarray(arrays["alive"], dtype=bool)
    survival = _survival(terminal)
    if survival != list(E034_SURVIVAL):
        raise ValueError("teacher dataset survival does not reproduce E034")
    expected_alive = _alive_from_survival(E034_SURVIVAL)
    if not np.array_equal(alive, expected_alive):
        raise ValueError("teacher dataset alive mask does not match terminals")
    replay_terminal = np.asarray(arrays["replay_terminal"], dtype=bool)
    replay_alive = np.asarray(arrays["replay_alive"], dtype=bool)
    replay_survival = _survival(replay_terminal)
    if not np.array_equal(
        replay_alive, _alive_from_survival(replay_survival)
    ):
        raise ValueError("teacher replay alive mask does not match terminals")
    successful = np.asarray(E034_SURVIVAL) == 32
    replay_successful = np.asarray(replay_survival) == 32
    if not np.array_equal(replay_successful, successful):
        raise ValueError("teacher replay changes the E034 success set")
    failed_drift = np.abs(
        np.asarray(replay_survival)[~successful]
        - np.asarray(E034_SURVIVAL)[~successful]
    )
    maximum_failed_drift = int(np.max(failed_drift))
    if maximum_failed_drift > 1:
        raise ValueError("teacher failed-row replay drift exceeds one transition")

    raw = np.asarray(arrays["raw_action"])
    effective = np.asarray(arrays["effective_action"])
    if not np.array_equal(effective, np.clip(raw, -1.0, 1.0)):
        raise ValueError("teacher effective actions do not match final clipping")
    if not np.allclose(
        raw,
        np.asarray(arrays["parent_action"])
        + np.asarray(arrays["correction"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("teacher raw action does not equal parent plus correction")

    if not np.array_equal(np.asarray(arrays["success_mask"]), successful):
        raise ValueError("teacher success mask does not match E034 survival")
    action_mask = np.broadcast_to(alive[..., None], raw.shape)
    clipped = np.abs(raw) > 1.0

    def clip_fraction(start_mask: np.ndarray) -> float:
        selected = action_mask & start_mask[:, None, None]
        return float(np.mean(clipped[selected]))

    return {
        "survival": survival,
        "replay_survival": replay_survival,
        "maximum_failed_survival_drift": maximum_failed_drift,
        "successful_starts": int(np.sum(successful)),
        "teacher_rows": int(np.sum(successful) * 32),
        "all_clip_fraction": clip_fraction(np.ones(24, dtype=bool)),
        "recovered_clip_fraction": clip_fraction(successful),
        "failed_clip_fraction": clip_fraction(~successful),
    }


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


def publish_teacher_dataset(
    *,
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Publish validated tensors atomically before a hash-bound manifest."""
    validation = validate_teacher_arrays(arrays)
    output_directory = output_directory.resolve()
    dataset_path = output_directory / "e034_recovery_teacher_dataset.npz"
    summary_path = output_directory / "summary.json"
    _write_npz(dataset_path, arrays)
    manifest = {
        "valid": True,
        "protocol": PROTOCOL,
        **validation,
        **dict(provenance),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
    }
    _write_json(summary_path, manifest)
    return manifest


def collect_teacher_arrays(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    bank_path: Path,
    oracle_evidence_path: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    """Replay the immutable E034 correction tape and capture policy inputs."""
    import jax
    import jax.numpy as jnp

    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.evaluate_g1_tracking import _load_policy
    from tools.run_g1_action_sequence_recovery_oracle import (
        _build_environment,
        _load_failure_rows,
    )

    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    rows = _load_failure_rows(bank_path)
    env = _build_environment(hparams, reference_path)
    actor, actor_params, normalizer_state = _load_policy(
        env, checkpoint_path, seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)
    with np.load(oracle_evidence_path, allow_pickle=False) as archive:
        correction_tape = np.asarray(archive["correction_tape"])
        oracle_terminal = np.asarray(archive["candidate_terminal"], dtype=bool)
    if correction_tape.shape != (24, 32, 29):
        raise ValueError("E034 correction tape shape does not match")
    if not np.isfinite(correction_tape).all():
        raise ValueError("E034 correction tape must be finite")

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

    keys = jax.random.split(jax.random.PRNGKey(seed), 24)
    profile = get_solver_profile("g1-4x5")
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

    def parent_action(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        return actor.apply(actor_params, normalized).astype(jnp.float64)

    def replay_one(initial_state, corrections):
        def step(carry, correction):
            state, alive = carry
            parent = parent_action(state)
            raw = parent + correction
            effective = jnp.clip(raw, -1.0, 1.0)
            next_state = env.step(state, raw)
            errors = jnp.stack(
                [
                    next_state.metrics["termination_anchor_z_error"],
                    next_state.metrics["termination_anchor_xy_error"],
                    next_state.metrics["termination_gravity_z_error"],
                    next_state.metrics["termination_distal_z_error"],
                ]
            ) / thresholds
            terminal = next_state.info["terminal"] > 0.5
            output = (
                state.obs,
                state.info["phase"],
                parent,
                correction,
                raw,
                effective,
                alive,
                terminal,
                next_state.reward,
                errors,
            )
            return (next_state, alive & ~terminal), output

        return jax.lax.scan(
            step,
            (initial_state, jnp.asarray(True)),
            corrections,
        )[1]

    with solver_context(profile):
        replay = jax.jit(jax.vmap(replay_one))(
            initial_states,
            jnp.asarray(correction_tape),
        )
    names = (
        "actor_obs",
        "phase",
        "parent_action",
        "correction",
        "raw_action",
        "effective_action",
        "replay_alive",
        "replay_terminal",
        "reward",
        "normalized_termination_errors",
    )
    arrays = {name: np.asarray(value) for name, value in zip(names, replay)}
    oracle_survival = _survival(oracle_terminal)
    arrays["terminal"] = oracle_terminal
    arrays["alive"] = _alive_from_survival(oracle_survival)
    arrays["success_mask"] = np.asarray(E034_SURVIVAL) == 32
    return arrays


def _zero_seed(value: str) -> int:
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("E034 replay seed must be exactly zero")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--oracle-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=_zero_seed, default=0)
    return parser


def main() -> None:
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from tools.build_g1_e023_carried_reset_bank import validate_code_commit
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets
    from tools.run_g1_tracking_shac import configure_jax

    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    code_commit = validate_code_commit(repository, args.code_commit)
    expected_inputs = (
        (args.checkpoint, EXPECTED_RESUME_SHA256, "checkpoint"),
        (args.hparams, EXPECTED_RESUME_HPARAMS_SHA256, "hparams"),
        (args.reference_path, EXPECTED_LAFAN_REFERENCE_SHA256, "reference"),
        (args.source_bank, EXPECTED_SOURCE_BANK_SHA256, "source bank"),
        (args.oracle_evidence, EXPECTED_ORACLE_SHA256, "oracle evidence"),
    )
    for path, expected, label in expected_inputs:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"teacher {label} SHA-256 does not match")
    hparams = json.loads(args.hparams.read_text(encoding="utf-8"))
    runtime_assets = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    configure_jax()
    arrays = collect_teacher_arrays(
        checkpoint_path=args.checkpoint.resolve(),
        hparams_path=args.hparams.resolve(),
        reference_path=args.reference_path.resolve(),
        bank_path=args.source_bank.resolve(),
        oracle_evidence_path=args.oracle_evidence.resolve(),
        seed=args.seed,
    )
    manifest = publish_teacher_dataset(
        output_directory=args.output_directory,
        arrays=arrays,
        provenance={
            "code_commit": code_commit,
            "checkpoint_sha256": EXPECTED_RESUME_SHA256,
            "hparams_sha256": EXPECTED_RESUME_HPARAMS_SHA256,
            "reference_sha256": EXPECTED_LAFAN_REFERENCE_SHA256,
            "source_bank_sha256": EXPECTED_SOURCE_BANK_SHA256,
            "oracle_evidence_sha256": EXPECTED_ORACLE_SHA256,
            "seed": args.seed,
            **runtime_assets,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
