"""Build an immutable actor-context-faithful G1 carried reset bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


PROTOCOL = "g1-history-carried-reset-bank-v1"
SOURCE_PHASES = (0, 100, 200, 300, 400)
SOURCE_SURVIVAL = (70, 63, 95, 70, 44)
HISTORY_LEN = 10
LOOKAHEAD_STEPS = (4, 8, 12)
RESIDUAL_HIDDEN = 256
TERMINATION_THRESHOLDS = np.array(
    [0.25, 1.3, 0.8, 0.4], dtype=np.float64
)


def select_preterminal_indices(
    step_count: int,
    min_remaining: int = 6,
    max_remaining: int = 29,
) -> np.ndarray:
    """Select pre-step indices in a fixed transitions-to-terminal band."""
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 1
        or isinstance(min_remaining, bool)
        or not isinstance(min_remaining, int)
        or isinstance(max_remaining, bool)
        or not isinstance(max_remaining, int)
        or min_remaining < 1
        or max_remaining < min_remaining
        or max_remaining > step_count
    ):
        raise ValueError("invalid preterminal selection bounds")
    remaining = np.arange(step_count, 0, -1, dtype=np.int32)
    return np.flatnonzero(
        (remaining >= min_remaining) & (remaining <= max_remaining)
    )


def _finite_array(
    arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"bank is missing {name}")
    value = np.asarray(arrays[name])
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} does not match {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _integer_array(
    arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    value = _finite_array(arrays, name, shape)
    integer = value.astype(np.int32)
    if not np.array_equal(value, integer):
        raise ValueError(f"{name} must be integer-valued")
    return integer


def validate_history_bank(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_source_phases: tuple[int, ...],
    expected_survival: tuple[int, ...],
    history_len: int,
    frame_dim: int,
) -> dict[str, object]:
    """Validate the exact context-rich carried-state scientific contract."""
    if (
        len(expected_source_phases) == 0
        or len(expected_source_phases) != len(expected_survival)
        or len(set(expected_source_phases)) != len(expected_source_phases)
        or history_len < 1
        or frame_dim < 1
    ):
        raise ValueError("invalid expected carried-bank contract")
    selected_by_source = [
        select_preterminal_indices(steps) for steps in expected_survival
    ]
    rows_per_source = [int(indices.size) for indices in selected_by_source]
    rows = sum(rows_per_source)
    if rows < 1:
        raise ValueError("carried bank must contain rows")

    qpos = _finite_array(arrays, "qpos", (rows, 36))
    _finite_array(arrays, "qvel", (rows, 35))
    phase = _integer_array(arrays, "phase", (rows,))
    _finite_array(arrays, "last_act", (rows, 29))
    actor_history = _finite_array(
        arrays,
        "actor_obs_history",
        (rows, history_len, frame_dim),
    )
    fresh_frame = _finite_array(
        arrays, "fresh_actor_frame", (rows, frame_dim)
    )
    _finite_array(arrays, "action", (rows, 29))
    source_start = _integer_array(
        arrays, "source_start_phase", (rows,)
    )
    source_step = _integer_array(arrays, "source_step", (rows,))
    remaining = _integer_array(
        arrays, "transitions_to_terminal", (rows,)
    )
    terminal = _finite_array(arrays, "terminal", (rows,))
    termination_errors = _finite_array(
        arrays, "termination_errors", (rows, 4)
    )
    thresholds = _finite_array(
        arrays, "termination_thresholds", (4,)
    )
    if np.any(thresholds <= 0.0):
        raise ValueError("termination thresholds must be positive")
    if np.any(terminal > 0.5):
        raise ValueError("carried bank rows must be nonterminal")

    row_start = 0
    for expected_phase, expected_steps, expected_indices in zip(
        expected_source_phases,
        expected_survival,
        selected_by_source,
        strict=True,
    ):
        row_end = row_start + expected_indices.size
        source_slice = slice(row_start, row_end)
        if not np.array_equal(
            source_start[source_slice],
            np.full(expected_indices.size, expected_phase, dtype=np.int32),
        ):
            raise ValueError("source start phase does not match contract")
        if not np.array_equal(source_step[source_slice], expected_indices):
            raise ValueError("source step does not match survival contract")
        if not np.array_equal(
            phase[source_slice], expected_phase + expected_indices
        ):
            raise ValueError("phase does not follow the source rollout")
        if not np.array_equal(
            remaining[source_slice], expected_steps - expected_indices
        ):
            raise ValueError(
                "transitions-to-terminal does not match source survival"
            )
        row_start = row_end

    quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("carried bank root quaternions must be normalized")
    clearance = 1.0 - termination_errors / thresholds[None, :]
    minimum_clearance = float(np.min(clearance))
    if minimum_clearance <= 0.0:
        raise ValueError("carried bank rows cross hard termination limits")
    history_error = float(
        np.max(np.abs(actor_history[:, -1] - fresh_frame))
    )
    if history_error > 1e-10:
        raise ValueError("stored last history frame is inconsistent")
    return {
        "valid": True,
        "protocol": PROTOCOL,
        "rows": rows,
        "rows_per_source": rows_per_source,
        "source_phases": list(expected_source_phases),
        "source_survival": list(expected_survival),
        "minimum_transitions_to_terminal": int(np.min(remaining)),
        "maximum_transitions_to_terminal": int(np.max(remaining)),
        "minimum_hard_limit_clearance": minimum_clearance,
        "maximum_history_frame_error": history_error,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz_atomically(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _termination_errors(env, state) -> np.ndarray:
    body_pos, body_quat, _, _ = env._body_state(state.data)
    errors = env.termination_errors(
        phase=state.info["phase"],
        body_pos=body_pos,
        body_quat=body_quat,
    )
    return np.asarray(
        [
            errors["anchor_z_error"],
            errors["anchor_xy_error"],
            errors["gravity_z_error"],
            errors["distal_z_error"],
        ],
        dtype=np.float64,
    )


def _collect_source(
    env,
    action_fn: Callable,
    *,
    source_phase: int,
    expected_survival: int,
    seed: int,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    state = env.reset_at_phase(
        jax.random.PRNGKey(seed),
        jnp.array(0.0),
        jnp.array(source_phase),
    )
    records: dict[str, list[np.ndarray | int | float]] = {
        name: []
        for name in (
            "qpos",
            "qvel",
            "phase",
            "last_act",
            "actor_obs_history",
            "fresh_actor_frame",
            "action",
            "source_start_phase",
            "source_step",
            "transitions_to_terminal",
            "terminal",
            "termination_errors",
        )
    }
    for step in range(expected_survival):
        action = action_fn(state)
        records["qpos"].append(np.asarray(state.data.qpos))
        records["qvel"].append(np.asarray(state.data.qvel))
        records["phase"].append(int(state.info["phase"]))
        records["last_act"].append(np.asarray(state.info["last_act"]))
        records["actor_obs_history"].append(
            np.asarray(state.info["actor_obs_history"])
        )
        records["fresh_actor_frame"].append(
            np.asarray(env._get_actor_obs(state.data, state.info))
        )
        records["action"].append(np.asarray(action))
        records["source_start_phase"].append(source_phase)
        records["source_step"].append(step)
        records["transitions_to_terminal"].append(
            expected_survival - step
        )
        records["terminal"].append(0.0)
        records["termination_errors"].append(_termination_errors(env, state))
        state = env.step(state, action)
        if float(state.done) > 0.5:
            break
    if len(records["phase"]) != expected_survival:
        raise ValueError(
            f"source phase {source_phase} survived "
            f"{len(records['phase'])}, expected {expected_survival}"
        )
    if float(state.info["terminal"]) <= 0.5:
        raise ValueError(
            f"source phase {source_phase} did not end in a terminal transition"
        )
    selected = select_preterminal_indices(expected_survival)
    return {
        name: np.asarray(values)[selected]
        for name, values in records.items()
    }


def collect_bank(
    checkpoint_path: Path,
    reference_path: Path,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Collect the fixed E008 preterminal bank with the evaluation actor."""
    import jax.numpy as jnp

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
    )
    from src.core.data_structures import Normalizer
    from src.core.networks import Actor
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.evaluate_g1_flax_phase_grid import evaluate_actor_action
    from tools.evaluate_g1_tracking import (
        configure_jax,
        make_evaluation_env,
        scale_policy_action,
    )

    configure_jax()
    profile = get_solver_profile("g1-4x5")
    env = make_evaluation_env(
        "g1_tracking_rmr_50hz_source_step",
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference_path,
        reference_stride=1,
        actor_history_len=HISTORY_LEN,
        actor_reference_lookahead_steps=LOOKAHEAD_STEPS,
        actor_reference_preview_mode="delta",
        reference_residual_control=True,
        reference_residual_scale=0.5,
    )
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if not isinstance(checkpoint.actor_params, FrozenPreviewResidualParams):
        raise ValueError("checkpoint is not a frozen residual preview actor")
    actor = Actor(
        env.action_dim,
        hidden=(512, 256, 128),
        squash=getattr(env, "squash_actor_actions", True),
        layer_norm=True,
        zero_output=False,
    )
    residual_actor = PreviewResidualAdapter(
        action_dim=env.action_dim,
        hidden_dim=RESIDUAL_HIDDEN,
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    def action_fn(state):
        normalized = env.normalize_actor_obs(
            normalizer, checkpoint.normalizer, state.obs
        ).astype(jnp.float32)
        return scale_policy_action(
            evaluate_actor_action(
                actor,
                checkpoint.actor_params,
                normalized,
                residual_actor=residual_actor,
                history_len=HISTORY_LEN,
                treatment_frame_dim=env.actor_frame_obs_dim,
            ),
            1.0,
        ).astype(jnp.float64)

    sources = []
    with solver_context(profile):
        for phase, survival in zip(
            SOURCE_PHASES, SOURCE_SURVIVAL, strict=True
        ):
            sources.append(
                _collect_source(
                    env,
                    action_fn,
                    source_phase=phase,
                    expected_survival=survival,
                    seed=seed,
                )
            )
    names = tuple(sources[0])
    arrays = {
        name: np.concatenate([source[name] for source in sources], axis=0)
        for name in names
    }
    arrays["termination_thresholds"] = TERMINATION_THRESHOLDS.copy()
    return arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.resolve()
    reference_path = args.reference_path.resolve()
    for path, expected in (
        (checkpoint_path, args.checkpoint_sha256),
        (reference_path, args.reference_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    arrays = collect_bank(
        checkpoint_path,
        reference_path,
        seed=args.seed,
    )
    summary = validate_history_bank(
        arrays,
        expected_source_phases=SOURCE_PHASES,
        expected_survival=SOURCE_SURVIVAL,
        history_len=HISTORY_LEN,
        frame_dim=int(arrays["actor_obs_history"].shape[-1]),
    )
    output_npz = args.output_npz.resolve()
    output_json = args.output_json.resolve()
    _write_npz_atomically(output_npz, arrays)
    payload = {
        **summary,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": args.checkpoint_sha256,
        "reference_path": str(reference_path),
        "reference_sha256": args.reference_sha256,
        "bank_path": str(output_npz),
        "bank_sha256": _sha256(output_npz),
        "history_len": HISTORY_LEN,
        "actor_frame_obs_dim": int(
            arrays["actor_obs_history"].shape[-1]
        ),
        "lookahead_steps": list(LOOKAHEAD_STEPS),
        "preview_mode": "delta",
        "residual_hidden": RESIDUAL_HIDDEN,
        "solver_profile": "g1-4x5",
        "seed": args.seed,
    }
    _write_json_atomically(output_json, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
