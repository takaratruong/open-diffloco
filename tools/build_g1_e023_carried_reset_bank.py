"""Build a history-faithful pre-failure reset bank from E023."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from tools.build_g1_history_carried_reset_bank import (
    HISTORY_LEN,
    LOOKAHEAD_STEPS,
    TERMINATION_THRESHOLDS,
    _collect_source,
    _write_json_atomically,
    _write_npz_atomically,
    validate_history_bank,
    validate_observed_survival,
)
from tools.run_g1_root_recovery_continuation import validate_runtime_assets


PROTOCOL = "g1-e023-history-carried-reset-bank-v1"
E023_SOURCE_PHASES = (0, 50)
E023_TOTAL_STEPS = 1_572_864
EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)

_E023_HPARAMS: dict[str, object] = {
    "total_steps": E023_TOTAL_STEPS,
    "env_variant": "g1_tracking_rmr_50hz_action_parity",
    "actor_hidden": [512, 256, 128],
    "actor_layer_norm": True,
    "actor_zero_output": True,
    "actor_history_len": HISTORY_LEN,
    "actor_reference_lookahead_steps": list(LOOKAHEAD_STEPS),
    "actor_reference_preview_mode": "delta",
    "reference_residual_control": True,
    "reference_residual_scale": 1.0,
    "reference_reset_noise_scale": 0.0,
    "domain_randomization": False,
    "actor_observation_noise": False,
    "squash_actor_mean": False,
    "clip_sampled_actor_actions": False,
    "solver_profile": "g1-4x5",
    "solver_iterations": 4,
    "solver_ls_iterations": 5,
    "reference_stride": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_e023_hparams(hparams: Mapping[str, Any]) -> dict[str, object]:
    """Require the exact E023 actor/environment contract used by collection."""
    for key, expected in _E023_HPARAMS.items():
        actual = hparams.get(key)
        if isinstance(expected, list):
            try:
                actual = list(actual)
            except TypeError as error:
                raise ValueError("E023 hparams do not match collection") from error
        if actual != expected:
            raise ValueError(
                f"E023 hparams do not match collection at {key}: "
                f"{actual!r} != {expected!r}"
            )
    return dict(_E023_HPARAMS)


def build_e023_bank_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    observed_survival: tuple[int, ...],
    frame_dim: int,
) -> dict[str, object]:
    """Validate exactly two 24-row E023 pre-failure bands."""
    if len(observed_survival) != len(E023_SOURCE_PHASES):
        raise ValueError("E023 bank requires exactly two source survivals")
    observed = validate_observed_survival(
        observed_survival, source_count=len(E023_SOURCE_PHASES)
    )
    summary = validate_history_bank(
        arrays,
        expected_source_phases=E023_SOURCE_PHASES,
        expected_survival=observed,
        history_len=HISTORY_LEN,
        frame_dim=frame_dim,
    )
    if summary["rows"] != 48 or summary["rows_per_source"] != [24, 24]:
        raise ValueError("E023 bank must contain two exact 24-row bands")
    return {**summary, "protocol": PROTOCOL}


def validate_code_commit(repository: Path, expected_commit: str) -> str:
    """Require the collector's exact clean registered source commit."""
    if len(expected_commit) != 40:
        raise ValueError("code commit must be a full Git SHA-1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError("collector code commit does not match registration")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tools"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("collector source checkout must be clean")
    return head


def collect_e023_bank(
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    *,
    seed: int,
    source_phases: tuple[int, ...] = E023_SOURCE_PHASES,
    require_reference_match: bool = True,
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    """Collect E023 pre-failure actor contexts on the requested reference."""
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
    )
    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.evaluate_g1_flax_phase_grid import (
        evaluate_actor_action,
        prepare_phase_grid_action,
    )
    from tools.evaluate_g1_tracking import (
        _load_policy,
        build_compiled_step,
        configure_jax,
        make_evaluation_env,
    )

    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    contract = validate_e023_hparams(hparams)
    if not source_phases or len(set(source_phases)) != len(source_phases):
        raise ValueError("source phases must be a nonempty unique tuple")
    if any(not isinstance(phase, int) or phase < 0 for phase in source_phases):
        raise ValueError("source phases must be nonnegative integers")
    if (
        require_reference_match
        and Path(str(hparams.get("reference_path", ""))).resolve()
        != reference_path
    ):
        raise ValueError("E023 hparams reference path does not match input")
    configure_jax()
    profile = get_solver_profile("g1-4x5")
    env = make_evaluation_env(
        str(contract["env_variant"]),
        solver_iterations=int(contract["solver_iterations"]),
        solver_ls_iterations=int(contract["solver_ls_iterations"]),
        reference_path=reference_path,
        reference_stride=int(contract["reference_stride"]),
        actor_history_len=int(contract["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            contract["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(
            contract["actor_reference_preview_mode"]
        ),
        actor_observation_noise=False,
        domain_randomization=False,
        reference_reset_noise_scale=0.0,
        reference_residual_control=True,
        reference_residual_scale=1.0,
    )
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if isinstance(checkpoint.actor_params, FrozenPreviewResidualParams):
        raise ValueError("E023 checkpoint must contain a plain Flax actor")
    if int(np.asarray(checkpoint.step)) != E023_TOTAL_STEPS:
        raise ValueError("E023 checkpoint step does not match selected parent")
    for leaf in jax.tree_util.tree_leaves(checkpoint.actor_params):
        if not np.isfinite(np.asarray(leaf)).all():
            raise ValueError("E023 actor contains nonfinite parameters")
    actor, actor_params, normalizer_state = _load_policy(
        env, checkpoint_path, seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)
    compiled_step = build_compiled_step(env)

    def action_fn(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        action = evaluate_actor_action(
            actor, actor_params, normalized
        )
        return prepare_phase_grid_action(
            action,
            clip_sampled_actor_actions=bool(
                contract["clip_sampled_actor_actions"]
            ),
        ).astype(jnp.float64)

    sources = []
    survival = []
    with solver_context(profile):
        for phase in source_phases:
            source, observed = _collect_source(
                env,
                action_fn,
                source_phase=phase,
                seed=seed,
                step_fn=compiled_step,
            )
            sources.append(source)
            survival.append(observed)
    arrays = {
        name: np.concatenate([source[name] for source in sources], axis=0)
        for name in sources[0]
    }
    arrays["termination_thresholds"] = TERMINATION_THRESHOLDS.copy()
    return arrays, tuple(survival)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--hparams-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    hparams = args.hparams.resolve()
    reference = args.reference_path.resolve()
    for path, expected in (
        (checkpoint, args.checkpoint_sha256),
        (hparams, args.hparams_sha256),
        (reference, args.reference_sha256),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"input SHA-256 mismatch: {path}")
    if args.reference_sha256 != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference SHA-256 is not the registered walk")
    code_commit = validate_code_commit(repository, args.code_commit)
    hparams_payload = json.loads(hparams.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams_payload)
    runtime_assets = validate_runtime_assets(
        Path(str(hparams_payload["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    arrays, observed_survival = collect_e023_bank(
        checkpoint, hparams, reference, seed=args.seed
    )
    frame_dim = int(arrays["actor_obs_history"].shape[-1])
    summary = build_e023_bank_summary(
        arrays,
        observed_survival=observed_survival,
        frame_dim=frame_dim,
    )
    output_npz = args.output_npz.resolve()
    _write_npz_atomically(output_npz, arrays)
    payload = {
        **summary,
        **runtime_assets,
        "code_commit": code_commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "hparams_path": str(hparams),
        "hparams_sha256": args.hparams_sha256,
        "reference_path": str(reference),
        "reference_sha256": args.reference_sha256,
        "bank_path": str(output_npz),
        "bank_sha256": _sha256(output_npz),
        "history_len": HISTORY_LEN,
        "actor_frame_obs_dim": frame_dim,
        "lookahead_steps": list(LOOKAHEAD_STEPS),
        "preview_mode": "delta",
        "solver_profile": "g1-4x5",
        "seed": args.seed,
    }
    _write_json_atomically(args.output_json.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
