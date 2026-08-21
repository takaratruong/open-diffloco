"""Calibrate the G1 AHAC contact threshold from the frozen E023 actor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_zero_assistance_consolidation import (
    _git_output,
    _write_json_atomically,
)


CALIBRATION_PHASES = (0, 25, 50, 75, 100)
QUANTILE = 0.9
QUANTILE_METHOD = "linear"
PROVENANCE_KEYS = (
    "checkpoint_sha256",
    "reference_sha256",
    "model_sha256",
    "controller_sha256",
    "code_commit",
)


def _finite_signals(signals_by_phase: dict[int, list[float]]) -> np.ndarray:
    if tuple(sorted(signals_by_phase)) != CALIBRATION_PHASES:
        raise ValueError("calibration phase grid is not exact")
    vectors = []
    for phase in CALIBRATION_PHASES:
        values = np.asarray(signals_by_phase[phase], dtype=np.float64)
        if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
            raise ValueError(f"phase {phase} contact signals must be finite")
        if np.any(values < 0.0):
            raise ValueError(f"phase {phase} contact signals must be nonnegative")
        vectors.append(values)
    return np.concatenate(vectors)


def _validate_provenance(provenance: dict[str, object]) -> None:
    if set(PROVENANCE_KEYS) - set(provenance):
        raise ValueError("calibration provenance is incomplete")
    for key in PROVENANCE_KEYS:
        expected_length = 40 if key == "code_commit" else 64
        value = provenance[key]
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"calibration provenance {key} is invalid")


def build_calibration_payload(
    signals_by_phase: dict[int, list[float]],
    *,
    provenance: dict[str, str],
) -> dict[str, Any]:
    """Build one deterministic, inspectable linear-P90 calibration record."""
    all_signals = _finite_signals(signals_by_phase)
    _validate_provenance(provenance)
    threshold = float(np.quantile(all_signals, QUANTILE, method=QUANTILE_METHOD))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("calibrated contact threshold must be positive and finite")
    payload = {
        "protocol": "g1-ahac-contact-calibration-v1",
        "valid": True,
        "phases": list(CALIBRATION_PHASES),
        "quantile": QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "signals_by_phase": {
            str(phase): [float(value) for value in signals_by_phase[phase]]
            for phase in CALIBRATION_PHASES
        },
        "sample_count_by_phase": {
            str(phase): len(signals_by_phase[phase])
            for phase in CALIBRATION_PHASES
        },
        "sample_count": int(all_signals.size),
        "threshold": threshold,
        "provenance": dict(provenance),
    }
    validate_calibration_payload(payload)
    return payload


def validate_calibration_payload(payload: dict[str, object]) -> float:
    """Fail closed on a malformed, drifted, or nonfinite calibration."""
    if payload.get("valid") is not True:
        raise ValueError("calibration is not valid")
    if payload.get("phases") != list(CALIBRATION_PHASES):
        raise ValueError("calibration phase grid is not exact")
    if (
        payload.get("quantile") != QUANTILE
        or payload.get("quantile_method") != QUANTILE_METHOD
    ):
        raise ValueError("calibration quantile contract drifted")
    raw = payload.get("signals_by_phase")
    if not isinstance(raw, dict):
        raise ValueError("calibration phase signals are missing")
    expected_keys = {str(phase) for phase in CALIBRATION_PHASES}
    if set(raw) != expected_keys:
        raise ValueError("calibration phase signals are incomplete")
    signals = {
        phase: raw[str(phase)]
        for phase in CALIBRATION_PHASES
    }
    all_signals = _finite_signals(signals)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("calibration provenance is missing")
    _validate_provenance(provenance)
    expected = float(np.quantile(all_signals, QUANTILE, method=QUANTILE_METHOD))
    actual = payload.get("threshold")
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or float(actual) <= 0.0
        or float(actual) != expected
    ):
        raise ValueError("calibration threshold does not match raw signals")
    counts = payload.get("sample_count_by_phase")
    if counts != {str(phase): len(signals[phase]) for phase in CALIBRATION_PHASES}:
        raise ValueError("calibration phase sample counts do not match")
    if payload.get("sample_count") != int(all_signals.size):
        raise ValueError("calibration sample count does not match")
    return float(actual)


def publish_calibration(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish a validated calibration JSON document."""
    validate_calibration_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(path, payload)


def collect_contact_signals(
    *, checkpoint: Path, hparams: dict[str, object]
) -> dict[int, list[float]]:
    """Replay the frozen actor and collect one signal per active transition."""
    import jax
    import jax.numpy as jnp

    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.evaluate_g1_tracking import (
        _load_policy,
        make_evaluation_env,
        remaining_reference_transitions,
        reset_evaluation_state,
    )

    profile = get_solver_profile(str(hparams["solver_profile"]))
    env = make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        reference_path=str(hparams["reference_path"]),
        reference_stride=int(hparams["reference_stride"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            int(value) for value in hparams["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(hparams["actor_reference_preview_mode"]),
        actor_observation_noise=False,
        domain_randomization=False,
        reference_reset_noise_scale=0.0,
        reference_residual_control=bool(hparams["reference_residual_control"]),
        reference_residual_scale=float(hparams["reference_residual_scale"]),
    )
    actor, actor_params, normalizer_state = _load_policy(env, checkpoint, 0)
    compiled_step = jax.jit(env.step)
    signals: dict[int, list[float]] = {}
    for phase in CALIBRATION_PHASES:
        state = reset_evaluation_state(
            env,
            reset_key=jax.random.PRNGKey(0),
            difficulty=jnp.asarray(0.0, dtype=jnp.float64),
            phase=phase,
            sample_training_reset=False,
            profile=profile,
        )
        phase_signals = []
        for _ in range(
            remaining_reference_transitions(
                env.reference_length, phase, env.reference_stride
            )
        ):
            normalized = env.normalize_actor_obs(
                Normalizer(env.actor_frame_obs_dim), normalizer_state, state.obs
            ).astype(jnp.float32)
            action = actor.apply(actor_params, normalized).astype(jnp.float64)
            with solver_context(profile):
                state = compiled_step(state, action)
            value = float(state.info["transition_contact_stiffness"])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("runtime contact stiffness is nonfinite")
            phase_signals.append(value)
            if float(state.done) > 0.5:
                break
        signals[phase] = phase_signals
    return signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    import jax

    args = build_parser().parse_args()
    jax.config.update("jax_enable_x64", True)
    repository = Path(__file__).resolve().parents[1]
    head = _git_output(repository, "rev-parse", "HEAD")
    if args.code_commit != head or _git_output(repository, "status", "--porcelain"):
        raise ValueError("calibration requires the registered clean code commit")
    checkpoint = args.checkpoint.resolve()
    hparams_path = checkpoint.with_name("hparams.json")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    reference = args.reference_path.resolve()
    model = args.model_path.resolve()
    controller = args.controller_path.resolve()
    if Path(str(hparams["reference_path"])).resolve() != reference:
        raise ValueError("checkpoint reference path does not match calibration")
    if Path(str(hparams["xml_path"])).resolve() != model:
        raise ValueError("checkpoint model path does not match calibration")
    provenance = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "reference_sha256": sha256_file(reference),
        "model_sha256": sha256_file(model),
        "controller_sha256": sha256_file(controller),
        "code_commit": head,
    }
    signals = collect_contact_signals(checkpoint=checkpoint, hparams=hparams)
    payload = build_calibration_payload(signals, provenance=provenance)
    publish_calibration(args.output.resolve(), payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
