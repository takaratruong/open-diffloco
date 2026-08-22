"""Audit a hard-forward, compliant-backward G1 SHAC transition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.evaluate_g1_ivw_h_gradients import (
    EXPECTED_CONTROLLER_SHA256,
    EXPECTED_INPUT_SHA256,
    EXPECTED_MODEL_SHA256,
    PHASES,
    REPLICAS_PER_PHASE,
    SOLVERS,
    TAPE_SEEDS,
    _action_standard_deviation,
    _atomic_json,
    _atomic_npz,
    _capture_one_population,
    _tree_matrix,
    _tree_vector,
    _validate_clean_source,
    _vector_cosine,
    build_fixed_phase_population,
)


PROTOCOL = "g1-backward-contact-compliance-v1"
HARD_TIME_CONSTANT = 0.02
SOFT_TIME_CONSTANT = 0.05
MODES = ("hard", "compliant")
OUTCOMES = frozenset(
    {
        "backward-compliance-robust",
        "backward-compliance-neutral",
        "backward-compliance-destructive",
        "invalid-execution",
    }
)
FORWARD_FIELDS = (
    "reward",
    "done",
    "terminal",
    "qpos",
    "qvel",
    "normalized_obs",
    "mean",
    "sampled_action",
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_summary(row: object) -> bool:
    if not isinstance(row, Mapping):
        return False
    scalars = (
        "pathwise_vjp_cosine_min",
        "pathwise_vjp_norm_ratio_min",
        "pathwise_vjp_norm_ratio_max",
        "finite_phase_count_min",
        "hard_mean_solver_cosine",
        "compliant_mean_solver_cosine",
        "hard_mean_tape_cosine",
        "compliant_mean_tape_cosine",
        "retained_hard_cosine",
        "retained_hard_norm_ratio",
    )
    vectors = (
        "hard_phase_solver_cosine",
        "compliant_phase_solver_cosine",
        "hard_phase_tape_cosine",
        "compliant_phase_tape_cosine",
    )
    if row.get("valid") is not True or row.get("forward_identical") is not True:
        return False
    if any(not _finite_number(row.get(name)) for name in scalars):
        return False
    if any(
        not isinstance(row.get(name), (list, tuple))
        or len(row[name]) != len(PHASES)
        or any(not _finite_number(value) for value in row[name])
        for name in vectors
    ):
        return False
    return (
        float(row["pathwise_vjp_cosine_min"]) >= 0.999
        and float(row["pathwise_vjp_norm_ratio_min"]) >= 0.999
        and float(row["pathwise_vjp_norm_ratio_max"]) <= 1.001
        and int(row["finite_phase_count_min"]) >= 16
    )


def classify_backward_contact_compliance(row: Mapping[str, object]) -> str:
    """Apply invalid, destructive, robust, then neutral precedence."""

    if not _valid_summary(row):
        return "invalid-execution"
    destructive = (
        float(row["retained_hard_cosine"]) < 0.5
        or not 0.25 <= float(row["retained_hard_norm_ratio"]) <= 4.0
        or any(
            float(compliant) < float(hard) - 0.05
            for hard, compliant in zip(
                row["hard_phase_solver_cosine"],
                row["compliant_phase_solver_cosine"],
            )
        )
        or any(
            float(compliant) < float(hard) - 0.05
            for hard, compliant in zip(
                row["hard_phase_tape_cosine"],
                row["compliant_phase_tape_cosine"],
            )
        )
    )
    if destructive:
        return "backward-compliance-destructive"
    robust = (
        float(row["compliant_mean_solver_cosine"])
        >= float(row["hard_mean_solver_cosine"]) + 0.05
        and float(row["compliant_mean_tape_cosine"])
        >= float(row["hard_mean_tape_cosine"]) + 0.05
    )
    return (
        "backward-compliance-robust"
        if robust
        else "backward-compliance-neutral"
    )


def validate_contact_model_delta(
    hard_solref: np.ndarray, compliant_solref: np.ndarray
) -> dict[str, object]:
    """Validate the exact registered 0.02-to-0.05 contact-only model delta."""

    hard = np.asarray(hard_solref, dtype=np.float64)
    compliant = np.asarray(compliant_solref, dtype=np.float64)
    if hard.ndim != 2 or hard.shape[1] != 2 or compliant.shape != hard.shape:
        raise ValueError("contact solref arrays must share shape (ngeom, 2)")
    if not np.isfinite(hard).all() or not np.isfinite(compliant).all():
        raise ValueError("contact solref arrays must be finite")
    if not np.array_equal(hard[:, 0], np.full(len(hard), HARD_TIME_CONSTANT)):
        raise ValueError("hard time constant does not match the registered model")
    if not np.array_equal(
        compliant[:, 0], np.full(len(compliant), SOFT_TIME_CONSTANT)
    ):
        raise ValueError("compliant time constant does not match the treatment")
    if not np.array_equal(hard[:, 1], compliant[:, 1]) or not np.array_equal(
        hard[:, 1], np.ones(len(hard))
    ):
        raise ValueError("contact dampratio must remain exactly one")
    return {
        "hard_time_constant": HARD_TIME_CONSTANT,
        "soft_time_constant": SOFT_TIME_CONSTANT,
        "dampratio": 1.0,
        "geom_count": int(len(hard)),
    }


def validate_forward_identity(
    hard: Mapping[str, object], surrogate: Mapping[str, object]
) -> bool:
    """Require every registered forward field to remain bit-identical."""

    for field in FORWARD_FIELDS:
        if field not in hard or field not in surrogate:
            raise ValueError(f"forward identity is missing {field}")
        if not np.array_equal(np.asarray(hard[field]), np.asarray(surrogate[field])):
            raise ValueError(f"forward identity drifted at {field}")
    return True


def validate_completion(path: Path) -> dict[str, object]:
    """Reopen a completion manifest and every hash-bound artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("valid") is not True
        or payload.get("protocol") != PROTOCOL
        or payload.get("outcome") not in OUTCOMES
    ):
        raise ValueError("completion contract is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("completion artifacts are missing")
    for name, expected_hash in artifacts.items():
        artifact = path.parent / str(name)
        if not artifact.is_file():
            raise ValueError(f"completion artifact is missing: {name}")
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"artifact hash mismatch for {name}")
    return payload


def _capture_auxiliary(
    capture: Mapping[str, object], *, hard_branch: bool = False
) -> dict[str, np.ndarray]:
    auxiliary = capture["auxiliary"]
    assert isinstance(auxiliary, Mapping)
    step_fields = {"reward", "done", "terminal", "qpos", "qvel"}
    return {
        field: np.asarray(
            auxiliary[f"hard_{field}"]
            if hard_branch and field in step_fields
            else auxiliary[field]
        )
        for field in FORWARD_FIELDS
    }


def run_audit(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    repository: Path,
    code_commit: str,
    output_directory: Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, object]:
    """Run the fixed fresh hard-versus-compliant backward gradient audit."""

    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.contact_compliance import with_contact_time_constant
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
        raise ValueError("backward-contact compliance seed must be zero")
    source = _validate_clean_source(repository.resolve(), code_commit)
    paths = {
        "checkpoint": checkpoint_path.resolve(),
        "hparams": hparams_path.resolve(),
        "reference": reference_path.resolve(),
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("backward-contact inputs do not match E023")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    if hparams.get("clip_sampled_actor_actions") is not False:
        raise ValueError("backward-contact audit requires unclipped E023 actions")
    if float(hparams.get("actor_bootstrap_scale", math.nan)) != 0.0:
        raise ValueError("backward-contact audit requires zero bootstrap")
    runtime = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    if (
        runtime["model_sha256"] != EXPECTED_MODEL_SHA256
        or runtime["controller_sha256"] != EXPECTED_CONTROLLER_SHA256
    ):
        raise ValueError("backward-contact runtime assets do not match E023")
    input_hashes.update(
        model=runtime["model_sha256"], controller=runtime["controller_sha256"]
    )

    population = build_fixed_phase_population(seed)
    phases = population["phase"]
    noise = population["noise"]
    solver_names = SOLVERS
    tape_indices = (0, 1)
    if smoke:
        selected = np.flatnonzero(phases == 100)[:8]
        phases = phases[selected]
        noise = noise[:1, selected]
        solver_names = SOLVERS[:1]
        tape_indices = (0,)
    count = int(len(phases))
    keys = jax.random.split(jax.random.PRNGKey(seed), count)

    captures: dict[tuple[str, str, int], dict[str, object]] = {}
    initial_arrays: dict[str, np.ndarray] | None = None
    actor_parameters = None
    model_delta: dict[str, object] | None = None
    for solver_name in solver_names:
        profile = get_solver_profile(solver_name)
        solver_hparams = {
            **hparams,
            "solver_iterations": profile.iterations,
            "solver_ls_iterations": profile.ls_iterations,
        }
        hard_env = _build_environment(solver_hparams, reference_path)
        compliant_env = copy.copy(hard_env)
        compliant_env.mjx_model = with_contact_time_constant(
            hard_env.mjx_model, SOFT_TIME_CONSTANT
        )
        current_delta = validate_contact_model_delta(
            np.asarray(hard_env.mjx_model.geom_solref),
            np.asarray(compliant_env.mjx_model.geom_solref),
        )
        if model_delta is None:
            model_delta = current_delta
        elif current_delta != model_delta:
            raise ValueError("contact model delta differs across solvers")
        with solver_context(profile):
            states = jax.jit(jax.vmap(hard_env.reset_at_phase))(
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

        actor, parameters, normalizer_state = _load_policy(
            hard_env,
            None,
            seed,
            actor_hidden=tuple(hparams["actor_hidden"]),
            actor_layer_norm=bool(hparams["actor_layer_norm"]),
            actor_zero_output=bool(hparams["actor_zero_output"]),
            training_initialization=True,
        )
        if actor_parameters is None:
            actor_parameters = parameters
        elif not np.array_equal(
            _tree_vector(parameters), _tree_vector(actor_parameters)
        ):
            raise ValueError("fresh actor parameters differ across solvers")
        normalizer = Normalizer(hard_env.actor_frame_obs_dim)
        sigma = _action_standard_deviation(hparams["action_noise_std_start"])
        for tape_index in tape_indices:
            for mode in MODES:
                with solver_context(profile):
                    capture = _capture_one_population(
                        env=hard_env,
                        gradient_env=(compliant_env if mode == "compliant" else None),
                        actor=actor,
                        parameters=parameters,
                        normalizer=normalizer,
                        normalizer_state=normalizer_state,
                        states=states,
                        epsilon=noise[tape_index],
                        phases=phases,
                        sigma=sigma,
                        gamma=float(hparams["gamma"]),
                        chunk_size=(count if smoke else REPLICAS_PER_PHASE),
                    )
                captures[(mode, solver_name, tape_index)] = capture
            compliant_capture = captures[("compliant", solver_name, tape_index)]
            validate_forward_identity(
                _capture_auxiliary(compliant_capture, hard_branch=True),
                _capture_auxiliary(compliant_capture),
            )
            try:
                validate_forward_identity(
                    _capture_auxiliary(captures[("hard", solver_name, tape_index)]),
                    _capture_auxiliary(compliant_capture, hard_branch=True),
                )
            except ValueError as exc:
                raise ValueError(
                    "separately compiled hard capture drifted from the compliant "
                    "capture's internal hard branch"
                ) from exc

    assert initial_arrays is not None and model_delta is not None
    if smoke:
        parity_rows = [captures[(mode, solver_names[0], 0)]["parity"] for mode in MODES]
        parity_cosines = np.concatenate(
            [
                np.asarray(row["cosine"])[np.asarray(row["finite"])]
                for row in parity_rows
            ]
        )
        parity_ratios = np.concatenate(
            [
                np.asarray(row["norm_ratio"])[np.asarray(row["finite"])]
                for row in parity_rows
            ]
        )
        report = {
            "valid": bool(
                len(parity_cosines) >= 4
                and np.min(parity_cosines) >= 0.999
                and np.min(parity_ratios) >= 0.999
                and np.max(parity_ratios) <= 1.001
            ),
            "scientific": False,
            "protocol": f"{PROTOCOL}-smoke",
            "forward_identical": True,
            "model_delta": model_delta,
            "pathwise_vjp_cosine_min": float(np.min(parity_cosines)),
            "pathwise_vjp_norm_ratio_min": float(np.min(parity_ratios)),
            "pathwise_vjp_norm_ratio_max": float(np.max(parity_ratios)),
            **source,
        }
        if not report["valid"]:
            raise ValueError(f"backward-contact smoke failed: {report}")
        _atomic_json(output_directory.resolve() / "smoke_summary.json", report)
        return report

    arrays: dict[str, np.ndarray] = {
        "phase": np.asarray(phases, dtype=np.int32),
        "noise": np.asarray(noise, dtype=np.float32),
        "initial_qpos": initial_arrays["qpos"],
        "initial_qvel": initial_arrays["qvel"],
        "initial_actor_obs_history": initial_arrays["history"],
    }
    aggregate_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    task_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    finite_phase_counts: list[int] = []
    parity_cosines: list[np.ndarray] = []
    parity_ratios: list[np.ndarray] = []
    for key, capture in captures.items():
        mode, solver_name, tape_index = key
        gradient = capture["gradients"]["ordinary"]
        aggregated = aggregate_audit_direction(
            gradient,
            phases,
            phase_count=125,
            clip_norm=1.0,
            alpha=0.5,
            iterations=32,
        )
        aggregate_vectors[key] = _tree_vector(aggregated.combined_gradient)
        task_vectors[key] = _tree_matrix(aggregated.task_gradients)
        prefix = f"{mode}_{solver_name}_tape{tape_index}".replace("-", "_")
        arrays[f"{prefix}_combined"] = aggregate_vectors[key]
        arrays[f"{prefix}_task"] = task_vectors[key]
        stats = per_env_gradient_statistics(gradient)
        arrays[f"{prefix}_env_norm"] = np.asarray(stats["raw_norm_by_env"])
        parity = capture["parity"]
        finite = np.asarray(parity["finite"], dtype=np.bool_)
        arrays[f"{prefix}_pathwise_vjp_cosine"] = np.asarray(parity["cosine"])
        arrays[f"{prefix}_pathwise_vjp_norm_ratio"] = np.asarray(
            parity["norm_ratio"]
        )
        parity_cosines.append(np.asarray(parity["cosine"])[finite])
        parity_ratios.append(np.asarray(parity["norm_ratio"])[finite])
        finite_phase_counts.extend(capture["finite_phase_counts"]["ordinary"])
    nominal_hard = _capture_auxiliary(captures[("hard", SOLVERS[0], 0)])
    nominal_compliant = _capture_auxiliary(
        captures[("compliant", SOLVERS[0], 0)]
    )
    for field in FORWARD_FIELDS:
        arrays[f"nominal_hard_{field}"] = nominal_hard[field]
        arrays[f"nominal_compliant_{field}"] = nominal_compliant[field]

    def reliability(mode: str):
        solver = [
            _vector_cosine(
                aggregate_vectors[(mode, SOLVERS[0], tape)],
                aggregate_vectors[(mode, SOLVERS[1], tape)],
            )
            for tape in (0, 1)
        ]
        tape = [
            _vector_cosine(
                aggregate_vectors[(mode, solver_name, 0)],
                aggregate_vectors[(mode, solver_name, 1)],
            )
            for solver_name in SOLVERS
        ]
        phase_solver = np.mean(
            [
                [
                    _vector_cosine(
                        task_vectors[(mode, SOLVERS[0], tape_id)][phase_id],
                        task_vectors[(mode, SOLVERS[1], tape_id)][phase_id],
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
                        task_vectors[(mode, solver_name, 0)][phase_id],
                        task_vectors[(mode, solver_name, 1)][phase_id],
                    )
                    for phase_id in range(5)
                ]
                for solver_name in SOLVERS
            ],
            axis=0,
        )
        return solver, tape, phase_solver, phase_tape

    hard_solver, hard_tape, hard_phase_solver, hard_phase_tape = reliability("hard")
    compliant_solver, compliant_tape, compliant_phase_solver, compliant_phase_tape = (
        reliability("compliant")
    )
    nominal_hard_gradient = aggregate_vectors[("hard", SOLVERS[0], 0)]
    nominal_compliant_gradient = aggregate_vectors[("compliant", SOLVERS[0], 0)]
    all_parity_cosines = np.concatenate(parity_cosines)
    all_parity_ratios = np.concatenate(parity_ratios)
    summary: dict[str, object] = {
        "valid": True,
        "forward_identical": True,
        "pathwise_vjp_cosine_min": float(np.min(all_parity_cosines)),
        "pathwise_vjp_norm_ratio_min": float(np.min(all_parity_ratios)),
        "pathwise_vjp_norm_ratio_max": float(np.max(all_parity_ratios)),
        "finite_phase_count_min": int(min(finite_phase_counts)),
        "hard_solver_cosine": hard_solver,
        "compliant_solver_cosine": compliant_solver,
        "hard_tape_cosine": hard_tape,
        "compliant_tape_cosine": compliant_tape,
        "hard_mean_solver_cosine": float(np.mean(hard_solver)),
        "compliant_mean_solver_cosine": float(np.mean(compliant_solver)),
        "hard_mean_tape_cosine": float(np.mean(hard_tape)),
        "compliant_mean_tape_cosine": float(np.mean(compliant_tape)),
        "hard_phase_solver_cosine": hard_phase_solver.tolist(),
        "compliant_phase_solver_cosine": compliant_phase_solver.tolist(),
        "hard_phase_tape_cosine": hard_phase_tape.tolist(),
        "compliant_phase_tape_cosine": compliant_phase_tape.tolist(),
        "retained_hard_cosine": _vector_cosine(
            nominal_hard_gradient, nominal_compliant_gradient
        ),
        "retained_hard_norm_ratio": float(
            np.linalg.norm(nominal_compliant_gradient)
            / np.linalg.norm(nominal_hard_gradient)
        ),
    }
    outcome = classify_backward_contact_compliance(summary)
    if outcome == "invalid-execution":
        raise ValueError(f"backward-contact evidence is invalid: {summary}")

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
            "input_sha256": input_hashes,
            "model_delta": model_delta,
            "solvers": list(SOLVERS),
            "tape_seeds": list(TAPE_SEEDS),
            "seed": seed,
            **source,
        },
    )
    _atomic_npz(evidence_path, arrays)
    _atomic_json(
        summary_path,
        {
            **summary,
            "protocol": PROTOCOL,
            "outcome": outcome,
            "model_delta": model_delta,
            "gradient_evidence_sha256": sha256_file(evidence_path),
            "input_sha256": input_hashes,
            **source,
        },
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].bar(
        ["hard", "compliant"],
        [summary["hard_mean_solver_cosine"], summary["compliant_mean_solver_cosine"]],
        color=["#777777", "#2a78c5"],
    )
    axes[0].set_title("paired solver")
    axes[1].bar(
        ["hard", "compliant"],
        [summary["hard_mean_tape_cosine"], summary["compliant_mean_tape_cosine"]],
        color=["#999999", "#39a96b"],
    )
    axes[1].set_title("independent noise tape")
    for axis in axes:
        axis.set_ylim(-1.0, 1.0)
        axis.set_ylabel("gradient cosine")
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
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        reference_path=args.reference_path,
        repository=args.repository,
        code_commit=args.code_commit,
        output_directory=args.output_directory,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
