"""Evaluate and select zero-head recovery-feature SHAC checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
from statistics import median
import subprocess
from typing import Mapping, Sequence

import numpy as np

from tools.evaluate_g1_e038_recovery_transfer import (
    _load_all_bank_rows,
    parameter_tree_sha256,
    validate_e023_hparams,
    validate_paired_evidence,
    validate_parent_checkpoint,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_lafan_anchored_carried_recovery import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
)
from tools.run_g1_rmr_noise_h24_continuation import (
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
)


BANK_ROWS = 120
HORIZON = 32
ORDINARY_FLOORS = (116, 63, 49, 39, 47)
COMPLETE_SUFFIXES = (499, 399, 299, 199, 99)
EXPECTED_BANK_SHA256 = "d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5"
EVALUATION_CHECKPOINTS = {
    1_671_168: 8,
    1_769_472: 16,
    1_966_080: 32,
    2_359_296: 64,
}
_INPUT_HASH_NAMES = frozenset(
    {
        "parent_checkpoint",
        "parent_hparams",
        "candidate_checkpoint",
        "reference",
        "source_bank",
        "model",
        "controller",
    }
)
PROTOCOL = "g1-zero-head-feature-transfer-carried-evaluation-v1"


def _integer_vector(
    values: object, *, length: int, maximum: int | None = None
) -> tuple[int, ...]:
    if (
        not isinstance(values, (list, tuple))
        or len(values) != length
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (maximum is not None and value > maximum)
            for value in values
        )
    ):
        raise ValueError("zero-head feature selection record is invalid")
    return tuple(values)


def select_checkpoint(
    records: Sequence[Mapping[str, object]],
    *,
    parent_survival: Sequence[int],
) -> dict[str, object]:
    """Select only carried-state-safe checkpoints, then maximize recovery."""
    parent = _integer_vector(
        list(parent_survival), length=BANK_ROWS, maximum=HORIZON
    )
    normalized: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for record in records:
        update = record.get("update")
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError("zero-head feature selection record is invalid")
        carried = _integer_vector(
            record.get("carried_survival"),
            length=BANK_ROWS,
            maximum=HORIZON,
        )
        ordinary = _integer_vector(
            record.get("ordinary_survival"), length=len(ORDINARY_FLOORS)
        )
        normalized.append((update, carried, ordinary))
    eligible = [
        row
        for row in normalized
        if all(value >= floor for value, floor in zip(row[1], parent, strict=True))
    ]
    improved = [
        row
        for row in eligible
        if any(value > floor for value, floor in zip(row[1], parent, strict=True))
        or any(
            value > floor
            for value, floor in zip(row[2], ORDINARY_FLOORS, strict=True)
        )
    ]
    if not improved:
        return {
            "valid": True,
            "outcome": "zero-head-features-insufficient",
            "eligible_updates": [row[0] for row in eligible],
            "selected_update": None,
            "selected_carried_survival": None,
            "selected_ordinary_survival": None,
        }

    def key(row):
        update, carried, ordinary = row
        return (
            sum(value >= HORIZON for value in carried[:24]),
            sum(value >= HORIZON for value in carried),
            min(carried),
            median(carried),
            sum(carried) / len(carried),
            min(ordinary),
            median(ordinary),
            sum(ordinary) / len(ordinary),
            -update,
        )

    update, carried, ordinary = max(improved, key=key)
    solved = all(value >= HORIZON for value in carried) and all(
        value >= target
        for value, target in zip(ordinary, COMPLETE_SUFFIXES, strict=True)
    )
    return {
        "valid": True,
        "outcome": (
            "zero-head-features-solve"
            if solved
            else "zero-head-features-advance"
        ),
        "eligible_updates": [row[0] for row in eligible],
        "selected_update": update,
        "selected_carried_survival": list(carried),
        "selected_ordinary_survival": list(ordinary),
    }


def _zero_seed(value: str) -> int:
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("E041 evaluation requires seed zero")
    return seed


def _sha256(value: str) -> str:
    if len(value) != 64:
        raise argparse.ArgumentTypeError("SHA-256 must contain 64 characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise argparse.ArgumentTypeError("SHA-256 must be hexadecimal") from error
    return value


def _git_sha1(value: str) -> str:
    if len(value) != 40:
        raise argparse.ArgumentTypeError("Git commit must contain 40 characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Git commit must be hexadecimal") from error
    return value


def _registered_step(value: str) -> int:
    step = int(value)
    if step not in EVALUATION_CHECKPOINTS:
        raise argparse.ArgumentTypeError(
            "candidate step is not registered for E041 selection"
        )
    return step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-hparams", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-sha256", type=_sha256, required=True)
    parser.add_argument("--candidate-step", type=_registered_step, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", type=_git_sha1, required=True)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--seed", type=_zero_seed, default=0)
    return parser


def _write_npz_atomically(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_digest_map(values: object) -> dict[str, str]:
    if not isinstance(values, Mapping) or set(values) != _INPUT_HASH_NAMES:
        raise ValueError("input hash manifest is not exact")
    result = dict(values)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in result.values()
    ):
        raise ValueError("input hash manifest is not exact")
    return result


def validate_code_provenance(expected_commit: str) -> dict[str, str]:
    """Require the exact clean evaluator source commit."""
    repository = Path(__file__).resolve().parents[1]
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise ValueError("evaluator code commit must be a full hexadecimal SHA-1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError("evaluator code commit does not match checkout")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "src", "tools"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "tools",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if diff or untracked:
        raise ValueError("evaluator source checkout has executable dirty changes")
    return {
        "repository": str(repository),
        "code_commit": head,
        "dirty_patch_sha256": hashlib.sha256(diff).hexdigest(),
    }


def validate_candidate_checkpoint(
    candidate_state,
    *,
    candidate_step: int,
    parent_params,
    parent_normalizer,
) -> dict[str, object]:
    """Require a frozen-parent 328-256-29 residual at one registered update."""
    import jax

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        split_residual_adapter_params,
    )

    if candidate_step not in EVALUATION_CHECKPOINTS:
        raise ValueError("candidate step is not on the E041 checkpoint grid")
    if (
        not hasattr(candidate_state, "step")
        or int(np.asarray(candidate_state.step)) != candidate_step
        or not isinstance(candidate_state.actor_params, FrozenPreviewResidualParams)
    ):
        raise ValueError("candidate checkpoint does not match E041 structure")
    parent_exact = parameter_tree_sha256(
        candidate_state.actor_params.parent
    ) == parameter_tree_sha256(parent_params)
    normalizer_exact = parameter_tree_sha256(
        candidate_state.normalizer
    ) == parameter_tree_sha256(parent_normalizer)
    dense0, auxiliary = split_residual_adapter_params(
        candidate_state.actor_params.adapter
    )
    shape_exact = bool(
        dense0.shape == (328, 256)
        and auxiliary.dense0_bias.shape == (256,)
        and auxiliary.dense1_kernel.shape == (256, 29)
        and auxiliary.dense1_bias.shape == (29,)
    )
    finite = all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(candidate_state.actor_params)
    )
    valid = parent_exact and normalizer_exact and shape_exact and finite
    if not valid:
        raise ValueError("candidate checkpoint frozen-state contract failed")
    return {
        "valid": True,
        "candidate_step": candidate_step,
        "candidate_update": EVALUATION_CHECKPOINTS[candidate_step],
        "parent_parameters_exact": parent_exact,
        "normalizer_exact": normalizer_exact,
        "adapter_shape_exact": shape_exact,
        "parameters_finite": bool(finite),
    }


def publish_evaluation(
    *,
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Publish complete paired carried evidence and a manifest last."""
    validation = validate_paired_evidence(arrays)
    input_hashes = _validate_digest_map(provenance.get("input_sha256"))
    code_provenance = provenance.get("code_provenance")
    if (
        not isinstance(code_provenance, Mapping)
        or set(code_provenance)
        != {"repository", "code_commit", "dirty_patch_sha256"}
    ):
        raise ValueError("code provenance manifest is not exact")
    candidate_validation = provenance.get("candidate_validation")
    if (
        not isinstance(candidate_validation, Mapping)
        or candidate_validation.get("valid") is not True
        or candidate_validation.get("candidate_step")
        != provenance.get("candidate_step")
        or candidate_validation.get("candidate_update")
        != EVALUATION_CHECKPOINTS.get(provenance.get("candidate_step"))
    ):
        raise ValueError("candidate validation manifest is not exact")
    parent = np.asarray(validation["parent_survival"], dtype=np.int32)
    candidate = np.asarray(validation["expert_survival"], dtype=np.int32)
    output_directory = output_directory.resolve()
    evidence_path = output_directory / "paired_rollouts.npz"
    summary_path = output_directory / "summary.json"
    _write_npz_atomically(evidence_path, arrays)
    manifest = {
        "valid": True,
        "protocol": PROTOCOL,
        "candidate_step": int(provenance["candidate_step"]),
        "candidate_update": EVALUATION_CHECKPOINTS[
            int(provenance["candidate_step"])
        ],
        "seed": 0,
        "solver_profile": "g1-4x5",
        "code_provenance": dict(code_provenance),
        "input_sha256": input_hashes,
        "candidate_validation": dict(candidate_validation),
        **validation,
        "candidate_survival": validation["expert_survival"],
        "carried_no_regression": bool(np.all(candidate >= parent)),
        "carried_improvement_count": int(np.sum(candidate > parent)),
        "carried_regression_count": int(np.sum(candidate < parent)),
        "paired_rollouts_path": str(evidence_path),
        "paired_rollouts_sha256": sha256_file(evidence_path),
    }
    _write_json_atomically(summary_path, manifest)
    return manifest


def run_evaluation(
    *,
    parent_checkpoint_path: Path,
    parent_hparams_path: Path,
    candidate_checkpoint_path: Path,
    candidate_sha256: str,
    candidate_step: int,
    reference_path: Path,
    bank_path: Path,
    output_directory: Path,
    code_commit: str,
    seed: int,
) -> dict[str, object]:
    """Evaluate one immutable E041 checkpoint against E023 on all 120 states."""
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.residual_preview_adapter import (
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
        split_residual_adapter_params,
    )
    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.evaluate_g1_tracking import _load_policy
    from tools.run_g1_action_sequence_recovery_oracle import _build_environment
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets

    if seed != 0:
        raise ValueError("E041 evaluation seed must be zero")
    code_provenance = validate_code_provenance(code_commit)
    paths = tuple(
        path.resolve()
        for path in (
            parent_checkpoint_path,
            parent_hparams_path,
            candidate_checkpoint_path,
            reference_path,
            bank_path,
        )
    )
    (
        parent_checkpoint_path,
        parent_hparams_path,
        candidate_checkpoint_path,
        reference_path,
        bank_path,
    ) = paths
    for path in paths:
        if not path.is_file():
            raise ValueError(f"required evaluation input is missing: {path}")
    expected = {
        parent_checkpoint_path: EXPECTED_RESUME_SHA256,
        parent_hparams_path: EXPECTED_RESUME_HPARAMS_SHA256,
        candidate_checkpoint_path: candidate_sha256,
        reference_path: EXPECTED_LAFAN_REFERENCE_SHA256,
        bank_path: EXPECTED_BANK_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"evaluation input SHA-256 does not match: {path.name}")
    hparams = json.loads(parent_hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    runtime_assets = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    rows = _load_all_bank_rows(bank_path)
    profile = get_solver_profile("g1-4x5")
    env = _build_environment(hparams, reference_path)
    with parent_checkpoint_path.open("rb") as stream:
        parent_state = pickle.load(stream)
    validate_parent_checkpoint(parent_state)
    parent_actor, parent_params, parent_normalizer = _load_policy(
        env, parent_checkpoint_path, seed
    )
    with candidate_checkpoint_path.open("rb") as stream:
        candidate_state = pickle.load(stream)
    candidate_validation = validate_candidate_checkpoint(
        candidate_state,
        candidate_step=candidate_step,
        parent_params=parent_params,
        parent_normalizer=parent_normalizer,
    )
    candidate_params = candidate_state.actor_params
    adapter_kernel, _ = split_residual_adapter_params(candidate_params.adapter)
    residual_actor = PreviewResidualAdapter(
        action_dim=env.action_dim, hidden_dim=int(adapter_kernel.shape[1])
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)

    def make_state(qpos, qvel, phase, last_act, history, rng):
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

    keys = jax.random.split(jax.random.PRNGKey(seed), BANK_ROWS)
    with solver_context(profile):
        initial_states = jax.vmap(make_state)(
            jnp.asarray(rows["qpos"]),
            jnp.asarray(rows["qvel"]),
            jnp.asarray(rows["phase"], dtype=jnp.int32),
            jnp.asarray(rows["last_act"]),
            jnp.asarray(rows["actor_obs_history"]),
            keys,
        )
    thresholds = jnp.asarray(rows["termination_thresholds"], dtype=jnp.float64)

    def rollout_arm(initial_state, *, candidate: bool):
        def step(carry, _):
            state, alive = carry
            normalized = env.normalize_actor_obs(
                normalizer, parent_normalizer, state.obs
            ).astype(jnp.float32)
            if candidate:
                raw_action, parent_action, correction = apply_frozen_preview_residual(
                    parent_actor,
                    residual_actor,
                    candidate_params,
                    normalized,
                    history_len=env.actor_history_len,
                    treatment_frame_dim=env.actor_frame_obs_dim,
                )
            else:
                parent_action = parent_actor.apply(parent_params, normalized)
                correction = jnp.zeros_like(parent_action)
                raw_action = parent_action
            # Training forms parent + residual in float32, then casts the
            # already-composed action to float64 immediately before env.step.
            # Preserve those exact actor-boundary operands in the evidence.
            next_state = env.step(state, raw_action.astype(jnp.float64))
            terminal = next_state.info["terminal"] > 0.5
            normalized_errors = jnp.stack(
                (
                    next_state.metrics["termination_anchor_z_error"],
                    next_state.metrics["termination_anchor_xy_error"],
                    next_state.metrics["termination_gravity_z_error"],
                    next_state.metrics["termination_distal_z_error"],
                )
            ) / thresholds
            output = (
                state.data.qpos,
                state.info["phase"],
                parent_action,
                correction,
                raw_action,
                jnp.clip(raw_action, -1.0, 1.0),
                alive,
                terminal,
                next_state.reward,
                normalized_errors,
            )
            return (next_state, alive & ~terminal), output

        return jax.lax.scan(
            step, (initial_state, jnp.asarray(True)), None, HORIZON
        )[1]

    tensor_names = (
        "qpos",
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
    evidence: dict[str, np.ndarray] = {
        "source_start_phase": np.asarray(rows["source_start_phase"]),
        "initial_qpos": np.asarray(rows["qpos"]),
        "initial_qvel": np.asarray(rows["qvel"]),
        "initial_phase": np.asarray(rows["phase"], dtype=np.int32),
        "initial_last_act": np.asarray(rows["last_act"]),
        "initial_actor_obs_history": np.asarray(rows["actor_obs_history"]),
        "initial_rng_key": np.asarray(keys, dtype=np.uint32),
    }
    with solver_context(profile):
        for arm, candidate in (("parent", False), ("expert", True)):
            rollout = jax.jit(
                jax.vmap(lambda state: rollout_arm(state, candidate=candidate))
            )(initial_states)
            evidence.update(
                {
                    f"{arm}_{name}": np.asarray(value)
                    for name, value in zip(tensor_names, rollout, strict=True)
                }
            )
    return publish_evaluation(
        output_directory=output_directory,
        arrays=evidence,
        provenance={
            "candidate_step": candidate_step,
            "code_provenance": code_provenance,
            "candidate_validation": candidate_validation,
            "input_sha256": {
                "parent_checkpoint": EXPECTED_RESUME_SHA256,
                "parent_hparams": EXPECTED_RESUME_HPARAMS_SHA256,
                "candidate_checkpoint": candidate_sha256,
                "reference": EXPECTED_LAFAN_REFERENCE_SHA256,
                "source_bank": EXPECTED_BANK_SHA256,
                "model": runtime_assets["model_sha256"],
                "controller": runtime_assets["controller_sha256"],
            },
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    summary = run_evaluation(
        parent_checkpoint_path=args.parent_checkpoint,
        parent_hparams_path=args.parent_hparams,
        candidate_checkpoint_path=args.candidate_checkpoint,
        candidate_sha256=args.candidate_sha256,
        candidate_step=args.candidate_step,
        reference_path=args.reference_path,
        bank_path=args.source_bank,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
