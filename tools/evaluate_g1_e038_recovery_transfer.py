"""Pure contracts for the preregistered E038 recovery transfer evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np

from tools.build_g1_e023_carried_reset_bank import (
    E023_TOTAL_STEPS,
    validate_e023_hparams,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_e023_lafan_anchored_carried_recovery import (
    EXPECTED_LAFAN_REFERENCE_SHA256,
)
from tools.run_g1_progressive_recovery_expert import EXPECTED_SOURCE_BANK_SHA256
from tools.run_g1_rmr_noise_h24_continuation import (
    EXPECTED_RESUME_HPARAMS_SHA256,
    EXPECTED_RESUME_SHA256,
)


SOURCE_PHASES = (0, 100, 200, 300, 400)
ROWS_PER_SOURCE = 24
BANK_ROWS = len(SOURCE_PHASES) * ROWS_PER_SOURCE
HORIZON = 32
ACTION_DIM = 29
QPOS_DIM = 36
TERMINATION_ERROR_DIM = 4
PROTOCOL = "g1-e038-recovery-expert-transfer-v1"
EXPECTED_PARENT_CHECKPOINT_SHA256 = EXPECTED_RESUME_SHA256
EXPECTED_PARENT_HPARAMS_SHA256 = EXPECTED_RESUME_HPARAMS_SHA256
EXPECTED_REFERENCE_SHA256 = EXPECTED_LAFAN_REFERENCE_SHA256
EXPECTED_EXPERT_CHECKPOINT_SHA256 = (
    "373fd6528d135dac65b38c35728800da693780558a03bb0cca6a412e314f7bd2"
)

_ARM_TENSOR_SHAPES = {
    "qpos": (HORIZON, QPOS_DIM),
    "phase": (HORIZON,),
    "parent_action": (HORIZON, ACTION_DIM),
    "correction": (HORIZON, ACTION_DIM),
    "raw_action": (HORIZON, ACTION_DIM),
    "effective_action": (HORIZON, ACTION_DIM),
    "alive": (HORIZON,),
    "terminal": (HORIZON,),
    "reward": (HORIZON,),
    "normalized_termination_errors": (HORIZON, TERMINATION_ERROR_DIM),
}
_PAIRED_TENSOR_NAMES = frozenset(
    {
        "source_start_phase",
        "initial_qpos",
        "initial_qvel",
        "initial_phase",
        "initial_last_act",
        "initial_actor_obs_history",
        "initial_rng_key",
    }
    | {
        f"{arm}_{name}"
        for arm in ("parent", "expert")
        for name in _ARM_TENSOR_SHAPES
    }
)
_INPUT_HASH_NAMES = frozenset(
    {
        "parent_checkpoint",
        "hparams",
        "reference",
        "source_bank",
        "expert_checkpoint",
        "model",
        "controller",
    }
)
_PARAMETER_HASH_NAMES = frozenset({"parent", "normalizer", "expert"})


def _zero_seed(value: str) -> int:
    """Parse the only seed registered for this deterministic evaluation."""
    seed = int(value)
    if seed != 0:
        raise argparse.ArgumentTypeError("seed must be exactly zero")
    return seed


def validate_bank_layout(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Require the immutable E027 bank's five ordered 24-row source bands."""
    if "source_start_phase" not in arrays:
        raise ValueError("bank is missing source_start_phase")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in normalized.items():
        if name not in {"source_start_phase", "termination_thresholds"} and (
            value.ndim == 0 or value.shape[0] != BANK_ROWS
        ):
            raise ValueError(f"{name} must contain exactly 120 rows")
    source_start_phase = normalized["source_start_phase"]
    expected = np.repeat(
        np.asarray(SOURCE_PHASES, dtype=np.int32), ROWS_PER_SOURCE
    )
    try:
        is_exact = (
            source_start_phase.shape == (BANK_ROWS,)
            and np.isfinite(source_start_phase).all()
            and np.array_equal(source_start_phase, expected)
        )
    except TypeError:
        is_exact = False
    if not is_exact:
        raise ValueError("bank source groups must be five ordered 24-row bands")
    return normalized


def survival_from_terminals(terminals: np.ndarray) -> list[int]:
    """Report each H32 rollout's first terminal step, or H32 if it survives."""
    values = np.asarray(terminals)
    if values.ndim != 2 or values.shape[1] != HORIZON:
        raise ValueError("terminal evidence must have shape (rows, 32)")
    if not np.isfinite(values).all():
        raise ValueError("terminal evidence must be finite")
    return [
        int(indices[0]) if indices.size else HORIZON
        for indices in (np.flatnonzero(row) for row in values)
    ]


def _survival_vector(values: Sequence[int | float], *, label: str) -> np.ndarray:
    result = np.asarray(values)
    if (
        result.shape != (BANK_ROWS,)
        or not np.issubdtype(result.dtype, np.number)
        or not np.isfinite(result).all()
        or not np.equal(result, np.floor(result)).all()
        or np.any(result < 0)
        or np.any(result > HORIZON)
    ):
        raise ValueError(f"{label} survival evidence is malformed")
    return result.astype(np.int32)


def classify_transfer(
    parent_survival: Sequence[int | float],
    expert_survival: Sequence[int | float],
    source_start_phase: Sequence[int | float],
    execution_valid: bool,
) -> str:
    """Apply E038's preregistered transfer outcome map."""
    if not execution_valid:
        return "invalid-execution"
    parent = _survival_vector(parent_survival, label="parent")
    expert = _survival_vector(expert_survival, label="expert")
    phases = validate_bank_layout(
        {"source_start_phase": np.asarray(source_start_phase)}
    )["source_start_phase"]
    phase_zero = phases == 0
    untouched = ~phase_zero
    phase_zero_successes = int(np.sum(expert[phase_zero] >= HORIZON))
    no_regressions = bool(np.all(expert >= parent))
    newly_recovered = int(
        np.sum((expert[untouched] >= HORIZON) & (parent[untouched] < HORIZON))
    )
    median_regressed = bool(
        np.median(expert[untouched]) < np.median(parent[untouched])
    )
    has_improvements = bool(np.any(expert > parent))
    has_regressions = bool(np.any(expert < parent))

    if (
        phase_zero_successes < 10
        or median_regressed
        or (has_regressions and not has_improvements)
    ):
        return "recovery-expert-destructive"
    if phase_zero_successes in (10, 11) or (
        has_improvements and has_regressions
    ):
        return "recovery-expert-mixed-transfer"
    if phase_zero_successes >= 12 and no_regressions:
        if newly_recovered >= 10:
            return "recovery-expert-generalizes"
        return "recovery-expert-local-only"
    return "recovery-expert-mixed-transfer"


def _expected_paired_shape(name: str) -> tuple[int, ...]:
    initial_shapes = {
        "source_start_phase": (BANK_ROWS,),
        "initial_qpos": (BANK_ROWS, QPOS_DIM),
        "initial_qvel": (BANK_ROWS, 35),
        "initial_phase": (BANK_ROWS,),
        "initial_last_act": (BANK_ROWS, ACTION_DIM),
        "initial_actor_obs_history": (BANK_ROWS, 10, 328),
        "initial_rng_key": (BANK_ROWS, 2),
    }
    if name in initial_shapes:
        return initial_shapes[name]
    arm, tensor = name.split("_", maxsplit=1)
    if arm not in {"parent", "expert"} or tensor not in _ARM_TENSOR_SHAPES:
        raise ValueError(f"unregistered paired evidence tensor: {name}")
    return (BANK_ROWS, *_ARM_TENSOR_SHAPES[tensor])


def _alive_from_terminals(terminals: np.ndarray) -> np.ndarray:
    alive = np.ones(terminals.shape, dtype=bool)
    for row, terminal in enumerate(terminals):
        indices = np.flatnonzero(terminal)
        if indices.size:
            alive[row, int(indices[0]) + 1 :] = False
    return alive


def _validate_forward_canonicalized_qpos(
    actual: np.ndarray,
    requested: np.ndarray,
    *,
    arm: str,
) -> float:
    """Allow only MJX forward's roundoff-scale root-quaternion normalization."""
    if not (
        np.array_equal(actual[:, :3], requested[:, :3])
        and np.array_equal(actual[:, 7:], requested[:, 7:])
    ):
        raise ValueError(
            f"paired {arm} initial qpos non-quaternion coordinates do not "
            "match bank"
        )
    quaternion_delta = np.abs(actual[:, 3:7] - requested[:, 3:7])
    if not np.allclose(
        actual[:, 3:7], requested[:, 3:7], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            f"paired {arm} initial root quaternion does not match bank after "
            "bounded MJX forward canonicalization"
        )
    return float(np.max(quaternion_delta))


def validate_paired_evidence(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Fail closed on E038's exact two-arm H32 evidence contract."""
    if set(arrays) != _PAIRED_TENSOR_NAMES:
        missing = sorted(_PAIRED_TENSOR_NAMES - set(arrays))
        unexpected = sorted(set(arrays) - _PAIRED_TENSOR_NAMES)
        raise ValueError(
            f"paired evidence tensor set is not exact; missing={missing}, "
            f"unexpected={unexpected}"
        )
    evidence = {name: np.asarray(value) for name, value in arrays.items()}
    validate_bank_layout({"source_start_phase": evidence["source_start_phase"]})
    for name, value in evidence.items():
        expected_shape = _expected_paired_shape(name)
        if value.shape != expected_shape:
            raise ValueError(
                f"paired evidence {name} shape {value.shape} does not match "
                f"{expected_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"paired evidence {name} must be finite")

    integer_tensors = (
        "source_start_phase",
        "initial_phase",
        "parent_phase",
        "expert_phase",
    )
    for name in integer_tensors:
        if not np.issubdtype(evidence[name].dtype, np.integer):
            raise ValueError(f"paired evidence {name} must have integer dtype")
    if evidence["initial_rng_key"].dtype != np.dtype(np.uint32):
        raise ValueError("paired evidence initial RNG keys must have uint32 dtype")
    for arm in ("parent", "expert"):
        for name in ("alive", "terminal"):
            if evidence[f"{arm}_{name}"].dtype != np.dtype(bool):
                raise ValueError(
                    f"paired evidence {arm} {name} must have boolean dtype"
                )

    parent_qpos = evidence["parent_qpos"]
    expert_qpos = evidence["expert_qpos"]
    if not np.array_equal(parent_qpos[:, 0], expert_qpos[:, 0]):
        raise ValueError("paired initial qpos does not match")
    if not np.array_equal(
        evidence["parent_phase"][:, 0], evidence["expert_phase"][:, 0]
    ):
        raise ValueError("paired initial phase does not match")
    qpos_canonicalization = {
        arm: _validate_forward_canonicalized_qpos(
            evidence[f"{arm}_qpos"][:, 0],
            evidence["initial_qpos"],
            arm=arm,
        )
        for arm in ("parent", "expert")
    }
    for arm in ("parent", "expert"):
        if not np.array_equal(
            evidence[f"{arm}_phase"][:, 0], evidence["initial_phase"]
        ):
            raise ValueError(f"paired {arm} initial phase does not match bank")

    for arm in ("parent", "expert"):
        raw = evidence[f"{arm}_raw_action"]
        effective = evidence[f"{arm}_effective_action"]
        if not np.array_equal(effective, np.clip(raw, -1.0, 1.0)):
            raise ValueError(
                f"paired evidence {arm} effective action does not match "
                "final clipping"
            )
        if not np.array_equal(
            raw,
            evidence[f"{arm}_parent_action"] + evidence[f"{arm}_correction"],
        ):
            raise ValueError(
                f"paired evidence {arm} raw action does not equal parent plus "
                "correction"
            )
        terminal = np.asarray(evidence[f"{arm}_terminal"], dtype=bool)
        if not np.array_equal(
            np.asarray(evidence[f"{arm}_alive"], dtype=bool),
            _alive_from_terminals(terminal),
        ):
            raise ValueError(f"paired evidence {arm} alive mask does not match terminals")

    parent_survival = survival_from_terminals(evidence["parent_terminal"])
    expert_survival = survival_from_terminals(evidence["expert_terminal"])
    parent_values = np.asarray(parent_survival, dtype=np.int32)
    expert_values = np.asarray(expert_survival, dtype=np.int32)
    phase_zero = evidence["source_start_phase"] == 0
    untouched = ~phase_zero
    return {
        "source_rows": [ROWS_PER_SOURCE] * len(SOURCE_PHASES),
        "initial_qpos_max_abs_canonicalization": max(
            qpos_canonicalization.values()
        ),
        "parent_survival": parent_survival,
        "expert_survival": expert_survival,
        "parent_full_horizon_count": int(
            np.sum(parent_values >= HORIZON)
        ),
        "expert_full_horizon_count": int(
            np.sum(expert_values >= HORIZON)
        ),
        "phase_zero_expert_successes": int(
            np.sum(expert_values[phase_zero] >= HORIZON)
        ),
        "untouched_newly_recovered": int(
            np.sum(
                (expert_values[untouched] >= HORIZON)
                & (parent_values[untouched] < HORIZON)
            )
        ),
        "any_survival_improvement": bool(np.any(expert_values > parent_values)),
        "any_survival_regression": bool(np.any(expert_values < parent_values)),
        "untouched_parent_median_survival": float(np.median(parent_values[untouched])),
        "untouched_expert_median_survival": float(np.median(expert_values[untouched])),
    }


def parameter_tree_sha256(tree: object) -> str:
    """Hash a JAX pytree's treedef, paths, leaf dtypes, shapes, and bytes."""
    import jax

    digest = hashlib.sha256()
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    digest.update(repr(treedef).encode("utf-8"))
    for path, value in paths_and_leaves:
        digest.update(repr(path).encode("utf-8"))
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(b"array")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_parameter_immutability(
    before: Mapping[str, str], after: Mapping[str, str]
) -> bool:
    """Require the parent, normalizer, and expert parameter bytes to be frozen."""
    if set(before) != _PARAMETER_HASH_NAMES or set(after) != _PARAMETER_HASH_NAMES:
        raise ValueError("parameter hash manifest is not exact")
    if dict(before) != dict(after):
        raise ValueError("frozen evaluation parameters changed")
    return True


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


def _validate_digest_map(
    values: Mapping[str, str], *, expected: frozenset[str], label: str
) -> dict[str, str]:
    if set(values) != expected or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise ValueError(f"{label} hash manifest is not exact")
    return dict(values)


def publish_evaluation(
    *,
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Atomically publish paired evidence before its hash-bound final manifest."""
    validation = validate_paired_evidence(arrays)
    input_hashes = _validate_digest_map(
        provenance.get("input_sha256", {}),
        expected=_INPUT_HASH_NAMES,
        label="input",
    )
    before = _validate_digest_map(
        provenance.get("parameter_sha256_before", {}),
        expected=_PARAMETER_HASH_NAMES,
        label="parameter-before",
    )
    after = _validate_digest_map(
        provenance.get("parameter_sha256_after", {}),
        expected=_PARAMETER_HASH_NAMES,
        label="parameter-after",
    )
    validate_parameter_immutability(before, after)
    code_commit = provenance.get("code_commit")
    if not isinstance(code_commit, str) or not code_commit:
        raise ValueError("code commit is required for paired evidence")

    output_directory = output_directory.resolve()
    evidence_path = output_directory / "paired_rollouts.npz"
    summary_path = output_directory / "summary.json"
    _write_npz_atomically(evidence_path, arrays)
    outcome = classify_transfer(
        validation["parent_survival"],
        validation["expert_survival"],
        np.asarray(arrays["source_start_phase"]),
        execution_valid=True,
    )
    manifest = {
        "valid": True,
        "protocol": PROTOCOL,
        "outcome": outcome,
        "horizon": HORIZON,
        "seed": provenance.get("seed", 0),
        "solver_profile": "g1-4x5",
        "code_commit": code_commit,
        "input_sha256": input_hashes,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameter_immutability": True,
        **validation,
        "paired_rollouts_path": str(evidence_path),
        "paired_rollouts_sha256": sha256_file(evidence_path),
    }
    _write_json_atomically(summary_path, manifest)
    return manifest


def _load_all_bank_rows(bank_path: Path) -> dict[str, np.ndarray]:
    with np.load(bank_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays = validate_bank_layout(arrays)
    required_shapes = {
        "qpos": (BANK_ROWS, QPOS_DIM),
        "qvel": (BANK_ROWS, 35),
        "phase": (BANK_ROWS,),
        "last_act": (BANK_ROWS, ACTION_DIM),
        "actor_obs_history": (BANK_ROWS, 10, 328),
        "termination_thresholds": (TERMINATION_ERROR_DIM,),
    }
    for name, expected_shape in required_shapes.items():
        value = arrays.get(name)
        if value is None or value.shape != expected_shape:
            raise ValueError(f"E027 bank {name} shape does not match")
        if not np.isfinite(value).all():
            raise ValueError(f"E027 bank {name} must be finite")
    if np.any(arrays["termination_thresholds"] <= 0.0):
        raise ValueError("E027 bank termination thresholds must be positive")
    return arrays


def validate_exact_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Require the preregistered E023, E027, LAFAN, and E038 artifacts."""
    expected = {
        "parent_checkpoint": EXPECTED_PARENT_CHECKPOINT_SHA256,
        "hparams": EXPECTED_PARENT_HPARAMS_SHA256,
        "reference": EXPECTED_REFERENCE_SHA256,
        "source_bank": EXPECTED_SOURCE_BANK_SHA256,
        "expert_checkpoint": EXPECTED_EXPERT_CHECKPOINT_SHA256,
    }
    if set(paths) != set(expected):
        raise ValueError("E038 input hash names are not exact")
    for name, expected_digest in expected.items():
        if sha256_file(paths[name]) != expected_digest:
            raise ValueError(f"E038 {name} SHA-256 does not match")
    return expected


def validate_parent_checkpoint(checkpoint: object) -> None:
    """Require the selected E023 plain Flax actor at its exact final step."""
    from src.algorithms.shac.residual_preview_adapter import FrozenPreviewResidualParams

    if not hasattr(checkpoint, "actor_params") or not hasattr(checkpoint, "step"):
        raise ValueError("E023 checkpoint does not expose a TrainState actor")
    if int(np.asarray(checkpoint.step)) != E023_TOTAL_STEPS:
        raise ValueError("E023 checkpoint step does not match selected parent")
    actor_params = checkpoint.actor_params
    if isinstance(actor_params, FrozenPreviewResidualParams):
        raise ValueError("E023 checkpoint must contain a plain Flax actor")
    if not isinstance(actor_params, Mapping) or set(actor_params) != {"params"}:
        raise ValueError("E023 checkpoint plain actor identity does not match")
    layers = actor_params["params"]
    expected_layers = {
        "Dense_0",
        "Dense_1",
        "Dense_2",
        "Dense_3",
        "LayerNorm_0",
        "LayerNorm_1",
        "LayerNorm_2",
    }
    if not isinstance(layers, Mapping) or set(layers) != expected_layers:
        raise ValueError("E023 checkpoint plain actor identity does not match")
    import jax

    if not all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(actor_params)
    ):
        raise ValueError("E023 actor contains nonfinite parameters")


def run_evaluation(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    bank_path: Path,
    expert_checkpoint_path: Path,
    output_directory: Path,
    seed: int,
    code_commit: str,
) -> dict[str, object]:
    """Evaluate the immutable E038 expert and E023 parent from every E027 row."""
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.residual_preview_adapter import (
        PreviewResidualAdapter,
        current_treatment_frame,
    )
    from src.core.data_structures import Normalizer
    from src.core.networks import Actor
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.run_g1_action_sequence_recovery_oracle import _build_environment
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets
    from tools.evaluate_g1_tracking import _load_policy

    if seed != 0:
        raise ValueError("E038 evaluation seed must be exactly zero")
    checkpoint_path = checkpoint_path.resolve()
    hparams_path = hparams_path.resolve()
    reference_path = reference_path.resolve()
    bank_path = bank_path.resolve()
    expert_checkpoint_path = expert_checkpoint_path.resolve()
    for path, label in (
        (checkpoint_path, "parent checkpoint"),
        (hparams_path, "hparams"),
        (reference_path, "reference"),
        (bank_path, "source bank"),
        (expert_checkpoint_path, "expert checkpoint"),
    ):
        if not path.is_file():
            raise ValueError(f"E038 {label} is missing")
    if hparams_path != checkpoint_path.with_name("hparams.json"):
        raise ValueError("E023 hparams must be the parent checkpoint sibling")
    input_hashes = validate_exact_input_hashes(
        {
            "parent_checkpoint": checkpoint_path,
            "hparams": hparams_path,
            "reference": reference_path,
            "source_bank": bank_path,
            "expert_checkpoint": expert_checkpoint_path,
        }
    )

    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    runtime_assets = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    input_hashes.update(
        {
            "model": runtime_assets["model_sha256"],
            "controller": runtime_assets["controller_sha256"],
        }
    )
    rows = _load_all_bank_rows(bank_path)
    profile = get_solver_profile("g1-4x5")
    if (
        int(hparams["solver_iterations"]) != profile.iterations
        or int(hparams["solver_ls_iterations"]) != profile.ls_iterations
    ):
        raise ValueError("E038 evaluation requires the g1-4x5 solver profile")
    env = _build_environment(hparams, reference_path)
    with checkpoint_path.open("rb") as stream:
        parent_checkpoint = pickle.load(stream)
    validate_parent_checkpoint(parent_checkpoint)
    actor, actor_params, normalizer_state = _load_policy(env, checkpoint_path, seed)
    if not isinstance(actor, Actor):
        raise ValueError("E023 policy loader did not return the plain Flax actor")
    normalizer = Normalizer(env.actor_frame_obs_dim)
    with expert_checkpoint_path.open("rb") as stream:
        expert_params = pickle.load(stream)
    expert = PreviewResidualAdapter(action_dim=env.action_dim, hidden_dim=256)
    parameter_before = {
        "parent": parameter_tree_sha256(actor_params),
        "normalizer": parameter_tree_sha256(normalizer_state),
        "expert": parameter_tree_sha256(expert_params),
    }

    if env.action_dim != ACTION_DIM:
        raise ValueError("E038 evaluator action dimension does not match")
    if int(env.actor_history_len) != rows["actor_obs_history"].shape[1]:
        raise ValueError("E027 actor history does not match the E023 environment")
    if int(env.actor_frame_obs_dim) != rows["actor_obs_history"].shape[2]:
        raise ValueError("E027 actor frame does not match the E023 environment")

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

    def rollout_arm(initial_state, *, use_expert: bool):
        def step(carry, _):
            state, alive = carry
            normalized = env.normalize_actor_obs(
                normalizer, normalizer_state, state.obs
            ).astype(jnp.float32)
            parent_action = actor.apply(actor_params, normalized).astype(jnp.float64)
            if use_expert:
                frame = current_treatment_frame(
                    normalized,
                    history_len=env.actor_history_len,
                    treatment_frame_dim=env.actor_frame_obs_dim,
                )
                correction = expert.apply(expert_params, frame).astype(jnp.float64)
            else:
                correction = jnp.zeros_like(parent_action)
            raw_action = parent_action + correction
            next_state = env.step(state, raw_action)
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

        return jax.lax.scan(step, (initial_state, jnp.asarray(True)), None, HORIZON)[1]

    tensor_names = tuple(_ARM_TENSOR_SHAPES)
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
        for arm, use_expert in (("parent", False), ("expert", True)):
            rollout = jax.jit(
                jax.vmap(lambda state: rollout_arm(state, use_expert=use_expert))
            )(initial_states)
            evidence.update(
                {
                    f"{arm}_{name}": np.asarray(value)
                    for name, value in zip(tensor_names, rollout, strict=True)
                }
            )
    parameter_after = {
        "parent": parameter_tree_sha256(actor_params),
        "normalizer": parameter_tree_sha256(normalizer_state),
        "expert": parameter_tree_sha256(expert_params),
    }
    return publish_evaluation(
        output_directory=output_directory,
        arrays=evidence,
        provenance={
            "code_commit": code_commit,
            "seed": seed,
            "input_sha256": input_hashes,
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded E038 evaluation parser, pinning seed and solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--seed", type=_zero_seed, default=0)
    return parser


def main() -> None:
    from tools.build_g1_e023_carried_reset_bank import validate_code_commit
    from tools.run_g1_tracking_shac import configure_jax

    args = build_parser().parse_args()
    if args.solver_profile != "g1-4x5":
        raise ValueError("E038 evaluation requires the g1-4x5 solver profile")
    repository = Path(__file__).resolve().parents[1]
    configure_jax()
    summary = run_evaluation(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        reference_path=args.reference_path,
        bank_path=args.source_bank,
        expert_checkpoint_path=args.expert_checkpoint,
        output_directory=args.output_directory,
        seed=args.seed,
        code_commit=validate_code_commit(repository, args.code_commit),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
