"""Audit frozen E023 SHAC objective directions on one fixed G1 population."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any, Mapping

import numpy as np

from tools.prepare_g1_rmr_reference import sha256_file


POPULATION = 512
CARRIED_POPULATION = 128
EXACT_POPULATION = 384
BANK_ROWS = 120
ACTION_DIM = 29
MAX_HORIZON = 48
DIRECTIONS = ("h24_a", "h24_b", "h48_a", "bootstrap_a")
EXPECTED_INPUT_SHA256 = {
    "checkpoint": "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f",
    "hparams": "a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650",
    "reference": "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db",
    "bank": "d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5",
    "expert": "373fd6528d135dac65b38c35728800da693780558a03bb0cca6a412e314f7bd2",
}
EXPECTED_MODEL_SHA256 = "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
EXPECTED_CONTROLLER_SHA256 = "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
PROTOCOL = "g1-objective-direction-audit-v1"
BASE_DISPLACEMENT = 0.09495018422603607
PROPOSAL_DIRECTIONS = ("h24", "h48", "bootstrap")
PROPOSAL_MULTIPLIERS = (0.125, 0.25, 0.5, 1.0)
ORDINARY_BASELINE = (116, 63, 49, 39, 47)
REGISTERED_OUTCOMES = frozenset(
    {
        "stochastic-gradient-inconsistent",
        "current-h24-direction-useful",
        "short-horizon-credit-misaligned",
        "terminal-bootstrap-useful",
        "cross-state-objective-conflict",
        "aligned-local-step-insufficient",
        "direction-audit-inconclusive",
        "invalid-execution",
    }
)


def build_fixed_population_indices(seed: int) -> dict[str, np.ndarray]:
    """Return the immutable 128 carried plus 384 exact-reference population."""
    if seed != 0:
        raise ValueError("objective-direction population seed must be zero")
    rng = np.random.default_rng(seed)
    repeats = rng.choice(BANK_ROWS, size=CARRIED_POPULATION - BANK_ROWS, replace=False)
    carried = np.concatenate((np.arange(BANK_ROWS), repeats)).astype(np.int32)
    exact = np.floor(
        np.linspace(0, 499, EXACT_POPULATION, endpoint=False)
    ).astype(np.int32)
    return {
        "source_kind": np.concatenate(
            (
                np.ones(CARRIED_POPULATION, dtype=np.int8),
                np.zeros(EXACT_POPULATION, dtype=np.int8),
            )
        ),
        "source_index": np.concatenate((carried, exact)),
    }


def build_fixed_noise_tapes(seed: int) -> dict[str, np.ndarray]:
    """Generate two immutable standard-normal epsilon tapes."""
    if seed != 0:
        raise ValueError("objective-direction noise seed must be zero")
    rng_a = np.random.default_rng(41_024 + seed)
    rng_b = np.random.default_rng(82_048 + seed)
    shape = (POPULATION, MAX_HORIZON, ACTION_DIM)
    return {
        "a": rng_a.standard_normal(shape, dtype=np.float32),
        "b": rng_b.standard_normal(shape, dtype=np.float32),
    }


def validate_common_noise_prefix(prefix: np.ndarray, full: np.ndarray) -> bool:
    """Require a bit-exact H24 prefix of the H48 tape."""
    prefix = np.asarray(prefix)
    full = np.asarray(full)
    if (
        prefix.shape != (POPULATION, 24, ACTION_DIM)
        or full.shape != (POPULATION, MAX_HORIZON, ACTION_DIM)
        or prefix.dtype != np.float32
        or full.dtype != np.float32
        or not np.array_equal(prefix, full[:, :24])
    ):
        raise ValueError("H24 noise is not the exact H48 common prefix")
    return True


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite numeric evidence")
    return array


def validate_gradient_artifacts(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Validate the complete fixed-population gradient evidence."""
    required = {
        "source_kind",
        "source_index",
        "phase",
        "rng_key",
        "initial_qpos",
        "initial_qvel",
        "initial_last_act",
        "initial_actor_obs_history",
        "noise_tape_a",
        "noise_tape_b",
        "h24_tape_env_cosine",
        "h24_h48_env_cosine",
        "h24_a_bootstrap_env_cosine",
        "h24_b_h48_env_cosine",
        "h24_b_bootstrap_env_cosine",
        "h48_bootstrap_env_cosine",
    }
    for direction in DIRECTIONS:
        required.update(
            {
                f"{direction}_combined",
                f"{direction}_task",
                f"{direction}_counts",
                f"{direction}_cosine",
                f"{direction}_weights",
                f"{direction}_env_norm",
            }
        )
    if set(arrays) != required:
        missing = sorted(required - set(arrays))
        extra = sorted(set(arrays) - required)
        raise ValueError(f"gradient artifact schema differs: missing={missing}, extra={extra}")
    population = int(np.asarray(arrays["source_kind"]).shape[0])
    if population != POPULATION:
        raise ValueError("scientific gradient evidence requires 512 states")
    if np.asarray(arrays["source_kind"]).shape != (POPULATION,):
        raise ValueError("source kind shape does not match")
    if np.asarray(arrays["source_index"]).shape != (POPULATION,):
        raise ValueError("source index shape does not match")
    if np.asarray(arrays["phase"]).shape != (POPULATION,):
        raise ValueError("phase shape does not match")
    if np.asarray(arrays["rng_key"]).shape != (POPULATION, 2):
        raise ValueError("RNG key shape does not match")
    if np.asarray(arrays["rng_key"]).dtype != np.uint32:
        raise ValueError("RNG keys must be uint32")
    identity_shapes = {
        "initial_qpos": (POPULATION, 36),
        "initial_qvel": (POPULATION, 35),
        "initial_last_act": (POPULATION, ACTION_DIM),
        "initial_actor_obs_history": (POPULATION, 10, 328),
    }
    for name, shape in identity_shapes.items():
        if _require_finite(name, arrays[name]).shape != shape:
            raise ValueError(f"{name} identity shape does not match")
    validate_common_noise_prefix(
        np.asarray(arrays["noise_tape_a"])[:, :24],
        np.asarray(arrays["noise_tape_a"]),
    )
    if np.array_equal(arrays["noise_tape_a"], arrays["noise_tape_b"]):
        raise ValueError("noise tapes must be independent")
    vector_width = None
    direction_norms: dict[str, float] = {}
    for direction in DIRECTIONS:
        combined = _require_finite(
            f"{direction} combined", arrays[f"{direction}_combined"]
        )
        task = _require_finite(f"{direction} task", arrays[f"{direction}_task"])
        counts = _require_finite(
            f"{direction} counts", arrays[f"{direction}_counts"]
        )
        cosine = _require_finite(
            f"{direction} cosine", arrays[f"{direction}_cosine"]
        )
        weights = _require_finite(
            f"{direction} weights", arrays[f"{direction}_weights"]
        )
        env_norm = _require_finite(
            f"{direction} env norm", arrays[f"{direction}_env_norm"]
        )
        if combined.ndim != 1 or combined.size == 0:
            raise ValueError("combined gradient vector must be nonempty")
        vector_width = combined.size if vector_width is None else vector_width
        if combined.size != vector_width or task.shape != (5, vector_width):
            raise ValueError("gradient vector widths do not agree")
        if counts.shape != (5,) or np.any(counts <= 0):
            raise ValueError("every direction must occupy all five phase bins")
        if cosine.shape != (5, 5) or weights.shape != (5,):
            raise ValueError("CAGrad diagnostic shapes do not match")
        if env_norm.shape != (POPULATION,) or np.any(env_norm < 0):
            raise ValueError("per-environment gradient norms do not match")
        norm = float(np.linalg.norm(combined))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("combined gradient direction must be nonzero")
        direction_norms[direction] = norm
    for name in (
        "h24_tape_env_cosine",
        "h24_h48_env_cosine",
        "h24_a_bootstrap_env_cosine",
        "h24_b_h48_env_cosine",
        "h24_b_bootstrap_env_cosine",
        "h48_bootstrap_env_cosine",
    ):
        values = _require_finite(name, arrays[name])
        if values.shape != (POPULATION,) or np.any(np.abs(values) > 1.000001):
            raise ValueError("per-environment cosine evidence does not match")
    return {
        "valid": True,
        "protocol": PROTOCOL,
        "population": POPULATION,
        "carried_population": int(np.sum(np.asarray(arrays["source_kind"]) == 1)),
        "exact_population": int(np.sum(np.asarray(arrays["source_kind"]) == 0)),
        "direction_norms": direction_norms,
        "h24_tape_env_cosine_mean": float(np.mean(arrays["h24_tape_env_cosine"])),
        "h24_h48_env_cosine_mean": float(np.mean(arrays["h24_h48_env_cosine"])),
    }


def publish_gradient_artifacts(
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    provenance: Mapping[str, object],
    scientific: bool,
) -> dict[str, object]:
    """Publish hash-bound evidence, writing the completion marker last."""
    validation = validate_gradient_artifacts(arrays)
    code_commit = provenance.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise ValueError("a full code commit is required")
    input_hashes = provenance.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("input SHA-256 provenance is required")
    output_directory = output_directory.resolve()
    artifact = output_directory / "gradient_evidence.npz"
    summary_path = output_directory / "gradient_summary.json"
    completion_path = output_directory / "completion.json"
    _atomic_npz(artifact, arrays)
    summary = {
        **validation,
        "scientific": bool(scientific),
        "code_commit": code_commit,
        "input_sha256": dict(input_hashes),
        "gradient_artifact_path": str(artifact),
        "gradient_artifact_sha256": sha256_file(artifact),
    }
    _atomic_json(summary_path, summary)
    completion = {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }
    _atomic_json(completion_path, completion)
    return completion


def validate_completion_manifest(path: Path) -> dict[str, object]:
    """Reopen a completed publication and independently verify its hashes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        raise ValueError("completion manifest is invalid")
    artifact = Path(str(payload.get("gradient_artifact_path", "")))
    summary = Path(str(payload.get("summary_path", "")))
    if (
        not artifact.is_file()
        or sha256_file(artifact) != payload.get("gradient_artifact_sha256")
        or not summary.is_file()
        or sha256_file(summary) != payload.get("summary_sha256")
    ):
        raise ValueError("completion artifact SHA-256 does not match")
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    validate_gradient_artifacts(arrays)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    if summary_payload.get("gradient_artifact_sha256") != payload.get(
        "gradient_artifact_sha256"
    ):
        raise ValueError("completion summary does not match")
    return payload


def proposal_specs() -> list[dict[str, object]]:
    """Return the twelve immutable equal-norm line-search specifications."""
    return [
        {
            "label": f"{direction}-{str(multiplier).replace('.', 'p')}",
            "direction": direction,
            "multiplier": multiplier,
            "displacement": BASE_DISPLACEMENT * multiplier,
        }
        for direction in PROPOSAL_DIRECTIONS
        for multiplier in PROPOSAL_MULTIPLIERS
    ]


def validate_line_search_evidence(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Validate the complete H32 carried grid and bounded ordinary follow-up."""
    shapes = {
        "baseline_carried_survival": (BANK_ROWS,),
        "candidate_carried_survival": (12, BANK_ROWS),
        "selected_proposal_index": (3,),
        "baseline_ordinary_survival": (5,),
        "selected_ordinary_survival": (3, 5),
        "full_gate": (3,),
    }
    if set(arrays) != set(shapes):
        raise ValueError("line-search evidence schema is not exact")
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"line-search {name} shape does not match")
        if not np.issubdtype(value.dtype, np.number) and value.dtype != bool:
            raise ValueError(f"line-search {name} dtype does not match")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"line-search {name} must be finite")
    baseline = np.asarray(arrays["baseline_carried_survival"])
    candidates = np.asarray(arrays["candidate_carried_survival"])
    if np.any(baseline < 0) or np.any(baseline > 32) or np.any(candidates < 0) or np.any(candidates > 32):
        raise ValueError("line-search H32 survival lies outside the registered horizon")
    selected = np.asarray(arrays["selected_proposal_index"], dtype=np.int64)
    for direction_index, proposal_index in enumerate(selected):
        allowed = range(direction_index * 4, direction_index * 4 + 4)
        if proposal_index != -1 and proposal_index not in allowed:
            raise ValueError("selected proposal does not belong to its direction")
    ordinary = np.asarray(arrays["selected_ordinary_survival"])
    for row, proposal_index in zip(ordinary, selected, strict=True):
        if proposal_index == -1 and not np.array_equal(row, np.full(5, -1)):
            raise ValueError("unselected direction has ordinary evidence")
        if proposal_index != -1 and np.any(row < 0):
            raise ValueError("selected direction lacks ordinary evidence")
    if not np.array_equal(
        np.asarray(arrays["baseline_ordinary_survival"]),
        np.asarray(ORDINARY_BASELINE),
    ):
        raise ValueError("ordinary baseline does not reproduce E023")
    return {"valid": True, "proposal_count": 12}


def publish_final_manifest(
    output_directory: Path,
    *,
    artifacts: Mapping[str, Path],
    outcome: str,
    code_commit: str,
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Write the diagnostic completion marker after every bound artifact."""
    expected = {
        "preflight.json",
        "gradient_evidence.npz",
        "gradient_summary.json",
        "line_search.npz",
        "line_search.json",
        "selection.json",
        "cosine_heatmap.png",
        "survival_plot.png",
    }
    if set(artifacts) != expected or outcome not in REGISTERED_OUTCOMES:
        raise ValueError("final objective-direction publication is incomplete")
    if len(code_commit) != 40 or any(
        not isinstance(value, str) or len(value) != 64
        for value in input_sha256.values()
    ):
        raise ValueError("final objective-direction provenance is invalid")
    artifact_rows: dict[str, dict[str, str]] = {}
    for name in sorted(expected):
        path = artifacts[name].resolve()
        if not path.is_file() or path.name != name:
            raise ValueError("final objective-direction artifact is missing")
        artifact_rows[name] = {"path": str(path), "sha256": sha256_file(path)}
    payload = {
        "valid": True,
        "protocol": f"{PROTOCOL}-completion",
        "outcome": outcome,
        "code_commit": code_commit,
        "input_sha256": dict(input_sha256),
        "artifacts": artifact_rows,
        "diagnostic_only": True,
    }
    completion_path = output_directory.resolve() / "completion.json"
    _atomic_json(completion_path, payload)
    return payload


def validate_final_manifest(path: Path) -> dict[str, object]:
    """Independently rehash every final diagnostic artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("valid") is not True
        or payload.get("outcome") not in REGISTERED_OUTCOMES
        or payload.get("diagnostic_only") is not True
    ):
        raise ValueError("final objective-direction manifest is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("final objective-direction artifact map is invalid")
    for name, row in artifacts.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise ValueError("final objective-direction artifact row is invalid")
        artifact = Path(str(row.get("path", "")))
        if not artifact.is_file() or sha256_file(artifact) != row.get("sha256"):
            raise ValueError("final objective-direction artifact hash does not match")
    return payload


def build_preflight(
    paths: Mapping[str, Path], *, code_commit: str, repository: Path
) -> dict[str, object]:
    """Bind the exact frozen inputs, runtime assets, and clean source commit."""
    if set(paths) != set(EXPECTED_INPUT_SHA256):
        raise ValueError("objective-direction input names are not exact")
    hashes = {name: sha256_file(path.resolve()) for name, path in paths.items()}
    if hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("objective-direction input SHA-256 does not match")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tools"],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout
    if head != code_commit or len(code_commit) != 40 or dirty:
        raise ValueError("objective-direction source provenance does not match")
    return {
        "valid": True,
        "protocol": f"{PROTOCOL}-preflight",
        "code_commit": code_commit,
        "input_sha256": hashes,
    }


def _tree_matrix(tree: Any) -> np.ndarray:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("gradient tree is empty")
    rows = int(np.asarray(leaves[0]).shape[0])
    return np.concatenate(
        [np.asarray(leaf).reshape(rows, -1) for leaf in leaves], axis=1
    )


def _tree_vector(tree: Any) -> np.ndarray:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    return np.concatenate([np.asarray(leaf).reshape(-1) for leaf in leaves])


def _env_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=1)
    denom = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)


def _vector_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("objective-direction cosine requires nonzero finite vectors")
    value = float(np.dot(left, right) / denominator)
    if not np.isfinite(value):
        raise ValueError("objective-direction cosine is nonfinite")
    return value


def _first_terminal_survival(
    terminals: np.ndarray, maximum_steps: np.ndarray
) -> np.ndarray:
    terminal_values = np.asarray(terminals, dtype=bool)
    limits = np.asarray(maximum_steps, dtype=np.int32)
    if terminal_values.ndim != 2 or limits.shape != (terminal_values.shape[0],):
        raise ValueError("terminal survival evidence shape does not match")
    first = np.argmax(terminal_values, axis=1)
    has_terminal = np.any(terminal_values, axis=1)
    return np.where(has_terminal, first, limits).astype(np.int32)


def ordinary_steps_from_done(done_trace: list[bool], *, maximum_steps: int) -> int:
    """Match the canonical phase-grid count, including the terminal transition."""
    if (
        maximum_steps < 1
        or not done_trace
        or len(done_trace) > maximum_steps
        or any(not isinstance(value, (bool, np.bool_)) for value in done_trace)
        or (not done_trace[-1] and len(done_trace) != maximum_steps)
    ):
        raise ValueError("ordinary done trace does not match its rollout limit")
    if any(done_trace[:-1]):
        raise ValueError("ordinary done trace continued after termination")
    return len(done_trace)


def _plot_diagnostics(
    output_directory: Path,
    *,
    aggregate_cosines: np.ndarray,
    baseline_carried: np.ndarray,
    candidate_carried: np.ndarray,
    baseline_ordinary: np.ndarray,
    selected_ordinary: np.ndarray,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    cosine_path = output_directory / "cosine_heatmap.png"
    survival_path = output_directory / "survival_plot.png"
    temporary_cosine = cosine_path.with_name(f".{cosine_path.name}.tmp.png")
    temporary_survival = survival_path.with_name(f".{survival_path.name}.tmp.png")
    names = ("H24-A", "H24-B", "H48-A", "Bootstrap")
    figure, axis = plt.subplots(figsize=(5.5, 4.7))
    image = axis.imshow(aggregate_cosines, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(4), names, rotation=25, ha="right")
    axis.set_yticks(range(4), names)
    for row in range(4):
        for column in range(4):
            axis.text(column, row, f"{aggregate_cosines[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="gradient cosine")
    figure.tight_layout()
    figure.savefig(temporary_cosine, dpi=160)
    plt.close(figure)
    os.replace(temporary_cosine, cosine_path)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    carried_delta = candidate_carried - baseline_carried[None, :]
    axes[0].boxplot([carried_delta[index] for index in range(12)], showfliers=False)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("H32 carried survival deltas")
    axes[0].set_xlabel("proposal index")
    axes[0].set_ylabel("steps vs E023")
    x = np.arange(5)
    axes[1].plot(x, baseline_ordinary, marker="o", label="E023")
    for index, values in enumerate(selected_ordinary):
        if np.all(values >= 0):
            axes[1].plot(x, values, marker="o", label=PROPOSAL_DIRECTIONS[index])
    axes[1].set_xticks(x, ("0", "100", "200", "300", "400"))
    axes[1].set_title("ordinary replay-free survival")
    axes[1].set_xlabel("start phase")
    axes[1].set_ylabel("steps")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(temporary_survival, dpi=160)
    plt.close(figure)
    os.replace(temporary_survival, survival_path)
    return cosine_path, survival_path


def run_gradient_capture(
    *,
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    bank_path: Path,
    expert_path: Path,
    output_directory: Path,
    seed: int,
    code_commit: str,
    repository: Path,
    smoke: bool = False,
) -> dict[str, object]:
    """Capture four training-effective adapter directions on one frozen batch."""
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.algorithm import squeeze_value_head
    from src.algorithms.shac.objective_direction_audit import (
        aggregate_audit_direction,
        classify_objective_direction_audit,
        normalized_descent_proposal,
        ordinary_componentwise_safe,
        select_carried_safe_candidate,
    )
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
        transplant_zero_head_recovery_features,
    )
    from src.core.data_structures import Normalizer
    from src.core.networks import Critic
    from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
    from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
    from tools.build_g1_e023_carried_reset_bank import validate_e023_hparams
    from tools.evaluate_g1_e038_recovery_transfer import (
        _load_all_bank_rows,
        validate_parent_checkpoint,
    )
    from tools.evaluate_g1_tracking import _load_policy, build_compiled_step
    from tools.run_g1_action_sequence_recovery_oracle import _build_environment
    from tools.run_g1_root_recovery_continuation import validate_runtime_assets

    if seed != 0:
        raise ValueError("objective-direction audit seed must be zero")
    input_paths = {
        "checkpoint": checkpoint_path.resolve(),
        "hparams": hparams_path.resolve(),
        "reference": reference_path.resolve(),
        "bank": bank_path.resolve(),
        "expert": expert_path.resolve(),
    }
    preflight = build_preflight(
        input_paths, code_commit=code_commit, repository=repository.resolve()
    )
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    runtime = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    if (
        runtime["model_sha256"] != EXPECTED_MODEL_SHA256
        or runtime["controller_sha256"] != EXPECTED_CONTROLLER_SHA256
    ):
        raise ValueError("objective-direction runtime assets do not match")
    preflight["input_sha256"].update(
        model=runtime["model_sha256"], controller=runtime["controller_sha256"]
    )
    env = _build_environment(hparams, reference_path)
    profile = get_solver_profile("g1-4x5")
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    validate_parent_checkpoint(checkpoint)
    actor, parent_params, actor_norm_state = _load_policy(env, checkpoint_path, seed)
    actor_normalizer = Normalizer(env.actor_frame_obs_dim)
    critic_normalizer = Normalizer(env.critic_obs_dim)
    critic_layers = checkpoint.target_critic_params["params"]
    dense_names = sorted(
        (name for name in critic_layers if name.startswith("Dense_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    critic_hidden = tuple(
        int(critic_layers[name]["kernel"].shape[-1]) for name in dense_names[:-1]
    )
    critic = Critic(hidden=critic_hidden)
    residual = PreviewResidualAdapter(action_dim=env.action_dim, hidden_dim=256)
    template = residual.init(
        jax.random.PRNGKey(13),
        jnp.zeros(env.actor_frame_obs_dim, dtype=jnp.float32),
    )
    with expert_path.open("rb") as stream:
        expert_params = pickle.load(stream)
    adapter_params, transfer = transplant_zero_head_recovery_features(
        template, expert_params
    )
    if transfer.get("output_head_zero") is not True:
        raise ValueError("objective-direction adapter head is not exactly zero")

    population = build_fixed_population_indices(seed)
    rows = _load_all_bank_rows(bank_path)
    carried_indices = population["source_index"][:CARRIED_POPULATION]
    exact_phases = population["source_index"][CARRIED_POPULATION:]
    keys = jax.random.split(jax.random.PRNGKey(seed), POPULATION)

    def carried_state(qpos, qvel, phase, last_act, history, rng):
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

    with solver_context(profile):
        carried_states = jax.jit(jax.vmap(carried_state))(
            jnp.asarray(rows["qpos"][carried_indices]),
            jnp.asarray(rows["qvel"][carried_indices]),
            jnp.asarray(rows["phase"][carried_indices], dtype=jnp.int32),
            jnp.asarray(rows["last_act"][carried_indices]),
            jnp.asarray(rows["actor_obs_history"][carried_indices]),
            keys[:CARRIED_POPULATION],
        )
        exact_states = jax.jit(jax.vmap(env.reset_at_phase))(
            keys[CARRIED_POPULATION:],
            jnp.zeros(EXACT_POPULATION, dtype=jnp.float64),
            jnp.asarray(exact_phases, dtype=jnp.int32),
        )
    states = jax.tree_util.tree_map(
        lambda left, right: jnp.concatenate((left, right), axis=0),
        carried_states,
        exact_states,
    )
    phases = np.concatenate(
        (np.asarray(rows["phase"][carried_indices], dtype=np.int32), exact_phases)
    )
    composite = FrozenPreviewResidualParams(parent_params, adapter_params)
    normalized = jax.vmap(
        lambda obs: env.normalize_actor_obs(actor_normalizer, actor_norm_state, obs)
    )(states.obs).astype(jnp.float32)
    candidate_action, parent_action, correction = apply_frozen_preview_residual(
        actor,
        residual,
        composite,
        normalized,
        history_len=env.actor_history_len,
        treatment_frame_dim=env.actor_frame_obs_dim,
    )
    if not np.array_equal(np.asarray(candidate_action), np.asarray(parent_action)) or not np.array_equal(
        np.asarray(correction), np.zeros_like(np.asarray(correction))
    ):
        raise ValueError("zero-head adapter does not preserve E023 actions")

    noise_std = jnp.asarray(hparams["action_noise_std_end"], dtype=jnp.float64)
    if noise_std.shape != (ACTION_DIM,) or not bool(jnp.all(jnp.isfinite(noise_std))):
        raise ValueError("E023 final RMR action-noise vector does not match")
    gamma = float(hparams["gamma"])
    clip_actions = bool(hparams["clip_sampled_actor_actions"])

    def loss(adapter, initial_state, noise, bootstrap_scale, *, horizon):
        def rollout_step(state, epsilon):
            _obs_key, env_key = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_key})
            obs_norm = env.normalize_actor_obs(
                actor_normalizer, actor_norm_state, state.obs
            ).astype(jnp.float32)
            action, _, _ = apply_frozen_preview_residual(
                actor,
                residual,
                FrozenPreviewResidualParams(parent_params, adapter),
                obs_norm,
                history_len=env.actor_history_len,
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            noisy = action.astype(jnp.float64) + epsilon.astype(jnp.float64) * noise_std
            if clip_actions:
                noisy = jnp.clip(noisy, -1.0, 1.0)
            next_state = env.step(state, noisy)
            return next_state, (
                next_state.reward,
                next_state.done,
                next_state.info["terminal"],
                next_state.info["bootstrap_critic_obs"],
            )

        final_state, trajectory = jax.lax.scan(
            rollout_step, initial_state, noise, length=horizon
        )
        reward, done, terminal, bootstrap_critic_obs = trajectory
        bootstrap_normalized = critic_normalizer.normalize(
            checkpoint.critic_normalizer, bootstrap_critic_obs
        ).astype(jnp.float32)
        bootstrap_v = squeeze_value_head(
            critic.apply(checkpoint.target_critic_params, bootstrap_normalized)
        )

        def accumulate(carry, values):
            total, running, discount = carry
            step_reward, step_done, step_terminal, value_next = values
            next_discount = discount * gamma
            running = running + discount * step_reward
            truncation = bootstrap_scale * (1.0 - step_terminal) * next_discount * value_next
            total = total + jnp.where(step_done, running + truncation, 0.0)
            return (
                total,
                jnp.where(step_done, 0.0, running),
                jnp.where(step_done, 1.0, next_discount),
            ), None

        (total, running, discount), _ = jax.lax.scan(
            accumulate,
            (jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0)),
            (reward, done, terminal, bootstrap_v),
        )
        final_obs = critic_normalizer.normalize(
            checkpoint.critic_normalizer,
            env._get_critic_obs(final_state.data, final_state.info),
        ).astype(jnp.float32)
        final_v = squeeze_value_head(
            critic.apply(checkpoint.target_critic_params, final_obs)
        )
        final_bootstrap = jnp.where(
            done[-1], 0.0, bootstrap_scale * discount * final_v
        )
        return -(total + running + final_bootstrap) / horizon

    tapes = build_fixed_noise_tapes(seed)

    if smoke:
        one_state = jax.tree_util.tree_map(lambda value: value[0], states)
        one_tape = jnp.asarray(tapes["a"][0])

        def smoke_gradient(*, horizon: int, bootstrap_scale: float):
            function = jax.jit(
                jax.grad(
                    lambda adapter: loss(
                        adapter,
                        one_state,
                        one_tape[:horizon],
                        bootstrap_scale,
                        horizon=horizon,
                    )
                )
            )
            with solver_context(profile):
                return jax.device_get(function(adapter_params))

        h24_smoke = smoke_gradient(horizon=24, bootstrap_scale=0.0)
        h48_smoke = smoke_gradient(horizon=48, bootstrap_scale=0.0)
        full_smoke = smoke_gradient(horizon=24, bootstrap_scale=1.0)
        bootstrap_smoke = jax.tree_util.tree_map(
            lambda full, immediate: full - immediate, full_smoke, h24_smoke
        )
        norms = {
            "h24": float(np.linalg.norm(_tree_vector(h24_smoke))),
            "h48": float(np.linalg.norm(_tree_vector(h48_smoke))),
            "bootstrap": float(np.linalg.norm(_tree_vector(bootstrap_smoke))),
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in norms.values()):
            raise ValueError("compiled objective-direction smoke gradient is invalid")
        smoke_summary = {
            "valid": True,
            "scientific": False,
            "protocol": f"{PROTOCOL}-compiled-smoke",
            "code_commit": code_commit,
            "common_noise_prefix": validate_common_noise_prefix(
                tapes["a"][:, :24], tapes["a"]
            ),
            "gradient_norms": norms,
            "input_sha256": preflight["input_sha256"],
        }
        _atomic_json(output_directory.resolve() / "smoke_summary.json", smoke_summary)
        return smoke_summary

    def capture(noise: np.ndarray, *, horizon: int, bootstrap_scale: float):
        gradient_fn = jax.jit(
            jax.vmap(
                jax.grad(
                    lambda adapter, state, epsilon: loss(
                        adapter, state, epsilon, bootstrap_scale, horizon=horizon
                    )
                ),
                in_axes=(None, 0, 0),
            )
        )
        chunks = []
        with solver_context(profile):
            for start in range(0, POPULATION, 256):
                stop = start + 256
                state_chunk = jax.tree_util.tree_map(lambda value: value[start:stop], states)
                chunk = gradient_fn(
                    adapter_params,
                    state_chunk,
                    jnp.asarray(noise[start:stop, :horizon]),
                )
                chunks.append(jax.device_get(chunk))
        return jax.tree_util.tree_map(
            lambda first, second: np.concatenate((first, second), axis=0),
            chunks[0],
            chunks[1],
        )

    gradient_trees = {
        "h24_a": capture(tapes["a"], horizon=24, bootstrap_scale=0.0),
        "h24_b": capture(tapes["b"], horizon=24, bootstrap_scale=0.0),
        "h48_a": capture(tapes["a"], horizon=48, bootstrap_scale=0.0),
    }
    h24_bootstrapped = capture(tapes["a"], horizon=24, bootstrap_scale=1.0)
    gradient_trees["bootstrap_a"] = jax.tree_util.tree_map(
        lambda full, immediate: full - immediate,
        h24_bootstrapped,
        gradient_trees["h24_a"],
    )
    matrices = {
        name: _tree_matrix(tree).astype(np.float32, copy=False)
        for name, tree in gradient_trees.items()
    }
    arrays: dict[str, np.ndarray] = {
        "source_kind": population["source_kind"],
        "source_index": population["source_index"],
        "phase": phases,
        "rng_key": np.asarray(keys, dtype=np.uint32),
        "initial_qpos": np.asarray(states.data.qpos),
        "initial_qvel": np.asarray(states.data.qvel),
        "initial_last_act": np.asarray(states.info["last_act"]),
        "initial_actor_obs_history": np.asarray(states.info["actor_obs_history"]),
        "noise_tape_a": tapes["a"],
        "noise_tape_b": tapes["b"],
        "h24_tape_env_cosine": _env_cosine(matrices["h24_a"], matrices["h24_b"]),
        "h24_h48_env_cosine": _env_cosine(matrices["h24_a"], matrices["h48_a"]),
        "h24_a_bootstrap_env_cosine": _env_cosine(
            matrices["h24_a"], matrices["bootstrap_a"]
        ),
        "h24_b_h48_env_cosine": _env_cosine(
            matrices["h24_b"], matrices["h48_a"]
        ),
        "h24_b_bootstrap_env_cosine": _env_cosine(
            matrices["h24_b"], matrices["bootstrap_a"]
        ),
        "h48_bootstrap_env_cosine": _env_cosine(
            matrices["h48_a"], matrices["bootstrap_a"]
        ),
    }
    aggregate_trees: dict[str, Any] = {}
    for name, tree in gradient_trees.items():
        aggregated = aggregate_audit_direction(
            tree,
            phases,
            phase_count=env.reference_length,
            clip_norm=1.0,
            alpha=0.5,
            iterations=32,
        )
        aggregate_trees[name] = aggregated.combined_gradient
        arrays.update(
            {
                f"{name}_combined": _tree_vector(aggregated.combined_gradient),
                f"{name}_task": _tree_matrix(aggregated.task_gradients),
                f"{name}_counts": np.asarray(aggregated.env_counts, dtype=np.int32),
                f"{name}_cosine": np.asarray(aggregated.cosine_matrix),
                f"{name}_weights": np.asarray(aggregated.weights),
                f"{name}_env_norm": np.linalg.norm(matrices[name], axis=1),
            }
        )
    gradient_validation = validate_gradient_artifacts(arrays)
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    preflight_path = output_directory / "preflight.json"
    gradient_path = output_directory / "gradient_evidence.npz"
    gradient_summary_path = output_directory / "gradient_summary.json"
    _atomic_json(preflight_path, preflight)
    _atomic_npz(gradient_path, arrays)

    ordered_names = ("h24_a", "h24_b", "h48_a", "bootstrap_a")
    aggregate_cosines = np.eye(4, dtype=np.float64)
    aggregate_dots = np.zeros((4, 4), dtype=np.float64)
    phase_pairwise_cosines = np.zeros((4, 4, 5), dtype=np.float64)
    for row in range(4):
        aggregate_dots[row, row] = float(
            np.dot(
                arrays[f"{ordered_names[row]}_combined"],
                arrays[f"{ordered_names[row]}_combined"],
            )
        )
        phase_pairwise_cosines[row, row] = 1.0
        for column in range(row + 1, 4):
            cosine = _vector_cosine(
                arrays[f"{ordered_names[row]}_combined"],
                arrays[f"{ordered_names[column]}_combined"],
            )
            aggregate_cosines[row, column] = cosine
            aggregate_cosines[column, row] = cosine
            dot = float(
                np.dot(
                    arrays[f"{ordered_names[row]}_combined"],
                    arrays[f"{ordered_names[column]}_combined"],
                )
            )
            aggregate_dots[row, column] = dot
            aggregate_dots[column, row] = dot
            phase_values = np.asarray(
                [
                    _vector_cosine(
                        arrays[f"{ordered_names[row]}_task"][index],
                        arrays[f"{ordered_names[column]}_task"][index],
                    )
                    for index in range(5)
                ]
            )
            phase_pairwise_cosines[row, column] = phase_values
            phase_pairwise_cosines[column, row] = phase_values
    phase_h24_tape = np.asarray(
        [
            _vector_cosine(
                arrays["h24_a_task"][index], arrays["h24_b_task"][index]
            )
            for index in range(5)
        ]
    )
    phase_h24_h48 = np.asarray(
        [
            _vector_cosine(
                arrays["h24_a_task"][index], arrays["h48_a_task"][index]
            )
            for index in range(5)
        ]
    )
    gradient_summary = {
        **gradient_validation,
        "scientific": True,
        "code_commit": code_commit,
        "input_sha256": preflight["input_sha256"],
        "gradient_artifact_path": str(gradient_path),
        "gradient_artifact_sha256": sha256_file(gradient_path),
        "aggregate_cosine_matrix": aggregate_cosines.tolist(),
        "aggregate_dot_matrix": aggregate_dots.tolist(),
        "phase_pairwise_cosines": phase_pairwise_cosines.tolist(),
        "h24_tape_phase_cosines": phase_h24_tape.tolist(),
        "h24_h48_phase_cosines": phase_h24_h48.tolist(),
    }
    _atomic_json(gradient_summary_path, gradient_summary)
    del gradient_trees, matrices, h24_bootstrapped

    def rollout_terminals(params: Any, initial_states: Any, maximum, *, horizon: int):
        def scan_step(current_states, elapsed):
            normalized_obs = env.normalize_actor_obs(
                actor_normalizer, actor_norm_state, current_states.obs
            ).astype(jnp.float32)
            action, _, _ = apply_frozen_preview_residual(
                actor,
                residual,
                FrozenPreviewResidualParams(parent_params, params),
                normalized_obs,
                history_len=env.actor_history_len,
                treatment_frame_dim=env.actor_frame_obs_dim,
            )
            if clip_actions:
                action = jnp.clip(action, -1.0, 1.0)
            next_states = jax.vmap(env.step)(current_states, action.astype(jnp.float64))
            terminal = (next_states.info["terminal"] > 0.5) & (elapsed < maximum)
            return next_states, terminal

        return jax.lax.scan(
            scan_step, initial_states, jnp.arange(horizon, dtype=jnp.int32)
        )[1]

    compiled_rollout_terminals = jax.jit(
        rollout_terminals, static_argnames=("horizon",)
    )

    def rollout_survival(
        params: Any,
        initial_states: Any,
        *,
        horizon: int,
        maximum_steps: np.ndarray,
    ) -> np.ndarray:
        maximum = jnp.asarray(maximum_steps, dtype=jnp.int32)
        with solver_context(profile):
            terminal_steps = compiled_rollout_terminals(
                params, initial_states, maximum, horizon=horizon
            )
        return _first_terminal_survival(
            np.asarray(terminal_steps).T, np.asarray(maximum_steps, dtype=np.int32)
        )

    bank_states = jax.tree_util.tree_map(lambda value: value[:BANK_ROWS], carried_states)
    bank_limits = np.full(BANK_ROWS, 32, dtype=np.int32)
    baseline_carried = rollout_survival(
        adapter_params, bank_states, horizon=32, maximum_steps=bank_limits
    )
    specs = proposal_specs()
    direction_tree_names = {
        "h24": "h24_a",
        "h48": "h48_a",
        "bootstrap": "bootstrap_a",
    }
    candidates: list[Any] = []
    candidate_carried_rows: list[np.ndarray] = []
    for spec in specs:
        candidate = normalized_descent_proposal(
            adapter_params,
            aggregate_trees[direction_tree_names[str(spec["direction"])]],
            displacement=float(spec["displacement"]),
        )
        candidates.append(candidate)
        candidate_carried_rows.append(
            rollout_survival(
                candidate, bank_states, horizon=32, maximum_steps=bank_limits
            )
        )
    candidate_carried = np.stack(candidate_carried_rows)
    selected_indices = np.full(3, -1, dtype=np.int32)
    selected_rows: list[dict[str, object] | None] = []
    for direction_index, direction in enumerate(PROPOSAL_DIRECTIONS):
        offset = direction_index * 4
        rows_for_direction = [
            {
                "proposal_index": offset + local_index,
                "label": specs[offset + local_index]["label"],
                "multiplier": specs[offset + local_index]["multiplier"],
                "candidate_survival": candidate_carried[offset + local_index].tolist(),
            }
            for local_index in range(4)
        ]
        selected = select_carried_safe_candidate(
            rows_for_direction, baseline_survival=baseline_carried.tolist()
        )
        selected_rows.append(selected)
        if selected is not None:
            selected_indices[direction_index] = int(selected["proposal_index"])

    ordinary_phases = np.asarray((0, 100, 200, 300, 400), dtype=np.int32)
    ordinary_limits = 499 - ordinary_phases
    compiled_step = build_compiled_step(env)

    def evaluate_ordinary(params: Any) -> np.ndarray:
        survival: list[int] = []
        with solver_context(profile):
            for phase, maximum in zip(
                ordinary_phases, ordinary_limits, strict=True
            ):
                state = env.reset_at_phase(
                    jax.random.PRNGKey(seed),
                    jnp.asarray(0.0, dtype=jnp.float64),
                    jnp.asarray(phase, dtype=jnp.int32),
                )
                done_trace: list[bool] = []
                for _ in range(int(maximum)):
                    normalized_obs = env.normalize_actor_obs(
                        actor_normalizer, actor_norm_state, state.obs
                    ).astype(jnp.float32)
                    action, _, _ = apply_frozen_preview_residual(
                        actor,
                        residual,
                        FrozenPreviewResidualParams(parent_params, params),
                        normalized_obs,
                        history_len=env.actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                    )
                    if clip_actions:
                        action = jnp.clip(action, -1.0, 1.0)
                    state = compiled_step(state, action.astype(jnp.float64))
                    done = bool(np.asarray(state.done) > 0.5)
                    done_trace.append(done)
                    if done:
                        break
                survival.append(
                    ordinary_steps_from_done(
                        done_trace, maximum_steps=int(maximum)
                    )
                )
        return np.asarray(survival, dtype=np.int32)

    baseline_ordinary = evaluate_ordinary(adapter_params)
    if tuple(map(int, baseline_ordinary)) != ORDINARY_BASELINE:
        raise ValueError(
            f"E023 ordinary baseline drifted: {baseline_ordinary.tolist()}"
        )
    selected_ordinary = np.full((3, 5), -1, dtype=np.int32)
    full_gate = np.zeros(3, dtype=bool)
    for direction_index, proposal_index in enumerate(selected_indices):
        if proposal_index < 0:
            continue
        survival = evaluate_ordinary(candidates[int(proposal_index)])
        selected_ordinary[direction_index] = survival
        full_gate[direction_index] = ordinary_componentwise_safe(
            survival.tolist(), ORDINARY_BASELINE
        )
    line_search_arrays = {
        "baseline_carried_survival": baseline_carried,
        "candidate_carried_survival": candidate_carried,
        "selected_proposal_index": selected_indices,
        "baseline_ordinary_survival": baseline_ordinary,
        "selected_ordinary_survival": selected_ordinary,
        "full_gate": full_gate,
    }
    line_search_validation = validate_line_search_evidence(line_search_arrays)
    line_search_path = output_directory / "line_search.npz"
    line_search_summary_path = output_directory / "line_search.json"
    _atomic_npz(line_search_path, line_search_arrays)
    line_search_summary = {
        **line_search_validation,
        "specs": specs,
        "selected": selected_rows,
        "full_gate": {
            direction: bool(full_gate[index])
            for index, direction in enumerate(PROPOSAL_DIRECTIONS)
        },
        "line_search_artifact_path": str(line_search_path),
        "line_search_artifact_sha256": sha256_file(line_search_path),
    }
    _atomic_json(line_search_summary_path, line_search_summary)

    total_gains = candidate_carried - baseline_carried[None, :]
    best_gain = [
        int(np.max(np.sum(total_gains[index * 4 : index * 4 + 4], axis=1)))
        for index in range(3)
    ]
    every_direction_mixed = all(
        any(
            np.any(total_gains[proposal_index] > 0)
            and np.any(total_gains[proposal_index] < 0)
            for proposal_index in range(index * 4, index * 4 + 4)
        )
        for index in range(3)
    )
    outcome = classify_objective_direction_audit(
        execution_valid=True,
        h24_tape_cosine=float(aggregate_cosines[0, 1]),
        h24_tape_phase_cosines=phase_h24_tape.tolist(),
        h24_h48_cosine=float(aggregate_cosines[0, 2]),
        h24_h48_phase_cosines=phase_h24_h48.tolist(),
        aggregate_pairwise_cosines=[
            float(aggregate_cosines[0, 2]),
            float(aggregate_cosines[0, 3]),
            float(aggregate_cosines[2, 3]),
        ],
        full_gate_by_direction={
            direction: bool(full_gate[index])
            for index, direction in enumerate(PROPOSAL_DIRECTIONS)
        },
        h48_carried_strictly_better=best_gain[1] > best_gain[0],
        every_direction_mixed=every_direction_mixed,
    )
    selection_path = output_directory / "selection.json"
    selection = {
        "valid": True,
        "protocol": f"{PROTOCOL}-selection",
        "outcome": outcome,
        "selected_proposal_index": selected_indices.tolist(),
        "full_gate": line_search_summary["full_gate"],
        "diagnostic_only": True,
        "retained_policy": None,
    }
    _atomic_json(selection_path, selection)
    cosine_path, survival_path = _plot_diagnostics(
        output_directory,
        aggregate_cosines=aggregate_cosines,
        baseline_carried=baseline_carried,
        candidate_carried=candidate_carried,
        baseline_ordinary=baseline_ordinary,
        selected_ordinary=selected_ordinary,
    )
    return publish_final_manifest(
        output_directory,
        artifacts={
            "preflight.json": preflight_path,
            "gradient_evidence.npz": gradient_path,
            "gradient_summary.json": gradient_summary_path,
            "line_search.npz": line_search_path,
            "line_search.json": line_search_summary_path,
            "selection.json": selection_path,
            "cosine_heatmap.png": cosine_path,
            "survival_plot.png": survival_path,
        },
        outcome=outcome,
        code_commit=code_commit,
        input_sha256=preflight["input_sha256"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    from tools.run_g1_tracking_shac import configure_jax

    args = build_parser().parse_args()
    configure_jax()
    manifest = run_gradient_capture(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        reference_path=args.reference_path,
        bank_path=args.source_bank,
        expert_path=args.expert_checkpoint,
        output_directory=args.output_directory,
        seed=args.seed,
        code_commit=args.code_commit,
        repository=args.repository,
        smoke=args.smoke,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
