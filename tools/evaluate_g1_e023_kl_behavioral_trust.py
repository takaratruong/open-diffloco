"""Evaluate PPO-style KL backtracking against protected E023 behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.core.rmr_action_noise import RMR_WALK_MODEL_999_ACTION_STD
from tools.build_g1_e023_carried_reset_bank import validate_code_commit


CANDIDATE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
E023_FLOORS = (116, 99, 67, 49, 24)
MEAN_KL_LIMIT = 0.01
P95_STATE_KL_LIMIT = 0.02
PHASES = (0, 25, 50, 75, 100)
TRACKING_ERROR_KEYS = (
    "mean_body_position_error",
    "mean_body_orientation_error",
    "mean_body_linear_velocity_error",
    "mean_body_angular_velocity_error",
    "mean_anchor_position_error",
    "mean_anchor_orientation_error",
)
EXPECTED_SOURCE_SHA256 = (
    "2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f"
)
EXPECTED_SOURCE_HPARAMS_SHA256 = (
    "a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650"
)
EXPECTED_AUDIT_BANK_SHA256 = (
    "dd17aca2ffa94fd5843dfae0b01c46f893fb7480ac429957b9a5e7ccfc084df2"
)
EXPECTED_REFERENCE_SHA256 = (
    "b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca"
)


# A standard-library type keeps artifacts importable whether this tool is invoked
# as a module or as a script; a script-local dataclass would pickle as __main__.
EvaluationActorState = SimpleNamespace


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _alpha_filename(alpha: float) -> str:
    return f"candidate_alpha_{alpha:.5f}".replace(".", "p") + ".pkl"


def _finite_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{label} leaves must be finite numeric arrays")
    return array


def interpolate_actor_tree(source: Any, proposal: Any, *, alpha: float) -> Any:
    """Interpolate one registered actor proposal from an immutable source."""
    if alpha not in CANDIDATE_ALPHAS:
        raise ValueError("alpha is not on the registered candidate grid")
    if jax.tree_util.tree_structure(source) != jax.tree_util.tree_structure(
        proposal
    ):
        raise ValueError("source and proposal actor structures differ")

    def interpolate(source_leaf: Any, proposal_leaf: Any) -> Any:
        source_array = _finite_array(source_leaf, label="source actor")
        proposal_array = _finite_array(proposal_leaf, label="proposal actor")
        if source_array.shape != proposal_array.shape:
            raise ValueError("source and proposal actor structures differ")
        if source_array.dtype != proposal_array.dtype:
            raise ValueError("source and proposal actor dtypes differ")
        if alpha == 1.0:
            return proposal_leaf
        return source_leaf + alpha * (proposal_leaf - source_leaf)

    return jax.tree_util.tree_map(interpolate, source, proposal)


def build_evaluation_candidate(
    source_state: Any,
    proposal_state: Any,
    *,
    alpha: float,
) -> EvaluationActorState:
    """Build a compact candidate using only source normalization state."""
    for state, label in (
        (source_state, "source"),
        (proposal_state, "proposal"),
    ):
        if not hasattr(state, "actor_params") or not hasattr(state, "normalizer"):
            raise ValueError(f"{label} checkpoint lacks actor state")
    return EvaluationActorState(
        actor_params=interpolate_actor_tree(
            source_state.actor_params,
            proposal_state.actor_params,
            alpha=alpha,
        ),
        normalizer=source_state.normalizer,
        evaluation_only=True,
        resumable=False,
    )


def publish_evaluation_candidates(
    source_checkpoint: str | Path,
    proposal_checkpoint: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Atomically publish compact, hash-bound evaluation candidates."""
    source_path = Path(source_checkpoint).resolve()
    proposal_path = Path(proposal_checkpoint).resolve()
    output = Path(output_directory).resolve()
    hparams_path = source_path.with_name("hparams.json")
    for path, label in (
        (source_path, "source checkpoint"),
        (proposal_path, "proposal checkpoint"),
        (hparams_path, "source hparams"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is missing")
    manifest_path = output / "candidate_manifest.json"
    if manifest_path.exists():
        raise ValueError("candidate manifest already exists")
    with source_path.open("rb") as stream:
        source_state = pickle.load(stream)
    with proposal_path.open("rb") as stream:
        proposal_state = pickle.load(stream)
    output.mkdir(parents=True, exist_ok=True)
    published_hparams = output / "hparams.json"
    _atomic_write_bytes(published_hparams, hparams_path.read_bytes())

    rows: list[dict[str, Any]] = []
    for alpha in CANDIDATE_ALPHAS:
        candidate = build_evaluation_candidate(
            source_state, proposal_state, alpha=alpha
        )
        checkpoint_path = output / _alpha_filename(alpha)
        if checkpoint_path.exists():
            raise ValueError("candidate checkpoint already exists")
        _atomic_write_bytes(
            checkpoint_path,
            pickle.dumps(candidate, protocol=pickle.HIGHEST_PROTOCOL),
        )
        rows.append(
            {
                "alpha": alpha,
                "checkpoint_path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "evaluation_only": True,
                "resumable": False,
            }
        )
    manifest = {
        "protocol": "g1-e023-kl-evaluation-candidates-v1",
        "evaluation_only": True,
        "resumable": False,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "proposal_checkpoint": str(proposal_path),
        "proposal_checkpoint_sha256": _sha256(proposal_path),
        "hparams_path": str(published_hparams),
        "hparams_sha256": _sha256(published_hparams),
        "candidates": rows,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def validate_candidate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Revalidate every candidate and provenance hash from disk."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "g1-e023-kl-evaluation-candidates-v1"
        or payload.get("evaluation_only") is not True
        or payload.get("resumable") is not False
    ):
        raise ValueError("candidate manifest contract is invalid")
    provenance = (
        ("source_checkpoint", "source_checkpoint_sha256"),
        ("proposal_checkpoint", "proposal_checkpoint_sha256"),
        ("hparams_path", "hparams_sha256"),
    )
    for path_key, hash_key in provenance:
        artifact = Path(payload.get(path_key, ""))
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or _sha256(artifact) != payload.get(hash_key)
        ):
            raise ValueError(f"{path_key} SHA-256 does not match")
    rows = payload.get("candidates")
    if (
        not isinstance(rows, list)
        or tuple(row.get("alpha") for row in rows) != CANDIDATE_ALPHAS
    ):
        raise ValueError("candidate manifest grid is invalid")
    for row in rows:
        checkpoint = Path(row.get("checkpoint_path", ""))
        if (
            row.get("evaluation_only") is not True
            or row.get("resumable") is not False
            or checkpoint.parent.resolve() != path.parent
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or _sha256(checkpoint) != row.get("sha256")
        ):
            raise ValueError("candidate checkpoint SHA-256 does not match")
        with checkpoint.open("rb") as stream:
            candidate = pickle.load(stream)
        if (
            not isinstance(candidate, EvaluationActorState)
            or candidate.evaluation_only is not True
            or candidate.resumable is not False
        ):
            raise ValueError("candidate checkpoint contract is invalid")
    return payload


def actor_means_from_bank(
    state: Any,
    bank_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Apply one standard E023 actor to immutable audit-bank histories."""
    if not hasattr(state, "actor_params") or not hasattr(state, "normalizer"):
        raise ValueError("checkpoint lacks actor state")
    with np.load(Path(bank_path), allow_pickle=False) as archive:
        histories = np.asarray(archive["initial_actor_obs_history"])
        noise_tape = np.asarray(archive["noise_tape_a"])
    if (
        histories.ndim != 3
        or histories.shape[1:] != (10, 328)
        or noise_tape.ndim != 3
        or noise_tape.shape[0] != histories.shape[0]
        or noise_tape.shape[2] != 29
    ):
        raise ValueError("audit bank does not match the E023 actor contract")
    count = histories.shape[0] if limit is None else limit
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(histories):
        raise ValueError("audit bank limit is invalid")
    frames = jnp.asarray(histories[:count])
    normalized = Normalizer(328).normalize(state.normalizer, frames)
    flattened = normalized.reshape(count, 3280).astype(jnp.float32)
    actor = Actor(
        29,
        hidden=(512, 256, 128),
        squash=False,
        layer_norm=True,
        zero_output=False,
    )
    means = np.asarray(actor.apply(state.actor_params, flattened))
    epsilon = noise_tape[:count, 0, :]
    if means.shape != (count, 29) or not np.isfinite(means).all():
        raise ValueError("actor means are invalid")
    if epsilon.shape != means.shape or not np.isfinite(epsilon).all():
        raise ValueError("audit epsilon is invalid")
    return {"actor_mean": means, "epsilon": epsilon}


def publish_proximity_evidence(
    source_checkpoint: str | Path,
    candidate_manifest: str | Path,
    audit_bank: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Evaluate every compact actor and publish raw Gaussian evidence."""
    source_path = Path(source_checkpoint).resolve()
    manifest_path = Path(candidate_manifest).resolve()
    bank_path = Path(audit_bank).resolve()
    output = Path(output_directory).resolve()
    manifest = validate_candidate_manifest(manifest_path)
    with source_path.open("rb") as stream:
        source_state = pickle.load(stream)
    source_report = actor_means_from_bank(source_state, bank_path)
    candidate_means: list[np.ndarray] = []
    for row in manifest["candidates"]:
        with Path(row["checkpoint_path"]).open("rb") as stream:
            candidate_state = pickle.load(stream)
        candidate_report = actor_means_from_bank(candidate_state, bank_path)
        if not np.array_equal(
            candidate_report["epsilon"], source_report["epsilon"]
        ):
            raise ValueError("candidate epsilon does not match the source")
        candidate_means.append(candidate_report["actor_mean"])
    candidate_array = np.stack(candidate_means)
    std = np.asarray(RMR_WALK_MODEL_999_ACTION_STD)
    summary = build_proximity_summary(
        source_report["actor_mean"],
        candidate_array,
        std,
        source_report["epsilon"],
    )
    raw_reports = [
        gaussian_proximity(
            source_report["actor_mean"],
            candidate,
            std,
            source_report["epsilon"],
        )
        for candidate in candidate_array
    ]
    evidence_path = output / "proximity_evidence.npz"
    _atomic_write_npz(
        evidence_path,
        candidate_alphas=np.asarray(CANDIDATE_ALPHAS, dtype=np.float64),
        source_actor_mean=source_report["actor_mean"],
        candidate_actor_mean=candidate_array,
        action_std=std,
        epsilon=source_report["epsilon"],
        kl_by_state_action=np.stack(
            [row["kl_by_state_action"] for row in raw_reports]
        ),
        kl_by_state=np.stack([row["kl_by_state"] for row in raw_reports]),
        log_ratio_by_state=np.stack(
            [row["log_ratio_by_state"] for row in raw_reports]
        ),
    )
    summary.update(
        {
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": _sha256(source_path),
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": _sha256(manifest_path),
            "audit_bank": str(bank_path),
            "audit_bank_sha256": _sha256(bank_path),
            "evidence_path": str(evidence_path),
            "evidence_sha256": _sha256(evidence_path),
        }
    )
    _atomic_write_json(output / "proximity_summary.json", summary)
    return summary


def gaussian_proximity(
    old_mean: Any,
    new_mean: Any,
    std: Any,
    epsilon: Any,
) -> dict[str, Any]:
    """Measure fixed-variance Gaussian KL and common-sample log ratios."""
    old = _finite_array(old_mean, label="old mean").astype(np.float64)
    new = _finite_array(new_mean, label="new mean").astype(np.float64)
    noise = _finite_array(epsilon, label="epsilon").astype(np.float64)
    scale = np.asarray(std)
    if old.ndim != 2 or old.shape[0] == 0 or old.shape[1] == 0:
        raise ValueError("policy means must be a nonempty state-action matrix")
    if new.shape != old.shape or noise.shape != old.shape:
        raise ValueError("policy mean and epsilon shapes must match")
    if scale.shape != (old.shape[1],):
        raise ValueError("standard deviation shape does not match actions")
    if not np.isfinite(scale).all():
        raise ValueError("standard deviation must be finite")
    if np.any(scale <= 0):
        raise ValueError("standard deviation must be positive")
    scale = scale.astype(np.float64)

    delta = new - old
    kl_by_state_action = np.square(delta) / (2.0 * np.square(scale))
    kl_by_state = np.sum(kl_by_state_action, axis=1)
    sampled_action = old + scale * noise
    old_quadratic = np.square((sampled_action - old) / scale)
    new_quadratic = np.square((sampled_action - new) / scale)
    log_ratio_by_state = -0.5 * np.sum(new_quadratic - old_quadratic, axis=1)

    mean_kl = float(np.mean(kl_by_state_action))
    p95_state_kl = float(np.percentile(kl_by_state, 95))
    max_state_kl = float(np.max(kl_by_state))
    absolute_log_ratio = np.abs(log_ratio_by_state)
    return {
        "kl_by_state_action": kl_by_state_action,
        "kl_by_state": kl_by_state,
        "log_ratio_by_state": log_ratio_by_state,
        "mean_kl": mean_kl,
        "p95_state_kl": p95_state_kl,
        "max_state_kl": max_state_kl,
        "mean_log_ratio": float(np.mean(log_ratio_by_state)),
        "p95_abs_log_ratio": float(np.percentile(absolute_log_ratio, 95)),
        "max_abs_log_ratio": float(np.max(absolute_log_ratio)),
        "kl_feasible": (
            mean_kl <= MEAN_KL_LIMIT
            and p95_state_kl <= P95_STATE_KL_LIMIT
        ),
    }


def _proximity_scalars(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "mean_kl",
            "p95_state_kl",
            "max_state_kl",
            "mean_log_ratio",
            "p95_abs_log_ratio",
            "max_abs_log_ratio",
            "kl_feasible",
        )
    }


def build_proximity_summary(
    source_actor_mean: Any,
    candidate_actor_mean: Any,
    std: Any,
    epsilon: Any,
) -> dict[str, Any]:
    """Summarize raw source/candidate actor evidence in registered order."""
    source = _finite_array(source_actor_mean, label="source actor mean")
    candidates = _finite_array(
        candidate_actor_mean, label="candidate means"
    )
    if candidates.shape != (len(CANDIDATE_ALPHAS), *source.shape):
        raise ValueError("candidate means do not match the registered grid")
    identity = gaussian_proximity(source, source, std, epsilon)
    if (
        identity["mean_kl"] != 0.0
        or identity["p95_state_kl"] != 0.0
        or identity["max_state_kl"] != 0.0
        or np.any(identity["log_ratio_by_state"] != 0.0)
    ):
        raise ValueError("source identity proximity is not exact zero")
    rows: list[dict[str, Any]] = []
    for alpha, candidate in zip(CANDIDATE_ALPHAS, candidates, strict=True):
        report = gaussian_proximity(source, candidate, std, epsilon)
        rows.append(
            {
                "alpha": alpha,
                "valid": True,
                **_proximity_scalars(report),
            }
        )
    return {
        "protocol": "g1-e023-gaussian-proximity-v1",
        "valid": True,
        "mean_kl_limit": MEAN_KL_LIMIT,
        "p95_state_kl_limit": P95_STATE_KL_LIMIT,
        "source_identity": _proximity_scalars(identity),
        "candidates": rows,
    }


def _validated_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if tuple(row.get("alpha") for row in rows) != CANDIDATE_ALPHAS:
        raise ValueError("rows do not match the registered candidate grid")
    validated: list[dict[str, Any]] = []
    for row in rows:
        if row.get("valid") is not True or not isinstance(
            row.get("kl_feasible"), bool
        ):
            raise ValueError("candidate row is invalid")
        survival = row.get("survival")
        if (
            not isinstance(survival, list)
            or len(survival) != len(E023_FLOORS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in survival
            )
        ):
            raise ValueError("candidate survival is invalid")
        validated.append(dict(row))
    return validated


def select_kl_behavioral_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_survival: Sequence[int] = E023_FLOORS,
) -> dict[str, Any]:
    """Select the largest KL-feasible candidate preserving every E023 phase."""
    if (
        len(source_survival) != len(E023_FLOORS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < floor
            for value, floor in zip(
                source_survival, E023_FLOORS, strict=True
            )
        )
    ):
        raise ValueError("source survival is below the protected E023 floor")
    candidates = _validated_rows(rows)
    kl_feasible = [row for row in candidates if row["kl_feasible"]]
    safe = [
        row
        for row in kl_feasible
        if all(
            value >= floor
            for value, floor in zip(row["survival"], E023_FLOORS, strict=True)
        )
    ]
    if safe:
        selected = safe[0]
        advances = any(
            value > floor
            for value, floor in zip(
                selected["survival"], E023_FLOORS, strict=True
            )
        ) or selected.get("tracking_advances") is True
        outcome = "kl-trust-advances" if advances else "kl-trust-preserves-only"
        selected_alpha: float | None = float(selected["alpha"])
    elif kl_feasible:
        outcome = "kl-feasible-behaviorally-unsafe"
        selected_alpha = None
    else:
        outcome = "no-nonzero-kl-candidate"
        selected_alpha = None
    return {
        "protocol": "g1-e023-kl-behavioral-selection-v1",
        "valid": True,
        "source_survival": list(source_survival),
        "outcome": outcome,
        "selected_alpha": selected_alpha,
    }


def _phase_grid_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("tracking_velocity_kernel") != "exponential":
        raise ValueError("phase grid must use the E023 exponential kernel")
    if payload.get("solver_profile") != "g1-4x5":
        raise ValueError("phase grid solver does not match E023")
    summary = payload.get("summary")
    results = payload.get("results")
    if (
        not isinstance(summary, Mapping)
        or tuple(summary.get("phases", ())) != PHASES
        or not isinstance(results, list)
        or len(results) != len(PHASES)
    ):
        raise ValueError("phase grid contract is invalid")
    survival = summary.get("survival")
    if (
        not isinstance(survival, list)
        or len(survival) != len(PHASES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in survival
        )
        or survival != [row.get("steps") for row in results]
        or [row.get("phase") for row in results] != list(PHASES)
    ):
        raise ValueError("phase grid survival is invalid")
    metric_keys = ("mean_reward", *TRACKING_ERROR_KEYS)
    metrics: dict[str, float] = {}
    for key in metric_keys:
        values = np.asarray([row.get(key, np.nan) for row in results], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("phase grid tracking metrics are invalid")
        metrics[key] = float(np.mean(values))
    return {"survival": survival, "tracking_metrics": metrics}


def _tracking_advances(
    source: Mapping[str, float], candidate: Mapping[str, float]
) -> bool:
    return bool(
        candidate["mean_reward"] > source["mean_reward"]
        and candidate["mean_body_position_error"]
        < source["mean_body_position_error"]
        and all(
            candidate[key] <= source[key] * 1.05
            for key in TRACKING_ERROR_KEYS
        )
    )


def build_phase_selection(
    proximity_summary: Mapping[str, Any],
    source_phase_grid: Mapping[str, Any],
    candidate_phase_grids: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join KL evidence with exact replay-free componentwise behavior."""
    if proximity_summary.get("valid") is not True:
        raise ValueError("proximity evidence is invalid")
    proximity_rows = proximity_summary.get("candidates")
    if (
        not isinstance(proximity_rows, list)
        or tuple(row.get("alpha") for row in proximity_rows)
        != CANDIDATE_ALPHAS
        or len(candidate_phase_grids) != len(CANDIDATE_ALPHAS)
    ):
        raise ValueError("proximity candidate grid is invalid")
    source = _phase_grid_evidence(source_phase_grid)
    if any(
        value < floor
        for value, floor in zip(
            source["survival"], E023_FLOORS, strict=True
        )
    ):
        raise ValueError("source phase grid is below protected E023")
    rows: list[dict[str, Any]] = []
    for proximity, phase_grid in zip(
        proximity_rows, candidate_phase_grids, strict=True
    ):
        behavioral = _phase_grid_evidence(phase_grid)
        rows.append(
            {
                **dict(proximity),
                "survival": behavioral["survival"],
                "tracking_metrics": behavioral["tracking_metrics"],
                "tracking_advances": _tracking_advances(
                    source["tracking_metrics"],
                    behavioral["tracking_metrics"],
                ),
            }
        )
    selection = select_kl_behavioral_candidate(
        rows, source_survival=source["survival"]
    )
    return {
        **selection,
        "source_tracking_metrics": source["tracking_metrics"],
        "candidates": rows,
    }


def build_phase_grid_command(
    *,
    checkpoint: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    code_commit: str,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the exact current-code E023 replay-free evaluator command."""
    return [
        python_executable,
        "-m",
        "tools.evaluate_g1_flax_phase_grid",
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--reference-path",
        str(Path(reference_path).resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--phases",
        *(str(phase) for phase in PHASES),
        "--seed",
        "0",
        "--solver-profile",
        "g1-4x5",
        "--code-commit",
        code_commit,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--audit-bank", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    return parser


def _validate_runtime_inputs(
    *,
    repository: Path,
    source_checkpoint: Path,
    proposal_checkpoint: Path,
    audit_bank: Path,
    reference_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    commit = validate_code_commit(repository, code_commit)
    source_hparams = source_checkpoint.with_name("hparams.json")
    expected = (
        (source_checkpoint, EXPECTED_SOURCE_SHA256, "source checkpoint"),
        (source_hparams, EXPECTED_SOURCE_HPARAMS_SHA256, "source hparams"),
        (audit_bank, EXPECTED_AUDIT_BANK_SHA256, "audit bank"),
        (reference_path, EXPECTED_REFERENCE_SHA256, "reference"),
    )
    for path, digest, label in expected:
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"{label} SHA-256 does not match")
    if proposal_checkpoint.is_symlink() or not proposal_checkpoint.is_file():
        raise ValueError("proposal checkpoint is missing")
    with source_checkpoint.open("rb") as stream:
        source_state = pickle.load(stream)
    with proposal_checkpoint.open("rb") as stream:
        proposal_state = pickle.load(stream)
    if int(source_state.step) != 1_572_864 or int(proposal_state.step) != 1_585_152:
        raise ValueError("source/proposal checkpoint steps do not match")
    if jax.tree_util.tree_structure(source_state.actor_params) != jax.tree_util.tree_structure(
        proposal_state.actor_params
    ):
        raise ValueError("source/proposal actor structures differ")
    return {
        "protocol": "g1-e023-kl-behavioral-preflight-v1",
        "valid": True,
        "code_commit": commit,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "source_hparams_sha256": EXPECTED_SOURCE_HPARAMS_SHA256,
        "proposal_checkpoint": str(proposal_checkpoint),
        "proposal_checkpoint_sha256": _sha256(proposal_checkpoint),
        "audit_bank": str(audit_bank),
        "audit_bank_sha256": EXPECTED_AUDIT_BANK_SHA256,
        "reference_path": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "phases": list(PHASES),
        "candidate_alphas": list(CANDIDATE_ALPHAS),
        "mean_kl_limit": MEAN_KL_LIMIT,
        "p95_state_kl_limit": P95_STATE_KL_LIMIT,
    }


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    source = args.source_checkpoint.resolve()
    proposal = args.proposal_checkpoint.resolve()
    bank = args.audit_bank.resolve()
    reference = args.reference_path.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = _validate_runtime_inputs(
        repository=repository,
        source_checkpoint=source,
        proposal_checkpoint=proposal,
        audit_bank=bank,
        reference_path=reference,
        code_commit=args.code_commit,
    )
    _atomic_write_json(output / "preflight.json", preflight)
    manifest = publish_evaluation_candidates(
        source, proposal, output / "candidates"
    )
    proximity = publish_proximity_evidence(
        source,
        output / "candidates" / "candidate_manifest.json",
        bank,
        output,
    )

    phase_directory = output / "phase_grids"
    source_output = phase_directory / "source.json"
    commands = [
        build_phase_grid_command(
            checkpoint=source,
            reference_path=reference,
            output_path=source_output,
            code_commit=args.code_commit,
        )
    ]
    candidate_outputs: list[Path] = []
    for row in manifest["candidates"]:
        candidate_output = phase_directory / (
            f"alpha_{float(row['alpha']):.5f}".replace(".", "p") + ".json"
        )
        candidate_outputs.append(candidate_output)
        commands.append(
            build_phase_grid_command(
                checkpoint=row["checkpoint_path"],
                reference_path=reference,
                output_path=candidate_output,
                code_commit=args.code_commit,
            )
        )
    for command in commands:
        subprocess.run(command, cwd=repository, check=True)
    source_grid = json.loads(source_output.read_text(encoding="utf-8"))
    candidate_grids = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in candidate_outputs
    ]
    selection = build_phase_selection(
        proximity, source_grid, candidate_grids
    )
    selection.update(
        {
            "preflight_sha256": _sha256(output / "preflight.json"),
            "candidate_manifest_sha256": _sha256(
                output / "candidates" / "candidate_manifest.json"
            ),
            "proximity_summary_sha256": _sha256(
                output / "proximity_summary.json"
            ),
            "source_phase_grid": str(source_output),
            "source_phase_grid_sha256": _sha256(source_output),
            "candidate_phase_grids": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in candidate_outputs
            ],
        }
    )
    _atomic_write_json(output / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
