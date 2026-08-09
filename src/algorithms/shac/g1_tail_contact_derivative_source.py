"""Host-only E011 source receipts for the bounded E012 audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EVIDENCE_RELATIVE = Path("seed-1/evidence")
_RECEIPT_FILENAMES = (
    "outcome.json",
    "validity.json",
    "failure_weight_receipts.json",
    "estimator_receipts.json",
)
_SHARD_SEEDS = (0, 1, 2, 3)
_REQUIRED_VALIDITY_KEYS = frozenset(
    {
        "frozen_hashes",
        "weight_receipts_exact",
        "uniform_reproduction_exact",
        "tail_reproduction_exact",
        "stability_evidence_exact",
        "aggregate_gradients_finite_nonzero",
        "candidate_trees_finite_nonzero",
        "functional_steps_valid",
        "rollouts_fresh_replay_free_complete_finite",
    }
)
_RECOMPUTATION_BOOLEAN_KEYS = frozenset(
    {
        "weight_receipts_exact",
        "uniform_reproduction_exact",
        "tail_reproduction_exact",
        "stability_evidence_exact",
    }
)


@dataclass(frozen=True)
class E011SourceReceipts:
    """Validated E011 documents and shard data consumed by E012."""

    run_dir: Path
    evidence_dir: Path
    outcome: Mapping[str, Any]
    validity: Mapping[str, bool]
    failure_weight_receipts: Mapping[str, Any]
    estimator_receipts: Mapping[str, Any]
    losses_by_shard: Mapping[int, tuple[float, ...]]
    initial_phases_by_shard: Mapping[int, tuple[int, ...]]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_shard_rows(value: Any, *, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(_SHARD_SEEDS):
        raise ValueError(f"{label} shard identities must be exactly 0, 1, 2, 3")
    rows = {}
    for row in value:
        row = _require_mapping(row, label=f"{label} shard row")
        seed = row.get("seed")
        if type(seed) is not int or seed in rows:
            raise ValueError(f"{label} shard identities must be exactly 0, 1, 2, 3")
        rows[seed] = row
    if set(rows) != set(_SHARD_SEEDS):
        raise ValueError(f"{label} shard identities must be exactly 0, 1, 2, 3")
    return {seed: rows[seed] for seed in _SHARD_SEEDS}


def _validate_outcome(value: Any) -> Mapping[str, Any]:
    outcome = _require_mapping(value, label="outcome receipt")
    for key in ("verdict", "reason", "decision_metrics"):
        if key not in outcome:
            raise ValueError(f"outcome receipt is missing required {key}")
    if outcome["verdict"] != "failure-aware-unstable":
        raise ValueError("E011 verdict must be failure-aware-unstable")
    if not isinstance(outcome["reason"], str) or not outcome["reason"]:
        raise ValueError("outcome receipt reason must be a nonempty string")
    _require_mapping(outcome["decision_metrics"], label="outcome decision_metrics")
    return outcome


def _validate_validity(value: Any) -> Mapping[str, bool]:
    validity = _require_mapping(value, label="validity receipt")
    missing = sorted(_REQUIRED_VALIDITY_KEYS - set(validity))
    extra = sorted(set(validity) - _REQUIRED_VALIDITY_KEYS)
    if missing or extra:
        raise ValueError(
            f"validity receipt key mismatch: missing={missing}, extra={extra}"
        )
    failed = sorted(key for key, valid in validity.items() if valid is not True)
    if failed:
        raise ValueError("validity checks are not true: " + ", ".join(failed))
    return validity


def _validated_losses(value: Any, *, seed: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 64:
        raise ValueError(f"weighting shard {seed} losses must contain 64 values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"weighting shard {seed} losses must be finite numbers")
    return tuple(float(item) for item in value)


def _validated_phases(value: Any, *, seed: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 64:
        raise ValueError(
            f"weighting shard {seed} initial_phases must contain 64 values"
        )
    if any(type(item) is not int or not 0 <= item < 500 for item in value):
        raise ValueError(
            f"weighting shard {seed} initial_phases must be integers in [0, 500)"
        )
    return tuple(value)


def _validate_failure_receipts(
    value: Any,
) -> tuple[Mapping[str, Any], dict[int, tuple[float, ...]], dict[int, tuple[int, ...]]]:
    receipts = _require_mapping(value, label="failure-weight receipt")
    for key in ("weighting", "independent_recomputation", "gradient_hashes"):
        if key not in receipts:
            raise ValueError(f"failure-weight receipt is missing required {key}")

    weighting = _require_shard_rows(receipts["weighting"], label="weighting")
    losses = {}
    phases = {}
    for seed, row in weighting.items():
        if "losses" not in row:
            raise ValueError(f"weighting shard {seed} is missing losses")
        if "initial_phases" not in row:
            raise ValueError(f"weighting shard {seed} is missing initial_phases")
        losses[seed] = _validated_losses(row["losses"], seed=seed)
        phases[seed] = _validated_phases(row["initial_phases"], seed=seed)

    recomputation = _require_mapping(
        receipts["independent_recomputation"],
        label="independent recomputation receipt",
    )
    missing = sorted(_RECOMPUTATION_BOOLEAN_KEYS - set(recomputation))
    if missing:
        raise ValueError(f"independent recomputation receipt is missing {missing}")
    failed = sorted(
        key for key in _RECOMPUTATION_BOOLEAN_KEYS if recomputation[key] is not True
    )
    if failed:
        raise ValueError(
            "independent recomputation checks are not true: " + ", ".join(failed)
        )
    _require_mapping(
        recomputation.get("independent_host_recomputation"),
        label="independent host recomputation receipt",
    )

    gradient_hashes = _require_mapping(
        receipts["gradient_hashes"], label="gradient hashes receipt"
    )
    if "per_shard" not in gradient_hashes or "aggregate" not in gradient_hashes:
        raise ValueError("gradient hashes receipt is missing required structures")
    hash_rows = _require_shard_rows(
        gradient_hashes["per_shard"], label="gradient hashes"
    )
    for seed, row in hash_rows.items():
        for key in ("per_environment_clipped", "uniform", "tail"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(
                    f"gradient hashes shard {seed} is missing required {key}"
                )
    aggregate = _require_mapping(
        gradient_hashes["aggregate"], label="aggregate gradient hashes"
    )
    for key in ("uniform", "tail"):
        if not isinstance(aggregate.get(key), str) or not aggregate[key]:
            raise ValueError(f"aggregate gradient hashes are missing required {key}")
    return receipts, losses, phases


def _validate_estimator_receipts(value: Any) -> Mapping[str, Any]:
    receipts = _require_mapping(value, label="estimator receipt")
    if receipts.get("shared_rollout_identity") is not True:
        raise ValueError("estimator shared rollout identity must be true")
    algorithmic = _require_mapping(
        receipts.get("algorithmic_validity"), label="algorithmic validity receipt"
    )
    if not algorithmic:
        raise ValueError("algorithmic validity receipt must not be empty")
    failed = sorted(key for key, valid in algorithmic.items() if valid is not True)
    if failed:
        raise ValueError(
            "algorithmic validity checks are not true: " + ", ".join(failed)
        )
    per_shard = _require_mapping(
        receipts.get("per_shard"), label="estimator per_shard receipt"
    )
    expected_keys = {str(seed) for seed in _SHARD_SEEDS}
    if set(per_shard) != expected_keys:
        raise ValueError("estimator shard keys must be exactly 0, 1, 2, 3")
    for seed in _SHARD_SEEDS:
        row = _require_mapping(per_shard[str(seed)], label=f"estimator shard {seed}")
        for key in ("pathwise", "score"):
            if key not in row:
                raise ValueError(f"estimator shard {seed} is missing required {key}")
            _require_mapping(row[key], label=f"estimator shard {seed} {key}")
        if row["pathwise"] != row["score"]:
            raise ValueError(f"estimator shard {seed} pathwise and score receipts differ")
    return receipts


def validate_e011_source_receipts(
    *,
    run_dir: Path,
    outcome: Any,
    validity: Any,
    failure_weight_receipts: Any,
    estimator_receipts: Any,
) -> E011SourceReceipts:
    """Validate decoded E011 receipts without simulator execution or writes."""

    resolved_run_dir = Path(run_dir).resolve()
    validated_outcome = _validate_outcome(outcome)
    validated_validity = _validate_validity(validity)
    validated_failure, losses, phases = _validate_failure_receipts(
        failure_weight_receipts
    )
    validated_estimators = _validate_estimator_receipts(estimator_receipts)
    return E011SourceReceipts(
        run_dir=resolved_run_dir,
        evidence_dir=resolved_run_dir / _EVIDENCE_RELATIVE,
        outcome=validated_outcome,
        validity=validated_validity,
        failure_weight_receipts=validated_failure,
        estimator_receipts=validated_estimators,
        losses_by_shard=losses,
        initial_phases_by_shard=phases,
    )


def load_e011_source_receipts(run_dir: Path) -> E011SourceReceipts:
    """Load the four source receipts from an already verified E011 run."""

    resolved_run_dir = Path(run_dir).resolve()
    evidence_dir = resolved_run_dir / _EVIDENCE_RELATIVE
    manifest_path = evidence_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load frozen E011 manifest manifest.json: {error}"
        ) from error
    manifest = _require_mapping(manifest, label="frozen E011 manifest")
    artifacts = _require_mapping(
        manifest.get("artifacts"), label="frozen E011 manifest artifacts"
    )
    documents = {}
    for name in _RECEIPT_FILENAMES:
        path = evidence_dir / name
        try:
            encoded = path.read_bytes()
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load required E011 receipt {name}: {error}") from error
        expected_sha256 = artifacts.get(name)
        if (
            not isinstance(expected_sha256, str)
            or hashlib.sha256(encoded).hexdigest() != expected_sha256
        ):
            raise ValueError(f"E011 receipt {name} SHA-256 differs from manifest")
        try:
            documents[name] = json.loads(encoded)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load required E011 receipt {name}: {error}") from error
    return validate_e011_source_receipts(
        run_dir=resolved_run_dir,
        outcome=documents["outcome.json"],
        validity=documents["validity.json"],
        failure_weight_receipts=documents["failure_weight_receipts.json"],
        estimator_receipts=documents["estimator_receipts.json"],
    )
