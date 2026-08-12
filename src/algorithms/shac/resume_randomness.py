"""Exact, opt-in randomness rekeying for resumed SHAC training."""

from __future__ import annotations

import hashlib
import json
import os
from numbers import Integral
from pathlib import Path
from typing import Any

import jax
import numpy as np


TRAINER_KEY_NAMESPACE = 0x54524149
ENVIRONMENT_KEY_NAMESPACE = 0x454E5652
AUDIT_FILENAME = "resume_randomness_audit.json"


def _validated_environment_rng(state: Any) -> jax.Array:
    info = state.env_state.info
    if not isinstance(info, dict) or "rng" not in info:
        raise ValueError('resumed env_state must contain info["rng"]')
    rng = info["rng"]
    if getattr(rng, "ndim", None) != 2 or rng.shape[-1] != 2:
        raise ValueError(
            'resumed env_state.info["rng"] must have shape (num_envs, 2)'
        )
    return rng


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("resume random seed must be a non-negative integer")
    return int(seed)


def rekey_resumed_train_state(state: Any, *, seed: int) -> Any:
    """Replace only the trainer and per-environment RNG leaves."""
    seed = _validate_seed(seed)
    old_environment_rng = _validated_environment_rng(state)
    root = jax.random.PRNGKey(seed)
    trainer_key = jax.random.fold_in(root, TRAINER_KEY_NAMESPACE)
    environment_root = jax.random.fold_in(root, ENVIRONMENT_KEY_NAMESPACE)
    environment_keys = jax.random.split(
        environment_root, old_environment_rng.shape[0]
    )
    info = dict(state.env_state.info)
    info["rng"] = environment_keys
    return state.replace(
        key=trainer_key,
        env_state=state.env_state.replace(info=info),
    )


def apply_resume_randomness_setting(
    state: Any | None,
    *,
    seed: int | None,
) -> tuple[Any | None, dict[str, object] | None]:
    """Apply one explicit resume rekey, preserving legacy exact resume."""
    if seed is None:
        return state, None
    if state is None:
        raise ValueError("resume_random_seed requires a resumed checkpoint")
    candidate = rekey_resumed_train_state(state, seed=seed)
    audit = resume_randomness_audit(state, candidate, seed=seed)
    if not audit["valid"]:
        raise ValueError("resume randomness audit failed")
    return candidate, audit


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tree_bytes_exact(first: Any, second: Any) -> bool:
    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(second):
        return False
    for first_leaf, second_leaf in zip(
        jax.tree_util.tree_leaves(first),
        jax.tree_util.tree_leaves(second),
        strict=True,
    ):
        first_array = np.asarray(first_leaf)
        second_array = np.asarray(second_leaf)
        if (
            first_array.dtype != second_array.dtype
            or first_array.shape != second_array.shape
            or first_array.tobytes(order="C") != second_array.tobytes(order="C")
        ):
            return False
    return True


def resume_randomness_audit(
    original_state: Any,
    candidate_state: Any,
    *,
    seed: int,
) -> dict[str, object]:
    """Prove that a resume rekey changed both authorized RNG leaves only."""
    seed = _validate_seed(seed)
    old_environment_rng = _validated_environment_rng(original_state)
    new_environment_rng = _validated_environment_rng(candidate_state)
    if old_environment_rng.shape != new_environment_rng.shape:
        raise ValueError("resume rekey changed the environment RNG shape")

    restored_info = dict(candidate_state.env_state.info)
    restored_info["rng"] = old_environment_rng
    restored = candidate_state.replace(
        key=original_state.key,
        env_state=candidate_state.env_state.replace(info=restored_info),
    )
    trainer_changed = not _tree_bytes_exact(
        original_state.key, candidate_state.key
    )
    environment_changed = bool(
        np.all(
            np.any(
                np.asarray(old_environment_rng)
                != np.asarray(new_environment_rng),
                axis=-1,
            )
        )
    )
    non_rng_exact = _tree_bytes_exact(original_state, restored)
    valid = trainer_changed and environment_changed and non_rng_exact
    return {
        "schema_version": 1,
        "resume_random_seed": seed,
        "trainer_key_namespace": TRAINER_KEY_NAMESPACE,
        "environment_key_namespace": ENVIRONMENT_KEY_NAMESPACE,
        "environment_count": int(old_environment_rng.shape[0]),
        "old_trainer_key_sha256": _array_sha256(original_state.key),
        "new_trainer_key_sha256": _array_sha256(candidate_state.key),
        "old_environment_keys_sha256": _array_sha256(old_environment_rng),
        "new_environment_keys_sha256": _array_sha256(new_environment_rng),
        "trainer_key_changed": trainer_changed,
        "environment_keys_changed": environment_changed,
        "non_rng_state_exact": non_rng_exact,
        "valid": valid,
    }


def persist_resume_randomness_audit(
    directory: str | Path, payload: dict[str, object]
) -> Path:
    """Atomically persist one JSON resume-randomness audit."""
    output = Path(directory) / AUDIT_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output
