import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from src.core.data_structures import EnvState, NormState, TrainState


def _state(*, rng: jax.Array | None = None) -> TrainState:
    info = {
        "rng": (
            jax.random.split(jax.random.PRNGKey(9), 4)
            if rng is None
            else rng
        ),
        "phase": jp.arange(4, dtype=jp.int32),
    }
    return TrainState(
        key=jax.random.PRNGKey(7),
        env_state=EnvState(
            data=jp.arange(3.0),
            obs=jp.arange(8.0).reshape(4, 2),
            reward=jp.arange(4.0),
            done=jp.zeros(4),
            info=info,
            metrics={"tracking": jp.ones(4)},
        ),
        actor_params={"weight": jp.arange(6.0).reshape(2, 3)},
        critic_params={"weight": jp.arange(2.0)},
        target_critic_params={"weight": jp.arange(2.0)},
        normalizer=NormState(
            mean=jp.arange(2.0), var=jp.ones(2), count=jp.array(4.0)
        ),
        actor_opt={"moment": jp.arange(3.0)},
        critic_opt={"moment": jp.arange(2.0)},
        step=jp.array(123, dtype=jp.int32),
    )


def _assert_tree_exact(first, second) -> None:
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(
        second
    )
    for first_leaf, second_leaf in zip(
        jax.tree_util.tree_leaves(first),
        jax.tree_util.tree_leaves(second),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def _restore_authorized_rngs(candidate: TrainState, original: TrainState) -> TrainState:
    info = dict(candidate.env_state.info)
    info["rng"] = original.env_state.info["rng"]
    return candidate.replace(
        key=original.key,
        env_state=candidate.env_state.replace(info=info),
    )


def test_resume_rekey_is_deterministic_disjoint_and_preserves_non_rng_state() -> None:
    from src.algorithms.shac.resume_randomness import (
        rekey_resumed_train_state,
        resume_randomness_audit,
    )

    original = _state()
    first = rekey_resumed_train_state(original, seed=1)
    repeated = rekey_resumed_train_state(original, seed=1)
    other = rekey_resumed_train_state(original, seed=2)

    _assert_tree_exact(first, repeated)
    assert not np.array_equal(np.asarray(first.key), np.asarray(original.key))
    assert not np.array_equal(
        np.asarray(first.env_state.info["rng"]),
        np.asarray(original.env_state.info["rng"]),
    )
    assert not np.array_equal(np.asarray(first.key), np.asarray(other.key))
    assert not np.array_equal(
        np.asarray(first.env_state.info["rng"]),
        np.asarray(other.env_state.info["rng"]),
    )
    assert not any(
        np.array_equal(np.asarray(first.key), np.asarray(environment_key))
        for environment_key in np.asarray(first.env_state.info["rng"])
    )
    _assert_tree_exact(_restore_authorized_rngs(first, original), original)

    audit = resume_randomness_audit(original, first, seed=1)
    assert audit["resume_random_seed"] == 1
    assert audit["trainer_key_changed"] is True
    assert audit["environment_keys_changed"] is True
    assert audit["non_rng_state_exact"] is True
    assert audit["valid"] is True
    assert len(audit["old_trainer_key_sha256"]) == 64
    assert len(audit["new_environment_keys_sha256"]) == 64
    assert "key" not in audit


def test_resume_randomness_setting_is_default_off_and_requires_resume() -> None:
    from src.algorithms.shac.resume_randomness import (
        apply_resume_randomness_setting,
    )

    original = _state()
    unchanged, audit = apply_resume_randomness_setting(original, seed=None)
    assert unchanged is original
    assert audit is None

    with pytest.raises(ValueError, match="requires a resumed checkpoint"):
        apply_resume_randomness_setting(None, seed=1)

    changed, audit = apply_resume_randomness_setting(original, seed=1)
    assert audit is not None
    assert audit["valid"] is True
    assert changed is not original


def test_train_exposes_and_persists_opt_in_resume_randomness() -> None:
    import inspect

    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters
    assert parameters["resume_random_seed"].default is None
    source = inspect.getsource(train)
    assert "apply_resume_randomness_setting(" in source
    assert "persist_resume_randomness_audit(" in source
    assert '"resume_random_seed": resume_random_seed' in source


@pytest.mark.parametrize(
    ("seed", "message"),
    [
        (True, "non-negative integer"),
        (-1, "non-negative integer"),
        (1.5, "non-negative integer"),
        (2**32, "32-bit"),
    ],
)
def test_resume_rekey_rejects_invalid_seed(seed: object, message: str) -> None:
    from src.algorithms.shac.resume_randomness import rekey_resumed_train_state

    with pytest.raises(ValueError, match=message):
        rekey_resumed_train_state(_state(), seed=seed)


def test_resume_rekey_rejects_missing_or_malformed_environment_rng() -> None:
    from src.algorithms.shac.resume_randomness import rekey_resumed_train_state

    original = _state()
    missing = original.replace(
        env_state=original.env_state.replace(info={"phase": jp.arange(4)})
    )
    with pytest.raises(ValueError, match=r'info\["rng"\]'):
        rekey_resumed_train_state(missing, seed=1)

    for malformed in (jp.zeros((4,), dtype=jp.uint32), jp.zeros((4, 3), dtype=jp.uint32)):
        with pytest.raises(ValueError, match=r"shape \(num_envs, 2\)"):
            rekey_resumed_train_state(_state(rng=malformed), seed=1)


def test_resume_randomness_audit_persists_atomically(tmp_path: Path) -> None:
    from src.algorithms.shac.resume_randomness import (
        persist_resume_randomness_audit,
        rekey_resumed_train_state,
        resume_randomness_audit,
    )

    original = _state()
    candidate = rekey_resumed_train_state(original, seed=1)
    payload = resume_randomness_audit(original, candidate, seed=1)
    output = persist_resume_randomness_audit(tmp_path, payload)

    assert output == tmp_path / "resume_randomness_audit.json"
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / ".resume_randomness_audit.json.tmp").exists()
