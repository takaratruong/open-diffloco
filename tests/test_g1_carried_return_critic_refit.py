from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

from tools.refit_g1_carried_return_critic import (
    calibration_candidate_key,
    noncritic_state_drift,
    phase_splits,
    replace_critic_state,
    trajectory_rows,
)
from tools.consolidate_g1_carried_return_critic import (
    CONFIRMATION_PHASES,
    load_frozen_dataset,
)


def test_phase_splits_are_disjoint_and_immutable():
    splits = phase_splits()
    assert splits["fit"] == tuple(range(10, 400, 20))
    assert splits["validation"] == (20, 120, 220, 320, 380)
    assert splits["test"] == (0, 100, 200, 300, 400)
    assert len(set().union(*map(set, splits.values()))) == 30


def test_trajectory_rows_stop_at_first_natural_terminal():
    observations = np.arange(24, dtype=np.float32).reshape(6, 4)
    rewards = np.ones(6)
    rows = trajectory_rows(
        observations,
        rewards,
        terminals=[False, False, True, False, False, False],
        gamma=0.5,
    )
    assert rows["critic_observations"].shape == (3, 4)
    np.testing.assert_allclose(rows["returns"], [1.75, 1.5, 1.0])
    with pytest.raises(ValueError):
        trajectory_rows(
            observations, rewards, terminals=[False] * 6, gamma=0.5
        )


def test_candidate_key_prefers_gate_then_nrmse_rank_and_earlier_step():
    failing = calibration_candidate_key(
        {"rank_correlation": 0.9, "nrmse": 0.3}, step=20
    )
    passing = calibration_candidate_key(
        {"rank_correlation": 0.8, "nrmse": 0.25}, step=40
    )
    better = calibration_candidate_key(
        {"rank_correlation": 0.85, "nrmse": 0.2}, step=60
    )
    earlier = calibration_candidate_key(
        {"rank_correlation": 0.85, "nrmse": 0.2}, step=40
    )
    assert passing > failing
    assert better > passing
    assert earlier > better


@dataclass(frozen=True)
class FakeState:
    actor_params: object
    actor_opt: object
    normalizer: object
    env_state: object
    critic_params: object
    target_critic_params: object
    critic_opt: object


def test_replace_critic_state_preserves_every_noncritic_leaf_exactly():
    original = FakeState(
        actor_params={"w": jnp.array([1.0])},
        actor_opt={"m": jnp.array([2.0])},
        normalizer={"mean": jnp.array([3.0])},
        env_state={"q": jnp.array([4.0])},
        critic_params={"w": jnp.array([5.0])},
        target_critic_params={"w": jnp.array([6.0])},
        critic_opt={"m": jnp.array([7.0])},
    )
    candidate = replace_critic_state(
        original,
        critic_params={"w": jnp.array([8.0])},
        critic_opt={"m": jnp.array([9.0])},
        replace_fn=replace,
    )
    assert noncritic_state_drift(original, candidate) == {
        "actor_params": 0.0,
        "actor_opt": 0.0,
        "normalizer": 0.0,
        "env_state": 0.0,
        "valid": True,
    }
    np.testing.assert_array_equal(candidate.critic_params["w"], [8.0])
    np.testing.assert_array_equal(candidate.target_critic_params["w"], [8.0])
    np.testing.assert_array_equal(candidate.critic_opt["m"], [9.0])


def test_confirmation_phases_are_unseen_and_dataset_loader_is_exact(tmp_path):
    assert CONFIRMATION_PHASES == (5, 105, 205, 305, 405)
    assert set(CONFIRMATION_PHASES).isdisjoint(
        set().union(*map(set, phase_splits().values()))
    )
    payload = {}
    expected_phases = tuple(
        (*phase_splits()["fit"], *phase_splits()["validation"], *phase_splits()["test"])
    )
    for phase in expected_phases:
        payload[f"phase_{phase}_critic_observations"] = np.full((2, 3), phase)
        payload[f"phase_{phase}_rewards"] = np.ones(2)
        payload[f"phase_{phase}_returns"] = np.array([1.5, 1.0])
    path = tmp_path / "dataset.npz"
    np.savez_compressed(path, **payload)
    loaded = load_frozen_dataset(path)
    assert tuple(loaded) == expected_phases
    np.testing.assert_array_equal(
        loaded[110]["critic_observations"], np.full((2, 3), 110)
    )
