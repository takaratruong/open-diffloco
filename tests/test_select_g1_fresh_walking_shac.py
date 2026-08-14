from __future__ import annotations

import pytest


PHASES = (0, 25, 50, 75, 100)
SURVIVAL = (124, 99, 74, 49, 24)


def _results(*, reward: float, body_position: float, secondary: float = 1.0):
    return [
        {
            "phase": phase,
            "steps": steps,
            "terminal": False,
            "mean_reward": reward,
            "mean_body_position_error": body_position,
            "mean_body_orientation_error": secondary,
            "mean_body_linear_velocity_error": secondary,
            "mean_body_angular_velocity_error": secondary,
            "mean_anchor_position_error": secondary,
            "mean_anchor_orientation_error": secondary,
        }
        for phase, steps in zip(PHASES, SURVIVAL, strict=True)
    ]


def _payload(
    *,
    step: int = 98_304,
    reward_delta: float,
    body_position_delta: float,
    secondary_delta: float = 0.0,
):
    return {
        "protocol": "g1-rmr-walk-replay-free-five-phase-v1",
        "source_controller": "zero_residual",
        "reference_transitions": 124,
        "checkpoint_path": f"/tmp/checkpoint_step_{step:06d}.pkl",
        "source": {
            "results": _results(reward=1.0, body_position=1.0),
            "summary": {"completed_suffix": [True] * 5},
        },
        "candidate": {
            "results": _results(
                reward=1.0 + reward_delta,
                body_position=1.0 + body_position_delta,
                secondary=1.0 + secondary_delta,
            ),
            "summary": {"completed_suffix": [True] * 5},
            "action_root_mean_square": 0.1,
        },
    }


def _training_row(step: int = 98_304):
    return {
        "step": step,
        "actor_cagrad_valid": True,
        "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
        "actor_cagrad_bin_gradient_norms": [0.5, 0.4, 0.3, 0.2, 0.1],
        "actor_grad": 0.2,
        "actor_update_norm": 0.1,
    }


def test_selector_requires_learning_beyond_zero_residual() -> None:
    from tools.select_g1_fresh_walking_shac import (
        select_fresh_walking_checkpoint,
    )

    matching = _payload(reward_delta=0.0, body_position_delta=0.0)
    report = select_fresh_walking_checkpoint([matching], [_training_row()])

    assert report["outcome"] == "stable-partial-learning"
    assert report["selected_checkpoint"] is None


def test_selector_selects_earliest_checkpoint_after_metric_ties() -> None:
    from tools.select_g1_fresh_walking_shac import (
        select_fresh_walking_checkpoint,
    )

    early = _payload(
        step=98_304, reward_delta=0.01, body_position_delta=-0.01
    )
    late = _payload(
        step=196_608, reward_delta=0.01, body_position_delta=-0.01
    )
    report = select_fresh_walking_checkpoint(
        [late, early], [_training_row(98_304), _training_row(196_608)]
    )

    assert report["outcome"] == "fresh-shac-learns-walk"
    assert report["selected_checkpoint"]["step"] == 98_304


def test_selector_rejects_secondary_regression_above_five_percent() -> None:
    from tools.select_g1_fresh_walking_shac import (
        select_fresh_walking_checkpoint,
    )

    regressed = _payload(
        reward_delta=0.01,
        body_position_delta=-0.01,
        secondary_delta=0.051,
    )
    report = select_fresh_walking_checkpoint([regressed], [_training_row()])

    assert report["outcome"] == "stable-partial-learning"
    assert report["records"][0]["eligible"] is False


def test_selector_rejects_nonfinite_cagrad_counts() -> None:
    from tools.select_g1_fresh_walking_shac import (
        select_fresh_walking_checkpoint,
    )

    training = _training_row()
    training["actor_cagrad_bin_counts"][0] = float("nan")

    with pytest.raises(ValueError, match="CAGrad telemetry is invalid"):
        select_fresh_walking_checkpoint(
            [_payload(reward_delta=0.01, body_position_delta=-0.01)],
            [training],
        )
