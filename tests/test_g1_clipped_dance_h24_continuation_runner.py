from pathlib import Path


def test_h24_continuation_changes_only_horizon_and_mechanical_budget(
    tmp_path: Path,
) -> None:
    from tools.run_g1_clipped_dance_continuation import (
        build_clipped_dance_continuation_kwargs,
    )
    from tools.run_g1_clipped_dance_h24_continuation import (
        CONTINUATION_END_STEP,
        build_clipped_dance_h24_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e012-selected.pkl"
    parent = build_clipped_dance_continuation_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_clipped_dance_h24_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }

    assert delta == {
        "unroll_length": 24,
        "total_steps": CONTINUATION_END_STEP,
        "checkpoint_interval": 98_304,
    }
    assert candidate["num_envs"] == 256
    assert candidate["gradient_accumulation_steps"] == 2
    assert candidate["actor_per_env_grad_clip"] == 1.0
    assert expected_checkpoint_steps() == (1_867_776, 1_966_080)
