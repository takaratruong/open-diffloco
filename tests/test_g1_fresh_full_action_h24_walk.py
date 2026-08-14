from pathlib import Path

import json

import pytest


def test_h24_builder_changes_only_horizon_and_cadence() -> None:
    from tools.run_g1_fresh_full_action_h24_walk import (
        build_fresh_full_action_h24_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        build_fresh_ppo_action_contract_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    baseline = build_fresh_ppo_action_contract_kwargs(
        "g1-4x5", reference, 0
    )
    treatment = build_fresh_full_action_h24_kwargs(
        "g1-4x5", reference, 0
    )

    changed = {
        key
        for key in set(baseline) | set(treatment)
        if baseline.get(key) != treatment.get(key)
    }
    assert changed == {
        "unroll_length",
        "total_steps",
        "checkpoint_interval",
    }
    assert treatment["unroll_length"] == 24
    assert treatment["total_steps"] == 1_572_864
    assert treatment["checkpoint_interval"] == 196_608
    assert treatment["num_envs"] == 256
    assert treatment["gradient_accumulation_steps"] == 2
    assert treatment["total_steps"] // (256 * 2 * 24) == 128
    assert expected_checkpoint_steps() == tuple(
        range(196_608, 1_572_864 + 1, 196_608)
    )


def test_h24_validator_rejects_an_h12_persisted_horizon(
    tmp_path: Path,
) -> None:
    from tools.run_g1_fresh_full_action_h24_walk import (
        TOTAL_STEPS,
        build_fresh_full_action_h24_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )

    reference = Path(
        "/home/ubuntu/projects/diffsim2real/outputs/"
        "rmr_motion_walk_win137_212_named.npz"
    )
    expected = build_fresh_full_action_h24_kwargs(
        "g1-4x5", reference, 0
    )
    hparams = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in expected.items()
        if key not in {"checkpoint_interval", "diagnose", "resume_from"}
    }
    hparams.update(
        squash_actor_mean=False,
        clip_sampled_actor_actions=False,
        unroll_length=12,
    )
    (tmp_path / "hparams.json").write_text(json.dumps(hparams))

    with pytest.raises(ValueError, match="unroll_length"):
        validate_training_artifacts(
            tmp_path,
            expected_kwargs=expected,
            expected_steps=expected_checkpoint_steps(),
            total_steps=TOTAL_STEPS,
            protocol="g1-fresh-full-action-h24-walk-training-v1",
        )
