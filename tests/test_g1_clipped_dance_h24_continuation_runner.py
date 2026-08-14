from pathlib import Path
import json


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


def test_h24_validator_uses_exact_checkpoint_grid_not_absent_hparam(
    tmp_path: Path,
) -> None:
    from tools.run_g1_clipped_dance_h24_continuation import (
        CONTINUATION_END_STEP,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    hparams = {
        "unroll_length": 24,
        "total_steps": CONTINUATION_END_STEP,
        "actor_per_env_grad_clip": 1.0,
        "num_envs": 256,
        "gradient_accumulation_steps": 2,
    }
    (tmp_path / "hparams.json").write_text(json.dumps(hparams))
    rows = []
    for step in expected_checkpoint_steps():
        (tmp_path / f"checkpoint_step_{step}.pkl").write_bytes(b"checkpoint")
        rows.append(
            {
                "step": step,
                "actor_cagrad_valid": True,
                "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
                "actor_cagrad_bin_gradient_norms": [0.1] * 5,
                "actor_preview_frozen_parameter_drift_max_abs": 0.0,
                "actor_preview_frozen_moment_drift_max_abs": 0.0,
                "actor_preview_normalizer_drift_max_abs": 0.0,
                "torso_wrench_assistance_scale_current": 0.0,
                "torso_wrench_assistance_active_fraction": 0.0,
                "torso_wrench_assistance_max_force": 0.0,
                "torso_wrench_assistance_max_torque": 0.0,
            }
        )
    (tmp_path / "checkpoint_phase_metrics.json").write_text(json.dumps(rows))

    assert validate_training_artifacts(tmp_path)["valid"] is True
