from pathlib import Path
import json


def test_behavioral_trust_proposal_changes_only_one_update_budget(tmp_path: Path):
    from tools.run_g1_clipped_dance_h24_continuation import (
        build_clipped_dance_h24_kwargs,
    )
    from tools.run_g1_behavioral_trust_proposal import (
        PROPOSAL_END_STEP,
        build_behavioral_trust_proposal_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "dance.npz"
    checkpoint = tmp_path / "e013-selected.pkl"
    parent = build_clipped_dance_h24_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    candidate = build_behavioral_trust_proposal_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }

    assert delta == {
        "total_steps": PROPOSAL_END_STEP,
        "checkpoint_interval": 12_288,
    }
    assert candidate["unroll_length"] == 24
    assert candidate["actor_per_env_grad_clip"] == 1.0
    assert candidate["num_envs"] == 256
    assert candidate["gradient_accumulation_steps"] == 2
    assert expected_checkpoint_steps() == (PROPOSAL_END_STEP,)


def test_behavioral_trust_proposal_validator_requires_exact_single_checkpoint(
    tmp_path: Path,
) -> None:
    from tools.run_g1_behavioral_trust_proposal import (
        PROPOSAL_END_STEP,
        validate_training_artifacts,
    )

    (tmp_path / "hparams.json").write_text(
        json.dumps(
            {
                "unroll_length": 24,
                "total_steps": PROPOSAL_END_STEP,
                "actor_per_env_grad_clip": 1.0,
                "num_envs": 256,
                "gradient_accumulation_steps": 2,
            }
        )
    )
    (tmp_path / f"checkpoint_step_{PROPOSAL_END_STEP}.pkl").write_bytes(b"state")
    (tmp_path / "checkpoint_phase_metrics.json").write_text(
        json.dumps(
            [
                {
                    "step": PROPOSAL_END_STEP,
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
            ]
        )
    )

    assert validate_training_artifacts(tmp_path)["valid"] is True
