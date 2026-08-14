from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_builder_changes_only_the_fresh_action_contract() -> None:
    from tools.run_g1_fresh_fixed_noise_training import (
        build_fresh_fixed_noise_kwargs,
    )
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        build_fresh_ppo_action_contract_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    baseline = build_fresh_fixed_noise_kwargs(
        "g1-4x5",
        reference,
        0,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    treatment = build_fresh_ppo_action_contract_kwargs(
        "g1-4x5", reference, 0
    )

    changed = {
        key
        for key in set(baseline) | set(treatment)
        if baseline.get(key) != treatment.get(key)
    }
    assert changed == {"env_variant", "reference_residual_scale"}
    assert treatment["env_variant"] == "g1_tracking_rmr_50hz_action_parity"
    assert treatment["reference_residual_scale"] == 1.0
    assert treatment["resume_from"] is None
    assert treatment["actor_zero_output"] is True
    assert treatment["action_noise_std_start"] == 0.2
    assert treatment["action_noise_std_end"] == 0.2
    assert treatment["actor_lr"] == 1e-3
    assert treatment["actor_bootstrap_scale"] == 0.0
    assert treatment["unroll_length"] == 12
    assert treatment["num_envs"] == 256
    assert treatment["gradient_accumulation_steps"] == 2
    assert treatment["actor_cagrad"] is True
    assert treatment["actor_per_env_grad_clip"] == 1.0


def test_action_parity_environment_exposes_unbounded_training_contract() -> None:
    from src.envs.g1_tracking.environment import (
        DEFAULT_CONTROLLER_PATH,
        DEFAULT_MODEL_PATH,
    )
    from src.envs.go2.environment import Go2Env
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        build_fresh_ppo_action_contract_kwargs,
    )

    kwargs = build_fresh_ppo_action_contract_kwargs(
        "g1-4x5",
        Path(
            "/home/ubuntu/projects/diffsim2real/outputs/"
            "rmr_motion_walk_win137_212_named.npz"
        ),
        0,
    )
    env = Go2Env(
        variant=kwargs["env_variant"],
        xml_path=DEFAULT_MODEL_PATH,
        controller_path=DEFAULT_CONTROLLER_PATH,
        reference_path=kwargs["reference_path"],
        reference_residual_control=kwargs["reference_residual_control"],
        reference_residual_scale=kwargs["reference_residual_scale"],
        reference_stride=kwargs["reference_stride"],
        actor_reference_lookahead_steps=kwargs[
            "actor_reference_lookahead_steps"
        ],
        actor_reference_preview_mode=kwargs["actor_reference_preview_mode"],
        solver_iterations=4,
        solver_ls_iterations=5,
    )

    assert env.squash_actor_mean is False
    assert env.clip_sampled_actor_actions is False
    assert env.reference_residual_scale == 1.0


def test_training_validator_requires_persisted_unbounded_contract(
    tmp_path: Path,
) -> None:
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        validate_training_artifacts,
    )

    hparams = {
        "env_variant": "g1_tracking_rmr_50hz_source_step",
        "reference_residual_scale": 0.5,
        "squash_actor_mean": True,
        "clip_sampled_actor_actions": True,
    }
    (tmp_path / "hparams.json").write_text(json.dumps(hparams))

    with pytest.raises(ValueError, match="action contract"):
        validate_training_artifacts(tmp_path)


def test_training_validator_reads_actor_update_norms_from_diag_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.run_g1_fresh_ppo_action_contract_walk import (
        TOTAL_STEPS,
        build_fresh_ppo_action_contract_kwargs,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    reference = Path(
        "/home/ubuntu/projects/diffsim2real/outputs/"
        "rmr_motion_walk_win137_212_named.npz"
    )
    hparams = build_fresh_ppo_action_contract_kwargs("g1-4x5", reference, 0)
    hparams = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in hparams.items()
        if key not in {"checkpoint_interval", "diagnose", "resume_from"}
    }
    hparams.update(
        squash_actor_mean=False,
        clip_sampled_actor_actions=False,
    )
    (tmp_path / "hparams.json").write_text(json.dumps(hparams))

    steps = expected_checkpoint_steps()
    for step in steps:
        (tmp_path / f"checkpoint_step_{step:06d}.pkl").write_bytes(b"state")
    (tmp_path / "checkpoint_latest.pkl").write_bytes(b"state")
    (tmp_path / "policy_final.pkl").write_bytes(b"state")
    monkeypatch.setattr(
        "tools.run_g1_fresh_ppo_action_contract_walk._validate_checkpoint",
        lambda path, step: "final" if step == TOTAL_STEPS else f"sha-{step}",
    )

    rows = []
    for step in steps:
        rows.append(
            {
                "step": step,
                "action_noise_current": 0.2,
                "actor_bootstrap_scale_current": 0.0,
                "actor_cagrad_valid": True,
                "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
                "actor_cagrad_bin_gradient_norms": [0.2] * 5,
                "actor_cagrad_bin_losses": [-0.1] * 5,
                "actor_cagrad_weights": [0.2] * 5,
                "actor_cagrad_gram_matrix": np.eye(5).tolist(),
                "actor_cagrad_cosine_matrix": np.eye(5).tolist(),
                "actor_cagrad_objective": 0.1,
                "actor_cagrad_dual_gap": 0.0,
                "actor_cagrad_uniform_combined_cosine": 0.5,
                "actor_cagrad_combined_norm": 0.2,
            }
        )
    (tmp_path / "checkpoint_phase_metrics.json").write_text(json.dumps(rows))
    (tmp_path / "diag_log.json").write_text(
        json.dumps([{"actor_grad": 0.2, "actor_update_norm": 0.1}])
    )

    result = validate_training_artifacts(tmp_path)

    assert result["valid"] is True
