import json
from pathlib import Path

import numpy as np


def test_fresh_fixed_noise_builder_is_exact_and_has_dense_checkpoints() -> None:
    from tools.run_g1_fresh_fixed_noise_training import (
        CHECKPOINT_INTERVAL,
        TOTAL_STEPS,
        build_fresh_fixed_noise_kwargs,
    )

    kwargs = build_fresh_fixed_noise_kwargs(
        "g1-4x5", Path("/tmp/reference.npz"), seed=3
    )

    assert TOTAL_STEPS == 786_432
    assert CHECKPOINT_INTERVAL == 98_304
    assert kwargs["total_steps"] == TOTAL_STEPS
    assert kwargs["checkpoint_interval"] == CHECKPOINT_INTERVAL
    assert kwargs["env_variant"] == "g1_tracking_rmr_50hz_source_step"
    assert kwargs["resume_from"] is None
    assert kwargs["actor_zero_output"] is True
    assert kwargs["actor_hidden"] == (512, 256, 128)
    assert kwargs["actor_layer_norm"] is True
    assert kwargs["action_noise_std_start"] == 0.2
    assert kwargs["action_noise_std_end"] == 0.2
    assert kwargs["action_noise_schedule_steps"] == TOTAL_STEPS
    assert kwargs["actor_observation_noise"] is False
    assert kwargs["reference_reset_noise_scale"] == 0.0
    assert kwargs["reference_root_reset_noise_probability"] == 0.0
    assert kwargs["carried_reset_probability"] == 0.0
    assert kwargs["domain_randomization"] is False
    assert kwargs["friction_range"] == (1.0, 1.0)
    assert kwargs["mass_range"] == (1.0, 1.0)
    assert kwargs["kp_range"] == (35.0, 35.0)
    assert kwargs["kd_range"] == (0.5, 0.5)
    assert kwargs["com_offset_range"] == (0.0, 0.0, 0.0)
    assert kwargs["push_velocity_range"] == (0.0, 0.0)
    assert kwargs["terrain_bump_std"] == 0.0
    assert kwargs["torso_wrench_assistance"] is False
    assert kwargs["reference_residual_control"] is True
    assert kwargs["reference_residual_scale"] == 0.5
    assert kwargs["unroll_length"] == 12
    assert kwargs["num_envs"] == 256
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["actor_cagrad"] is True
    assert kwargs["actor_reference_lookahead_steps"] == (4, 8, 12)
    assert kwargs["actor_reference_preview_mode"] == "delta"
    assert kwargs["actor_bootstrap_scale"] == 0.0


def test_episode_action_diagnostics_expose_saturation() -> None:
    from tools.log_g1_training_episodes import action_diagnostics

    mean = np.vstack((np.full(29, 2.0), np.zeros(29)))
    epsilon = np.vstack((np.full(29, -1.0), np.ones(29)))
    noisy = mean + 0.2 * epsilon
    effective = np.clip(noisy, -1.0, 1.0)

    result = action_diagnostics(
        action_mean=mean,
        epsilon=epsilon,
        action_std=np.array(0.2),
        noisy_action=noisy,
        effective_action=effective,
    )

    assert result["samples"] == 58
    assert result["noise_rms"] == 0.2
    assert result["mean_action_outside_fraction"] == 0.5
    assert result["noisy_action_outside_fraction"] == 0.5
    assert result["effective_action_saturation_fraction"] == 0.5
    assert result["effective_action_max_abs"] == 1.0


def test_episode_logger_discovers_exact_registered_checkpoints(tmp_path: Path) -> None:
    from tools.log_g1_training_episodes import discover_checkpoints

    for step in (98_304, 196_608, 294_912):
        (tmp_path / f"checkpoint_step_{step:06d}.pkl").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_latest.pkl").write_bytes(b"latest")

    checkpoints = discover_checkpoints(
        tmp_path, checkpoint_interval=98_304, total_steps=294_912
    )

    assert [path.name for path in checkpoints] == [
        "checkpoint_step_098304.pkl",
        "checkpoint_step_196608.pkl",
        "checkpoint_step_294912.pkl",
    ]


def test_episode_manifest_requires_clean_and_noisy_evidence(tmp_path: Path) -> None:
    from tools.log_g1_training_episodes import build_episode_manifest

    checkpoint = tmp_path / "checkpoint_step_098304.pkl"
    checkpoint.write_bytes(b"checkpoint")
    noisy = tmp_path / "checkpoint_step_098304" / "noisy"
    clean = tmp_path / "checkpoint_step_098304" / "clean"
    noisy.mkdir(parents=True)
    clean.mkdir(parents=True)
    (noisy / "training_rollout.mp4").write_bytes(b"video")
    (noisy / "training_slice_h12.mp4").write_bytes(b"slice")
    (noisy / "contact_sheet.png").write_bytes(b"png")
    (clean / "evaluation.mp4").write_bytes(b"clean-video")
    (clean / "contact_sheet.png").write_bytes(b"clean-png")
    for directory, steps in ((noisy, 12), (clean, 20)):
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "steps": steps,
                    "checkpoint_sha256": __import__(
                        "hashlib"
                    ).sha256(b"checkpoint").hexdigest(),
                    "training_distribution_rollout": directory == noisy,
                    "training_observation_noise": False,
                    "training_exact_reset_phase": 0 if directory == noisy else None,
                }
            )
        )
    np.savez_compressed(
        noisy / "training_action_noise.npz",
        action_mean=np.zeros((2, 29)),
        epsilon=np.ones((2, 29)),
        action_std=np.array(0.2),
        noisy_action=np.full((2, 29), 0.2),
        effective_action=np.full((2, 29), 0.2),
    )

    manifest = build_episode_manifest(
        checkpoints=[checkpoint], output_root=tmp_path
    )

    assert manifest["valid"] is True
    assert manifest["episodes"][0]["step"] == 98_304
    assert manifest["episodes"][0]["noisy"]["steps"] == 12
    assert manifest["episodes"][0]["clean"]["steps"] == 20
    assert manifest["episodes"][0]["actions"]["noise_rms"] == 0.2
