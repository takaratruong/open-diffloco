import json
from pathlib import Path


def test_fixed_020_runner_uses_exact_scalar_for_entire_continuation() -> None:
    from tools.run_g1_fixed_020_action_noise_continuation import (
        FIXED_ACTION_NOISE_STD,
        build_fixed_action_noise_kwargs,
    )

    kwargs = build_fixed_action_noise_kwargs(
        "g1-4x5", Path("/tmp/reference.npz"), 0, Path("/tmp/parent.pkl")
    )

    assert FIXED_ACTION_NOISE_STD == 0.2
    assert kwargs["action_noise_std_start"] == 0.2
    assert kwargs["action_noise_std_end"] == 0.2
    assert kwargs["allow_resume_action_noise_change"] is True
    assert kwargs["total_steps"] == 2_064_384
    assert kwargs["action_noise_schedule_steps"] == 2_064_384
    assert kwargs["actor_observation_noise"] is False
    assert kwargs["reference_reset_noise_scale"] == 0.0
    assert kwargs["domain_randomization"] is False
    assert kwargs["push_velocity_range"] == (0.0, 0.0)


def test_fixed_020_staged_resume_disables_every_other_training_perturbation(
    tmp_path: Path,
) -> None:
    from tools.run_g1_fixed_020_action_noise_continuation import (
        build_fixed_resume_hparams,
    )

    source = {
        "actor_observation_noise": True,
        "reference_reset_noise_scale": 1.0,
        "reference_root_reset_noise_multiplier": 2.0,
        "reference_root_reset_noise_probability": 0.5,
        "domain_randomization": True,
        "kp_range": [25.0, 45.0],
        "kd_range": [0.3, 0.7],
        "friction_range": [0.5, 2.0],
        "mass_range": [0.85, 1.15],
        "com_offset_range": [0.05, 0.05, 0.04],
        "push_velocity_range": [-1.0, 1.0],
        "terrain_bump_std": 0.4,
        "carried_reset_probability": 0.0,
    }

    staged = build_fixed_resume_hparams(source)

    assert staged["actor_observation_noise"] is False
    assert staged["reference_reset_noise_scale"] == 0.0
    assert staged["reference_root_reset_noise_probability"] == 0.0
    assert staged["domain_randomization"] is False
    assert staged["kp_range"] == [35.0, 35.0]
    assert staged["kd_range"] == [0.5, 0.5]
    assert staged["friction_range"] == [1.0, 1.0]
    assert staged["mass_range"] == [1.0, 1.0]
    assert staged["com_offset_range"] == [0.0, 0.0, 0.0]
    assert staged["push_velocity_range"] == [0.0, 0.0]
    assert staged["terrain_bump_std"] == 0.0
    assert source["actor_observation_noise"] is True
    json.dumps(staged)
