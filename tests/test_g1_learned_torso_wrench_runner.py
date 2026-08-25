from __future__ import annotations


def test_runner_changes_only_reference_reset_and_learned_wrench_treatment():
    from tools.run_g1_learned_torso_wrench import (
        END_STEP,
        START_STEP,
        build_learned_wrench_kwargs,
        expected_checkpoint_steps,
    )

    kwargs = build_learned_wrench_kwargs(
        "g1-4x5",
        "/tmp/prefix.npz",
        0,
        "/tmp/e026.pkl",
    )
    assert kwargs["resume_from"] == "/tmp/e026.pkl"
    assert kwargs["reference_path"] == "/tmp/prefix.npz"
    assert kwargs["allow_resume_reference_path_change"] is True
    assert kwargs["carried_reset_bank_path"] is None
    assert kwargs["carried_reset_probability"] == 0.0
    assert kwargs["allow_resume_carried_reset_change"] is True
    assert kwargs["actor_policy_anchor_weight"] == 0.0
    assert kwargs["allow_resume_actor_policy_anchor_source_change"] is True
    assert kwargs["actor_cagrad"] is False
    assert kwargs["allow_resume_actor_cagrad_change"] is True
    assert kwargs["actor_learned_torso_wrench"] is True
    assert kwargs["actor_learned_torso_wrench_hidden"] == 256
    assert kwargs["actor_learned_torso_wrench_scale"] == 1.0
    assert kwargs["actor_learned_torso_wrench_penalty"] == 0.0
    assert kwargs["allow_resume_actor_learned_torso_wrench_start"] is True
    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == expected_checkpoint_steps()
    assert expected_checkpoint_steps()[0] == START_STEP + 8 * 512 * 24
    assert expected_checkpoint_steps()[-1] == END_STEP


def test_training_validation_requires_finite_wrench_only_updates(tmp_path):
    import json
    import pickle
    from types import SimpleNamespace

    import jax.numpy as jp

    from src.algorithms.shac.learned_torso_wrench import (
        FrozenControllerWrenchParams,
    )
    from tools.run_g1_learned_torso_wrench import validate_training_artifacts

    run = tmp_path / "run"
    run.mkdir()
    steps = (100,)
    state = SimpleNamespace(
        step=jp.asarray(100),
        actor_params=FrozenControllerWrenchParams(
            controller={"x": jp.asarray([1.0])},
            wrench={"x": jp.asarray([0.1])},
        ),
    )
    for name in ("checkpoint_step_100.pkl", "checkpoint_latest.pkl", "policy_final.pkl"):
        with (run / name).open("wb") as stream:
            pickle.dump(state, stream)
    (run / "hparams.json").write_text(
        json.dumps(
            {
                "actor_learned_torso_wrench": True,
                "actor_learned_torso_wrench_scale": 1.0,
                "actor_learned_torso_wrench_penalty": 0.0,
            }
        )
    )
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(
            [
                {
                    "step": 100,
                    "actor_preview_frozen_parameter_drift_max_abs": 0.0,
                    "actor_preview_frozen_moment_drift_max_abs": 0.0,
                    "actor_preview_normalizer_drift_max_abs": 0.0,
                    "actor_preview_gradient_norm": 1.0,
                    "actor_preview_update_norm": 0.1,
                    "actor_preview_valid": True,
                    "learned_torso_wrench_rms_force": 10.0,
                    "learned_torso_wrench_rms_torque": 1.0,
                    "learned_torso_wrench_max_force": 20.0,
                    "learned_torso_wrench_max_torque": 2.0,
                    "learned_torso_wrench_saturation_fraction": 0.1,
                    "learned_torso_wrench_valid": True,
                }
            ]
        )
    )

    report = validate_training_artifacts(run, expected_steps=steps)
    assert report["valid"] is True
    assert report["checkpoint_steps"] == [100]
