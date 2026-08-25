from __future__ import annotations

import inspect


def test_train_exposes_fail_closed_learned_wrench_configuration():
    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters
    assert parameters["actor_learned_torso_wrench"].default is False
    assert parameters["actor_learned_torso_wrench_hidden"].default == 256
    assert parameters["actor_learned_torso_wrench_scale"].default == 1.0
    assert parameters["actor_learned_torso_wrench_scale_end"].default is None
    assert parameters["actor_learned_torso_wrench_scale_start_step"].default == 0
    assert parameters["actor_learned_torso_wrench_scale_end_step"].default == 1
    assert parameters["actor_learned_torso_wrench_condition_on_scale"].default is False
    assert parameters["actor_learned_torso_wrench_train_controller"].default is False
    assert parameters["actor_learned_torso_wrench_penalty"].default == 0.0
    assert (
        parameters["allow_resume_actor_learned_torso_wrench_start"].default
        is False
    )
    assert (
        parameters["allow_resume_actor_learned_torso_wrench_change"].default
        is False
    )


def test_learned_wrench_is_written_after_policy_and_before_environment_step():
    from src.algorithms.shac import algorithm

    source = inspect.getsource(algorithm.train)
    head = source.index("learned_torso_wrench_normalized = apply_learned_torso_wrench")
    write = source.index(
        "world_wrench=learned_torso_wrench", head
    )
    step = source.index("candidate_unreplayed_state = env.step", write)
    assert head < write < step
    assert "analytic and learned torso wrench treatments are mutually exclusive" in source
    assert "and not actor_learned_torso_wrench" in source
    residual_validation = source.index(
        "validate_residual_preview_adapter_configuration("
    )
    learned_override = source.index(
        "actor_cagrad=(actor_cagrad or actor_learned_torso_wrench)",
        residual_validation,
    )
    assert residual_validation < learned_override
    assert source.count(
        "actor_cagrad=(actor_cagrad or actor_learned_torso_wrench)"
    ) == 2


def test_learned_wrench_checkpoint_metadata_and_telemetry_are_persisted():
    from src.algorithms.shac import algorithm

    source = inspect.getsource(algorithm.train)
    for fragment in (
        '"actor_learned_torso_wrench": actor_learned_torso_wrench',
        '"actor_learned_torso_wrench_scale": (',
        '"actor_learned_torso_wrench_scale_end": (',
        '"actor_learned_torso_wrench_condition_on_scale": (',
        '"actor_learned_torso_wrench_train_controller": (',
        '"learned_torso_wrench_rms_force": float(',
        '"learned_torso_wrench_rms_torque": float(',
        '"learned_torso_wrench_saturation_fraction": float(',
        'metrics["learned_torso_wrench_valid"]',
    ):
        assert fragment in source


def test_learned_wrench_anneal_is_step_scheduled_and_scale_conditioned():
    from src.algorithms.shac import algorithm

    source = inspect.getsource(algorithm.train)
    schedule = source.index("current_learned_wrench_scale = (")
    gradient = source.index("actor_grad_fn = jax.value_and_grad", schedule)
    assert schedule < gradient
    assert "current_learned_wrench_scale," in source
    assert "build_learned_wrench_anneal_mask" in source
    assert "migrate_learned_wrench_scale_conditioning" in source
    assert "resuming an existing learned wrench head is not yet supported" not in source
