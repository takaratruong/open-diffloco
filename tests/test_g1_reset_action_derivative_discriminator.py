from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np


def test_smooth_reference_state_loss_is_zero_at_reference_and_scaled() -> None:
    from experiments.g1_reset_action_derivative_discriminator.run import (
        smooth_reference_state_loss,
    )

    reference_qpos = jnp.zeros(9, dtype=jnp.float64)
    reference_qvel = jnp.zeros(8, dtype=jnp.float64)
    assert (
        smooth_reference_state_loss(
            reference_qpos,
            reference_qvel,
            reference_qpos,
            reference_qvel,
        )
        == 0.0
    )

    qpos = reference_qpos.at[0].set(0.3)
    expected = 0.5 / reference_qpos.size
    np.testing.assert_allclose(
        smooth_reference_state_loss(
            qpos,
            reference_qvel,
            reference_qpos,
            reference_qvel,
        ),
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_two_objective_derivative_helper_matches_analytic_jacobian() -> None:
    from experiments.g1_reset_action_derivative_discriminator.run import (
        compute_two_objective_derivatives,
    )

    action = jnp.asarray([0.2, -0.4, 0.5], dtype=jnp.float64)
    direction = jnp.asarray([1.0, -1.0, 1.0], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    objectives = (
        lambda value: jnp.sum(value**2),
        lambda value: value[0] - 2.0 * value[2],
    )

    result = compute_two_objective_derivatives(
        objectives,
        action,
        direction=direction,
        finite_difference_epsilon=1e-5,
    )
    expected_jacobian = np.asarray(
        [[0.4, -0.8, 1.0], [1.0, 0.0, -2.0]], dtype=np.float64
    )
    np.testing.assert_allclose(result["reverse_jacobian"], expected_jacobian)
    np.testing.assert_allclose(result["forward_jacobian"], expected_jacobian)
    expected_primal = np.asarray([objective(action) for objective in objectives])
    np.testing.assert_allclose(result["source_primal"], expected_primal)
    np.testing.assert_allclose(result["reverse_primal"], expected_primal)
    np.testing.assert_allclose(result["forward_primal"], expected_primal)
    np.testing.assert_allclose(
        result["finite_difference_directional"],
        expected_jacobian @ np.asarray(direction),
        rtol=1e-9,
        atol=1e-10,
    )


def test_two_objectives_are_structurally_isolated_from_singular_pullbacks() -> None:
    from experiments.g1_reset_action_derivative_discriminator.run import (
        compute_two_objective_derivatives,
    )

    result = compute_two_objective_derivatives(
        (
            lambda value: value[0],
            lambda value: jnp.arccos(jnp.clip(value[0], 0.0, 1.0)),
        ),
        jnp.asarray([1.0], dtype=jnp.float64),
        direction=jnp.asarray([1.0], dtype=jnp.float64),
        finite_difference_epsilon=1e-5,
    )
    assert np.isfinite(np.asarray(result["reverse_jacobian"])[0]).all()
    assert not np.isfinite(np.asarray(result["reverse_jacobian"])[1]).all()


def test_model_order_action_round_trips_through_environment_permutation() -> None:
    from experiments.g1_reset_action_derivative_discriminator.run import (
        actor_action_from_model_action,
    )

    actor_to_model = np.asarray([2, 0, 3, 1], dtype=np.int64)
    model_to_actor = np.argsort(actor_to_model)
    actor_action = np.asarray([10.0, 20.0, 30.0, 40.0])
    model_action = actor_action[actor_to_model]
    recovered = actor_action_from_model_action(model_action, model_to_actor)
    np.testing.assert_array_equal(recovered[actor_to_model], model_action)
    np.testing.assert_array_equal(recovered, actor_action)


def test_reset_qpos_gate_allows_only_quaternion_normalization_roundoff() -> None:
    from experiments.g1_reset_action_derivative_discriminator.run import (
        validate_reconstructed_reset_qpos,
    )

    captured = np.arange(36, dtype=np.float64)
    captured[3:7] = np.asarray([0.5, 0.4, 0.3, 0.2])
    reconstructed = captured.copy()
    reconstructed[3] = np.nextafter(reconstructed[3], np.inf)
    expected_delta = float(reconstructed[3] - captured[3])
    assert validate_reconstructed_reset_qpos(reconstructed, captured) == expected_delta

    wrong_joint = reconstructed.copy()
    wrong_joint[7] = np.nextafter(wrong_joint[7], np.inf)
    with np.testing.assert_raises(ValueError):
        validate_reconstructed_reset_qpos(wrong_joint, captured)

    wrong_quaternion = captured.copy()
    wrong_quaternion[3] += 1e-10
    with np.testing.assert_raises(ValueError):
        validate_reconstructed_reset_qpos(wrong_quaternion, captured)


def test_probe_builds_only_the_diffsim_common_environment(monkeypatch) -> None:
    from experiments.g1_reset_action_derivative_discriminator import run

    captured = {}
    marker = object()

    def fake_make(variant, **kwargs):
        captured.update(variant=variant, kwargs=kwargs)
        return marker

    monkeypatch.setattr(run, "make_evaluation_env", fake_make)
    hparams = {
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "actor_history_len": 10,
        "actor_reference_preview_mode": "delta",
        "tracking_velocity_kernel": "exponential",
        "tracking_root_velocity_weight": 1.0,
    }
    result = run.build_common_probe_env("/reference.npz", hparams)

    assert result is marker
    assert captured["variant"] == hparams["env_variant"]
    assert captured["kwargs"]["actor_history_len"] == 10
    assert captured["kwargs"]["actor_reference_preview_mode"] == "delta"
    assert captured["kwargs"]["tracking_root_velocity_weight"] == 1.0
    assert captured["kwargs"]["reference_residual_control"] is True
    assert captured["kwargs"]["reference_residual_scale"] == 1.0
    assert captured["kwargs"]["solver_iterations"] == 4
    assert captured["kwargs"]["solver_ls_iterations"] == 5


def _flags(ppo: list[bool], diffsim: list[bool]) -> dict[str, list[bool]]:
    interleaved = []
    for ppo_value, diffsim_value in zip(ppo, diffsim, strict=True):
        interleaved.extend((ppo_value, diffsim_value))
    return {"case_pass": interleaved}


def test_classifier_separates_physics_reward_and_action_dependence() -> None:
    from experiments.g1_reset_action_derivative_discriminator.run import (
        classify_derivative_cases,
    )

    all_valid = [True] * 5
    one_bad = [True, True, False, True, True]

    result = classify_derivative_cases(
        arms=["ppo", "diffsim"] * 5,
        measurement_valid=True,
        smooth_report=_flags(all_valid, one_bad),
        reward_report=_flags(all_valid, all_valid),
    )
    assert result["outcome"] == "diffsim-action-only-smooth-derivative-failure"

    result = classify_derivative_cases(
        arms=["ppo", "diffsim"] * 5,
        measurement_valid=True,
        smooth_report=_flags(all_valid, all_valid),
        reward_report=_flags(one_bad, one_bad),
    )
    assert result["outcome"] == "smooth-physics-valid-reward-derivative-failure"

    result = classify_derivative_cases(
        arms=["ppo", "diffsim"] * 5,
        measurement_valid=True,
        smooth_report=_flags(all_valid, all_valid),
        reward_report=_flags(all_valid, all_valid),
    )
    assert result["outcome"] == "reset-boundary-derivatives-valid"

    result = classify_derivative_cases(
        arms=["ppo", "diffsim"] * 5,
        measurement_valid=False,
        smooth_report=_flags(all_valid, all_valid),
        reward_report=_flags(all_valid, all_valid),
    )
    assert result["outcome"] == "invalid-measurement"
    assert result["scientifically_interpretable"] is False


def test_runner_reuses_captured_actions_without_policy_or_optimizer_work() -> None:
    from experiments.g1_reset_action_derivative_discriminator import run

    source = inspect.getsource(run)
    assert "paired_trajectories.npz" in source
    assert "load_source_actor_policy" not in source
    assert "_load_policy" not in source
    assert "train(" not in source
    assert '"policy_evaluation_computed": False' in source
    assert '"optimizer_update_retained": False' in source


def test_runner_builds_one_compiled_callable_for_two_exact_invocations() -> None:
    from experiments.g1_reset_action_derivative_discriminator import run

    source = inspect.getsource(run._run)
    assert source.count("_build_compiled_probe(") == 1
    assert source.count("compiled_probe(states, actions)") == 2
