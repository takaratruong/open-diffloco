from __future__ import annotations

import inspect

import jax
import jax.numpy as jp
import numpy as np
import pytest


def test_double_critic_exposes_two_independent_heads() -> None:
    from src.core.networks import DoubleCritic

    network = DoubleCritic(hidden=(8,))
    params = network.init(jax.random.PRNGKey(7), jp.ones((3, 5)))
    values = network.apply(params, jp.ones((3, 5)))

    assert values.shape == (3, 2)
    head0 = params["params"]["critic_0"]["Dense_1"]["kernel"]
    head1 = params["params"]["critic_1"]["Dense_1"]["kernel"]
    assert not np.array_equal(np.asarray(head0), np.asarray(head1))


def test_conservative_value_uses_minimum_double_head_and_legacy_squeeze() -> None:
    from src.algorithms.shac.ahac import conservative_value

    double = jp.asarray([[3.0, 2.0], [-1.0, 4.0]])
    legacy = jp.asarray([[3.0], [-1.0]])

    np.testing.assert_array_equal(conservative_value(double, double=True), [2.0, -1.0])
    np.testing.assert_array_equal(conservative_value(legacy, double=False), [3.0, -1.0])


def test_double_critic_loss_trains_both_heads_against_same_target() -> None:
    from src.algorithms.shac.ahac import critic_value_loss

    predictions = jp.asarray([[1.0, 3.0], [2.0, 4.0]])
    targets = jp.asarray([2.0, 2.0])
    result = critic_value_loss(predictions, targets, double=True)

    np.testing.assert_allclose(result.total, 1.5)
    np.testing.assert_allclose(result.head_losses, [0.5, 2.5])
    np.testing.assert_allclose(result.disagreement, 2.0)


def test_double_critic_loss_excludes_inactive_horizon_slots() -> None:
    from src.algorithms.shac.ahac import critic_value_loss

    predictions = jp.asarray([[1.0, 3.0], [100.0, -100.0]])
    targets = jp.asarray([2.0, 0.0])
    result = critic_value_loss(
        predictions,
        targets,
        double=True,
        active_mask=jp.asarray([True, False]),
    )

    np.testing.assert_allclose(result.total, 1.0)
    np.testing.assert_allclose(result.head_losses, [1.0, 1.0])
    np.testing.assert_allclose(result.disagreement, 2.0)


def test_select_active_tree_freezes_inactive_environment_carry() -> None:
    from src.algorithms.shac.ahac import select_active_tree

    previous = {"x": jp.asarray([1.0, 2.0]), "n": jp.asarray(3)}
    candidate = {"x": jp.asarray([9.0, 8.0]), "n": jp.asarray(7)}

    active = select_active_tree(previous, candidate, jp.asarray(True))
    inactive = select_active_tree(previous, candidate, jp.asarray(False))

    np.testing.assert_array_equal(active["x"], candidate["x"])
    np.testing.assert_array_equal(inactive["x"], previous["x"])
    assert int(active["n"]) == 7
    assert int(inactive["n"]) == 3


def test_ahac_resume_fails_closed_on_missing_or_changed_metadata() -> None:
    from src.algorithms.shac.ahac import resolve_ahac_resume_settings

    requested = {
        "ahac": True,
        "ahac_horizon_min": 8,
        "ahac_horizon_max": 24,
        "ahac_contact_threshold": 50.0,
        "ahac_dual_lr": 5e-4,
        "ahac_critic_max_iterations": 64,
        "ahac_critic_tolerance": 0.2,
    }

    with pytest.raises(ValueError, match="metadata"):
        resolve_ahac_resume_settings(
            requested=requested,
            resumed_hparams=None,
            is_resume=True,
            allow_change=False,
        )
    with pytest.raises(ValueError, match="match"):
        resolve_ahac_resume_settings(
            requested=requested,
            resumed_hparams={**requested, "ahac_horizon_min": 9},
            is_resume=True,
            allow_change=False,
        )
    assert resolve_ahac_resume_settings(
        requested=requested,
        resumed_hparams=requested,
        is_resume=True,
        allow_change=False,
    ) == requested


def test_legacy_shac_resume_does_not_require_ahac_metadata() -> None:
    from src.algorithms.shac.ahac import resolve_ahac_resume_settings

    requested = {
        "ahac": False,
        "ahac_horizon_min": 8,
        "ahac_horizon_max": 24,
        "ahac_contact_threshold": 500.0,
        "ahac_dual_lr": 5e-4,
        "ahac_critic_max_iterations": 64,
        "ahac_critic_tolerance": 0.2,
    }

    assert resolve_ahac_resume_settings(
        requested=requested,
        resumed_hparams={},
        is_resume=True,
        allow_change=False,
    ) == requested
    assert resolve_ahac_resume_settings(
        requested=requested,
        resumed_hparams=None,
        is_resume=True,
        allow_change=False,
    ) == requested


def test_train_signature_and_source_wire_all_ahac_contracts() -> None:
    from src.algorithms.shac.algorithm import train

    signature = inspect.signature(train)
    for name in (
        "ahac",
        "ahac_horizon_min",
        "ahac_horizon_max",
        "ahac_contact_threshold",
        "ahac_dual_lr",
        "ahac_critic_max_iterations",
        "ahac_critic_tolerance",
        "allow_resume_ahac_change",
    ):
        assert name in signature.parameters

    source = inspect.getsource(train)
    assert "DoubleCritic" in source
    assert "contact_stiffness" in source
    assert "active_horizon_mask" in source
    assert "update_horizon_dual" in source
    assert "critic_convergence" in source
    assert '"ahac_horizon"' in source
    assert '"ahac_contact_stiffness_mean"' in source


def test_g1_reset_and_step_metrics_share_contact_stiffness_leaf() -> None:
    from src.envs.g1_tracking.environment import G1TrackingEnv

    initial_source = inspect.getsource(G1TrackingEnv._init_metrics)
    info_source = inspect.getsource(G1TrackingEnv._base_info)
    step_source = inspect.getsource(G1TrackingEnv.step)

    assert '"contact_stiffness": zero' in initial_source
    assert '"transition_contact_stiffness": jp.array(0.0)' in info_source
    assert '"contact_stiffness": transition_contact_stiffness' in step_source


def test_checkpoint_ahac_telemetry_is_complete_and_fails_closed() -> None:
    from src.algorithms.shac.algorithm import build_checkpoint_ahac_telemetry

    metrics = {
        "ahac_horizon": 8.5,
        "ahac_horizon_before_update": 8.0,
        "ahac_active_transitions": 8,
        "ahac_dual_mean": 0.1,
        "ahac_dual_max": 0.2,
        "ahac_contact_stiffness_mean": 12.0,
        "ahac_contact_stiffness_max": 20.0,
        "ahac_contact_threshold": 15.0,
        "ahac_horizon_valid": True,
        "ahac_critic_iterations": 17,
        "ahac_critic_converged": True,
        "ahac_critic_loss_history": jp.asarray([1.0, 0.9, 0.85, 0.82, 0.8]),
        "ahac_critic_head_losses": jp.asarray([0.7, 0.9]),
        "ahac_critic_head_disagreement": 0.3,
    }

    report = build_checkpoint_ahac_telemetry(metrics)
    assert report["ahac_valid"] is True
    assert report["ahac_critic_iterations"] == 17
    assert report["ahac_active_transitions"] == 8
    assert report["ahac_critic_head_losses"] == [0.7, 0.9]

    with pytest.raises(ValueError, match="finite"):
        build_checkpoint_ahac_telemetry(
            {**metrics, "ahac_contact_stiffness_max": np.inf}
        )
