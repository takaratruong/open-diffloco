import inspect

import jax
import jax.numpy as jp
import numpy as np
import pytest

from src.algorithms.shac.contact_truncation import (
    contact_gradient_barrier,
    contact_topology_event_from_info,
    resolve_contact_topology_truncation_resume_setting,
)


def test_contact_barrier_preserves_primal_and_zeros_event_vjp() -> None:
    def objective(value, event):
        tree = {"reward": value**2, "state": 3.0 * value}
        result = contact_gradient_barrier(tree, event, enabled=True)
        return result["reward"] + result["state"]

    assert objective(2.0, True) == 10.0
    assert jax.grad(objective)(2.0, True) == 0.0
    assert jax.grad(objective)(2.0, False) == 7.0


def test_disabled_barrier_is_exact_legacy_identity() -> None:
    tree = {
        "float": jp.asarray([1.0, 2.0]),
        "integer": jp.asarray(3, dtype=jp.int32),
        "boolean": jp.asarray(True),
    }

    actual = contact_gradient_barrier(tree, True, enabled=False)

    assert actual is tree


def test_dynamic_barrier_flag_shares_one_compiled_primal() -> None:
    def value_and_gradient(value, enabled):
        def objective(argument):
            result = contact_gradient_barrier(
                argument * argument,
                jp.asarray(True),
                enabled=enabled,
            )
            return result

        return jax.value_and_grad(objective)(value)

    compiled = jax.jit(value_and_gradient)
    ordinary_value, ordinary_gradient = compiled(2.0, jp.asarray(False))
    truncated_value, truncated_gradient = compiled(2.0, jp.asarray(True))

    assert np.asarray(ordinary_value).tobytes() == np.asarray(
        truncated_value
    ).tobytes()
    assert ordinary_gradient == 4.0
    assert truncated_gradient == 0.0


def test_disabled_non_g1_rollout_does_not_require_contact_event_info() -> None:
    assert not bool(contact_topology_event_from_info({}, enabled=False))


def test_enabled_rollout_requires_contact_event_info() -> None:
    with pytest.raises(KeyError, match="transition_contact_topology_event"):
        contact_topology_event_from_info({}, enabled=True)


def test_post_event_direct_actor_gradient_remains_live() -> None:
    def two_step_loss(parameter):
        first = {"state": parameter * 3.0, "reward": parameter**2}
        first = contact_gradient_barrier(first, True, enabled=True)
        second_reward = first["state"] + parameter * 5.0
        return first["reward"] + second_reward

    assert two_step_loss(2.0) == 20.0
    assert jax.grad(two_step_loss)(2.0) == 5.0


@pytest.mark.parametrize("requested", (False, True))
def test_fresh_contact_truncation_setting_is_explicit(requested: bool) -> None:
    assert (
        resolve_contact_topology_truncation_resume_setting(
            requested=requested,
            resumed_hparams=None,
            is_resume=False,
            allow_change=False,
        )
        is requested
    )


def test_resume_requires_explicit_authority_for_contact_truncation_change() -> None:
    with pytest.raises(ValueError, match="explicit resume authority"):
        resolve_contact_topology_truncation_resume_setting(
            requested=True,
            resumed_hparams={
                "actor_contact_topology_gradient_truncation": False
            },
            is_resume=True,
            allow_change=False,
        )


@pytest.mark.parametrize("resumed_hparams", (None, {}, {"wrong": False}))
def test_legacy_resume_treats_missing_contact_metadata_as_disabled(
    resumed_hparams,
) -> None:
    assert not resolve_contact_topology_truncation_resume_setting(
        requested=False,
        resumed_hparams=resumed_hparams,
        is_resume=True,
        allow_change=False,
    )


def test_legacy_resume_requires_authority_to_start_contact_truncation() -> None:
    with pytest.raises(ValueError, match="explicit resume authority"):
        resolve_contact_topology_truncation_resume_setting(
            requested=True,
            resumed_hparams=None,
            is_resume=True,
            allow_change=False,
        )


def test_legacy_resume_can_start_contact_truncation_with_authority() -> None:
    assert resolve_contact_topology_truncation_resume_setting(
        requested=True,
        resumed_hparams=None,
        is_resume=True,
        allow_change=True,
    )


def test_explicit_resume_authority_allows_contact_truncation_change() -> None:
    assert resolve_contact_topology_truncation_resume_setting(
        requested=True,
        resumed_hparams={"actor_contact_topology_gradient_truncation": False},
        is_resume=True,
        allow_change=True,
    )


@pytest.mark.parametrize(
    ("requested", "allow_change"),
    ((1, False), (False, 1)),
)
def test_contact_truncation_settings_must_be_boolean(
    requested, allow_change
) -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_contact_topology_truncation_resume_setting(
            requested=requested,
            resumed_hparams={
                "actor_contact_topology_gradient_truncation": False
            },
            is_resume=True,
            allow_change=allow_change,
        )


def test_train_exposes_default_off_contact_truncation_flags() -> None:
    from src.algorithms.shac.algorithm import train

    signature = inspect.signature(train)
    assert (
        signature.parameters[
            "actor_contact_topology_gradient_truncation"
        ].default
        is False
    )
    assert (
        signature.parameters[
            "allow_resume_actor_contact_topology_gradient_truncation_change"
        ].default
        is False
    )


def test_actor_loss_uses_gradient_state_but_raw_event_telemetry() -> None:
    from src.algorithms.shac import algorithm

    source = inspect.getsource(algorithm.train)
    compact = "".join(source.split())
    assert "gradient_next_state = contact_gradient_barrier(" in source
    assert 'gradient_next_state.reward' in source
    assert '"contact_topology_event"' in source
    assert (
        "contact_topology_event_from_info(candidate_next_state.info,"
        in compact
    )
    assert '"contact_topology_event_count"' in source
    assert '"contact_topology_event_fraction"' in source
