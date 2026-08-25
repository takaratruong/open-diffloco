import inspect
import hashlib
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from src.algorithms.shac.residual_preview_adapter import FrozenPreviewResidualParams


def test_policy_anchor_penalty_is_zero_for_identical_actions():
    from src.algorithms.shac.algorithm import policy_anchor_penalty

    action = jnp.asarray([0.2, -0.4, 0.7])
    assert float(policy_anchor_penalty(action, action, weight=1.0)) == 0.0


def test_policy_anchor_penalty_is_quadratic_with_finite_candidate_gradient():
    from src.algorithms.shac.algorithm import policy_anchor_penalty

    parent = jnp.asarray([0.0, 0.0])
    candidate = jnp.asarray([1.0, -1.0])
    value = policy_anchor_penalty(candidate, parent, weight=0.5)
    gradient = jax.grad(policy_anchor_penalty)(candidate, parent, weight=0.5)

    assert float(value) == pytest.approx(0.5)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, jnp.asarray([0.5, -0.5]))


def test_policy_anchor_configuration_is_default_off_and_fail_closed():
    from src.algorithms.shac.algorithm import (
        train,
        validate_actor_policy_anchor_configuration,
    )

    assert inspect.signature(train).parameters[
        "actor_policy_anchor_weight"
    ].default == 0.0
    validate_actor_policy_anchor_configuration(
        weight=0.0,
        initial_full_actor_policy=None,
        resume_from=None,
    )
    with pytest.raises(ValueError, match="full actor"):
        validate_actor_policy_anchor_configuration(
            weight=1.0,
            initial_full_actor_policy=None,
            resume_from=None,
        )
    with pytest.raises(ValueError, match="fresh run"):
        validate_actor_policy_anchor_configuration(
            weight=1.0,
            initial_full_actor_policy=object(),
            resume_from="checkpoint.pkl",
        )
    for invalid in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="non-negative and finite"):
            validate_actor_policy_anchor_configuration(
                weight=invalid,
                initial_full_actor_policy=object(),
                resume_from=None,
            )


def test_policy_anchor_allows_explicit_frozen_residual_resume():
    from src.algorithms.shac.algorithm import (
        validate_actor_policy_anchor_configuration,
    )

    validate_actor_policy_anchor_configuration(
        weight=1.0,
        initial_full_actor_policy=None,
        resume_from="checkpoint.pkl",
        actor_residual_preview_adapter=True,
        allow_resume_actor_residual_preview_adapter_start=True,
    )
    with pytest.raises(ValueError, match="explicit frozen residual"):
        validate_actor_policy_anchor_configuration(
            weight=1.0,
            initial_full_actor_policy=None,
            resume_from="checkpoint.pkl",
            actor_residual_preview_adapter=True,
            allow_resume_actor_residual_preview_adapter_start=False,
        )


def test_train_wires_policy_anchor_into_rollout_loss_and_hparams():
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)
    assert "policy_anchor_penalty(" in source
    assert '"actor_policy_anchor_squared_error"' in source
    assert '"actor_policy_anchor_weight": actor_policy_anchor_weight' in source


def test_preview_anchors_reuse_expanded_candidate_and_parent_actions():
    """The 154-wide frozen parent must never receive the 328-wide preview obs."""
    from src.algorithms.shac.algorithm import train

    source = inspect.getsource(train)
    assert "if actor_residual_preview_adapter or actor_preview_adapter:" in source
    assert "(action, parent_action)" in source


def test_policy_anchor_checkpoint_telemetry_is_finite_and_explicit():
    from src.algorithms.shac.algorithm import build_policy_anchor_telemetry

    assert build_policy_anchor_telemetry(
        {"actor_policy_anchor_squared_error": jnp.asarray(0.25)},
        weight=1.0,
    ) == {
        "actor_policy_anchor_weight": 1.0,
        "actor_policy_anchor_squared_error": 0.25,
        "actor_policy_anchor_valid": True,
    }
    with pytest.raises(ValueError, match="finite"):
        build_policy_anchor_telemetry(
            {"actor_policy_anchor_squared_error": jnp.asarray(float("nan"))},
            weight=1.0,
        )


def test_source_policy_anchor_loads_an_exact_frozen_residual(tmp_path):
    from src.algorithms.shac.algorithm import load_policy_anchor_source

    source = FrozenPreviewResidualParams(
        parent={"p": jnp.asarray([1.0])},
        adapter={"a": jnp.asarray([2.0])},
    )
    path = tmp_path / "source.pkl"
    path.write_bytes(pickle.dumps(SimpleNamespace(actor_params=source)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_policy_anchor_source(path, expected_sha256=digest)

    assert isinstance(loaded, FrozenPreviewResidualParams)
    assert float(loaded.adapter["a"][0]) == 2.0
    with pytest.raises(ValueError, match="SHA-256"):
        load_policy_anchor_source(path, expected_sha256="0" * 64)


def test_source_policy_anchor_configuration_is_explicit_and_fail_closed():
    from src.algorithms.shac.algorithm import (
        validate_policy_anchor_source_configuration,
    )

    validate_policy_anchor_source_configuration(
        path="source.pkl",
        sha256="a" * 64,
        weight=1.0,
        actor_residual_preview_adapter=True,
    )
    validate_policy_anchor_source_configuration(
        path=None,
        sha256=None,
        weight=1.0,
        actor_residual_preview_adapter=True,
    )
    with pytest.raises(ValueError, match="path and SHA-256"):
        validate_policy_anchor_source_configuration(
            path="source.pkl",
            sha256=None,
            weight=1.0,
            actor_residual_preview_adapter=True,
        )
    with pytest.raises(ValueError, match="residual preview"):
        validate_policy_anchor_source_configuration(
            path="source.pkl",
            sha256="a" * 64,
            weight=1.0,
            actor_residual_preview_adapter=False,
        )
    with pytest.raises(ValueError, match="positive anchor weight"):
        validate_policy_anchor_source_configuration(
            path="source.pkl",
            sha256="a" * 64,
            weight=0.0,
            actor_residual_preview_adapter=True,
        )


def test_source_policy_anchor_requires_the_same_frozen_parent():
    from src.algorithms.shac.algorithm import validate_policy_anchor_source_parent

    source = FrozenPreviewResidualParams(
        parent={"p": jnp.asarray([1.0])}, adapter={"a": jnp.asarray([2.0])}
    )
    candidate = FrozenPreviewResidualParams(
        parent={"p": jnp.asarray([1.0])}, adapter={"a": jnp.asarray([3.0])}
    )
    validate_policy_anchor_source_parent(source, candidate)

    changed_parent = FrozenPreviewResidualParams(
        parent={"p": jnp.asarray([1.01])}, adapter=candidate.adapter
    )
    with pytest.raises(ValueError, match="frozen parent"):
        validate_policy_anchor_source_parent(source, changed_parent)


def test_train_wires_hash_bound_source_policy_as_optional_anchor_target():
    from src.algorithms.shac.algorithm import train

    signature = inspect.signature(train).parameters
    assert signature["actor_policy_anchor_source_path"].default is None
    assert signature["actor_policy_anchor_source_sha256"].default is None
    source = inspect.getsource(train)
    assert "load_policy_anchor_source(" in source
    assert "policy_anchor_source_params" in source
    assert '"actor_policy_anchor_source_sha256"' in source
    assert "actor_policy_anchor_source_sha256" in source


def test_source_policy_anchor_resume_change_requires_explicit_authority():
    from src.algorithms.shac.algorithm import (
        resolve_policy_anchor_source_resume_setting,
    )

    saved = {
        "actor_policy_anchor_source_path": None,
        "actor_policy_anchor_source_sha256": None,
    }
    with pytest.raises(ValueError, match="explicit authority"):
        resolve_policy_anchor_source_resume_setting(
            saved,
            requested_path="source.pkl",
            requested_sha256="a" * 64,
            allow_change=False,
        )
    assert resolve_policy_anchor_source_resume_setting(
        saved,
        requested_path="source.pkl",
        requested_sha256="a" * 64,
        allow_change=True,
    ) == ("source.pkl", "a" * 64)
    with pytest.raises(ValueError, match="metadata"):
        resolve_policy_anchor_source_resume_setting(
            {},
            requested_path=None,
            requested_sha256=None,
            allow_change=False,
        )
    assert resolve_policy_anchor_source_resume_setting(
        {},
        requested_path="source.pkl",
        requested_sha256="a" * 64,
        allow_change=True,
    ) == ("source.pkl", "a" * 64)
