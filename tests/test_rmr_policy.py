"""Focused public-boundary proofs for the pure-JAX RMR actor seam.

These exercise the three frozen API names:
    rmr_policy_from_state_dict(model_state_dict, normalizer_state_dict)
    apply_rmr_policy(policy, observations)
    compose_rmr_residual(policy, observations, residual)
"""
import os
import sys

import contextlib

import numpy as np
import jax
import jax.numpy as jnp

try:  # pytest is optional; the file is also runnable directly.
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback shim
    class _PytestShim:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc} to be raised")

    pytest = _PytestShim()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.rmr_policy import (  # noqa: E402
    rmr_policy_from_state_dict,
    apply_rmr_policy,
    compose_bounded_rmr_residual,
    compose_rmr_residual,
)


def _state_dicts():
    """A 3-linear-layer actor with non-monotonic numeric indices (0, 2, 10)."""
    rng = np.random.default_rng(0)
    obs_dim, h1, h2, act_dim = 4, 5, 6, 3
    # Insert keys out of numeric order to prove ordering is by index, not
    # insertion or lexical order ("10" < "2" lexically).
    model = {
        "actor.10.weight": rng.standard_normal((act_dim, h2)).astype(np.float32),
        "actor.10.bias": rng.standard_normal((act_dim,)).astype(np.float32),
        "actor.2.weight": rng.standard_normal((h2, h1)).astype(np.float32),
        "actor.2.bias": rng.standard_normal((h2,)).astype(np.float32),
        "actor.0.weight": rng.standard_normal((h1, obs_dim)).astype(np.float32),
        "actor.0.bias": rng.standard_normal((h1,)).astype(np.float32),
    }
    normalizer = {
        "_mean": rng.standard_normal((obs_dim,)).astype(np.float32),
        "_std": (np.abs(rng.standard_normal((obs_dim,))) + 0.5).astype(np.float32),
    }
    return model, normalizer, obs_dim, act_dim


def _numpy_forward(model, normalizer, obs):
    x = (obs - normalizer["_mean"]) / (normalizer["_std"] + 1e-8)
    idxs = sorted({int(k.split(".")[1]) for k in model if k.startswith("actor.")})
    for i, idx in enumerate(idxs):
        w = model[f"actor.{idx}.weight"]
        b = model[f"actor.{idx}.bias"]
        x = x @ w.T + b
        if i < len(idxs) - 1:
            x = np.where(x > 0, x, np.expm1(x))  # ELU, alpha=1.0
    return x


def test_inference_matches_reference_unbatched():
    model, normalizer, obs_dim, _ = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(1).standard_normal((obs_dim,)).astype(np.float32)

    got = np.asarray(apply_rmr_policy(policy, obs))
    expected = _numpy_forward(model, normalizer, obs)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_inference_matches_reference_batched():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(2).standard_normal((7, obs_dim)).astype(np.float32)

    got = np.asarray(apply_rmr_policy(policy, obs))
    expected = _numpy_forward(model, normalizer, obs)

    assert got.shape == (7, act_dim)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_source_row_normalizer_preserves_unbatched_action_shape():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    normalizer = {
        key: value.reshape(1, -1) for key, value in normalizer.items()
    }
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(9).standard_normal((obs_dim,)).astype(np.float32)

    got = np.asarray(apply_rmr_policy(policy, obs))
    expected = _numpy_forward(model, normalizer, obs)[0]

    assert got.shape == (act_dim,)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_zero_residual_preserves_source_action():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(3).standard_normal((4, obs_dim)).astype(np.float32)

    source = np.asarray(apply_rmr_policy(policy, obs))
    composed = np.asarray(
        compose_rmr_residual(policy, obs, jnp.zeros((4, act_dim)))
    )
    np.testing.assert_allclose(composed, source, rtol=1e-6, atol=1e-6)


def test_nonzero_residual_adds_exactly():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(4).standard_normal((obs_dim,)).astype(np.float32)
    residual = np.random.default_rng(5).standard_normal((act_dim,)).astype(np.float32)

    source = np.asarray(apply_rmr_policy(policy, obs))
    composed = np.asarray(compose_rmr_residual(policy, obs, residual))
    np.testing.assert_allclose(composed, source + residual, rtol=1e-6, atol=1e-6)


def test_bounded_residual_is_zero_at_initialization_and_respects_limit():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(10).standard_normal((obs_dim,)).astype(np.float32)
    source = np.asarray(apply_rmr_policy(policy, obs))

    zero = np.asarray(
        compose_bounded_rmr_residual(
            policy, obs, jnp.zeros(act_dim), action_scale=0.1
        )
    )
    saturated = np.asarray(
        compose_bounded_rmr_residual(
            policy, obs, jnp.full(act_dim, 100.0), action_scale=0.1
        )
    )

    np.testing.assert_array_equal(zero, source)
    np.testing.assert_allclose(saturated - source, 0.1, atol=1e-6)


def test_residual_jacobian_is_identity():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(6).standard_normal((obs_dim,)).astype(np.float32)
    residual = jnp.zeros((act_dim,))

    jac = jax.jacobian(lambda r: compose_rmr_residual(policy, obs, r))(residual)
    np.testing.assert_allclose(np.asarray(jac), np.eye(act_dim), rtol=1e-6, atol=1e-6)


def test_compose_is_jittable():
    model, normalizer, obs_dim, act_dim = _state_dicts()
    policy = rmr_policy_from_state_dict(model, normalizer)
    obs = np.random.default_rng(7).standard_normal((obs_dim,)).astype(np.float32)
    residual = np.random.default_rng(8).standard_normal((act_dim,)).astype(np.float32)

    jitted = jax.jit(compose_rmr_residual)
    got = np.asarray(jitted(policy, obs, residual))
    expected = np.asarray(compose_rmr_residual(policy, obs, residual))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_missing_bias_fails_closed():
    model, normalizer, _, _ = _state_dicts()
    del model["actor.10.bias"]
    with pytest.raises((KeyError, ValueError)):
        rmr_policy_from_state_dict(model, normalizer)


def test_no_actor_layers_fails_closed():
    _, normalizer, _, _ = _state_dicts()
    with pytest.raises((KeyError, ValueError)):
        rmr_policy_from_state_dict({}, normalizer)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} FOCUSED TESTS PASSED")
