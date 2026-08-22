import json
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tools.evaluate_g1_ivw_h_gradients import (
    ACTION_DIM,
    ACTORS,
    HORIZON,
    PHASES,
    POPULATION,
    PROTOCOL,
    REPLICAS_PER_PHASE,
    build_fixed_phase_population,
    classify_ivw_h_gradient_audit,
    push_action_gradients_to_policy,
    validate_completion,
)


def _actor_row(
    *,
    solver_gain=0.06,
    tape_gain=0.06,
    retained_cosine=0.8,
    retained_norm_ratio=1.0,
    phase_regression=0.0,
):
    ordinary_solver = 0.5
    ordinary_tape = 0.4
    return {
        "pathwise_vjp_cosine_min": 0.9995,
        "pathwise_vjp_norm_ratio_min": 0.9995,
        "pathwise_vjp_norm_ratio_max": 1.0005,
        "finite_phase_count_min": 24,
        "ordinary_mean_solver_cosine": ordinary_solver,
        "ivw_h_mean_solver_cosine": ordinary_solver + solver_gain,
        "ordinary_mean_tape_cosine": ordinary_tape,
        "ivw_h_mean_tape_cosine": ordinary_tape + tape_gain,
        "ordinary_phase_solver_cosine": [0.5] * 5,
        "ivw_h_phase_solver_cosine": [0.5 - phase_regression] * 5,
        "ordinary_phase_tape_cosine": [0.4] * 5,
        "ivw_h_phase_tape_cosine": [0.4 - phase_regression] * 5,
        "retained_pathwise_cosine": retained_cosine,
        "retained_pathwise_norm_ratio": retained_norm_ratio,
    }


def _evidence(**row_kwargs):
    return {
        "valid": True,
        "actors": {actor: _actor_row(**row_kwargs) for actor in ACTORS},
    }


def test_registered_population_has_two_distinct_replayable_tapes():
    population = build_fixed_phase_population(seed=0)

    assert population["phase"].shape == (POPULATION,)
    assert population["noise"].shape == (
        2,
        POPULATION,
        HORIZON,
        ACTION_DIM,
    )
    np.testing.assert_array_equal(
        population["phase"], np.repeat(PHASES, REPLICAS_PER_PHASE)
    )
    assert not np.array_equal(population["noise"][0], population["noise"][1])
    for index, tape_seed in enumerate((913_024, 913_025)):
        expected = np.random.default_rng(tape_seed).standard_normal(
            (POPULATION, HORIZON, ACTION_DIM), dtype=np.float32
        )
        np.testing.assert_array_equal(population["noise"][index], expected)


def test_registered_population_rejects_other_seed():
    with pytest.raises(ValueError, match="seed must be zero"):
        build_fixed_phase_population(seed=1)


def test_classifier_selects_robust_only_when_both_gains_clear():
    assert classify_ivw_h_gradient_audit(_evidence()) == "ivw-h-robust"
    assert (
        classify_ivw_h_gradient_audit(_evidence(solver_gain=0.049))
        == "ivw-h-neutral"
    )
    assert (
        classify_ivw_h_gradient_audit(_evidence(tape_gain=0.049))
        == "ivw-h-neutral"
    )


def test_destructive_precedes_robust_when_retained_signal_fails():
    assert (
        classify_ivw_h_gradient_audit(_evidence(retained_cosine=0.49))
        == "ivw-h-destructive"
    )
    assert (
        classify_ivw_h_gradient_audit(_evidence(retained_norm_ratio=4.01))
        == "ivw-h-destructive"
    )
    assert (
        classify_ivw_h_gradient_audit(_evidence(phase_regression=0.051))
        == "ivw-h-destructive"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence.update(valid=False),
        lambda evidence: evidence["actors"].pop("fresh"),
        lambda evidence: evidence["actors"]["fresh"].update(
            pathwise_vjp_cosine_min=0.998
        ),
        lambda evidence: evidence["actors"]["fresh"].update(
            finite_phase_count_min=15
        ),
        lambda evidence: evidence["actors"]["fresh"].update(
            ordinary_mean_solver_cosine=np.nan
        ),
    ],
)
def test_classifier_fails_closed_on_invalid_evidence(mutation):
    evidence = _evidence()
    mutation(evidence)
    assert classify_ivw_h_gradient_audit(evidence) == "invalid-execution"


def test_action_node_vjp_matches_direct_recurrent_pathwise_gradient():
    params = {"gain": jnp.asarray([0.7, -0.2])}
    epsilon = jnp.asarray(
        [[0.1, -0.2], [0.3, 0.05], [-0.4, 0.1]], dtype=jnp.float32
    )

    def actor_apply(parameters, observations):
        return observations * parameters["gain"]

    def rollout(parameters, delta):
        def step(state, values):
            epsilon_t, delta_t = values
            observation = jnp.stack([state, 1.0 + state])
            mean = actor_apply(parameters, observation)
            action = mean + epsilon_t + delta_t
            next_state = 0.8 * state + jnp.sum(jnp.tanh(action))
            return next_state, observation

        final, observations = jax.lax.scan(
            step,
            jnp.asarray(0.25, dtype=jnp.float32),
            (epsilon, delta),
        )
        return final**2, observations

    zero = jnp.zeros_like(epsilon)
    direct = jax.grad(lambda p: rollout(p, zero)[0])(params)
    (_, observations), action_gradient = jax.value_and_grad(
        lambda delta: rollout(params, delta), has_aux=True
    )(zero)
    pushed = push_action_gradients_to_policy(
        actor_apply,
        params,
        observations,
        action_gradient,
    )

    np.testing.assert_allclose(
        pushed["gain"], direct["gain"], rtol=1e-6, atol=1e-7
    )


def test_completion_reopens_hash_bound_artifacts(tmp_path: Path):
    artifact = tmp_path / "gradient_summary.json"
    artifact.write_text('{"valid": true}\n', encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    completion = tmp_path / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "valid": True,
                "protocol": PROTOCOL,
                "outcome": "ivw-h-neutral",
                "artifacts": {artifact.name: digest},
            }
        ),
        encoding="utf-8",
    )

    assert validate_completion(completion)["outcome"] == "ivw-h-neutral"
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_completion(completion)
