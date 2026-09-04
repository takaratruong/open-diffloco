import inspect

import jax
import jax.numpy as jnp

BOUNDARIES = (
    "random_inputs",
    "first_actor_action",
    "first_mjx_substep",
    "first_mjx_control_step",
    "first_env_step",
    "rollout",
    "actor_cagrad",
    "learned_dynamics",
    "critic",
)

FIRST_MJX_SUBSTEP_COMPONENTS = (
    "integrated_state",
    "acceleration_state",
    "constraint_force",
    "contact_state",
)

FIRST_MJX_SUBSTEP_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "qfrc_applied",
    "qfrc_passive",
    "position_input_qpos",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "xanchor",
    "xaxis",
    "subtree_com",
    "subtree_local_position",
    "subtree_local_mass",
    "subtree_scan_position",
    "subtree_scan_mass",
    "subtree_divided",
    "subtree_selected",
    "rne_input_qvel",
    "cdof",
    "cdof_dot",
    "cvel",
    "cinert",
    "qfrc_bias",
    "qfrc_actuator",
    "actuator_force",
    "xfrc_applied",
    "xfrc_accumulated",
    "qfrc_smooth",
    "qfrc_constraint",
    "efc_force",
    "contact",
)


def _metrics(value):
    metrics = {
        f"determinism_{name}_fingerprint": jnp.asarray(
            [value, value + index], dtype=jnp.uint32
        )
        for index, name in enumerate(BOUNDARIES)
    }
    metrics.update(
        {
            f"determinism_first_mjx_substep_{name}_fingerprint": (
                jnp.asarray([value, value + index], dtype=jnp.uint32)
            )
            for index, name in enumerate(FIRST_MJX_SUBSTEP_COMPONENTS)
        }
    )
    metrics.update(
        {
            f"determinism_first_mjx_substep_field_{name}_fingerprint": (
                jnp.asarray([value, value + index], dtype=jnp.uint32)
            )
            for index, name in enumerate(FIRST_MJX_SUBSTEP_FIELDS)
        }
    )
    metrics[
        "determinism_first_mjx_substep_field_subtree_selected_fingerprint"
    ] = metrics[
        "determinism_first_mjx_substep_field_subtree_com_fingerprint"
    ]
    return metrics


def test_probe_reuses_one_jitted_callable_and_accepts_exact_replay():
    from src.algorithms.shac.algorithm import run_determinism_probe

    traces = []

    @jax.jit
    def step(state):
        traces.append(True)
        return state + 1.0, _metrics(jnp.asarray(7, dtype=jnp.uint32))

    jax.block_until_ready(step(jnp.asarray(0.0)))
    report = run_determinism_probe(step, jnp.asarray(0.0))

    assert len(traces) == 1
    assert report["valid"] is True
    assert report["first_mismatch_boundary"] is None
    assert report["full_state_exact"] is True
    assert report["metrics_exact"] is True
    assert report["protocol"] == "shac-compiled-update-determinism-v9"
    assert report["subtree_com_probe_consistency"] == {
        "first": True,
        "second": True,
        "valid": True,
    }
    assert all(
        component["exact"]
        for component in report["first_mjx_substep_components"].values()
    )
    assert all(
        field["exact"]
        for field in report["first_mjx_substep_fields"].values()
    )


def test_probe_reports_the_first_mismatching_boundary():
    from src.algorithms.shac.algorithm import run_determinism_probe

    class ChangingStep:
        def __init__(self):
            self.calls = 0

        def __call__(self, state):
            self.calls += 1
            metrics = _metrics(jnp.asarray(self.calls, dtype=jnp.uint32))
            return state + self.calls, metrics

    report = run_determinism_probe(ChangingStep(), jnp.asarray(0.0))

    assert report["valid"] is False
    assert report["first_mismatch_boundary"] == "random_inputs"
    assert report["full_state_exact"] is False
    assert report["metrics_exact"] is False


def test_probe_serializes_the_complete_cagrad_population(monkeypatch):
    import src.algorithms.shac.algorithm as algorithm

    def build_report(metrics, *, input_step, computed_output_step):
        assert bool(metrics["actor_cagrad_valid"]) is True
        return {
            "protocol": "sentinel-cagrad-population",
            "input_step": input_step,
            "computed_output_step": computed_output_step,
        }

    monkeypatch.setattr(algorithm, "build_cagrad_population_report", build_report)

    def step(state):
        metrics = _metrics(jnp.asarray(7, dtype=jnp.uint32))
        metrics["actor_cagrad_valid"] = jnp.asarray(True)
        return state + 12, metrics

    report = algorithm.run_determinism_probe(step, jnp.asarray(10))

    assert report["cagrad_population"] == {
        "protocol": "sentinel-cagrad-population",
        "input_step": 10,
        "computed_output_step": 22,
    }


def test_probe_serializes_exact_forward_jvp_population():
    from src.algorithms.shac.algorithm import run_determinism_probe

    def step(state):
        metrics = _metrics(jnp.asarray(7, dtype=jnp.uint32))
        metrics.update(
            {
                "actor_forward_jvp_primal_by_env": jnp.asarray(
                    [1.0, 2.0, 3.0]
                ),
                "actor_forward_jvp_tangent_by_env": jnp.asarray(
                    [0.25, -0.5, 0.0]
                ),
                "actor_forward_jvp_direction_fingerprint": jnp.asarray(
                    [11, 12, 13, 14], dtype=jnp.uint32
                ),
                "actor_forward_jvp_direction_norm": jnp.asarray(1.0),
                "actor_forward_jvp_trainable_scalar_count": jnp.asarray(
                    17, dtype=jnp.int32
                ),
                "actor_cagrad_losses_by_env": jnp.asarray(
                    [1.0, 2.0, 3.0]
                ),
                "actor_cagrad_gradient_finite_by_env": jnp.asarray(
                    [False, True, False]
                ),
            }
        )
        return state + 1.0, metrics

    report = run_determinism_probe(step, jnp.asarray(0.0))

    assert report["valid"] is True
    assert report["actor_forward_jvp"] == {
        "protocol": "shac-actor-forward-jvp-population-v1",
        "valid": True,
        "population_size": 3,
        "direction_fingerprint": [11, 12, 13, 14],
        "direction_norm": 1.0,
        "trainable_scalar_count": 17,
        "primal_matches_reverse_losses": True,
        "repeat_primal_exact": True,
        "repeat_tangent_exact": True,
        "finite_count": 3,
        "nonfinite_count": 0,
        "nonzero_count": 2,
        "finite_by_env": [True, True, True],
        "nonzero_by_env": [True, True, False],
        "reverse_gradient_finite_by_env": [False, True, False],
        "finite_on_reverse_invalid_count": 2,
        "reverse_invalid_count": 2,
        "primals_by_env": [1.0, 2.0, 3.0],
        "directional_derivatives_by_env": [0.25, -0.5, 0.0],
    }


def test_masked_rademacher_direction_is_unit_trainable_and_repeatable():
    from src.algorithms.shac.algorithm import build_masked_rademacher_direction

    params = {
        "frozen": jnp.arange(3.0),
        "trainable": jnp.arange(4.0).reshape(2, 2),
    }
    mask = {
        "frozen": jnp.zeros((3,), dtype=bool),
        "trainable": jnp.ones((2, 2), dtype=bool),
    }

    first = build_masked_rademacher_direction(params, mask, seed=17)
    second = build_masked_rademacher_direction(params, mask, seed=17)
    changed = build_masked_rademacher_direction(params, mask, seed=18)

    assert jnp.array_equal(first["frozen"], jnp.zeros((3,)))
    assert jnp.isclose(
        jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(first))),
        1.0,
    )
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(first), jax.tree.leaves(second), strict=True
        )
    )
    assert any(
        not jnp.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(first), jax.tree.leaves(changed), strict=True
        )
    )


def test_probe_reports_first_mjx_substep_components_without_causal_ordering():
    from src.algorithms.shac.algorithm import run_determinism_probe

    class ConstraintChangingStep:
        def __init__(self):
            self.calls = 0

        def __call__(self, state):
            self.calls += 1
            metrics = _metrics(jnp.asarray(7, dtype=jnp.uint32))
            if self.calls == 2:
                metrics["determinism_first_mjx_substep_fingerprint"] += 1
                metrics[
                    "determinism_first_mjx_substep_constraint_force_fingerprint"
                ] += 1
            return state, metrics

    report = run_determinism_probe(
        ConstraintChangingStep(), jnp.asarray(0.0)
    )

    assert report["first_mismatch_boundary"] == "first_mjx_substep"
    assert report["mismatching_first_mjx_substep_components"] == [
        "constraint_force"
    ]
    assert report["first_mjx_substep_components"]["integrated_state"][
        "exact"
    ] is True
    assert report["first_mjx_substep_components"]["constraint_force"][
        "exact"
    ] is False


def test_probe_reports_individual_first_mjx_substep_fields():
    from src.algorithms.shac.algorithm import run_determinism_probe

    class SmoothAccelerationChangingStep:
        def __init__(self):
            self.calls = 0

        def __call__(self, state):
            self.calls += 1
            metrics = _metrics(jnp.asarray(7, dtype=jnp.uint32))
            if self.calls == 2:
                metrics["determinism_first_mjx_substep_fingerprint"] += 1
                metrics[
                    "determinism_first_mjx_substep_acceleration_state_fingerprint"
                ] += 1
                metrics[
                    "determinism_first_mjx_substep_field_qacc_smooth_fingerprint"
                ] += 1
            return state, metrics

    report = run_determinism_probe(
        SmoothAccelerationChangingStep(), jnp.asarray(0.0)
    )

    assert report["first_mismatch_boundary"] == "first_mjx_substep"
    assert report["mismatching_first_mjx_substep_fields"] == [
        "qacc_smooth"
    ]
    assert report["first_mjx_substep_fields"]["qfrc_smooth"][
        "exact"
    ] is True
    assert report["first_mjx_substep_fields"]["qacc_smooth"][
        "exact"
    ] is False


def test_probe_rejects_subtree_reconstruction_that_misses_production():
    from src.algorithms.shac.algorithm import run_determinism_probe

    def step(state):
        metrics = _metrics(jnp.asarray(7, dtype=jnp.uint32))
        metrics[
            "determinism_first_mjx_substep_field_"
            "subtree_selected_fingerprint"
        ] += 1
        return state, metrics

    report = run_determinism_probe(step, jnp.asarray(0.0))

    assert report["first_mismatch_boundary"] is None
    assert report["subtree_com_probe_consistency"] == {
        "first": False,
        "second": False,
        "valid": False,
    }
    assert report["valid"] is False


def test_boundary_fingerprint_is_stable_and_change_sensitive():
    from src.algorithms.shac.algorithm import tree_bit_fingerprint

    original = {
        "float": jnp.asarray([1.0, -2.0], dtype=jnp.float32),
        "integer": jnp.asarray([3, 4], dtype=jnp.int32),
    }
    identical = {name: value.copy() for name, value in original.items()}
    changed = {**identical, "integer": jnp.asarray([3, 5], dtype=jnp.int32)}

    first = tree_bit_fingerprint(original)
    second = tree_bit_fingerprint(identical)
    different = tree_bit_fingerprint(changed)

    assert jnp.array_equal(first, second)
    assert not jnp.array_equal(first, different)


def test_train_probe_runs_after_compile_and_before_the_training_loop():
    from src.algorithms.shac.algorithm import (
        FIRST_MJX_SUBSTEP_COMPONENTS as IMPLEMENTED_COMPONENTS,
    )
    from src.algorithms.shac.algorithm import (
        FIRST_MJX_SUBSTEP_FIELDS as IMPLEMENTED_FIELDS,
    )
    from src.algorithms.shac.algorithm import train

    parameters = inspect.signature(train).parameters
    assert parameters["determinism_probe_output"].default is None

    source = inspect.getsource(train)
    compile_complete = source.index("state = canonicalize_tree_like")
    probe = source.index("run_determinism_probe(train_step, state)")
    training_loop = source.index("for i in range(start_iter, total_iters)")
    assert compile_complete < probe < training_loop

    for name in BOUNDARIES:
        assert f'"determinism_{name}_fingerprint"' in source
    assert IMPLEMENTED_COMPONENTS == FIRST_MJX_SUBSTEP_COMPONENTS
    assert IMPLEMENTED_FIELDS == FIRST_MJX_SUBSTEP_FIELDS
    assert "determinism_first_mjx_substep_{name}_fingerprint" in source
    assert "field_{name}_fingerprint" in source


def test_g1_step_exposes_raw_mjx_probe_boundaries():
    from src.envs.g1_tracking.environment import (
        G1TrackingEnv,
        _subtree_com_probe_values,
    )

    source = inspect.getsource(G1TrackingEnv.step)

    first_mjx_step = source.index("mjx.step(")
    first_substep = source.index(
        '"determinism_mjx_substep_fingerprint"'
    )
    control_step = source.index(
        '"determinism_mjx_control_step_fingerprint"'
    )
    returned_state = source.index("return EnvState(")

    assert first_mjx_step < first_substep < returned_state
    assert first_mjx_step < control_step < returned_state
    component = source.index("determinism_mjx_substep_{name}_fingerprint")
    assert first_mjx_step < component < returned_state
    field = source.index("field_{name}_fingerprint")
    assert '"position_input_qpos": position_input_qpos' in source
    assert '"rne_input_qvel": rne_input_qvel' in source
    assert "next_data," in source
    assert "data.qpos," in source
    assert "data.qvel," in source
    assert "subtree_probe_values," in source
    assert first_mjx_step < field < returned_state
    for name in FIRST_MJX_SUBSTEP_COMPONENTS:
        assert f'"{name}"' in source

    subtree_source = inspect.getsource(_subtree_com_probe_values)
    assert "mjx_scan.body_tree(" in subtree_source
    assert "reverse=True" in subtree_source
    assert '"subtree_scan_position"' in subtree_source
    assert '"subtree_scan_mass"' in subtree_source
    assert '"subtree_selected"' in subtree_source
