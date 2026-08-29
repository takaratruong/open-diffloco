import inspect

import jax
import jax.numpy as jnp

BOUNDARIES = (
    "random_inputs",
    "first_actor_action",
    "first_env_step",
    "rollout",
    "actor_cagrad",
    "learned_dynamics",
    "critic",
)


def _metrics(value):
    return {
        f"determinism_{name}_fingerprint": jnp.asarray(
            [value, value + index], dtype=jnp.uint32
        )
        for index, name in enumerate(BOUNDARIES)
    }


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
