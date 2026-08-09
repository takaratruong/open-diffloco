import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac import g1_tail_contact_derivative_smoke as smoke
from src.algorithms.shac.g1_gradient_audit import FirstActionObjective
from src.algorithms.shac.g1_tail_contact_derivative_smoke import (
    CompiledCaseSmoke,
    compile_case_kernels,
    run_compiled_case_smoke,
)


class FakeFirstActionObjective:
    def __init__(self):
        self.action = jnp.linspace(-0.5, 0.5, 29, dtype=jnp.float64)
        dones = jnp.zeros((48,), dtype=jnp.bool_)
        actions = jnp.tile(self.action[None, :], (48, 1))
        self.nominal_trajectory = SimpleNamespace(dones=dones, actions=actions)

    def objective(self, action):
        return jnp.sum(jnp.square(action))

    def rollout(self, action):
        return (
            SimpleNamespace(
                dones=self.nominal_trajectory.dones,
                actions=jnp.tile(action[None, :], (48, 1)),
            ),
            object(),
        )

    def build(self):
        return FirstActionObjective(
            nominal_trajectory=self.nominal_trajectory,
            nominal_final_state=object(),
            nominal_first_action=self.action,
            nominal_objective=self.objective(self.action),
            rollout=self.rollout,
            objective=self.objective,
        )


class CompiledTailContactDerivativeSmokeTest(unittest.TestCase):
    def test_lowers_and_compiles_both_kernels_without_execution_timing(self):
        original_jit = jax.jit
        wrappers = []

        class LoweredKernel:
            def __init__(self, lowered):
                self.lowered = lowered
                self.compile_calls = 0

            def compile(self):
                self.compile_calls += 1
                return self.lowered.compile()

        class MustLowerBeforeExecution:
            def __init__(self, function):
                self.jitted = original_jit(function)
                self.lower_calls = 0
                self.lowered = None

            def __call__(self, *args, **kwargs):
                raise AssertionError("kernel execution must follow lower().compile()")

            def lower(self, *args, **kwargs):
                self.lower_calls += 1
                self.lowered = LoweredKernel(self.jitted.lower(*args, **kwargs))
                return self.lowered

        def capture_jit(function):
            wrapper = MustLowerBeforeExecution(function)
            wrappers.append(wrapper)
            return wrapper

        with mock.patch.object(smoke.jax, "jit", side_effect=capture_jit):
            compile_case_kernels(FakeFirstActionObjective().build())

        self.assertEqual(len(wrappers), 2)
        self.assertTrue(
            all(
                wrapper.lower_calls == 1
                and wrapper.lowered is not None
                and wrapper.lowered.compile_calls == 1
                for wrapper in wrappers
            )
        )

    def test_compiles_one_reverse_and_one_directional_jvp_kernel(self):
        original_jit = jax.jit
        with mock.patch.object(smoke.jax, "jit", wraps=original_jit) as jit:
            compiled = compile_case_kernels(FakeFirstActionObjective().build())

        self.assertIsInstance(compiled, CompiledCaseSmoke)
        self.assertEqual(jit.call_count, 2)
        self.assertEqual(
            (compiled.shard_seed, compiled.phase_bin, compiled.direction_seed),
            (0, 0, 12001),
        )
        self.assertGreaterEqual(compiled.reverse_compile_duration_seconds, 0.0)
        self.assertGreaterEqual(compiled.forward_compile_duration_seconds, 0.0)

    def test_runs_one_cached_reverse_and_three_sequential_complete_jvp_sweeps(self):
        fake = FakeFirstActionObjective()
        result = run_compiled_case_smoke(compile_case_kernels(fake.build()))

        self.assertEqual(result.reverse_gradient.shape, (29,))
        self.assertEqual(result.forward_gradients.shape, (3, 29))
        self.assertEqual(result.forward_primals.shape, (3,))
        self.assertEqual(result.forward_cached_sweep_durations_seconds.shape, (3,))
        np.testing.assert_array_equal(result.reverse_gradient, 2.0 * fake.action)
        for gradient in result.forward_gradients:
            np.testing.assert_array_equal(gradient, 2.0 * fake.action)
        self.assertGreaterEqual(result.reverse_cached_duration_seconds, 0.0)
        self.assertTrue(
            np.all(result.forward_cached_sweep_durations_seconds >= 0.0)
        )
        self.assertTrue(result.comparison.forward_valid)
        self.assertTrue(result.comparison.reverse_parity_valid)

    def test_runs_exactly_one_cached_reverse_and_eighty_seven_jvp_calls(self):
        compiled = compile_case_kernels(FakeFirstActionObjective().build())
        reverse_kernel = mock.Mock(wraps=compiled.reverse_kernel)
        directional_jvp_kernel = mock.Mock(wraps=compiled.directional_jvp_kernel)

        run_compiled_case_smoke(
            replace(
                compiled,
                reverse_kernel=reverse_kernel,
                directional_jvp_kernel=directional_jvp_kernel,
            )
        )

        self.assertEqual(reverse_kernel.call_count, 1)
        self.assertEqual(directional_jvp_kernel.call_count, 3 * 29)

    def test_blocks_every_directional_jvp_before_launching_the_next(self):
        compiled = compile_case_kernels(FakeFirstActionObjective().build())
        events = []
        original_block_until_ready = jax.block_until_ready

        def directional_jvp_kernel(*args, **kwargs):
            events.append("jvp")
            return compiled.directional_jvp_kernel(*args, **kwargs)

        def block_until_ready(value):
            events.append("block")
            return original_block_until_ready(value)

        with mock.patch.object(
            smoke.jax, "block_until_ready", side_effect=block_until_ready
        ):
            run_compiled_case_smoke(
                replace(
                    compiled,
                    directional_jvp_kernel=directional_jvp_kernel,
                )
            )

        self.assertEqual(events[1 : 1 + 2 * 3 * 29], ["jvp", "block"] * (3 * 29))

    def test_uses_canonical_direction_and_centered_probes(self):
        fake = FakeFirstActionObjective()
        result = run_compiled_case_smoke(compile_case_kernels(fake.build()))
        expected_direction = jax.random.normal(
            jax.random.PRNGKey(12001), (29,), dtype=jnp.float64
        )
        expected_direction = expected_direction / jnp.linalg.norm(expected_direction)

        np.testing.assert_array_equal(result.direction, expected_direction)
        np.testing.assert_array_equal(
            result.positive_action,
            fake.action + 0.001 * expected_direction,
        )
        np.testing.assert_array_equal(
            result.negative_action,
            fake.action - 0.001 * expected_direction,
        )
        self.assertEqual(result.probe_objectives.shape, (2,))
        self.assertEqual(result.probe_durations_seconds.shape, (2,))
        self.assertTrue(result.positive_done_exact)
        self.assertTrue(result.negative_done_exact)
        self.assertTrue(result.positive_support_exact)
        self.assertTrue(result.negative_support_exact)

    def test_rejects_every_noncanonical_case_identity_before_compilation(self):
        diagnostic = FakeFirstActionObjective().build()
        for kwargs in (
            {"shard_seed": 1},
            {"phase_bin": 1},
            {"direction_seed": 12002},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "canonical case"
            ):
                compile_case_kernels(diagnostic, **kwargs)


if __name__ == "__main__":
    unittest.main()
