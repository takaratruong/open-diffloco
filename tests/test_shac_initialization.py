import unittest

import jax
import jax.numpy as jnp

from src.algorithms.shac.initialization import (
    canonicalize_normalizer_dtype,
    canonicalize_step_dtype,
    canonicalize_tree_like,
    commit_tree_to_local_device,
)
from src.core.data_structures import Normalizer


class ShacInitializationTest(unittest.TestCase):
    def test_value_head_squeeze_preserves_single_step_time_axis(self):
        from src.algorithms.shac.algorithm import squeeze_value_head

        one_step_batch = jnp.ones((1, 1))

        self.assertEqual(squeeze_value_head(one_step_batch).shape, (1,))

    def test_normalizer_uses_rollout_observation_dtype_before_warmup(self):
        state = Normalizer(3).init()

        canonical = canonicalize_normalizer_dtype(state, jnp.dtype("float64"))

        self.assertEqual(canonical.mean.dtype, jnp.dtype("float64"))
        self.assertEqual(canonical.var.dtype, jnp.dtype("float64"))
        self.assertEqual(canonical.count.dtype, jnp.dtype("float64"))

    def test_step_is_a_nonweak_device_integer_before_first_jit(self):
        step = canonicalize_step_dtype(0)

        self.assertFalse(getattr(step, "weak_type", True))
        expected = jnp.dtype("int64" if jax.config.x64_enabled else "int32")
        self.assertEqual(step.dtype, expected)

    def test_state_tree_matches_warmup_output_signature(self):
        current = {
            "scalar": 0,
            "array": jnp.ones(2, dtype=jnp.float32),
        }
        warmup_output = {
            "scalar": jnp.asarray(1, dtype=jnp.int64),
            "array": jnp.ones(2, dtype=jnp.float64),
        }

        canonical = canonicalize_tree_like(current, warmup_output)

        self.assertEqual(canonical["scalar"].dtype, jnp.dtype("int64"))
        self.assertFalse(canonical["scalar"].weak_type)
        self.assertEqual(canonical["array"].dtype, jnp.dtype("float64"))
        self.assertEqual(
            canonical["array"].sharding, warmup_output["array"].sharding
        )

    def test_committed_initial_tree_reuses_warmup_jit_signature(self):
        traces = []

        @jax.jit
        def update(state):
            traces.append(True)
            return {"value": state["value"] + 1.0}

        state = commit_tree_to_local_device(
            {"value": jnp.ones(2, dtype=jnp.float64)}
        )
        self.assertTrue(state["value"].committed)

        warmup_state = update(state)
        jax.block_until_ready(warmup_state["value"])
        replay_state = canonicalize_tree_like(state, warmup_state)
        replay_state = update(replay_state)
        jax.block_until_ready(replay_state["value"])

        self.assertEqual(len(traces), 1)


if __name__ == "__main__":
    unittest.main()
