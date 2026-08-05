import unittest

import jax
import jax.numpy as jnp

from src.algorithms.shac.initialization import (
    canonicalize_normalizer_dtype,
    canonicalize_step_dtype,
)
from src.core.data_structures import Normalizer


class ShacInitializationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
