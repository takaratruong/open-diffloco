import math
import unittest

import jax.numpy as jnp

from src.algorithms.shac.algorithm import (
    actor_bootstrap_scale_at_step,
    resolve_actor_bootstrap_resume_scale,
)


class ActorBootstrapScheduleTest(unittest.TestCase):
    def test_zero_delay_preserves_existing_scale_from_step_zero(self):
        scale = actor_bootstrap_scale_at_step(jnp.array(0), 0.75, 0)
        self.assertEqual(float(scale), 0.75)
        self.assertEqual(scale.dtype, jnp.dtype("float32"))

    def test_positive_delay_is_zero_before_boundary_and_full_at_boundary(self):
        self.assertEqual(
            float(actor_bootstrap_scale_at_step(jnp.array(61_439), 1.0, 61_440)),
            0.0,
        )
        self.assertEqual(
            float(actor_bootstrap_scale_at_step(jnp.array(61_440), 1.0, 61_440)),
            1.0,
        )

    def test_resume_scale_restores_checkpoint_without_explicit_authority(self):
        self.assertEqual(
            resolve_actor_bootstrap_resume_scale(
                {"actor_bootstrap_scale": 0.75},
                requested_scale=0.75,
                allow_change=False,
            ),
            0.75,
        )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_actor_bootstrap_resume_scale(
                {"actor_bootstrap_scale": 1.0},
                requested_scale=0.0,
                allow_change=False,
            )

    def test_resume_scale_allows_explicit_zero_treatment(self):
        self.assertEqual(
            resolve_actor_bootstrap_resume_scale(
                {"actor_bootstrap_scale": 1.0},
                requested_scale=0.0,
                allow_change=True,
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_actor_bootstrap_resume_scale(
                {}, requested_scale=0.0, allow_change=False
            )

    def test_resume_scale_rejects_invalid_values_and_authority(self):
        for value in (True, -0.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_actor_bootstrap_resume_scale(
                        {"actor_bootstrap_scale": 1.0},
                        requested_scale=value,
                        allow_change=True,
                    )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            resolve_actor_bootstrap_resume_scale(
                {"actor_bootstrap_scale": 1.0},
                requested_scale=0.0,
                allow_change=1,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint.*finite"):
            resolve_actor_bootstrap_resume_scale(
                {"actor_bootstrap_scale": math.nan},
                requested_scale=0.0,
                allow_change=True,
            )


if __name__ == "__main__":
    unittest.main()
