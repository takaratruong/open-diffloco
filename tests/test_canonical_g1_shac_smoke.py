import math
import unittest
from pathlib import Path


def valid_receipt():
    return {
        "schema_version": 1,
        "solver_profile": "g1-4x5",
        "num_envs": 2,
        "unroll_length": 1,
        "actor_updates": 1,
        "critic_iterations": 16,
        "distinct_model_count": 2,
        "sampled_model_sha256": ["a" * 64, "b" * 64],
        "zero_head_reference_target_max_error": 0.0,
        "reward": 0.25,
        "actor_grad_finite_fraction": 1.0,
        "critic_grad_finite_fraction": 1.0,
        "actor_grad_raw_median": 3.0,
        "actor_grad_raw_max": 5.0,
        "critic_grad_raw_median": 2.0,
        "critic_grad_raw_max": 4.0,
        "optimizer_update_norm": 0.01,
    }


class CanonicalG1ShacSmokeTest(unittest.TestCase):
    def test_smoke_receipt_requires_real_randomization_and_update(self):
        from tools.smoke_canonical_g1_shac import validate_smoke_receipt

        receipt = valid_receipt()
        validate_smoke_receipt(receipt)

        self.assertEqual(receipt["num_envs"], 2)
        self.assertEqual(receipt["distinct_model_count"], 2)
        self.assertLessEqual(
            receipt["zero_head_reference_target_max_error"], 1e-12
        )
        self.assertEqual(receipt["actor_grad_finite_fraction"], 1.0)
        self.assertEqual(receipt["critic_grad_finite_fraction"], 1.0)
        self.assertGreater(receipt["optimizer_update_norm"], 0.0)

    def test_smoke_receipt_rejects_missing_nonfinite_or_replayed_evidence(self):
        from tools.smoke_canonical_g1_shac import validate_smoke_receipt

        for name, value in (
            ("reward", math.nan),
            ("actor_grad_finite_fraction", 0.5),
            ("optimizer_update_norm", 0.0),
            ("distinct_model_count", 1),
        ):
            receipt = valid_receipt()
            receipt[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_smoke_receipt(receipt)

        receipt = valid_receipt()
        del receipt["reward"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_smoke_receipt(receipt)

    def test_smoke_kwargs_bound_only_execution_budget(self):
        from tools.smoke_canonical_g1_shac import build_smoke_kwargs

        kwargs = build_smoke_kwargs(
            "upstream-1x5", Path("/tmp/dance.npz"), seed=9
        )

        self.assertEqual(kwargs["num_envs"], 2)
        self.assertEqual(kwargs["unroll_length"], 1)
        self.assertEqual(kwargs["total_steps"], 2)
        self.assertEqual(kwargs["checkpoint_interval"], 2)
        self.assertEqual(kwargs["critic_iterations"], 16)
        self.assertTrue(kwargs["domain_randomization"])
        self.assertTrue(kwargs["reference_residual_control"])
        self.assertTrue(kwargs["actor_observation_noise"])


if __name__ == "__main__":
    unittest.main()
