import unittest
from pathlib import Path


class G1CAGradContinuationRunnerTest(unittest.TestCase):
    def test_contract_is_exact_e006_continuation_plus_cagrad_treatment(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs
        from tools.run_g1_cagrad_continuation import (
            build_cagrad_continuation_kwargs,
        )

        reference = Path("/tmp/dance.npz")
        checkpoint = Path("/tmp/e006-policy.pkl")
        base = build_canonical_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        base.update(
            gradient_accumulation_steps=2,
            total_steps=1_179_648,
            checkpoint_interval=196_608,
        )
        candidate = build_cagrad_continuation_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )

        treatment = {
            "actor_cagrad": True,
            "actor_cagrad_alpha": 0.5,
            "actor_cagrad_iterations": 32,
            "actor_phase_bin_count": 5,
        }
        self.assertEqual(set(candidate), set(base) | set(treatment))
        self.assertEqual(
            {name: candidate[name] for name in treatment}, treatment
        )
        self.assertEqual(
            {name: candidate[name] for name in base}, base
        )
        self.assertEqual(candidate["num_envs"], 256)
        self.assertEqual(candidate["gradient_accumulation_steps"], 2)
        self.assertEqual(candidate["unroll_length"], 12)
        self.assertEqual(candidate["total_steps"], 1_179_648)
        self.assertEqual(candidate["checkpoint_interval"], 196_608)
        self.assertEqual(
            candidate["resume_from"], str(checkpoint.resolve())
        )

    def test_parser_requires_resume_and_rejects_scientific_overrides(self):
        from tools.run_g1_cagrad_continuation import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit) as raised:
            parser.parse_args(["--solver-profile", "g1-4x5"])
        self.assertEqual(raised.exception.code, 2)

        for override in (
            ["--num-envs", "512"],
            ["--gradient-accumulation-steps", "4"],
            ["--total-steps", "3145728"],
            ["--unroll-length", "24"],
            ["--actor-lr", "0.001"],
            ["--actor-cagrad", "false"],
            ["--actor-cagrad-alpha", "0.25"],
            ["--actor-cagrad-iterations", "64"],
            ["--actor-phase-bin-count", "10"],
        ):
            arguments = [
                "--solver-profile",
                "g1-4x5",
                "--resume-from",
                "/tmp/e006-policy.pkl",
                *override,
            ]
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
