import unittest
from pathlib import Path


class G1CAGradFutureReferenceRunnerTest(unittest.TestCase):
    def test_contract_is_exact_e008_h12_continuation_plus_preview(self):
        from tools.run_g1_cagrad_continuation import (
            build_cagrad_continuation_kwargs,
        )
        from tools.run_g1_cagrad_future_reference_continuation import (
            build_cagrad_future_reference_kwargs,
        )

        reference = Path("/tmp/dance.npz")
        checkpoint = Path("/tmp/e008-policy-final.pkl")
        parent = build_cagrad_continuation_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        parent["total_steps"] = 1_572_864
        candidate = build_cagrad_future_reference_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        treatment = {
            "actor_reference_lookahead_steps": (4, 8, 12),
            "allow_resume_actor_reference_lookahead_upgrade": True,
        }

        self.assertEqual(set(candidate), set(parent) | set(treatment))
        self.assertEqual(
            {key: candidate[key] for key in parent}, parent
        )
        self.assertEqual(
            {key: candidate[key] for key in treatment}, treatment
        )
        self.assertEqual(candidate["unroll_length"], 12)
        self.assertEqual(candidate["num_envs"], 256)
        self.assertEqual(candidate["gradient_accumulation_steps"], 2)
        self.assertTrue(candidate["actor_cagrad"])
        self.assertEqual(candidate["total_steps"], 1_572_864)
        self.assertEqual(candidate["checkpoint_interval"], 196_608)

        resumed_step = 1_179_648
        added = candidate["total_steps"] - resumed_step
        transitions_per_update = (
            candidate["num_envs"]
            * candidate["gradient_accumulation_steps"]
            * candidate["unroll_length"]
        )
        self.assertEqual(added, 393_216)
        self.assertEqual(added // transitions_per_update, 64)
        self.assertEqual(
            resumed_step + candidate["checkpoint_interval"], 1_376_256
        )

    def test_parser_requires_resume_and_rejects_scientific_overrides(self):
        from tools.run_g1_cagrad_future_reference_continuation import (
            build_parser,
        )

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--solver-profile", "g1-4x5"])
        for override in (
            ["--actor-reference-lookahead-steps", "4", "8", "12"],
            ["--allow-resume-actor-reference-lookahead-upgrade"],
            ["--num-envs", "512"],
            ["--total-steps", "2000000"],
            ["--unroll-length", "24"],
            ["--actor-lr", "0.001"],
        ):
            with self.subTest(override=override):
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        [
                            "--solver-profile",
                            "g1-4x5",
                            "--resume-from",
                            "/tmp/e008.pkl",
                            *override,
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
