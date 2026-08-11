import unittest
from pathlib import Path


class G1CAGradH24ContinuationRunnerTest(unittest.TestCase):
    def test_contract_is_exact_e008_continuation_with_only_horizon_treatment(self):
        from tools.run_g1_cagrad_continuation import (
            build_cagrad_continuation_kwargs,
        )
        from tools.run_g1_cagrad_h24_continuation import (
            build_cagrad_h24_continuation_kwargs,
        )

        reference = Path("/tmp/dance.npz")
        checkpoint = Path("/tmp/e008-policy-final.pkl")
        parent = build_cagrad_continuation_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        parent["total_steps"] = 1_572_864
        candidate = build_cagrad_h24_continuation_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )

        self.assertEqual(set(candidate), set(parent))
        differing = {
            name for name in parent if parent[name] != candidate[name]
        }
        self.assertEqual(differing, {"unroll_length"})
        self.assertEqual(candidate["unroll_length"], 24)
        self.assertEqual(candidate["num_envs"], 256)
        self.assertEqual(candidate["gradient_accumulation_steps"], 2)
        self.assertEqual(candidate["actor_cagrad"], True)
        self.assertEqual(candidate["actor_phase_bin_count"], 5)
        self.assertEqual(candidate["actor_cagrad_alpha"], 0.5)
        self.assertEqual(candidate["actor_cagrad_iterations"], 32)
        self.assertEqual(candidate["total_steps"], 1_572_864)
        self.assertEqual(candidate["checkpoint_interval"], 196_608)
        self.assertEqual(
            candidate["resume_from"], str(checkpoint.resolve())
        )

        resumed_step = 1_179_648
        added_transitions = candidate["total_steps"] - resumed_step
        steps_per_update = (
            candidate["num_envs"]
            * candidate["gradient_accumulation_steps"]
            * candidate["unroll_length"]
        )
        self.assertEqual(added_transitions, 393_216)
        self.assertEqual(added_transitions // steps_per_update, 32)
        self.assertEqual(
            resumed_step + candidate["checkpoint_interval"], 1_376_256
        )
        self.assertEqual(
            resumed_step + 2 * candidate["checkpoint_interval"],
            candidate["total_steps"],
        )

    def test_parser_requires_resume_and_rejects_scientific_overrides(self):
        from tools.run_g1_cagrad_h24_continuation import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit) as raised:
            parser.parse_args(["--solver-profile", "g1-4x5"])
        self.assertEqual(raised.exception.code, 2)

        for override in (
            ["--num-envs", "512"],
            ["--gradient-accumulation-steps", "4"],
            ["--total-steps", "2000000"],
            ["--unroll-length", "12"],
            ["--checkpoint-interval", "100000"],
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
                "/tmp/e008-policy-final.pkl",
                *override,
            ]
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
