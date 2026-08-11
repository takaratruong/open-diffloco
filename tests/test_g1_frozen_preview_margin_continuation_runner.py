import unittest
from pathlib import Path


class G1FrozenPreviewMarginContinuationRunnerTest(unittest.TestCase):
    def test_contract_changes_only_objective_cadence_and_endpoint(self):
        from tools.run_g1_frozen_preview_adapter_continuation import (
            build_frozen_preview_adapter_kwargs,
        )
        from tools.run_g1_frozen_preview_margin_continuation import (
            build_frozen_preview_margin_kwargs,
        )

        reference = Path("/tmp/dance.npz")
        checkpoint = Path("/tmp/e011-midpoint.pkl")
        parent = build_frozen_preview_adapter_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        candidate = build_frozen_preview_margin_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )

        changed = {
            key
            for key in set(parent) | set(candidate)
            if parent.get(key) != candidate.get(key)
        }
        self.assertEqual(
            changed,
            {
                "checkpoint_interval",
                "termination_margin_weight",
                "allow_resume_termination_margin_change",
            },
        )
        self.assertEqual(candidate["termination_margin_weight"], 0.5)
        self.assertIs(
            candidate["allow_resume_termination_margin_change"], True
        )
        self.assertEqual(candidate["total_steps"], 1_572_864)
        self.assertEqual(candidate["checkpoint_interval"], 49_152)
        self.assertEqual(candidate["actor_reference_lookahead_steps"], (4, 8, 12))
        self.assertIs(candidate["actor_preview_adapter"], True)
        self.assertIs(candidate["actor_cagrad"], True)
        self.assertEqual(candidate["unroll_length"], 12)
        self.assertEqual(
            candidate["num_envs"]
            * candidate["gradient_accumulation_steps"]
            * candidate["unroll_length"],
            6_144,
        )
        self.assertEqual(
            candidate["total_steps"] - 1_376_256,
            32 * 6_144,
        )

    def test_parser_requires_resume_and_rejects_scientific_overrides(self):
        from tools.run_g1_frozen_preview_margin_continuation import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--solver-profile", "g1-4x5"])
        for override in (
            ["--termination-margin-weight", "1.0"],
            ["--num-envs", "512"],
            ["--total-steps", "2000000"],
            ["--unroll-length", "24"],
        ):
            with self.subTest(override=override):
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        [
                            "--solver-profile",
                            "g1-4x5",
                            "--resume-from",
                            "/tmp/e011-midpoint.pkl",
                            *override,
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
