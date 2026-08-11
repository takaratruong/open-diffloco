import unittest
from pathlib import Path


class G1FrozenPreviewDenseCheckpointRunnerTest(unittest.TestCase):
    def test_contract_changes_only_checkpoint_interval(self):
        from tools.run_g1_frozen_preview_adapter_continuation import (
            build_frozen_preview_adapter_kwargs,
        )
        from tools.run_g1_frozen_preview_dense_checkpoint_continuation import (
            build_frozen_preview_dense_checkpoint_kwargs,
        )

        reference = Path("/tmp/dance.npz")
        checkpoint = Path("/tmp/e008-policy-final.pkl")
        parent = build_frozen_preview_adapter_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        candidate = build_frozen_preview_dense_checkpoint_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )

        self.assertEqual(set(candidate), set(parent))
        self.assertEqual(
            {
                key: value
                for key, value in candidate.items()
                if key != "checkpoint_interval"
            },
            {
                key: value
                for key, value in parent.items()
                if key != "checkpoint_interval"
            },
        )
        self.assertEqual(parent["checkpoint_interval"], 196_608)
        self.assertEqual(candidate["checkpoint_interval"], 49_152)
        self.assertEqual(
            candidate["num_envs"]
            * candidate["gradient_accumulation_steps"]
            * candidate["unroll_length"],
            6_144,
        )
        self.assertEqual(candidate["total_steps"], 1_572_864)

    def test_parser_requires_resume_and_rejects_scientific_overrides(self):
        from tools.run_g1_frozen_preview_dense_checkpoint_continuation import (
            build_parser,
        )

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--solver-profile", "g1-4x5"])
        for override in (
            ["--checkpoint-interval", "98304"],
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
                            "/tmp/e008.pkl",
                            *override,
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
