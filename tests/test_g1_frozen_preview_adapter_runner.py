import unittest
from pathlib import Path


class G1FrozenPreviewAdapterRunnerTest(unittest.TestCase):
    def test_contract_changes_only_adapter_flag_from_future_parent(self):
        from tools.run_g1_cagrad_future_reference_continuation import (
            build_cagrad_future_reference_kwargs,
        )
        from tools.run_g1_frozen_preview_adapter_continuation import (
            build_frozen_preview_adapter_kwargs,
        )

        reference = Path("/tmp/dance.npz")
        checkpoint = Path("/tmp/e008-policy-final.pkl")
        parent = build_cagrad_future_reference_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )
        candidate = build_frozen_preview_adapter_kwargs(
            "g1-4x5", reference, seed=0, resume_from=checkpoint
        )

        self.assertEqual(
            set(candidate), set(parent) | {"actor_preview_adapter"}
        )
        self.assertEqual(
            {key: candidate[key] for key in parent}, parent
        )
        self.assertIs(candidate["actor_preview_adapter"], True)
        self.assertEqual(candidate["total_steps"], 1_572_864)
        self.assertEqual(candidate["checkpoint_interval"], 196_608)
        self.assertEqual(candidate["unroll_length"], 12)
        self.assertEqual(candidate["num_envs"], 256)
        self.assertEqual(candidate["gradient_accumulation_steps"], 2)
        self.assertEqual(
            candidate["actor_reference_lookahead_steps"], (4, 8, 12)
        )

    def test_parser_requires_resume_and_rejects_scientific_overrides(self):
        from tools.run_g1_frozen_preview_adapter_continuation import (
            build_parser,
        )

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--solver-profile", "g1-4x5"])
        for override in (
            ["--actor-preview-adapter"],
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
