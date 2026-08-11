import unittest
from pathlib import Path


class G1Effective512ShacRunnerTest(unittest.TestCase):
    def test_contract_changes_only_effective_batch_and_derived_budget(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs
        from tools.run_g1_effective512_shac import build_effective512_kwargs

        canonical = build_canonical_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        candidate = build_effective512_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )

        self.assertEqual(set(candidate), set(canonical))
        differing = {
            name for name in canonical if canonical[name] != candidate[name]
        }
        self.assertEqual(
            differing, {"gradient_accumulation_steps", "total_steps"}
        )
        self.assertEqual(candidate["num_envs"], 256)
        self.assertEqual(candidate["gradient_accumulation_steps"], 2)
        self.assertEqual(candidate["unroll_length"], 12)
        self.assertEqual(candidate["total_steps"], 786_432)
        self.assertEqual(256 * 2 * 12 * 128, candidate["total_steps"])

    def test_parser_rejects_scientific_overrides(self):
        from tools.run_g1_effective512_shac import build_parser

        parser = build_parser()
        for override in (
            ["--num-envs", "512"],
            ["--gradient-accumulation-steps", "4"],
            ["--total-steps", "3145728"],
            ["--unroll-length", "24"],
            ["--actor-lr", "0.001"],
        ):
            arguments = ["--solver-profile", "g1-4x5", *override]
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
