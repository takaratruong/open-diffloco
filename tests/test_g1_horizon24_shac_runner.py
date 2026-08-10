import unittest
from pathlib import Path


class G1Horizon24ShacRunnerTest(unittest.TestCase):
    def test_contract_differs_only_in_horizon_and_bounded_budget(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs
        from tools.run_g1_horizon24_shac import build_horizon24_kwargs

        canonical = build_canonical_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        candidate = build_horizon24_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )

        self.assertEqual(set(candidate), set(canonical))
        differing = {
            name for name in canonical if canonical[name] != candidate[name]
        }
        self.assertEqual(differing, {"total_steps", "unroll_length"})
        self.assertEqual(candidate["total_steps"], 393_216)
        self.assertEqual(candidate["unroll_length"], 24)
        self.assertNotIn("resume_from", candidate)

    def test_parser_rejects_resume_and_scientific_overrides(self):
        from tools.run_g1_horizon24_shac import build_parser

        parser = build_parser()
        for arguments in (
            ["--resume-from", "/tmp/checkpoint.pkl"],
            ["--actor-lr", "0.001"],
            ["--total-steps", "8000000"],
            ["--unroll-length", "12"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
