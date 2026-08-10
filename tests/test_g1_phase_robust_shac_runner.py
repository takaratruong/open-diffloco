import unittest
from pathlib import Path


class G1PhaseRobustRunnerTest(unittest.TestCase):
    def test_contract_changes_only_phase_weighting_from_e004(self):
        from tools.run_g1_horizon24_shac import build_horizon24_kwargs
        from tools.run_g1_phase_robust_shac import build_phase_robust_kwargs

        parent = build_horizon24_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        parent.update(
            actor_phase_robust_weighting=False,
            actor_phase_bin_count=5,
            actor_phase_robust_fraction=0.5,
        )
        candidate = build_phase_robust_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        self.assertEqual(set(candidate), set(parent))
        differing = {
            name for name in parent if parent[name] != candidate[name]
        }
        self.assertEqual(differing, {"actor_phase_robust_weighting"})
        self.assertTrue(candidate["actor_phase_robust_weighting"])

    def test_parser_rejects_resume_and_scientific_overrides(self):
        from tools.run_g1_phase_robust_shac import build_parser

        parser = build_parser()
        for arguments in (
            ["--resume-from", "/tmp/checkpoint.pkl"],
            ["--robust-fraction", "1.0"],
            ["--phase-bin-count", "10"],
            ["--total-steps", "8000000"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    parser.parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
