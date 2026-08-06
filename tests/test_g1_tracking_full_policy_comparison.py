import unittest


class G1TrackingFullPolicyComparisonTest(unittest.TestCase):
    def test_cli_requires_full_and_source_checkpoints(self):
        from tools.compare_g1_tracking_full_policy import build_parser

        args = build_parser().parse_args(
            [
                "--checkpoint",
                "/tmp/full.pkl",
                "--rmr-policy-checkpoint",
                "/tmp/source.pt",
                "--output",
                "/tmp/comparison.json",
                "--phases",
                "0",
                "30",
                "60",
                "90",
            ]
        )

        self.assertEqual(args.checkpoint.name, "full.pkl")
        self.assertEqual(args.phases, [0, 30, 60, 90])


if __name__ == "__main__":
    unittest.main()
