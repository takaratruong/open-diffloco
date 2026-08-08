from pathlib import Path
import unittest

import numpy as np

from tools.analyze_g1_collocation_transfer import build_parser, summarize_warm_start


class G1CollocationTransferTest(unittest.TestCase):
    def test_capture_cli_transports_explicit_reference_contract(self):
        args = build_parser().parse_args(
            [
                "--checkpoint",
                "/tmp/actor.pkl",
                "--output-dir",
                "/tmp/output",
                "--reference-path",
                "/tmp/dance.npz",
                "--reference-stride",
                "1",
            ]
        )

        self.assertEqual(args.reference_path, Path("/tmp/dance.npz"))
        self.assertEqual(args.reference_stride, 1)

    def test_complete_finite_rollout_is_admitted_as_warm_start(self):
        records = np.array(
            [
                [0.08, 0.0, 0.10, 0.08, 0.06, 0.15, 0.3, 0.8],
                [0.09, 0.0, 0.12, 0.07, 0.05, 0.14, 0.4, 0.7],
            ],
            dtype=np.float64,
        )
        errors = np.array(
            [
                [0.10, 0.20, 0.10, 0.10],
                [0.20, 0.30, 0.20, 0.15],
            ],
            dtype=np.float64,
        )
        actions = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=np.float64)
        qpos = np.zeros((2, 4), dtype=np.float64)
        qvel = np.zeros((2, 3), dtype=np.float64)

        summary = summarize_warm_start(
            records=records,
            termination_errors=errors,
            actions=actions,
            qpos=qpos,
            qvel=qvel,
            expected_steps=2,
        )

        self.assertTrue(summary["collocation_warm_start_admitted"])
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["state_knots"], 2)
        self.assertEqual(summary["collocation_intervals"], 1)
        self.assertEqual(summary["terminal_count"], 0)
        self.assertAlmostEqual(
            summary["minimum_normalized_hard_limit_clearance"], 0.2
        )
        self.assertAlmostEqual(summary["fraction_actions_abs_ge_0p95"], 0.25)

    def test_terminal_or_incomplete_rollout_is_rejected(self):
        records = np.array(
            [[0.08, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
            dtype=np.float64,
        )
        errors = np.zeros((1, 4), dtype=np.float64)
        actions = np.zeros((1, 2), dtype=np.float64)
        qpos = np.zeros((1, 4), dtype=np.float64)
        qvel = np.zeros((1, 3), dtype=np.float64)

        summary = summarize_warm_start(
            records=records,
            termination_errors=errors,
            actions=actions,
            qpos=qpos,
            qvel=qvel,
            expected_steps=2,
        )

        self.assertFalse(summary["collocation_warm_start_admitted"])

    def test_nonfinite_trajectory_is_rejected_before_summary(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_warm_start(
                records=np.zeros((1, 8), dtype=np.float64),
                termination_errors=np.zeros((1, 4), dtype=np.float64),
                actions=np.zeros((1, 2), dtype=np.float64),
                qpos=np.array([[np.nan, 0.0]], dtype=np.float64),
                qvel=np.zeros((1, 1), dtype=np.float64),
                expected_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
