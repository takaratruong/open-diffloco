import unittest

import numpy as np


class G1TrackingResidualComparisonTest(unittest.TestCase):
    def test_action_delta_summary_reports_scale_and_saturation(self):
        from tools.compare_g1_tracking_residual import (
            summarize_action_deltas,
        )

        deltas = np.asarray(
            [
                [0.01, -0.02],
                [0.095, -0.10],
            ],
            dtype=np.float64,
        )

        summary = summarize_action_deltas(deltas, bound=0.1)

        self.assertAlmostEqual(summary["mean_abs"], 0.05625)
        self.assertAlmostEqual(
            summary["root_mean_square"],
            float(np.sqrt(np.mean(np.square(deltas)))),
        )
        self.assertAlmostEqual(summary["max_abs"], 0.1)
        self.assertAlmostEqual(summary["fraction_at_95pct_bound"], 0.5)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["action_dim"], 2)

    def test_action_delta_summary_rejects_invalid_evidence(self):
        from tools.compare_g1_tracking_residual import (
            summarize_action_deltas,
        )

        invalid = (
            (np.empty((0, 2)), 0.1),
            (np.asarray([0.1, 0.2]), 0.1),
            (np.asarray([[0.1, np.nan]]), 0.1),
            (np.asarray([[0.1, 0.2]]), 0.0),
        )
        for deltas, bound in invalid:
            with self.subTest(deltas=deltas, bound=bound):
                with self.assertRaises(ValueError):
                    summarize_action_deltas(deltas, bound=bound)

    def test_aggregate_delta_preserves_mean_steps_field(self):
        from tools.compare_g1_tracking_residual import (
            SUMMARY_FIELDS,
            summary_delta,
        )

        baseline = {
            "mean_steps": 37.5,
            **{field: 1.0 for field in SUMMARY_FIELDS},
        }
        candidate = {
            "mean_steps": 38.0,
            **{field: 1.25 for field in SUMMARY_FIELDS},
        }

        delta = summary_delta(candidate, baseline)

        self.assertNotIn("steps", delta)
        self.assertEqual(delta["mean_steps"], 0.5)

    def test_comparison_cli_accepts_a_fixed_body_mass_scale(self):
        from tools.compare_g1_tracking_residual import build_parser

        args = build_parser().parse_args(
            [
                "--checkpoint",
                "/tmp/residual.pkl",
                "--rmr-policy-checkpoint",
                "/tmp/source.pt",
                "--output",
                "/tmp/comparison.json",
                "--phases",
                "0",
                "30",
                "60",
                "90",
                "--body-mass-scale",
                "1.15",
            ]
        )

        self.assertEqual(args.body_mass_scale, 1.15)

    def test_comparison_cli_accepts_a_nominal_source_baseline(self):
        from tools.compare_g1_tracking_residual import build_parser

        args = build_parser().parse_args(
            [
                "--checkpoint",
                "/tmp/residual.pkl",
                "--rmr-policy-checkpoint",
                "/tmp/source.pt",
                "--output",
                "/tmp/comparison.json",
                "--phases",
                "0",
                "30",
                "60",
                "90",
                "--body-mass-scale",
                "1.15",
                "--baseline-body-mass-scale",
                "1.0",
            ]
        )

        self.assertEqual(args.baseline_body_mass_scale, 1.0)

    def test_rollout_summary_reduces_each_registered_metric(self):
        from tools.compare_g1_tracking_residual import summarize_records

        records = np.asarray(
            [
                [0.10, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                [0.20, 1.0, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            ]
        )

        summary = summarize_records(records)

        self.assertEqual(summary["steps"], 2)
        self.assertTrue(summary["terminal"])
        self.assertAlmostEqual(summary["mean_reward"], 0.15)
        self.assertAlmostEqual(summary["mean_anchor_position_error"], 0.02)
        self.assertAlmostEqual(summary["mean_body_angular_velocity_error"], 0.07)


if __name__ == "__main__":
    unittest.main()
