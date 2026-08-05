import unittest

import numpy as np


class G1TrackingResidualComparisonTest(unittest.TestCase):
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
