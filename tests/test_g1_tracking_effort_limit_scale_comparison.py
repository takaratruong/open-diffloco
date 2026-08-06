import tempfile
import unittest
from pathlib import Path


ERROR_KEYS = (
    "mean_anchor_position_error",
    "mean_anchor_orientation_error",
    "mean_body_position_error",
    "mean_body_orientation_error",
    "mean_body_linear_velocity_error",
    "mean_body_angular_velocity_error",
)


def candidate(
    *,
    scale,
    nominal_terminals=0,
    shifted_terminals=0,
    reward_delta=-0.001,
    worsened_errors=6,
):
    deltas = {
        key: 0.01 if index < worsened_errors else -0.01
        for index, key in enumerate(ERROR_KEYS)
    }
    deltas["mean_reward"] = reward_delta
    return {
        "effort_limit_scale": scale,
        "aggregate": {
            "nominal": {"terminal_count": nominal_terminals},
            "shifted": {"terminal_count": shifted_terminals},
            "delta_shifted_minus_nominal": deltas,
        },
    }


class G1TrackingEffortLimitScaleComparisonTest(unittest.TestCase):
    def test_cli_preserves_registered_scale_order(self):
        from tools.compare_g1_tracking_effort_limit_scales import build_parser

        args = build_parser().parse_args(
            [
                "--effort-limit-scales",
                "1.0",
                "0.8",
                "0.7",
                "0.6",
                "0.5",
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

        self.assertEqual(
            args.effort_limit_scales,
            [1.0, 0.8, 0.7, 0.6, 0.5],
        )

    def test_gate_includes_exact_material_reward_threshold(self):
        from tools.compare_g1_tracking_effort_limit_scales import (
            candidate_passes,
        )

        self.assertTrue(
            candidate_passes(
                candidate(scale=0.8, reward_delta=-0.001),
                minimum_reward_drop=0.001,
            )
        )

    def test_gate_rejects_terminal_or_only_three_worsened_errors(self):
        from tools.compare_g1_tracking_effort_limit_scales import (
            candidate_passes,
        )

        self.assertFalse(
            candidate_passes(
                candidate(scale=0.8, shifted_terminals=1),
                minimum_reward_drop=0.001,
            )
        )
        self.assertFalse(
            candidate_passes(
                candidate(scale=0.8, worsened_errors=3),
                minimum_reward_drop=0.001,
            )
        )

    def test_selects_first_passing_scale_in_input_order(self):
        from tools.compare_g1_tracking_effort_limit_scales import (
            select_earliest_scale,
        )

        candidates = [
            candidate(scale=0.8, reward_delta=-0.0009),
            candidate(scale=0.7, worsened_errors=4),
            candidate(scale=0.6, worsened_errors=6),
        ]

        selected = select_earliest_scale(
            candidates,
            minimum_reward_drop=0.001,
        )

        self.assertEqual(selected["effort_limit_scale"], 0.7)

    def test_returns_none_when_no_shift_passes(self):
        from tools.compare_g1_tracking_effort_limit_scales import (
            select_earliest_scale,
        )

        candidates = [
            candidate(scale=0.8, reward_delta=-0.0009),
            candidate(scale=0.7, shifted_terminals=1),
            candidate(scale=0.6, worsened_errors=3),
        ]

        self.assertIsNone(
            select_earliest_scale(
                candidates,
                minimum_reward_drop=0.001,
            )
        )

    def test_finite_guard_rejects_nan_and_atomic_writer_replaces_output(self):
        from tools.compare_g1_tracking_effort_limit_scales import (
            _assert_finite_document,
            _write_json_atomically,
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            _assert_finite_document({"nested": [{"value": float("nan")}]})

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            output.write_text('{"old": true}\n')
            _write_json_atomically(output, {"new": True})

            self.assertEqual(output.read_text(), '{\n  "new": true\n}\n')
            self.assertFalse((output.parent / f".{output.name}.tmp").exists())


if __name__ == "__main__":
    unittest.main()
