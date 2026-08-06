import unittest


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
    step,
    source_terminals=0,
    full_terminals=0,
    reward_delta=0.001,
    improved_errors=6,
):
    deltas = {
        key: -0.01 if index < improved_errors else 0.01
        for index, key in enumerate(ERROR_KEYS)
    }
    deltas["mean_reward"] = reward_delta
    return {
        "step": step,
        "checkpoint": f"/tmp/checkpoint_step_{step:06d}.pkl",
        "aggregate": {
            "source": {"terminal_count": source_terminals},
            "full_policy": {"terminal_count": full_terminals},
            "delta_full_policy_minus_source": deltas,
        },
    }


class G1TrackingFullPolicyCheckpointComparisonTest(unittest.TestCase):
    def test_cli_preserves_checkpoint_order(self):
        from tools.compare_g1_tracking_full_policy_checkpoints import (
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--checkpoints",
                "/tmp/a.pkl",
                "/tmp/b.pkl",
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
            [path.name for path in args.checkpoints],
            ["a.pkl", "b.pkl"],
        )

    def test_selects_first_candidate_meeting_every_strict_gate(self):
        from tools.compare_g1_tracking_full_policy_checkpoints import (
            select_earliest_candidate,
        )

        candidates = [
            candidate(step=3_072, full_terminals=1),
            candidate(step=6_144, reward_delta=0.0),
            candidate(step=9_216, improved_errors=3),
            candidate(step=12_288, improved_errors=4),
            candidate(step=24_576, improved_errors=6),
        ]

        selected = select_earliest_candidate(candidates)

        self.assertEqual(selected["step"], 12_288)

    def test_returns_none_when_no_candidate_passes(self):
        from tools.compare_g1_tracking_full_policy_checkpoints import (
            select_earliest_candidate,
        )

        candidates = [
            candidate(step=3_072, full_terminals=1),
            candidate(step=6_144, reward_delta=-0.001),
            candidate(step=9_216, improved_errors=3),
        ]

        self.assertIsNone(select_earliest_candidate(candidates))


if __name__ == "__main__":
    unittest.main()
