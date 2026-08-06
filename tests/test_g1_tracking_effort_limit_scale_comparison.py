import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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
    def test_main_publishes_effective_provenance_for_every_environment(self):
        import tools.compare_g1_tracking_effort_limit_scales as comparison

        def fake_env(scale):
            return SimpleNamespace(
                body_mass_scale=1.0,
                effort_limit_scale=scale,
                mj_model=SimpleNamespace(
                    opt=SimpleNamespace(iterations=4, ls_iterations=5)
                ),
                reference=SimpleNamespace(
                    qpos=SimpleNamespace(shape=(1, 1))
                ),
            )

        def fake_evaluate_source(
            *,
            env,
            source_policy,
            phases,
            seed,
            max_steps,
            result_key,
        ):
            del source_policy, seed, max_steps
            reward = 1.0 if env.effort_limit_scale == 1.0 else 0.9
            error = 0.0 if env.effort_limit_scale == 1.0 else 0.1
            summary = {
                "steps": 1,
                "terminal": False,
                "mean_reward": reward,
                **{
                    field: error
                    for field in comparison.TRACKING_ERROR_FIELDS
                },
            }
            aggregate = {
                "mean_steps": 1.0,
                "terminal_count": 0,
                "mean_reward": reward,
                **{
                    field: error
                    for field in comparison.TRACKING_ERROR_FIELDS
                },
            }
            return (
                [
                    {
                        "phase": phase,
                        result_key: dict(summary),
                    }
                    for phase in phases
                ],
                aggregate,
            )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint = directory_path / "source.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = directory_path / "comparison.json"
            argv = [
                "compare_g1_tracking_effort_limit_scales.py",
                "--effort-limit-scales",
                "1.0",
                "0.7",
                "--rmr-policy-checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--phases",
                "0",
            ]
            with (
                mock.patch.object(comparison, "configure_jax"),
                mock.patch.object(
                    comparison,
                    "make_evaluation_env",
                    side_effect=[fake_env(1.0), fake_env(0.7)],
                ),
                mock.patch.object(
                    comparison,
                    "load_rmr_policy",
                    return_value=object(),
                ),
                mock.patch.object(
                    comparison,
                    "_evaluate_source",
                    side_effect=fake_evaluate_source,
                ),
                mock.patch.object(sys, "argv", argv),
            ):
                comparison.main()

            document = json.loads(output.read_text())

        self.assertIn("environment", document["nominal"])
        self.assertIn("environment", document["candidates"][0])
        self.assertEqual(
            document["nominal"]["environment"],
            {
                "body_mass_scale": 1.0,
                "effort_limit_scale": 1.0,
                "solver_iterations": 4,
                "solver_ls_iterations": 5,
            },
        )
        self.assertEqual(
            document["candidates"][0]["environment"],
            {
                "body_mass_scale": 1.0,
                "effort_limit_scale": 0.7,
                "solver_iterations": 4,
                "solver_ls_iterations": 5,
            },
        )

    def test_effective_environment_provenance_records_causal_inputs(self):
        import tools.compare_g1_tracking_effort_limit_scales as comparison
        from tools.evaluate_g1_tracking import make_evaluation_env

        self.assertTrue(
            hasattr(comparison, "_effective_environment_provenance"),
            "comparison must record effective environment provenance",
        )
        env = make_evaluation_env(
            "g1_tracking_rmr_50hz_validated",
            solver_iterations=4,
            solver_ls_iterations=5,
            body_mass_scale=1.0,
            effort_limit_scale=0.7,
        )

        provenance = comparison._effective_environment_provenance(
            env,
            expected_effort_limit_scale=0.7,
        )

        self.assertEqual(
            provenance,
            {
                "body_mass_scale": 1.0,
                "effort_limit_scale": 0.7,
                "solver_iterations": 4,
                "solver_ls_iterations": 5,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "effort-limit scale"):
            comparison._effective_environment_provenance(
                env,
                expected_effort_limit_scale=0.8,
            )

    def test_cli_preserves_registered_scale_order(self):
        from tools.compare_g1_tracking_effort_limit_scales import build_parser

        argv = [
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
        args = build_parser().parse_args(argv)

        self.assertEqual(
            args.effort_limit_scales,
            [1.0, 0.8, 0.7, 0.6, 0.5],
        )
        self.assertEqual(args.solver_iterations, 4)
        self.assertEqual(args.solver_ls_iterations, 5)

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                argv
                + [
                    "--solver-iterations",
                    "10",
                    "--solver-ls-iterations",
                    "20",
                ]
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
