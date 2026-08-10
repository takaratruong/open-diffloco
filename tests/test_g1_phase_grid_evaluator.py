import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from tools.evaluate_g1_phase_grid import (
    PHASES,
    build_parser,
    build_evaluator_command,
    build_phase_grid_payload,
    classify_phase_grid,
    enrich_phase_summary,
    make_contact_sheet,
    validate_grid,
    validate_phase_summary,
)


class G1PhaseGridEvaluatorTest(unittest.TestCase):
    def test_cli_requires_registered_inputs_and_four_gpu_ids(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--checkpoint",
                "/artifacts/policy.pkl",
                "--reference-path",
                "/artifacts/reference.npz",
                "--phase-zero-dir",
                "/artifacts/phase0",
                "--output-dir",
                "/run/evidence",
                "--gpu-ids",
                "1",
                "2",
                "3",
                "4",
            ]
        )

        self.assertEqual(args.gpu_ids, ["1", "2", "3", "4"])
        self.assertEqual(args.phases, PHASES)
        self.assertEqual(
            args.reference_sha256,
            "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db",
        )
        self.assertEqual(args.solver_profile, "g1-4x5")

    def test_grid_requires_exact_registered_phases_and_four_gpus(self) -> None:
        validate_grid(PHASES, ("1", "2", "3", "4"))

        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_grid((0, 100, 200, 400), ("1", "2", "3", "4"))
        with self.assertRaisesRegex(ValueError, "four GPU"):
            validate_grid(PHASES, ("1", "2", "3"))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_grid(PHASES, ("1", "1", "3", "4"))

    def test_decision_classifies_broad_phase_local_competence(self) -> None:
        decision = classify_phase_grid(
            {0: 80, 100: 45, 200: 50, 300: 40, 400: 99},
            {0: False, 100: False, 200: False, 300: False, 400: True},
        )

        self.assertEqual(decision, "broad-phase-local-competence")

    def test_decision_classifies_immediate_phase_local_difficulty(self) -> None:
        decision = classify_phase_grid(
            {0: 80, 100: 45, 200: 12, 300: 50, 400: 60},
            {phase: False for phase in PHASES},
        )

        self.assertEqual(decision, "phase-local-difficulty")

    def test_decision_classifies_low_median_phase_local_difficulty(self) -> None:
        decision = classify_phase_grid(
            {0: 80, 100: 20, 200: 30, 300: 39, 400: 60},
            {phase: False for phase in PHASES},
        )

        self.assertEqual(decision, "phase-local-difficulty")

    def test_decision_classifies_mixed_evidence(self) -> None:
        decision = classify_phase_grid(
            {0: 80, 100: 35, 200: 40, 300: 50, 400: 55},
            {phase: False for phase in PHASES},
        )

        self.assertEqual(decision, "mixed-evidence")

    def test_child_command_is_the_existing_exact_evaluator(self) -> None:
        command = build_evaluator_command(
            python=Path("/env/python"),
            evaluator=Path("/repo/tools/evaluate_g1_tracking.py"),
            checkpoint=Path("/artifacts/policy_final.pkl"),
            reference=Path("/artifacts/reference.npz"),
            output_dir=Path("/run/phase_100"),
            phase=100,
            solver_profile="g1-4x5",
        )

        self.assertEqual(
            command[:2],
            ["/env/python", "/repo/tools/evaluate_g1_tracking.py"],
        )
        self.assertEqual(
            command[command.index("--checkpoint") + 1],
            "/artifacts/policy_final.pkl",
        )
        self.assertEqual(
            command[command.index("--reference-path") + 1],
            "/artifacts/reference.npz",
        )
        self.assertEqual(
            command[command.index("--output-dir") + 1], "/run/phase_100"
        )
        self.assertEqual(command[command.index("--phase") + 1], "100")
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertIn("g1_tracking_rmr_50hz_source_step", command)
        self.assertEqual(
            command[command.index("--solver-iterations") + 1], "4"
        )
        self.assertEqual(
            command[command.index("--solver-ls-iterations") + 1], "5"
        )
        self.assertEqual(
            command[command.index("--solver-profile") + 1], "g1-4x5"
        )
        self.assertEqual(
            command[command.index("--actor-history-len") + 1], "10"
        )
        self.assertIn("--reference-residual-control", command)
        self.assertNotIn("--random-actor-output-head", command)

    def test_phase_summary_must_match_exact_phase_and_reference(self) -> None:
        summary = {
            "steps": 25,
            "terminal": True,
            "evaluation_start_phase": 100,
            "reference_sha256": "reference-sha",
            "reference_transitions": 499,
            "remaining_reference_transitions": 399,
            "completed_reference_suffix": False,
        }

        validate_phase_summary(summary, phase=100, reference_sha256="reference-sha")
        with self.assertRaisesRegex(ValueError, "phase"):
            validate_phase_summary(summary, phase=200, reference_sha256="reference-sha")
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_phase_summary(summary, phase=100, reference_sha256="wrong")

    def test_aggregate_payload_records_exact_grid_and_decision(self) -> None:
        summaries = {
            phase: {
                "steps": steps,
                "terminal": phase != 400,
                "completed_reference_suffix": phase == 400,
            }
            for phase, steps in zip(PHASES, (80, 45, 50, 40, 99))
        }

        payload = build_phase_grid_payload(
            summaries,
            checkpoint_sha256="checkpoint-sha",
            reference_sha256="reference-sha",
            solver_profile="g1-4x5",
        )

        self.assertEqual(payload["phases"], list(PHASES))
        self.assertEqual(payload["steps"]["100"], 45)
        self.assertEqual(payload["decision"], "broad-phase-local-competence")
        self.assertEqual(payload["checkpoint_sha256"], "checkpoint-sha")
        self.assertEqual(payload["reference_sha256"], "reference-sha")
        self.assertEqual(payload["solver_profile"], "g1-4x5")

    def test_phase_sidecar_records_nominal_replay_free_protocol(self) -> None:
        summary = {"steps": 12}

        enriched = enrich_phase_summary(
            summary,
            solver_profile="upstream-1x5",
            checkpoint_sha256="checkpoint-sha",
        )

        self.assertEqual(enriched["solver_profile"], "upstream-1x5")
        self.assertEqual(enriched["checkpoint_sha256"], "checkpoint-sha")
        self.assertEqual(enriched["randomization"], "disabled-nominal")
        self.assertFalse(enriched["actor_observation_noise"])
        self.assertEqual(enriched["reset_mode"], "exact-reference-phase")

    def test_contact_sheet_is_written_from_finite_rgb_frames(self) -> None:
        frames = [
            np.full((24, 48, 3), value, dtype=np.uint8)
            for value in (0, 64, 128, 255)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contact_sheet.png"

            make_contact_sheet(frames, output)

            image = imageio.imread(output)
            self.assertEqual(image.ndim, 3)
            self.assertEqual(image.shape[-1], 3)
            self.assertGreater(image.shape[0], 24)
            self.assertGreater(image.shape[1], 48)


if __name__ == "__main__":
    unittest.main()
