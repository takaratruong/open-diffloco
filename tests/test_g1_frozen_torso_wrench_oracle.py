"""Contract tests for the frozen E008 paired torso-wrench evaluator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jp
import numpy as np


@dataclass(frozen=True)
class _Data:
    xfrc_applied: jp.ndarray

    def replace(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class _State:
    data: _Data

    def replace(self, **changes):
        return replace(self, **changes)


class G1FrozenTorsoWrenchOracleTest(unittest.TestCase):
    def test_parser_defaults_to_the_frozen_five_phase_protocol(self) -> None:
        from tools.evaluate_g1_frozen_torso_wrench_oracle import (
            EXPECTED_SUFFIX_TRANSITIONS,
            PHASES,
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--checkpoint",
                "/artifacts/e008.pkl",
                "--reference-path",
                "/artifacts/reference.npz",
                "--output",
                "/evidence/oracle.json",
            ]
        )

        self.assertEqual(args.phases, PHASES)
        self.assertEqual(EXPECTED_SUFFIX_TRANSITIONS, (499, 399, 299, 199, 99))
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.solver_profile, "g1-4x5")
        self.assertEqual(args.assistance_scale, 1.0)

    def test_paired_reset_reuses_one_identical_exact_reference_state(self) -> None:
        from tools.evaluate_g1_frozen_torso_wrench_oracle import paired_reset

        class Environment:
            def __init__(self):
                self.calls = []

            def reset_at_phase(self, key, difficulty, phase):
                self.calls.append((key, difficulty, phase))
                return _State(_Data(jp.zeros((3, 6))))

        env = Environment()
        unassisted, assisted = paired_reset(env, phase=100, seed=7)

        self.assertIs(unassisted, assisted)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(int(env.calls[0][2]), 100)

    def test_disabled_injection_overwrites_a_stale_torso_row_with_exact_zeros(
        self,
    ) -> None:
        from tools.evaluate_g1_frozen_torso_wrench_oracle import inject_wrench

        state = _State(_Data(jp.full((4, 6), 3.0)))
        injected = inject_wrench(
            state,
            torso_body_id=2,
            world_wrench=jp.zeros(6),
        )

        np.testing.assert_array_equal(
            np.asarray(injected.data.xfrc_applied[2]), np.zeros(6)
        )
        np.testing.assert_array_equal(
            np.asarray(injected.data.xfrc_applied[1]), np.full(6, 3.0)
        )

    def test_wrench_telemetry_reports_phase_force_torque_and_absolute_work(
        self,
    ) -> None:
        from src.evaluation.g1_torso_wrench_oracle import TorsoWrenchParameters
        from tools.evaluate_g1_frozen_torso_wrench_oracle import (
            summarize_wrench_trace,
        )

        parameters = TorsoWrenchParameters(
            nominal_total_mass=1.0,
            gravity_magnitude=100.0,
        )
        trace = np.array(
            [
                [3.0, 4.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )

        summary = summarize_wrench_trace(trace, parameters=parameters, dt=0.02)

        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["max_force"], 5.0)
        self.assertAlmostEqual(summary["rms_force"], np.sqrt(12.5))
        self.assertEqual(summary["max_torque"], 2.0)
        self.assertAlmostEqual(summary["rms_torque"], np.sqrt(2.5))
        self.assertEqual(summary["absolute_wrench_power"], 11.0)
        self.assertAlmostEqual(summary["absolute_wrench_work"], 0.22)
        self.assertTrue(summary["finite"])
        self.assertTrue(summary["force_cap_compliant"])
        self.assertTrue(summary["torque_cap_compliant"])

    def test_provenance_hashes_are_immutable_and_must_match_requested_inputs(
        self,
    ) -> None:
        from tools.evaluate_g1_frozen_torso_wrench_oracle import frozen_provenance

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "e008.pkl"
            reference = Path(directory) / "reference.npz"
            checkpoint.write_bytes(b"frozen-checkpoint")
            reference.write_bytes(b"frozen-reference")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()

            provenance = frozen_provenance(
                checkpoint=checkpoint,
                reference=reference,
                expected_checkpoint_sha256=checkpoint_hash,
                expected_reference_sha256=reference_hash,
                solver_profile="g1-4x5",
                torso_body_id=16,
            )

            self.assertEqual(provenance["checkpoint_sha256"], checkpoint_hash)
            self.assertEqual(provenance["reference_sha256"], reference_hash)
            self.assertEqual(provenance["torso_body_id"], 16)
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                frozen_provenance(
                    checkpoint=checkpoint,
                    reference=reference,
                    expected_checkpoint_sha256="0" * 64,
                    expected_reference_sha256=reference_hash,
                    solver_profile="g1-4x5",
                    torso_body_id=16,
                )

    def test_pass_gate_requires_every_exact_suffix_and_clean_wrench_evidence(
        self,
    ) -> None:
        from tools.evaluate_g1_frozen_torso_wrench_oracle import (
            EXPECTED_SUFFIX_TRANSITIONS,
            PHASES,
            passes_oracle_gate,
        )

        assisted = {
            phase: {
                "steps": steps,
                "terminal": False,
                "completed_reference_suffix": True,
            }
            for phase, steps in zip(PHASES, EXPECTED_SUFFIX_TRANSITIONS, strict=True)
        }
        telemetry = {
            phase: {
                "finite": True,
                "force_cap_compliant": True,
                "torque_cap_compliant": True,
            }
            for phase in PHASES
        }

        self.assertTrue(passes_oracle_gate(assisted, telemetry))
        assisted[200] = {**assisted[200], "steps": 298}
        self.assertFalse(passes_oracle_gate(assisted, telemetry))


if __name__ == "__main__":
    unittest.main()
