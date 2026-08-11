"""Contract tests for the frozen E008 paired torso-wrench evaluator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.assertEqual(
            args.checkpoint_sha256,
            "fbea5e272d1431c08753a3600014623cd5577e34e01aeeba18b16af46d369377",
        )
        self.assertEqual(
            args.reference_sha256,
            "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db",
        )
        required = [
            "--checkpoint",
            "/artifacts/e008.pkl",
            "--reference-path",
            "/artifacts/reference.npz",
            "--output",
            "/evidence/oracle.json",
        ]
        for override in (
            ("--seed", "1"),
            ("--phases", "0", "100", "200", "300", "399"),
            ("--assistance-scale", "0.5"),
            ("--solver-profile", "upstream-1x5"),
            ("--checkpoint-sha256", "0" * 64),
            ("--reference-sha256", "0" * 64),
        ):
            with self.subTest(override=override):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args([*required, *override])

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
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                frozen_provenance(
                    checkpoint=checkpoint,
                    reference=reference,
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

        unassisted = {
            phase: {"steps": steps}
            for phase, steps in zip(PHASES, (70, 63, 95, 70, 44), strict=True)
        }
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

        self.assertTrue(passes_oracle_gate(unassisted, assisted, telemetry))
        assisted[200] = {**assisted[200], "steps": 298}
        self.assertFalse(passes_oracle_gate(unassisted, assisted, telemetry))
        assisted[200] = {
            "steps": 299,
            "terminal": False,
            "completed_reference_suffix": True,
        }
        unassisted[300] = {"steps": 69}
        self.assertFalse(passes_oracle_gate(unassisted, assisted, telemetry))

    def test_cap_telemetry_accepts_float32_cap_rounding(self) -> None:
        from src.evaluation.g1_torso_wrench_oracle import TorsoWrenchParameters
        from tools.evaluate_g1_frozen_torso_wrench_oracle import (
            summarize_wrench_trace,
        )

        parameters = TorsoWrenchParameters(
            nominal_total_mass=1.0,
            gravity_magnitude=100.0,
        )
        trace = np.array(
            [[100.000005, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6],
            dtype=np.float64,
        )

        summary = summarize_wrench_trace(trace, parameters=parameters, dt=0.02)

        self.assertTrue(summary["force_cap_compliant"])

    def test_frozen_e008_environment_contract_restores_delta_preview_layout(
        self,
    ) -> None:
        from tools.evaluate_g1_frozen_torso_wrench_oracle import (
            frozen_e008_environment_kwargs,
        )

        kwargs = frozen_e008_environment_kwargs()

        self.assertEqual(kwargs["actor_history_len"], 10)
        self.assertEqual(kwargs["actor_reference_lookahead_steps"], (4, 8, 12))
        self.assertEqual(kwargs["actor_reference_preview_mode"], "delta")
        self.assertTrue(kwargs["reference_residual_control"])
        self.assertEqual(kwargs["reference_residual_scale"], 0.5)

    def test_composite_e008_checkpoint_uses_residual_action_path(self) -> None:
        import jax

        from src.algorithms.shac.residual_preview_adapter import (
            FrozenPreviewResidualParams,
            PreviewResidualAdapter,
        )
        from src.core.networks import Actor
        from tools.evaluate_g1_frozen_torso_wrench_oracle import (
            evaluate_frozen_e008_action,
            load_frozen_e008_policy,
        )

        environment = SimpleNamespace(
            action_dim=2,
            actor_obs_dim=3280,
            actor_frame_obs_dim=328,
            squash_actor_actions=True,
        )
        actor = Actor(
            2,
            hidden=(512, 256, 128),
            squash=True,
            layer_norm=True,
            zero_output=False,
        )
        observations = jp.zeros((1, 3280), dtype=jp.float32)
        residual = PreviewResidualAdapter(action_dim=2, hidden_dim=256)
        composite = FrozenPreviewResidualParams(
            parent=actor.init(jax.random.PRNGKey(1), observations),
            adapter=residual.init(
                jax.random.PRNGKey(2), jp.zeros((1, 328), dtype=jp.float32)
            ),
        )
        checkpoint_state = SimpleNamespace(
            actor_params=composite,
            normalizer=SimpleNamespace(),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "e008.pkl"
            with checkpoint.open("wb") as stream:
                pickle.dump(checkpoint_state, stream)
            loaded_actor, loaded_params, loaded_residual, normalizer = (
                load_frozen_e008_policy(environment, checkpoint)
            )

        self.assertIsInstance(loaded_params, FrozenPreviewResidualParams)
        self.assertEqual(loaded_residual.hidden_dim, 256)
        self.assertIsInstance(normalizer, SimpleNamespace)
        action = evaluate_frozen_e008_action(
            loaded_actor,
            loaded_params,
            observations,
            residual_actor=loaded_residual,
            treatment_frame_dim=environment.actor_frame_obs_dim,
        )
        self.assertEqual(action.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
