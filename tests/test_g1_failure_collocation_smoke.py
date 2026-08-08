import unittest
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import mujoco
import numpy as np

from tools.smoke_g1_failure_collocation import (
    active_contact_rows,
    build_parser,
    require_identity_equalities,
    summarize_derivative,
)


class G1FailureCollocationSmokeTest(unittest.TestCase):
    def test_cli_help_runs_outside_repository_working_directory(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "smoke_g1_failure_collocation.py"
        )

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_cli_requires_pinned_reference_contract(self):
        args = build_parser().parse_args(
            [
                "--reference-path",
                "/tmp/lafan.npz",
                "--reference-sha256",
                "a" * 64,
                "--checkpoint-path",
                "/tmp/actor.pkl",
                "--config-path",
                "/tmp/hparams.json",
                "--grail-commit",
                "c" * 40,
                "--code-commit",
                "b" * 40,
                "--output",
                "/tmp/smoke.json",
            ]
        )

        self.assertEqual(args.reference_path, Path("/tmp/lafan.npz"))
        self.assertEqual(args.reference_sha256, "a" * 64)
        self.assertEqual(args.checkpoint_path, Path("/tmp/actor.pkl"))
        self.assertEqual(args.config_path, Path("/tmp/hparams.json"))
        self.assertEqual(args.grail_commit, "c" * 40)
        self.assertEqual(args.code_commit, "b" * 40)

    def test_derivative_summary_reports_finite_leaf_norms(self):
        summary = summarize_derivative(
            (jnp.array([3.0, 4.0]), jnp.array([0.0]))
        )

        self.assertTrue(summary["finite"])
        self.assertEqual(summary["leaf_count"], 2)
        self.assertAlmostEqual(summary["l2_norm"], 5.0)

    def test_derivative_summary_hard_fails_nonfinite_leaf(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_derivative(jnp.array([jnp.nan]))

    def test_identity_gate_rejects_nonzero_physical_defect(self):
        require_identity_equalities(np.array([0.0, 5e-9]), atol=1e-8)
        with self.assertRaisesRegex(ValueError, "identity"):
            require_identity_equalities(
                np.array([0.0, 1.9e-2]), atol=1e-8
            )

    def test_active_contact_rows_detects_penetrating_free_body(self):
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            "<geom type='plane' size='1 1 .1'/>"
            "<body><freejoint/><geom type='sphere' size='.1' mass='1'/>"
            "</body></worldbody></mujoco>"
        )
        qpos = np.array([[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]])
        qvel = np.zeros((1, 6))

        self.assertEqual(active_contact_rows(model, qpos, qvel), (0,))


if __name__ == "__main__":
    unittest.main()
