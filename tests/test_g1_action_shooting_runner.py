import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import mujoco
import numpy as np

from src.envs.g1_tracking.action_shooting import (
    ForwardGradientReport,
    support_trace_from_states,
)
from src.envs.g1_tracking.fixed_solver import CONVERGENCE_SCAN
from tools.run_g1_action_shooting_gate import (
    CHECKPOINT_SHA256,
    CONFIG_SHA256,
    REFERENCE_SHA256,
    G1PhysicalRuntime,
    PhysicalEvaluation,
    atomic_savez,
    atomic_write_json,
    build_parser,
    classify_gate,
    execute_gate,
    main,
    required_support_transition_present,
    validate_registered_args,
    validate_success_artifacts,
    validate_training_hparams,
    verify_sha256,
)


class G1ActionShootingRunnerTest(unittest.TestCase):
    def test_parser_freezes_registered_defaults(self):
        args = build_parser().parse_args(
            ["--output-dir", "/tmp/evidence", "--code-commit", "a" * 40]
        )

        self.assertEqual(args.start_phase, 105)
        self.assertEqual(args.horizon, 12)
        self.assertEqual(args.solver_iterations, 4)
        self.assertEqual(args.solver_ls_iterations, 5)
        self.assertEqual(args.solver_gradient_semantic, CONVERGENCE_SCAN)
        self.assertEqual(args.gradient_repeat_count, 2)
        self.assertEqual(args.finite_difference_epsilon, 1e-3)
        self.assertEqual(args.action_deviation_weight, 1e-3)
        self.assertEqual(args.max_iterations, 3)
        self.assertEqual(args.trust_radius, 0.02)
        self.assertEqual(args.line_search_alphas, [1.0, 0.5, 0.25, 0.125])
        self.assertEqual(args.minimum_mean_reward_gain, 0.001)
        self.assertEqual(args.reference_sha256, REFERENCE_SHA256)
        self.assertEqual(args.checkpoint_sha256, CHECKPOINT_SHA256)
        self.assertEqual(args.config_sha256, CONFIG_SHA256)
        validate_registered_args(args)
        args.horizon = 11
        with self.assertRaisesRegex(ValueError, "horizon"):
            validate_registered_args(args)

    def test_hash_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"pinned")

            digest = verify_sha256(path, expected=None, label="input")
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_sha256(path, expected="0" * 64, label="input")

    def test_training_hparams_bind_the_unchanged_task(self):
        hparams = {
            "env_variant": "g1_tracking_rmr_50hz_validated",
            "reference_sha256": REFERENCE_SHA256,
            "reference_fps": 50.0,
            "reference_stride": 1,
            "reference_states": 500,
            "reference_transitions": 499,
            "termination_margin_weight": 0.0,
            "actor_history_len": 1,
            "residual_action_scale": 0.0,
        }

        validate_training_hparams(hparams)
        hparams["reference_stride"] = 2
        with self.assertRaisesRegex(ValueError, "reference_stride"):
            validate_training_hparams(hparams)

    def test_support_trace_detects_actual_left_foot_liftoff(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <geom name="floor" type="plane" size="2 2 .1"/>
                <body name="left" pos="-.2 0 0">
                  <freejoint/>
                  <geom name="left_foot_collision" type="sphere" size=".1"/>
                </body>
                <body name="right" pos=".2 0 0">
                  <freejoint/>
                  <geom name="right_foot_collision" type="sphere" size=".1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        qpos = np.zeros((2, model.nq), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[:, 10] = 1.0
        qpos[1, 2] = 1.0
        qvel = np.zeros((2, model.nv), dtype=np.float64)

        trace = support_trace_from_states(model, qpos, qvel)

        np.testing.assert_array_equal(trace.support, [[True, True], [False, True]])
        self.assertEqual(trace.switch_count, 1)

    def test_required_phase_106_transition_is_right_only_to_bilateral(self):
        phases = np.arange(105, 118, dtype=np.int32)
        support = np.ones((13, 2), dtype=bool)
        support[0] = [False, True]

        self.assertTrue(required_support_transition_present(phases, support))
        support[7] = [False, True]
        self.assertFalse(required_support_transition_present(phases, support))
        support[7] = [True, True]
        support[0] = [True, True]
        support[1] = [False, True]
        self.assertFalse(required_support_transition_present(phases, support))

    def test_classification_requires_gradient_contact_and_material_reward(self):
        gradient = {
            "passed": True,
            "maximum_primal_error": 0.0,
            "maximum_repeat_error": 0.0,
            "fd_relative_error": 0.0,
        }
        initial = {
            "mean_reward": 0.08,
            "terminal_count": 0,
            "support_switch_count": 1,
            "required_support_switch_present": True,
        }
        candidate = {
            "mean_reward": 0.0811,
            "terminal_count": 0,
            "support_switch_count": 1,
            "required_support_switch_present": True,
        }

        self.assertEqual(
            classify_gate(gradient, initial, candidate, accepted_steps=1),
            "contact-shooting-authorized",
        )
        candidate["mean_reward"] = 0.0809
        self.assertEqual(
            classify_gate(gradient, initial, candidate, accepted_steps=1),
            "finite-contact-no-material-step",
        )
        initial["required_support_switch_present"] = False
        self.assertEqual(
            classify_gate(gradient, initial, candidate, accepted_steps=1),
            "contact-window-invalid",
        )
        initial["required_support_switch_present"] = True
        initial["support_switch_count"] = 0
        initial["required_support_switch_present"] = False
        self.assertEqual(
            classify_gate(gradient, initial, candidate, accepted_steps=1),
            "contact-window-invalid",
        )
        gradient["passed"] = False
        self.assertEqual(
            classify_gate(gradient, initial, candidate, accepted_steps=1),
            "action-gradient-identity-blocked",
        )

    def test_atomic_artifacts_are_strict_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "summary.json"
            npz_path = root / "rollout.npz"

            atomic_write_json(json_path, {"verdict": "pass", "value": 1.0})
            atomic_savez(
                npz_path, phases=np.arange(106, 118), actions=np.zeros((12, 29))
            )

            self.assertEqual(json.loads(json_path.read_text())["verdict"], "pass")
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertEqual(archive["actions"].shape, (12, 29))
            with self.assertRaises(ValueError):
                atomic_write_json(json_path, {"bad": float("nan")})

    def test_execute_gate_rejects_preexisting_or_symlink_output_root(self):
        class UnusedRuntime:
            def preflight(self):
                raise AssertionError("output validation must happen first")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(existing),
                    "--code-commit",
                    "a" * 40,
                ]
            )
            with self.assertRaisesRegex(ValueError, "pre-existing"):
                execute_gate(args, runtime=UnusedRuntime())

            target = root / "target"
            target.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(target, target_is_directory=True)
            args.output_dir = symlink
            with self.assertRaisesRegex(ValueError, "symlink"):
                execute_gate(args, runtime=UnusedRuntime())

    def test_main_does_not_publish_failure_into_preexisting_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "run_g1_action_shooting_gate.py",
                "--output-dir",
                str(root),
                "--code-commit",
                "a" * 40,
            ]

            with (
                patch("sys.argv", argv),
                self.assertRaisesRegex(ValueError, "pre-existing"),
            ):
                main()

            self.assertEqual(tuple(root.iterdir()), ())

    def test_success_artifact_set_must_be_exact_regular_files(self):
        names = {
            "preflight.json",
            "initial_rollout.npz",
            "gradient_gate.json",
            "optimization_trace.json",
            "candidate_rollout.npz",
            "summary.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                (root / name).write_bytes(b"evidence")

            validate_success_artifacts(root)
            (root / "extra.txt").write_text("unexpected")
            with self.assertRaisesRegex(ValueError, "exactly"):
                validate_success_artifacts(root)

    def test_execute_gate_publishes_complete_bounded_trace_with_fake_runtime(self):
        class FakeRuntime:
            def __init__(self):
                self.nominal = np.zeros((12, 29), dtype=np.float64)

            def preflight(self):
                return {"passed": True, "protocol": "fake"}

            def nominal_actions(self):
                return np.array(self.nominal, copy=True)

            def evaluate(self, actions):
                displacement = float(np.max(actions))
                return PhysicalEvaluation(
                    objective=float(np.mean(np.square(actions - 0.02))),
                    feasible=True,
                    summary={
                        "mean_reward": 0.08 + 0.1 * displacement,
                        "terminal_count": 0,
                        "support_switch_count": 1,
                        "required_support_switch_present": True,
                    },
                    arrays={
                        "phases": np.arange(106, 118),
                        "actions": np.asarray(actions),
                    },
                )

            def gradient_preflight(self, actions):
                del actions
                return {
                    "passed": True,
                    "scalar_jvps": 348,
                    "maximum_primal_error": 0.0,
                    "maximum_repeat_error": 0.0,
                    "fd_relative_error": 0.0,
                }

            def gradient(self, actions):
                return 2.0 * (actions - 0.02) / actions.size

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(root),
                    "--code-commit",
                    "a" * 40,
                ]
            )

            summary = execute_gate(args, runtime=FakeRuntime())

            expected = {
                "preflight.json",
                "initial_rollout.npz",
                "gradient_gate.json",
                "optimization_trace.json",
                "candidate_rollout.npz",
                "summary.json",
            }
            self.assertEqual({path.name for path in root.iterdir()}, expected)
            self.assertEqual(summary["classification"], "contact-shooting-authorized")
            hashed_artifacts = expected - {"summary.json"}
            self.assertEqual(set(summary["artifact_sha256"]), hashed_artifacts)
            for name in hashed_artifacts:
                self.assertEqual(
                    summary["artifact_sha256"][name],
                    hashlib.sha256((root / name).read_bytes()).hexdigest(),
                )
            trace = json.loads((root / "optimization_trace.json").read_text())
            self.assertEqual(len(trace["iterations"]), 3)
            self.assertEqual(trace["accepted_steps"], 3)

    def test_execute_gate_does_not_optimize_invalid_contact_window(self):
        class FakeRuntime:
            def __init__(self):
                self.nominal = np.zeros((12, 29), dtype=np.float64)

            def preflight(self):
                return {"passed": True}

            def nominal_actions(self):
                return self.nominal

            def evaluate(self, actions):
                return PhysicalEvaluation(
                    objective=0.0,
                    feasible=False,
                    summary={
                        "mean_reward": 0.08,
                        "terminal_count": 0,
                        "support_switch_count": 0,
                        "required_support_switch_present": False,
                    },
                    arrays={"actions": np.asarray(actions)},
                )

            def gradient_preflight(self, actions):
                del actions
                return {"passed": True}

            def gradient(self, actions):
                del actions
                raise AssertionError("invalid contact must block optimization")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "evidence"
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--code-commit",
                    "a" * 40,
                ]
            )

            summary = execute_gate(args, runtime=FakeRuntime())

        self.assertEqual(summary["classification"], "contact-window-invalid")
        self.assertEqual(summary["accepted_steps"], 0)

    def test_gradient_preflight_requires_both_fd_support_probes(self):
        runtime = object.__new__(G1PhysicalRuntime)
        runtime.args = SimpleNamespace(
            gradient_repeat_count=2,
            finite_difference_epsilon=1e-3,
            horizon=12,
        )
        runtime.env = SimpleNamespace(action_dim=29)
        runtime._gradient_cache = {}
        runtime._ensure_nominal = lambda: None
        zero_gradient = np.zeros((12, 29), dtype=np.float64)
        runtime._canonical_gradient = lambda actions: ForwardGradientReport(
            value=0.0,
            gradient=zero_gradient,
            scalar_jvps=348,
            maximum_primal_error=0.0,
        )
        runtime._objective = lambda actions: jnp.sum(jnp.square(actions))
        probes = []

        def evaluate(actions):
            probes.append(np.asarray(actions))
            return PhysicalEvaluation(
                objective=0.0,
                feasible=len(probes) == 1,
                summary={},
                arrays={},
            )

        runtime.evaluate = evaluate

        gate = runtime.gradient_preflight(zero_gradient)

        self.assertEqual(len(probes), 2)
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["fd_positive_support_safe"])
        self.assertFalse(gate["fd_negative_support_safe"])


if __name__ == "__main__":
    unittest.main()
