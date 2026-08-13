import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np


class G1RmrActionSpaceParityRunnerTest(unittest.TestCase):
    def test_decoupled_early_learning_kwargs_change_only_bounded_budget(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_decoupled_early_learning_kwargs,
            build_decoupled_exploration_kwargs,
        )

        full = build_decoupled_exploration_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=3
        )
        early = build_decoupled_early_learning_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=3
        )

        differing = {
            key
            for key in full
            if not np.array_equal(np.asarray(full[key]), np.asarray(early[key]))
        }
        self.assertEqual(
            differing,
            {
                "total_steps",
                "curriculum_grace",
                "curriculum_steps",
            },
        )
        self.assertEqual(early["total_steps"], 98_304)
        self.assertEqual(early["checkpoint_interval"], 98_304)
        self.assertEqual(early["curriculum_grace"], 98_304)
        self.assertEqual(early["curriculum_steps"], 1)

    def test_decoupled_kwargs_bound_mean_without_clipping_noise(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_decoupled_exploration_kwargs,
        )

        kwargs = build_decoupled_exploration_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=3
        )

        self.assertEqual(
            kwargs["env_variant"],
            "g1_tracking_rmr_50hz_decoupled_exploration",
        )
        self.assertEqual(kwargs["reference_residual_scale"], 1.0)
        self.assertEqual(kwargs["action_noise_std_start"], 1.0)

    def test_decoupled_gate_validator_requires_bounded_h12_action_tape(self):
        from src.core.rmr_action_noise import RMR_ACTION_STD
        from tools.run_g1_rmr_action_space_parity import validate_gate_artifacts

        with TemporaryDirectory() as directory:
            run = Path(directory)
            hparams = {
                "total_steps": 6_144,
                "env_variant": "g1_tracking_rmr_50hz_decoupled_exploration",
                "squash_actor_actions": False,
                "squash_actor_mean": True,
                "clip_sampled_actor_actions": False,
                "actor_observation_noise": False,
                "reference_reset_noise_scale": 0.0,
                "reference_residual_control": True,
                "reference_residual_scale": 1.0,
                "kp_range": [35.0, 35.0],
                "kd_range": [0.5, 0.5],
                "friction_range": [1.0, 1.0],
                "mass_range": [1.0, 1.0],
                "com_offset_range": [0.0, 0.0, 0.0],
                "domain_randomization": False,
                "randomization_com_body_name": "torso_link",
                "randomization_uses_curriculum": False,
                "push_velocity_range": [0.0, 0.0],
                "action_noise_std_start": 1.0,
                "action_noise_std_end": np.asarray(RMR_ACTION_STD).tolist(),
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
            }
            (run / "hparams.json").write_text(json.dumps(hparams))
            (run / "checkpoint_step_006144.pkl").write_bytes(b"checkpoint")
            (run / "checkpoint_phase_metrics.json").write_text(
                json.dumps([{"step": 6_144, "actor_cagrad_valid": True,
                             "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
                             "actor_cagrad_combined_norm": 2.0}])
            )
            (run / "diag_log.json").write_text(
                json.dumps([{"actor_grad": 2.0, "actor_update_norm": 0.1}])
            )
            tape = run / "gate_training_rollout"
            tape.mkdir()
            np.savez_compressed(
                tape / "training_action_noise.npz",
                action_mean=np.zeros((12, 29)),
                epsilon=np.ones((12, 29)),
                action_std=np.ones(29),
                noisy_action=np.ones((12, 29)),
                effective_action=np.ones((12, 29)),
            )
            from tools.prepare_g1_rmr_reference import sha256_file

            (tape / "summary.json").write_text(
                json.dumps(
                    {
                        "training_distribution_rollout": True,
                        "training_checkpoint_step": 6_144,
                        "training_exact_reset_phase": 0,
                        "checkpoint_sha256": sha256_file(
                            run / "checkpoint_step_006144.pkl"
                        ),
                        "steps": 12,
                    }
                )
            )

            result = validate_gate_artifacts(
                run,
                env_variant="g1_tracking_rmr_50hz_decoupled_exploration",
            )
            self.assertTrue(result["bounded_mean_rollout_valid"])

            np.savez_compressed(
                tape / "training_action_noise.npz",
                action_mean=np.full((12, 29), 1.1),
                epsilon=np.ones((12, 29)),
                action_std=np.ones(29),
                noisy_action=np.ones((12, 29)),
                effective_action=np.ones((12, 29)),
            )
            with self.assertRaisesRegex(ValueError, "actor mean"):
                validate_gate_artifacts(
                    run,
                    env_variant=(
                        "g1_tracking_rmr_50hz_decoupled_exploration"
                    ),
                )

    def test_fresh_parity_kwargs_use_linear_full_scale_nominal_gains(self):
        from src.core.rmr_action_noise import RMR_ACTION_STD
        from tools.run_g1_rmr_action_space_parity import (
            build_rmr_action_space_parity_kwargs,
        )

        kwargs = build_rmr_action_space_parity_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=7
        )

        self.assertEqual(
            kwargs["env_variant"], "g1_tracking_rmr_50hz_action_parity"
        )
        self.assertTrue(kwargs["reference_residual_control"])
        self.assertEqual(kwargs["reference_residual_scale"], 1.0)
        self.assertEqual(kwargs["action_scale"], 1.0)
        self.assertEqual(kwargs["kp_range"], (35.0, 35.0))
        self.assertEqual(kwargs["kd_range"], (0.5, 0.5))
        self.assertFalse(kwargs["domain_randomization"])
        self.assertEqual(kwargs["friction_range"], (1.0, 1.0))
        self.assertEqual(kwargs["mass_range"], (1.0, 1.0))
        self.assertEqual(kwargs["com_offset_range"], (0.0, 0.0, 0.0))
        self.assertEqual(kwargs["push_velocity_range"], (0.0, 0.0))
        self.assertEqual(kwargs["push_interval_s"], 2.0)
        self.assertFalse(kwargs["actor_observation_noise"])
        self.assertEqual(kwargs["reference_reset_noise_scale"], 0.0)
        self.assertEqual(kwargs["action_noise_std_start"], 1.0)
        np.testing.assert_array_equal(
            kwargs["action_noise_std_end"], RMR_ACTION_STD
        )
        self.assertEqual(kwargs["action_noise_schedule_steps"], 800_000)
        self.assertEqual(kwargs["total_steps"], 786_432)
        self.assertEqual(kwargs["checkpoint_interval"], 98_304)
        self.assertNotIn("resume_from", kwargs)

    def test_fresh_parity_kwargs_keep_proven_shac_recipe(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_rmr_action_space_parity_kwargs,
        )

        kwargs = build_rmr_action_space_parity_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=11
        )

        self.assertEqual(kwargs["unroll_length"], 12)
        self.assertEqual(kwargs["num_envs"], 256)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 2)
        self.assertTrue(kwargs["actor_cagrad"])
        self.assertEqual(kwargs["actor_phase_bin_count"], 5)
        self.assertEqual(kwargs["actor_reference_lookahead_steps"], (4, 8, 12))
        self.assertEqual(kwargs["solver_iterations"], 4)
        self.assertEqual(kwargs["solver_ls_iterations"], 5)
        self.assertEqual(kwargs["reference_stride"], 1)
        self.assertEqual(kwargs["seed"], 11)

    def test_rmr_ranges_fix_mass_and_gains_but_randomize_friction_and_com(self):
        import jax
        import jax.numpy as jnp

        from src.envs.g1_tracking.randomization import (
            G1RandomizationRanges,
            sample_g1_randomization,
        )

        ranges = G1RandomizationRanges(
            friction=(0.3, 1.6),
            mass=(1.0, 1.0),
            kp_scale=(1.0, 1.0),
            kd_scale=(1.0, 1.0),
            com_offset=(0.025, 0.05, 0.05),
        )
        keys = jax.random.split(jax.random.PRNGKey(5), 32)
        samples = jax.vmap(
            lambda key: sample_g1_randomization(key, jnp.array(1.0), ranges)
        )(keys)

        np.testing.assert_array_equal(samples["kp_scale"], np.ones(32))
        np.testing.assert_array_equal(samples["kd_scale"], np.ones(32))
        np.testing.assert_array_equal(samples["mass_scale"], np.ones(32))
        self.assertGreater(np.ptp(np.asarray(samples["friction_scale"])), 0.5)
        self.assertGreater(np.ptp(np.asarray(samples["com_offset"])), 0.02)

    def test_gate_kwargs_change_only_budget_and_curriculum_endpoint(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_parity_gate_kwargs,
            build_rmr_action_space_parity_kwargs,
        )

        full = build_rmr_action_space_parity_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=13
        )
        gate = build_parity_gate_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=13
        )

        differing = set()
        for key in full:
            if hasattr(full[key], "shape") and tuple(full[key].shape):
                np.testing.assert_array_equal(full[key], gate[key])
            elif full[key] != gate[key]:
                differing.add(key)
        self.assertEqual(
            differing,
            {"total_steps", "checkpoint_interval", "curriculum_grace", "curriculum_steps"},
        )
        self.assertEqual(gate["total_steps"], 6_144)
        self.assertEqual(gate["checkpoint_interval"], 6_144)
        self.assertEqual(gate["curriculum_grace"], 6_144)
        self.assertEqual(gate["curriculum_steps"], 1)

    def test_parser_does_not_accept_a_resume_checkpoint(self):
        from tools.run_g1_rmr_action_space_parity import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--solver-profile",
                    "g1-4x5",
                    "--code-commit",
                    "0" * 40,
                    "--resume-from",
                    "/tmp/old.pkl",
                ]
            )

    def test_parser_accepts_an_explicit_one_update_gate(self):
        from tools.run_g1_rmr_action_space_parity import build_parser

        args = build_parser().parse_args(
            [
                "--solver-profile",
                "g1-4x5",
                "--code-commit",
                "0" * 40,
                "--gate-only",
            ]
        )

        self.assertTrue(args.gate_only)

    def test_parser_accepts_early_learning_only_with_decoupled_exploration(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_parser,
            validate_mode_args,
        )

        args = build_parser().parse_args(
            [
                "--solver-profile",
                "g1-4x5",
                "--code-commit",
                "0" * 40,
                "--early-learning-gate",
                "--decoupled-exploration",
            ]
        )
        validate_mode_args(args)
        self.assertTrue(args.early_learning_gate)

        missing_decoupling = build_parser().parse_args(
            [
                "--solver-profile",
                "g1-4x5",
                "--code-commit",
                "0" * 40,
                "--early-learning-gate",
            ]
        )
        with self.assertRaisesRegex(ValueError, "decoupled"):
            validate_mode_args(missing_decoupling)

    def test_parser_rejects_both_gate_modes(self):
        from tools.run_g1_rmr_action_space_parity import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--solver-profile",
                    "g1-4x5",
                    "--code-commit",
                    "0" * 40,
                    "--gate-only",
                    "--early-learning-gate",
                ]
            )

    def test_preflight_binds_clean_code_reference_and_runtime_assets(self):
        from tools.run_g1_rmr_action_space_parity import (
            EXPECTED_REFERENCE_SHA256,
            validate_preflight,
        )

        repository = Path("/tmp/repository")
        reference = Path("/tmp/dance.npz")
        commit = "a" * 40
        with (
            mock.patch(
                "tools.run_g1_rmr_action_space_parity._git_output",
                side_effect=(commit, ""),
            ),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch(
                "tools.run_g1_rmr_action_space_parity.sha256_file",
                return_value=EXPECTED_REFERENCE_SHA256,
            ),
            mock.patch(
                "tools.run_g1_rmr_action_space_parity.validate_runtime_assets",
                return_value={
                    "model_sha256": "model",
                    "controller_sha256": "controller",
                },
            ),
        ):
            result = validate_preflight(
                repository=repository,
                reference_path=reference,
                code_commit=commit,
            )

        self.assertEqual(result["code_commit"], commit)
        self.assertEqual(result["reference_sha256"], EXPECTED_REFERENCE_SHA256)
        self.assertEqual(result["model_sha256"], "model")
        self.assertEqual(result["controller_sha256"], "controller")
        self.assertEqual(result["reference_residual_scale"], 1.0)
        self.assertFalse(result["normalized_action_clip"])
        self.assertEqual(
            result["environment_variant"],
            "g1_tracking_rmr_50hz_action_parity",
        )
        self.assertEqual(result["joint_velocity_observation_noise"], 0.0)
        self.assertTrue(result["exact_reference_resets"])
        self.assertEqual(result["randomization_com_body_name"], "torso_link")
        self.assertFalse(result["randomization_uses_curriculum"])
        self.assertFalse(result["domain_randomization"])
        self.assertEqual(result["com_offset_range"], [0.0, 0.0, 0.0])
        self.assertEqual(result["push_velocity_range"], [0.0, 0.0])
        self.assertEqual(
            result["remaining_rmr_randomization_gaps"],
            [
                "friction-and-restitution-material-buckets",
                "joint-default-position-offsets",
                "pushes-disabled",
                "torso-com-randomization-disabled",
            ],
        )
        self.assertEqual(result["kp_range"], [35.0, 35.0])
        self.assertEqual(result["kd_range"], [0.5, 0.5])

    def test_preflight_rejects_a_dirty_worktree(self):
        from tools.run_g1_rmr_action_space_parity import validate_preflight

        commit = "b" * 40
        with (
            mock.patch(
                "tools.run_g1_rmr_action_space_parity._git_output",
                side_effect=(commit, " M src/file.py"),
            ),
            self.assertRaisesRegex(ValueError, "clean"),
        ):
            validate_preflight(
                repository=Path("/tmp/repository"),
                reference_path=Path("/tmp/dance.npz"),
                code_commit=commit,
            )

    def test_execute_runs_the_one_update_gate_after_preflight(self):
        from tools import run_g1_rmr_action_space_parity as runner

        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            args = Namespace(
                solver_profile="g1-4x5",
                reference_path=Path("/tmp/dance.npz"),
                seed=17,
                output_root=output_root,
                code_commit="c" * 40,
                gate_only=True,
            )
            with (
                mock.patch.object(
                    runner,
                    "validate_preflight",
                    return_value={"valid": True},
                ) as preflight,
                mock.patch.object(runner, "configure_jax"),
                mock.patch.object(
                    runner, "get_solver_profile", return_value=object()
                ),
                mock.patch.object(runner, "solver_context") as context,
                mock.patch.object(
                    runner,
                    "train",
                    return_value=(None, "training_runs/gate"),
                ) as train,
                mock.patch.object(
                    runner,
                    "validate_gate_artifacts",
                    return_value={"valid": True, "step": 6_144},
                ) as validate_gate,
                mock.patch.object(runner, "_write_json_atomically") as write,
            ):
                context.return_value.__enter__.return_value = None
                result = runner.execute(args)

        preflight.assert_called_once()
        self.assertEqual(
            write.call_args_list,
            [
                mock.call(
                    output_root / "action_space_parity_preflight.json",
                    {"valid": True},
                ),
                mock.call(
                    output_root / "action_space_parity_gate_validation.json",
                    {"valid": True, "step": 6_144},
                ),
            ],
        )
        validate_gate.assert_called_once_with(
            (output_root / "training_runs/gate").resolve()
        )
        self.assertEqual(train.call_args.kwargs["total_steps"], 6_144)
        self.assertEqual(train.call_args.kwargs["reference_residual_scale"], 1.0)
        self.assertEqual(result, (output_root / "training_runs/gate").resolve())

    def test_gate_validation_requires_the_executed_parity_contract(self):
        from src.core.rmr_action_noise import RMR_ACTION_STD
        from tools.run_g1_rmr_action_space_parity import validate_gate_artifacts

        with TemporaryDirectory() as directory:
            run = Path(directory)
            hparams = {
                "total_steps": 6_144,
                "env_variant": "g1_tracking_rmr_50hz_action_parity",
                "squash_actor_actions": False,
                "actor_observation_noise": False,
                "reference_reset_noise_scale": 0.0,
                "reference_residual_control": True,
                "reference_residual_scale": 1.0,
                "kp_range": [35.0, 35.0],
                "kd_range": [0.5, 0.5],
                "friction_range": [1.0, 1.0],
                "mass_range": [1.0, 1.0],
                "com_offset_range": [0.0, 0.0, 0.0],
                "domain_randomization": False,
                "randomization_com_body_name": "torso_link",
                "randomization_uses_curriculum": False,
                "push_velocity_range": [0.0, 0.0],
                "action_noise_std_start": 1.0,
                "action_noise_std_end": np.asarray(RMR_ACTION_STD).tolist(),
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
            }
            (run / "hparams.json").write_text(json.dumps(hparams))
            (run / "checkpoint_step_006144.pkl").write_bytes(b"checkpoint")
            (run / "checkpoint_phase_metrics.json").write_text(
                json.dumps(
                    [
                        {
                            "step": 6_144,
                            "action_noise_current": 1.0,
                            "actor_cagrad_valid": True,
                            "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
                            "actor_cagrad_combined_norm": 2.0,
                        }
                    ]
                )
            )
            (run / "diag_log.json").write_text(
                json.dumps(
                    [{"step": 6_144, "actor_grad": 2.0, "actor_update_norm": 0.1}]
                )
            )

            result = validate_gate_artifacts(run)
            self.assertTrue(result["valid"])

            hparams["squash_actor_actions"] = True
            (run / "hparams.json").write_text(json.dumps(hparams))
            with self.assertRaisesRegex(ValueError, "squash"):
                validate_gate_artifacts(run)


if __name__ == "__main__":
    unittest.main()
