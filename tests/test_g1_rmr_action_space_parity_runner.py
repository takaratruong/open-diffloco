import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np


def _write_early_learning_fixture(run: Path) -> None:
    from src.core.rmr_action_noise import RMR_ACTION_STD
    from tools.prepare_g1_rmr_reference import sha256_file

    hparams = {
        "total_steps": 98_304,
        "env_variant": "g1_tracking_rmr_50hz_upstream_boundary",
        "squash_actor_actions": False,
        "squash_actor_mean": True,
        "clip_sampled_actor_actions": True,
        "actor_observation_noise": False,
        "reference_reset_noise_scale": 0.0,
        "reference_residual_control": True,
        "reference_residual_scale": 0.5,
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "friction_range": [1.0, 1.0],
        "mass_range": [1.0, 1.0],
        "com_offset_range": [0.0, 0.0, 0.0],
        "domain_randomization": False,
        "randomization_com_body_name": "torso_link",
        "randomization_uses_curriculum": False,
        "push_velocity_range": [0.0, 0.0],
        "action_noise_std_start": np.asarray(RMR_ACTION_STD).tolist(),
        "action_noise_std_end": np.asarray(RMR_ACTION_STD).tolist(),
        "action_noise_schedule_steps": 800_000,
        "actor_cagrad": True,
        "actor_phase_bin_count": 5,
        "gradient_accumulation_steps": 2,
        "num_envs": 256,
        "unroll_length": 12,
        "actor_reference_lookahead_steps": [4, 8, 12],
        "reference_path": "/tmp/dance.npz",
        "reference_stride": 1,
        "actor_history_len": 10,
        "actor_reference_preview_mode": "absolute",
        "solver_profile": "g1-4x5",
        "seed": 0,
        "effort_limit_scale": 1.0,
        "terrain": False,
        "torso_wrench_assistance": False,
        "actor_observe_torso_wrench_assistance": False,
        "actor_torso_wrench_assistance_conditioning": False,
        "curriculum_grace": 98_304,
        "curriculum_steps": 1,
        "actor_lr": 5e-3,
        "actor_layer_norm": True,
        "actor_hidden": [512, 256, 128],
    }
    (run / "hparams.json").write_text(json.dumps(hparams))
    checkpoint = run / "checkpoint_step_098304.pkl"
    checkpoint.write_bytes(b"checkpoint")
    (run / "checkpoint_phase_metrics.json").write_text(
        json.dumps(
            [
                {
                    "step": 98_304,
                    "actor_cagrad_valid": True,
                    "actor_cagrad_bin_counts": [1, 2, 3, 4, 5],
                    "actor_cagrad_combined_norm": 2.0,
                }
            ]
        )
    )
    (run / "diag_log.json").write_text(
        json.dumps(
            [
                {
                    "step": 6_144,
                    "actor_grad": 2.0,
                    "actor_update_norm": 0.1,
                },
                {
                    "step": 67_584,
                    "actor_grad": 1.5,
                    "actor_update_norm": 0.08,
                },
            ]
        )
    )
    evidence = run / "early_learning_evidence" / "checkpoint_step_098304"
    noisy = evidence / "noisy"
    clean = evidence / "clean"
    noisy.mkdir(parents=True)
    clean.mkdir(parents=True)
    checkpoint_sha = sha256_file(checkpoint)
    common = {
        "checkpoint_sha256": checkpoint_sha,
        "reference_sha256": (
            "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
        ),
        "solver_profile": "g1-4x5",
        "evaluation_start_phase": 0,
    }
    (noisy / "summary.json").write_text(
        json.dumps(
            {
                **common,
                "steps": 120,
                "training_distribution_rollout": True,
                "training_observation_noise": False,
                "training_exact_reset_phase": 0,
                "training_checkpoint_step": 98_304,
            }
        )
    )
    (clean / "summary.json").write_text(
        json.dumps(
            {
                **common,
                "steps": 45,
                "training_distribution_rollout": False,
            }
        )
    )
    mean = np.full((120, 29), 0.1)
    epsilon = np.full((120, 29), 10.0)
    std = np.asarray(RMR_ACTION_STD, dtype=np.float64)
    np.savez_compressed(
        noisy / "training_action_noise.npz",
        action_mean=mean,
        epsilon=epsilon,
        action_std=std,
        noisy_action=mean + epsilon * std,
        effective_action=np.clip(mean + epsilon * std, -1.0, 1.0),
    )
    columns = np.asarray(["step", "done", "terminal"])
    values = np.zeros((120, 3))
    values[:, 0] = np.arange(120)
    values[51, 1:] = 1.0
    np.savez_compressed(
        noisy / "evaluation.npz", columns=columns, values=values
    )
    clean_values = np.zeros((45, 3))
    clean_values[:, 0] = np.arange(45)
    clean_values[-1, 1:] = 1.0
    np.savez_compressed(
        clean / "evaluation.npz",
        columns=columns,
        values=clean_values,
    )
    for path in (
        noisy / "training_rollout.mp4",
        noisy / "contact_sheet.png",
        clean / "evaluation.mp4",
        clean / "contact_sheet.png",
    ):
        path.write_bytes(b"evidence")


class G1RmrActionSpaceParityRunnerTest(unittest.TestCase):
    def test_early_learning_commands_render_noisy_training_and_clean_rollouts(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_early_learning_rollout_commands,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)
            noisy, clean = build_early_learning_rollout_commands(
                repository=Path("/repo"), run_directory=run
            )

        self.assertIn("--training-distribution-rollout", noisy)
        self.assertIn("--continue-training-after-terminal", noisy)
        self.assertEqual(noisy[noisy.index("--max-steps") + 1], "120")
        self.assertIn("--exact-training-reset-phase", noisy)
        self.assertNotIn("--training-distribution-rollout", clean)
        self.assertEqual(clean[clean.index("--phase") + 1], "0")

    def test_early_learning_validator_requires_usable_mean_and_survival(self):
        from tools.run_g1_rmr_action_space_parity import (
            validate_early_learning_artifacts,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)

            result = validate_early_learning_artifacts(run)
            self.assertTrue(result["valid"])
            self.assertEqual(result["step"], 98_304)
            self.assertEqual(result["updates"], 16)
            self.assertEqual(result["noisy_first_episode_survival"], 52)
            self.assertEqual(result["clean_phase_zero_survival"], 45)
            self.assertLess(result["actor_mean_saturation_fraction"], 0.20)

            noisy = (
                run
                / "early_learning_evidence"
                / "checkpoint_step_098304"
                / "noisy"
            )
            with np.load(noisy / "training_action_noise.npz") as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["action_mean"] = np.full((120, 29), 0.99)
            payload["noisy_action"] = (
                payload["action_mean"]
                + payload["epsilon"] * payload["action_std"]
            )
            payload["effective_action"] = np.clip(
                payload["noisy_action"], -1.0, 1.0
            )
            np.savez_compressed(noisy / "training_action_noise.npz", **payload)
            with self.assertRaisesRegex(ValueError, "saturation"):
                validate_early_learning_artifacts(run)

    def test_early_learning_validator_rejects_short_clean_rollout(self):
        from tools.run_g1_rmr_action_space_parity import (
            validate_early_learning_artifacts,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)
            clean_summary = (
                run
                / "early_learning_evidence"
                / "checkpoint_step_098304"
                / "clean"
                / "summary.json"
            )
            summary = json.loads(clean_summary.read_text())
            summary["steps"] = 39
            clean_summary.write_text(json.dumps(summary))
            clean_trajectory = clean_summary.parent / "evaluation.npz"
            with np.load(clean_trajectory) as archive:
                columns = archive["columns"]
                values = archive["values"][:39]
            np.savez_compressed(
                clean_trajectory,
                columns=columns,
                values=values,
            )

            with self.assertRaisesRegex(ValueError, "clean phase-zero"):
                validate_early_learning_artifacts(run)

    def test_early_learning_validator_rejects_missing_media(self):
        from tools.run_g1_rmr_action_space_parity import (
            validate_early_learning_artifacts,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)
            (
                run
                / "early_learning_evidence"
                / "checkpoint_step_098304"
                / "noisy"
                / "training_rollout.mp4"
            ).unlink()

            with self.assertRaisesRegex(ValueError, "media"):
                validate_early_learning_artifacts(run)

    def test_early_learning_validator_recomputes_noisy_action(self):
        from tools.run_g1_rmr_action_space_parity import (
            validate_early_learning_artifacts,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)
            tape = (
                run
                / "early_learning_evidence"
                / "checkpoint_step_098304"
                / "noisy"
                / "training_action_noise.npz"
            )
            with np.load(tape) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["noisy_action"] = np.zeros((120, 29))
            np.savez_compressed(tape, **payload)

            with self.assertRaisesRegex(ValueError, "reparameterized"):
                validate_early_learning_artifacts(run)

    def test_early_learning_validator_rejects_nonzero_seed(self):
        from tools.run_g1_rmr_action_space_parity import (
            validate_early_learning_artifacts,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)
            hparams_path = run / "hparams.json"
            hparams = json.loads(hparams_path.read_text())
            hparams["seed"] = 1
            hparams_path.write_text(json.dumps(hparams))

            with self.assertRaisesRegex(ValueError, "seed"):
                validate_early_learning_artifacts(run)

    def test_early_learning_validator_reconstructs_clean_survival(self):
        from tools.run_g1_rmr_action_space_parity import (
            validate_early_learning_artifacts,
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            _write_early_learning_fixture(run)
            clean = (
                run
                / "early_learning_evidence"
                / "checkpoint_step_098304"
                / "clean"
            )
            with np.load(clean / "evaluation.npz") as archive:
                columns = archive["columns"]
            np.savez_compressed(
                clean / "evaluation.npz",
                columns=columns,
                values=np.zeros((39, 3)),
            )

            with self.assertRaisesRegex(ValueError, "clean trajectory"):
                validate_early_learning_artifacts(run)

    def test_early_learning_kwargs_restore_upstream_action_boundary(self):
        from src.core.rmr_action_noise import RMR_ACTION_STD
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
            for key in set(full) | set(early)
            if key not in full
            or key not in early
            or not np.array_equal(np.asarray(full[key]), np.asarray(early[key]))
        }
        self.assertEqual(
            differing,
            {
                "total_steps",
                "curriculum_grace",
                "curriculum_steps",
                "action_noise_std_start",
                "env_variant",
                "reference_residual_scale",
            },
        )
        self.assertEqual(early["total_steps"], 98_304)
        self.assertEqual(early["checkpoint_interval"], 98_304)
        self.assertEqual(early["curriculum_grace"], 98_304)
        self.assertEqual(early["curriculum_steps"], 1)
        self.assertEqual(early["actor_lr"], 5e-3)
        np.testing.assert_array_equal(
            early["action_noise_std_start"], RMR_ACTION_STD
        )
        np.testing.assert_array_equal(
            early["action_noise_std_end"], RMR_ACTION_STD
        )
        self.assertEqual(
            early["env_variant"], "g1_tracking_rmr_50hz_upstream_boundary"
        )
        self.assertEqual(early["reference_residual_scale"], 0.5)

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

    def test_upstream_action_penalty_is_an_early_learning_treatment(self):
        from tools.run_g1_rmr_action_space_parity import (
            build_parser,
            selected_env_variant,
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
                "--upstream-action-penalty",
            ]
        )
        validate_mode_args(args)
        self.assertEqual(
            selected_env_variant(args),
            "g1_tracking_rmr_50hz_upstream_action_penalty",
        )

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

    def test_early_learning_mode_rejects_nonzero_seed_before_training(self):
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
                "--seed",
                "1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "seed zero"):
            validate_mode_args(args)

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

    def test_preflight_reports_upstream_penalty_boundary_truthfully(self):
        from tools.run_g1_rmr_action_space_parity import (
            EXPECTED_REFERENCE_SHA256,
            validate_preflight,
        )

        with (
            mock.patch(
                "tools.run_g1_rmr_action_space_parity._git_output",
                side_effect=("a" * 40, ""),
            ),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch(
                "tools.run_g1_rmr_action_space_parity.sha256_file",
                return_value=EXPECTED_REFERENCE_SHA256,
            ),
            mock.patch(
                "tools.run_g1_rmr_action_space_parity.validate_runtime_assets",
                return_value={},
            ),
        ):
            result = validate_preflight(
                repository=Path("/tmp/repository"),
                reference_path=Path("/tmp/dance.npz"),
                code_commit="a" * 40,
                env_variant=(
                    "g1_tracking_rmr_50hz_upstream_action_penalty"
                ),
            )

        self.assertEqual(result["reference_residual_scale"], 0.5)
        self.assertTrue(result["normalized_action_clip"])
        self.assertEqual(result["action_magnitude_weight"], 0.05)
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
