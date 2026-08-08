import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class G1TrackingRunnerTest(unittest.TestCase):
    def test_reference_hparams_capture_exact_training_artifact(self):
        from src.algorithms.shac.algorithm import reference_hparams_for_env

        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "dance.npz"
            reference_path.write_bytes(b"named reference fixture")
            env = SimpleNamespace(
                reference_path=str(reference_path),
                reference=SimpleNamespace(
                    fps=50.0,
                    qpos=SimpleNamespace(shape=(500, 36)),
                ),
                reference_stride=1,
                reference_transitions=499,
            )

            hparams = reference_hparams_for_env(env)

        self.assertEqual(hparams["reference_path"], str(reference_path.resolve()))
        self.assertEqual(
            hparams["reference_sha256"],
            hashlib.sha256(b"named reference fixture").hexdigest(),
        )
        self.assertEqual(hparams["reference_fps"], 50.0)
        self.assertEqual(hparams["reference_stride"], 1)
        self.assertEqual(hparams["reference_states"], 500)
        self.assertEqual(hparams["reference_transitions"], 499)

    def test_runner_disables_unregistered_randomization_and_uses_tracking_task(self):
        from tools.run_g1_tracking_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=4096,
            num_envs=2,
            seed=7,
            checkpoint_interval=1000,
        )
        self.assertEqual(kwargs["env_variant"], "g1_tracking")
        self.assertEqual(kwargs["actor_history_len"], 1)
        self.assertEqual(kwargs["unroll_length"], 16)
        self.assertEqual(kwargs["num_envs"], 2)
        self.assertEqual(kwargs["total_steps"], 4096)
        self.assertEqual(kwargs["seed"], 7)
        self.assertEqual(kwargs["actor_lr"], 1e-4)
        self.assertEqual(kwargs["action_noise_std_start"], 0.05)
        self.assertEqual(kwargs["action_noise_std_end"], 0.05)
        self.assertEqual(kwargs["actor_per_env_grad_clip"], 1.0)
        self.assertEqual(kwargs["critic_per_env_grad_clip"], 1.0)
        self.assertEqual(kwargs["actor_bootstrap_scale"], 1.0)
        self.assertEqual(kwargs["friction_range"], (1.0, 1.0))
        self.assertEqual(kwargs["mass_range"], (1.0, 1.0))
        self.assertEqual(kwargs["com_offset_range"], (0.0, 0.0, 0.0))
        self.assertEqual(kwargs["push_velocity_range"], (0.0, 0.0))
        self.assertFalse(kwargs["terrain"])

    def test_native_rmr_runner_matches_50hz_rollout_timebase(self):
        from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=196608,
            num_envs=32,
            seed=3,
            checkpoint_interval=49152,
        )
        self.assertEqual(kwargs["env_variant"], "g1_tracking_rmr_50hz")
        self.assertEqual(kwargs["unroll_length"], 24)
        self.assertEqual(kwargs["gamma"], 0.99)
        self.assertEqual(kwargs["gae_lambda"], 0.95)
        self.assertEqual(kwargs["max_episode_length"], 60)
        self.assertEqual(kwargs["reference_path"], DEFAULT_REFERENCE_PATH)
        self.assertEqual(kwargs["reference_stride"], 2)
        self.assertEqual(kwargs["total_steps"], 196608)
        self.assertEqual(kwargs["checkpoint_interval"], 49152)

    def test_native_rmr_runner_transports_named_reference_contract(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=393_216,
            num_envs=256,
            seed=0,
            checkpoint_interval=30_720,
            validated_task=True,
            reference_path="/tmp/dance.npz",
            reference_stride=1,
        )

        self.assertEqual(kwargs["reference_path"], "/tmp/dance.npz")
        self.assertEqual(kwargs["reference_stride"], 1)

    def test_native_rmr_runner_transports_exact_resume_checkpoint(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=589_824,
            num_envs=256,
            seed=0,
            checkpoint_interval=30_720,
            validated_task=True,
            reference_path="/tmp/dance.npz",
            reference_stride=1,
            resume_from="/tmp/policy_final.pkl",
        )

        self.assertEqual(kwargs["resume_from"], "/tmp/policy_final.pkl")

    def test_native_rmr_runner_can_match_go2_batch_and_noise_schedule(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=8_000_000,
            num_envs=256,
            seed=3,
            checkpoint_interval=400_000,
            actor_lr=5e-3,
            action_noise_std=0.5,
            action_noise_std_end=0.32,
            unroll_length=12,
        )
        self.assertEqual(kwargs["unroll_length"], 12)
        self.assertEqual(kwargs["action_noise_std_start"], 0.5)
        self.assertEqual(kwargs["action_noise_std_end"], 0.32)

    def test_native_rmr_runner_can_select_compact_random_actor(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=1,
            num_envs=1,
            seed=3,
            checkpoint_interval=1,
            action_noise_std=1.0,
            unroll_length=1,
            actor_hidden=(512, 512),
            actor_layer_norm=False,
            actor_zero_output=False,
        )

        self.assertEqual(kwargs["actor_hidden"], (512, 512))
        self.assertFalse(kwargs["actor_layer_norm"])
        self.assertFalse(kwargs["actor_zero_output"])
        self.assertEqual(kwargs["action_noise_std_start"], 1.0)
        self.assertEqual(kwargs["action_noise_std_end"], 1.0)

    def test_native_rmr_runner_transports_gradient_accumulation(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=1_572_864,
            num_envs=256,
            seed=3,
            checkpoint_interval=122_880,
            unroll_length=12,
            gradient_accumulation_steps=4,
        )

        self.assertEqual(kwargs["num_envs"], 256)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 4)

    def test_native_rmr_runner_transports_delayed_actor_bootstrap(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=393_216,
            num_envs=256,
            seed=3,
            checkpoint_interval=30_720,
            actor_bootstrap_scale=1.0,
            actor_bootstrap_delay_steps=61_440,
        )

        self.assertEqual(kwargs["actor_bootstrap_scale"], 1.0)
        self.assertEqual(kwargs["actor_bootstrap_delay_steps"], 61_440)

    def test_native_rmr_runner_rejects_invalid_actor_bootstrap_delay(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        for delay_steps in (-1, 1.5, True):
            with self.subTest(delay_steps=delay_steps):
                with self.assertRaisesRegex(
                    ValueError, "actor_bootstrap_delay_steps"
                ):
                    build_train_kwargs(
                        steps=393_216,
                        num_envs=256,
                        seed=3,
                        checkpoint_interval=30_720,
                        actor_bootstrap_delay_steps=delay_steps,
                    )

    def test_native_rmr_runner_rejects_invalid_gradient_accumulation(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        for accumulation_steps in (0, -1, 1.5, True):
            with self.subTest(accumulation_steps=accumulation_steps):
                with self.assertRaisesRegex(
                    ValueError, "gradient_accumulation_steps"
                ):
                    build_train_kwargs(
                        steps=49_152,
                        num_envs=256,
                        seed=3,
                        checkpoint_interval=12_288,
                        gradient_accumulation_steps=accumulation_steps,
                    )

    def test_native_rmr_runner_can_select_linear_unbounded_action_support(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            unbounded_actions=True,
        )

        self.assertEqual(
            kwargs["env_variant"], "g1_tracking_rmr_50hz_unbounded"
        )

    def test_native_rmr_runner_can_select_validated_task_authority(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
        )

        self.assertEqual(
            kwargs["env_variant"],
            "g1_tracking_rmr_50hz_validated",
        )

    def test_native_rmr_runner_transports_a_fixed_body_mass_scale(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            body_mass_scale=1.15,
        )

        self.assertEqual(kwargs["mass_range"], (1.15, 1.15))

    def test_native_rmr_runner_rejects_nonpositive_body_mass_scale(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        with self.assertRaisesRegex(ValueError, "body_mass_scale"):
            build_train_kwargs(
                steps=65_536,
                num_envs=256,
                seed=3,
                checkpoint_interval=16_384,
                validated_task=True,
                body_mass_scale=0.0,
            )

    def test_native_rmr_runner_transports_a_fixed_effort_limit_scale(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            effort_limit_scale=0.7,
        )

        self.assertEqual(kwargs["effort_limit_scale"], 0.7)

    def test_native_rmr_runner_rejects_invalid_effort_limit_scale(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "effort_limit_scale",
                ):
                    build_train_kwargs(
                        steps=65_536,
                        num_envs=256,
                        seed=3,
                        checkpoint_interval=16_384,
                        validated_task=True,
                        effort_limit_scale=scale,
                    )

    def test_native_rmr_runner_transports_termination_margin_weight(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            termination_margin_weight=0.5,
        )

        self.assertEqual(kwargs["termination_margin_weight"], 0.5)

    def test_native_rmr_runner_can_opt_in_to_resume_margin_treatment(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=786_432,
            num_envs=256,
            seed=3,
            checkpoint_interval=49_152,
            validated_task=True,
            termination_margin_weight=0.5,
            allow_resume_termination_margin_change=True,
            resume_from="/tmp/policy_final.pkl",
        )

        self.assertTrue(kwargs["allow_resume_termination_margin_change"])

    def test_native_rmr_runner_transports_reference_reset_noise_scale(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            reference_reset_noise_scale=1.0,
        )

        self.assertEqual(kwargs["reference_reset_noise_scale"], 1.0)

    def test_native_rmr_runner_transports_carried_reset_bank(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            carried_reset_bank_path="/tmp/carried_states.npz",
            carried_reset_probability=0.5,
            carried_reset_bank_start=64,
        )

        self.assertEqual(
            kwargs["carried_reset_bank_path"], "/tmp/carried_states.npz"
        )
        self.assertEqual(kwargs["carried_reset_probability"], 0.5)
        self.assertEqual(kwargs["carried_reset_bank_start"], 64)

    def test_native_rmr_runner_rejects_invalid_carried_reset_options(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        with self.assertRaisesRegex(ValueError, "carried_reset_bank_path"):
            build_train_kwargs(
                steps=65_536,
                num_envs=256,
                seed=3,
                checkpoint_interval=16_384,
                carried_reset_probability=0.5,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_train_kwargs(
                steps=65_536,
                num_envs=256,
                seed=3,
                checkpoint_interval=16_384,
                reference_reset_noise_scale=1.0,
                carried_reset_bank_path="/tmp/carried_states.npz",
                carried_reset_probability=0.5,
            )

    def test_native_rmr_runner_rejects_invalid_reference_reset_noise_scale(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        for scale in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "reference_reset_noise_scale",
                ):
                    build_train_kwargs(
                        steps=65_536,
                        num_envs=256,
                        seed=3,
                        checkpoint_interval=16_384,
                        validated_task=True,
                        reference_reset_noise_scale=scale,
                    )

    def test_native_rmr_runner_rejects_invalid_termination_margin_weight(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        for weight in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(
                    ValueError,
                    "termination_margin_weight",
                ):
                    build_train_kwargs(
                        steps=65_536,
                        num_envs=256,
                        seed=3,
                        checkpoint_interval=16_384,
                        validated_task=True,
                        termination_margin_weight=weight,
                    )

    def test_native_rmr_runner_can_train_a_bounded_source_policy_residual(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        source_policy = object()
        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            source_actor_policy=source_policy,
            residual_action_scale=0.1,
            differentiate_source_feedback=False,
        )

        self.assertIs(kwargs["source_actor_policy"], source_policy)
        self.assertEqual(kwargs["residual_action_scale"], 0.1)
        self.assertFalse(kwargs["differentiate_source_feedback"])

    def test_native_rmr_runner_can_initialize_the_complete_source_actor(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        full_actor = object()
        kwargs = build_train_kwargs(
            steps=65_536,
            num_envs=256,
            seed=3,
            checkpoint_interval=16_384,
            validated_task=True,
            initial_full_actor_policy=full_actor,
        )

        self.assertIs(kwargs["initial_full_actor_policy"], full_actor)
        self.assertIsNone(kwargs["source_actor_policy"])

    def test_full_actor_initialization_rejects_residual_composition(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_train_kwargs(
                steps=65_536,
                num_envs=256,
                seed=3,
                checkpoint_interval=16_384,
                validated_task=True,
                source_actor_policy=object(),
                residual_action_scale=0.1,
                initial_full_actor_policy=object(),
            )

    def test_full_actor_initialization_requires_validated_source_order(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        with self.assertRaisesRegex(ValueError, "validated_task"):
            build_train_kwargs(
                steps=65_536,
                num_envs=256,
                seed=3,
                checkpoint_interval=16_384,
                initial_full_actor_policy=object(),
            )

    def test_native_rmr_runner_rejects_residual_scale_without_source_policy(self):
        from tools.run_g1_tracking_rmr50_shac import build_train_kwargs

        with self.assertRaisesRegex(ValueError, "source_actor_policy"):
            build_train_kwargs(
                steps=65_536,
                num_envs=256,
                seed=3,
                checkpoint_interval=16_384,
                validated_task=True,
                residual_action_scale=0.1,
            )


if __name__ == "__main__":
    unittest.main()
