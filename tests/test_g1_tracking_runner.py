import unittest


class G1TrackingRunnerTest(unittest.TestCase):
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
        self.assertEqual(kwargs["total_steps"], 196608)
        self.assertEqual(kwargs["checkpoint_interval"], 49152)

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
            "g1_tracking_rmr_50hz_source_step_robust",
        )


if __name__ == "__main__":
    unittest.main()
