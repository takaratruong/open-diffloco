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
        self.assertEqual(kwargs["actor_bootstrap_scale"], 1.0)
        self.assertEqual(kwargs["friction_range"], (1.0, 1.0))
        self.assertEqual(kwargs["mass_range"], (1.0, 1.0))
        self.assertEqual(kwargs["com_offset_range"], (0.0, 0.0, 0.0))
        self.assertEqual(kwargs["push_velocity_range"], (0.0, 0.0))
        self.assertFalse(kwargs["terrain"])


if __name__ == "__main__":
    unittest.main()
