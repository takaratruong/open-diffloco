import unittest
from pathlib import Path
from unittest import mock


class CanonicalG1ShacRunnerTest(unittest.TestCase):
    def test_canonical_kwargs_match_open_diffloco_contract(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs

        kwargs = build_canonical_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        expected = {
            "total_steps": 8_000_000,
            "num_envs": 256,
            "unroll_length": 12,
            "critic_iterations": 16,
            "actor_lr": 5e-3,
            "critic_lr": 5e-4,
            "action_noise_std_start": 0.5,
            "action_noise_std_end": 0.32,
            "actor_bootstrap_scale": 1.0,
            "actor_history_len": 10,
            "actor_hidden": (512, 256, 128),
            "actor_layer_norm": True,
            "actor_zero_output": True,
            "actor_per_env_grad_clip": None,
            "critic_per_env_grad_clip": None,
            "friction_range": (0.5, 2.0),
            "mass_range": (0.85, 1.15),
            "kp_range": (25.0, 45.0),
            "kd_range": (0.3, 0.7),
            "com_offset_range": (0.05, 0.05, 0.04),
            "push_velocity_range": (-1.0, 1.0),
            "push_interval_s": 4.0,
            "terrain": False,
            "domain_randomization": True,
            "reference_reset_noise_scale": 1.0,
            "reference_residual_control": True,
            "reference_residual_scale": 0.5,
            "actor_observation_noise": True,
            "solver_iterations": 4,
            "solver_ls_iterations": 5,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(kwargs[name], value)

        self.assertEqual(
            kwargs["env_variant"],
            "g1_tracking_rmr_50hz_source_step",
        )
        self.assertEqual(kwargs["solver_profile"], "g1-4x5")
        self.assertEqual(kwargs["checkpoint_interval"], 393_216)
        self.assertEqual(kwargs["curriculum_grace"], 800_000)
        self.assertEqual(kwargs["curriculum_steps"], 6_400_000)
        self.assertEqual(kwargs["reference_path"], "/tmp/dance.npz")
        self.assertEqual(kwargs["reference_stride"], 1)
        self.assertEqual(kwargs["seed"], 42)

    def test_solver_profiles_change_only_registered_solver_fields(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs

        stock = build_canonical_kwargs(
            "upstream-1x5", Path("/tmp/dance.npz"), seed=7
        )
        fixed = build_canonical_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=7
        )

        differing = {
            name for name in stock if stock[name] != fixed[name]
        }
        self.assertEqual(
            differing,
            {"solver_profile", "solver_iterations"},
        )

    def test_canonical_kwargs_transport_exact_resume_checkpoint(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs

        checkpoint = Path("/tmp/canonical/checkpoint_step_2359296.pkl")
        resumed = build_canonical_kwargs(
            "g1-4x5",
            Path("/tmp/dance.npz"),
            seed=0,
            resume_from=checkpoint,
        )
        fresh = build_canonical_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=0
        )

        self.assertEqual(resumed["resume_from"], str(checkpoint.resolve()))
        self.assertNotIn("resume_from", fresh)

    def test_parser_rejects_scientific_overrides(self):
        from tools.run_canonical_g1_shac import build_parser

        with mock.patch(
            "sys.argv", ["canonical-g1", "--actor-lr", "0.001"]
        ):
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
