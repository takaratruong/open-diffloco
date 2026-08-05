import hashlib
import unittest
from argparse import Namespace
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]


class FrozenAlgorithmContractTest(unittest.TestCase):
    def test_upstream_algorithm_files_are_unchanged(self):
        expected = {
            "src/algorithms/shac/algorithm.py": (
                "69a3a9a1c6fa38a666abb245141b92225ef3701dbb8ca645c7a21e4e52228f30"
            ),
            "src/algorithms/jave/algorithm.py": (
                "6312c2f91c67dade4f0144875f75fce5beca5d7128df0ae3dea51a4f8eb7ec04"
            ),
            "src/core/networks.py": (
                "b67b3956535a5c2ae899c71e7e9eb065ad12421c129bf4056d1e4e3730439b81"
            ),
        }
        for relative, digest in expected.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)


class HumanoidModelContractTest(unittest.TestCase):
    def test_model_exposes_expected_mjx_contract(self):
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "src/envs/humanoid/models/humanoid_mjx.xml")
        )
        self.assertEqual((model.nq, model.nv, model.nu), (35, 34, 28))
        self.assertEqual(model.opt.iterations, 1)
        self.assertEqual(model.opt.ls_iterations, 5)
        self.assertEqual(model.nkey, 1)
        self.assertEqual(model.keyframe("home").qpos.shape, (35,))
        self.assertAlmostEqual(model.keyframe("home").qpos[2], 0.88)

        data = mujoco.MjData(model)
        data.qpos[:] = model.keyframe("home").qpos
        mujoco.mj_forward(model, data)
        for name in ("FL", "FR", "RL", "RR"):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.assertGreaterEqual(site_id, 0)
            self.assertLess(abs(data.site_xpos[site_id, 2]), 0.03)
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), 0
            )


class HumanoidEnvironmentContractTest(unittest.TestCase):
    def test_upstream_factory_selects_humanoid_adapter(self):
        from src.envs.go2.environment import get_go2_env_class
        from src.envs.humanoid.environment import HumanoidEnv

        self.assertIs(
            get_go2_env_class("humanoid_blind_linvel_nokinref"),
            HumanoidEnv,
        )

    def test_environment_infers_humanoid_dimensions(self):
        from src.envs.humanoid.environment import HumanoidEnv

        env = HumanoidEnv(
            xml_path="src/envs/humanoid/models/humanoid_mjx.xml",
        )
        self.assertEqual(env.action_dim, 28)
        self.assertEqual(env.actor_frame_obs_dim, 96)
        self.assertEqual(env.actor_obs_dim, 960)
        self.assertEqual(env.critic_obs_dim, 97)
        self.assertEqual(tuple(env._foot_site_ids.shape), (4,))
        self.assertEqual(tuple(env._foot_geom_ids.shape), (4,))
        self.assertEqual(len(set(map(int, env._foot_body_ids))), 2)
        self.assertAlmostEqual(env.target_height, 0.88)
        self.assertAlmostEqual(env.termination_height, 0.528)

    def test_visualizer_falls_back_to_a_torso_tracking_camera(self):
        from src.envs.humanoid.environment import HumanoidEnv
        from src.visualization.go2 import _make_env_render_camera

        env = HumanoidEnv(
            xml_path="src/envs/humanoid/models/humanoid_mjx.xml",
        )
        camera = _make_env_render_camera(env)
        self.assertIsInstance(camera, mujoco.MjvCamera)
        self.assertEqual(camera.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(camera.trackbodyid, env.torso_body_id)
        self.assertLessEqual(camera.distance, 3.0)


class HumanoidCliContractTest(unittest.TestCase):
    def test_humanoid_config_selects_frozen_upstream_defaults(self):
        from src.main import _apply_config

        args = Namespace(embodiment=None, config="humanoid.yaml", gpu=3)
        configured = _apply_config(
            args,
            {
                "embodiment": "humanoid",
                "algorithm": "shac",
                "steps": 8_000_000,
            },
        )
        self.assertEqual(configured.embodiment, "humanoid")
        self.assertEqual(configured.variant, "humanoid_blind_linvel_nokinref")
        self.assertEqual(
            configured.model_xml,
            "src/envs/humanoid/models/humanoid_mjx.xml",
        )
        self.assertEqual(configured.actor_lr, 5e-3)
        self.assertEqual(configured.critic_lr, 5e-4)
        self.assertEqual(configured.action_scale, 0.5)

    def test_gate_runner_pins_upstream_shac_parameters(self):
        from tools.run_humanoid_shac import build_train_kwargs

        kwargs = build_train_kwargs(
            steps=3_072,
            num_envs=256,
            seed=7,
            checkpoint_interval=100_000,
        )
        self.assertEqual(kwargs["total_steps"], 3_072)
        self.assertEqual(kwargs["num_envs"], 256)
        self.assertEqual(kwargs["unroll_length"], 12)
        self.assertEqual(kwargs["critic_iterations"], 16)
        self.assertEqual(kwargs["actor_lr"], 5e-3)
        self.assertEqual(kwargs["critic_lr"], 5e-4)
        self.assertEqual(kwargs["gamma"], 0.99)
        self.assertEqual(kwargs["gae_lambda"], 0.95)
        self.assertEqual(kwargs["target_update_rate"], 0.01)
        self.assertEqual(kwargs["action_scale"], 0.5)
        self.assertEqual(kwargs["cmd_vel_x_range"], (-2.0, 2.0))
        self.assertEqual(kwargs["action_noise_std_start"], 0.5)
        self.assertEqual(kwargs["action_noise_std_end"], 0.32)
        self.assertEqual(kwargs["env_variant"], "humanoid_blind_linvel_nokinref")
        self.assertEqual(kwargs["curriculum_grace"], 0)
        self.assertEqual(kwargs["curriculum_steps"], 1)

    def test_gate_runner_enables_upstream_x64_mode(self):
        import jax

        from tools.run_humanoid_shac import configure_jax

        jax.config.update("jax_enable_x64", False)
        self.assertFalse(jax.config.jax_enable_x64)
        configure_jax()
        self.assertTrue(jax.config.jax_enable_x64)


if __name__ == "__main__":
    unittest.main()
