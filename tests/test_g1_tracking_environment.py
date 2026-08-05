import math
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np


MODEL = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
REFERENCE = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/w02_rmrspec_grounded.npz"
)
CONTROLLER = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)


class G1TrackingEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        cls.env = G1TrackingEnv(
            xml_path=str(MODEL),
            reference_path=str(REFERENCE),
            controller_path=str(CONTROLLER),
            actor_history_len=1,
        )

    def test_environment_dimensions_and_timebase_match_rmr_port(self):
        env = self.env
        self.assertEqual(env.action_dim, 29)
        self.assertEqual(env.actor_frame_obs_dim, 154)
        self.assertEqual(env.actor_obs_dim, 154)
        self.assertEqual(env.critic_obs_dim, 286)
        self.assertEqual(env.n_frames, 5)
        self.assertAlmostEqual(env.dt, 0.01)
        state = env.reset(jax.random.PRNGKey(2), jnp.array(0.0))
        self.assertEqual(state.obs.shape, (154,))
        self.assertEqual(
            state.info["bootstrap_critic_obs"].shape, (286,)
        )

    def test_open_diffloco_factory_selects_tracking_environment(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv
        from src.envs.go2.environment import get_go2_env_class

        self.assertIs(get_go2_env_class("g1_tracking"), G1TrackingEnv)

    def test_reference_state_initialization_is_exact(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(7), jnp.array(0.0))
        phase = int(state.info["phase"])

        np.testing.assert_allclose(
            state.data.qpos, env.reference.qpos[phase], atol=1e-7
        )
        np.testing.assert_allclose(
            state.data.qvel, env.reference.qvel[phase], atol=1e-7
        )
        reward, components = env._tracking_reward(state.data, state.info)
        self.assertAlmostEqual(float(reward), 5.0, places=5)
        for value in components.values():
            self.assertAlmostEqual(float(value), 1.0, places=5)

    def test_fixed_phase_reset_supports_paired_evaluation(self):
        env = self.env
        state = env.reset_at_phase(
            jax.random.PRNGKey(17), jnp.array(0.0), jnp.array(37)
        )

        self.assertEqual(int(state.info["phase"]), 37)
        np.testing.assert_allclose(
            state.data.qpos, env.reference.qpos[37], atol=1e-7
        )

    def test_actor_observation_begins_with_matching_reference_command(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(11), jnp.array(0.0))
        phase = int(state.info["phase"])
        frame = state.obs.reshape(env.actor_history_len, -1)[-1]

        expected_command = np.concatenate(
            [
                env.reference.qpos[phase, 7:],
                env.reference.qvel[phase, 6:],
            ]
        )
        np.testing.assert_allclose(frame[:58], expected_command, atol=1e-7)
        np.testing.assert_allclose(
            frame[58:64],
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            atol=1e-5,
        )

    def test_global_anchor_translation_does_not_corrupt_relative_body_pose(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(13), jnp.array(0.0))
        translated_qpos = state.data.qpos.at[0].add(0.3)
        translated = mjx.forward(
            env.mjx_model, state.data.replace(qpos=translated_qpos)
        )

        _, components = env._tracking_reward(translated, state.info)
        self.assertAlmostEqual(
            float(components["anchor_position"]), math.exp(-1.0), places=5
        )
        self.assertAlmostEqual(
            float(components["body_position"]), 1.0, places=5
        )

    def test_one_step_advances_exactly_one_reference_frame(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(3), jnp.array(0.0))
        next_state = env.step(state, jnp.zeros(env.action_dim))

        expected_phase = (int(state.info["phase"]) + 1) % env.reference_length
        self.assertEqual(int(next_state.info["phase"]), expected_phase)
        self.assertTrue(np.isfinite(np.asarray(next_state.data.qpos)).all())
        self.assertTrue(np.isfinite(float(next_state.reward)))


if __name__ == "__main__":
    unittest.main()
