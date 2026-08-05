import unittest
from unittest import mock
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tools.evaluate_g1_tracking import (
    configure_jax,
    load_rmr_policy,
    make_evaluation_env,
    scale_policy_action,
)

RMR_CHECKPOINT = Path(
    "/home/ubuntu/projects/rmr_tracking/logs/rsl_rl/g1_flat/"
    "2026-06-10_11-11-49_walk_win137_212/model_4999.pt"
)
RMR_ACTIONS = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)


class G1TrackingEvaluatorTest(unittest.TestCase):
    def test_evaluator_enables_training_precision(self):
        with mock.patch.object(jax.config, "update") as update:
            configure_jax()

        update.assert_called_once_with("jax_enable_x64", True)

    def test_unbounded_native_timebase_is_available_for_action_tape_control(self):
        env = make_evaluation_env("g1_tracking_rmr_50hz_unbounded")

        self.assertAlmostEqual(env.dt, 0.02)
        self.assertEqual(env.reference_stride, 2)
        self.assertFalse(env.squash_actor_actions)

    def test_source_policy_reset_action_agrees_with_logged_rollout(self):
        env = make_evaluation_env("g1_tracking_rmr_50hz_unbounded")
        state = env.reset_at_phase(
            jax.random.PRNGKey(0), jnp.array(0.0), jnp.array(0)
        )
        policy = load_rmr_policy(RMR_CHECKPOINT)
        predicted = np.asarray(policy(state.obs))
        with np.load(RMR_ACTIONS, allow_pickle=False) as archive:
            logged = np.asarray(archive["action"][0])

        cosine = np.dot(predicted, logged) / (
            np.linalg.norm(predicted) * np.linalg.norm(logged)
        )
        self.assertGreater(cosine, 0.9)

    def test_source_step_controls_are_available_without_external_models(self):
        source_step = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step"
        )
        robust = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step_robust"
        )

        self.assertEqual(source_step.n_frames, 4)
        self.assertAlmostEqual(source_step.mj_model.opt.timestep, 0.005)
        self.assertEqual(source_step.mj_model.ngeom, robust.mj_model.ngeom)
        self.assertEqual(robust.mj_model.opt.iterations, 10)
        self.assertEqual(robust.mj_model.opt.ls_iterations, 20)

    def test_source_step_evaluator_can_screen_solver_budgets(self):
        candidate = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step",
            solver_iterations=4,
            solver_ls_iterations=10,
        )

        self.assertEqual(candidate.mj_model.opt.iterations, 4)
        self.assertEqual(candidate.mj_model.opt.ls_iterations, 10)

    def test_action_gain_scales_policy_without_changing_direction(self):
        action = jnp.array([-0.8, 0.2, 1.0])
        np.testing.assert_allclose(
            scale_policy_action(action, 0.25),
            np.array([-0.2, 0.05, 0.25]),
        )

    def test_action_gain_must_interpolate_zero_and_learned_policy(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            scale_policy_action(jnp.zeros(3), 1.1)


if __name__ == "__main__":
    unittest.main()
