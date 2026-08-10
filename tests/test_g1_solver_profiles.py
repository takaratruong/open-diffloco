import unittest

import jax
import jax.numpy as jnp
import numpy as np


MODEL = (
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
REFERENCE = "/home/ubuntu/projects/diffsim2real/outputs/w02_rmrspec_grounded.npz"
CONTROLLER = "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"


class G1SolverProfileTest(unittest.TestCase):
    def test_registered_solver_profiles_are_exact(self):
        from src.envs.g1_tracking.solver_profiles import (
            SolverProfile,
            get_solver_profile,
        )

        self.assertEqual(
            get_solver_profile("upstream-1x5"),
            SolverProfile(1, 5, False),
        )
        self.assertEqual(
            get_solver_profile("g1-4x5"),
            SolverProfile(4, 5, True),
        )
        self.assertEqual(
            get_solver_profile("diagnostic-10x20"),
            SolverProfile(10, 20, True),
        )

    def test_unknown_profile_is_rejected(self):
        from src.envs.g1_tracking.solver_profiles import get_solver_profile

        with self.assertRaisesRegex(ValueError, "unknown solver profile"):
            get_solver_profile("almost-stock")

    def test_solver_context_restores_stock_solver_after_exception(self):
        from mujoco.mjx._src import solver

        from src.envs.g1_tracking.solver_profiles import (
            get_solver_profile,
            solver_context,
        )

        original = solver.solve
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with solver_context(get_solver_profile("g1-4x5")):
                self.assertIsNot(solver.solve, original)
                raise RuntimeError("injected")
        self.assertIs(solver.solve, original)

    def test_stock_profile_does_not_patch_solver(self):
        from mujoco.mjx._src import solver

        from src.envs.g1_tracking.solver_profiles import (
            get_solver_profile,
            solver_context,
        )

        original = solver.solve
        with solver_context(get_solver_profile("upstream-1x5")):
            self.assertIs(solver.solve, original)
        self.assertIs(solver.solve, original)

    def test_stock_and_fixed_one_iteration_match(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv
        from src.envs.g1_tracking.solver_profiles import (
            SolverProfile,
            solver_context,
        )

        env = G1TrackingEnv(
            xml_path=MODEL,
            reference_path=REFERENCE,
            controller_path=CONTROLLER,
            actor_history_len=1,
            physics_substeps=1,
            solver_iterations=1,
            solver_ls_iterations=5,
        )
        state = env.reset_at_phase(
            jax.random.PRNGKey(23), jnp.array(0.0), jnp.array(8)
        )
        action = jnp.linspace(-0.2, 0.2, env.action_dim)

        stock = env.step(state, action)
        with solver_context(SolverProfile(1, 5, True)):
            fixed = env.step(state, action)

        # GPU constraint reductions are not bitwise deterministic across two
        # separately traced solver callables.  One physical step must still be
        # numerically equivalent well below task-level tolerances.
        np.testing.assert_allclose(
            stock.data.qpos, fixed.data.qpos, rtol=0.0, atol=1e-8
        )
        np.testing.assert_allclose(
            stock.data.qvel, fixed.data.qvel, rtol=0.0, atol=1e-8
        )
        np.testing.assert_allclose(
            stock.reward, fixed.reward, rtol=0.0, atol=1e-8
        )
        np.testing.assert_array_equal(stock.done, fixed.done)


if __name__ == "__main__":
    unittest.main()
