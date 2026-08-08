import unittest

from mujoco.mjx._src import solver as mjx_solver

from src.envs.g1_tracking.fixed_solver import (
    CONVERGENCE_SCAN,
    FIXED_SOLVER_AND_LINESEARCH_SCAN,
    _fixed_loop_scan,
    _solve_with_fixed_outer_loop,
    active_solver_gradient_semantic,
    fixed_mjx_solver_outer_loop,
)


class G1FixedSolverTest(unittest.TestCase):
    def test_convergence_context_patches_only_solver_and_restores_everything(self):
        original_solve = mjx_solver.solve
        original_loop = mjx_solver._while_loop_scan

        with fixed_mjx_solver_outer_loop(semantic=CONVERGENCE_SCAN):
            self.assertIs(mjx_solver.solve, _solve_with_fixed_outer_loop)
            self.assertIs(mjx_solver._while_loop_scan, original_loop)
            self.assertEqual(active_solver_gradient_semantic(), CONVERGENCE_SCAN)

        self.assertIs(mjx_solver.solve, original_solve)
        self.assertIs(mjx_solver._while_loop_scan, original_loop)
        self.assertIsNone(active_solver_gradient_semantic())

    def test_fixed_linesearch_context_patches_both_scoped_loops_and_restores(self):
        original_solve = mjx_solver.solve
        original_loop = mjx_solver._while_loop_scan

        with fixed_mjx_solver_outer_loop(semantic=FIXED_SOLVER_AND_LINESEARCH_SCAN):
            self.assertIs(mjx_solver.solve, _solve_with_fixed_outer_loop)
            self.assertIs(mjx_solver._while_loop_scan, _fixed_loop_scan)
            self.assertEqual(
                active_solver_gradient_semantic(), FIXED_SOLVER_AND_LINESEARCH_SCAN
            )

        self.assertIs(mjx_solver.solve, original_solve)
        self.assertIs(mjx_solver._while_loop_scan, original_loop)
        self.assertIsNone(active_solver_gradient_semantic())

    def test_fixed_loop_scan_ignores_early_stop_and_runs_exact_maximum(self):
        def body(value):
            return value + 1

        result = _fixed_loop_scan(lambda value: value < 1, body, 0, 4)

        self.assertEqual(result, 4)
