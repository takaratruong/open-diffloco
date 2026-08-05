import unittest


class G1TrackingGateTest(unittest.TestCase):
    def test_gate_can_select_validated_task_authority(self):
        from tools.run_g1_tracking_gates import make_gate_env

        env = make_gate_env("g1_tracking_rmr_50hz_source_step_robust")

        self.assertEqual(env.n_frames, 4)
        self.assertAlmostEqual(env.mj_model.opt.timestep, 0.005)
        self.assertEqual(env.mj_model.opt.iterations, 10)
        self.assertEqual(env.mj_model.opt.ls_iterations, 20)
        self.assertFalse(env.squash_actor_actions)


if __name__ == "__main__":
    unittest.main()
