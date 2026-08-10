import importlib
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


class G1RandomizationSamplingTest(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module(
                "src.envs.g1_tracking.randomization"
            )
        except ModuleNotFoundError:
            self.fail("G1 randomization module is missing")

    def test_difficulty_zero_matches_upstream_randomization_semantics(self):
        module = self._module()

        values = module.sample_g1_randomization(
            jax.random.PRNGKey(1),
            jnp.array(0.0),
            module.CANONICAL_G1_RANDOMIZATION,
        )

        self.assertGreaterEqual(float(values["friction_scale"]), 0.5)
        self.assertLessEqual(float(values["friction_scale"]), 2.0)
        self.assertGreaterEqual(float(values["mass_scale"]), 0.85)
        self.assertLessEqual(float(values["mass_scale"]), 1.15)
        self.assertEqual(float(values["kp_scale"]), 1.0)
        self.assertEqual(float(values["kd_scale"]), 1.0)
        np.testing.assert_array_equal(values["com_offset"], np.zeros(3))

    def test_difficulty_one_samples_complete_registered_ranges(self):
        module = self._module()
        keys = jax.random.split(jax.random.PRNGKey(2), 512)
        samples = jax.vmap(
            lambda key: module.sample_g1_randomization(
                key,
                jnp.array(1.0),
                module.CANONICAL_G1_RANDOMIZATION,
            )
        )(keys)

        friction = np.asarray(samples["friction_scale"])
        mass = np.asarray(samples["mass_scale"])
        kp = np.asarray(samples["kp_scale"])
        kd = np.asarray(samples["kd_scale"])
        com = np.asarray(samples["com_offset"])
        self.assertTrue(np.all((0.5 <= friction) & (friction <= 2.0)))
        self.assertTrue(np.all((0.85 <= mass) & (mass <= 1.15)))
        self.assertTrue(
            np.all((25.0 / 35.0 <= kp) & (kp <= 45.0 / 35.0))
        )
        self.assertTrue(np.all((0.3 / 0.5 <= kd) & (kd <= 0.7 / 0.5)))
        self.assertTrue(
            np.all(np.abs(com) <= np.array([0.05, 0.05, 0.04]))
        )
        self.assertGreater(np.ptp(friction), 1.0)
        self.assertGreater(np.ptp(mass), 0.2)
        self.assertGreater(np.ptp(kp), 0.4)
        self.assertGreater(np.ptp(kd), 0.5)

    def test_sampling_is_reproducible(self):
        module = self._module()
        key = jax.random.PRNGKey(3)

        first = module.sample_g1_randomization(
            key,
            jnp.array(0.75),
            module.CANONICAL_G1_RANDOMIZATION,
        )
        second = module.sample_g1_randomization(
            key,
            jnp.array(0.75),
            module.CANONICAL_G1_RANDOMIZATION,
        )

        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


class G1RandomizedModelTest(unittest.TestCase):
    def _environment(self):
        from src.envs.g1_tracking.environment import G1TrackingEnv

        try:
            return G1TrackingEnv(
                xml_path=MODEL,
                reference_path=REFERENCE,
                controller_path=CONTROLLER,
                actor_history_len=1,
                domain_randomization=True,
                friction_range=(0.5, 2.0),
                mass_range=(0.85, 1.15),
                kp_range=(25.0, 45.0),
                kd_range=(0.3, 0.7),
                com_offset_range=(0.05, 0.05, 0.04),
            )
        except ValueError as exc:
            self.fail(f"canonical randomized G1 environment rejected: {exc}")

    def test_randomized_model_scales_registered_physical_arrays(self):
        env = self._environment()
        info = {
            "friction_scale": jnp.array(1.5),
            "mass_scale": jnp.array(1.1),
            "kp_scale": jnp.array(0.8),
            "kd_scale": jnp.array(1.2),
            "com_offset": jnp.array([0.01, -0.02, 0.03]),
        }

        model = env._get_randomized_model(info)

        np.testing.assert_allclose(
            model.geom_friction, 1.5 * env.base_friction
        )
        np.testing.assert_allclose(model.body_mass, 1.1 * env.base_mass)
        np.testing.assert_allclose(
            model.body_inertia, 1.1 * env.base_inertia
        )
        expected_ipos = np.asarray(env.base_ipos).copy()
        expected_ipos[env.pelvis_body_id] += np.asarray(info["com_offset"])
        np.testing.assert_allclose(model.body_ipos, expected_ipos)

    def test_training_resets_sample_and_carry_distinct_models(self):
        env = self._environment()

        first = env.reset(jax.random.PRNGKey(10), jnp.array(1.0))
        second = env.reset(jax.random.PRNGKey(11), jnp.array(1.0))

        for state in (first, second):
            self.assertGreaterEqual(float(state.info["friction_scale"]), 0.5)
            self.assertLessEqual(float(state.info["friction_scale"]), 2.0)
            self.assertGreaterEqual(float(state.info["mass_scale"]), 0.85)
            self.assertLessEqual(float(state.info["mass_scale"]), 1.15)
        self.assertNotEqual(
            float(first.info["friction_scale"]),
            float(second.info["friction_scale"]),
        )
        self.assertNotEqual(
            float(first.info["mass_scale"]),
            float(second.info["mass_scale"]),
        )

    def test_exact_phase_reset_remains_nominal_for_evaluation(self):
        env = self._environment()

        state = env.reset_at_phase(
            jax.random.PRNGKey(12),
            jnp.array(1.0),
            jnp.array(3),
        )

        self.assertEqual(float(state.info["friction_scale"]), 1.0)
        self.assertEqual(float(state.info["mass_scale"]), 1.0)
        self.assertEqual(float(state.info["kp_scale"]), 1.0)
        self.assertEqual(float(state.info["kd_scale"]), 1.0)
        np.testing.assert_array_equal(state.info["com_offset"], np.zeros(3))

    def test_vectorized_step_uses_each_carried_randomization(self):
        env = self._environment()
        state = env.reset_at_phase(
            jax.random.PRNGKey(13),
            jnp.array(1.0),
            jnp.array(3),
        )
        light = state.replace(
            info={
                **state.info,
                "friction_scale": jnp.array(0.5),
                "mass_scale": jnp.array(0.85),
                "kp_scale": jnp.array(25.0 / 35.0),
                "kd_scale": jnp.array(0.3 / 0.5),
                "com_offset": jnp.array([-0.05, 0.05, -0.04]),
            }
        )
        heavy = state.replace(
            info={
                **state.info,
                "friction_scale": jnp.array(2.0),
                "mass_scale": jnp.array(1.15),
                "kp_scale": jnp.array(45.0 / 35.0),
                "kd_scale": jnp.array(0.7 / 0.5),
                "com_offset": jnp.array([0.05, -0.05, 0.04]),
            }
        )
        batched = jax.tree_util.tree_map(
            lambda left, right: jnp.stack((left, right)),
            light,
            heavy,
        )

        stepped = jax.vmap(env.step)(
            batched,
            jnp.zeros((2, env.action_dim)),
        )

        difference = np.linalg.norm(
            np.asarray(stepped.data.qpos[0] - stepped.data.qpos[1])
        )
        self.assertGreater(difference, 1e-8)


if __name__ == "__main__":
    unittest.main()
