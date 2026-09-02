import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.actor_returns import (
    discounted_actor_return,
    resolve_actor_return_semantics,
)


class ActorReturnSemanticsTest(unittest.TestCase):
    def test_multi_episode_is_bit_exact_with_the_legacy_accumulator(self):
        rewards = jnp.array([0.13, -0.2, 0.7, 0.11], dtype=jnp.float64)
        dones = jnp.array([False, True, False, False])
        terminals = jnp.array([False, False, False, False])
        values = jnp.array([0.3, 0.4, 0.5, 0.6], dtype=jnp.float32)
        active = jnp.array([True, True, True, True])
        gamma = 0.99
        scale = jnp.asarray(0.75, dtype=jnp.float32)

        def legacy_step(carry, values_at_step):
            total, running, discount = carry
            reward, done, terminal, value, is_active = values_at_step
            next_discount = jnp.where(is_active, discount * gamma, discount)
            running = running + discount * reward
            truncation_bootstrap = scale * (1.0 - terminal) * next_discount * value
            total = total + jnp.where(done, running + truncation_bootstrap, 0.0)
            running = jnp.where(done, 0.0, running)
            discount = jnp.where(done, 1.0, next_discount)
            return (total, running, discount), None

        (total, running, final_discount), _ = jax.lax.scan(
            legacy_step,
            (0.0, 0.0, 1.0),
            (rewards, dones, terminals, values, active),
        )
        legacy = (
            total
            + running
            + jnp.where(dones[-1], 0.0, scale * final_discount * jnp.float32(0.9))
        )
        implemented = discounted_actor_return(
            rewards=rewards,
            dones=dones,
            terminals=terminals,
            bootstrap_values=values,
            active=active,
            final_value=jnp.float32(0.9),
            gamma=gamma,
            bootstrap_scale=scale,
            semantics="multi_episode",
        ).total

        self.assertEqual(
            np.asarray(implemented).tobytes(), np.asarray(legacy).tobytes()
        )

    def test_multi_episode_preserves_rewards_after_resets(self):
        result = discounted_actor_return(
            rewards=jnp.array([1.0, 2.0, 3.0, 4.0]),
            dones=jnp.array([False, True, False, True]),
            terminals=jnp.array([False, True, False, True]),
            bootstrap_values=jnp.zeros(4),
            active=jnp.ones(4, dtype=bool),
            final_value=jnp.array(0.0),
            gamma=1.0,
            bootstrap_scale=0.0,
            semantics="multi_episode",
        )

        self.assertEqual(float(result.total), 10.0)
        np.testing.assert_array_equal(result.reward_mask, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(
            result.post_first_done_mask,
            np.array([False, False, True, True]),
        )

    def test_first_terminal_excludes_the_reset_episode(self):
        result = discounted_actor_return(
            rewards=jnp.array([1.0, 2.0, 3.0, 4.0]),
            dones=jnp.array([False, True, False, True]),
            terminals=jnp.array([False, True, False, True]),
            bootstrap_values=jnp.zeros(4),
            active=jnp.ones(4, dtype=bool),
            final_value=jnp.array(0.0),
            gamma=1.0,
            bootstrap_scale=0.0,
            semantics="first_terminal",
        )

        self.assertEqual(float(result.total), 3.0)
        np.testing.assert_array_equal(
            result.reward_mask,
            np.array([True, True, False, False]),
        )
        np.testing.assert_array_equal(
            result.post_first_done_mask,
            np.array([False, False, True, True]),
        )

    def test_first_terminal_keeps_time_limit_bootstrap_but_not_final_bootstrap(
        self,
    ):
        result = discounted_actor_return(
            rewards=jnp.array([1.0, 2.0, 100.0]),
            dones=jnp.array([False, True, False]),
            terminals=jnp.array([False, False, False]),
            bootstrap_values=jnp.array([0.0, 5.0, 0.0]),
            active=jnp.ones(3, dtype=bool),
            final_value=jnp.array(7.0),
            gamma=0.5,
            bootstrap_scale=1.0,
            semantics="first_terminal",
        )

        # 1 + .5*2 + .25*V(s_2); the reset reward and final-state value vanish.
        self.assertEqual(float(result.total), 3.25)

    def test_true_terminal_has_no_bootstrap(self):
        result = discounted_actor_return(
            rewards=jnp.array([1.0, 2.0]),
            dones=jnp.array([False, True]),
            terminals=jnp.array([False, True]),
            bootstrap_values=jnp.array([0.0, 100.0]),
            active=jnp.ones(2, dtype=bool),
            final_value=jnp.array(100.0),
            gamma=0.5,
            bootstrap_scale=1.0,
            semantics="first_terminal",
        )

        self.assertEqual(float(result.total), 2.0)

    def test_no_done_uses_the_final_bootstrap_and_respects_active_horizon(
        self,
    ):
        result = discounted_actor_return(
            rewards=jnp.array([1.0, 2.0, 100.0]),
            dones=jnp.zeros(3, dtype=bool),
            terminals=jnp.zeros(3, dtype=bool),
            bootstrap_values=jnp.zeros(3),
            active=jnp.array([True, True, False]),
            final_value=jnp.array(10.0),
            gamma=0.5,
            bootstrap_scale=1.0,
            semantics="first_terminal",
        )

        self.assertEqual(float(result.total), 4.5)
        np.testing.assert_array_equal(
            result.reward_mask,
            np.array([True, True, False]),
        )

    def test_first_terminal_removes_post_reset_gradient(self):
        def objective(parameter, semantics):
            rewards = parameter * jnp.array([1.0, 2.0, 100.0])
            return discounted_actor_return(
                rewards=rewards,
                dones=jnp.array([False, True, False]),
                terminals=jnp.array([False, True, False]),
                bootstrap_values=jnp.zeros(3),
                active=jnp.ones(3, dtype=bool),
                final_value=jnp.array(0.0),
                gamma=1.0,
                bootstrap_scale=0.0,
                semantics=semantics,
            ).total

        multi_gradient = jax.grad(objective)(jnp.array(1.0), "multi_episode")
        first_gradient = jax.grad(objective)(jnp.array(1.0), "first_terminal")

        self.assertEqual(float(multi_gradient), 103.0)
        self.assertEqual(float(first_gradient), 3.0)

    def test_resume_requires_explicit_authority_for_semantic_change(self):
        self.assertEqual(
            resolve_actor_return_semantics(
                {"actor_return_semantics": "multi_episode"},
                requested="multi_episode",
                is_resume=True,
                allow_change=False,
            ),
            "multi_episode",
        )
        with self.assertRaisesRegex(ValueError, "explicit resume authority"):
            resolve_actor_return_semantics(
                {"actor_return_semantics": "multi_episode"},
                requested="first_terminal",
                is_resume=True,
                allow_change=False,
            )
        self.assertEqual(
            resolve_actor_return_semantics(
                {"actor_return_semantics": "multi_episode"},
                requested="first_terminal",
                is_resume=True,
                allow_change=True,
            ),
            "first_terminal",
        )

    def test_legacy_checkpoint_defaults_to_multi_episode(self):
        self.assertEqual(
            resolve_actor_return_semantics(
                {},
                requested="multi_episode",
                is_resume=True,
                allow_change=False,
            ),
            "multi_episode",
        )
        with self.assertRaisesRegex(ValueError, "explicit resume authority"):
            resolve_actor_return_semantics(
                {},
                requested="first_terminal",
                is_resume=True,
                allow_change=False,
            )

    def test_invalid_semantics_and_authority_fail_closed(self):
        for requested in (None, "first_done", 1):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    resolve_actor_return_semantics(
                        None,
                        requested=requested,
                        is_resume=False,
                        allow_change=False,
                    )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            resolve_actor_return_semantics(
                None,
                requested="multi_episode",
                is_resume=False,
                allow_change=1,
            )

    def test_telemetry_serializes_boundary_counts_and_rejects_bad_values(self):
        from src.algorithms.shac.algorithm import build_actor_return_telemetry

        metrics = {
            "actor_return_done_env_count": 2,
            "actor_return_done_event_count": 3,
            "actor_return_included_transition_count": 7,
            "actor_return_post_first_done_transition_count": 4,
            "actor_return_post_first_done_env_count": 2,
            "actor_return_mean": 1.25,
            "actor_return_post_first_done_reward_sum": 0.75,
            "actor_return_post_first_done_reward_mean": 0.1875,
        }

        result = build_actor_return_telemetry(metrics, semantics="first_terminal")

        self.assertEqual(result["actor_return_semantics"], "first_terminal")
        self.assertEqual(result["actor_return_done_event_count"], 3)
        with self.assertRaisesRegex(ValueError, "telemetry is invalid"):
            build_actor_return_telemetry(
                {**metrics, "actor_return_mean": float("nan")},
                semantics="first_terminal",
            )


if __name__ == "__main__":
    unittest.main()
