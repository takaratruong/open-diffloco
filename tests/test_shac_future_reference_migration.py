import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.core.data_structures import EnvState, Normalizer, TrainState
from src.core.networks import Actor


HISTORY_LEN = 10
LEGACY_FRAME_DIM = 154
TREATMENT_FRAME_DIM = 328
ACTION_DIM = 3


def _future_reference_command(phase):
    return jnp.arange(174, dtype=jnp.float32) + phase.astype(jnp.float32)


def _build_states():
    actor = Actor(
        ACTION_DIM,
        hidden=(8,),
        squash=True,
        layer_norm=False,
        zero_output=False,
    )
    key = jax.random.PRNGKey(4)
    legacy_params = actor.init(
        key, jnp.zeros((1, HISTORY_LEN * LEGACY_FRAME_DIM))
    )
    treatment_params = actor.init(
        key, jnp.zeros((1, HISTORY_LEN * TREATMENT_FRAME_DIM))
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(1e-3)
    )
    legacy_opt = optimizer.init(legacy_params)
    _, legacy_opt = optimizer.update(
        jax.tree_util.tree_map(jnp.ones_like, legacy_params),
        legacy_opt,
        legacy_params,
    )
    treatment_opt = optimizer.init(treatment_params)

    batch_size = 2
    history = jnp.arange(
        batch_size * HISTORY_LEN * LEGACY_FRAME_DIM,
        dtype=jnp.float32,
    ).reshape(batch_size, HISTORY_LEN, LEGACY_FRAME_DIM)
    env_state = EnvState(
        data={"sentinel": jnp.arange(batch_size)},
        obs=history.reshape(batch_size, -1),
        reward=jnp.zeros(batch_size),
        done=jnp.zeros(batch_size),
        info={
            "phase": jnp.array([37, 3], dtype=jnp.int32),
            "step": jnp.array([9, 0], dtype=jnp.int32),
            "actor_obs_history": history,
            "bootstrap_obs": (history + 0.5).reshape(batch_size, -1),
        },
        metrics={"sentinel": jnp.ones(batch_size)},
    )
    normalizer = Normalizer(LEGACY_FRAME_DIM).init().replace(
        mean=jnp.linspace(-1.0, 1.0, LEGACY_FRAME_DIM),
        var=jnp.linspace(0.5, 1.5, LEGACY_FRAME_DIM),
        count=jnp.array(1234.0),
    )
    common = {
        "key": jax.random.PRNGKey(8),
        "env_state": env_state,
        "critic_params": {"critic": jnp.array([2.0])},
        "target_critic_params": {"critic": jnp.array([3.0])},
        "critic_normalizer": Normalizer(2).init(),
        "critic_opt": (jnp.array(5.0),),
        "step": jnp.array(1_179_648),
    }
    resumed = TrainState(
        actor_params=legacy_params,
        normalizer=normalizer,
        actor_opt=legacy_opt,
        **common,
    )
    initialized = TrainState(
        actor_params=treatment_params,
        normalizer=Normalizer(TREATMENT_FRAME_DIM).init(),
        actor_opt=treatment_opt,
        **common,
    )
    env = types.SimpleNamespace(
        actor_history_len=HISTORY_LEN,
        actor_frame_obs_dim=TREATMENT_FRAME_DIM,
        actor_future_reference_dim=174,
        actor_reference_lookahead_steps=(4, 8, 12),
        reference_stride=1,
        reference_length=64,
        _future_reference_command=_future_reference_command,
    )
    return actor, resumed, initialized, env


class FutureReferenceMigrationTest(unittest.TestCase):
    def test_expands_each_history_block_without_moving_legacy_rows(self):
        from src.algorithms.shac.future_reference_migration import (
            expand_history_input_rows,
        )

        value = jnp.arange(
            HISTORY_LEN * LEGACY_FRAME_DIM * 2, dtype=jnp.float32
        ).reshape(HISTORY_LEN * LEGACY_FRAME_DIM, 2)

        expanded = expand_history_input_rows(
            value,
            history_len=HISTORY_LEN,
            old_frame_dim=LEGACY_FRAME_DIM,
            new_frame_dim=TREATMENT_FRAME_DIM,
        ).reshape(HISTORY_LEN, TREATMENT_FRAME_DIM, 2)

        np.testing.assert_array_equal(
            expanded[:, :LEGACY_FRAME_DIM],
            np.asarray(value).reshape(HISTORY_LEN, LEGACY_FRAME_DIM, 2),
        )
        np.testing.assert_array_equal(expanded[:, LEGACY_FRAME_DIM:], 0.0)

    def test_migration_preserves_actor_output_and_non_actor_state(self):
        from src.algorithms.shac.future_reference_migration import (
            future_reference_migration_report,
            migrate_future_reference_train_state,
            validate_future_reference_migration_report,
        )

        actor, resumed, initialized, env = _build_states()

        migrated = migrate_future_reference_train_state(
            resumed, initialized, env, expected_history_len=HISTORY_LEN
        )
        report = future_reference_migration_report(
            resumed,
            migrated,
            actor,
            legacy_frame_dim=LEGACY_FRAME_DIM,
            treatment_frame_dim=TREATMENT_FRAME_DIM,
            history_len=HISTORY_LEN,
        )
        validate_future_reference_migration_report(report)

        old_kernel = resumed.actor_params["params"]["Dense_0"][
            "kernel"
        ].reshape(HISTORY_LEN, LEGACY_FRAME_DIM, -1)
        new_kernel = migrated.actor_params["params"]["Dense_0"][
            "kernel"
        ].reshape(HISTORY_LEN, TREATMENT_FRAME_DIM, -1)
        np.testing.assert_array_equal(
            new_kernel[:, :LEGACY_FRAME_DIM], old_kernel
        )
        np.testing.assert_array_equal(
            new_kernel[:, LEGACY_FRAME_DIM:], 0.0
        )

        old_mu = resumed.actor_opt[1][0].mu["params"]["Dense_0"][
            "kernel"
        ].reshape(HISTORY_LEN, LEGACY_FRAME_DIM, -1)
        new_mu = migrated.actor_opt[1][0].mu["params"]["Dense_0"][
            "kernel"
        ].reshape(HISTORY_LEN, TREATMENT_FRAME_DIM, -1)
        np.testing.assert_array_equal(new_mu[:, :LEGACY_FRAME_DIM], old_mu)
        np.testing.assert_array_equal(new_mu[:, LEGACY_FRAME_DIM:], 0.0)
        old_nu = resumed.actor_opt[1][0].nu["params"]["Dense_0"][
            "kernel"
        ].reshape(HISTORY_LEN, LEGACY_FRAME_DIM, -1)
        new_nu = migrated.actor_opt[1][0].nu["params"]["Dense_0"][
            "kernel"
        ].reshape(HISTORY_LEN, TREATMENT_FRAME_DIM, -1)
        np.testing.assert_array_equal(new_nu[:, :LEGACY_FRAME_DIM], old_nu)
        np.testing.assert_array_equal(new_nu[:, LEGACY_FRAME_DIM:], 0.0)

        self.assertEqual(migrated.env_state.obs.shape, (2, 3280))
        self.assertEqual(
            migrated.env_state.info["actor_obs_history"].shape,
            (2, 10, 328),
        )
        self.assertEqual(
            migrated.env_state.info["bootstrap_obs"].shape, (2, 3280)
        )
        np.testing.assert_array_equal(
            migrated.normalizer.mean[:LEGACY_FRAME_DIM],
            resumed.normalizer.mean,
        )
        np.testing.assert_array_equal(
            migrated.normalizer.var[:LEGACY_FRAME_DIM],
            resumed.normalizer.var,
        )
        self.assertTrue(
            np.isfinite(
                np.asarray(migrated.normalizer.mean[LEGACY_FRAME_DIM:])
            ).all()
        )
        self.assertTrue(
            np.isfinite(
                np.asarray(migrated.normalizer.var[LEGACY_FRAME_DIM:])
            ).all()
        )
        np.testing.assert_array_equal(
            migrated.critic_params, resumed.critic_params
        )
        np.testing.assert_array_equal(
            migrated.target_critic_params, resumed.target_critic_params
        )
        np.testing.assert_array_equal(migrated.key, resumed.key)
        self.assertEqual(int(migrated.step), int(resumed.step))
        self.assertTrue(report["valid"])
        self.assertTrue(report["non_input_actor_params_exact"])
        self.assertTrue(report["legacy_optimizer_rows_exact"])
        self.assertTrue(report["new_optimizer_rows_zero"])
        self.assertTrue(report["other_optimizer_leaves_exact"])
        self.assertLessEqual(report["max_action_absolute_error"], 1e-7)
        self.assertLessEqual(report["max_action_relative_error"], 1e-7)

    def test_migration_fails_closed_on_unsupported_shapes(self):
        from src.algorithms.shac.future_reference_migration import (
            expand_history_input_rows,
            migrate_future_reference_train_state,
        )

        _, resumed, initialized, env = _build_states()
        with self.assertRaisesRegex(ValueError, "history length"):
            migrate_future_reference_train_state(
                resumed, initialized, env, expected_history_len=1
            )
        with self.assertRaisesRegex(ValueError, "input-row shape"):
            expand_history_input_rows(
                jnp.zeros((17, 3)),
                history_len=HISTORY_LEN,
                old_frame_dim=LEGACY_FRAME_DIM,
                new_frame_dim=TREATMENT_FRAME_DIM,
            )

    def test_report_validation_rejects_action_drift(self):
        from src.algorithms.shac.future_reference_migration import (
            validate_future_reference_migration_report,
        )

        with self.assertRaisesRegex(ValueError, "equivalence"):
            validate_future_reference_migration_report(
                {
                    "valid": False,
                    "max_action_absolute_error": 1e-3,
                    "max_action_relative_error": 1e-3,
                }
            )


if __name__ == "__main__":
    unittest.main()
