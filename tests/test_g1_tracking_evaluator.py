import unittest
from unittest import mock
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from src.core.networks import Actor
from tools.evaluate_g1_tracking import (
    _load_policy,
    build_parser,
    configure_jax,
    load_rmr_policy,
    make_evaluation_env,
    remaining_reference_transitions,
    scale_policy_action,
    summarize_stability_errors,
)

RMR_CHECKPOINT = Path(
    "/home/ubuntu/projects/rmr_tracking/logs/rsl_rl/g1_flat/"
    "2026-06-10_11-11-49_walk_win137_212/model_4999.pt"
)
RMR_ACTIONS = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)


class G1TrackingEvaluatorTest(unittest.TestCase):
    def test_parser_defaults_to_complete_named_reference_suffix(self):
        args = build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/g1-evaluation",
                "--reference-path",
                "/tmp/dance.npz",
                "--reference-stride",
                "1",
            ]
        )

        self.assertIsNone(args.max_steps)
        self.assertEqual(args.reference_path, Path("/tmp/dance.npz"))
        self.assertEqual(args.reference_stride, 1)
        self.assertEqual(args.actor_reference_lookahead_steps, ())
        self.assertEqual(args.actor_reference_preview_mode, "absolute")
        self.assertEqual(
            build_parser().parse_args(
                [
                    "--output-dir",
                    "/tmp/g1-evaluation",
                    "--actor-reference-preview-mode",
                    "delta",
                ]
            ).actor_reference_preview_mode,
            "delta",
        )

    def test_complete_suffix_uses_every_carried_reference_transition(self):
        self.assertEqual(remaining_reference_transitions(500, 0, 1), 499)
        self.assertEqual(remaining_reference_transitions(500, 120, 1), 379)
        self.assertEqual(remaining_reference_transitions(121, 0, 2), 60)

    def test_stability_summary_reports_maximum_termination_errors(self):
        summary = summarize_stability_errors(
            {
                "anchor_z_error": np.asarray([0.01, 0.12]),
                "anchor_xy_error": np.asarray([0.04, 0.08]),
                "gravity_z_error": np.asarray([0.10, 0.31]),
                "distal_z_error": np.asarray([0.03, 0.09]),
            }
        )

        self.assertEqual(summary["max_anchor_z_error"], 0.12)
        self.assertEqual(summary["max_anchor_xy_error"], 0.08)
        self.assertEqual(summary["max_gravity_z_error"], 0.31)
        self.assertEqual(summary["max_distal_z_error"], 0.09)

    def test_checkpoint_loader_infers_compact_actor_architecture(self):
        env = SimpleNamespace(
            action_dim=3,
            squash_actor_actions=False,
            actor_obs_dim=7,
            actor_frame_obs_dim=7,
        )
        expected_actor = Actor(
            3,
            hidden=(512, 512),
            squash=False,
            layer_norm=False,
            zero_output=False,
        )
        params = expected_actor.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 7), dtype=jnp.float32),
        )
        state = SimpleNamespace(actor_params=params, normalizer="normalizer")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            with checkpoint.open("wb") as handle:
                pickle.dump(state, handle)
            actor, loaded_params, normalizer = _load_policy(
                env, checkpoint, seed=123
            )

        self.assertEqual(tuple(actor.hidden), (512, 512))
        self.assertFalse(actor.layer_norm)
        self.assertFalse(actor.squash)
        self.assertEqual(normalizer, "normalizer")
        np.testing.assert_allclose(
            actor.apply(loaded_params, jnp.zeros((1, 7))),
            expected_actor.apply(params, jnp.zeros((1, 7))),
        )

    def test_checkpoint_loader_applies_conditioned_residual_with_zero_assistance(self):
        from src.algorithms.shac.residual_preview_adapter import (
            FrozenPreviewResidualParams,
            PreviewResidualAdapter,
            apply_frozen_preview_residual,
        )

        env = SimpleNamespace(
            action_dim=2,
            squash_actor_actions=True,
            actor_obs_dim=15,
            actor_frame_obs_dim=5,
            actor_history_len=3,
        )
        parent = Actor(
            2,
            hidden=(4,),
            squash=True,
            layer_norm=False,
            zero_output=False,
        )
        residual = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
        parent_params = parent.init(
            jax.random.PRNGKey(11), jnp.zeros((1, 15), dtype=jnp.float32)
        )
        adapter_params = residual.init(
            jax.random.PRNGKey(12), jnp.zeros((1, 6), dtype=jnp.float32)
        )
        params = FrozenPreviewResidualParams(parent_params, adapter_params)
        state = SimpleNamespace(actor_params=params, normalizer="normalizer")
        observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            with checkpoint.open("wb") as handle:
                pickle.dump(state, handle)
            actor, loaded_params, normalizer = _load_policy(
                env, checkpoint, seed=0
            )

        expected, _, _ = apply_frozen_preview_residual(
            parent,
            residual,
            params,
            observations,
            history_len=3,
            treatment_frame_dim=5,
            assistance_scale=jnp.asarray(0.0),
        )
        np.testing.assert_array_equal(
            actor.apply(loaded_params, observations), expected
        )
        self.assertEqual(normalizer, "normalizer")

    def test_checkpoint_loader_applies_legacy_residual_without_assistance_input(self):
        from src.algorithms.shac.residual_preview_adapter import (
            FrozenPreviewResidualParams,
            PreviewResidualAdapter,
            apply_frozen_preview_residual,
        )

        env = SimpleNamespace(
            action_dim=2,
            squash_actor_actions=True,
            actor_obs_dim=15,
            actor_frame_obs_dim=5,
            actor_history_len=3,
        )
        parent = Actor(
            2,
            hidden=(4,),
            squash=True,
            layer_norm=False,
            zero_output=False,
        )
        residual = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
        parent_params = parent.init(
            jax.random.PRNGKey(21), jnp.zeros((1, 15), dtype=jnp.float32)
        )
        adapter_params = residual.init(
            jax.random.PRNGKey(22), jnp.zeros((1, 5), dtype=jnp.float32)
        )
        params = FrozenPreviewResidualParams(parent_params, adapter_params)
        state = SimpleNamespace(actor_params=params, normalizer="normalizer")
        observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            with checkpoint.open("wb") as handle:
                pickle.dump(state, handle)
            actor, loaded_params, normalizer = _load_policy(
                env, checkpoint, seed=0
            )

        expected, _, _ = apply_frozen_preview_residual(
            parent,
            residual,
            params,
            observations,
            history_len=3,
            treatment_frame_dim=5,
            assistance_scale=None,
        )
        np.testing.assert_array_equal(
            actor.apply(loaded_params, observations), expected
        )
        self.assertEqual(normalizer, "normalizer")

    def test_loader_recreates_exact_compact_training_initialization(self):
        env = SimpleNamespace(
            action_dim=3,
            squash_actor_actions=False,
            actor_obs_dim=7,
            actor_frame_obs_dim=7,
        )
        actor, params, _normalizer = _load_policy(
            env,
            checkpoint=None,
            seed=0,
            actor_hidden=(512, 512),
            actor_layer_norm=False,
            actor_zero_output=False,
            training_initialization=True,
        )
        _unused, actor_key, _critic_key, _env_key = jax.random.split(
            jax.random.PRNGKey(0), 4
        )
        expected = actor.init(
            actor_key, jnp.zeros((1, 7), dtype=jnp.float32)
        )

        self.assertEqual(tuple(actor.hidden), (512, 512))
        self.assertFalse(actor.layer_norm)
        np.testing.assert_allclose(
            actor.apply(params, jnp.zeros((1, 7))),
            actor.apply(expected, jnp.zeros((1, 7))),
        )

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

    def test_action_parity_evaluation_preserves_unbounded_actor(self):
        env = make_evaluation_env(
            "g1_tracking_rmr_50hz_action_parity",
            actor_history_len=10,
            actor_reference_lookahead_steps=(4, 8, 12),
            actor_reference_preview_mode="delta",
            reference_residual_control=True,
            reference_residual_scale=1.0,
        )

        self.assertFalse(env.squash_actor_actions)
        self.assertEqual(env.actor_history_len, 10)
        self.assertEqual(env.reference_residual_scale, 1.0)

    def test_source_step_evaluator_can_screen_solver_budgets(self):
        candidate = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step",
            solver_iterations=4,
            solver_ls_iterations=10,
        )

        self.assertEqual(candidate.mj_model.opt.iterations, 4)
        self.assertEqual(candidate.mj_model.opt.ls_iterations, 10)

    def test_evaluator_can_recreate_canonical_actor_observation_and_control(self):
        candidate = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step",
            actor_history_len=10,
            reference_residual_control=True,
            reference_residual_scale=0.5,
        )

        self.assertEqual(candidate.actor_history_len, 10)
        self.assertEqual(candidate.actor_obs_dim, 1540)
        self.assertTrue(candidate.reference_residual_control)
        self.assertEqual(candidate.reference_residual_scale, 0.5)
        self.assertFalse(candidate.actor_observation_noise)

    def test_evaluator_can_recreate_future_reference_actor_observation(self):
        candidate = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step",
            actor_history_len=10,
            actor_reference_lookahead_steps=(4, 8, 12),
            reference_residual_control=True,
            reference_residual_scale=0.5,
        )

        self.assertEqual(
            candidate.actor_reference_lookahead_steps, (4, 8, 12)
        )
        self.assertEqual(candidate.actor_frame_obs_dim, 328)
        self.assertEqual(candidate.actor_obs_dim, 3280)

    def test_evaluator_forwards_delta_future_reference_mode(self):
        candidate = make_evaluation_env(
            "g1_tracking_rmr_50hz_source_step",
            actor_history_len=10,
            actor_reference_lookahead_steps=(4, 8, 12),
            actor_reference_preview_mode="delta",
            reference_residual_control=True,
            reference_residual_scale=0.5,
        )

        self.assertEqual(candidate.actor_reference_preview_mode, "delta")
        self.assertEqual(candidate.actor_obs_dim, 3280)

    def test_validated_evaluator_uses_smallest_passing_solver_budget(self):
        validated = make_evaluation_env(
            "g1_tracking_rmr_50hz_validated"
        )

        self.assertEqual(validated.mj_model.opt.iterations, 4)
        self.assertEqual(validated.mj_model.opt.ls_iterations, 5)

    def test_evaluator_transports_a_fixed_body_mass_scale(self):
        shifted = make_evaluation_env(
            "g1_tracking_rmr_50hz_validated",
            body_mass_scale=1.15,
        )

        self.assertEqual(shifted.body_mass_scale, 1.15)

    def test_evaluator_cli_accepts_a_fixed_body_mass_scale(self):
        from tools.evaluate_g1_tracking import build_parser

        args = build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/g1-evaluation",
                "--body-mass-scale",
                "1.15",
            ]
        )

        self.assertEqual(args.body_mass_scale, 1.15)

    def test_evaluator_cli_accepts_named_solver_profile(self):
        args = build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/g1-evaluation",
                "--solver-profile",
                "g1-4x5",
            ]
        )

        self.assertEqual(args.solver_profile, "g1-4x5")

    def test_evaluator_transports_a_fixed_effort_limit_scale(self):
        shifted = make_evaluation_env(
            "g1_tracking_rmr_50hz_validated",
            effort_limit_scale=0.7,
        )

        self.assertEqual(shifted.effort_limit_scale, 0.7)

    def test_evaluator_cli_accepts_a_fixed_effort_limit_scale(self):
        from tools.evaluate_g1_tracking import build_parser

        args = build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/g1-evaluation",
                "--effort-limit-scale",
                "0.7",
            ]
        )

        self.assertEqual(args.effort_limit_scale, 0.7)

    def test_action_gain_scales_policy_without_changing_direction(self):
        action = jnp.array([-0.8, 0.2, 1.0])
        np.testing.assert_allclose(
            scale_policy_action(action, 0.25),
            np.array([-0.2, 0.05, 0.25]),
        )

    def test_action_gain_must_interpolate_zero_and_learned_policy(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            scale_policy_action(jnp.zeros(3), 1.1)

    def test_evaluator_cli_accepts_standalone_full_rmr_actor_checkpoint(self):
        from tools.evaluate_g1_tracking import build_parser

        args = build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/g1-evaluation",
                "--checkpoint",
                "/tmp/full-actor.pkl",
                "--full-rmr-actor",
            ]
        )

        self.assertTrue(args.full_rmr_actor)


if __name__ == "__main__":
    unittest.main()
