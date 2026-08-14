def _row(phase, steps, terminal=False):
    return {
        "phase": phase,
        "steps": steps,
        "terminal": terminal,
        "mean_reward": 1.0,
        "mean_anchor_position_error": 0.1,
        "mean_anchor_orientation_error": 0.2,
        "mean_body_position_error": 0.3,
        "mean_body_orientation_error": 0.4,
        "mean_body_linear_velocity_error": 0.5,
        "mean_body_angular_velocity_error": 0.6,
    }


def test_flax_phase_grid_payload_records_exact_suffix_completion():
    from tools.evaluate_g1_flax_phase_grid import build_payload

    phases = (0, 100, 200, 300, 400)
    results = [_row(phase, 499 - phase) for phase in phases]
    payload = build_payload(
        results,
        phases=phases,
        reference_transitions=499,
        checkpoint_path="/tmp/checkpoint_step_1.pkl",
        checkpoint_sha256="a" * 64,
        reference_path="/tmp/reference.npz",
        reference_sha256="b" * 64,
        solver_profile="g1-4x5",
        actor_reference_preview_mode="delta",
        actor_residual_preview_adapter=True,
        actor_residual_preview_hidden=256,
        actor_residual_preview_trainable_parameter_count=91_677,
    )

    assert payload["summary"]["survival"] == [499, 399, 299, 199, 99]
    assert payload["summary"]["completed_suffix"] == [True] * 5
    assert payload["results"] == results
    assert payload["actor_history_len"] == 10
    assert payload["actor_reference_lookahead_steps"] == [4, 8, 12]
    assert payload["actor_reference_preview_mode"] == "delta"
    assert payload["actor_residual_preview_adapter"] is True
    assert payload["actor_residual_preview_hidden"] == 256
    assert payload["actor_residual_preview_trainable_parameter_count"] == 91_677
    assert payload["actor_assistance_conditioning_scale"] == 0.0
    assert payload["post_policy_action_clip"] is True


def test_flax_phase_grid_parser_defaults_absolute_and_accepts_delta():
    from tools.evaluate_g1_flax_phase_grid import build_parser

    required = [
        "--checkpoint",
        "/tmp/checkpoint.pkl",
        "--reference-path",
        "/tmp/reference.npz",
        "--output",
        "/tmp/summary.json",
    ]
    parser = build_parser()

    assert (
        parser.parse_args(required).actor_reference_preview_mode
        == "absolute"
    )
    assert (
        parser.parse_args(
            [*required, "--actor-reference-preview-mode", "delta"]
        ).actor_reference_preview_mode
        == "delta"
    )
    residual = parser.parse_args(
        [*required, "--actor-residual-preview-adapter"]
    )
    assert residual.actor_residual_preview_adapter is True
    assert residual.actor_residual_preview_hidden == 256
    upstream = parser.parse_args(
        [
            *required,
            "--env-variant",
            "g1_tracking_rmr_50hz_upstream_action_penalty",
        ]
    )
    assert upstream.env_variant == (
        "g1_tracking_rmr_50hz_upstream_action_penalty"
    )


def test_evaluator_residual_action_matches_training_composition():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
    )
    from src.core.networks import Actor
    from tools.evaluate_g1_flax_phase_grid import evaluate_actor_action

    parent_actor = Actor(
        action_dim=2,
        hidden=(4,),
        squash=True,
        layer_norm=False,
        zero_output=False,
    )
    residual_actor = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 10.0
    params = FrozenPreviewResidualParams(
        parent=parent_actor.init(jax.random.PRNGKey(1), observations),
        adapter=residual_actor.init(
            jax.random.PRNGKey(2), jnp.zeros((1, 5), dtype=jnp.float32)
        ),
    )
    training_action, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        params,
        observations,
        history_len=3,
        treatment_frame_dim=5,
    )

    evaluation_action = evaluate_actor_action(
        parent_actor,
        params,
        observations,
        residual_actor=residual_actor,
        history_len=3,
        treatment_frame_dim=5,
    )

    np.testing.assert_array_equal(evaluation_action, training_action)


def test_phase_grid_applies_training_post_policy_boundary():
    import jax.numpy as jnp
    import numpy as np

    from tools.evaluate_g1_flax_phase_grid import prepare_phase_grid_action

    raw = jnp.asarray([-1.4, -0.2, 1.7])

    np.testing.assert_array_equal(
        prepare_phase_grid_action(raw, clip_sampled_actor_actions=True),
        jnp.asarray([-1.0, -0.2, 1.0]),
    )
    np.testing.assert_array_equal(
        prepare_phase_grid_action(raw, clip_sampled_actor_actions=False),
        raw,
    )


def test_flax_phase_grid_routes_rollouts_through_one_compiled_step():
    from pathlib import Path

    source = Path("tools/evaluate_g1_flax_phase_grid.py").read_text()

    assert "compiled_step = build_compiled_step(env)" in source
    assert source.count("step_fn=compiled_step") == 1
