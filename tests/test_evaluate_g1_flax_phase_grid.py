from pathlib import Path


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
        actor_history_len=1,
        actor_observe_motion_anchor_position=True,
        tracking_velocity_kernel="pseudo_huber",
        actor_residual_preview_adapter=True,
        actor_residual_preview_hidden=256,
        actor_residual_preview_trainable_parameter_count=91_677,
        seed=0,
        code_provenance={
            "repository": "/tmp/repository",
            "code_commit": "c" * 40,
            "dirty_patch_sha256": "0" * 64,
        },
    )

    assert payload["summary"]["survival"] == [499, 399, 299, 199, 99]
    assert payload["summary"]["completed_suffix"] == [True] * 5
    assert payload["results"] == results
    assert payload["actor_history_len"] == 1
    assert payload["actor_reference_lookahead_steps"] == [4, 8, 12]
    assert payload["actor_reference_preview_mode"] == "delta"
    assert payload["actor_observe_motion_anchor_position"] is True
    assert payload["tracking_velocity_kernel"] == "pseudo_huber"
    assert payload["actor_residual_preview_adapter"] is True
    assert payload["actor_residual_preview_hidden"] == 256
    assert payload["actor_residual_preview_trainable_parameter_count"] == 91_677
    assert payload["actor_assistance_conditioning_scale"] == 0.0
    assert payload["post_policy_action_clip"] is True
    assert payload["seed"] == 0
    assert payload["code_provenance"]["code_commit"] == "c" * 40


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

    assert parser.parse_args(required).actor_reference_preview_mode is None
    assert parser.parse_args(required).code_commit is None
    assert parser.parse_args([*required, "--code-commit", "d" * 40]).code_commit == (
        "d" * 40
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
    gated = parser.parse_args(
        [
            *required,
            "--actor-residual-preview-adapter",
            "--actor-state-gated-recovery-support",
            "/tmp/support.npz",
        ]
    )
    assert gated.actor_state_gated_recovery_support == Path(
        "/tmp/support.npz"
    )


def test_phase_grid_loads_environment_contract_from_checkpoint_hparams(
    tmp_path,
):
    import json
    import pytest

    from tools.evaluate_g1_flax_phase_grid import (
        load_checkpoint_environment_contract,
    )

    checkpoint = tmp_path / "checkpoint_step_098304.pkl"
    checkpoint.write_bytes(b"checkpoint")
    hparams = {
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "reference_stride": 1,
        "actor_history_len": 10,
        "actor_reference_lookahead_steps": (4, 8, 12),
        "actor_reference_preview_mode": "delta",
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "solver_profile": "g1-4x5",
        "squash_actor_mean": False,
        "clip_sampled_actor_actions": False,
    }
    expected = {
        **hparams,
        "actor_reference_lookahead_steps": (4, 8, 12),
        "tracking_velocity_kernel": "exponential",
    }

    for stored, expected_value in ((None, False), (False, False), (True, True)):
        candidate = dict(hparams)
        candidate["actor_reference_lookahead_steps"] = [4, 8, 12]
        if stored is not None:
            candidate["actor_observe_motion_anchor_position"] = stored
        (tmp_path / "hparams.json").write_text(json.dumps(candidate))

        contract = load_checkpoint_environment_contract(checkpoint)

        assert contract == {
            **expected,
            "actor_observe_motion_anchor_position": expected_value,
        }

    candidate = dict(hparams)
    candidate["actor_reference_lookahead_steps"] = [4, 8, 12]
    candidate["tracking_velocity_kernel"] = "pseudo_huber"
    (tmp_path / "hparams.json").write_text(json.dumps(candidate))
    contract = load_checkpoint_environment_contract(checkpoint)
    assert contract["tracking_velocity_kernel"] == "pseudo_huber"

    candidate["tracking_velocity_kernel"] = "unknown"
    (tmp_path / "hparams.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="evaluation contract is invalid"):
        load_checkpoint_environment_contract(checkpoint)

    for invalid in ("false", 1, None):
        candidate = dict(hparams)
        candidate["actor_reference_lookahead_steps"] = [4, 8, 12]
        candidate["actor_observe_motion_anchor_position"] = invalid
        (tmp_path / "hparams.json").write_text(json.dumps(candidate))

        with pytest.raises(ValueError, match="evaluation contract is invalid"):
            load_checkpoint_environment_contract(checkpoint)


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


def test_gated_evaluator_routes_exact_pre_step_phase():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from src.algorithms.shac.progressive_recovery_expert import (
        RecoverySupport,
        apply_state_gated_recovery,
    )
    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
    )
    from src.core.networks import Actor
    from tools.evaluate_g1_flax_phase_grid import evaluate_gated_actor_action

    actor = Actor(2, hidden=(4,), squash=True, layer_norm=False)
    adapter = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    obs = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 10.0
    params = FrozenPreviewResidualParams(
        parent=actor.init(jax.random.PRNGKey(1), obs),
        adapter=adapter.init(
            jax.random.PRNGKey(2), jnp.zeros((1, 5), dtype=jnp.float32)
        ),
    )
    support = RecoverySupport(
        anchors=jnp.asarray([[1.0, 1.1, 1.2, 1.3, 1.4]]),
        radius=jnp.asarray(1.0),
        phase_min=8,
        phase_max=12,
        taper=2,
    )
    expected = apply_state_gated_recovery(
        actor,
        adapter,
        params,
        obs,
        jnp.asarray([10]),
        support,
        history_len=3,
        treatment_frame_dim=5,
    )
    actual = evaluate_gated_actor_action(
        actor,
        adapter,
        params,
        obs,
        jnp.asarray([10]),
        support,
        history_len=3,
        treatment_frame_dim=5,
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


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


def test_flax_phase_grid_main_does_not_hardcode_history_for_actor_paths():
    source = Path("tools/evaluate_g1_flax_phase_grid.py").read_text()

    assert "history_len=ACTOR_HISTORY_LEN" not in source
    assert source.count('history_len=contract["actor_history_len"]') >= 2
