import hashlib
import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.networks import Actor
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualParams,
)
from src.algorithms.shac.counterfactual_wrench_distillation import (
    resolve_leg_action_indices,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
)
from src.core.rmr_action_noise import RMR_ACTION_STD_JOINT_NAMES
from tools.run_g1_counterfactual_wrench_distillation import (
    END_STEP,
    START_STEP,
    build_counterfactual_kwargs,
    expected_checkpoint_steps,
    validate_training_artifacts,
)
from tools.evaluate_g1_tracking import _load_policy


def test_registered_budget_is_exactly_32_effective512_h24_updates():
    assert START_STEP == 1_769_472
    assert END_STEP == START_STEP + 32 * 512 * 24
    assert expected_checkpoint_steps() == (
        1_867_776,
        1_966_080,
        2_064_384,
        2_162_688,
    )


def test_builder_enables_only_registered_counterfactual_treatment(tmp_path):
    teacher = tmp_path / "teacher.pkl"
    feasibility = tmp_path / "feasibility.json"
    teacher.write_bytes(b"teacher")
    feasibility.write_text("{}")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = build_counterfactual_kwargs(
        "g1-4x5",
        "/reference.npz",
        0,
        "/e026.pkl",
        teacher,
        sha(teacher),
        feasibility,
        sha(feasibility),
    )
    assert kwargs["actor_counterfactual_wrench_distillation"] is True
    assert kwargs["actor_frozen_controller_residual"] is True
    assert kwargs["actor_cagrad"] is True
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["num_envs"] == 256
    assert kwargs["unroll_length"] == 24
    assert kwargs["actor_capture_point_tracking"] is False
    assert kwargs["actor_centroidal_propulsion"] is False
    assert kwargs["actor_learned_torso_wrench"] is False
    assert kwargs["torso_wrench_assistance"] is False
    assert kwargs["domain_randomization"] is False
    assert kwargs["actor_observation_noise"] is False
    assert kwargs["push_velocity_range"] == (0.0, 0.0)
    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_steps"] == expected_checkpoint_steps()


def _finite_adapter():
    actor = PreviewResidualAdapter(action_dim=12, hidden_dim=256)
    return actor.init(jax.random.PRNGKey(0), jnp.zeros((1, 328)))


def test_validation_rejects_nonzero_nonleg_or_student_wrench(tmp_path):
    source = SimpleNamespace(
        actor_params={"p": np.ones(1)},
        normalizer={"n": np.ones(1)},
    )
    source_path = tmp_path / "source.pkl"
    with source_path.open("wb") as stream:
        pickle.dump(source, stream)
    run = tmp_path / "run"
    run.mkdir()
    (run / "hparams.json").write_text(json.dumps({
        "actor_counterfactual_wrench_distillation": True,
        "actor_counterfactual_wrench_teacher_sha256": "a" * 64,
        "actor_counterfactual_wrench_feasibility_sha256": "b" * 64,
        "actor_counterfactual_wrench_leg_indices": list(
            resolve_leg_action_indices(RMR_ACTION_STD_JOINT_NAMES)
        ),
        "actor_counterfactual_wrench_loss_weight": 1.0,
        "actor_counterfactual_wrench_residual_magnitude_weight": 0.01,
        "actor_counterfactual_wrench_residual_temporal_weight": 0.001,
        "action_noise_std_end": 0.2,
        "actor_frozen_controller_residual": True,
        "actor_capture_point_tracking": False,
        "actor_capture_point_delta": 0.1,
        "actor_capture_point_weight": 1.0,
        "actor_learned_torso_wrench": False,
        "torso_wrench_assistance": False,
        "total_steps": END_STEP,
    }))
    rows = []
    for step in expected_checkpoint_steps():
        state = SimpleNamespace(
            step=step,
            actor_params=FrozenControllerResidualParams(
                source.actor_params, _finite_adapter()
            ),
            normalizer=source.normalizer,
        )
        with (run / f"checkpoint_step_{step}.pkl").open("wb") as stream:
            pickle.dump(state, stream)
        rows.append({
            "step": step,
            "actor_preview_valid": True,
            "actor_preview_gradient_norm": 1.0,
            "actor_preview_update_norm": 1.0,
            "actor_preview_frozen_parameter_drift_max_abs": 0.0,
            "actor_preview_frozen_moment_drift_max_abs": 0.0,
            "actor_preview_normalizer_drift_max_abs": 0.0,
            "actor_counterfactual_valid": True,
            "actor_counterfactual_valid_count": 1,
            "actor_counterfactual_invalid_count": 0,
            "actor_counterfactual_nonleg_max_abs": 1e-6,
            "actor_counterfactual_student_wrench_max_abs": 0.0,
            "action_noise_current": 0.2,
            "actor_bootstrap_scale_current": 0.0,
            "actor_cagrad_bin_counts": [1, 1, 1, 1, 1],
            "actor_cagrad_bin_gradient_norms": [0.5] * 5,
            "actor_cagrad_bin_losses": [0.1] * 5,
            "actor_cagrad_weights": [0.2] * 5,
            "actor_cagrad_gram_matrix": np.eye(5).tolist(),
            "actor_cagrad_cosine_matrix": np.eye(5).tolist(),
            "actor_cagrad_objective": 0.1,
            "actor_cagrad_dual_gap": 0.0,
            "actor_cagrad_uniform_combined_cosine": 0.5,
            "actor_cagrad_combined_norm": 0.5,
            "actor_cagrad_valid": True,
            **{
                f"actor_counterfactual_{name}": 0.1
                for name in (
                    "loss", "base_linear_loss", "base_angular_loss",
                    "centroidal_linear_loss", "centroidal_angular_loss",
                    "cosine", "student_rms", "teacher_rms",
                    "normalized_error_rms", "residual_rms",
                    "residual_max_abs", "residual_bound_fraction",
                    "teacher_wrench_rms",
                )
            },
        })
    (run / "checkpoint_phase_metrics.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="counterfactual"):
        validate_training_artifacts(
            run,
            source_checkpoint=source_path,
            teacher_sha256="a" * 64,
            feasibility_sha256="b" * 64,
        )


def test_tracking_evaluator_scatter_loads_twelve_leg_adapter(tmp_path):
    env = SimpleNamespace(
        action_dim=29,
        squash_actor_actions=False,
        squash_actor_mean=False,
        actor_obs_dim=15,
        actor_frame_obs_dim=5,
        actor_history_len=3,
        actor_joint_names=RMR_ACTION_STD_JOINT_NAMES,
    )
    parent_actor = Actor(29, hidden=(4,), squash=False, zero_output=False)
    e026_actor = PreviewResidualAdapter(action_dim=29, hidden_dim=4)
    leg_actor = PreviewResidualAdapter(action_dim=12, hidden_dim=4)
    parent_params = parent_actor.init(
        jax.random.PRNGKey(1), jnp.zeros((1, 15))
    )
    e026_params = e026_actor.init(
        jax.random.PRNGKey(2), jnp.zeros((1, 5))
    )
    leg_params = leg_actor.init(
        jax.random.PRNGKey(3), jnp.zeros((1, 5))
    )
    params = FrozenControllerResidualParams(
        FrozenPreviewResidualParams(parent_params, e026_params),
        leg_params,
    )
    checkpoint = tmp_path / "checkpoint.pkl"
    with checkpoint.open("wb") as stream:
        pickle.dump(SimpleNamespace(actor_params=params, normalizer="n"), stream)
    actor, loaded, normalizer = _load_policy(env, checkpoint, seed=0)
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15)

    def apply_e026(e026, obs):
        return apply_frozen_preview_residual(
            parent_actor,
            e026_actor,
            e026,
            obs,
            history_len=3,
            treatment_frame_dim=5,
        )[0]

    from src.algorithms.shac.frozen_controller_residual import (
        apply_frozen_controller_residual,
    )

    expected = apply_frozen_controller_residual(
        apply_e026,
        leg_actor,
        params,
        observations,
        history_len=3,
        frame_dim=5,
        residual_action_indices=resolve_leg_action_indices(
            RMR_ACTION_STD_JOINT_NAMES
        ),
    )[0]
    np.testing.assert_array_equal(actor.apply(loaded, observations), expected)
    assert normalizer == "n"
