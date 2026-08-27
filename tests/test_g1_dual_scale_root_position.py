from pathlib import Path
import json
import pickle

import jax
import numpy as np
import pytest


def test_pair_has_exact_short_checkpoint_grid() -> None:
    from tools.run_g1_dual_scale_root_position import (
        END_STEP,
        START_STEP,
        expected_checkpoint_steps,
    )

    assert START_STEP == 1_867_776
    assert END_STEP == 2_064_384
    assert expected_checkpoint_steps() == (
        1_916_928,
        1_966_080,
        2_015_232,
        2_064_384,
    )


def test_pair_arms_differ_only_in_registered_kernel_authority() -> None:
    from tools.run_g1_dual_scale_root_position import build_arm_kwargs

    checkpoint = Path("/tmp/checkpoint_step_1867776.pkl")
    control = build_arm_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, kernel="exponential"
    )
    treatment = build_arm_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, kernel="dual_scale"
    )

    assert control["tracking_anchor_position_kernel"] == "exponential"
    assert control["allow_resume_tracking_anchor_position_kernel_change"] is False
    assert treatment["tracking_anchor_position_kernel"] == "dual_scale"
    assert treatment["allow_resume_tracking_anchor_position_kernel_change"] is True
    assert treatment["tracking_root_velocity_weight"] == 1.0
    assert treatment["allow_resume_tracking_root_velocity_change"] is False
    for key in (
        "tracking_anchor_position_kernel",
        "allow_resume_tracking_anchor_position_kernel_change",
    ):
        control.pop(key)
        treatment.pop(key)
    assert control.keys() == treatment.keys()
    for key in control:
        control_leaves, control_tree = jax.tree_util.tree_flatten(control[key])
        treatment_leaves, treatment_tree = jax.tree_util.tree_flatten(
            treatment[key]
        )
        assert control_tree == treatment_tree, key
        for actual, expected in zip(
            control_leaves, treatment_leaves, strict=True
        ):
            np.testing.assert_array_equal(actual, expected, err_msg=key)


def test_quadratic_arm_is_an_explicit_single_kernel_delta() -> None:
    from tools.run_g1_dual_scale_root_position import build_arm_kwargs

    checkpoint = Path("/tmp/checkpoint_step_1867776.pkl")
    control = build_arm_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, kernel="exponential"
    )
    treatment = build_arm_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, kernel="quadratic"
    )

    assert treatment["tracking_anchor_position_kernel"] == "quadratic"
    assert treatment["allow_resume_tracking_anchor_position_kernel_change"] is True
    for key in (
        "tracking_anchor_position_kernel",
        "allow_resume_tracking_anchor_position_kernel_change",
    ):
        control.pop(key)
        treatment.pop(key)
    assert control.keys() == treatment.keys()
    for key in control:
        control_leaves, control_tree = jax.tree_util.tree_flatten(control[key])
        treatment_leaves, treatment_tree = jax.tree_util.tree_flatten(
            treatment[key]
        )
        assert control_tree == treatment_tree, key
        for actual, expected in zip(
            control_leaves, treatment_leaves, strict=True
        ):
            np.testing.assert_array_equal(actual, expected, err_msg=key)


def _records(*survivals):
    from tools.run_g1_dual_scale_root_position import expected_checkpoint_steps

    return {
        step: {"checkpoint_sha256": f"{index + 1:064x}", "survival": list(row)}
        for index, (step, row) in enumerate(
            zip(expected_checkpoint_steps(), survivals, strict=True)
        )
    }


def test_selector_requires_safe_treatment_and_beats_control() -> None:
    from tools.run_g1_dual_scale_root_position import classify_pair

    result = classify_pair(
        control=_records(
            (136, 144, 84, 90, 79),
            (136, 145, 84, 90, 79),
            (135, 160, 100, 100, 100),
            (130, 170, 110, 110, 110),
        ),
        treatment=_records(
            (137, 145, 85, 91, 80),
            (138, 146, 86, 92, 81),
            (136, 144, 84, 90, 79),
            (135, 200, 200, 200, 200),
        ),
        source_survival=[136, 144, 84, 90, 79],
    )

    assert result["outcome"] == "dual-scale-advances"
    assert result["selected_treatment_step"] == 1_966_080
    assert result["selected_treatment_survival"] == [138, 146, 86, 92, 81]


def test_selector_rejects_uncorroborated_source() -> None:
    from tools.run_g1_dual_scale_root_position import classify_pair

    with pytest.raises(ValueError, match="source E002"):
        classify_pair(
            control=_records(*([(136, 144, 84, 90, 79)] * 4)),
            treatment=_records(*([(137, 145, 85, 91, 80)] * 4)),
            source_survival=[136, 144, 84, 90, 78],
        )


def test_selector_labels_quadratic_treatment_without_changing_gates() -> None:
    from tools.run_g1_dual_scale_root_position import classify_pair

    result = classify_pair(
        control=_records(*([(136, 144, 84, 90, 79)] * 4)),
        treatment=_records(
            (137, 145, 85, 91, 80),
            (136, 144, 84, 90, 79),
            (135, 200, 200, 200, 200),
            (130, 210, 210, 210, 210),
        ),
        source_survival=[136, 144, 84, 90, 79],
        treatment_label="quadratic",
    )

    assert result["protocol"] == "g1-quadratic-root-position-pair-v1"
    assert result["outcome"] == "quadratic-advances"


def test_arm_training_validation_rejects_wrong_checkpoint_step(tmp_path) -> None:
    from src.algorithms.shac.frozen_controller_residual import (
        FrozenControllerResidualOptState,
        FrozenControllerResidualParams,
    )
    from src.core.data_structures import NormState, TrainState
    from tools.run_g1_dual_scale_root_position import (
        END_STEP,
        expected_checkpoint_steps,
        validate_arm_training_artifacts,
    )

    source = tmp_path / "checkpoint_step_1867776.pkl"
    source_hparams = {
        "actor_frozen_controller_residual": True,
        "tracking_root_velocity_weight": 1.0,
        "action_noise_std_start": [0.1] * 29,
        "total_steps": 2_162_688,
    }
    source.with_name("hparams.json").write_text(json.dumps(source_hparams))
    normalizer = NormState(
        mean=np.zeros(2), var=np.ones(2), count=np.array(1.0)
    )
    actor_params = FrozenControllerResidualParams(
        parent={"p": np.array([1.0])}, adapter={"a": np.array([0.0])}
    )
    actor_opt = FrozenControllerResidualOptState(
        parent_optimizer_state={"m": np.array([0.0])},
        adapter_optimizer_state={"m": np.array([0.0])},
    )

    def state(step: int) -> TrainState:
        return TrainState(
            key=np.array([0, 1], dtype=np.uint32),
            env_state=None,
            actor_params=actor_params,
            critic_params={},
            target_critic_params={},
            normalizer=normalizer,
            actor_opt=actor_opt,
            critic_opt={},
            step=step,
        )

    with source.open("wb") as stream:
        pickle.dump(state(1_867_776), stream)
    run = tmp_path / "run"
    run.mkdir()
    hparams = {
        **source_hparams,
        "tracking_anchor_position_kernel": "quadratic",
        "allow_resume_tracking_anchor_position_kernel_change": True,
        "allow_resume_tracking_root_velocity_change": False,
        "checkpoint_steps": list(expected_checkpoint_steps()),
        "reference_path_migration_artifact": None,
        "total_steps": END_STEP,
    }
    (run / "hparams.json").write_text(json.dumps(hparams))
    for step in expected_checkpoint_steps():
        with (run / f"checkpoint_step_{step}.pkl").open("wb") as stream:
            pickle.dump(state(step), stream)
    metrics = []
    for step in expected_checkpoint_steps():
        metrics.append(
            {
                "step": step,
                "actor_preview_valid": True,
                "actor_preview_gradient_norm": 0.1,
                "actor_preview_update_norm": 0.1,
                "actor_preview_frozen_parameter_drift_max_abs": 0.0,
                "actor_preview_frozen_moment_drift_max_abs": 0.0,
                "actor_preview_normalizer_drift_max_abs": 0.0,
                "actor_cagrad_valid": True,
                "actor_cagrad_bin_counts": [1] * 5,
                "actor_cagrad_bin_gradient_norms": [0.5] * 5,
                "actor_cagrad_bin_losses": [0.1] * 5,
                "actor_cagrad_weights": [0.2] * 5,
                "actor_cagrad_gram_matrix": [[0.1] * 5 for _ in range(5)],
                "actor_cagrad_cosine_matrix": [[0.1] * 5 for _ in range(5)],
                "actor_cagrad_combined_norm": 0.1,
                "actor_cagrad_dual_gap": 0.0,
                "actor_cagrad_objective": 0.1,
                "actor_cagrad_uniform_combined_cosine": 0.5,
                "actor_bootstrap_scale_current": 0.0,
                "action_noise_current": [0.1] * 29,
            }
        )
    (run / "checkpoint_phase_metrics.json").write_text(json.dumps(metrics))

    valid = validate_arm_training_artifacts(
        run, source_checkpoint=source, kernel="quadratic"
    )
    assert valid["valid"] is True

    with (run / f"checkpoint_step_{END_STEP}.pkl").open("wb") as stream:
        pickle.dump(state(END_STEP + 1), stream)

    with pytest.raises(ValueError, match="checkpoint structure"):
        validate_arm_training_artifacts(
            run, source_checkpoint=source, kernel="quadratic"
        )
