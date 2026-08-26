import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.algorithms.shac import algorithm
from src.algorithms.shac.counterfactual_wrench_distillation import (
    load_counterfactual_feasibility,
    resolve_counterfactual_wrench_resume_setting,
)


def _write_feasibility(tmp_path: Path) -> tuple[Path, str]:
    npz_path = tmp_path / "counterfactual_wrench_feasibility.npz"
    np.savez(npz_path, target_changes=np.ones((5, 12)))
    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    payload = {
        "protocol": "g1-counterfactual-wrench-feasibility-v1",
        "valid": True,
        "outcome": "leg-counterfactual-feasible",
        "phases": [0, 25, 50, 75, 100],
        "phase_counts": {str(phase): 1 for phase in (0, 25, 50, 75, 100)},
        "row_count": 5,
        "threshold": 0.5,
        "median_normalized_residual": 0.4,
        "target_rms": [float(value) for value in range(1, 13)],
        "npz_file": npz_path.name,
        "npz_sha256": npz_sha,
        "teacher_checkpoint_sha256": "a" * 64,
        "teacher_tree_sha256": "b" * 64,
        "e026_tree_sha256": "c" * 64,
        "wrench_tree_sha256": "d" * 64,
    }
    path = tmp_path / "counterfactual_wrench_feasibility.json"
    path.write_text(json.dumps(payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_feasibility_loader_binds_artifact_and_target_rms(tmp_path):
    path, sha = _write_feasibility(tmp_path)
    report = load_counterfactual_feasibility(path, expected_sha256=sha)
    np.testing.assert_array_equal(report.target_rms, np.arange(1.0, 13.0))
    assert report.teacher_checkpoint_sha256 == "a" * 64
    assert report.e026_tree_sha256 == "c" * 64


@pytest.mark.parametrize("mutation", ["sha", "outcome", "rms", "npz"])
def test_feasibility_loader_fails_closed(tmp_path, mutation):
    path, sha = _write_feasibility(tmp_path)
    payload = json.loads(path.read_text())
    if mutation == "sha":
        sha = "0" * 64
    elif mutation == "outcome":
        payload["outcome"] = "leg-counterfactual-not-feasible"
        path.write_text(json.dumps(payload))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "rms":
        payload["target_rms"][3] = 0.0
        path.write_text(json.dumps(payload))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        path.with_suffix(".npz").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        load_counterfactual_feasibility(path, expected_sha256=sha)


def test_counterfactual_resume_requires_explicit_upgrade_and_exact_sources():
    saved = {"actor_counterfactual_wrench_distillation": False}
    with pytest.raises(ValueError, match="authority"):
        resolve_counterfactual_wrench_resume_setting(
            saved,
            requested=True,
            teacher_sha256="a" * 64,
            feasibility_sha256="b" * 64,
            allow_start=False,
            is_resume=True,
        )
    enabled, upgrade = resolve_counterfactual_wrench_resume_setting(
        saved,
        requested=True,
        teacher_sha256="a" * 64,
        feasibility_sha256="b" * 64,
        allow_start=True,
        is_resume=True,
    )
    assert enabled and upgrade
    with pytest.raises(ValueError, match="metadata"):
        resolve_counterfactual_wrench_resume_setting(
            None,
            requested=True,
            teacher_sha256="a" * 64,
            feasibility_sha256="b" * 64,
            allow_start=True,
            is_resume=True,
        )


def test_train_exposes_disabled_counterfactual_contract():
    signature = inspect.signature(algorithm.train)
    assert signature.parameters[
        "actor_counterfactual_wrench_distillation"
    ].default is False
    assert signature.parameters[
        "actor_counterfactual_wrench_teacher_path"
    ].default is None
    assert signature.parameters[
        "actor_counterfactual_wrench_teacher_sha256"
    ].default is None
    assert signature.parameters[
        "actor_counterfactual_wrench_feasibility_path"
    ].default is None
    assert signature.parameters[
        "actor_counterfactual_wrench_feasibility_sha256"
    ].default is None


def test_train_wires_zero_wrench_teacher_and_leg_only_student():
    source = inspect.getsource(algorithm.train)
    assert "resolve_leg_action_indices(" in source
    assert "env.actor_joint_names" in source
    assert "action_dim=(12 if actor_counterfactual_wrench_distillation" in source
    assert "residual_action_indices=counterfactual_leg_indices" in source
    assert "counterfactual_teacher_params" in source
    assert "teacher_noisy_action = noisy_action - _residual_action" in source
    assert "teacher_state = state.replace" in source
    assert "jp.zeros_like(state.data.xfrc_applied)" in source
    assert "teacher_next_state = env.step(" in source
    assert "teacher_state, teacher_noisy_action" in source
    assert "jax.lax.stop_gradient(" in source
    assert "teacher_next_state" in source
    assert "counterfactual_transition_loss(" in source
    assert "candidate_unreplayed_state.done == 0" in source
    assert "counterfactual_teacher_next_state.done == 0" in source
    assert "counterfactual_done_match" in source
    assert '"counterfactual_done_mismatch"' in source
    assert '"counterfactual_integrity"' in source
    assert "counterfactual_invalid_count == 0" in source
    assert "actor_objective + counterfactual_objective" in source
    assert "counterfactual_residual_temporal_weight" in source


def test_counterfactual_target_is_not_an_environment_reward():
    source = inspect.getsource(algorithm.train)
    reward_block = source[source.index('"reward": jp.where'):source.index('"done": jp.where')]
    assert "counterfactual" not in reward_block


def test_checkpoint_counterfactual_telemetry_fails_closed():
    metrics = {
        f"actor_counterfactual_{name}": 0.1
        for name in (
            "loss",
            "base_linear_loss",
            "base_angular_loss",
            "centroidal_linear_loss",
            "centroidal_angular_loss",
            "cosine",
            "student_rms",
            "teacher_rms",
            "normalized_error_rms",
            "residual_rms",
            "residual_max_abs",
            "residual_bound_fraction",
            "teacher_wrench_rms",
        )
    }
    metrics.update(
        actor_counterfactual_nonleg_max_abs=0.0,
        actor_counterfactual_student_wrench_max_abs=0.0,
        actor_counterfactual_valid_count=12,
        actor_counterfactual_invalid_count=0,
        actor_counterfactual_done_mismatch_count=1,
        actor_counterfactual_valid=True,
    )
    report = algorithm.build_counterfactual_wrench_telemetry(metrics)
    assert report["actor_counterfactual_valid"] is True
    assert report["actor_counterfactual_done_mismatch_count"] == 1
    assert report["actor_counterfactual_valid_count"] == 12
    metrics["actor_counterfactual_nonleg_max_abs"] = 1e-12
    with pytest.raises(ValueError):
        algorithm.build_counterfactual_wrench_telemetry(metrics)
    metrics["actor_counterfactual_nonleg_max_abs"] = 0.0
    metrics["actor_counterfactual_invalid_count"] = 1
    with pytest.raises(ValueError):
        algorithm.build_counterfactual_wrench_telemetry(metrics)
