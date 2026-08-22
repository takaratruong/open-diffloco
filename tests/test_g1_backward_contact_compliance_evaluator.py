import json
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_g1_backward_contact_compliance import (
    HARD_TIME_CONSTANT,
    PROTOCOL,
    SOFT_TIME_CONSTANT,
    classify_backward_contact_compliance,
    validate_contact_model_delta,
    validate_forward_identity,
    validate_completion,
)


def _row():
    return {
        "valid": True,
        "forward_identical": True,
        "pathwise_vjp_cosine_min": 0.9999,
        "pathwise_vjp_norm_ratio_min": 0.9999,
        "pathwise_vjp_norm_ratio_max": 1.0001,
        "finite_phase_count_min": 20,
        "hard_mean_solver_cosine": 0.70,
        "compliant_mean_solver_cosine": 0.77,
        "hard_mean_tape_cosine": 0.60,
        "compliant_mean_tape_cosine": 0.66,
        "hard_phase_solver_cosine": [0.7] * 5,
        "compliant_phase_solver_cosine": [0.75] * 5,
        "hard_phase_tape_cosine": [0.6] * 5,
        "compliant_phase_tape_cosine": [0.65] * 5,
        "retained_hard_cosine": 0.8,
        "retained_hard_norm_ratio": 1.2,
    }


def test_classifier_selects_robust_only_when_both_means_advance():
    assert classify_backward_contact_compliance(_row()) == (
        "backward-compliance-robust"
    )
    row = _row()
    row["compliant_mean_tape_cosine"] = 0.64
    assert classify_backward_contact_compliance(row) == (
        "backward-compliance-neutral"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retained_hard_cosine", 0.49),
        ("retained_hard_norm_ratio", 4.01),
        ("compliant_phase_solver_cosine", [0.64] + [0.75] * 4),
        ("compliant_phase_tape_cosine", [0.54] + [0.65] * 4),
    ],
)
def test_classifier_rejects_destructive_directions(field, value):
    row = _row()
    row[field] = value
    assert classify_backward_contact_compliance(row) == (
        "backward-compliance-destructive"
    )


def test_classifier_fails_closed_on_incomplete_or_nonfinite_evidence():
    row = _row()
    row.pop("forward_identical")
    assert classify_backward_contact_compliance(row) == "invalid-execution"
    row = _row()
    row["hard_mean_solver_cosine"] = np.nan
    assert classify_backward_contact_compliance(row) == "invalid-execution"


def test_contact_model_delta_accepts_only_fixed_time_constant_change():
    hard = np.asarray([[HARD_TIME_CONSTANT, 1.0]] * 3)
    soft = np.asarray([[SOFT_TIME_CONSTANT, 1.0]] * 3)

    assert validate_contact_model_delta(hard, soft) == {
        "hard_time_constant": HARD_TIME_CONSTANT,
        "soft_time_constant": SOFT_TIME_CONSTANT,
        "dampratio": 1.0,
        "geom_count": 3,
    }
    with pytest.raises(ValueError, match="hard time constant"):
        validate_contact_model_delta(hard + [0.01, 0.0], soft)
    with pytest.raises(ValueError, match="dampratio"):
        validate_contact_model_delta(hard, soft + [0.0, 0.1])


def test_forward_identity_is_bit_exact_and_complete():
    hard = {
        "reward": np.asarray([1.0]),
        "done": np.asarray([False]),
        "terminal": np.asarray([False]),
        "qpos": np.asarray([[1.0, 2.0]]),
        "qvel": np.asarray([[3.0]]),
        "normalized_obs": np.asarray([[4.0]]),
        "mean": np.asarray([[5.0]]),
        "sampled_action": np.asarray([[6.0]]),
    }
    assert validate_forward_identity(hard, dict(hard)) is True
    changed = dict(hard)
    changed["qpos"] = np.asarray([[1.0, np.nextafter(2.0, 3.0)]])
    with pytest.raises(ValueError, match="qpos"):
        validate_forward_identity(hard, changed)
    incomplete = dict(hard)
    incomplete.pop("terminal")
    with pytest.raises(ValueError, match="terminal"):
        validate_forward_identity(hard, incomplete)


def test_completion_reopens_every_hash_bound_artifact(tmp_path: Path):
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    completion = tmp_path / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "valid": True,
                "protocol": PROTOCOL,
                "outcome": "backward-compliance-neutral",
                "artifacts": {artifact.name: digest},
            }
        ),
        encoding="utf-8",
    )
    assert validate_completion(completion)["outcome"] == (
        "backward-compliance-neutral"
    )
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_completion(completion)
