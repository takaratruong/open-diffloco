from __future__ import annotations

import numpy as np
import pytest


def _teacher_arrays() -> dict[str, np.ndarray]:
    success = np.zeros(24, dtype=bool)
    success[list(range(12)) + [13]] = True
    return {
        "actor_obs": np.zeros((24, 32, 3280), dtype=np.float32),
        "parent_action": np.zeros((24, 32, 29), dtype=np.float32),
        "correction": np.full((24, 32, 29), 0.25, dtype=np.float32),
        "effective_action": np.full((24, 32, 29), 0.25, dtype=np.float32),
        "success_mask": success,
    }


def test_select_teacher_rows_preserves_noncontiguous_success_set():
    from tools.run_g1_e034_recovery_expert import select_teacher_rows

    arrays = _teacher_arrays()
    arrays["actor_obs"][:, :, 0] = np.arange(24)[:, None]

    rows = select_teacher_rows(arrays)

    assert rows["actor_obs"].shape == (416, 3280)
    np.testing.assert_array_equal(
        np.unique(rows["actor_obs"][:, 0]), list(range(12)) + [13]
    )
    assert rows["correction"].shape == (416, 29)


def test_select_teacher_rows_rejects_wrong_success_mask():
    from tools.run_g1_e034_recovery_expert import select_teacher_rows

    arrays = _teacher_arrays()
    arrays["success_mask"][12] = True

    with pytest.raises(ValueError, match="success mask"):
        select_teacher_rows(arrays)


def test_imitation_loss_includes_correction_and_effective_action():
    from tools.run_g1_e034_recovery_expert import imitation_loss

    parent = np.asarray([[0.9, -0.9]], dtype=np.float32)
    target_correction = np.asarray([[0.5, -0.5]], dtype=np.float32)
    target_effective = np.asarray([[1.0, -1.0]], dtype=np.float32)

    exact = imitation_loss(
        target_correction, parent, target_correction, target_effective
    )
    wrong = imitation_loss(
        np.zeros_like(target_correction),
        parent,
        target_correction,
        target_effective,
    )

    assert float(exact) == pytest.approx(0.0)
    assert float(wrong) > 0.0


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ([32] * 13 + [30] * 11, "state-conditioned-recovery-reproduced"),
        ([32] * 4 + list(range(24, 4, -1)), "state-conditioned-recovery-partial"),
        (list(range(28, 4, -1)), "state-conditioned-recovery-insufficient"),
    ],
)
def test_classify_recovery_expert(candidate, expected):
    from tools.run_g1_e034_recovery_expert import classify_recovery_expert

    baseline = list(range(28, 4, -1))
    success_mask = np.zeros(24, dtype=bool)
    success_mask[list(range(12)) + [13]] = True
    if expected == "state-conditioned-recovery-reproduced":
        candidate = np.maximum(candidate, baseline).tolist()

    assert (
        classify_recovery_expert(
            baseline_survival=baseline,
            candidate_survival=candidate,
            teacher_success_mask=success_mask,
            execution_valid=True,
        )
        == expected
    )


def test_classify_recovery_expert_rejects_regression_from_reproduced():
    from tools.run_g1_e034_recovery_expert import classify_recovery_expert

    baseline = list(range(28, 4, -1))
    candidate = [32] * 13 + [30] * 11
    candidate[-1] = 4
    success_mask = np.zeros(24, dtype=bool)
    success_mask[list(range(12)) + [13]] = True

    assert classify_recovery_expert(
        baseline_survival=baseline,
        candidate_survival=candidate,
        teacher_success_mask=success_mask,
        execution_valid=True,
    ) != "state-conditioned-recovery-reproduced"


def test_zero_seed_is_enforced():
    from tools.run_g1_e034_recovery_expert import _zero_seed

    assert _zero_seed("0") == 0
    with pytest.raises(Exception, match="exactly zero"):
        _zero_seed("1")
