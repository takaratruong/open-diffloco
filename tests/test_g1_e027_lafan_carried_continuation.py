from pathlib import Path

import numpy as np


def test_e027_continuation_changes_only_endpoint_and_parent(tmp_path: Path):
    from tools.run_g1_e023_lafan_anchored_carried_recovery import (
        build_lafan_recovery_kwargs,
    )
    from tools.run_g1_e027_lafan_carried_continuation import (
        CONTINUATION_END_STEP,
        build_lafan_continuation_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "lafan.npz"
    bank = tmp_path / "bank.npz"
    old = tmp_path / "checkpoint_step_1572864.pkl"
    new = tmp_path / "checkpoint_step_2359296.pkl"
    parent = build_lafan_recovery_kwargs("g1-4x5", reference, 0, old, bank)
    treatment = build_lafan_continuation_kwargs(
        "g1-4x5", reference, 0, new, bank
    )
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }
    assert changed == {"resume_from", "total_steps"}
    assert CONTINUATION_END_STEP == 3_145_728
    assert expected_checkpoint_steps() == (
        2_457_600,
        2_555_904,
        2_654_208,
        2_752_512,
        2_850_816,
        2_949_120,
        3_047_424,
        3_145_728,
    )


def test_exact_continuation_rows_require_frozen_parent():
    from tools.run_g1_e027_lafan_carried_continuation import (
        expected_checkpoint_steps,
        validate_exact_continuation_rows,
    )

    rows = [
        {
            "step": step,
            "actor_preview_frozen_parameter_drift_max_abs": 0.0,
            "actor_preview_frozen_moment_drift_max_abs": 0.0,
            "actor_preview_normalizer_drift_max_abs": 0.0,
            "actor_preview_gradient_norm": 0.1,
            "actor_preview_update_norm": 0.01,
            "actor_preview_valid": True,
        }
        for step in expected_checkpoint_steps()
    ]
    assert validate_exact_continuation_rows(rows)["valid"] is True
    rows[-1]["actor_preview_frozen_parameter_drift_max_abs"] = 1e-8
    try:
        validate_exact_continuation_rows(rows)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("frozen drift was accepted")
