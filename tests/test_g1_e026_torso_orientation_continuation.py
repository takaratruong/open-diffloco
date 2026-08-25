from pathlib import Path


def test_registered_torso_continuation_steps_and_kwargs():
    from tools.run_g1_e026_torso_orientation_continuation import (
        END_STEP,
        START_STEP,
        build_torso_orientation_kwargs,
        expected_checkpoint_steps,
    )

    checkpoint = Path("/tmp/checkpoint_step_1769472.pkl")
    bank = Path("/tmp/carried.npz")
    kwargs = build_torso_orientation_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, bank
    )

    assert START_STEP == 1_769_472
    assert END_STEP == 2_162_688
    assert expected_checkpoint_steps() == (
        1_867_776,
        1_966_080,
        2_064_384,
        2_162_688,
    )
    assert kwargs["resume_from"] == str(checkpoint.resolve())
    assert kwargs["total_steps"] == END_STEP
    assert kwargs["checkpoint_interval"] == 98_304
    assert kwargs["tracking_torso_orientation_weight"] == 1.0
    assert kwargs["allow_resume_tracking_torso_orientation_change"] is True
    assert kwargs["carried_reset_bank_path"] == str(bank.resolve())
    assert kwargs["carried_reset_probability"] == 0.25
    assert kwargs["actor_residual_preview_adapter"] is True
    assert kwargs["actor_policy_anchor_weight"] == 1.0
    assert kwargs["reference_reset_noise_scale"] == 0.0
    assert kwargs["domain_randomization"] is False


def _record(
    update, survival, mean_pitch, max_pitch, *, position=None, orientation=None
):
    return {
        "update": update,
        "survival": list(survival),
        "body_position_error_ratio": position or [1.0] * 5,
        "body_orientation_error_ratio": orientation or [1.0] * 5,
        "phase0_tail_mean_abs_pitch_degrees": mean_pitch,
        "phase0_tail_max_abs_pitch_degrees": max_pitch,
    }


def test_selector_requires_survival_tracking_and_tail_stability():
    from tools.run_g1_e026_torso_orientation_continuation import (
        select_torso_orientation_checkpoint,
    )

    records = [
        _record(8, (124, 99, 74, 49, 24), 9.4, 17.0),
        _record(16, (124, 99, 73, 49, 24), 8.0, 16.0),
        _record(
            24,
            (124, 99, 74, 49, 24),
            7.0,
            16.0,
            position=[1.0, 1.06, 1.0, 1.0, 1.0],
        ),
        _record(32, (124, 99, 74, 49, 24), 8.5, 16.5),
    ]

    selected = select_torso_orientation_checkpoint(records)

    assert selected["outcome"] == "torso-objective-stabilizes-short-walk"
    assert selected["eligible_updates"] == [8, 32]
    assert selected["selected_update"] == 32
    assert selected["selected_survival"] == [124, 99, 74, 49, 24]


def test_selector_distinguishes_redistribution_from_insufficient():
    from tools.run_g1_e026_torso_orientation_continuation import (
        select_torso_orientation_checkpoint,
    )

    redistributed = select_torso_orientation_checkpoint(
        [_record(8, (123, 99, 74, 49, 24), 8.0, 16.0)]
    )
    insufficient = select_torso_orientation_checkpoint(
        [_record(8, (124, 99, 74, 49, 24), 10.0, 17.0)]
    )

    assert redistributed["outcome"] == "torso-objective-redistributes"
    assert insufficient["outcome"] == "torso-objective-insufficient"
