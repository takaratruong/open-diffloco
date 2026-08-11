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
    )

    assert payload["summary"]["survival"] == [499, 399, 299, 199, 99]
    assert payload["summary"]["completed_suffix"] == [True] * 5
    assert payload["results"] == results
    assert payload["actor_history_len"] == 10
    assert payload["actor_reference_lookahead_steps"] == [4, 8, 12]
