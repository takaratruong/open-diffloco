from pathlib import Path

import numpy as np
import pytest

from experiments.g1_e002_first_terminal_audit.run import (
    END_STEP,
    START_STEP,
    TRANSITIONS_PER_UPDATE,
    build_probe_kwargs,
    compare_probe_reports,
)


def test_probe_kwargs_change_only_actor_return_boundary_between_arms():
    common = dict(
        profile_name="g1-4x5",
        reference_path=Path("/tmp/reference.npz"),
        seed=0,
        resume_from=Path("/tmp/checkpoint_step_1867776.pkl"),
    )
    control = build_probe_kwargs(
        **common,
        semantics="multi_episode",
        probe_output=Path("/tmp/control.json"),
    )
    treatment = build_probe_kwargs(
        **common,
        semantics="first_terminal",
        probe_output=Path("/tmp/treatment.json"),
    )

    def differs(left, right):
        if hasattr(left, "shape") or hasattr(right, "shape"):
            return not np.array_equal(np.asarray(left), np.asarray(right))
        return left != right

    changed = {
        key
        for key in set(control) | set(treatment)
        if differs(control.get(key), treatment.get(key))
    }

    assert START_STEP == 1_867_776
    assert END_STEP == START_STEP + TRANSITIONS_PER_UPDATE
    assert changed == {
        "actor_return_semantics",
        "allow_resume_actor_return_semantics_change",
        "determinism_probe_output",
    }
    assert control["actor_return_semantics"] == "multi_episode"
    assert treatment["actor_return_semantics"] == "first_terminal"
    assert control["allow_resume_actor_return_semantics_change"] is False
    assert treatment["allow_resume_actor_return_semantics_change"] is True
    assert control["actor_bootstrap_scale"] == treatment["actor_bootstrap_scale"] == 0.0
    assert control["actor_cagrad"] is treatment["actor_cagrad"] is True


def _summary(*, included: int, grad: float) -> dict[str, object]:
    return {
        "actor_loss": -0.1,
        "actor_return_mean": 2.0 if included == TRANSITIONS_PER_UPDATE else 1.5,
        "actor_return_done_env_count": 3.0,
        "actor_return_done_event_count": 4.0,
        "actor_return_included_transition_count": float(included),
        "actor_return_post_first_done_transition_count": 5.0,
        "actor_return_post_first_done_env_count": 3.0,
        "actor_return_post_first_done_reward_sum": 256.0,
        "actor_return_post_first_done_reward_mean": 0.1,
        "actor_grad": grad,
        "actor_update_norm": grad * 2.0,
        "actor_grad_population_mean_norm": grad,
        "actor_grad_population_rms_norm": grad * 3.0,
        "actor_grad_population_variance_trace": grad * 4.0,
        "actor_grad_population_cancellation_ratio": 0.2,
        "actor_grad_population_noise_scale": 24.0,
        "actor_grad_population_esnr": 2.0,
    }


def _report(*, included: int, grad: float, actor_fingerprint: int = 8):
    physical = {
        name: {"first": [index, 7], "second": [index, 7], "exact": True}
        for index, name in enumerate(
            (
                "random_inputs",
                "first_actor_action",
                "first_mjx_substep",
                "first_mjx_control_step",
                "first_env_step",
                "rollout",
                "learned_dynamics",
                "critic",
            )
        )
    }
    physical["actor_cagrad"] = {
        "first": [actor_fingerprint, 9],
        "second": [actor_fingerprint, 9],
        "exact": True,
    }
    return {
        "valid": True,
        "input_step": START_STEP,
        "input_state_sha256": "a" * 64,
        "ordinary_training_loop_entered": False,
        "compiled_callable_reused": True,
        "full_state_exact": True,
        "metrics_exact": True,
        "boundaries": physical,
        "actor_update_summary": _summary(included=included, grad=grad),
    }


def test_comparison_requires_matched_physics_and_detects_gradient_change():
    control = _report(included=TRANSITIONS_PER_UPDATE, grad=0.1)
    treatment = _report(
        included=TRANSITIONS_PER_UPDATE - 5,
        grad=0.2,
        actor_fingerprint=10,
    )

    result = compare_probe_reports(control, treatment)

    assert result["outcome"] == "first-terminal-changes-gradient"
    assert result["actor_update_changed"] is True
    assert result["post_first_done_transition_count"] == 5
    assert result["post_reset_return_contribution_mean"] == 0.5
    assert result["gradient_ratios"]["actor_grad"] == 2.0
    assert result["optimizer_update_retained"] is False
    assert result["retained_policy"] is None


def test_comparison_rejects_a_physical_rollout_mismatch():
    control = _report(included=TRANSITIONS_PER_UPDATE, grad=0.1)
    treatment = _report(
        included=TRANSITIONS_PER_UPDATE - 5,
        grad=0.2,
        actor_fingerprint=10,
    )
    treatment["boundaries"]["rollout"]["first"] = [99, 7]

    with pytest.raises(ValueError, match="one physical rollout"):
        compare_probe_reports(control, treatment)


def test_comparison_rejects_prefix_accounting_that_does_not_close():
    control = _report(included=TRANSITIONS_PER_UPDATE, grad=0.1)
    treatment = _report(
        included=TRANSITIONS_PER_UPDATE - 4,
        grad=0.2,
        actor_fingerprint=10,
    )

    with pytest.raises(ValueError, match="prefix accounting"):
        compare_probe_reports(control, treatment)
