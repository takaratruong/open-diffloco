from pathlib import Path


def _report(*, first_mismatch=None, state_exact=True, metrics_exact=True):
    boundaries = {
        name: {
            "first": [1, 2, 3, 4],
            "second": [1, 2, 3, 4],
            "exact": name != first_mismatch,
        }
        for name in ("rollout", "actor_cagrad", "learned_dynamics", "critic")
    }
    return {
        "protocol": "shac-compiled-update-determinism-v1",
        "valid": first_mismatch is None and state_exact and metrics_exact,
        "boundaries": boundaries,
        "first_mismatch_boundary": first_mismatch,
        "full_state_exact": state_exact,
        "metrics_exact": metrics_exact,
    }


def test_probe_kwargs_reuse_the_existing_jave_branch_route():
    from tools.run_g1_compiled_update_determinism import build_probe_kwargs

    output = Path("/tmp/probe/determinism_probe.json")
    kwargs = build_probe_kwargs(
        "g1-4x5",
        Path("/tmp/reference.npz"),
        0,
        Path("/tmp/checkpoint_step_1880064.pkl"),
        output,
    )

    assert kwargs["resume_from"] == "/tmp/checkpoint_step_1880064.pkl"
    assert kwargs["jave_vg_weight"] == 0.0
    assert kwargs["jave_collect_transitions"] is True
    assert kwargs["actor_bootstrap_scale"] == 1.0
    assert kwargs["determinism_probe_output"] == str(output)


def test_probe_classification_localizes_the_first_boundary():
    from tools.run_g1_compiled_update_determinism import classify_probe

    assert classify_probe(_report())["outcome"] == "compiled-update-exact"
    mismatch = classify_probe(_report(first_mismatch="learned_dynamics"))
    assert mismatch["outcome"] == (
        "compiled-update-diverges-learned-dynamics"
    )
    assert mismatch["first_mismatch_boundary"] == "learned_dynamics"


def test_probe_classification_preserves_an_unlocalized_failure():
    from tools.run_g1_compiled_update_determinism import classify_probe

    result = classify_probe(_report(state_exact=False))
    assert result["outcome"] == "compiled-update-diverges-unlocalized"
    assert result["first_mismatch_boundary"] is None
