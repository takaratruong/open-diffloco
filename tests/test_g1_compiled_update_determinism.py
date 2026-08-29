import json
from pathlib import Path

FIRST_MJX_SUBSTEP_COMPONENTS = (
    "integrated_state",
    "acceleration_state",
    "constraint_force",
    "contact_state",
)

FIRST_MJX_SUBSTEP_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "qfrc_applied",
    "qfrc_passive",
    "qfrc_bias",
    "qfrc_actuator",
    "actuator_force",
    "xfrc_applied",
    "xfrc_accumulated",
    "qfrc_smooth",
    "qfrc_constraint",
    "efc_force",
    "contact",
)


def _report(
    *,
    first_mismatch=None,
    component_mismatches=(),
    field_mismatches=(),
    state_exact=True,
    metrics_exact=True,
):
    boundaries = {
        name: {
            "first": [1, 2, 3, 4],
            "second": [1, 2, 3, 4],
            "exact": name != first_mismatch,
        }
        for name in (
            "random_inputs",
            "first_actor_action",
            "first_mjx_substep",
            "first_mjx_control_step",
            "first_env_step",
            "rollout",
            "actor_cagrad",
            "learned_dynamics",
            "critic",
        )
    }
    components = {
        name: {
            "first": [1, 2, 3, 4],
            "second": [1, 2, 3, 4],
            "exact": name not in component_mismatches,
        }
        for name in FIRST_MJX_SUBSTEP_COMPONENTS
    }
    fields = {
        name: {
            "first": [1, 2, 3, 4],
            "second": [1, 2, 3, 4],
            "exact": name not in field_mismatches,
        }
        for name in FIRST_MJX_SUBSTEP_FIELDS
    }
    return {
        "protocol": "shac-compiled-update-determinism-v6",
        "valid": (
            first_mismatch is None
            and not component_mismatches
            and not field_mismatches
            and state_exact
            and metrics_exact
        ),
        "boundaries": boundaries,
        "first_mismatch_boundary": first_mismatch,
        "first_mjx_substep_components": components,
        "mismatching_first_mjx_substep_components": list(
            component_mismatches
        ),
        "first_mjx_substep_fields": fields,
        "mismatching_first_mjx_substep_fields": list(field_mismatches),
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

    first_step = classify_probe(_report(first_mismatch="first_env_step"))
    assert first_step["outcome"] == "compiled-update-diverges-first-env-step"

    mjx_substep = classify_probe(
        _report(
            first_mismatch="first_mjx_substep",
            component_mismatches=("constraint_force", "contact_state"),
            field_mismatches=("qacc", "efc_force"),
        )
    )
    assert mjx_substep["outcome"] == (
        "compiled-update-diverges-first-mjx-substep"
    )
    assert mjx_substep["mismatching_first_mjx_substep_components"] == [
        "constraint_force",
        "contact_state",
    ]
    assert mjx_substep["mismatching_first_mjx_substep_fields"] == [
        "qacc",
        "efc_force",
    ]

    mjx_control_step = classify_probe(
        _report(first_mismatch="first_mjx_control_step")
    )
    assert mjx_control_step["outcome"] == (
        "compiled-update-diverges-first-mjx-control-step"
    )


def test_probe_classification_preserves_an_unlocalized_failure():
    from tools.run_g1_compiled_update_determinism import classify_probe

    result = classify_probe(_report(state_exact=False))
    assert result["outcome"] == "compiled-update-diverges-unlocalized"
    assert result["first_mismatch_boundary"] is None


def test_probe_artifact_validation_accepts_sorted_json_boundary_keys(
    tmp_path, monkeypatch
):
    import tools.run_g1_compiled_update_determinism as probe

    report_path = tmp_path / "determinism_probe.json"
    run_directory = tmp_path / "training_run"
    run_directory.mkdir()
    report = _report(
        first_mismatch="first_env_step", state_exact=False, metrics_exact=False
    )
    report["boundaries"] = {
        name: report["boundaries"][name]
        for name in sorted(report["boundaries"])
    }
    report["first_mjx_substep_fields"] = {
        name: report["first_mjx_substep_fields"][name]
        for name in sorted(report["first_mjx_substep_fields"])
    }
    report.update(
        input_step=probe.SOURCE_STEP,
        input_state_sha256=probe.EXPECTED_SOURCE_STATE_SHA256,
        compiled_callable_reused=True,
        ordinary_training_loop_entered=False,
        first_state_sha256="1" * 64,
        second_state_sha256="2" * 64,
        first_metrics_sha256="3" * 64,
        second_metrics_sha256="4" * 64,
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (run_directory / "hparams.json").write_text(
        json.dumps(
            {
                "determinism_probe": True,
                "determinism_probe_output": str(report_path.resolve()),
                "actor_bootstrap_scale": 1.0,
                "jave_vg_weight": 0.0,
                "jave_collect_transitions": True,
                "actor_cagrad": True,
                "gradient_accumulation_steps": 2,
                "num_envs": 256,
                "unroll_length": 24,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        probe,
        "numeric_tree_sha256",
        lambda _state: probe.EXPECTED_SOURCE_STATE_SHA256,
    )

    validation = probe.validate_probe_artifacts(
        report_path=report_path,
        run_directory=run_directory,
        returned_state=object(),
    )

    assert validation["valid"] is True
    assert validation["probe"]["first_mismatch_boundary"] == "first_env_step"
