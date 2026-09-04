from __future__ import annotations

from collections import Counter

import numpy as np
import pytest


PHYSICAL_BOUNDARIES = (
    "random_inputs",
    "first_actor_action",
    "first_mjx_substep",
    "first_mjx_control_step",
    "first_env_step",
    "rollout",
    "learned_dynamics",
    "critic",
)


def _population(scale: float, gradient_mask: list[bool]) -> dict[str, object]:
    phases = [index % 271 for index in range(512)]
    supports = [index % 4 for index in range(512)]
    terminals = [index % 2 for index in range(512)]
    losses = [scale + index / 1000.0 for index in range(512)]
    bins = [min(phase * 5 // 271, 4) for phase in phases]
    loss_counts = [Counter(bins)[index] for index in range(5)]
    gradient_counts = [
        sum(
            bin_index == target and finite
            for bin_index, finite in zip(bins, gradient_mask)
        )
        for target in range(5)
    ]
    bad_rows = [
        {
            "index": index,
            "phase": phases[index],
            "start_support_mode": supports[index],
            "terminal_mode": terminals[index],
            "loss": losses[index],
            "loss_finite": True,
        }
        for index, finite in enumerate(gradient_mask)
        if not finite
    ]
    return {
        "protocol": "shac-cagrad-population-v1",
        "report_valid": True,
        "actor_cagrad_valid": all(gradient_mask),
        "classification": (
            "cagrad-valid"
            if all(gradient_mask)
            else "nonfinite-per-environment-gradients"
        ),
        "input_step": 1_867_776,
        "computed_output_step": 1_880_064,
        "computed_candidate_state_persisted": False,
        "subgates": {
            "loss_bins_occupied": True,
            "losses_finite": True,
            "loss_valid": True,
            "gradient_bins_occupied": True,
            "gradient_norms_finite": True,
            "counts_match": all(gradient_mask),
            "solver_valid": True,
            "reduction_valid": True,
        },
        "phase_bins": {
            "loss_counts": loss_counts,
            "gradient_counts": gradient_counts,
            "missing_gradient_contributors": [
                loss - gradient for loss, gradient in zip(loss_counts, gradient_counts)
            ],
        },
        "population": {
            "size": 512,
            "finite_gradient_count": sum(gradient_mask),
            "finite_loss_count": 512,
            "gradient_finite_by_env": gradient_mask,
            "loss_finite_by_env": [True] * 512,
            "start_phases": phases,
            "start_support_modes": supports,
            "terminal_modes": terminals,
            "losses_by_env": losses,
            "nonfinite_gradient_environments": bad_rows,
            "nonfinite_loss_environments": [],
        },
        "ahac": {"actor_bootstrap_scale_current": scale},
    }


def _probe(scale: float, gradient_mask: list[bool]) -> dict[str, object]:
    boundaries = {
        name: {"first": [index], "second": [index], "exact": True}
        for index, name in enumerate(PHYSICAL_BOUNDARIES)
    }
    boundaries["actor_cagrad"] = {
        "first": [int(scale * 10)],
        "second": [int(scale * 10)],
        "exact": True,
    }
    return {
        "valid": True,
        "input_step": 1_867_776,
        "input_state_sha256": "same-state",
        "compiled_callable_reused": True,
        "ordinary_training_loop_entered": False,
        "full_state_exact": True,
        "metrics_exact": True,
        "boundaries": boundaries,
        "cagrad_population": _population(scale, gradient_mask),
    }


def test_population_validator_accepts_an_explicit_output_step() -> None:
    from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
        _validate_population_report,
    )

    report = _population(0.0, [True] * 512)
    report["computed_output_step"] = 1_868_288

    summary = _validate_population_report(
        report,
        expected_scale=0.0,
        expected_output_step=1_868_288,
    )

    assert summary["finite_count"] == 512


def test_probe_kwargs_change_only_bootstrap_scale_and_output(tmp_path) -> None:
    from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
        build_probe_kwargs,
    )

    reference = tmp_path / "reference.npz"
    checkpoint = tmp_path / "checkpoint.pkl"
    control = build_probe_kwargs(
        "g1-4x5",
        reference,
        0,
        checkpoint,
        arm="scale_one",
        probe_output=tmp_path / "one.json",
    )
    treatment = build_probe_kwargs(
        "g1-4x5",
        reference,
        0,
        checkpoint,
        arm="scale_zero",
        probe_output=tmp_path / "zero.json",
    )

    assert control["actor_bootstrap_scale"] == 1.0
    assert treatment["actor_bootstrap_scale"] == 0.0
    assert control["allow_ahac_actor_bootstrap_ablation"] is True
    assert treatment["allow_ahac_actor_bootstrap_ablation"] is True
    assert control["allow_resume_actor_bootstrap_scale_change"] is True
    assert treatment["allow_resume_actor_bootstrap_scale_change"] is True

    def equal(left, right) -> bool:
        if hasattr(left, "shape") or hasattr(right, "shape"):
            return np.array_equal(np.asarray(left), np.asarray(right))
        return bool(left == right)

    differing = {
        key
        for key in set(control) | set(treatment)
        if not equal(control.get(key), treatment.get(key))
    }
    assert differing == {"actor_bootstrap_scale", "determinism_probe_output"}


def test_zero_bootstrap_complete_recovery_is_classified() -> None:
    from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
        compare_probe_reports,
    )

    control_mask = [False] * 186 + [True] * 326
    control = _probe(1.0, control_mask)
    treatment = _probe(0.0, [True] * 512)

    result = compare_probe_reports(
        control,
        treatment,
        source_failure=control["cagrad_population"],
    )

    assert result["outcome"] == "zero-bootstrap-restores-all-adjoints"
    assert result["scale_one_finite_gradient_count"] == 326
    assert result["scale_zero_finite_gradient_count"] == 512
    assert result["recovered_gradient_count"] == 186
    assert result["introduced_nonfinite_gradient_count"] == 0
    assert all(result["physical_boundaries_match"].values())
    assert result["optimizer_update_retained"] is False
    assert result["retained_policy"] is None


@pytest.mark.parametrize(
    ("treatment_mask", "outcome"),
    (
        (
            [True] * 100 + [False] * 86 + [True] * 326,
            "zero-bootstrap-partially-restores-adjoints",
        ),
        (
            [False] * 186 + [True] * 326,
            "zero-bootstrap-preserves-identical-invalid-mask",
        ),
        (
            [True] * 186 + [False] * 186 + [True] * 140,
            "zero-bootstrap-redistributes-or-worsens-invalid-adjoints",
        ),
    ),
)
def test_incomplete_zero_bootstrap_outcomes_are_exhaustive(
    treatment_mask: list[bool], outcome: str
) -> None:
    from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
        compare_probe_reports,
    )

    control_mask = [False] * 186 + [True] * 326
    control = _probe(1.0, control_mask)
    treatment = _probe(0.0, treatment_mask)

    result = compare_probe_reports(
        control,
        treatment,
        source_failure=control["cagrad_population"],
    )

    assert result["outcome"] == outcome


def test_parser_exposes_no_scientific_overrides() -> None:
    from experiments.g1_ahac_bootstrap_gradient_discriminator.run import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--reference-path",
        "/tmp/reference.npz",
        "--resume-from",
        "/tmp/checkpoint.pkl",
        "--output-root",
        "/tmp/output",
        "--resume-audit",
        "/tmp/resume.json",
        "--ahac-fidelity-audit",
        "/tmp/fidelity.json",
        "--critic-calibration-audit",
        "/tmp/calibration.json",
        "--source-run-json",
        "/tmp/run.json",
        "--source-audit",
        "/tmp/audit.json",
        "--source-failure",
        "/tmp/failure.json",
        "--code-commit",
        "0" * 40,
    ]
    build_parser().parse_args(required)
    for override in (
        ["--actor-bootstrap-scale", "0.5"],
        ["--num-envs", "1024"],
        ["--unroll-length", "12"],
        ["--seed", "1"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])
