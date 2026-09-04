from __future__ import annotations

import numpy as np
import pytest


def test_probe_kwargs_add_only_excision_and_forward_jvp(tmp_path) -> None:
    from experiments.g1_ahac_bootstrap_gradient_discriminator.run import (
        build_probe_kwargs as build_source_kwargs,
    )
    from experiments.g1_ahac_reverse_forward_discriminator.run import (
        build_probe_kwargs,
    )

    reference = tmp_path / "reference.npz"
    checkpoint = tmp_path / "checkpoint.pkl"
    probe = tmp_path / "probe.json"
    source = build_source_kwargs(
        "g1-4x5",
        reference,
        0,
        checkpoint,
        arm="scale_zero",
        probe_output=probe,
    )
    treatment = build_probe_kwargs(
        "g1-4x5",
        reference,
        0,
        checkpoint,
        probe_output=probe,
    )

    assert treatment["actor_bootstrap_scale"] == 0.0
    assert treatment["actor_bootstrap_graph_mode"] == "excised"
    assert treatment["actor_forward_jvp_probe"] is True

    def equal(left, right) -> bool:
        if hasattr(left, "shape") or hasattr(right, "shape"):
            return np.array_equal(np.asarray(left), np.asarray(right))
        return bool(left == right)

    differing = {
        key
        for key in set(source) | set(treatment)
        if not equal(source.get(key), treatment.get(key))
    }
    assert differing == {
        "actor_bootstrap_graph_mode",
        "actor_forward_jvp_probe",
    }


@pytest.mark.parametrize(
    ("structural_mask", "forward_mask", "outcome"),
    (
        (
            [True] * 512,
            [True] * 512,
            "structural-excision-restores-all-reverse-adjoints",
        ),
        (
            [True] * 100 + [False] * 86 + [True] * 326,
            [True] * 512,
            "structural-excision-partially-restores-reverse-adjoints",
        ),
        (
            [True] * 186 + [False] * 186 + [True] * 140,
            [True] * 512,
            "structural-excision-redistributes-or-worsens-reverse-adjoints",
        ),
        (
            [False] * 186 + [True] * 326,
            [True] * 512,
            "identical-reverse-mask-dense-forward-direction-finite",
        ),
        (
            [False] * 186 + [True] * 326,
            [False] + [True] * 511,
            "dense-forward-direction-shares-reverse-invalidity",
        ),
        (
            [False] * 186 + [True] * 326,
            [True] * 200 + [False] + [True] * 311,
            "dense-forward-direction-diverges-outside-reverse-invalidity",
        ),
    ),
)
def test_reverse_forward_outcomes_are_exhaustive(
    structural_mask: list[bool],
    forward_mask: list[bool],
    outcome: str,
) -> None:
    from experiments.g1_ahac_reverse_forward_discriminator.run import (
        classify_reverse_forward_masks,
    )

    source_mask = [False] * 186 + [True] * 326
    result = classify_reverse_forward_masks(
        source_mask=source_mask,
        structural_mask=structural_mask,
        forward_finite_mask=forward_mask,
    )

    assert result["outcome"] == outcome
    assert result["source_finite_reverse_count"] == 326
    assert result["structural_finite_reverse_count"] == sum(structural_mask)
    assert result["forward_finite_count"] == sum(forward_mask)


def test_forward_jvp_report_is_reconstructed_from_complete_vectors() -> None:
    from experiments.g1_ahac_reverse_forward_discriminator.run import (
        validate_forward_jvp_report,
    )

    reverse_mask = [False] * 186 + [True] * 326
    losses = [index / 1000.0 for index in range(512)]
    tangents = [None, -0.25, *([0.0] * 510)]
    report = {
        "protocol": "shac-actor-forward-jvp-population-v1",
        "valid": True,
        "population_size": 512,
        "direction_fingerprint": [1, 2, 3, 4],
        "direction_norm": 1.0,
        "trainable_scalar_count": 91_677,
        "primal_matches_reverse_losses": True,
        "repeat_primal_exact": True,
        "repeat_tangent_exact": True,
        "finite_count": 511,
        "nonfinite_count": 1,
        "nonzero_count": 1,
        "finite_by_env": [False, *([True] * 511)],
        "nonzero_by_env": [False, True, *([False] * 510)],
        "reverse_gradient_finite_by_env": reverse_mask,
        "finite_on_reverse_invalid_count": 185,
        "reverse_invalid_count": 186,
        "primals_by_env": losses,
        "directional_derivatives_by_env": tangents,
    }

    result = validate_forward_jvp_report(
        report,
        expected_losses=losses,
        expected_reverse_mask=reverse_mask,
    )

    assert result["finite_mask"] == [False, *([True] * 511)]
    assert result["nonfinite_indices"] == [0]
    assert result["nonzero_on_reverse_invalid_count"] == 1


def test_parser_exposes_no_scientific_overrides() -> None:
    from experiments.g1_ahac_reverse_forward_discriminator.run import build_parser

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
        "--source-e017-run-json",
        "/tmp/run.json",
        "--source-e017-audit",
        "/tmp/audit.json",
        "--source-e017-probe",
        "/tmp/probe.json",
        "--source-e017-arm",
        "/tmp/arm.json",
        "--source-e017-comparison",
        "/tmp/comparison.json",
        "--source-e017-completion",
        "/tmp/completion.json",
        "--source-e017-receipt",
        "/tmp/receipt.json",
        "--code-commit",
        "0" * 40,
    ]
    build_parser().parse_args(required)
    for override in (
        ["--actor-bootstrap-graph-mode", "connected"],
        ["--forward-jvp-seed", "1"],
        ["--num-envs", "1024"],
        ["--seed", "1"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])
