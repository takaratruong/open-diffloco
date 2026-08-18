from pathlib import Path


def test_builder_changes_only_velocity_kernel_from_e023():
    from tools.run_g1_pseudo_huber_velocity_h24_walk import (
        build_pseudo_huber_velocity_h24_kwargs,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = Path("/tmp/walk.npz")
    control = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_pseudo_huber_velocity_h24_kwargs(
        "g1-4x5", reference, 0
    )

    assert treatment["tracking_velocity_kernel"] == "pseudo_huber"
    assert treatment["allow_resume_tracking_velocity_kernel_change"] is False
    assert {
        key: value
        for key, value in treatment.items()
        if key
        not in {
            "tracking_velocity_kernel",
            "allow_resume_tracking_velocity_kernel_change",
        }
    } == control


def test_preflight_records_exact_single_scientific_delta(monkeypatch):
    import tools.run_g1_pseudo_huber_velocity_h24_walk as runner

    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_kwargs: {
            "protocol": "parent",
            "valid": True,
            "code_commit": "a" * 40,
        },
    )

    report = runner.validate_preflight(
        repository=Path("/tmp/repository"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="a" * 40,
    )

    assert report["protocol"] == "g1-pseudo-huber-velocity-h24-walk-preflight-v1"
    assert report["scientific_delta"] == ["tracking_velocity_kernel"]
    assert report["tracking_velocity_kernel"] == "pseudo_huber"


def test_parser_requires_pinned_runtime_inputs():
    from tools.run_g1_pseudo_huber_velocity_h24_walk import build_parser

    args = build_parser().parse_args(
        [
            "--solver-profile",
            "g1-4x5",
            "--reference-path",
            "/tmp/walk.npz",
            "--code-commit",
            "b" * 40,
        ]
    )

    assert args.solver_profile == "g1-4x5"
    assert args.reference_path == Path("/tmp/walk.npz")
    assert args.seed == 0
    assert args.code_commit == "b" * 40
