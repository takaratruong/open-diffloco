from pathlib import Path


def test_treatment_conditions_are_the_approved_minimal_pair():
    from tools.evaluate_g1_learned_wrench_vertical_discriminator import (
        TREATMENT_CONDITIONS,
    )

    assert TREATMENT_CONDITIONS == (
        ("vertical-force-only-a", "vertical-force-only"),
        ("vertical-force-only-b", "vertical-force-only"),
        (
            "horizontal-force-and-torque-a",
            "horizontal-force-and-torque",
        ),
        (
            "horizontal-force-and-torque-b",
            "horizontal-force-and-torque",
        ),
    )


def test_classification_requires_within_treatment_agreement():
    from tools.evaluate_g1_learned_wrench_vertical_discriminator import (
        classify_vertical_discriminator,
    )

    assert (
        classify_vertical_discriminator(
            vertical=[True, True], no_vertical=[False, False]
        )["outcome"]
        == "vertical-alone-sufficient-no-vertical-insufficient"
    )
    assert (
        classify_vertical_discriminator(
            vertical=[False, False], no_vertical=[False, False]
        )["outcome"]
        == "vertical-required-but-not-alone-sufficient"
    )
    assert (
        classify_vertical_discriminator(
            vertical=[True, True], no_vertical=[True, True]
        )["outcome"]
        == "vertical-not-required"
    )
    mixed = classify_vertical_discriminator(
        vertical=[True, False], no_vertical=[False, False]
    )
    assert mixed == {
        "outcome": "replicate-divergence",
        "treatments_unanimous": False,
    }


def test_default_xla_child_environment_removes_ambient_flags():
    from tools.evaluate_g1_learned_wrench_vertical_discriminator import (
        default_xla_environment,
    )

    environment = default_xla_environment(
        {
            "PATH": "/bin",
            "CUDA_VISIBLE_DEVICES": "2",
            "XLA_FLAGS": "ambient-is-forbidden",
        }
    )

    assert "XLA_FLAGS" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_parser_binds_reused_control_matrix_and_exact_inputs(tmp_path):
    from tools.evaluate_g1_learned_wrench_vertical_discriminator import build_parser

    args = build_parser().parse_args(
        [
            "--control-matrix",
            str(tmp_path / "replay_matrix.json"),
            "--control-matrix-sha256",
            "d" * 64,
            "--checkpoint",
            str(tmp_path / "checkpoint.pkl"),
            "--checkpoint-sha256",
            "a" * 64,
            "--reference-path",
            str(tmp_path / "reference.npz"),
            "--reference-sha256",
            "b" * 64,
            "--evaluator-sha256",
            "c" * 64,
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert args.control_matrix == Path(tmp_path / "replay_matrix.json")
    assert args.output_root == Path(tmp_path / "output")


def test_control_matrix_accepts_behavioral_not_bit_exact_current_defaults():
    from tools.evaluate_g1_learned_wrench_vertical_discriminator import (
        validate_control_matrix,
    )

    payload = {
        "protocol": "g1-learned-wrench-replay-matrix-v1",
        "checkpoint_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "phase": 0,
        "solver_profile": "g1-4x5",
        "rows": [
            {
                "condition": "current-default-a",
                "source": "current",
                "execution": "default",
                "steps": 271,
                "completed": True,
                "evaluation_content_sha256": "1" * 64,
            },
            {
                "condition": "current-default-b",
                "source": "current",
                "execution": "default",
                "steps": 271,
                "completed": True,
                "evaluation_content_sha256": "2" * 64,
            },
        ],
    }

    rows = validate_control_matrix(
        payload,
        checkpoint_sha256="a" * 64,
        reference_sha256="b" * 64,
        phase=0,
        solver_profile="g1-4x5",
        expected_steps=271,
    )

    assert [row["condition"] for row in rows] == [
        "current-default-a",
        "current-default-b",
    ]
