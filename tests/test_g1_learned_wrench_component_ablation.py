import numpy as np


def test_conditions_cover_complete_component_factorial_with_repeat_control():
    from tools.evaluate_g1_learned_wrench_component_ablation import (
        COMPONENT_CONDITIONS,
    )

    assert COMPONENT_CONDITIONS == (
        ("full-control-a", "full"),
        ("full-control-b", "full"),
        ("force-only", "force-only"),
        ("vertical-force-and-torque", "vertical-force-and-torque"),
        ("horizontal-force-and-torque", "horizontal-force-and-torque"),
        ("vertical-force-only", "vertical-force-only"),
        ("horizontal-force-only", "horizontal-force-only"),
        ("torque-only", "torque-only"),
        ("zero", "zero"),
    )


def test_classification_identifies_vertical_force_as_necessary_and_sufficient():
    from tools.evaluate_g1_learned_wrench_component_ablation import (
        classify_component_ablation,
    )

    completed = {
        "full-control-a": True,
        "full-control-b": True,
        "force-only": True,
        "vertical-force-and-torque": True,
        "horizontal-force-and-torque": False,
        "vertical-force-only": True,
        "horizontal-force-only": False,
        "torque-only": False,
        "zero": False,
    }

    assert (
        classify_component_ablation(completed, controls_exact=True)
        == "vertical-force-alone-sufficient-no-vertical-insufficient"
    )
    assert (
        classify_component_ablation(completed, controls_exact=False)
        == "invalid-control-repeat"
    )


def test_evaluator_command_preserves_route_and_changes_only_component_mask(tmp_path):
    from tools.evaluate_g1_learned_wrench_component_ablation import (
        build_evaluator_command,
    )

    command = build_evaluator_command(
        python=tmp_path / "python",
        evaluator=tmp_path / "evaluate.py",
        checkpoint=tmp_path / "checkpoint.pkl",
        reference=tmp_path / "reference.npz",
        output_dir=tmp_path / "output",
        components="vertical-force-only",
        phase=0,
        solver_profile="g1-4x5",
    )

    assert command[:2] == [str(tmp_path / "python"), str(tmp_path / "evaluate.py")]
    assert command[command.index("--checkpoint") + 1] == str(
        tmp_path / "checkpoint.pkl"
    )
    assert command[command.index("--reference-path") + 1] == str(
        tmp_path / "reference.npz"
    )
    assert command[command.index("--learned-wrench-components") + 1] == (
        "vertical-force-only"
    )
    assert command[command.index("--phase") + 1] == "0"
    assert command[command.index("--solver-profile") + 1] == "g1-4x5"
    assert command[command.index("--actor-reference-lookahead-steps") + 1 :] == [
        "4",
        "8",
        "12",
    ]


def test_npz_content_digest_ignores_container_metadata(tmp_path):
    from tools.evaluate_g1_learned_wrench_component_ablation import (
        npz_content_sha256,
    )

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez_compressed(first, b=np.asarray(["x"]), a=np.arange(6).reshape(2, 3))
    np.savez_compressed(second, a=np.arange(6).reshape(2, 3), b=np.asarray(["x"]))

    assert npz_content_sha256(first) == npz_content_sha256(second)
