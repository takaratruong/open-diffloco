from pathlib import Path


def _arm_payload(kernel: str, offset: int = 0) -> dict[str, object]:
    from tools.run_g1_dual_scale_root_position import expected_checkpoint_steps

    rows = (
        (136 + offset, 144 + offset, 84 + offset, 90 + offset, 79 + offset),
        (136, 144, 84, 90, 79),
        (135, 160, 100, 100, 100),
        (130, 170, 110, 110, 110),
    )
    return {
        "kernel": kernel,
        "source_survival": [136, 144, 84, 90, 79],
        "candidates": {
            str(step): {
                "checkpoint_sha256": f"{index + 1:064x}",
                "survival": list(row),
            }
            for index, (step, row) in enumerate(
                zip(expected_checkpoint_steps(), rows, strict=True)
            )
        },
    }


def test_arm_commands_differ_only_by_kernel_output_and_device() -> None:
    from tools.run_g1_quadratic_root_position_pair import build_arm_command

    common = {
        "solver_profile": "g1-4x5",
        "reference": Path("/tmp/reference.npz"),
        "checkpoint": Path("/tmp/checkpoint.pkl"),
        "code_commit": "a" * 40,
        "seed": 0,
    }
    control = build_arm_command(
        **common, output=Path("/tmp/control"), kernel="exponential"
    )
    treatment = build_arm_command(
        **common, output=Path("/tmp/treatment"), kernel="quadratic"
    )

    assert control[-4:] == ["--kernel", "exponential", "--seed", "0"]
    assert treatment[-4:] == ["--kernel", "quadratic", "--seed", "0"]
    assert control[control.index("--output-root") + 1] == "/tmp/control"
    assert treatment[treatment.index("--output-root") + 1] == "/tmp/treatment"


def test_pair_aggregation_uses_quadratic_componentwise_gate() -> None:
    from tools.run_g1_quadratic_root_position_pair import classify_arm_payloads

    result = classify_arm_payloads(
        _arm_payload("exponential"),
        _arm_payload("quadratic", offset=1),
    )

    assert result["outcome"] == "quadratic-advances"
    assert result["policy_retained"] is True
    assert result["selected_treatment_survival"] == [137, 145, 85, 91, 80]


def test_quadratic_render_uses_explicit_quadratic_kernel() -> None:
    from tools.run_g1_quadratic_root_position_pair import (
        build_quadratic_render_command,
    )

    command = build_quadratic_render_command(
        checkpoint=Path("/tmp/checkpoint.pkl"),
        reference=Path("/tmp/reference.npz"),
        output=Path("/tmp/preview"),
    )

    index = command.index("--tracking-anchor-position-kernel")
    assert command[index + 1] == "quadratic"
    assert "--training-distribution-rollout" not in command


def test_pair_workers_start_in_isolated_process_groups(
    tmp_path, monkeypatch
) -> None:
    from tools import run_g1_quadratic_root_position_pair as pair

    calls = []

    class FinishedProcess:
        pid = 12345

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(*args, **kwargs):
        calls.append(kwargs)
        return FinishedProcess()

    monkeypatch.setattr(pair.subprocess, "Popen", fake_popen)
    pair._run_arms(
        {"control": ["control"], "treatment": ["treatment"]},
        {"control": "2", "treatment": "3"},
        tmp_path,
    )

    assert len(calls) == 2
    assert all(call["start_new_session"] is True for call in calls)
