import json
from pathlib import Path


def _payload(checkpoint: Path, survival, completed):
    phases = [0, 24, 48, 72, 96]
    return {
        "source_policy_path": str(checkpoint.resolve()),
        "source_policy_sha256": f"sha-{checkpoint.stem}",
        "reference_transitions": 120,
        "source": {
            "summary": {
                "phases": phases,
                "survival": survival,
                "completed_suffix": completed,
            }
        },
    }


def test_select_checkpoint_requires_all_suffixes_and_ranks_normalized_survival(
    tmp_path,
):
    from tools.screen_g1_rmr_parent_checkpoints import select_checkpoint

    weak = tmp_path / "model_500.pt"
    competent = tmp_path / "model_1000.pt"
    strong_incomplete = tmp_path / "model_1500.pt"
    payloads = [
        _payload(weak, [70, 60, 50, 40, 20], [False] * 5),
        _payload(
            competent,
            [120, 96, 72, 48, 24],
            [True, True, True, True, True],
        ),
        _payload(
            strong_incomplete,
            [119, 95, 71, 47, 24],
            [False, False, False, False, True],
        ),
    ]

    selected = select_checkpoint(payloads, phases=(0, 24, 48, 72, 96))

    assert selected["eligible"] is True
    assert selected["selected_checkpoint_path"] == str(competent.resolve())
    assert selected["selected_checkpoint_sha256"] == "sha-model_1000"
    assert selected["ranking"][0]["completed_suffix_count"] == 5
    assert selected["ranking"][0]["minimum_survival_fraction"] == 1.0


def test_select_checkpoint_reports_best_diagnostic_but_no_eligible_parent(
    tmp_path,
):
    from tools.screen_g1_rmr_parent_checkpoints import select_checkpoint

    first = tmp_path / "model_500.pt"
    second = tmp_path / "model_1000.pt"
    selected = select_checkpoint(
        [
            _payload(first, [60, 96, 72, 48, 24], [False, True, True, True, True]),
            _payload(second, [119, 95, 71, 47, 23], [False] * 5),
        ],
        phases=(0, 24, 48, 72, 96),
    )

    assert selected["eligible"] is False
    assert selected["selected_checkpoint_path"] is None
    assert selected["best_diagnostic_checkpoint_path"] == str(first.resolve())


def test_screen_assigns_one_visible_gpu_per_child_and_writes_manifest(
    monkeypatch, tmp_path,
):
    from tools import screen_g1_rmr_parent_checkpoints as screen

    checkpoints = [tmp_path / "model_500.pt", tmp_path / "model_1000.pt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"reference")
    calls = []

    def fake_run(command, *, env, stdout, stderr, text, check):
        del stdout, stderr, text, check
        calls.append((command, env["CUDA_VISIBLE_DEVICES"]))
        output = Path(command[command.index("--output") + 1])
        checkpoint = Path(command[command.index("--source-policy-checkpoint") + 1])
        output.write_text(
            json.dumps(
                _payload(
                    checkpoint,
                    [120, 96, 72, 48, 24],
                    [True, True, True, True, True],
                )
            )
        )

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(screen.subprocess, "run", fake_run)
    summary_path = tmp_path / "screen.json"
    result = screen.run_screen(
        checkpoints=checkpoints,
        reference_path=reference,
        output_root=tmp_path / "children",
        summary_output=summary_path,
        phases=(0, 24, 48, 72, 96),
        seed=0,
        solver_profile="g1-4x5",
        gpu_ids=("2", "3"),
    )

    assert {gpu for _, gpu in calls} == {"2", "3"}
    assert result["selection"]["eligible"] is True
    assert summary_path.is_file()
    assert len(result["evaluations"]) == 2
