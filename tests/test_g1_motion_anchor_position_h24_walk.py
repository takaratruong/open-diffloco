from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest


CONTROL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}


def test_builder_changes_only_root_position_and_execution_metadata() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        TOTAL_STEPS,
        build_motion_anchor_position_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_rmr_noise_h24_walk import (
        TOTAL_STEPS as E023_TOTAL_STEPS,
        build_rmr_noise_h24_kwargs,
    )

    reference = Path("/tmp/walk.npz")
    parent = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_motion_anchor_position_kwargs("g1-4x5", reference, 0)
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }

    assert changed == {
        "actor_observe_motion_anchor_position",
        "expected_actor_obs_dim",
        "total_steps",
    }
    assert treatment["actor_observe_motion_anchor_position"] is True
    assert treatment["actor_history_len"] == 10
    assert treatment["expected_actor_obs_dim"] == 3_310
    assert treatment["total_steps"] == TOTAL_STEPS == 393_216
    assert treatment["checkpoint_interval"] == 196_608
    assert treatment["action_noise_schedule_steps"] == E023_TOTAL_STEPS == 1_572_864
    assert expected_checkpoint_steps() == (196_608, 393_216)
    assert treatment["actor_reference_lookahead_steps"] == (4, 8, 12)
    assert treatment["actor_reference_preview_mode"] == "delta"


def test_import_does_not_initialize_jax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tools.run_g1_motion_anchor_position_h24_walk; "
                "assert 'jax' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_actor_input_contract_checks_root_position_history_width() -> None:
    from src.core.actor_input_contract import validate_actor_input_contract

    report = validate_actor_input_contract(
        expected_input_dim=3_310,
        environment_input_dim=3_310,
        first_layer_input_dim=3_310,
    )
    assert report == {
        "expected_actor_obs_dim": 3_310,
        "environment_actor_obs_dim": 3_310,
        "actor_first_layer_input_dim": 3_310,
        "valid": True,
    }
    with pytest.raises(ValueError):
        validate_actor_input_contract(
            expected_input_dim=3_310,
            environment_input_dim=3_309,
            first_layer_input_dim=3_310,
        )
    with pytest.raises(ValueError):
        validate_actor_input_contract(
            expected_input_dim=3_310,
            environment_input_dim=3_310,
            first_layer_input_dim=3_309,
        )


@pytest.mark.parametrize(
    ("treatment", "expected"),
    [
        (
            {16: (43, 36, 48, 49, 24), 32: CONTROL[32]},
            "root-position-early-advances",
        ),
        (CONTROL, "root-position-early-parity"),
        (
            {16: (46, 32, 48, 49, 24), 32: (45, 46, 53, 49, 24)},
            "root-position-early-mixed",
        ),
        (
            {16: (38, 34, 45, 46, 24), 32: (41, 46, 49, 45, 24)},
            "root-position-early-underperforms",
        ),
    ],
)
def test_classifier_covers_registered_outcomes(treatment, expected) -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_root_position_ablation,
    )

    assert classify_root_position_ablation(treatment) == expected


def test_advancement_requires_preserving_phase_100() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_root_position_ablation,
    )

    treatment = {16: (43, 36, 48, 49, 23), 32: (45, 50, 53, 49, 23)}

    assert classify_root_position_ablation(treatment) == "root-position-early-parity"


@pytest.mark.parametrize(
    "treatment",
    [
        {16: CONTROL[16]},
        {16: CONTROL[16], 32: (45, 50, 53, 49)},
        {16: CONTROL[16], 32: (45, 50, np.nan, 49, 24)},
        {16: CONTROL[16], 32: (45.0, 50, 53, 49, 24)},
        {16: CONTROL[16], 32: (125, 50, 53, 49, 24)},
    ],
)
def test_classifier_fails_closed_on_invalid_evidence(treatment) -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_root_position_ablation,
    )

    with pytest.raises(ValueError):
        classify_root_position_ablation(treatment)


def test_selection_uses_first_four_phase_key_and_earliest_tie() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        select_root_position_checkpoint,
    )

    assert select_root_position_checkpoint(CONTROL) == 32
    tied = {16: CONTROL[16], 32: CONTROL[16]}
    assert select_root_position_checkpoint(tied) == 16


def test_preflight_records_only_root_position_semantic_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_motion_anchor_position_h24_walk as runner

    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"protocol": "parent", "valid": True},
    )
    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="a" * 40,
    )

    assert report["valid"] is True
    assert report["scientific_delta"] == ["actor_observe_motion_anchor_position"]
    assert report["actor_observe_motion_anchor_position"] is True
    assert report["actor_history_len"] == 10
    assert report["actor_input_dim"] == 3_310
    assert report["total_updates"] == 32
    assert report["checkpoint_updates"] == [16, 32]
    assert report["action_noise_schedule_steps"] == 1_572_864


def test_preflight_fails_closed_when_parent_provenance_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_motion_anchor_position_h24_walk as runner

    def reject_parent(**_):
        raise ValueError("runtime code worktree must be clean")

    monkeypatch.setattr(runner, "validate_e023_preflight", reject_parent)
    with pytest.raises(ValueError, match="worktree must be clean"):
        runner.validate_preflight(
            repository=Path("/repo"),
            reference_path=Path("/tmp/walk.npz"),
            code_commit="a" * 40,
        )


def test_real_preflight_reports_root_position_frame_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_g1_motion_anchor_position_h24_walk as runner
    import tools.run_g1_one_frame_rmr_noise_h24_walk as pure_preflight

    reference = tmp_path / "walk.npz"
    model = tmp_path / "g1.xml"
    controller = tmp_path / "controller.npz"
    for path in (reference, model, controller):
        path.write_bytes(path.name.encode("utf-8"))
    expected_hashes = {
        reference: runner_hash(reference),
        model: runner_hash(model),
        controller: runner_hash(controller),
    }
    monkeypatch.setattr(pure_preflight, "EXPECTED_REFERENCE_SHA256", expected_hashes[reference])
    monkeypatch.setattr(pure_preflight, "EXPECTED_MODEL_PATH", model)
    monkeypatch.setattr(pure_preflight, "EXPECTED_MODEL_SHA256", expected_hashes[model])
    monkeypatch.setattr(pure_preflight, "EXPECTED_CONTROLLER_PATH", controller)
    monkeypatch.setattr(
        pure_preflight, "EXPECTED_CONTROLLER_SHA256", expected_hashes[controller]
    )
    monkeypatch.setattr(
        pure_preflight,
        "_git_output",
        lambda _repository, *arguments: "a" * 40
        if arguments == ("rev-parse", "HEAD")
        else "",
    )

    report = runner.validate_preflight(
        repository=tmp_path,
        reference_path=reference,
        code_commit="a" * 40,
    )

    assert report["actor_frame_obs_dim"] == 331
    assert report["actor_history_len"] == 10
    assert report["actor_input_dim"] == 3_310


def runner_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_preflight_rejects_invalid_code_without_importing_jax() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "from pathlib import Path",
                    "from tools.run_g1_motion_anchor_position_h24_walk "
                    "import validate_preflight",
                    "try:",
                    "    validate_preflight("
                    f"repository=Path({str(repository)!r}), "
                    "reference_path=Path('/missing.npz'), "
                    "code_commit='not-a-commit')",
                    "except ValueError as error:",
                    "    assert 'runtime code commit' in str(error)",
                    "else:",
                    "    raise AssertionError('invalid code commit was accepted')",
                    "assert 'jax' not in sys.modules",
                )
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_preflight_rejects_invalid_asset_without_importing_jax() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = Path(directory) / "repository"
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=repository, check=True
        )
        (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "initial"], cwd=repository, check=True
        )
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    (
                        "import sys",
                        "from pathlib import Path",
                        "from tools.run_g1_motion_anchor_position_h24_walk "
                        "import validate_preflight",
                        "try:",
                        "    validate_preflight("
                        f"repository=Path({str(repository)!r}), "
                        "reference_path=Path('/missing.npz'), "
                        f"code_commit={code_commit!r})",
                        "except ValueError as error:",
                        "    assert 'runtime reference SHA-256' in str(error)",
                        "else:",
                        "    raise AssertionError('invalid reference was accepted')",
                        "assert 'jax' not in sys.modules",
                    )
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert completed.returncode == 0, completed.stderr


def test_parser_requires_code_commit() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--solver-profile", "g1-4x5", "--reference-path", "/tmp/walk.npz"]
        )
