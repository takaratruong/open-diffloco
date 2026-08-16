from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest


CONTROL = {
    16: (42, 36, 48, 49, 24),
    32: (45, 50, 53, 49, 24),
}
E004_EARLY = {
    16: (43, 38, 48, 49, 24),
    32: (51, 66, 55, 49, 24),
}


def test_budget_contracts_are_exact_and_fail_closed() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        EARLY_BUDGET,
        FULL_BUDGET,
        resolve_budget,
    )

    assert EARLY_BUDGET.name == "early"
    assert EARLY_BUDGET.total_updates == 32
    assert EARLY_BUDGET.total_steps == 393_216
    assert EARLY_BUDGET.checkpoint_updates == (16, 32)
    assert EARLY_BUDGET.checkpoint_steps == (196_608, 393_216)
    assert FULL_BUDGET.name == "full"
    assert FULL_BUDGET.total_updates == 128
    assert FULL_BUDGET.total_steps == 1_572_864
    assert FULL_BUDGET.checkpoint_updates == (16, 32, 48, 64, 80, 96, 112, 128)
    assert FULL_BUDGET.checkpoint_steps == (
        196_608,
        393_216,
        589_824,
        786_432,
        983_040,
        1_179_648,
        1_376_256,
        1_572_864,
    )
    assert resolve_budget("early") is EARLY_BUDGET
    assert resolve_budget("full") is FULL_BUDGET
    with pytest.raises(ValueError, match="unknown execution budget"):
        resolve_budget("long")


def test_public_budget_apis_canonicalize_copies_and_reject_custom_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_motion_anchor_position_h24_walk as runner

    full_copy = runner.BudgetContract(
        name="full",
        total_updates=128,
        checkpoint_updates=(16, 32, 48, 64, 80, 96, 112, 128),
    )
    custom = runner.BudgetContract(
        name="full",
        total_updates=64,
        checkpoint_updates=(16, 32, 48, 64),
    )
    monkeypatch.setattr(
        runner,
        "validate_e023_preflight",
        lambda **_: {"protocol": "parent", "valid": True},
    )

    report = runner.validate_preflight(
        repository=Path("/repo"),
        reference_path=Path("/tmp/walk.npz"),
        code_commit="a" * 40,
        budget=full_copy,
    )

    assert report["protocol"] == (
        "g1-motion-anchor-position-h24-walk-preflight-full-v1"
    )
    assert report["budget"] == "full"
    with pytest.raises(ValueError, match="unregistered execution budget"):
        runner.expected_checkpoint_steps(custom)
    with pytest.raises(ValueError, match="unregistered execution budget"):
        runner.build_motion_anchor_position_kwargs(
            "g1-4x5", Path("/tmp/walk.npz"), 0, budget=custom
        )
    with pytest.raises(ValueError, match="unregistered execution budget"):
        runner.validate_preflight(
            repository=Path("/repo"),
            reference_path=Path("/tmp/walk.npz"),
            code_commit="a" * 40,
            budget=custom,
        )
    with pytest.raises(ValueError, match="unregistered execution budget"):
        runner.validate_budget_training_artifacts(
            Path("/tmp/run"), expected_kwargs={}, budget=custom
        )


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


def test_full_builder_changes_only_root_position_observation_from_e023() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        FULL_BUDGET,
        build_motion_anchor_position_kwargs,
        expected_checkpoint_steps,
    )
    from tools.run_g1_rmr_noise_h24_walk import build_rmr_noise_h24_kwargs

    reference = Path("/tmp/walk.npz")
    parent = build_rmr_noise_h24_kwargs("g1-4x5", reference, 0)
    treatment = build_motion_anchor_position_kwargs(
        "g1-4x5", reference, 0, budget=FULL_BUDGET
    )
    changed = {
        key
        for key in set(parent) | set(treatment)
        if not np.array_equal(parent.get(key), treatment.get(key))
    }

    assert changed == {
        "actor_observe_motion_anchor_position",
        "expected_actor_obs_dim",
    }
    assert treatment["total_steps"] == 1_572_864
    assert treatment["action_noise_schedule_steps"] == 1_572_864
    assert expected_checkpoint_steps(FULL_BUDGET) == FULL_BUDGET.checkpoint_steps


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


@pytest.mark.parametrize(
    ("update_128", "expected"),
    [
        ((117, 99, 67, 49, 24), "root-position-full-advances"),
        ((114, 99, 67, 49, 24), "root-position-full-parity"),
        ((120, 95, 67, 49, 24), "root-position-full-mixed"),
        ((110, 95, 64, 45, 24), "root-position-full-underperforms"),
    ],
)
def test_full_budget_classifier_covers_registered_outcomes(
    update_128, expected
) -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_full_budget_root_position,
    )

    treatment = {
        **E004_EARLY,
        64: (71, 60, 51, 49, 24),
        128: update_128,
    }

    assert classify_full_budget_root_position(treatment) == expected


def test_full_budget_advancement_requires_preserving_phase_100() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_full_budget_root_position,
    )

    treatment = {
        **E004_EARLY,
        64: (71, 60, 51, 49, 24),
        128: (117, 99, 67, 49, 23),
    }

    assert classify_full_budget_root_position(treatment) == (
        "root-position-full-parity"
    )


@pytest.mark.parametrize(
    "treatment",
    [
        {**E004_EARLY, 64: (71, 60, 51, 49, 24)},
        {
            **E004_EARLY,
            48: (50, 50, 50, 49, 24),
            64: (71, 60, 51, 49, 24),
            128: (116, 99, 67, 49, 24),
        },
        {
            16: (44, 38, 48, 49, 24),
            32: E004_EARLY[32],
            64: (71, 60, 51, 49, 24),
            128: (116, 99, 67, 49, 24),
        },
        {
            16: E004_EARLY[16],
            32: (51, 65, 55, 49, 24),
            64: (71, 60, 51, 49, 24),
            128: (116, 99, 67, 49, 24),
        },
        {**E004_EARLY, 64: (71, 60, 51, 49), 128: (116, 99, 67, 49, 24)},
        {**E004_EARLY, 64: (71, 60, 51, 49, 24), 128: (116.0, 99, 67, 49, 24)},
        {**E004_EARLY, 64: (71, 60, 51, 49, 24), 128: (125, 99, 67, 49, 24)},
        {**E004_EARLY, 64: (71, 60, 51, 49, 24), 128: (116, 99, np.nan, 49, 24)},
    ],
)
def test_full_budget_classifier_rejects_invalid_or_noncorroborating_evidence(
    treatment,
) -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_full_budget_root_position,
    )

    with pytest.raises(ValueError):
        classify_full_budget_root_position(treatment)


@pytest.mark.parametrize(
    ("update_128", "expected"),
    [
        ((117, 99, 67, 49, 24), "root-position-full-advances"),
        ((114, 99, 67, 49, 24), "root-position-full-parity"),
        ((120, 95, 67, 49, 24), "root-position-full-mixed"),
        ((110, 95, 64, 45, 24), "root-position-full-underperforms"),
    ],
)
def test_matched_full_classifier_uses_only_full_budget_evidence(
    update_128, expected
) -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_matched_full_budget_root_position,
    )

    assert classify_matched_full_budget_root_position(
        {64: (71, 60, 51, 49, 24), 128: update_128}
    ) == expected


@pytest.mark.parametrize(
    "treatment",
    [
        {64: (71, 60, 51, 49, 24)},
        {
            16: (42, 36, 49, 47, 24),
            64: (71, 60, 51, 49, 24),
            128: (116, 99, 67, 49, 24),
        },
        {64: (71, 60, 51, 49), 128: (116, 99, 67, 49, 24)},
        {64: (71, 60, 51, 49, 24), 128: (116.0, 99, 67, 49, 24)},
    ],
)
def test_matched_full_classifier_fails_closed_on_malformed_evidence(
    treatment,
) -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import (
        classify_matched_full_budget_root_position,
    )

    with pytest.raises(ValueError):
        classify_matched_full_budget_root_position(treatment)


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


def test_full_preflight_records_all_archives_without_importing_jax(
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
        budget=runner.FULL_BUDGET,
    )

    assert report["valid"] is True
    assert report["budget"] == "full"
    assert report["total_updates"] == 128
    assert report["total_steps"] == 1_572_864
    assert report["checkpoint_updates"] == [16, 32, 48, 64, 80, 96, 112, 128]
    assert report["checkpoint_steps"] == [
        196_608,
        393_216,
        589_824,
        786_432,
        983_040,
        1_179_648,
        1_376_256,
        1_572_864,
    ]


def test_full_preflight_path_does_not_import_jax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "from pathlib import Path",
                    "import tools.run_g1_motion_anchor_position_h24_walk as runner",
                    "runner.validate_e023_preflight = lambda **_: "
                    "{'protocol': 'parent', 'valid': True}",
                    "report = runner.validate_preflight(",
                    "    repository=Path('/repo'),",
                    "    reference_path=Path('/tmp/walk.npz'),",
                    "    code_commit='a' * 40,",
                    "    budget=runner.FULL_BUDGET,",
                    ")",
                    "assert report['budget'] == 'full'",
                    "assert 'jax' not in sys.modules",
                )
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_full_artifact_validation_forwards_exact_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_g1_motion_anchor_position_h24_walk as runner

    captured = {}

    def fake_validator(run_directory, **kwargs):
        captured["run_directory"] = run_directory
        captured.update(kwargs)
        return {"valid": True}

    monkeypatch.setattr(runner, "validate_e023_training_artifacts", fake_validator)
    kwargs = {"total_steps": runner.FULL_BUDGET.total_steps}
    report = runner.validate_budget_training_artifacts(
        Path("/tmp/run"), expected_kwargs=kwargs, budget=runner.FULL_BUDGET
    )

    assert report == {"valid": True}
    assert captured["expected_kwargs"] is kwargs
    assert captured["expected_steps"] == runner.FULL_BUDGET.checkpoint_steps
    assert captured["total_steps"] == 1_572_864
    assert captured["protocol"] == (
        "g1-motion-anchor-position-h24-walk-training-full-v1"
    )


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
    monkeypatch.setattr(
        pure_preflight, "EXPECTED_REFERENCE_SHA256", expected_hashes[reference]
    )
    monkeypatch.setattr(pure_preflight, "EXPECTED_MODEL_PATH", model)
    monkeypatch.setattr(pure_preflight, "EXPECTED_MODEL_SHA256", expected_hashes[model])
    monkeypatch.setattr(pure_preflight, "EXPECTED_CONTROLLER_PATH", controller)
    monkeypatch.setattr(
        pure_preflight, "EXPECTED_CONTROLLER_SHA256", expected_hashes[controller]
    )
    monkeypatch.setattr(
        pure_preflight,
        "_git_output",
        lambda _repository, *arguments: (
            "a" * 40 if arguments == ("rev-parse", "HEAD") else ""
        ),
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
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=repository, check=True)
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


def test_parser_defaults_to_early_and_accepts_full_budget() -> None:
    from tools.run_g1_motion_anchor_position_h24_walk import build_parser

    parser = build_parser()
    common = [
        "--solver-profile",
        "g1-4x5",
        "--reference-path",
        "/tmp/walk.npz",
        "--code-commit",
        "a" * 40,
    ]

    assert parser.parse_args(common).budget == "early"
    assert parser.parse_args([*common, "--budget", "full"]).budget == "full"
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--budget", "long"])


def test_main_threads_full_budget_to_training_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_g1_motion_anchor_position_h24_walk as runner

    reference = tmp_path / "walk.npz"
    reference.write_bytes(b"reference")
    output_root = tmp_path / "output"
    args = argparse.Namespace(
        solver_profile="g1-4x5",
        reference_path=reference,
        seed=0,
        output_root=output_root,
        code_commit="a" * 40,
        budget="full",
    )
    captured = {}

    class FakeParser:
        def parse_args(self):
            return args

    def fake_preflight(**kwargs):
        captured["preflight_budget"] = kwargs["budget"]
        return {"valid": True}

    def fake_builder(profile_name, reference_path, seed, *, budget):
        captured["builder_budget"] = budget
        return {"total_steps": budget.total_steps}

    def fake_train(**kwargs):
        captured["train_kwargs"] = kwargs
        return None, "run"

    def fake_validator(run_directory, *, expected_kwargs, budget):
        captured["validation_run_directory"] = run_directory
        captured["validation_kwargs"] = expected_kwargs
        captured["validation_budget"] = budget
        return {"valid": True}

    monkeypatch.setattr(runner, "build_parser", lambda: FakeParser())
    monkeypatch.setattr(runner, "validate_preflight", fake_preflight)
    monkeypatch.setattr(runner, "build_motion_anchor_position_kwargs", fake_builder)
    monkeypatch.setattr(runner, "validate_budget_training_artifacts", fake_validator)
    monkeypatch.setitem(
        sys.modules,
        "src.algorithms.shac.algorithm",
        types.SimpleNamespace(train=fake_train),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.envs.g1_tracking.solver_profiles",
        types.SimpleNamespace(
            get_solver_profile=lambda name: name,
            solver_context=lambda _profile: nullcontext(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.run_g1_tracking_shac",
        types.SimpleNamespace(configure_jax=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.run_g1_zero_assistance_consolidation",
        types.SimpleNamespace(_write_json_atomically=lambda *_args: None),
    )

    runner.main()

    assert captured["preflight_budget"] is runner.FULL_BUDGET
    assert captured["builder_budget"] is runner.FULL_BUDGET
    assert captured["train_kwargs"] == {"total_steps": 1_572_864}
    assert captured["validation_kwargs"] == captured["train_kwargs"]
    assert captured["validation_budget"] is runner.FULL_BUDGET
    assert captured["validation_budget"].checkpoint_steps == (
        196_608,
        393_216,
        589_824,
        786_432,
        983_040,
        1_179_648,
        1_376_256,
        1_572_864,
    )
    assert captured["validation_run_directory"] == output_root / "run"
