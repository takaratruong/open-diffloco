import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest


def test_root_velocity_continuation_has_exact_update_grid() -> None:
    from tools.run_g1_root_velocity_continuation import (
        END_STEP,
        ROOT_VELOCITY_WEIGHT,
        expected_checkpoint_steps,
    )

    assert ROOT_VELOCITY_WEIGHT == 1.0
    assert END_STEP == 2_162_688
    assert expected_checkpoint_steps() == (
        1_867_776,
        1_966_080,
        2_064_384,
        2_162_688,
    )


def test_root_velocity_kwargs_change_only_registered_treatment() -> None:
    from tools.run_g1_root_velocity_continuation import (
        build_root_velocity_kwargs,
    )

    checkpoint = Path("/tmp/checkpoint_step_1769472.pkl")
    treatment = build_root_velocity_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, enabled=True
    )
    control = build_root_velocity_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, checkpoint, enabled=False
    )

    assert treatment["tracking_root_velocity_weight"] == 1.0
    assert treatment["allow_resume_tracking_root_velocity_change"] is True
    assert treatment["actor_capture_point_tracking"] is False
    assert treatment["actor_centroidal_propulsion"] is False
    assert treatment["actor_counterfactual_wrench_distillation"] is False
    assert control["tracking_root_velocity_weight"] == 0.0
    assert control["allow_resume_tracking_root_velocity_change"] is False
    for key in (
        "tracking_root_velocity_weight",
        "allow_resume_tracking_root_velocity_change",
    ):
        treatment.pop(key)
        control.pop(key)
    assert treatment.keys() == control.keys()
    for key in treatment:
        treatment_leaves, treatment_tree = jax.tree_util.tree_flatten(
            treatment[key]
        )
        control_leaves, control_tree = jax.tree_util.tree_flatten(control[key])
        assert treatment_tree == control_tree, key
        for actual, expected in zip(
            treatment_leaves, control_leaves, strict=True
        ):
            np.testing.assert_equal(
                np.asarray(actual), np.asarray(expected), err_msg=key
            )


def _candidates(*survivals: tuple[int, int, int, int, int]):
    from tools.run_g1_root_velocity_continuation import expected_checkpoint_steps

    return {
        step: {"checkpoint_sha256": f"{index + 1:064x}", "survival": list(row)}
        for index, (step, row) in enumerate(
            zip(expected_checkpoint_steps(), survivals, strict=True)
        )
    }


def test_root_velocity_selector_retains_only_componentwise_safe_gain() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    result = classify_selection(
        _candidates(
            (131, 114, 74, 71, 74),
            (132, 114, 74, 71, 74),
            (131, 115, 75, 72, 75),
            (130, 200, 200, 200, 200),
        ),
        source_survival=[131, 114, 74, 71, 74],
    )

    assert result["outcome"] == "root-velocity-advances"
    assert result["selected_step"] == 2_064_384
    assert result["selected_survival"] == [131, 115, 75, 72, 75]
    assert result["policy_retained"] is True


def test_root_velocity_selector_marks_aggregate_gain_with_regression() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    result = classify_selection(
        _candidates(
            (130, 120, 90, 90, 90),
            (120, 130, 100, 100, 100),
            (100, 100, 100, 100, 100),
            (131, 114, 74, 71, 74),
        ),
        source_survival=[131, 114, 74, 71, 74],
    )

    assert result["outcome"] == "root-velocity-redistributes"
    assert result["selected_step"] is None
    assert result["policy_retained"] is False


def test_root_velocity_selector_rejects_bad_grid() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    with pytest.raises(ValueError, match="exact checkpoint grid"):
        classify_selection({}, source_survival=[131, 114, 74, 71, 74])


def test_root_velocity_selector_requires_contemporary_e026_corroboration() -> None:
    from tools.run_g1_root_velocity_continuation import classify_selection

    with pytest.raises(ValueError, match="source E026 survival"):
        classify_selection(
            _candidates(
                (131, 114, 74, 71, 74),
                (132, 114, 74, 71, 74),
                (131, 115, 75, 72, 75),
                (130, 200, 200, 200, 200),
            ),
            source_survival=[131, 114, 74, 70, 74],
        )


def test_evaluation_corroborates_source_and_binds_render_before_selection(
    tmp_path, monkeypatch
) -> None:
    import tools.run_g1_root_velocity_continuation as runner

    source = tmp_path / "source" / "checkpoint_step_1769472.pkl"
    source.parent.mkdir()
    source.write_bytes(b"source")
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"reference")
    for step in runner.expected_checkpoint_steps():
        (run / f"checkpoint_step_{step}.pkl").write_bytes(str(step).encode())

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "tools.evaluate_g1_flax_phase_grid" in command:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            output = Path(command[command.index("--output") + 1])
            is_source = checkpoint == source
            output.write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                        "reference_sha256": hashlib.sha256(
                            reference.read_bytes()
                        ).hexdigest(),
                        "tracking_root_velocity_weight": 0.0 if is_source else 1.0,
                        "summary": {
                            "phases": [0, 25, 50, 75, 100],
                            "survival": (
                                [131, 114, 74, 71, 74]
                                if is_source
                                else [132, 115, 75, 72, 75]
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            assert not (tmp_path / "output" / "selection.json").exists()
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "evaluation_start_phase": 0,
                        "tracking_root_velocity_weight": 1.0,
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                        "reference_sha256": hashlib.sha256(
                            reference.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            (output / "evaluation.mp4").write_bytes(b"video")
            (output / "contact_sheet.png").write_bytes(b"image")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "_plot_survival",
        lambda selection, output: output.write_bytes(b"plot"),
    )

    result = runner.evaluate_and_select(
        run,
        source_checkpoint=source,
        reference=reference,
        output_root=tmp_path / "output",
        code_commit="f" * 40,
    )

    assert len(calls) == 6
    assert Path(calls[0][calls[0].index("--checkpoint") + 1]) == source
    assert result["source_survival"] == [131, 114, 74, 71, 74]
    assert result["source_checkpoint_sha256"] == hashlib.sha256(b"source").hexdigest()
    assert result["render_checkpoint_sha256"] == result["selected_checkpoint_sha256"]
    assert len(result["render_summary_sha256"]) == 64
    assert len(result["render_mp4_sha256"]) == 64
    assert len(result["render_contact_sheet_sha256"]) == 64
    assert (tmp_path / "output" / "selection.json").is_file()
