from pathlib import Path
import json
import pickle
from types import SimpleNamespace

import numpy as np


def test_fresh_walking_continuation_changes_only_resume_and_budget(
    tmp_path: Path,
) -> None:
    from tools.run_g1_fresh_fixed_noise_training import (
        build_fresh_fixed_noise_kwargs,
    )
    from tools.run_g1_fresh_walking_shac_continuation import (
        CONTINUATION_END_STEP,
        build_fresh_walking_continuation_kwargs,
        expected_checkpoint_steps,
    )

    reference = tmp_path / "walk.npz"
    checkpoint = tmp_path / "checkpoint_step_786432.pkl"
    parent = build_fresh_fixed_noise_kwargs(
        "g1-4x5",
        reference,
        0,
        actor_lr=1e-3,
        actor_per_env_grad_clip=1.0,
    )
    candidate = build_fresh_walking_continuation_kwargs(
        "g1-4x5", reference, 0, checkpoint
    )
    delta = {
        key: candidate.get(key)
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }

    assert delta == {
        "checkpoint_interval": 393_216,
        "resume_from": str(checkpoint),
        "total_steps": CONTINUATION_END_STEP,
    }
    assert expected_checkpoint_steps() == (
        1_179_648,
        1_572_864,
        1_966_080,
        2_359_296,
        2_752_512,
        3_145_728,
    )


def test_fresh_walking_continuation_parser_has_no_scientific_overrides() -> None:
    import pytest

    from tools.run_g1_fresh_walking_shac_continuation import build_parser

    required = [
        "--solver-profile",
        "g1-4x5",
        "--reference-path",
        "/tmp/walk.npz",
        "--resume-from",
        "/tmp/checkpoint_step_786432.pkl",
        "--code-commit",
        "0" * 40,
    ]
    args = build_parser().parse_args(required)
    assert args.resume_from == Path("/tmp/checkpoint_step_786432.pkl")

    for override in (
        ["--actor-lr", "0.0005"],
        ["--action-noise-std", "0.1"],
        ["--total-steps", "4000000"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([*required, *override])


def test_training_validation_uses_artifacts_for_unpersisted_resume_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tools.run_g1_fresh_walking_shac_continuation import (
        build_fresh_walking_continuation_kwargs,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    checkpoint = tmp_path / "parent" / "checkpoint_step_786432.pkl"
    checkpoint.parent.mkdir()
    kwargs = build_fresh_walking_continuation_kwargs(
        "g1-4x5", tmp_path / "walk.npz", 0, checkpoint
    )
    # train() does not persist these execution-only inputs in hparams.json.
    kwargs.pop("checkpoint_interval")
    kwargs.pop("resume_from")
    (tmp_path / "hparams.json").write_text(json.dumps(kwargs))

    rows = []
    for step in expected_checkpoint_steps():
        with (tmp_path / f"checkpoint_step_{step}.pkl").open("wb") as stream:
            pickle.dump(
                SimpleNamespace(
                    step=np.asarray(step), payload=np.asarray([1.0])
                ),
                stream,
            )
        rows.append(
            {
                "step": step,
                "actor_cagrad_valid": True,
                "actor_cagrad_bin_counts": [1.0] * 5,
                "actor_cagrad_bin_gradient_norms": [0.5] * 5,
                "actor_cagrad_bin_losses": [0.1] * 5,
                "actor_cagrad_weights": [0.2] * 5,
                "actor_cagrad_gram_matrix": [[0.1] * 5 for _ in range(5)],
                "actor_cagrad_cosine_matrix": [[0.1] * 5 for _ in range(5)],
                "actor_cagrad_objective": 0.1,
                "actor_cagrad_dual_gap": 0.1,
                "actor_cagrad_uniform_combined_cosine": 0.1,
                "actor_cagrad_combined_norm": 0.1,
                "actor_bootstrap_scale_current": 0.0,
                "action_noise_current": 0.2,
            }
        )
    final_bytes = (
        tmp_path / "checkpoint_step_3145728.pkl"
    ).read_bytes()
    (tmp_path / "checkpoint_latest.pkl").write_bytes(final_bytes)
    (tmp_path / "policy_final.pkl").write_bytes(final_bytes)
    (tmp_path / "checkpoint_phase_metrics.json").write_text(json.dumps(rows))

    monkeypatch.setattr(
        "tools.run_g1_fresh_walking_shac_continuation."
        "build_fresh_walking_continuation_kwargs",
        lambda *_args, **_kwargs: kwargs,
    )
    validation = validate_training_artifacts(tmp_path)
    assert validation["valid"] is True

    (tmp_path / "checkpoint_step_1179648.pkl").write_bytes(b"not a pickle")
    import pytest

    with pytest.raises(ValueError, match="invalid checkpoint"):
        validate_training_artifacts(tmp_path)


def test_training_validation_rejects_incomplete_cagrad_telemetry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pytest

    from tools.run_g1_fresh_walking_shac_continuation import (
        build_fresh_walking_continuation_kwargs,
        expected_checkpoint_steps,
        validate_training_artifacts,
    )

    checkpoint = tmp_path / "parent" / "checkpoint_step_786432.pkl"
    checkpoint.parent.mkdir()
    kwargs = build_fresh_walking_continuation_kwargs(
        "g1-4x5", tmp_path / "walk.npz", 0, checkpoint
    )
    kwargs.pop("checkpoint_interval")
    kwargs.pop("resume_from")
    (tmp_path / "hparams.json").write_text(json.dumps(kwargs))
    rows = []
    for step in expected_checkpoint_steps():
        with (tmp_path / f"checkpoint_step_{step}.pkl").open("wb") as stream:
            pickle.dump(SimpleNamespace(step=np.asarray(step)), stream)
        rows.append(
            {
                "step": step,
                "actor_cagrad_valid": True,
                "actor_cagrad_bin_counts": [1.0] * 5,
                "actor_cagrad_bin_gradient_norms": [0.5] * 5,
            }
        )
    final_bytes = (
        tmp_path / "checkpoint_step_3145728.pkl"
    ).read_bytes()
    (tmp_path / "checkpoint_latest.pkl").write_bytes(final_bytes)
    (tmp_path / "policy_final.pkl").write_bytes(final_bytes)
    (tmp_path / "checkpoint_phase_metrics.json").write_text(json.dumps(rows))
    monkeypatch.setattr(
        "tools.run_g1_fresh_walking_shac_continuation."
        "build_fresh_walking_continuation_kwargs",
        lambda *_args, **_kwargs: kwargs,
    )

    with pytest.raises(ValueError, match="CAGrad telemetry"):
        validate_training_artifacts(tmp_path)
