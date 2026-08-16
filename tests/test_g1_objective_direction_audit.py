from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_fixed_population_contains_every_carried_row_before_repeats():
    from tools.evaluate_g1_objective_directions import (
        build_fixed_population_indices,
    )

    population = build_fixed_population_indices(seed=0)

    assert population["source_kind"].shape == (512,)
    assert population["source_index"].shape == (512,)
    np.testing.assert_array_equal(population["source_kind"][:128], 1)
    np.testing.assert_array_equal(population["source_kind"][128:], 0)
    np.testing.assert_array_equal(population["source_index"][:120], np.arange(120))
    assert len(np.unique(population["source_index"][120:128])) == 8
    assert np.all(population["source_index"][120:128] < 120)
    assert np.all(population["source_index"][128:] >= 0)
    assert np.all(population["source_index"][128:] < 499)


def test_fixed_population_is_deterministic_and_seed_zero_only():
    from tools.evaluate_g1_objective_directions import (
        build_fixed_population_indices,
    )

    first = build_fixed_population_indices(seed=0)
    second = build_fixed_population_indices(seed=0)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    with pytest.raises(ValueError, match="seed.*zero"):
        build_fixed_population_indices(seed=1)


def test_noise_tapes_are_float32_independent_and_share_h24_prefix():
    from tools.evaluate_g1_objective_directions import (
        build_fixed_noise_tapes,
        validate_common_noise_prefix,
    )

    tapes = build_fixed_noise_tapes(seed=0)

    assert tapes["a"].shape == (512, 48, 29)
    assert tapes["b"].shape == (512, 48, 29)
    assert tapes["a"].dtype == np.float32
    assert tapes["b"].dtype == np.float32
    assert not np.array_equal(tapes["a"], tapes["b"])
    assert validate_common_noise_prefix(tapes["a"][:, :24], tapes["a"])
    with pytest.raises(ValueError, match="prefix"):
        validate_common_noise_prefix(tapes["b"][:, :24], tapes["a"])
    with pytest.raises(ValueError, match="seed.*zero"):
        build_fixed_noise_tapes(seed=2)


def test_zero_head_transplant_preserves_initial_action_identity():
    import jax
    import jax.numpy as jnp

    from src.algorithms.shac.residual_preview_adapter import (
        FrozenPreviewResidualParams,
        PreviewResidualAdapter,
        apply_frozen_preview_residual,
        transplant_zero_head_recovery_features,
    )

    class Parent:
        def apply(self, params, observations):
            return observations[..., :2] @ params["kernel"]

    residual = PreviewResidualAdapter(action_dim=2, hidden_dim=3)
    frame = jnp.arange(4, dtype=jnp.float32)
    template = residual.init(jax.random.PRNGKey(0), frame)
    expert = residual.init(jax.random.PRNGKey(1), frame)
    transferred, report = transplant_zero_head_recovery_features(template, expert)
    observations = jnp.arange(8, dtype=jnp.float32)
    parent_params = {"kernel": jnp.ones((2, 2), dtype=jnp.float32)}

    action, parent, correction = apply_frozen_preview_residual(
        Parent(),
        residual,
        FrozenPreviewResidualParams(parent_params, transferred),
        observations,
        history_len=2,
        treatment_frame_dim=4,
    )

    np.testing.assert_array_equal(np.asarray(action), np.asarray(parent))
    np.testing.assert_array_equal(np.asarray(correction), np.zeros(2, np.float32))
    assert report["output_head_zero"] is True


def _gradient_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "source_kind": np.r_[np.ones(128), np.zeros(384)].astype(np.int8),
        "source_index": np.arange(512, dtype=np.int32),
        "phase": np.arange(512, dtype=np.int32) % 499,
        "rng_key": np.zeros((512, 2), dtype=np.uint32),
        "initial_qpos": np.zeros((512, 36), dtype=np.float64),
        "initial_qvel": np.zeros((512, 35), dtype=np.float64),
        "initial_last_act": np.zeros((512, 29), dtype=np.float64),
        "initial_actor_obs_history": np.zeros((512, 10, 328), dtype=np.float64),
        "noise_tape_a": np.zeros((512, 48, 29), dtype=np.float32),
        "noise_tape_b": np.ones((512, 48, 29), dtype=np.float32),
    }
    for direction in ("h24_a", "h24_b", "h48_a", "bootstrap_a"):
        arrays[f"{direction}_combined"] = np.ones(7, dtype=np.float64)
        arrays[f"{direction}_task"] = np.ones((5, 7), dtype=np.float64)
        arrays[f"{direction}_counts"] = np.full(5, 1, dtype=np.int32)
        arrays[f"{direction}_cosine"] = np.eye(5, dtype=np.float64)
        arrays[f"{direction}_weights"] = np.full(5, 0.2, dtype=np.float64)
        arrays[f"{direction}_env_norm"] = np.ones(512, dtype=np.float64)
    arrays["h24_tape_env_cosine"] = np.ones(512, dtype=np.float64)
    arrays["h24_h48_env_cosine"] = np.ones(512, dtype=np.float64)
    arrays["h24_a_bootstrap_env_cosine"] = np.ones(512, dtype=np.float64)
    arrays["h24_b_h48_env_cosine"] = np.ones(512, dtype=np.float64)
    arrays["h24_b_bootstrap_env_cosine"] = np.ones(512, dtype=np.float64)
    arrays["h48_bootstrap_env_cosine"] = np.ones(512, dtype=np.float64)
    return arrays


def test_gradient_artifacts_require_all_directions_and_nonzero_bins():
    from tools.evaluate_g1_objective_directions import (
        validate_gradient_artifacts,
    )

    summary = validate_gradient_artifacts(_gradient_arrays())
    assert summary["valid"] is True
    assert summary["population"] == 512
    broken = _gradient_arrays()
    broken["h48_a_counts"][2] = 0
    with pytest.raises(ValueError, match="phase bins"):
        validate_gradient_artifacts(broken)


def test_manifest_last_publication_rejects_tampering(tmp_path: Path):
    from tools.evaluate_g1_objective_directions import (
        publish_gradient_artifacts,
        validate_completion_manifest,
    )

    manifest = publish_gradient_artifacts(
        tmp_path,
        _gradient_arrays(),
        provenance={
            "code_commit": "a" * 40,
            "input_sha256": {
                name: str(index) * 64
                for index, name in enumerate(
                    ("checkpoint", "hparams", "reference", "bank", "expert", "model", "controller"),
                    start=1,
                )
            },
        },
        scientific=False,
    )
    assert validate_completion_manifest(tmp_path / "completion.json")["valid"]
    assert manifest["scientific"] is False
    artifact = Path(manifest["gradient_artifact_path"])
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_completion_manifest(tmp_path / "completion.json")


def test_completion_is_written_after_summary(tmp_path: Path):
    from tools.evaluate_g1_objective_directions import publish_gradient_artifacts

    publish_gradient_artifacts(
        tmp_path,
        _gradient_arrays(),
        provenance={"code_commit": "b" * 40, "input_sha256": {}},
        scientific=False,
    )

    completion = json.loads((tmp_path / "completion.json").read_text())
    assert (tmp_path / "gradient_summary.json").is_file()
    assert completion["summary_sha256"]


def test_proposal_specs_are_exact_and_equal_norm():
    from tools.evaluate_g1_objective_directions import proposal_specs

    specs = proposal_specs()

    assert len(specs) == 12
    assert [row["direction"] for row in specs] == ["h24"] * 4 + ["h48"] * 4 + ["bootstrap"] * 4
    assert [row["multiplier"] for row in specs[:4]] == [0.125, 0.25, 0.5, 1.0]
    assert specs[0]["displacement"] == pytest.approx(
        0.09495018422603607 * 0.125, rel=0, abs=1e-15
    )
    assert specs[-1]["displacement"] == pytest.approx(
        0.09495018422603607, rel=0, abs=1e-15
    )
    assert len({row["label"] for row in specs}) == 12


def _line_search_arrays() -> dict[str, np.ndarray]:
    return {
        "baseline_carried_survival": np.full(120, 12, dtype=np.int32),
        "candidate_carried_survival": np.full((12, 120), 12, dtype=np.int32),
        "selected_proposal_index": np.asarray([-1, -1, -1], dtype=np.int32),
        "baseline_ordinary_survival": np.asarray([116, 63, 49, 39, 47], dtype=np.int32),
        "selected_ordinary_survival": np.full((3, 5), -1, dtype=np.int32),
        "full_gate": np.zeros(3, dtype=bool),
    }


def test_line_search_evidence_requires_complete_bounded_grid():
    from tools.evaluate_g1_objective_directions import validate_line_search_evidence

    summary = validate_line_search_evidence(_line_search_arrays())
    assert summary["valid"] is True
    assert summary["proposal_count"] == 12
    broken = _line_search_arrays()
    broken["candidate_carried_survival"][0, 0] = 33
    with pytest.raises(ValueError, match="H32"):
        validate_line_search_evidence(broken)


def test_final_manifest_binds_gradient_line_search_selection_and_plots(tmp_path: Path):
    from tools.evaluate_g1_objective_directions import (
        publish_final_manifest,
        validate_final_manifest,
    )

    artifacts = {}
    for name in (
        "preflight.json",
        "gradient_evidence.npz",
        "gradient_summary.json",
        "line_search.npz",
        "line_search.json",
        "selection.json",
        "cosine_heatmap.png",
        "survival_plot.png",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[name] = path
    completion = publish_final_manifest(
        tmp_path,
        artifacts=artifacts,
        outcome="direction-audit-inconclusive",
        code_commit="c" * 40,
        input_sha256={"checkpoint": "d" * 64},
    )
    assert validate_final_manifest(tmp_path / "completion.json")["outcome"] == "direction-audit-inconclusive"
    assert "retained_checkpoint" not in completion
    artifacts["selection.json"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash"):
        validate_final_manifest(tmp_path / "completion.json")
