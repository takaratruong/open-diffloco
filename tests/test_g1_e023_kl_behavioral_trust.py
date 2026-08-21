from __future__ import annotations

import pickle
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np
import pytest


E023_CHECKPOINT = Path(
    "/home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-023/"
    "20260815T023359Z/seed-0/training_runs/training_runs/"
    "shac_20260814_193402/checkpoint_step_1572864.pkl"
)
E023_AUDIT_BANK = Path(
    "/home/ubuntu/projects/diffsim2real-lab/runs/E-20260816-002/"
    "20260816T193304Z/seed-0/objective_audit/gradient_evidence.npz"
)


@dataclass(frozen=True)
class FakeState:
    actor_params: object
    normalizer: object
    ignored: object

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def test_interpolate_actor_tree_preserves_endpoints_and_midpoint():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        interpolate_actor_tree,
    )

    source = {"w": np.array([0.0, 2.0], dtype=np.float32)}
    proposal = {"w": np.array([2.0, 4.0], dtype=np.float32)}

    midpoint = interpolate_actor_tree(source, proposal, alpha=0.5)
    endpoint = interpolate_actor_tree(source, proposal, alpha=1.0)

    np.testing.assert_array_equal(midpoint["w"], [1.0, 3.0])
    np.testing.assert_array_equal(endpoint["w"], proposal["w"])


def test_interpolate_actor_tree_rejects_structure_nonfinite_and_unregistered_alpha():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        interpolate_actor_tree,
    )

    source = {"w": np.array([0.0, 2.0], dtype=np.float32)}
    proposal = {"w": np.array([2.0, 4.0], dtype=np.float32)}

    with pytest.raises(ValueError, match="structure"):
        interpolate_actor_tree(source, {"x": proposal["w"]}, alpha=0.5)
    with pytest.raises(ValueError, match="finite"):
        interpolate_actor_tree(
            source,
            {"w": np.array([np.nan, 4.0], dtype=np.float32)},
            alpha=0.5,
        )
    with pytest.raises(ValueError, match="registered"):
        interpolate_actor_tree(source, proposal, alpha=0.2)


def test_gaussian_proximity_matches_closed_form_and_zero_identity():
    from tools.evaluate_g1_e023_kl_behavioral_trust import gaussian_proximity

    old = np.zeros((2, 2), dtype=np.float64)
    new = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    std = np.array([2.0, 4.0], dtype=np.float64)
    epsilon = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float64)

    report = gaussian_proximity(old, new, std, epsilon)
    identity = gaussian_proximity(old, old, std, epsilon)

    np.testing.assert_allclose(
        report["kl_by_state_action"], [[0.125, 0.0], [0.0, 0.125]]
    )
    np.testing.assert_allclose(report["kl_by_state"], [0.125, 0.125])
    assert report["mean_kl"] == pytest.approx(0.0625)
    assert report["p95_state_kl"] == pytest.approx(0.125)
    assert report["max_state_kl"] == pytest.approx(0.125)
    assert np.isfinite(report["log_ratio_by_state"]).all()
    assert identity["mean_kl"] == 0.0
    assert identity["p95_state_kl"] == 0.0
    assert identity["max_state_kl"] == 0.0
    np.testing.assert_array_equal(identity["log_ratio_by_state"], 0.0)


@pytest.mark.parametrize(
    ("std", "match"),
    [
        (np.array([1.0]), "shape"),
        (np.array([1.0, 0.0]), "positive"),
        (np.array([1.0, np.nan]), "finite"),
    ],
)
def test_gaussian_proximity_rejects_invalid_standard_deviation(std, match):
    from tools.evaluate_g1_e023_kl_behavioral_trust import gaussian_proximity

    values = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=match):
        gaussian_proximity(values, values, std, values)


def _row(alpha, *, kl=True, survival=(116, 99, 67, 49, 24)):
    return {
        "alpha": alpha,
        "kl_feasible": kl,
        "survival": list(survival),
        "valid": True,
    }


def test_selector_chooses_largest_safe_alpha_and_marks_advancement():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    rows = [_row(alpha, kl=alpha <= 0.25) for alpha in CANDIDATE_ALPHAS]
    rows[2]["survival"] = [117, 99, 67, 49, 24]

    result = select_kl_behavioral_candidate(rows)

    assert result["outcome"] == "kl-trust-advances"
    assert result["selected_alpha"] == 0.25


def test_selector_does_not_compensate_for_any_phase_regression():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    rows = [_row(alpha, kl=False) for alpha in CANDIDATE_ALPHAS]
    rows[-1] = _row(
        CANDIDATE_ALPHAS[-1],
        kl=True,
        survival=(124, 99, 74, 49, 23),
    )

    result = select_kl_behavioral_candidate(rows)

    assert result["outcome"] == "kl-feasible-behaviorally-unsafe"
    assert result["selected_alpha"] is None


def test_selector_distinguishes_preservation_and_no_kl_candidate():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    preserving = [_row(alpha, kl=alpha == 0.125) for alpha in CANDIDATE_ALPHAS]
    absent = [_row(alpha, kl=False) for alpha in CANDIDATE_ALPHAS]

    assert (
        select_kl_behavioral_candidate(preserving)["outcome"]
        == "kl-trust-preserves-only"
    )
    assert (
        select_kl_behavioral_candidate(absent)["outcome"]
        == "no-nonzero-kl-candidate"
    )


def test_selector_rejects_incomplete_or_reordered_candidate_grid():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        select_kl_behavioral_candidate,
    )

    rows = [_row(alpha) for alpha in CANDIDATE_ALPHAS]
    with pytest.raises(ValueError, match="candidate grid"):
        select_kl_behavioral_candidate(rows[:-1])
    with pytest.raises(ValueError, match="candidate grid"):
        select_kl_behavioral_candidate(list(reversed(rows)))


def test_build_evaluation_candidate_contains_only_actor_and_source_normalizer():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        EvaluationActorState,
        build_evaluation_candidate,
    )

    source_normalizer = {"mean": np.array([1.0], dtype=np.float32)}
    source = FakeState(
        actor_params={"w": np.array([0.0, 2.0], dtype=np.float32)},
        normalizer=source_normalizer,
        ignored={"optimizer": "source"},
    )
    proposal = FakeState(
        actor_params={"w": np.array([2.0, 4.0], dtype=np.float32)},
        normalizer={"mean": np.array([9.0], dtype=np.float32)},
        ignored={"optimizer": "proposal"},
    )

    candidate = build_evaluation_candidate(source, proposal, alpha=0.5)

    assert isinstance(candidate, EvaluationActorState)
    np.testing.assert_array_equal(candidate.actor_params["w"], [1.0, 3.0])
    assert candidate.normalizer is source_normalizer
    assert not hasattr(candidate, "ignored")
    assert candidate.evaluation_only is True
    assert candidate.resumable is False


@pytest.mark.skipif(
    not E023_CHECKPOINT.is_file() or not E023_AUDIT_BANK.is_file(),
    reason="frozen E023 evidence is unavailable",
)
def test_real_e023_actor_applies_to_frozen_audit_histories_on_cpu():
    from tools.evaluate_g1_e023_kl_behavioral_trust import actor_means_from_bank

    with E023_CHECKPOINT.open("rb") as stream:
        state = pickle.load(stream)
    report = actor_means_from_bank(state, E023_AUDIT_BANK, limit=4)
    with np.load(E023_AUDIT_BANK, allow_pickle=False) as archive:
        expected_epsilon = archive["noise_tape_a"][:4, 0, :]

    assert report["actor_mean"].shape == (4, 29)
    assert report["epsilon"].shape == (4, 29)
    assert np.isfinite(report["actor_mean"]).all()
    np.testing.assert_array_equal(report["epsilon"], expected_epsilon)
    assert jax.default_backend() == "cpu"


def test_publish_candidates_is_compact_atomic_and_hash_bound(tmp_path):
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        EvaluationActorState,
        publish_evaluation_candidates,
        validate_candidate_manifest,
    )

    source = FakeState(
        actor_params={"w": np.array([0.0, 2.0], dtype=np.float32)},
        normalizer={"mean": np.array([1.0], dtype=np.float32)},
        ignored=np.ones(1000, dtype=np.float64),
    )
    proposal = FakeState(
        actor_params={"w": np.array([2.0, 4.0], dtype=np.float32)},
        normalizer={"mean": np.array([9.0], dtype=np.float32)},
        ignored=np.zeros(1000, dtype=np.float64),
    )
    source_path = tmp_path / "source.pkl"
    proposal_path = tmp_path / "proposal.pkl"
    source_path.write_bytes(pickle.dumps(source))
    proposal_path.write_bytes(pickle.dumps(proposal))
    (tmp_path / "hparams.json").write_text(
        json.dumps({"tracking_velocity_kernel": "exponential"}) + "\n"
    )

    manifest = publish_evaluation_candidates(
        source_path, proposal_path, tmp_path / "candidates"
    )
    validated = validate_candidate_manifest(
        tmp_path / "candidates" / "candidate_manifest.json"
    )

    assert manifest == validated
    assert [row["alpha"] for row in manifest["candidates"]] == list(
        CANDIDATE_ALPHAS
    )
    assert manifest["evaluation_only"] is True
    assert manifest["resumable"] is False
    for row in manifest["candidates"]:
        path = Path(row["checkpoint_path"])
        assert path.stat().st_size < source_path.stat().st_size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        with path.open("rb") as stream:
            candidate = pickle.load(stream)
        assert isinstance(candidate, EvaluationActorState)
        np.testing.assert_array_equal(
            candidate.normalizer["mean"], source.normalizer["mean"]
        )


def test_candidate_manifest_rejects_tampered_checkpoint(tmp_path):
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        publish_evaluation_candidates,
        validate_candidate_manifest,
    )

    state = FakeState(
        actor_params={"w": np.array([0.0], dtype=np.float32)},
        normalizer={"mean": np.array([1.0], dtype=np.float32)},
        ignored=None,
    )
    source_path = tmp_path / "source.pkl"
    proposal_path = tmp_path / "proposal.pkl"
    source_path.write_bytes(pickle.dumps(state))
    proposal_path.write_bytes(pickle.dumps(state))
    (tmp_path / "hparams.json").write_text("{}\n")
    output = tmp_path / "candidates"
    manifest = publish_evaluation_candidates(source_path, proposal_path, output)
    Path(manifest["candidates"][0]["checkpoint_path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_candidate_manifest(output / "candidate_manifest.json")


def test_build_proximity_summary_matches_raw_candidate_order():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        build_proximity_summary,
    )

    old = np.zeros((3, 2), dtype=np.float64)
    candidates = np.stack(
        [np.full_like(old, alpha * 0.01) for alpha in CANDIDATE_ALPHAS]
    )
    summary = build_proximity_summary(
        old,
        candidates,
        np.ones(2, dtype=np.float64),
        np.zeros_like(old),
    )

    assert [row["alpha"] for row in summary["candidates"]] == list(
        CANDIDATE_ALPHAS
    )
    assert summary["source_identity"]["mean_kl"] == 0.0
    assert summary["source_identity"]["max_state_kl"] == 0.0
    assert all(row["valid"] is True for row in summary["candidates"])
    assert summary["candidates"][0]["mean_kl"] > summary["candidates"][-1][
        "mean_kl"
    ]


def test_build_proximity_summary_rejects_candidate_shape_drift():
    from tools.evaluate_g1_e023_kl_behavioral_trust import build_proximity_summary

    old = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="candidate means"):
        build_proximity_summary(
            old,
            np.zeros((6, 3, 3), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.zeros_like(old),
        )


def _phase_grid(survival, *, kernel="exponential"):
    phases = [0, 25, 50, 75, 100]
    return {
        "protocol": "g1-flax-dance-replay-free-five-phase-v1",
        "solver_profile": "g1-4x5",
        "tracking_velocity_kernel": kernel,
        "summary": {"phases": phases, "survival": list(survival)},
        "results": [
            {
                "phase": phase,
                "steps": steps,
                "terminal": steps < maximum,
                "mean_reward": 1.0,
                "mean_body_position_error": 0.1,
                "mean_body_orientation_error": 0.1,
                "mean_body_linear_velocity_error": 0.1,
                "mean_body_angular_velocity_error": 0.1,
                "mean_anchor_position_error": 0.1,
                "mean_anchor_orientation_error": 0.1,
            }
            for phase, steps, maximum in zip(
                phases, survival, [124, 99, 74, 49, 24], strict=True
            )
        ],
    }


def test_build_phase_selection_joins_kl_and_replay_free_evidence():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        build_phase_selection,
    )

    proximity = {
        "valid": True,
        "candidates": [
            {"alpha": alpha, "valid": True, "kl_feasible": alpha <= 0.25}
            for alpha in CANDIDATE_ALPHAS
        ],
    }
    grids = [
        _phase_grid((117, 99, 67, 49, 24) if alpha == 0.25 else (115, 99, 67, 49, 24))
        for alpha in CANDIDATE_ALPHAS
    ]

    selection = build_phase_selection(
        proximity, _phase_grid((116, 99, 67, 49, 24)), grids
    )

    assert selection["outcome"] == "kl-trust-advances"
    assert selection["selected_alpha"] == 0.25
    assert selection["candidates"][2]["survival"] == [117, 99, 67, 49, 24]


def test_build_phase_selection_rejects_kernel_and_source_floor_drift():
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        CANDIDATE_ALPHAS,
        build_phase_selection,
    )

    proximity = {
        "valid": True,
        "candidates": [
            {"alpha": alpha, "valid": True, "kl_feasible": True}
            for alpha in CANDIDATE_ALPHAS
        ],
    }
    candidates = [_phase_grid((116, 99, 67, 49, 24))] * 6
    with pytest.raises(ValueError, match="exponential"):
        build_phase_selection(
            proximity,
            _phase_grid((116, 99, 67, 49, 24), kernel="pseudo_huber"),
            candidates,
        )
    with pytest.raises(ValueError, match="protected E023"):
        build_phase_selection(
            proximity,
            _phase_grid((115, 99, 67, 49, 24)),
            candidates,
        )


def test_cli_and_phase_grid_command_pin_the_exact_e023_evaluator(tmp_path):
    from tools.evaluate_g1_e023_kl_behavioral_trust import (
        build_parser,
        build_phase_grid_command,
    )

    args = build_parser().parse_args(
        [
            "--source-checkpoint",
            "/tmp/source.pkl",
            "--proposal-checkpoint",
            "/tmp/proposal.pkl",
            "--audit-bank",
            "/tmp/bank.npz",
            "--reference-path",
            "/tmp/reference.npz",
            "--output-root",
            "/tmp/output",
            "--code-commit",
            "a" * 40,
        ]
    )
    command = build_phase_grid_command(
        checkpoint=tmp_path / "candidate.pkl",
        reference_path=tmp_path / "reference.npz",
        output_path=tmp_path / "phase.json",
        code_commit="a" * 40,
        python_executable="/python",
    )

    assert args.seed == 0
    assert command == [
        "/python",
        "-m",
        "tools.evaluate_g1_flax_phase_grid",
        "--checkpoint",
        str((tmp_path / "candidate.pkl").resolve()),
        "--reference-path",
        str((tmp_path / "reference.npz").resolve()),
        "--output",
        str((tmp_path / "phase.json").resolve()),
        "--phases",
        "0",
        "25",
        "50",
        "75",
        "100",
        "--seed",
        "0",
        "--solver-profile",
        "g1-4x5",
        "--code-commit",
        "a" * 40,
    ]
