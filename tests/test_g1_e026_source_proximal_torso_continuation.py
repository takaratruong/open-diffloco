from pathlib import Path


def test_source_proximal_runner_changes_only_the_anchor_target():
    from tools.run_g1_e026_source_proximal_torso_continuation import (
        build_source_proximal_kwargs,
    )

    source = Path("/tmp/checkpoint_step_1769472.pkl")
    kwargs = build_source_proximal_kwargs(
        "g1-4x5", "/tmp/reference.npz", 0, source, "/tmp/carried.npz"
    )

    assert kwargs["actor_policy_anchor_weight"] == 1.0
    assert kwargs["actor_policy_anchor_source_path"] == str(source.resolve())
    assert len(kwargs["actor_policy_anchor_source_sha256"]) == 64
    assert kwargs["allow_resume_actor_policy_anchor_source_change"] is True
    assert kwargs["tracking_torso_orientation_weight"] == 1.0
    assert kwargs["total_steps"] == 2_162_688


def test_source_proximal_preflight_records_the_exact_scientific_delta(tmp_path):
    from tools.run_g1_e026_source_proximal_torso_continuation import (
        extend_source_proximal_preflight,
    )

    source = tmp_path / "source.pkl"
    source.write_bytes(b"source")
    report = extend_source_proximal_preflight(
        {"valid": True, "scientific_delta": ["old"]},
        source_path=source,
        source_sha256=__import__("hashlib").sha256(b"source").hexdigest(),
    )

    assert report["valid"] is True
    assert report["protocol"] == "g1-e026-source-proximal-torso-preflight-v1"
    assert report["policy_anchor_source_path"] == str(source.resolve())
    assert report["scientific_delta"] == [
        "actor_policy_anchor_source_path",
        "actor_policy_anchor_source_sha256",
        "allow_resume_actor_policy_anchor_source_change",
    ]
