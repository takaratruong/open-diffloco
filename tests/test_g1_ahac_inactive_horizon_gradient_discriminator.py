from __future__ import annotations

import inspect


def test_inactive_horizon_probe_changes_only_the_registered_graph_mode(
    monkeypatch,
) -> None:
    from experiments.g1_ahac_inactive_horizon_gradient_discriminator import (
        run,
    )

    baseline = {
        "actor_bootstrap_scale": 0.0,
        "actor_bootstrap_graph_mode": "connected",
        "actor_forward_jvp_probe": True,
        "determinism_probe_output": "/old/probe.json",
        "unchanged": object(),
    }
    monkeypatch.setattr(
        run,
        "build_source_probe_kwargs",
        lambda *args, **kwargs: dict(baseline),
    )

    connected = run.build_probe_kwargs(
        "g1-4x5",
        "/reference.npz",
        0,
        "/checkpoint.pkl",
        arm="connected",
        probe_output="/connected.json",
    )
    excised = run.build_probe_kwargs(
        "g1-4x5",
        "/reference.npz",
        0,
        "/checkpoint.pkl",
        arm="excised",
        probe_output="/excised.json",
    )

    assert connected["actor_bootstrap_scale"] == 0.0
    assert connected["actor_bootstrap_graph_mode"] == "excised"
    assert connected["actor_forward_jvp_probe"] is False
    assert connected["actor_inactive_horizon_gradient_mode"] == "connected"
    assert excised["actor_inactive_horizon_gradient_mode"] == "excised"
    assert connected["determinism_probe_output"] == "/connected.json"
    assert excised["determinism_probe_output"] == "/excised.json"
    assert {
        name
        for name in connected
        if connected[name] != excised[name]
    } == {
        "actor_inactive_horizon_gradient_mode",
        "determinism_probe_output",
    }


def test_inactive_horizon_mask_classification_counts_exact_rows() -> None:
    from experiments.g1_ahac_inactive_horizon_gradient_discriminator.run import (
        classify_gradient_masks,
    )

    connected = [True] * 326 + [False] * 186
    partial = connected.copy()
    partial[326:336] = [True] * 10
    result = classify_gradient_masks(connected, partial)

    assert result["outcome"] == "inactive-excision-partially-restores-adjoints"
    assert result["connected_finite_count"] == 326
    assert result["excised_finite_count"] == 336
    assert result["recovered_indices"] == list(range(326, 336))
    assert result["introduced_nonfinite_indices"] == []

    complete = classify_gradient_masks(connected, [True] * 512)
    assert complete["outcome"] == "inactive-excision-restores-all-adjoints"

    unchanged = classify_gradient_masks(connected, connected)
    assert unchanged["outcome"] == "inactive-excision-preserves-invalid-mask"


def test_inactive_horizon_runner_isolates_each_compiled_arm() -> None:
    from experiments.g1_ahac_inactive_horizon_gradient_discriminator import (
        run,
    )

    source = inspect.getsource(run._run_parent)

    assert "for arm in ARMS:" in source
    assert "subprocess.run(command, check=False)" in source
    assert "computed_optimizer_updates_per_arm" in source
    assert "persisted_optimizer_updates" in source
