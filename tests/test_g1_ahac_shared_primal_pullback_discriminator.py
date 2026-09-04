from __future__ import annotations

import inspect


def test_shared_primal_probe_changes_only_the_registered_runtime_mode(
    monkeypatch,
) -> None:
    from experiments.g1_ahac_shared_primal_pullback_discriminator import run

    baseline = {
        "actor_bootstrap_scale": 0.0,
        "actor_bootstrap_graph_mode": "excised",
        "actor_forward_jvp_probe": False,
        "actor_inactive_horizon_gradient_mode": "connected",
        "determinism_probe_output": "/old/probe.json",
        "unchanged": object(),
    }
    monkeypatch.setattr(
        run,
        "build_source_probe_kwargs",
        lambda *args, **kwargs: dict(baseline),
    )

    kwargs = run.build_probe_kwargs(
        "g1-4x5",
        "/reference.npz",
        0,
        "/checkpoint.pkl",
        probe_output="/paired.json",
    )

    assert kwargs["actor_bootstrap_scale"] == 0.0
    assert kwargs["actor_bootstrap_graph_mode"] == "excised"
    assert kwargs["actor_forward_jvp_probe"] is False
    assert kwargs["actor_inactive_horizon_gradient_mode"] == "runtime-paired"
    assert kwargs["determinism_probe_output"] == "/paired.json"
    assert kwargs["unchanged"] is baseline["unchanged"]


def test_runtime_pair_report_requires_shared_input_and_opposite_flags() -> None:
    from experiments.g1_ahac_shared_primal_pullback_discriminator.run import (
        extract_runtime_pair_arms,
    )

    connected = {
        "valid": True,
        "input_state_sha256": "same",
        "input_step": 11,
        "compiled_callable_reused": True,
        "ordinary_training_loop_entered": False,
        "runtime_excise_inactive": False,
    }
    excised = {**connected, "runtime_excise_inactive": True}
    report = {
        "protocol": "shac-runtime-pullback-pair-v1",
        "valid": True,
        "input_state_sha256": "same",
        "input_step": 11,
        "compiled_callable_reused": True,
        "ordinary_training_loop_entered": False,
        "arms": {"connected": connected, "excised": excised},
    }

    arms = extract_runtime_pair_arms(report)
    assert arms == (connected, excised)

    mismatched = {
        **report,
        "arms": {
            "connected": connected,
            "excised": {**excised, "input_state_sha256": "different"},
        },
    }
    try:
        extract_runtime_pair_arms(mismatched)
    except ValueError as error:
        assert "shared input" in str(error)
    else:
        raise AssertionError("mismatched runtime arms were accepted")


def test_shared_primal_runner_uses_one_train_invocation_without_subprocesses() -> None:
    from experiments.g1_ahac_shared_primal_pullback_discriminator import run

    source = inspect.getsource(run._run)
    assert source.count("train(**kwargs)") == 1
    assert "subprocess" not in source
    assert "runtime_paired_probe.json" in source
    assert "policy_retained" in source

