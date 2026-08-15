from __future__ import annotations

import hashlib
import inspect
import json
import pickle

import jax
import jax.numpy as jnp
import pytest


def _adapter_params():
    from src.algorithms.shac.residual_preview_adapter import PreviewResidualAdapter

    return PreviewResidualAdapter(action_dim=2, hidden_dim=4).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 5), dtype=jnp.float32)
    )


def test_train_exposes_and_persists_zero_head_feature_source():
    from src.algorithms.shac import algorithm

    signature = inspect.signature(algorithm.train)
    assert "actor_residual_preview_initial_adapter_path" in signature.parameters
    assert "actor_residual_preview_initial_adapter_sha256" in signature.parameters
    source = inspect.getsource(algorithm.train)
    assert '"actor_residual_preview_initial_adapter_path"' in source
    assert '"actor_residual_preview_initial_adapter_sha256"' in source
    assert source.index("load_zero_head_recovery_feature_adapter") < source.index(
        "initialize_residual_adapter_optimizer"
    )
    assert "if resume_from is None:" in source
    assert "is_resume=False" in source


def test_loader_hash_binds_expert_and_persists_manifest(tmp_path):
    from src.algorithms.shac.algorithm import (
        load_zero_head_recovery_feature_adapter,
        persist_zero_head_feature_transfer_report,
    )

    template = _adapter_params()
    expert = _adapter_params()
    expert["params"]["Dense_0"]["bias"] = jnp.arange(4, dtype=jnp.float32)
    source = tmp_path / "expert.pkl"
    source.write_bytes(pickle.dumps(expert))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    candidate, report = load_zero_head_recovery_feature_adapter(
        source,
        expected_sha256=digest,
        template_params=template,
    )
    path = persist_zero_head_feature_transfer_report(tmp_path, report)

    assert candidate["params"]["Dense_0"]["bias"].tolist() == [0, 1, 2, 3]
    assert report["source_path"] == str(source.resolve())
    assert report["source_sha256"] == digest
    assert json.loads(path.read_text()) == report
    with pytest.raises(ValueError, match="SHA-256"):
        load_zero_head_recovery_feature_adapter(
            source,
            expected_sha256="0" * 64,
            template_params=template,
        )
