from __future__ import annotations

import json
import pickle
from pathlib import Path

import flax
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
)
from src.core.networks import Actor


@flax.struct.dataclass
class ToyState:
    actor_params: object
    normalizer: object
    marker: jax.Array


def _params(*, conditioned: bool, offset: float = 0.0):
    width = 4 if conditioned else 3
    return FrozenPreviewResidualParams(
        parent=freeze({"params": {"frozen": jnp.asarray([1.0, 2.0])}}),
        adapter=freeze(
            {
                "params": {
                    "Dense_0": {
                        "kernel": jnp.arange(width * 2, dtype=jnp.float32).reshape(
                            width, 2
                        )
                        + offset,
                        "bias": jnp.asarray([0.1, 0.2]) + offset,
                    },
                    "Dense_1": {
                        "kernel": jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
                        + offset,
                        "bias": jnp.asarray([0.3, 0.4]) + offset,
                    },
                }
            }
        ),
    )


def test_project_conditioned_target_drops_only_assistance_row():
    from tools.build_g1_residual_path_checkpoints import (
        project_zero_scale_target_adapter,
    )

    source = _params(conditioned=False)
    target = _params(conditioned=True, offset=10.0)

    projected = project_zero_scale_target_adapter(source, target)

    np.testing.assert_array_equal(projected.parent["params"]["frozen"], [1.0, 2.0])
    np.testing.assert_array_equal(
        projected.adapter["params"]["Dense_0"]["kernel"],
        target.adapter["params"]["Dense_0"]["kernel"][:-1],
    )
    np.testing.assert_array_equal(
        projected.adapter["params"]["Dense_1"]["kernel"],
        target.adapter["params"]["Dense_1"]["kernel"],
    )


def test_projection_is_action_exact_at_zero_assistance():
    from tools.build_g1_residual_path_checkpoints import (
        project_zero_scale_target_adapter,
    )

    parent_actor = Actor(
        action_dim=2,
        hidden=(4,),
        squash=True,
        layer_norm=False,
        zero_output=False,
    )
    residual_actor = PreviewResidualAdapter(action_dim=2, hidden_dim=4)
    observations = jnp.arange(15, dtype=jnp.float32).reshape(1, 15) / 10
    parent = parent_actor.init(jax.random.PRNGKey(1), observations)
    source_adapter = residual_actor.init(
        jax.random.PRNGKey(2), jnp.zeros((1, 5), dtype=jnp.float32)
    )
    target_adapter = residual_actor.init(
        jax.random.PRNGKey(3), jnp.zeros((1, 6), dtype=jnp.float32)
    )
    source = FrozenPreviewResidualParams(parent, source_adapter)
    target = FrozenPreviewResidualParams(parent, target_adapter)
    projected = project_zero_scale_target_adapter(source, target)

    conditioned_action, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        target,
        observations,
        history_len=3,
        treatment_frame_dim=5,
        assistance_scale=jnp.asarray(0.0),
    )
    projected_action, _, _ = apply_frozen_preview_residual(
        parent_actor,
        residual_actor,
        projected,
        observations,
        history_len=3,
        treatment_frame_dim=5,
    )

    np.testing.assert_array_equal(projected_action, conditioned_action)


def test_interpolation_preserves_source_parent_and_uses_exact_alpha():
    from tools.build_g1_residual_path_checkpoints import (
        interpolate_residual_actor_params,
    )

    source = _params(conditioned=False)
    target = _params(conditioned=True, offset=8.0)

    result = interpolate_residual_actor_params(source, target, alpha=0.25)

    np.testing.assert_array_equal(result.parent, source.parent)
    for source_leaf, target_leaf, result_leaf in zip(
        jax.tree_util.tree_leaves(source.adapter),
        jax.tree_util.tree_leaves(
            interpolate_residual_actor_params(source, target, alpha=1.0).adapter
        ),
        jax.tree_util.tree_leaves(result.adapter),
        strict=True,
    ):
        np.testing.assert_array_equal(
            result_leaf, source_leaf + 0.25 * (target_leaf - source_leaf)
        )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_interpolation_rejects_invalid_alpha(alpha):
    from tools.build_g1_residual_path_checkpoints import (
        interpolate_residual_actor_params,
    )

    with pytest.raises(ValueError, match="alpha"):
        interpolate_residual_actor_params(
            _params(conditioned=False),
            _params(conditioned=True),
            alpha=alpha,
        )


def test_interpolation_rejects_parent_drift_and_nonfinite_adapter():
    from tools.build_g1_residual_path_checkpoints import (
        interpolate_residual_actor_params,
    )

    source = _params(conditioned=False)
    parent_drift = _params(conditioned=True)._replace(
        parent=freeze({"params": {"frozen": jnp.asarray([1.0, 3.0])}})
    )
    with pytest.raises(ValueError, match="parent"):
        interpolate_residual_actor_params(source, parent_drift, alpha=0.5)

    nonfinite = _params(conditioned=True)
    adapter = flax.core.unfreeze(nonfinite.adapter)
    adapter["params"]["Dense_1"]["bias"] = adapter["params"]["Dense_1"][
        "bias"
    ].at[0].set(jnp.nan)
    nonfinite = nonfinite._replace(adapter=freeze(adapter))
    with pytest.raises(ValueError, match="finite"):
        interpolate_residual_actor_params(source, nonfinite, alpha=0.5)


def test_builder_writes_deterministic_hash_bound_manifest_last(tmp_path: Path):
    from tools.build_g1_residual_path_checkpoints import build_path_checkpoints

    source_path = tmp_path / "source.pkl"
    target_path = tmp_path / "target.pkl"
    output = tmp_path / "out"
    source_state = ToyState(
        actor_params=_params(conditioned=False),
        normalizer=freeze({"mean": jnp.asarray([1.0, 2.0])}),
        marker=jnp.asarray(7),
    )
    target_state = ToyState(
        actor_params=_params(conditioned=True, offset=2.0),
        normalizer=source_state.normalizer,
        marker=jnp.asarray(9),
    )
    source_path.write_bytes(pickle.dumps(source_state, protocol=pickle.HIGHEST_PROTOCOL))
    target_path.write_bytes(pickle.dumps(target_state, protocol=pickle.HIGHEST_PROTOCOL))

    manifest = build_path_checkpoints(
        source_path=source_path,
        target_path=target_path,
        output_dir=output,
        arm="aware",
        alphas=(0.125, 0.5),
    )

    assert (output / "manifest.json").is_file()
    persisted = json.loads((output / "manifest.json").read_text())
    assert persisted == manifest
    assert [row["alpha"] for row in persisted["checkpoints"]] == [0.125, 0.5]
    assert [row["filename"] for row in persisted["checkpoints"]] == [
        "aware_alpha_0p125.pkl",
        "aware_alpha_0p5.pkl",
    ]
    assert all(len(row["sha256"]) == 64 for row in persisted["checkpoints"])
    for row in persisted["checkpoints"]:
        state = pickle.loads((output / row["filename"]).read_bytes())
        np.testing.assert_array_equal(state.marker, source_state.marker)
        np.testing.assert_array_equal(state.normalizer["mean"], [1.0, 2.0])


def test_builder_rejects_normalizer_drift(tmp_path: Path):
    from tools.build_g1_residual_path_checkpoints import build_path_checkpoints

    source_path = tmp_path / "source.pkl"
    target_path = tmp_path / "target.pkl"
    source_path.write_bytes(
        pickle.dumps(
            ToyState(
                _params(conditioned=False),
                freeze({"mean": jnp.asarray([1.0])}),
                jnp.asarray(1),
            )
        )
    )
    target_path.write_bytes(
        pickle.dumps(
            ToyState(
                _params(conditioned=True),
                freeze({"mean": jnp.asarray([2.0])}),
                jnp.asarray(1),
            )
        )
    )

    with pytest.raises(ValueError, match="normalizer"):
        build_path_checkpoints(
            source_path=source_path,
            target_path=target_path,
            output_dir=tmp_path / "out",
            arm="blind",
            alphas=(0.25,),
        )
