"""Build fail-closed residual-policy interpolation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    merge_residual_adapter_params,
    split_residual_adapter_params,
)


DEFAULT_ALPHAS = (0.125, 0.25, 0.5, 0.75)


def _tree_exact(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return left_structure == right_structure and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(leaves) and all(
        bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in leaves
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _alpha_token(alpha: float) -> str:
    return format(alpha, ".12g").replace(".", "p")


def project_zero_scale_target_adapter(
    source: FrozenPreviewResidualParams,
    target: FrozenPreviewResidualParams,
) -> FrozenPreviewResidualParams:
    """Remove only a target assistance row that is inactive at scale zero."""
    if not isinstance(source, FrozenPreviewResidualParams) or not isinstance(
        target, FrozenPreviewResidualParams
    ):
        raise ValueError("source and target must be frozen residual actors")
    if not _tree_exact(source.parent, target.parent):
        raise ValueError("source and target parent actors must be bit-identical")
    if not _tree_finite(source.adapter) or not _tree_finite(target.adapter):
        raise ValueError("source and target adapters must be finite")

    source_kernel, _ = split_residual_adapter_params(source.adapter)
    target_kernel, target_auxiliary = split_residual_adapter_params(
        target.adapter
    )
    if target_kernel.shape == source_kernel.shape:
        projected_kernel = target_kernel
    elif target_kernel.shape == (
        source_kernel.shape[0] + 1,
        source_kernel.shape[1],
    ):
        projected_kernel = target_kernel[:-1]
    else:
        raise ValueError(
            "target adapter must match source or add one assistance row"
        )
    projected_adapter = merge_residual_adapter_params(
        source.adapter,
        projected_kernel,
        target_auxiliary,
    )
    if jax.tree_util.tree_structure(projected_adapter) != jax.tree_util.tree_structure(
        source.adapter
    ):
        raise ValueError("projected target adapter structure does not match source")
    for source_leaf, projected_leaf in zip(
        jax.tree_util.tree_leaves(source.adapter),
        jax.tree_util.tree_leaves(projected_adapter),
        strict=True,
    ):
        if source_leaf.shape != projected_leaf.shape:
            raise ValueError("projected target adapter shape does not match source")
    return FrozenPreviewResidualParams(
        parent=source.parent,
        adapter=projected_adapter,
    )


def interpolate_residual_actor_params(
    source: FrozenPreviewResidualParams,
    target: FrozenPreviewResidualParams,
    *,
    alpha: float,
) -> FrozenPreviewResidualParams:
    """Interpolate adapter leaves while preserving the source parent exactly."""
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between zero and one")
    projected = project_zero_scale_target_adapter(source, target)
    adapter = jax.tree_util.tree_map(
        lambda start, end: start + alpha * (end - start),
        source.adapter,
        projected.adapter,
    )
    if not _tree_finite(adapter):
        raise ValueError("interpolated adapter must be finite")
    return FrozenPreviewResidualParams(parent=source.parent, adapter=adapter)


def _replace_actor_params(state: Any, actor_params: Any) -> Any:
    if hasattr(state, "replace"):
        return state.replace(actor_params=actor_params)
    if hasattr(state, "_replace"):
        return state._replace(actor_params=actor_params)
    raise ValueError("source checkpoint state does not support replacement")


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_path_checkpoints(
    *,
    source_path: Path,
    target_path: Path,
    output_dir: Path,
    arm: str,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """Build immutable diagnostic checkpoints and publish a manifest last."""
    source_path = source_path.resolve()
    target_path = target_path.resolve()
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("source and target checkpoint files are required")
    if not arm or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in arm):
        raise ValueError("arm must be a nonempty lowercase identifier")
    if not alphas or tuple(sorted(set(alphas))) != tuple(alphas):
        raise ValueError("alphas must be unique and strictly increasing")
    if any(not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alphas must be finite and between zero and one")

    with source_path.open("rb") as stream:
        source_state = pickle.load(stream)
    with target_path.open("rb") as stream:
        target_state = pickle.load(stream)
    for state, label in ((source_state, "source"), (target_state, "target")):
        if not hasattr(state, "actor_params") or not hasattr(state, "normalizer"):
            raise ValueError(f"{label} checkpoint lacks actor or normalizer state")
    if not _tree_exact(source_state.normalizer, target_state.normalizer):
        raise ValueError("source and target normalizers must be bit-identical")

    projected_target = project_zero_scale_target_adapter(
        source_state.actor_params,
        target_state.actor_params,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for alpha in alphas:
        actor_params = interpolate_residual_actor_params(
            source_state.actor_params,
            projected_target,
            alpha=alpha,
        )
        checkpoint = _replace_actor_params(source_state, actor_params)
        filename = f"{arm}_alpha_{_alpha_token(alpha)}.pkl"
        output_path = output_dir / filename
        _atomic_pickle(output_path, checkpoint)
        records.append(
            {
                "alpha": alpha,
                "filename": filename,
                "sha256": _sha256(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        )

    manifest = {
        "protocol": "g1-residual-path-checkpoints-v1",
        "arm": arm,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "target_path": str(target_path),
        "target_sha256": _sha256(target_path),
        "parent_exact": True,
        "normalizer_exact": True,
        "zero_scale_projection_exact": True,
        "checkpoints": records,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_path_checkpoints(
        source_path=args.source,
        target_path=args.target,
        output_dir=args.output_dir,
        arm=args.arm,
        alphas=tuple(args.alphas),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
