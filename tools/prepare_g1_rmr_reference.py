"""Validate and annotate a clean RMR CSV-to-NPZ motion archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REQUIRED_ARRAYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
METADATA_ARRAYS = ("joint_names", "root_body_name", "root_body_index")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Returns a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_arrays(
    arrays: Mapping[str, np.ndarray], joint_names: tuple[str, ...]
) -> tuple[int, int]:
    missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
    if missing:
        raise ValueError(f"RMR reference missing arrays: {missing}")
    conflicts = [key for key in METADATA_ARRAYS if key in arrays]
    if conflicts:
        raise ValueError(f"RMR reference already contains metadata: {conflicts}")

    fps_values = np.asarray(arrays["fps"]).reshape(-1)
    if fps_values.size != 1 or not np.isfinite(fps_values[0]):
        raise ValueError("fps must contain one finite value")
    fps = int(fps_values[0])
    if fps != 50 or float(fps_values[0]) != float(fps):
        raise ValueError("RMR reference fps must equal 50")

    joint_pos = arrays["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29 or joint_pos.shape[0] <= 0:
        raise ValueError("joint_pos must have shape (T, 29) with T > 0")
    frames = int(joint_pos.shape[0])
    if arrays["joint_vel"].shape != (frames, 29):
        raise ValueError("joint_vel must have shape (T, 29)")

    body_count: int | None = None
    trailing_sizes = {
        "body_pos_w": 3,
        "body_quat_w": 4,
        "body_lin_vel_w": 3,
        "body_ang_vel_w": 3,
    }
    for key, trailing_size in trailing_sizes.items():
        array = arrays[key]
        if array.ndim != 3 or array.shape[0] != frames or array.shape[2] != trailing_size:
            raise ValueError(f"{key} must have shape (T, B, {trailing_size})")
        if body_count is None:
            body_count = int(array.shape[1])
        elif array.shape[1] != body_count:
            raise ValueError(f"{key} rigid-body count does not match body_pos_w")
    if body_count is None or body_count <= 0:
        raise ValueError("RMR reference must contain at least one rigid body")

    if len(joint_names) != 29:
        raise ValueError("joint_names must contain exactly 29 names")
    if len(set(joint_names)) != 29 or any(not name.strip() for name in joint_names):
        raise ValueError("joint_names must contain unique nonempty names")

    for key, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{key} must be numeric")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key} must contain only finite values")
    return frames, fps


def prepare_reference(
    input_path: Path,
    output_path: Path,
    *,
    joint_names: Sequence[str],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Copies RMR arrays exactly and adds explicit ordering/provenance metadata."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists():
        raise FileExistsError(output_path)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    normalized_names = tuple(str(name) for name in joint_names)

    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    frames, fps = _validate_source_arrays(arrays, normalized_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_output.open("wb") as stream:
            np.savez(
                stream,
                **arrays,
                joint_names=np.asarray(normalized_names),
                root_body_name=np.asarray("pelvis"),
                root_body_index=np.asarray(0, dtype=np.int32),
            )
        temporary_output.replace(output_path)

        manifest: dict[str, Any] = {
            "format": "rmr_named_reference_v1",
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "frames": frames,
            "fps": fps,
            "joint_names": list(normalized_names),
            "root_body_name": "pelvis",
            "root_body_index": 0,
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
            "source": dict(source_metadata),
        }
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary_manifest.write_text(serialized)
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        manifest_path.with_name(manifest_path.name + ".tmp").unlink(missing_ok=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--controller-path",
        type=Path,
        required=True,
        help="RMR controller archive whose joint_names match the logger order.",
    )
    parser.add_argument(
        "--source-metadata-json",
        type=Path,
        required=True,
        help="JSON file containing pinned dataset/converter provenance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with np.load(args.controller_path, allow_pickle=False) as controller_archive:
        joint_names = tuple(map(str, controller_archive["joint_names"]))
    source_metadata = json.loads(args.source_metadata_json.read_text())
    manifest = prepare_reference(
        args.input_path,
        args.output_path,
        joint_names=joint_names,
        source_metadata=source_metadata,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
