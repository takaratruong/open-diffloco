"""Render clean and exact-distribution episodes from training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pkl$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not an archived checkpoint: {path.name}")
    return int(match.group(1))


def discover_checkpoints(
    run_directory: Path, *, checkpoint_interval: int, total_steps: int
) -> list[Path]:
    expected_steps = list(
        range(checkpoint_interval, total_steps + 1, checkpoint_interval)
    )
    checkpoints = [
        run_directory / f"checkpoint_step_{step:06d}.pkl"
        for step in expected_steps
    ]
    missing = [path.name for path in checkpoints if not path.is_file()]
    if missing:
        raise ValueError(f"missing archived checkpoints: {missing}")
    return checkpoints


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def action_diagnostics(
    *,
    action_mean: np.ndarray,
    epsilon: np.ndarray,
    action_std: np.ndarray,
    noisy_action: np.ndarray,
    effective_action: np.ndarray,
) -> dict[str, float | int]:
    arrays = {
        "action_mean": np.asarray(action_mean, dtype=np.float64),
        "epsilon": np.asarray(epsilon, dtype=np.float64),
        "noisy_action": np.asarray(noisy_action, dtype=np.float64),
        "effective_action": np.asarray(effective_action, dtype=np.float64),
    }
    shape = arrays["action_mean"].shape
    if len(shape) != 2 or shape[1] != 29:
        raise ValueError("action tape must have shape (steps, 29)")
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("action tape arrays must have identical shapes")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("action tape must be finite")
    std = np.asarray(action_std, dtype=np.float64)
    if std.ndim == 0:
        std = np.full((29,), float(std))
    if std.shape != (29,) or not np.isfinite(std).all():
        raise ValueError("action_std must be finite scalar or shape (29,)")
    injected_noise = arrays["epsilon"] * std
    return {
        "samples": int(np.prod(shape)),
        "mean_action_rms": _rms(arrays["action_mean"]),
        "noise_rms": _rms(injected_noise),
        "noisy_action_rms": _rms(arrays["noisy_action"]),
        "effective_action_rms": _rms(arrays["effective_action"]),
        "mean_action_outside_fraction": float(
            np.mean(np.abs(arrays["action_mean"]) > 1.0)
        ),
        "noisy_action_outside_fraction": float(
            np.mean(np.abs(arrays["noisy_action"]) > 1.0)
        ),
        "effective_action_saturation_fraction": float(
            np.mean(np.abs(arrays["effective_action"]) >= 1.0 - 1e-12)
        ),
        "effective_action_max_abs": float(
            np.max(np.abs(arrays["effective_action"]))
        ),
    }


def _load_summary(path: Path, *, checkpoint_sha256: str, noisy: bool) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"summary checkpoint mismatch: {path}")
    if summary.get("training_distribution_rollout") is not noisy:
        raise ValueError(f"summary rollout mode mismatch: {path}")
    if noisy and (
        summary.get("training_observation_noise") is not False
        or summary.get("training_exact_reset_phase") != 0
    ):
        raise ValueError("noisy episode is not the exact clean-observation phase-zero contract")
    return summary


def build_episode_manifest(
    *, checkpoints: list[Path], output_root: Path
) -> dict[str, Any]:
    episodes = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        checkpoint_sha = sha256_file(checkpoint)
        root = output_root / f"checkpoint_step_{step:06d}"
        noisy_dir = root / "noisy"
        clean_dir = root / "clean"
        required = (
            noisy_dir / "training_rollout.mp4",
            noisy_dir / "training_slice_h12.mp4",
            noisy_dir / "contact_sheet.png",
            noisy_dir / "training_action_noise.npz",
            clean_dir / "evaluation.mp4",
            clean_dir / "contact_sheet.png",
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise ValueError(f"episode evidence is incomplete for step {step}")
        noisy_summary = _load_summary(
            noisy_dir / "summary.json",
            checkpoint_sha256=checkpoint_sha,
            noisy=True,
        )
        clean_summary = _load_summary(
            clean_dir / "summary.json",
            checkpoint_sha256=checkpoint_sha,
            noisy=False,
        )
        with np.load(noisy_dir / "training_action_noise.npz") as archive:
            diagnostics = action_diagnostics(
                action_mean=archive["action_mean"],
                epsilon=archive["epsilon"],
                action_std=archive["action_std"],
                noisy_action=archive["noisy_action"],
                effective_action=archive["effective_action"],
            )
        episodes.append(
            {
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "noisy": noisy_summary,
                "clean": clean_summary,
                "actions": diagnostics,
                "noisy_video": str((noisy_dir / "training_rollout.mp4").resolve()),
                "clean_video": str((clean_dir / "evaluation.mp4").resolve()),
            }
        )
    return {
        "protocol": "g1-training-episode-log-v1",
        "valid": True,
        "episodes": episodes,
    }


def build_commands(
    *, checkpoint: Path, output_root: Path, evaluator: Path
) -> tuple[list[str], list[str]]:
    hparams = json.loads(
        (checkpoint.parent / "hparams.json").read_text(encoding="utf-8")
    )
    step = checkpoint_step(checkpoint)
    root = output_root / f"checkpoint_step_{step:06d}"
    common = [sys.executable, str(evaluator), "--checkpoint", str(checkpoint)]
    noisy = common + [
        "--output-dir",
        str(root / "noisy"),
        "--training-distribution-rollout",
        "--disable-training-observation-noise",
        "--exact-training-reset-phase",
        "0",
        "--seed",
        "0",
    ]
    clean = common + [
        "--output-dir",
        str(root / "clean"),
        "--env-variant",
        str(hparams["env_variant"]),
        "--phase",
        "0",
        "--reference-path",
        str(hparams["reference_path"]),
        "--reference-stride",
        str(hparams["reference_stride"]),
        "--actor-history-len",
        str(hparams["actor_history_len"]),
        "--actor-reference-preview-mode",
        str(hparams["actor_reference_preview_mode"]),
        "--reference-residual-control",
        "--reference-residual-scale",
        str(hparams["reference_residual_scale"]),
        "--solver-profile",
        str(hparams["solver_profile"]),
        "--seed",
        "0",
    ]
    lookahead = list(map(str, hparams["actor_reference_lookahead_steps"]))
    if lookahead:
        clean += ["--actor-reference-lookahead-steps", *lookahead]
    if not hparams["actor_layer_norm"]:
        clean.append("--no-actor-layer-norm")
    clean += ["--actor-hidden", *map(str, hparams["actor_hidden"])]
    return noisy, clean


def _run_commands(commands: list[list[str]], *, gpu: str, cwd: Path) -> None:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=gpu,
        JAX_ENABLE_X64="true",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONPATH=str(cwd),
    )
    for command in commands:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def render_all(
    *, checkpoints: list[Path], output_root: Path, gpus: list[str], repository: Path
) -> None:
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct device identifiers")
    evaluator = repository / "tools" / "evaluate_g1_tracking.py"
    all_commands = []
    for checkpoint in checkpoints:
        all_commands.extend(build_commands(
            checkpoint=checkpoint, output_root=output_root, evaluator=evaluator
        ))
    buckets = [all_commands[index::len(gpus)] for index in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_run_commands, bucket, gpu=gpu, cwd=repository)
            for bucket, gpu in zip(buckets, gpus, strict=True)
            if bucket
        ]
        for future in futures:
            future.result()


def build_progression_media(manifest: dict[str, Any], output_root: Path) -> dict:
    progression = output_root / "progression"
    progression.mkdir(parents=True, exist_ok=True)
    video_path = progression / "training_progression.mp4"
    sheet_path = progression / "training_progression.png"
    representative_frames = []
    with imageio.get_writer(video_path, fps=25, quality=8) as writer:
        for episode in manifest["episodes"]:
            reader = imageio.get_reader(episode["noisy_video"])
            first = None
            try:
                for raw in reader:
                    image = Image.fromarray(raw)
                    draw = ImageDraw.Draw(image)
                    label = (
                        f"step {episode['step']} | noisy training replay | "
                        f"survival {episode['noisy']['steps']} | "
                        f"sat {episode['actions']['effective_action_saturation_fraction']:.3f}"
                    )
                    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
                    draw.text((8, 7), label, fill=(255, 255, 255))
                    frame = np.asarray(image)
                    if first is None:
                        first = frame.copy()
                    writer.append_data(frame)
            finally:
                reader.close()
            if first is not None:
                representative_frames.append(first)
    if not representative_frames:
        raise ValueError("progression contains no video frames")
    thumb_width = 640
    thumbs = []
    for frame in representative_frames:
        image = Image.fromarray(frame)
        height = round(image.height * thumb_width / image.width)
        thumbs.append(image.resize((thumb_width, height)))
    columns = 2
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumbs[0].height))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * thumb_width, (index // columns) * image.height))
    sheet.save(sheet_path)
    return {
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": sha256_file(sheet_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    checkpoints = discover_checkpoints(
        args.run_directory.resolve(),
        checkpoint_interval=args.checkpoint_interval,
        total_steps=args.total_steps,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    render_all(
        checkpoints=checkpoints,
        output_root=output_root,
        gpus=args.gpus,
        repository=repository,
    )
    manifest = build_episode_manifest(
        checkpoints=checkpoints, output_root=output_root
    )
    manifest["progression"] = build_progression_media(manifest, output_root)
    path = output_root / "training_episode_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
