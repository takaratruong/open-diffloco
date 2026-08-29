"""Replay the E005 full-wrench positive control across source and XLA routes.

This is an evaluation-only provenance discriminator.  It runs the immutable
legacy E005 evaluator and the current evaluator, each with default and
deterministic XLA execution, twice on one inherited physical GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
)
from tools.prepare_g1_rmr_reference import sha256_file


MATRIX_CONDITIONS = (
    ("legacy-default-a", "legacy", "default"),
    ("legacy-default-b", "legacy", "default"),
    ("legacy-deterministic-a", "legacy", "deterministic"),
    ("legacy-deterministic-b", "legacy", "deterministic"),
    ("current-default-a", "current", "default"),
    ("current-default-b", "current", "default"),
    ("current-deterministic-a", "current", "deterministic"),
    ("current-deterministic-b", "current", "deterministic"),
)

DETERMINISTIC_XLA_FLAGS = "--xla_gpu_exclude_nondeterministic_ops"
LEGACY_COMMIT = "306301e323020b22a01b2ed1b0f45b907517a826"
LEGACY_EVALUATOR_SHA256 = (
    "ea034847373a7a62835076bb99a3260f0756a5b28b623eb5052ca9105dbbe81f"
)
TRAJECTORY_ARRAYS = (
    "values",
    "action_mean",
    "sampled_action",
    "effective_action",
    "joint_position",
    "joint_velocity",
    "qpos",
    "qvel",
    "reference_joint_position",
    "position_target",
    "learned_torso_wrench",
    "learned_torso_wrench_normalized",
)


def _hash_arrays(path: Path, names: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        selected = tuple(sorted(archive.files)) if names is None else names
        missing = [name for name in selected if name not in archive.files]
        if missing:
            raise ValueError(f"archive is missing arrays: {missing}")
        for name in selected:
            values = np.asarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(values.dtype.str.encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def npz_content_sha256(path: Path) -> str:
    """Hash every sorted array, excluding timestamped ZIP metadata."""
    return _hash_arrays(path)


def trajectory_content_sha256(path: Path) -> str:
    """Hash dynamic arrays shared by the legacy and current evaluators."""
    return _hash_arrays(path, TRAJECTORY_ARRAYS)


def build_evaluator_command(
    *,
    python: Path,
    evaluator: Path,
    checkpoint: Path,
    reference: Path,
    output_dir: Path,
    source: str,
    phase: int,
    solver_profile: str,
) -> list[str]:
    """Build the hparam-exact E005 evaluator command for one source route."""
    if source not in {"legacy", "current"}:
        raise ValueError(f"unknown source route: {source}")
    profile = get_solver_profile(solver_profile)
    command = [
        str(python),
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--reference-path",
        str(reference),
        "--output-dir",
        str(output_dir),
        "--seed",
        "0",
        "--phase",
        str(phase),
        "--render-every",
        "2",
        "--env-variant",
        "g1_tracking_rmr_50hz_action_parity",
        "--reference-stride",
        "1",
        "--actor-history-len",
        "10",
        "--actor-reference-preview-mode",
        "delta",
        "--reference-residual-control",
        "--reference-residual-scale",
        "1.0",
        "--solver-iterations",
        str(profile.iterations),
        "--solver-ls-iterations",
        str(profile.ls_iterations),
        "--solver-profile",
        solver_profile,
    ]
    if source == "current":
        command.extend(("--learned-wrench-components", "full"))
    command.extend(("--actor-reference-lookahead-steps", "4", "8", "12"))
    return command


def child_environment(
    ambient: Mapping[str, str], *, execution: str
) -> dict[str, str]:
    """Return an isolated child environment for one registered XLA route."""
    if execution not in {"default", "deterministic"}:
        raise ValueError(f"unknown execution route: {execution}")
    environment = dict(ambient)
    environment.pop("XLA_FLAGS", None)
    if execution == "deterministic":
        environment["XLA_FLAGS"] = DETERMINISTIC_XLA_FLAGS
    return environment


def classify_matrix(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify repeatable positive controls and coarse route effects."""
    expected = {
        f"{source}-{execution}"
        for source in ("legacy", "current")
        for execution in ("default", "deterministic")
    }
    if set(rows) != expected:
        raise ValueError("matrix rows do not match the registered 2x2 routes")

    eligible_routes = [
        route
        for route in (
            "legacy-default",
            "legacy-deterministic",
            "current-default",
            "current-deterministic",
        )
        if rows[route]["steps"] == [271, 271]
        and rows[route]["content_exact"] is True
    ]
    attribution: list[str] = []
    for source in ("legacy", "current"):
        if (
            rows[f"{source}-default"]["steps"]
            != rows[f"{source}-deterministic"]["steps"]
        ):
            attribution.append(f"xla-effect-{source}")
    for execution in ("default", "deterministic"):
        if (
            rows[f"legacy-{execution}"]["steps"]
            != rows[f"current-{execution}"]["steps"]
        ):
            attribution.append(f"source-effect-{execution}")
    for route in (
        "legacy-default",
        "legacy-deterministic",
        "current-default",
        "current-deterministic",
    ):
        if rows[route]["content_exact"] is not True:
            attribution.append(f"duplicate-divergence-{route}")

    return {
        "outcome": (
            "repeatable-positive-control-found"
            if eligible_routes
            else "no-repeatable-positive-control"
        ),
        "eligible_routes": eligible_routes,
        "attribution": attribution,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_repository(
    *, repository: Path, expected_commit: str, expected_evaluator_sha256: str
) -> Path:
    if not (repository / ".git").exists():
        # Linked worktrees expose .git as a file, so this is only a clearer
        # failure for paths that are not repositories at all.
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
    actual_commit = _git_output(repository, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"repository HEAD mismatch: {repository}: {actual_commit}"
        )
    dirty = _git_output(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"repository is dirty: {repository}:\n{dirty}")
    evaluator = repository / "tools" / "evaluate_g1_tracking.py"
    if sha256_file(evaluator) != expected_evaluator_sha256:
        raise ValueError(f"evaluator SHA-256 mismatch: {evaluator}")
    return evaluator


def _validate_condition(
    *,
    summary: dict[str, Any],
    archive_path: Path,
    source: str,
    checkpoint_sha256: str,
    reference_sha256: str,
    phase: int,
    solver_profile: str,
    expected_steps: int,
) -> None:
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("condition checkpoint SHA-256 does not match")
    if summary.get("reference_sha256") != reference_sha256:
        raise ValueError("condition reference SHA-256 does not match")
    if summary.get("evaluation_start_phase") != phase:
        raise ValueError("condition phase does not match")
    if summary.get("solver_profile") != solver_profile:
        raise ValueError("condition solver profile does not match")
    if summary.get("reference_stride") != 1:
        raise ValueError("condition reference stride does not match")
    if summary.get("actor_reference_lookahead_steps") != [4, 8, 12]:
        raise ValueError("condition lookahead does not match")
    if summary.get("actor_learned_torso_wrench") is not True:
        raise ValueError("condition did not load a learned-wrench checkpoint")
    if summary.get("remaining_reference_transitions") != expected_steps:
        raise ValueError("condition reference suffix length does not match")
    steps = summary.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("condition step count is not an integer")
    if not 1 <= steps <= expected_steps:
        raise ValueError("condition survival is outside the reference suffix")
    if bool(summary.get("completed_reference_suffix")) != (steps == expected_steps):
        raise ValueError("condition completion label conflicts with survival")

    if source == "current":
        exact_current_fields = {
            "environment_variant": "g1_tracking_rmr_50hz_action_parity",
            "actor_history_len": 10,
            "actor_reference_preview_mode": "delta",
            "reference_residual_control": True,
            "reference_residual_scale": 1.0,
            "learned_torso_wrench_components": "full",
            "learned_torso_wrench_component_mask": [1, 1, 1, 1, 1, 1],
        }
        for name, expected in exact_current_fields.items():
            if summary.get(name) != expected:
                raise ValueError(f"current condition field does not match: {name}")

    with np.load(archive_path, allow_pickle=False) as archive:
        if archive["values"].shape[0] != steps:
            raise ValueError("condition NPZ row count does not match summary")
        for name in TRAJECTORY_ARRAYS:
            values = np.asarray(archive[name])
            if values.shape[0] != steps:
                raise ValueError(f"condition NPZ rows do not match: {name}")
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"condition NPZ array is not numeric: {name}")
            if not np.isfinite(values).all():
                raise ValueError(f"condition NPZ array is non-finite: {name}")


def _route_rows(condition_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in condition_rows:
        grouped.setdefault(f'{row["source"]}-{row["execution"]}', []).append(row)
    routes: dict[str, dict[str, Any]] = {}
    for route, repetitions in grouped.items():
        if len(repetitions) != 2:
            raise ValueError(f"route does not have two repetitions: {route}")
        routes[route] = {
            "conditions": [row["condition"] for row in repetitions],
            "steps": [row["steps"] for row in repetitions],
            "completed": [row["completed"] for row in repetitions],
            "content_exact": (
                repetitions[0]["evaluation_content_sha256"]
                == repetitions[1]["evaluation_content_sha256"]
            ),
            "trajectory_exact": (
                repetitions[0]["trajectory_content_sha256"]
                == repetitions[1]["trajectory_content_sha256"]
            ),
        }
    return routes


def _write_plot(rows: list[dict[str, Any]], path: Path, expected_steps: int) -> None:
    labels = [row["condition"] for row in rows]
    steps = [row["steps"] for row in rows]
    colors = ["#2f855a" if row["completed"] else "#c53030" for row in rows]
    figure, axis = plt.subplots(figsize=(13, 5.8))
    positions = np.arange(len(rows))
    axis.bar(positions, steps, color=colors)
    axis.axhline(expected_steps, color="#2d3748", linestyle="--", linewidth=1)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("survived reference transitions")
    axis.set_title("E005 full-wrench replay: source x XLA execution")
    axis.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, steps, strict=True):
        axis.text(position, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repository", type=Path, required=True)
    parser.add_argument("--current-repository", type=Path, required=True)
    parser.add_argument("--legacy-commit", default=LEGACY_COMMIT)
    parser.add_argument(
        "--legacy-evaluator-sha256", default=LEGACY_EVALUATOR_SHA256
    )
    parser.add_argument("--current-commit")
    parser.add_argument("--current-evaluator-sha256")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--historical-evaluation", type=Path, required=True)
    parser.add_argument("--historical-evaluation-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--expected-completion-steps", type=int, default=271)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default="g1-4x5",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    legacy_repository = args.legacy_repository.resolve()
    current_repository = args.current_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    historical_evaluation = args.historical_evaluation.resolve()
    output_root = args.output_root.resolve()
    python = args.python.resolve()

    current_commit = args.current_commit or _git_output(
        current_repository, "rev-parse", "HEAD"
    )
    current_evaluator_path = (
        current_repository / "tools" / "evaluate_g1_tracking.py"
    )
    current_evaluator_sha256 = args.current_evaluator_sha256 or sha256_file(
        current_evaluator_path
    )
    evaluators = {
        "legacy": _validate_repository(
            repository=legacy_repository,
            expected_commit=args.legacy_commit,
            expected_evaluator_sha256=args.legacy_evaluator_sha256,
        ),
        "current": _validate_repository(
            repository=current_repository,
            expected_commit=current_commit,
            expected_evaluator_sha256=current_evaluator_sha256,
        ),
    }
    repositories = {
        "legacy": legacy_repository,
        "current": current_repository,
    }
    expected_hashes = (
        (checkpoint, args.checkpoint_sha256, "checkpoint"),
        (reference, args.reference_sha256, "reference"),
        (
            historical_evaluation,
            args.historical_evaluation_sha256,
            "historical evaluation",
        ),
    )
    for path, expected_sha256, label in expected_hashes:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} SHA-256 does not match")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)

    with np.load(historical_evaluation, allow_pickle=False) as archive:
        historical_steps = int(archive["values"].shape[0])
    if historical_steps != args.expected_completion_steps:
        raise ValueError("historical positive control does not have 271 steps")
    historical_trajectory_sha256 = trajectory_content_sha256(
        historical_evaluation
    )

    condition_rows: list[dict[str, Any]] = []
    for index, (condition, source, execution) in enumerate(
        MATRIX_CONDITIONS, start=1
    ):
        print(
            f"[{index}/{len(MATRIX_CONDITIONS)}] starting {condition}",
            flush=True,
        )
        condition_dir = output_root / condition
        command = build_evaluator_command(
            python=python,
            evaluator=evaluators[source],
            checkpoint=checkpoint,
            reference=reference,
            output_dir=condition_dir,
            source=source,
            phase=args.phase,
            solver_profile=args.solver_profile,
        )
        environment = child_environment(os.environ, execution=execution)
        environment["PYTHONPATH"] = str(repositories[source])
        with (
            (output_root / f"{condition}.stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout,
            (output_root / f"{condition}.stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr,
        ):
            subprocess.run(
                command,
                cwd=repositories[source],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        summary_path = condition_dir / "summary.json"
        archive_path = condition_dir / "evaluation.npz"
        video_path = condition_dir / "evaluation.mp4"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_condition(
            summary=summary,
            archive_path=archive_path,
            source=source,
            checkpoint_sha256=args.checkpoint_sha256,
            reference_sha256=args.reference_sha256,
            phase=args.phase,
            solver_profile=args.solver_profile,
            expected_steps=args.expected_completion_steps,
        )
        trajectory_sha256 = trajectory_content_sha256(archive_path)
        row = {
            "condition": condition,
            "source": source,
            "execution": execution,
            "xla_flags": (
                DETERMINISTIC_XLA_FLAGS
                if execution == "deterministic"
                else None
            ),
            "steps": int(summary["steps"]),
            "completed": bool(summary["completed_reference_suffix"]),
            "summary": str(summary_path.relative_to(output_root)),
            "summary_sha256": sha256_file(summary_path),
            "evaluation": str(archive_path.relative_to(output_root)),
            "evaluation_sha256": sha256_file(archive_path),
            "evaluation_content_sha256": npz_content_sha256(archive_path),
            "trajectory_content_sha256": trajectory_sha256,
            "historical_trajectory_exact": (
                trajectory_sha256 == historical_trajectory_sha256
            ),
            "video": str(video_path.relative_to(output_root)),
            "video_sha256": sha256_file(video_path),
            "command": command,
        }
        condition_rows.append(row)
        partial_payload = {
            "protocol": "g1-learned-wrench-replay-matrix-v1-partial",
            "historical_steps": historical_steps,
            "rows": condition_rows,
        }
        _write_json(output_root / "replay_matrix.partial.json", partial_payload)
        print(
            f"[{index}/{len(MATRIX_CONDITIONS)}] finished {condition}: "
            f'{row["steps"]} steps',
            flush=True,
        )

    routes = _route_rows(condition_rows)
    classification = classify_matrix(routes)
    payload = {
        "protocol": "g1-learned-wrench-replay-matrix-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "reference": str(reference),
        "reference_sha256": args.reference_sha256,
        "historical_evaluation": str(historical_evaluation),
        "historical_evaluation_sha256": args.historical_evaluation_sha256,
        "historical_trajectory_content_sha256": historical_trajectory_sha256,
        "historical_steps": historical_steps,
        "phase": args.phase,
        "solver_profile": args.solver_profile,
        "expected_completion_steps": args.expected_completion_steps,
        "repositories": {
            "legacy": {
                "path": str(legacy_repository),
                "commit": args.legacy_commit,
                "evaluator_sha256": args.legacy_evaluator_sha256,
            },
            "current": {
                "path": str(current_repository),
                "commit": current_commit,
                "evaluator_sha256": current_evaluator_sha256,
            },
        },
        **classification,
        "valid": bool(classification["eligible_routes"]),
        "routes": routes,
        "rows": condition_rows,
    }
    _write_json(output_root / "replay_matrix.json", payload)
    _write_plot(
        condition_rows,
        output_root / "replay_matrix.png",
        args.expected_completion_steps,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
