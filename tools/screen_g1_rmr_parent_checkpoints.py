"""Screen untouched RMR/PPO checkpoints under the exact five-phase MJX gate."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import SOLVER_PROFILES
from tools.evaluate_g1_rmr_phase_grid import DEFAULT_PHASES


def _checkpoint_step(path: str) -> int:
    match = re.search(r"model_(\d+)\.pt$", path)
    if match is None:
        raise ValueError(f"checkpoint name must end in model_<step>.pt: {path}")
    return int(match.group(1))


def select_checkpoint(
    payloads: list[dict], *, phases: tuple[int, ...]
) -> dict[str, object]:
    """Rank parents, but authorize only an actor completing every suffix."""
    if not payloads:
        raise ValueError("at least one checkpoint evaluation is required")
    ranking = []
    for payload in payloads:
        summary = payload["source"]["summary"]
        if tuple(summary["phases"]) != phases:
            raise ValueError("evaluation phase grid does not match the screen")
        reference_transitions = int(payload["reference_transitions"])
        remaining = [reference_transitions - phase for phase in phases]
        survival = [int(value) for value in summary["survival"]]
        fractions = [
            value / suffix
            for value, suffix in zip(survival, remaining, strict=True)
        ]
        completed = [bool(value) for value in summary["completed_suffix"]]
        checkpoint_path = str(Path(payload["source_policy_path"]).resolve())
        ranking.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": payload["source_policy_sha256"],
                "checkpoint_step": _checkpoint_step(checkpoint_path),
                "survival": survival,
                "survival_fraction": fractions,
                "completed_suffix": completed,
                "completed_suffix_count": sum(completed),
                "minimum_survival_fraction": min(fractions),
                "median_survival_fraction": float(median(fractions)),
                "mean_survival_fraction": float(mean(fractions)),
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["completed_suffix_count"],
            -row["minimum_survival_fraction"],
            -row["median_survival_fraction"],
            -row["mean_survival_fraction"],
            row["checkpoint_step"],
        )
    )
    best = ranking[0]
    eligible = best["completed_suffix_count"] == len(phases)
    return {
        "eligible": eligible,
        "selected_checkpoint_path": best["checkpoint_path"] if eligible else None,
        "selected_checkpoint_sha256": (
            best["checkpoint_sha256"] if eligible else None
        ),
        "best_diagnostic_checkpoint_path": best["checkpoint_path"],
        "best_diagnostic_checkpoint_sha256": best["checkpoint_sha256"],
        "ranking": ranking,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_screen(
    *,
    checkpoints: list[Path],
    reference_path: Path,
    output_root: Path,
    summary_output: Path,
    phases: tuple[int, ...],
    seed: int,
    solver_profile: str,
    gpu_ids: tuple[str, ...],
) -> dict[str, object]:
    """Evaluate checkpoints concurrently, with exclusive GPUs per child."""
    if not checkpoints or not gpu_ids:
        raise ValueError("checkpoints and GPU IDs must be nonempty")
    checkpoints = [path.resolve() for path in checkpoints]
    reference_path = reference_path.resolve()
    for path in [*checkpoints, reference_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint list contains duplicates")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique")
    output_root.mkdir(parents=True, exist_ok=True)
    available_gpus: queue.SimpleQueue[str] = queue.SimpleQueue()
    for gpu_id in gpu_ids:
        available_gpus.put(gpu_id)
    evaluator = Path(__file__).with_name("evaluate_g1_rmr_phase_grid.py").resolve()

    def evaluate(checkpoint: Path) -> dict[str, object]:
        gpu_id = available_gpus.get()
        output = output_root / f"{checkpoint.stem}.json"
        log = output_root / f"{checkpoint.stem}.log"
        command = [
            sys.executable,
            str(evaluator),
            "--source-policy-checkpoint",
            str(checkpoint),
            "--reference-path",
            str(reference_path),
            "--output",
            str(output),
            "--phases",
            *(str(phase) for phase in phases),
            "--seed",
            str(seed),
            "--solver-profile",
            solver_profile,
        ]
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=gpu_id,
            JAX_ENABLE_X64="true",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            PYTHONPATH=".",
        )
        try:
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{checkpoint.name} failed on GPU {gpu_id} with return code "
                    f"{completed.returncode}; see {log}"
                )
            if not output.is_file():
                raise RuntimeError(f"evaluator did not produce {output}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            if Path(payload["source_policy_path"]).resolve() != checkpoint:
                raise RuntimeError("evaluator checkpoint provenance mismatch")
            return {
                "checkpoint_path": str(checkpoint),
                "gpu_id": gpu_id,
                "output": str(output.resolve()),
                "log": str(log.resolve()),
                "payload": payload,
            }
        finally:
            available_gpus.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        evaluations = list(executor.map(evaluate, checkpoints))
    payloads = [record["payload"] for record in evaluations]
    result = {
        "protocol": "g1-rmr-walk-parent-checkpoint-screen-v1",
        "phases": list(phases),
        "reference_path": str(reference_path),
        "seed": seed,
        "solver_profile": solver_profile,
        "evaluations": evaluations,
        "selection": select_checkpoint(payloads, phases=phases),
    }
    _write_json(summary_output.resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--reference-path", type=Path, default=Path(DEFAULT_REFERENCE_PATH)
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--phases", type=int, nargs=5, default=DEFAULT_PHASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
        default="g1-4x5",
    )
    parser.add_argument("--gpu-ids", nargs="+", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screen(
        checkpoints=args.checkpoints,
        reference_path=args.reference_path,
        output_root=args.output_root,
        summary_output=args.summary_output,
        phases=tuple(args.phases),
        seed=args.seed,
        solver_profile=args.solver_profile,
        gpu_ids=tuple(args.gpu_ids),
    )
    print(json.dumps(result["selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
