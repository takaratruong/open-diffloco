"""Run paired true-scalar and blind torso-assistance continuations."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import traceback


E012_SELECTED_STEP = 1_671_168
STEPS_PER_UPDATE = 6_144
ASSISTED_UPDATES = 96
ZERO_TAIL_UPDATES = 32
ASSISTANCE_END_STEP = E012_SELECTED_STEP + ASSISTED_UPDATES * STEPS_PER_UPDATE
CONTINUATION_END_STEP = ASSISTANCE_END_STEP + ZERO_TAIL_UPDATES * STEPS_PER_UPDATE
CHECKPOINT_INTERVAL = 8 * STEPS_PER_UPDATE
ZERO_ASSISTANCE_FRACTION = 0.25
RESUME_RANDOM_SEED = 2
E012_SELECTED_CHECKPOINT_SHA256 = (
    "f375cadc9bf8b5cef26fc7414133071910fed393344c99bbacffea963aa9f4f7"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_checkpoint(path: str | Path) -> Path:
    """Require the exact selected E012 checkpoint before launching compute."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"E012 selected checkpoint is missing: {checkpoint}")
    actual = _sha256(checkpoint)
    if actual != E012_SELECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "E012 selected checkpoint SHA-256 mismatch: "
            f"expected {E012_SELECTED_CHECKPOINT_SHA256}, got {actual}"
        )
    return checkpoint


def build_assistance_observability_pair_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> tuple[dict, dict]:
    """Return paired contracts differing only in scalar observability."""
    from tools.run_g1_frozen_residual_preview_continuation import (
        build_frozen_residual_preview_kwargs,
    )

    common = build_frozen_residual_preview_kwargs(
        profile_name,
        Path(reference_path),
        seed,
        Path(resume_from),
    )
    common.update(
        total_steps=CONTINUATION_END_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        resume_random_seed=RESUME_RANDOM_SEED,
        torso_wrench_assistance=True,
        torso_wrench_assistance_start_step=E012_SELECTED_STEP,
        torso_wrench_assistance_end_step=ASSISTANCE_END_STEP,
        torso_wrench_assistance_zero_fraction=ZERO_ASSISTANCE_FRACTION,
        torso_wrench_assistance_continuous=True,
        actor_torso_wrench_assistance_conditioning=True,
        allow_resume_torso_wrench_assistance_change=True,
        allow_resume_assistance_conditioning_change=True,
    )
    aware = {**common, "actor_observe_torso_wrench_assistance": True}
    blind = {**common, "actor_observe_torso_wrench_assistance": False}
    return aware, blind


def validate_pair_delta(aware: dict, blind: dict) -> None:
    """Fail closed unless observability is the sole paired difference."""
    if set(aware) != set(blind):
        raise ValueError("paired assistance contracts have different keys")
    changed = {key for key in aware if aware[key] != blind[key]}
    if changed != {"actor_observe_torso_wrench_assistance"}:
        raise ValueError(f"invalid paired assistance delta: {sorted(changed)}")
    if aware["actor_observe_torso_wrench_assistance"] is not True:
        raise ValueError("aware arm must observe the assistance scalar")
    if blind["actor_observe_torso_wrench_assistance"] is not False:
        raise ValueError("blind arm must receive exact-zero conditioning")


def _run_arm(
    *,
    arm: str,
    observed: bool,
    device: str,
    profile_name: str,
    reference_path: str,
    seed: int,
    resume_from: str,
    output_root: str,
    result_queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        from src.algorithms.shac.algorithm import train
        from src.envs.g1_tracking.solver_profiles import (
            get_solver_profile,
            solver_context,
        )
        from tools.run_g1_tracking_shac import configure_jax

        configure_jax()
        aware, blind = build_assistance_observability_pair_kwargs(
            profile_name, reference_path, seed, resume_from
        )
        kwargs = aware if observed else blind
        arm_root = (Path(output_root) / arm).resolve()
        arm_root.mkdir(parents=True, exist_ok=False)
        profile = get_solver_profile(profile_name)
        previous_directory = Path.cwd()
        try:
            os.chdir(arm_root)
            with solver_context(profile):
                _, relative_save_dir = train(**kwargs)
        finally:
            os.chdir(previous_directory)
        result_queue.put(
            (arm, True, str((arm_root / relative_save_dir).resolve()))
        )
    except BaseException:
        result_queue.put((arm, False, traceback.format_exc()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", required=True)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("g1_assistance_observability_pair_runs"),
    )
    parser.add_argument("--aware-device", default="0")
    parser.add_argument("--blind-device", default="1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.aware_device == args.blind_device:
        raise ValueError("paired arms require distinct GPU devices")
    checkpoint = validate_parent_checkpoint(args.resume_from)
    if args.reference_path is None:
        from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH

        reference_path = Path(DEFAULT_REFERENCE_PATH).resolve()
    else:
        reference_path = args.reference_path.resolve()
    if not reference_path.is_file():
        raise ValueError(f"reference path is missing: {reference_path}")
    aware, blind = build_assistance_observability_pair_kwargs(
        args.solver_profile,
        reference_path,
        args.seed,
        checkpoint,
    )
    validate_pair_delta(aware, blind)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    common = {
        "profile_name": args.solver_profile,
        "reference_path": str(reference_path),
        "seed": args.seed,
        "resume_from": str(checkpoint),
        "output_root": str(output_root),
        "result_queue": result_queue,
    }
    processes = [
        context.Process(
            target=_run_arm,
            kwargs={
                **common,
                "arm": "aware",
                "observed": True,
                "device": args.aware_device,
            },
        ),
        context.Process(
            target=_run_arm,
            kwargs={
                **common,
                "arm": "blind",
                "observed": False,
                "device": args.blind_device,
            },
        ),
    ]
    for process in processes:
        process.start()
    results = [result_queue.get() for _ in processes]
    for process in processes:
        process.join()
    failures = [result for result in results if not result[1]]
    if failures:
        details = "\n".join(f"[{arm}]\n{detail}" for arm, _, detail in failures)
        raise RuntimeError(f"paired assistance run failed:\n{details}")
    for arm, _, path in sorted(results):
        print(f"{arm}: {path}")


if __name__ == "__main__":
    main()
