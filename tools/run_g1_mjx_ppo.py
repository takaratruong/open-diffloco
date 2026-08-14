"""Run the exact-environment G1 MJX PPO positive control."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

import jax

from src.algorithms.ppo.algorithm import train
from src.envs.g1_tracking.environment import (
    DEFAULT_CONTROLLER_PATH,
    DEFAULT_MODEL_PATH,
    G1TrackingRMR50HzActionParityEnv,
)
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_g1_tracking_shac import configure_jax


def validate_asset(path: Path, expected_sha256: str) -> Path:
    """Requires one exact regular file and SHA-256 digest."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if len(expected_sha256) != 64:
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
    with resolved.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {resolved}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return resolved


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_clean_commit(repo: Path, expected_commit: str) -> str:
    """Requires the exact clean tracked checkout registered for execution."""

    repo = Path(repo).resolve()
    observed = _git(repo, "rev-parse", "HEAD")
    if observed != expected_commit:
        raise ValueError(
            f"registered code commit mismatch: expected {expected_commit}, "
            f"observed {observed}"
        )
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("registered code checkout is dirty")
    return observed


def build_environment_kwargs(
    *, profile_name: str, reference_path: Path
) -> dict:
    """Returns the unchanged nominal MJX task used by the positive control."""

    profile = get_solver_profile(profile_name)
    return {
        "reference_path": str(Path(reference_path).resolve()),
        "reference_stride": 1,
        "actor_history_len": 1,
        "actor_observation_noise": False,
        "domain_randomization": False,
        "friction_range": (1.0, 1.0),
        "mass_range": (1.0, 1.0),
        "kp_range": (35.0, 35.0),
        "kd_range": (0.5, 0.5),
        "com_offset_range": (0.0, 0.0, 0.0),
        "reference_reset_noise_scale": 0.0,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
    }


def build_training_kwargs(*, seed: int, output_dir: Path) -> dict:
    """Returns the bounded 32-iteration PPO learning discriminator."""

    return {
        "output_dir": Path(output_dir).resolve(),
        "total_iterations": 32,
        "num_envs": 4096,
        "horizon": 24,
        "seed": seed,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "initial_action_std": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.0,
        "num_epochs": 4,
        "num_minibatches": 8,
        "max_grad_norm": 1.0,
        "checkpoint_interval_iterations": 8,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument(
        "--controller-path", type=Path, default=Path(DEFAULT_CONTROLLER_PATH)
    )
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--solver-profile", choices=tuple(sorted(SOLVER_PROFILES)), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    validate_clean_commit(repo, args.code_commit)
    reference = validate_asset(args.reference_path, args.reference_sha256)
    model = validate_asset(args.model_path, args.model_sha256)
    controller = validate_asset(args.controller_path, args.controller_sha256)
    configure_jax()
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError(
            "registered PPO run requires exactly one visible JAX GPU; got "
            f"backend={jax.default_backend()} devices={jax.devices()}"
        )

    environment_kwargs = build_environment_kwargs(
        profile_name=args.solver_profile,
        reference_path=reference,
    )
    environment_kwargs.update(
        xml_path=str(model),
        controller_path=str(controller),
    )
    training_kwargs = build_training_kwargs(
        seed=args.seed,
        output_dir=args.output_dir,
    )
    provenance = {
        "code_commit": args.code_commit,
        "dirty_patch": None,
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "solver_profile": args.solver_profile,
        "reference_path": str(reference),
        "reference_sha256": args.reference_sha256.lower(),
        "model_path": str(model),
        "model_sha256": args.model_sha256.lower(),
        "controller_path": str(controller),
        "controller_sha256": args.controller_sha256.lower(),
        "environment_kwargs": environment_kwargs,
    }
    profile = get_solver_profile(args.solver_profile)
    with solver_context(profile):
        env = G1TrackingRMR50HzActionParityEnv(**environment_kwargs)
        train(env=env, hparams=provenance, **training_kwargs)


if __name__ == "__main__":
    main()
