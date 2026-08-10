"""Run the immutable Open-DiffLoco-style G1 SHAC experiment."""

import argparse
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REFERENCE_PATH,
)
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_g1_tracking_shac import configure_jax


def build_canonical_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict:
    """Return the complete scientific contract; callers cannot override it."""
    profile = get_solver_profile(profile_name)
    return {
        "total_steps": 8_000_000,
        "unroll_length": 12,
        "num_envs": 256,
        "gradient_accumulation_steps": 1,
        "actor_lr": 5e-3,
        "critic_lr": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "target_update_rate": 0.01,
        "critic_iterations": 16,
        "xml_path": DEFAULT_MODEL_PATH,
        "action_scale": 0.5,
        "action_noise_std_start": 0.5,
        "action_noise_std_end": 0.32,
        "friction_range": (0.5, 2.0),
        "mass_range": (0.85, 1.15),
        "kp_range": (25.0, 45.0),
        "kd_range": (0.3, 0.7),
        "com_offset_range": (0.05, 0.05, 0.04),
        "push_velocity_range": (-1.0, 1.0),
        "push_interval_s": 4.0,
        "terrain": False,
        "domain_randomization": True,
        "zero_difficulty_frac": 0.0,
        "curriculum_grace": 800_000,
        "curriculum_steps": 6_400_000,
        "diagnose": True,
        "seed": seed,
        "checkpoint_interval": 393_216,
        "actor_history_len": 10,
        "actor_observation_noise": True,
        "env_variant": "g1_tracking_rmr_50hz_source_step",
        "actor_per_env_grad_clip": None,
        "critic_per_env_grad_clip": None,
        "actor_bootstrap_scale": 1.0,
        "actor_bootstrap_delay_steps": 0,
        "actor_hidden": (512, 256, 128),
        "actor_layer_norm": True,
        "actor_zero_output": True,
        "effort_limit_scale": 1.0,
        "termination_margin_weight": 0.0,
        "reference_reset_noise_scale": 1.0,
        "reference_residual_control": True,
        "reference_residual_scale": 0.5,
        "reference_path": str(reference_path),
        "reference_stride": 2,
        "solver_profile": profile_name,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Immutable canonical G1 SHAC run; scientific settings are fixed."
        )
    )
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=tuple(sorted(SOLVER_PROFILES)),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("canonical_g1_runs"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_canonical_kwargs(
        args.solver_profile,
        args.reference_path.resolve(),
        args.seed,
    )
    profile = get_solver_profile(args.solver_profile)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(profile):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    print((output_root / relative_save_dir).resolve())


if __name__ == "__main__":
    main()
