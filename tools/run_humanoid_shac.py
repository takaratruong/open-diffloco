"""Run the pinned Open-DiffLoco SHAC configuration without automatic rendering."""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configure_jax():
    """Match the upstream CLI's global float64 physics configuration."""
    import jax

    jax.config.update("jax_enable_x64", True)


def build_train_kwargs(
    *,
    steps: int,
    num_envs: int,
    seed: int,
    checkpoint_interval: int,
) -> dict:
    """Return the explicit upstream SHAC parameter contract."""
    return {
        "total_steps": steps,
        "unroll_length": 12,
        "num_envs": num_envs,
        "actor_lr": 5e-3,
        "critic_lr": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "target_update_rate": 0.01,
        "critic_iterations": 16,
        "use_lr_decay": False,
        "xml_path": str(ROOT / "src/envs/humanoid/models/humanoid_mjx.xml"),
        "action_scale": 0.5,
        "cmd_vel_x_range": (-1.5, 1.5),
        "cmd_vel_y_range": (-1.0, 1.0),
        "cmd_yaw_rate_range": (-1.5, 1.5),
        "cmd_zero_prob": (0.1, 0.7, 0.5),
        "cmd_ctrl_interval_range": (60, 140),
        "action_noise_std_start": 0.5,
        "action_noise_std_end": 0.32,
        "friction_range": (0.5, 2.0),
        "mass_range": (0.85, 1.15),
        "kp_range": (25.0, 45.0),
        "kd_range": (0.3, 0.7),
        "com_offset_range": (0.05, 0.05, 0.04),
        "push_velocity_range": (-1.0, 1.0),
        "push_interval_s": 4.0,
        "terrain_flat_prob": 0.2,
        "terrain_slope_max": 5.0,
        "terrain_bump_std": 0.4,
        "terrain_bump_decay": 0.4,
        "terrain": False,
        "zero_difficulty_frac": 0.0,
        "curriculum_grace": 0,
        "curriculum_steps": 1,
        "diagnose": True,
        "seed": seed,
        "resume_from": None,
        "checkpoint_interval": checkpoint_interval,
        "max_episode_length": 5_000,
        "actor_history_len": 10,
        "env_variant": "humanoid_blind_linvel_nokinref",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(output_root)

    from src.algorithms.shac.algorithm import train

    _, folder = train(
        **build_train_kwargs(
            steps=args.steps,
            num_envs=args.num_envs,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
        )
    )
    print((output_root / folder).resolve())


if __name__ == "__main__":
    main()
