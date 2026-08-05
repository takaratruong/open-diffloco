"""Launch the registered G1 RMR-style SHAC discriminator."""

import argparse

import jax

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_MODEL_PATH


def configure_jax() -> None:
    jax.config.update("jax_enable_x64", True)


def build_train_kwargs(
    *,
    steps: int,
    num_envs: int,
    seed: int,
    checkpoint_interval: int,
) -> dict:
    return {
        "total_steps": steps,
        "unroll_length": 16,
        "num_envs": num_envs,
        "actor_lr": 5e-3,
        "critic_lr": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "target_update_rate": 0.01,
        "critic_iterations": 16,
        "xml_path": DEFAULT_MODEL_PATH,
        "action_scale": 1.0,
        "action_noise_std_start": 0.5,
        "action_noise_std_end": 0.32,
        "friction_range": (1.0, 1.0),
        "mass_range": (1.0, 1.0),
        "kp_range": (1.0, 1.0),
        "kd_range": (1.0, 1.0),
        "com_offset_range": (0.0, 0.0, 0.0),
        "push_velocity_range": (0.0, 0.0),
        "push_interval_s": 1e9,
        "terrain": False,
        "diagnose": True,
        "zero_difficulty_frac": 1.0,
        "curriculum_grace": steps,
        "curriculum_steps": 1,
        "seed": seed,
        "checkpoint_interval": checkpoint_interval,
        "max_episode_length": 120,
        "actor_history_len": 1,
        "env_variant": "g1_tracking",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=1024)
    args = parser.parse_args()
    configure_jax()
    train(
        **build_train_kwargs(
            steps=args.steps,
            num_envs=args.num_envs,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
        )
    )


if __name__ == "__main__":
    main()
