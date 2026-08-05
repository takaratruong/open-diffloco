"""Launch robust G1 SHAC at the native RMR 50 Hz MDP timebase."""

import argparse

from src.algorithms.shac.algorithm import train
from tools.run_g1_tracking_shac import (
    build_train_kwargs as build_100hz_train_kwargs,
    configure_jax,
)


def build_train_kwargs(
    *,
    steps: int,
    num_envs: int,
    seed: int,
    checkpoint_interval: int,
    actor_lr: float = 1e-4,
    action_noise_std: float = 0.05,
    action_noise_std_end: float | None = None,
    actor_bootstrap_scale: float = 1.0,
    unroll_length: int = 24,
    unbounded_actions: bool = False,
    mjlab_plant: bool = False,
) -> dict:
    kwargs = build_100hz_train_kwargs(
        steps=steps,
        num_envs=num_envs,
        seed=seed,
        checkpoint_interval=checkpoint_interval,
        actor_lr=actor_lr,
        action_noise_std=action_noise_std,
        actor_bootstrap_scale=actor_bootstrap_scale,
    )
    kwargs.update(
        {
            "unroll_length": unroll_length,
            "action_noise_std_end": (
                action_noise_std
                if action_noise_std_end is None
                else action_noise_std_end
            ),
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_episode_length": 60,
            "env_variant": (
                "g1_tracking_rmr_50hz_mjlab"
                if mjlab_plant
                else "g1_tracking_rmr_50hz_unbounded"
                if unbounded_actions
                else "g1_tracking_rmr_50hz"
            ),
        }
    )
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=196608)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=49152)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--action-noise-std", type=float, default=0.05)
    parser.add_argument("--action-noise-std-end", type=float)
    parser.add_argument("--actor-bootstrap-scale", type=float, default=1.0)
    parser.add_argument("--unroll-length", type=int, default=24)
    parser.add_argument("--unbounded-actions", action="store_true")
    parser.add_argument("--mjlab-plant", action="store_true")
    args = parser.parse_args()

    configure_jax()
    train(
        **build_train_kwargs(
            steps=args.steps,
            num_envs=args.num_envs,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
            actor_lr=args.actor_lr,
            action_noise_std=args.action_noise_std,
            action_noise_std_end=args.action_noise_std_end,
            actor_bootstrap_scale=args.actor_bootstrap_scale,
            unroll_length=args.unroll_length,
            unbounded_actions=args.unbounded_actions,
            mjlab_plant=args.mjlab_plant,
        )
    )


if __name__ == "__main__":
    main()
