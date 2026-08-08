"""Launch robust G1 SHAC at the native RMR 50 Hz MDP timebase."""

import argparse
from contextlib import nullcontext
import math
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.core.rmr_policy import rmr_policy_from_state_dict
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.fixed_solver import fixed_mjx_solver_outer_loop
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
    actor_bootstrap_delay_steps: int = 0,
    unroll_length: int = 24,
    unbounded_actions: bool = False,
    validated_task: bool = False,
    source_actor_policy=None,
    initial_full_actor_policy=None,
    residual_action_scale: float = 0.0,
    differentiate_source_feedback: bool = True,
    body_mass_scale: float = 1.0,
    effort_limit_scale: float = 1.0,
    termination_margin_weight: float = 0.0,
    actor_hidden: tuple[int, ...] = (512, 256, 128),
    actor_layer_norm: bool = True,
    actor_zero_output: bool = True,
    gradient_accumulation_steps: int = 1,
    resume_from: str | Path | None = None,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    reference_stride: int = 2,
) -> dict:
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps < 1
    ):
        raise ValueError(
            "gradient_accumulation_steps must be a positive integer"
        )
    if (
        isinstance(actor_bootstrap_delay_steps, bool)
        or not isinstance(actor_bootstrap_delay_steps, int)
        or actor_bootstrap_delay_steps < 0
    ):
        raise ValueError(
            "actor_bootstrap_delay_steps must be a non-negative integer"
        )
    if (
        isinstance(reference_stride, bool)
        or not isinstance(reference_stride, int)
        or reference_stride < 1
    ):
        raise ValueError("reference_stride must be a positive integer")
    if (
        source_actor_policy is not None
        and initial_full_actor_policy is not None
    ):
        raise ValueError(
            "source_actor_policy and initial_full_actor_policy are "
            "mutually exclusive"
        )
    if initial_full_actor_policy is not None and not validated_task:
        raise ValueError(
            "initial_full_actor_policy requires validated_task source order"
        )
    if validated_task and unbounded_actions:
        raise ValueError(
            "validated_task already includes unbounded source actions"
        )
    if source_actor_policy is None and residual_action_scale != 0.0:
        raise ValueError(
            "residual_action_scale requires source_actor_policy"
        )
    if source_actor_policy is not None and residual_action_scale <= 0.0:
        raise ValueError(
            "source_actor_policy requires a positive residual_action_scale"
        )
    if not math.isfinite(body_mass_scale) or body_mass_scale <= 0.0:
        raise ValueError("body_mass_scale must be positive and finite")
    if not math.isfinite(effort_limit_scale) or effort_limit_scale <= 0.0:
        raise ValueError("effort_limit_scale must be positive and finite")
    if (
        isinstance(termination_margin_weight, bool)
        or not math.isfinite(termination_margin_weight)
        or termination_margin_weight < 0.0
    ):
        raise ValueError(
            "termination_margin_weight must be non-negative and finite"
        )
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
                "g1_tracking_rmr_50hz_validated"
                if validated_task
                else (
                    "g1_tracking_rmr_50hz_unbounded"
                    if unbounded_actions
                    else "g1_tracking_rmr_50hz"
                )
            ),
            "source_actor_policy": source_actor_policy,
            "initial_full_actor_policy": initial_full_actor_policy,
            "residual_action_scale": residual_action_scale,
            "differentiate_source_feedback": differentiate_source_feedback,
            "mass_range": (body_mass_scale, body_mass_scale),
            "effort_limit_scale": effort_limit_scale,
            "termination_margin_weight": termination_margin_weight,
            "actor_hidden": actor_hidden,
            "actor_layer_norm": actor_layer_norm,
            "actor_zero_output": actor_zero_output,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "resume_from": (
                None if resume_from is None else str(resume_from)
            ),
            "actor_bootstrap_delay_steps": actor_bootstrap_delay_steps,
            "reference_path": str(reference_path),
            "reference_stride": reference_stride,
        }
    )
    return kwargs


def load_source_actor_policy(checkpoint: Path):
    """Load an RSL-RL checkpoint once, then use pure JAX during training."""
    import torch

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    return rmr_policy_from_state_dict(
        payload["model_state_dict"],
        payload["obs_norm_state_dict"],
    )


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
    parser.add_argument("--actor-bootstrap-delay-steps", type=int, default=0)
    parser.add_argument("--unroll-length", type=int, default=24)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--reference-stride", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--unbounded-actions", action="store_true")
    parser.add_argument("--validated-task", action="store_true")
    parser.add_argument("--source-policy-checkpoint", type=Path)
    parser.add_argument("--initialize-full-policy-from", type=Path)
    parser.add_argument("--residual-action-scale", type=float, default=0.1)
    parser.add_argument("--body-mass-scale", type=float, default=1.0)
    parser.add_argument("--effort-limit-scale", type=float, default=1.0)
    parser.add_argument(
        "--termination-margin-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--actor-hidden",
        type=int,
        nargs="+",
        default=(512, 256, 128),
    )
    parser.add_argument(
        "--no-actor-layer-norm",
        action="store_true",
    )
    parser.add_argument(
        "--random-actor-output-head",
        action="store_true",
    )
    parser.add_argument(
        "--stop-gradient-source-feedback",
        action="store_true",
    )
    args = parser.parse_args()

    configure_jax()
    source_actor_policy = (
        load_source_actor_policy(args.source_policy_checkpoint)
        if args.source_policy_checkpoint is not None
        else None
    )
    initial_full_actor_policy = (
        load_source_actor_policy(args.initialize_full_policy_from)
        if args.initialize_full_policy_from is not None
        else None
    )
    solver_scope = (
        fixed_mjx_solver_outer_loop()
        if args.validated_task
        else nullcontext()
    )
    with solver_scope:
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
                actor_bootstrap_delay_steps=(
                    args.actor_bootstrap_delay_steps
                ),
                unroll_length=args.unroll_length,
                unbounded_actions=args.unbounded_actions,
                validated_task=args.validated_task,
                source_actor_policy=source_actor_policy,
                initial_full_actor_policy=initial_full_actor_policy,
                residual_action_scale=(
                    args.residual_action_scale
                    if source_actor_policy is not None
                    else 0.0
                ),
                differentiate_source_feedback=(
                    not args.stop_gradient_source_feedback
                ),
                body_mass_scale=args.body_mass_scale,
                effort_limit_scale=args.effort_limit_scale,
                termination_margin_weight=(
                    args.termination_margin_weight
                ),
                actor_hidden=tuple(args.actor_hidden),
                actor_layer_norm=not args.no_actor_layer_norm,
                actor_zero_output=not args.random_actor_output_head,
                gradient_accumulation_steps=(
                    args.gradient_accumulation_steps
                ),
                resume_from=args.resume,
                reference_path=args.reference_path,
                reference_stride=args.reference_stride,
            )
        )


if __name__ == "__main__":
    main()
