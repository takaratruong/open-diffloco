"""Paired replay-free evaluation and side-by-side G1/reference video."""

import argparse
import pickle
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.core.rmr_policy import bound_residual_action
from src.core.rmr_policy import apply_trainable_rmr_policy
from src.envs.g1_tracking.environment import G1TrackingEnv
from src.envs.go2.environment import get_go2_env_class


EVALUATION_ENV_VARIANTS = (
    "g1_tracking",
    "g1_tracking_rmr_50hz",
    "g1_tracking_rmr_50hz_unbounded",
    "g1_tracking_rmr_50hz_source_step",
    "g1_tracking_rmr_50hz_source_step_robust",
    "g1_tracking_rmr_50hz_validated",
)


def configure_jax() -> None:
    """Match the float64 precision used by G1 training."""
    jax.config.update("jax_enable_x64", True)


def make_evaluation_env(
    variant: str,
    *,
    solver_iterations: int | None = None,
    solver_ls_iterations: int | None = None,
    body_mass_scale: float = 1.0,
    effort_limit_scale: float = 1.0,
) -> G1TrackingEnv:
    """Build an exact-termination task on the requested control timebase."""
    if variant not in EVALUATION_ENV_VARIANTS:
        raise ValueError(f"unsupported evaluation environment: {variant}")
    if (solver_iterations is None) != (solver_ls_iterations is None):
        raise ValueError("both solver iteration budgets must be provided")
    kwargs = {
        "actor_history_len": 1,
        "mass_range": (body_mass_scale, body_mass_scale),
        "effort_limit_scale": effort_limit_scale,
    }
    if solver_iterations is not None:
        kwargs.update(
            solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations,
        )
    return get_go2_env_class(variant)(**kwargs)


def load_rmr_policy(checkpoint: Path):
    """Load the source RSL-RL actor without importing Isaac Lab."""
    import torch

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state = payload["model_state_dict"]
    layer_indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    layers = tuple(
        (state[f"actor.{index}.weight"], state[f"actor.{index}.bias"])
        for index in layer_indices
    )
    normalizer = payload["obs_norm_state_dict"]
    mean = normalizer["_mean"]
    std = normalizer["_std"]

    def policy(obs: jax.Array) -> jax.Array:
        with torch.inference_mode():
            value = torch.as_tensor(
                np.array(obs, copy=True),
                dtype=torch.float32,
            ).reshape(1, -1)
            value = (value - mean) / (std + 1e-8)
            for layer, (weight, bias) in enumerate(layers):
                value = torch.nn.functional.linear(value, weight, bias)
                if layer != len(layers) - 1:
                    value = torch.nn.functional.elu(value)
        return jnp.asarray(value.numpy()[0], dtype=jnp.float64)

    return policy


def scale_policy_action(action: jax.Array, gain: float) -> jax.Array:
    """Interpolates between the zero-action controller and a learned policy."""
    if not 0.0 <= gain <= 1.0:
        raise ValueError("action gain must be between 0 and 1")
    return action * gain


def _load_policy(
    env: G1TrackingEnv, checkpoint: Path | None, seed: int
):
    actor = Actor(
        env.action_dim,
        squash=getattr(env, "squash_actor_actions", True),
    )
    if checkpoint is not None:
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        return actor, state.actor_params, state.normalizer

    params = actor.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, env.actor_obs_dim), dtype=jnp.float32),
    )
    return actor, params, Normalizer(env.actor_frame_obs_dim).init()


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.7
    camera.azimuth = 120.0
    camera.elevation = -15.0
    return camera


def _render_pair(
    env: G1TrackingEnv,
    actual_qpos: np.ndarray,
    actual_qvel: np.ndarray,
    phase: int,
    actual_renderer: mujoco.Renderer,
    reference_renderer: mujoco.Renderer,
    actual_data: mujoco.MjData,
    reference_data: mujoco.MjData,
) -> np.ndarray:
    actual_data.qpos[:] = actual_qpos
    actual_data.qvel[:] = actual_qvel
    mujoco.mj_forward(env.mj_model, actual_data)
    reference_data.qpos[:] = env.reference.qpos[phase]
    reference_data.qvel[:] = env.reference.qvel[phase]
    mujoco.mj_forward(env.mj_model, reference_data)
    camera = _camera()
    camera.lookat[:] = env.reference.body_pos[phase, 0]
    actual_renderer.update_scene(actual_data, camera)
    reference_renderer.update_scene(reference_data, camera)
    return np.concatenate(
        (actual_renderer.render(), reference_renderer.render()), axis=1
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the replay-free evaluation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rmr-action-tape", type=Path)
    parser.add_argument("--rmr-policy-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--residual-action-scale", type=float, default=0.0)
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--solver-ls-iterations", type=int)
    parser.add_argument("--body-mass-scale", type=float, default=1.0)
    parser.add_argument("--effort-limit-scale", type=float, default=1.0)
    parser.add_argument("--full-rmr-actor", action="store_true")
    parser.add_argument(
        "--env-variant",
        choices=EVALUATION_ENV_VARIANTS,
        default="g1_tracking",
    )
    return parser


def main() -> None:
    configure_jax()
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 <= args.action_gain <= 1.0:
        parser.error("--action-gain must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = make_evaluation_env(
        args.env_variant,
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
        body_mass_scale=args.body_mass_scale,
        effort_limit_scale=args.effort_limit_scale,
    )
    controller_sources = (
        args.checkpoint,
        args.rmr_action_tape,
        args.rmr_policy_checkpoint,
    )
    paired_residual = (
        args.checkpoint is not None
        and args.rmr_policy_checkpoint is not None
        and args.residual_action_scale > 0.0
    )
    if args.full_rmr_actor and args.checkpoint is None:
        parser.error("--full-rmr-actor requires --checkpoint")
    if args.full_rmr_actor and (
        args.rmr_action_tape is not None
        or args.rmr_policy_checkpoint is not None
        or args.residual_action_scale != 0.0
    ):
        parser.error(
            "--full-rmr-actor is a standalone controller and cannot be "
            "combined with source or residual inputs"
        )
    if sum(source is not None for source in controller_sources) > 1 and not paired_residual:
        parser.error(
            "--checkpoint, --rmr-action-tape, and --rmr-policy-checkpoint "
            "are mutually exclusive unless checkpoint and RMR policy are "
            "paired with --residual-action-scale"
        )
    if args.residual_action_scale < 0.0:
        parser.error("--residual-action-scale must be non-negative")
    if args.residual_action_scale > 0.0 and not paired_residual:
        parser.error(
            "--residual-action-scale requires both --checkpoint and "
            "--rmr-policy-checkpoint"
        )
    actor = actor_params = normalizer_state = None
    full_rmr_actor = None
    action_tape = None
    rmr_policy = None
    if args.rmr_action_tape is not None:
        with np.load(args.rmr_action_tape, allow_pickle=False) as archive:
            source_names = tuple(map(str, archive["joint_names"]))
            permutation = np.array(
                [
                    source_names.index(name)
                    for name in env.actor_joint_names
                ]
            )
            action_tape = np.asarray(archive["action"][:, permutation])
    if args.rmr_policy_checkpoint is not None:
        if env.actor_joint_names != env.controller.actor_joint_names:
            parser.error(
                "--rmr-policy-checkpoint requires the source-order "
                "unbounded RMR environment"
            )
        rmr_policy = load_rmr_policy(args.rmr_policy_checkpoint)
    if args.full_rmr_actor:
        with args.checkpoint.open("rb") as handle:
            full_rmr_actor = pickle.load(handle).actor_params
    elif args.checkpoint is not None or not any(controller_sources):
        actor, actor_params, normalizer_state = _load_policy(
            env, args.checkpoint, args.seed
        )
    state = env.reset_at_phase(
        jax.random.PRNGKey(args.seed),
        jnp.array(0.0),
        jnp.array(args.phase),
    )

    actual_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    reference_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    actual_data = mujoco.MjData(env.mj_model)
    reference_data = mujoco.MjData(env.mj_model)
    frames = []
    records = []

    for step in range(args.max_steps):
        phase = int(state.info["phase"])
        if step % args.render_every == 0:
            frames.append(
                _render_pair(
                    env,
                    np.asarray(state.data.qpos),
                    np.asarray(state.data.qvel),
                    phase,
                    actual_renderer,
                    reference_renderer,
                    actual_data,
                    reference_data,
                )
            )
        if full_rmr_actor is not None:
            action = apply_trainable_rmr_policy(
                full_rmr_actor, state.obs
            ).astype(jnp.float64)
        elif action_tape is not None:
            # The logged RMR controller runs at 50 Hz; the grounded reference
            # and MJX task run at 100 Hz. Zero-order hold each logged action for
            # two reference frames without using it for policy training.
            action = jnp.asarray(
                action_tape[min(phase // 2, len(action_tape) - 1)]
            )
        elif rmr_policy is not None:
            action = rmr_policy(state.obs)
            if actor is not None:
                normalized = env.normalize_actor_obs(
                    Normalizer(env.actor_frame_obs_dim),
                    normalizer_state,
                    state.obs,
                ).astype(jnp.float32)
                residual_logits = actor.apply(actor_params, normalized)
                action = action + bound_residual_action(
                    residual_logits,
                    action_scale=args.residual_action_scale,
                ).astype(jnp.float64)
        else:
            normalized = env.normalize_actor_obs(
                Normalizer(env.actor_frame_obs_dim),
                normalizer_state,
                state.obs,
            ).astype(jnp.float32)
            action = scale_policy_action(
                actor.apply(actor_params, normalized).astype(jnp.float64),
                args.action_gain,
            )
        state = env.step(state, action)
        records.append(
            (
                step,
                phase,
                float(state.reward),
                float(state.done),
                float(state.info["terminal"]),
                float(state.metrics["anchor_position_error"]),
                float(state.metrics["anchor_orientation_error"]),
                float(state.metrics["body_position_error"]),
                float(state.metrics["body_orientation_error"]),
                float(state.metrics["body_linear_velocity_error"]),
                float(state.metrics["body_angular_velocity_error"]),
            )
        )
        if float(state.done) > 0.5:
            break

    columns = (
        "step",
        "phase",
        "reward",
        "done",
        "terminal",
        "anchor_position_error",
        "anchor_orientation_error",
        "body_position_error",
        "body_orientation_error",
        "body_linear_velocity_error",
        "body_angular_velocity_error",
    )
    values = np.asarray(records, dtype=np.float64)
    np.savez_compressed(
        args.output_dir / "evaluation.npz",
        columns=np.asarray(columns),
        values=values,
    )
    imageio.mimsave(
        args.output_dir / "evaluation.mp4",
        frames,
        fps=round(1.0 / (env.dt * args.render_every)),
        quality=8,
    )
    summary = {
        "steps": len(records),
        "terminal": bool(values[-1, 4] > 0.5),
        "mean_reward": float(np.mean(values[:, 2])),
        "mean_anchor_position_error": float(np.mean(values[:, 5])),
        "mean_anchor_orientation_error": float(np.mean(values[:, 6])),
        "mean_body_position_error": float(np.mean(values[:, 7])),
        "mean_body_orientation_error": float(np.mean(values[:, 8])),
        "mean_body_linear_velocity_error": float(np.mean(values[:, 9])),
        "mean_body_angular_velocity_error": float(np.mean(values[:, 10])),
        "action_gain": args.action_gain,
        "residual_action_scale": args.residual_action_scale,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "solver_iterations": int(env.mj_model.opt.iterations),
        "solver_ls_iterations": int(env.mj_model.opt.ls_iterations),
        "body_mass_scale": env.body_mass_scale,
        "effort_limit_scale": env.effort_limit_scale,
        "controller": (
            "full_rmr_actor"
            if full_rmr_actor is not None
            else "rmr_action_tape"
            if action_tape is not None
            else "rmr_residual_policy"
            if rmr_policy is not None and actor is not None
            else "rmr_policy"
            if rmr_policy is not None
            else "flax_policy"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
