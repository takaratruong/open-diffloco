"""Paired replay-free evaluation and side-by-side G1/reference video."""

import argparse
import math
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
from tools.prepare_g1_rmr_reference import sha256_file


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
    reference_path: str | Path | None = None,
    reference_stride: int | None = None,
    actor_history_len: int = 1,
    reference_residual_control: bool = False,
    reference_residual_scale: float = 0.5,
) -> G1TrackingEnv:
    """Build an exact-termination task on the requested control timebase."""
    if variant not in EVALUATION_ENV_VARIANTS:
        raise ValueError(f"unsupported evaluation environment: {variant}")
    if (solver_iterations is None) != (solver_ls_iterations is None):
        raise ValueError("both solver iteration budgets must be provided")
    kwargs = {
        "actor_history_len": actor_history_len,
        "actor_observation_noise": False,
        "mass_range": (body_mass_scale, body_mass_scale),
        "effort_limit_scale": effort_limit_scale,
        "reference_residual_control": reference_residual_control,
        "reference_residual_scale": reference_residual_scale,
    }
    if reference_path is not None:
        kwargs["reference_path"] = str(reference_path)
    if reference_stride is not None:
        kwargs["reference_stride"] = reference_stride
    if solver_iterations is not None:
        kwargs.update(
            solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations,
        )
    return get_go2_env_class(variant)(**kwargs)


def remaining_reference_transitions(
    reference_length: int, phase: int, reference_stride: int
) -> int:
    """Returns the number of carried transitions in a reference suffix."""
    if reference_length < 2:
        raise ValueError("reference must contain at least two states")
    if not 0 <= phase < reference_length - 1:
        raise ValueError("phase must leave at least one reference transition")
    if reference_stride < 1:
        raise ValueError("reference_stride must be positive")
    return math.ceil((reference_length - 1 - phase) / reference_stride)


def summarize_stability_errors(
    errors: dict[str, np.ndarray],
) -> dict[str, float]:
    """Summarizes exact pre-reset RMR termination errors."""
    required = (
        "anchor_z_error",
        "anchor_xy_error",
        "gravity_z_error",
        "distal_z_error",
    )
    missing = [key for key in required if key not in errors]
    if missing:
        raise ValueError(f"stability errors missing arrays: {missing}")
    summary = {}
    for key in required:
        values = np.asarray(errors[key], dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain finite values")
        summary[f"max_{key}"] = float(np.max(values))
    return summary


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
    env: G1TrackingEnv,
    checkpoint: Path | None,
    seed: int,
    *,
    actor_hidden: tuple[int, ...] = (512, 256, 128),
    actor_layer_norm: bool = True,
    actor_zero_output: bool = True,
    training_initialization: bool = False,
):
    if checkpoint is not None:
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        modules = state.actor_params["params"]
        dense_names = sorted(
            (name for name in modules if name.startswith("Dense_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if len(dense_names) < 2:
            raise ValueError(
                "checkpoint actor must contain at least one hidden Dense "
                "layer and one output Dense layer"
            )
        hidden = tuple(
            int(modules[name]["kernel"].shape[-1])
            for name in dense_names[:-1]
        )
        layer_norm = any(
            name.startswith("LayerNorm_") for name in modules
        )
        actor = Actor(
            env.action_dim,
            hidden=hidden,
            squash=getattr(env, "squash_actor_actions", True),
            layer_norm=layer_norm,
            # Initializers do not affect apply(), but using a nonzero head
            # describes both compact random-head checkpoints and legacy
            # checkpoints without changing their stored parameters.
            zero_output=False,
        )
        return actor, state.actor_params, state.normalizer

    actor = Actor(
        env.action_dim,
        hidden=actor_hidden,
        squash=getattr(env, "squash_actor_actions", True),
        layer_norm=actor_layer_norm,
        zero_output=actor_zero_output,
    )
    actor_key = jax.random.PRNGKey(seed)
    if training_initialization:
        _unused, actor_key, _critic_key, _env_key = jax.random.split(
            actor_key, 4
        )
    params = actor.init(
        actor_key,
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
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--reference-stride", type=int)
    parser.add_argument("--actor-history-len", type=int, default=1)
    parser.add_argument(
        "--reference-residual-control", action="store_true"
    )
    parser.add_argument(
        "--reference-residual-scale", type=float, default=0.5
    )
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--residual-action-scale", type=float, default=0.0)
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--solver-ls-iterations", type=int)
    parser.add_argument("--body-mass-scale", type=float, default=1.0)
    parser.add_argument("--effort-limit-scale", type=float, default=1.0)
    parser.add_argument("--full-rmr-actor", action="store_true")
    parser.add_argument(
        "--actor-hidden",
        type=int,
        nargs="+",
        default=(512, 256, 128),
    )
    parser.add_argument("--no-actor-layer-norm", action="store_true")
    parser.add_argument("--random-actor-output-head", action="store_true")
    parser.add_argument(
        "--training-initialization",
        action="store_true",
        help="recreate train()'s actor PRNG split for a no-checkpoint baseline",
    )
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
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = make_evaluation_env(
        args.env_variant,
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
        body_mass_scale=args.body_mass_scale,
        effort_limit_scale=args.effort_limit_scale,
        reference_path=args.reference_path,
        reference_stride=args.reference_stride,
        actor_history_len=args.actor_history_len,
        reference_residual_control=args.reference_residual_control,
        reference_residual_scale=args.reference_residual_scale,
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
            env,
            args.checkpoint,
            args.seed,
            actor_hidden=tuple(args.actor_hidden),
            actor_layer_norm=not args.no_actor_layer_norm,
            actor_zero_output=not args.random_actor_output_head,
            training_initialization=args.training_initialization,
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

    try:
        remaining = remaining_reference_transitions(
            env.reference_length,
            args.phase,
            env.reference_stride,
        )
    except ValueError as error:
        parser.error(str(error))
    step_limit = (
        remaining
        if args.max_steps is None
        else min(args.max_steps, remaining)
    )

    for step in range(step_limit):
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
        next_phase = min(
            phase + env.reference_stride,
            env.reference_length - 1,
        )
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
                next_phase,
                float(state.metrics["termination_anchor_z_error"]),
                float(state.metrics["termination_anchor_xy_error"]),
                float(state.metrics["termination_gravity_z_error"]),
                float(state.metrics["termination_distal_z_error"]),
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
        "transition_phase",
        "termination_anchor_z_error",
        "termination_anchor_xy_error",
        "termination_gravity_z_error",
        "termination_distal_z_error",
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
    stability_summary = summarize_stability_errors(
        {
            "anchor_z_error": values[:, 12],
            "anchor_xy_error": values[:, 13],
            "gravity_z_error": values[:, 14],
            "distal_z_error": values[:, 15],
        }
    )
    reference_path = Path(env.reference_path).resolve()
    true_terminal = bool(np.any(values[:, 4] > 0.5))
    completed_suffix = (
        len(records) == remaining
        and int(values[-1, 11]) == env.reference_length - 1
        and not true_terminal
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
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "reference_fps": env.reference.fps,
        "reference_stride": env.reference_stride,
        "reference_states": env.reference_length,
        "reference_transitions": env.reference_transitions,
        "evaluation_start_phase": args.phase,
        "remaining_reference_transitions": remaining,
        "requested_step_limit": args.max_steps,
        "completed_reference_suffix": completed_suffix,
        "intermediate_reset_occurred": true_terminal,
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
        **stability_summary,
    }
    (args.output_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
