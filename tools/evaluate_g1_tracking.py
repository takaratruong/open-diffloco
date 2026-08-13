"""Paired replay-free evaluation and side-by-side G1/reference video."""

import argparse
import json
import math
import pickle
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
    PreviewResidualAdapter,
    apply_frozen_preview_residual,
    split_residual_adapter_params,
)
from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.core.rmr_policy import apply_trainable_rmr_policy, bound_residual_action
from src.envs.g1_tracking.environment import G1TrackingEnv
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from src.envs.go2.environment import get_go2_env_class
from tools.prepare_g1_rmr_reference import sha256_file

EVALUATION_ENV_VARIANTS = (
    "g1_tracking",
    "g1_tracking_rmr_50hz",
    "g1_tracking_rmr_50hz_unbounded",
    "g1_tracking_rmr_50hz_source_step",
    "g1_tracking_rmr_50hz_source_step_robust",
    "g1_tracking_rmr_50hz_action_parity",
    "g1_tracking_rmr_50hz_decoupled_exploration",
    "g1_tracking_rmr_50hz_validated",
)


def training_action_noise_at_step(
    hparams: dict[str, object], step: int, *, action_dim: int
) -> np.ndarray:
    """Resolve the exact reparameterized action-noise scale at a checkpoint."""
    schedule_steps = int(hparams["action_noise_schedule_steps"])
    if schedule_steps <= 0 or step < 0:
        raise ValueError("training action-noise schedule must be positive")

    def endpoint(name: str) -> np.ndarray:
        value = np.asarray(hparams[name], dtype=np.float64)
        if value.ndim == 0:
            value = np.full((action_dim,), float(value), dtype=np.float64)
        if value.shape != (action_dim,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite scalar or action vector")
        return value

    start = endpoint("action_noise_std_start")
    end = endpoint("action_noise_std_end")
    progress = np.clip(step / schedule_steps, 0.0, 1.0)
    return start + progress * (end - start)


def validate_training_action_mean(action_mean: np.ndarray) -> None:
    """Reject a supposedly bounded actor mean that escaped [-1, 1]."""
    values = np.asarray(action_mean)
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or not np.isfinite(values).all()
        or np.max(np.abs(values), initial=0.0) > 1.0 + 1e-6
    ):
        raise ValueError("training actor mean is nonfinite or outside [-1, 1]")


def resolve_rollout_step_limit(
    *,
    remaining: int,
    requested: int | None,
    training_distribution_rollout: bool,
) -> int:
    """Keep training samples running through the environment's auto-resets."""
    if training_distribution_rollout:
        return 120 if requested is None else requested
    return remaining if requested is None else min(requested, remaining)


def resolve_training_visualization_controls(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve optional action-noise-only visualization overrides."""
    overrides_requested = (
        args.disable_training_observation_noise
        or args.exact_training_reset_phase is not None
    )
    if overrides_requested and not args.training_distribution_rollout:
        raise ValueError(
            "training visualization overrides require "
            "--training-distribution-rollout"
        )
    return argparse.Namespace(
        actor_observation_noise=(
            args.training_distribution_rollout
            and not args.disable_training_observation_noise
        ),
        exact_reset_phase=args.exact_training_reset_phase,
        continue_after_terminal=(
            args.training_distribution_rollout
            and args.exact_training_reset_phase is None
        ),
        force_zero_reset_noise=args.exact_training_reset_phase is not None,
    )


def build_compiled_step(env):
    """Compile the repeated MJX environment step once for evaluation."""
    return jax.jit(env.step)


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
    actor_reference_lookahead_steps: tuple[int, ...] = (),
    actor_reference_preview_mode: str = "absolute",
    actor_observation_noise: bool = False,
    domain_randomization: bool = False,
    friction_range: tuple[float, float] = (1.0, 1.0),
    kp_range: tuple[float, float] = (35.0, 35.0),
    kd_range: tuple[float, float] = (0.5, 0.5),
    com_offset_range: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reference_reset_noise_scale: float = 0.0,
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
        "actor_reference_lookahead_steps": actor_reference_lookahead_steps,
        "actor_reference_preview_mode": actor_reference_preview_mode,
        "actor_observation_noise": actor_observation_noise,
        "domain_randomization": domain_randomization,
        "friction_range": friction_range,
        "mass_range": (body_mass_scale, body_mass_scale),
        "kp_range": kp_range,
        "kd_range": kd_range,
        "com_offset_range": com_offset_range,
        "reference_reset_noise_scale": reference_reset_noise_scale,
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


def prepare_evaluation_action(
    action: jax.Array, *, squash: bool
) -> jax.Array:
    """Apply the same post-noise action boundary used by SHAC training."""
    return jnp.clip(action, -1.0, 1.0) if squash else action


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
        composite = isinstance(
            state.actor_params, FrozenPreviewResidualParams
        )
        modules = (
            state.actor_params.parent["params"]
            if composite
            else state.actor_params["params"]
        )
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
            squash=getattr(
                env,
                "squash_actor_mean",
                getattr(env, "squash_actor_actions", True),
            ),
            layer_norm=layer_norm,
            # Initializers do not affect apply(), but using a nonzero head
            # describes both compact random-head checkpoints and legacy
            # checkpoints without changing their stored parameters.
            zero_output=False,
        )
        if composite:
            parent_actor = actor
            adapter_kernel, _ = split_residual_adapter_params(
                state.actor_params.adapter
            )
            residual_actor = PreviewResidualAdapter(
                action_dim=env.action_dim,
                hidden_dim=int(adapter_kernel.shape[1]),
            )
            assistance_scale = (
                jnp.asarray(0.0, dtype=jnp.float32)
                if int(adapter_kernel.shape[0])
                == env.actor_frame_obs_dim + 1
                else None
            )

            class FrozenResidualCheckpointActor:
                def apply(self, params, observations):
                    action, _, _ = apply_frozen_preview_residual(
                        parent_actor,
                        residual_actor,
                        params,
                        observations,
                        history_len=env.actor_history_len,
                        treatment_frame_dim=env.actor_frame_obs_dim,
                        assistance_scale=assistance_scale,
                    )
                    return action

            actor = FrozenResidualCheckpointActor()
        return actor, state.actor_params, state.normalizer

    actor = Actor(
        env.action_dim,
        hidden=actor_hidden,
        squash=getattr(
            env,
            "squash_actor_mean",
            getattr(env, "squash_actor_actions", True),
        ),
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
        "--actor-reference-lookahead-steps",
        type=int,
        nargs="+",
        default=(),
    )
    parser.add_argument(
        "--actor-reference-preview-mode",
        choices=("absolute", "delta"),
        default="absolute",
    )
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
    parser.add_argument(
        "--solver-profile",
        choices=tuple(sorted(SOLVER_PROFILES)),
    )
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
        "--training-distribution-rollout",
        action="store_true",
        help=(
            "sample the checkpoint's randomized reset, observation noise, "
            "and reparameterized action noise instead of clean evaluation"
        ),
    )
    parser.add_argument(
        "--disable-training-observation-noise",
        action="store_true",
        help="retain training action noise but evaluate the actor observation cleanly",
    )
    parser.add_argument(
        "--exact-training-reset-phase",
        type=int,
        help="retain training action noise but reset exactly at this reference phase",
    )
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
    try:
        visualization_controls = resolve_training_visualization_controls(args)
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 <= args.action_gain <= 1.0:
        parser.error("--action-gain must be between 0 and 1")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    training_hparams = None
    checkpoint_step = None
    current_training_noise = None
    training_difficulty = None
    if args.training_distribution_rollout:
        if args.checkpoint is None:
            parser.error("--training-distribution-rollout requires --checkpoint")
        if args.rmr_action_tape is not None or args.rmr_policy_checkpoint is not None:
            parser.error(
                "training-distribution rollout requires a standalone SHAC checkpoint"
            )
        hparams_path = args.checkpoint.parent / "hparams.json"
        if not hparams_path.is_file():
            parser.error("training-distribution rollout requires sibling hparams.json")
        training_hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
        with args.checkpoint.open("rb") as stream:
            checkpoint_step = int(pickle.load(stream).step)
        if training_hparams.get("push_velocity_range") != [0.0, 0.0]:
            parser.error(
                "training-distribution visualization supports only zero-push runs"
            )
        args.env_variant = training_hparams["env_variant"]
        args.reference_path = Path(training_hparams["reference_path"])
        args.reference_stride = int(training_hparams["reference_stride"])
        args.actor_history_len = int(training_hparams["actor_history_len"])
        args.actor_reference_lookahead_steps = tuple(
            training_hparams["actor_reference_lookahead_steps"]
        )
        args.actor_reference_preview_mode = training_hparams[
            "actor_reference_preview_mode"
        ]
        args.reference_residual_control = bool(
            training_hparams["reference_residual_control"]
        )
        args.reference_residual_scale = float(
            training_hparams["reference_residual_scale"]
        )
        args.solver_iterations = int(training_hparams["solver_iterations"])
        args.solver_ls_iterations = int(
            training_hparams["solver_ls_iterations"]
        )
        args.solver_profile = training_hparams["solver_profile"]
        current_training_noise = training_action_noise_at_step(
            training_hparams,
            checkpoint_step,
            action_dim=29,
        )
        grace = int(training_hparams["curriculum_grace"])
        duration = int(training_hparams["curriculum_steps"])
        training_difficulty = float(
            np.clip((checkpoint_step - grace) / duration, 0.0, 1.0)
        )
    profile = (
        None
        if args.solver_profile is None
        else get_solver_profile(args.solver_profile)
    )
    if profile is not None and (
        args.solver_iterations not in (None, profile.iterations)
        or args.solver_ls_iterations not in (None, profile.ls_iterations)
    ):
        parser.error("solver iteration arguments do not match --solver-profile")
    if profile is not None:
        args.solver_iterations = profile.iterations
        args.solver_ls_iterations = profile.ls_iterations
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
        actor_reference_lookahead_steps=tuple(
            args.actor_reference_lookahead_steps
        ),
        actor_reference_preview_mode=args.actor_reference_preview_mode,
        actor_observation_noise=visualization_controls.actor_observation_noise,
        domain_randomization=(
            bool(training_hparams["domain_randomization"])
            if training_hparams is not None
            else False
        ),
        friction_range=(
            tuple(training_hparams["friction_range"])
            if training_hparams is not None
            else (1.0, 1.0)
        ),
        kp_range=(
            tuple(training_hparams["kp_range"])
            if training_hparams is not None
            else (35.0, 35.0)
        ),
        kd_range=(
            tuple(training_hparams["kd_range"])
            if training_hparams is not None
            else (0.5, 0.5)
        ),
        com_offset_range=(
            tuple(training_hparams["com_offset_range"])
            if training_hparams is not None
            else (0.0, 0.0, 0.0)
        ),
        reference_reset_noise_scale=(
            0.0
            if visualization_controls.force_zero_reset_noise
            else float(training_hparams["reference_reset_noise_scale"])
            if training_hparams is not None
            else 0.0
        ),
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
    reset_key, action_noise_key = jax.random.split(
        jax.random.PRNGKey(args.seed)
    )
    if (
        args.training_distribution_rollout
        and visualization_controls.exact_reset_phase is None
    ):
        state = env.reset(
            reset_key,
            jnp.asarray(training_difficulty, dtype=jnp.float64),
        )
    else:
        reset_phase = (
            args.phase
            if visualization_controls.exact_reset_phase is None
            else visualization_controls.exact_reset_phase
        )
        state = env.reset_at_phase(
            reset_key,
            jnp.array(0.0),
            jnp.array(reset_phase),
        )
    start_phase = int(state.info["phase"])
    compiled_step = build_compiled_step(env)

    actual_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    reference_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    actual_data = mujoco.MjData(env.mj_model)
    reference_data = mujoco.MjData(env.mj_model)
    frames = []
    records = []
    action_means = []
    action_epsilons = []
    noisy_actions = []
    effective_actions = []

    try:
        remaining = remaining_reference_transitions(
            env.reference_length,
            start_phase,
            env.reference_stride,
        )
    except ValueError as error:
        parser.error(str(error))
    step_limit = resolve_rollout_step_limit(
        remaining=remaining,
        requested=args.max_steps,
        training_distribution_rollout=args.training_distribution_rollout,
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
        policy_obs = state.obs
        if visualization_controls.actor_observation_noise:
            obs_rng, env_rng = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_rng})
            policy_obs = env._apply_obs_noise(state.obs, obs_rng)
        if full_rmr_actor is not None:
            action = apply_trainable_rmr_policy(
                full_rmr_actor, policy_obs
            ).astype(jnp.float64)
        elif action_tape is not None:
            # The logged RMR controller runs at 50 Hz; the grounded reference
            # and MJX task run at 100 Hz. Zero-order hold each logged action for
            # two reference frames without using it for policy training.
            action = jnp.asarray(
                action_tape[min(phase // 2, len(action_tape) - 1)]
            )
        elif rmr_policy is not None:
            action = rmr_policy(policy_obs)
            if actor is not None:
                normalized = env.normalize_actor_obs(
                    Normalizer(env.actor_frame_obs_dim),
                    normalizer_state,
                    policy_obs,
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
                policy_obs,
            ).astype(jnp.float32)
            action = scale_policy_action(
                actor.apply(actor_params, normalized).astype(jnp.float64),
                args.action_gain,
            )
        action_mean = action
        if args.training_distribution_rollout:
            epsilon = jax.random.normal(
                jax.random.fold_in(action_noise_key, step),
                (env.action_dim,),
                dtype=jnp.float64,
            )
            action = action_mean + epsilon * jnp.asarray(
                current_training_noise, dtype=jnp.float64
            )
            action_means.append(np.asarray(action_mean))
            action_epsilons.append(np.asarray(epsilon))
            noisy_actions.append(np.asarray(action))
        action = prepare_evaluation_action(
            action,
            squash=getattr(
                env,
                "clip_sampled_actor_actions",
                getattr(env, "squash_actor_actions", True),
            ),
        )
        if args.training_distribution_rollout:
            effective_actions.append(np.asarray(action))
        step_scope = (
            nullcontext() if profile is None else solver_context(profile)
        )
        with step_scope:
            state = compiled_step(state, action)
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
        if (
            float(state.done) > 0.5
            and not visualization_controls.continue_after_terminal
        ):
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
    rollout_name = (
        "training_rollout.mp4"
        if args.training_distribution_rollout
        else "evaluation.mp4"
    )
    imageio.mimsave(
        args.output_dir / rollout_name,
        frames,
        fps=round(1.0 / (env.dt * args.render_every)),
        quality=8,
    )
    if args.training_distribution_rollout:
        if getattr(
            env,
            "squash_actor_mean",
            getattr(env, "squash_actor_actions", True),
        ):
            validate_training_action_mean(np.asarray(action_means))
        imageio.mimsave(
            args.output_dir / "training_slice_h12.mp4",
            frames[: min(12, len(frames))],
            fps=round(1.0 / (env.dt * args.render_every)),
            quality=8,
        )
        np.savez_compressed(
            args.output_dir / "training_action_noise.npz",
            action_mean=np.asarray(action_means),
            epsilon=np.asarray(action_epsilons),
            action_std=np.asarray(current_training_noise),
            noisy_action=np.asarray(noisy_actions),
            effective_action=np.asarray(effective_actions),
        )
    from tools.evaluate_g1_phase_grid import make_contact_sheet

    make_contact_sheet(frames, args.output_dir / "contact_sheet.png")
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
        "solver_profile": args.solver_profile,
        "body_mass_scale": env.body_mass_scale,
        "effort_limit_scale": env.effort_limit_scale,
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "reference_fps": env.reference.fps,
        "reference_stride": env.reference_stride,
        "actor_reference_lookahead_steps": list(
            env.actor_reference_lookahead_steps
        ),
        "reference_states": env.reference_length,
        "reference_transitions": env.reference_transitions,
        "evaluation_start_phase": start_phase,
        "remaining_reference_transitions": remaining,
        "requested_step_limit": args.max_steps,
        "training_distribution_rollout": args.training_distribution_rollout,
        "training_observation_noise": visualization_controls.actor_observation_noise,
        "training_exact_reset_phase": visualization_controls.exact_reset_phase,
        "training_checkpoint_step": checkpoint_step,
        "training_difficulty": training_difficulty,
        "training_action_noise_std": (
            current_training_noise.tolist()
            if current_training_noise is not None
            else None
        ),
        "training_action_noise_rms": (
            float(np.sqrt(np.mean(np.square(current_training_noise))))
            if current_training_noise is not None
            else None
        ),
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
        "checkpoint_path": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "checkpoint_sha256": (
            sha256_file(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "actor_assistance_conditioning_scale": (
            0.0
            if isinstance(actor_params, FrozenPreviewResidualParams)
            else None
        ),
        **stability_summary,
    }
    (args.output_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
