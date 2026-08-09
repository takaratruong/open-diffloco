"""Exact shared-rollout runtime for the frozen E064 G1 gradient audit.

This module is intentionally separate from the production SHAC trainer.  It
materializes one trajectory and derives both the reparameterized pathwise and
detached Gaussian score gradients from that trajectory.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.gradient_audit import (
    detached_gaussian_score_loss,
    discounted_return_to_go,
)
from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.envs.g1_tracking.environment import G1TrackingRMR50HzValidatedEnv

PyTree = Any

E064_CHECKPOINT_SHA256 = (
    "6b5c6bb208f9acd9f5988fee201915f8aa67cba42c15231d361a4d2ae530a094"
)
E064_HPARAMS_SHA256 = (
    "98499799f221978510ee15b9417a4e408a6a0ae1aff95c9f84b48a4cc88a9c8b"
)
E064_REFERENCE_SHA256 = (
    "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
)
E064_XML_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
E064_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
E064_LIVE_RUNTIME_SHA256 = (
    "118b94a353ec0de85411e6179e681fddb364d79fd381f7e95380a8a84a474883"
)
E064_ACTOR_PARAMETERS_SHA256 = (
    "39b359b03d3415fc55b59efce1d5166ced0728b025bf526901b53af205603954"
)
E064_NORMALIZER_SHA256 = (
    "e4a3a615e1b2016dddb7c6f29359ac547cf25de2e01d5ba0511b052766282ec7"
)
E064_INITIAL_STATE_SHA256 = (
    "6ec9738c68eb9ce854e8aa430749aa6a6becf1c82081f390bf702d3f3fcd0a98"
)

E064_FROZEN_HPARAMS: Mapping[str, Any] = MappingProxyType({
    "action_noise_std_end": 0.1,
    "action_noise_std_start": 1.0,
    "action_scale": 1.0,
    "actor_bootstrap_delay_steps": 0,
    "actor_bootstrap_scale": 0.0,
    "actor_hidden": (512, 512),
    "actor_history_len": 1,
    "actor_kind": "flax",
    "actor_layer_norm": False,
    "actor_lr": 0.001,
    "actor_per_env_grad_clip": 1.0,
    "actor_zero_output": False,
    "algorithm": "shac",
    "best_reward": 0.07537891473167549,
    "cmd_ctrl_interval_range": (60, 140),
    "cmd_vel_x_range": (-2.0, 2.0),
    "cmd_vel_y_range": (-1.0, 1.0),
    "cmd_yaw_rate_range": (-1.5, 1.5),
    "cmd_zero_prob": (0.1, 0.7, 0.5),
    "com_offset_range": (0.0, 0.0, 0.0),
    "critic_iterations": 16,
    "critic_lr": 0.0005,
    "critic_per_env_grad_clip": 1.0,
    "curriculum_grace": 393216,
    "curriculum_steps": 1,
    "differentiate_source_feedback": True,
    "effective_num_envs": 64,
    "effort_limit_scale": 1.0,
    "env_variant": "g1_tracking_rmr_50hz_validated",
    "friction_range": (1.0, 1.0),
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "gradient_accumulation_steps": 1,
    "kd_range": (1.0, 1.0),
    "kp_range": (1.0, 1.0),
    "mass_range": (1.0, 1.0),
    "max_episode_length": 499,
    "num_envs": 64,
    "push_interval_s": 1_000_000_000.0,
    "push_velocity_range": (0.0, 0.0),
    "reference_fps": 50.0,
    "reference_path": (
        "/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/"
        "artifacts/E-20260808-000/reference/"
        "dance1_subject2_f122_422_50hz.npz"
    ),
    "reference_sha256": E064_REFERENCE_SHA256,
    "reference_states": 500,
    "reference_stride": 1,
    "reference_transitions": 499,
    "residual_action_scale": 0.0,
    "seed": 1,
    "source_actor_policy": False,
    "squash_actor_actions": False,
    "steps_per_actor_update": 3072,
    "target_update_rate": 0.01,
    "termination_margin_weight": 0.0,
    "terrain": False,
    "terrain_bump_decay": 0.4,
    "terrain_bump_std": 0.4,
    "terrain_flat_prob": 0.2,
    "terrain_slope_max": 5.0,
    "total_steps": 393216,
    "unroll_length": 48,
    "xml_path": (
        "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
        "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
    ),
    "zero_difficulty_frac": 1.0,
})


class SharedTrajectory(NamedTuple):
    """Materialized stochastic inputs and transition data for one population."""

    noise: jax.Array
    observation_rngs: jax.Array
    raw_observations: jax.Array
    observations: jax.Array
    normalized_observations: jax.Array
    means: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    initial_phase: jax.Array


class SharedGradientResult(NamedTuple):
    """Per-environment estimators derived from one shared trajectory."""

    losses: jax.Array
    trajectory: SharedTrajectory
    score_trajectory: SharedTrajectory
    final_states: PyTree
    pathwise_raw_gradients: PyTree
    pathwise_effective_gradients: PyTree
    pathwise_raw_norms: jax.Array
    pathwise_clip_scales: jax.Array
    pathwise_returns_to_go: jax.Array
    pathwise_score_losses: jax.Array
    score_means: jax.Array
    score_returns_to_go: jax.Array
    score_objectives: jax.Array
    score_losses: jax.Array
    score_gradients: PyTree


class ValidatedE064Contract(NamedTuple):
    """Small immutable receipt returned after fail-closed validation."""

    population: int
    horizon: int
    sigma: float
    gamma: float
    pathwise_clip_norm: float
    hparams_sha256: str
    actor_parameters_sha256: str
    normalizer_sha256: str
    initial_state_sha256: str
    env_variant: str
    reference_sha256: str
    actor_parameter_signature: tuple
    normalizer_signature: tuple
    initial_state_signature: tuple


class ValidatedCheckpointShapes(NamedTuple):
    """Pure shape/dtype contract used by tests and literal validation."""

    population: int
    horizon: int
    sigma: float
    actor_parameter_signature: tuple
    normalizer_signature: tuple
    initial_state_signature: tuple


def _require_unbounded_environment(env: Any) -> None:
    if getattr(env, "clip_actions", None) is not False:
        raise ValueError("the G1 audit requires an explicitly unbounded environment")
    if getattr(env, "squash_actor_actions", None) is not False:
        raise ValueError("the G1 audit requires unbounded, unsquashed actor actions")


def rollout_one_environment(
    actor_params: PyTree,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    env: Any,
    *,
    normalizer: Any,
    normalizer_state: PyTree,
    initial_state: PyTree,
    action_noise: jax.Array,
    sigma: float | jax.Array,
) -> tuple[SharedTrajectory, PyTree]:
    """Runs one E064-compatible shared trajectory.

    ``action_noise`` has shape ``(time, action)``.  The state's RNG is split in
    exactly the production order: the first child is offered to actor
    observation handling and the second child is carried by the environment.
    The validated G1 environment's observation transform is identity.
    """

    _require_unbounded_environment(env)
    if action_noise.ndim != 2 or action_noise.shape[0] < 1:
        raise ValueError("action_noise must have shape (nonzero time, action)")
    if isinstance(sigma, (int, float, np.number)) and (
        not np.isfinite(sigma) or sigma <= 0.0
    ):
        raise ValueError("sigma must be a positive finite scalar")
    sigma_array = jp.asarray(sigma)
    if sigma_array.ndim != 0:
        raise ValueError("sigma must be a scalar")

    state = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_state)
    initial_phase = jax.lax.stop_gradient(state.info["phase"])

    def step(state, noise_t):
        observation_rng, environment_rng = jax.random.split(state.info["rng"])
        state = state.replace(
            info={**state.info, "rng": environment_rng}
        )
        actor_observation = env._apply_obs_noise(
            state.obs, observation_rng
        )
        normalized_observation = env.normalize_actor_obs(
            normalizer,
            normalizer_state,
            actor_observation,
        ).astype(jp.float32)
        mean = actor_apply(actor_params, normalized_observation).astype(jp.float64)
        action = mean + sigma_array.astype(jp.float64) * noise_t.astype(jp.float64)
        next_state = env.step(state, action)
        transition = {
            "noise": noise_t,
            "observation_rngs": observation_rng,
            "raw_observations": state.obs,
            "observations": actor_observation,
            "normalized_observations": normalized_observation,
            "means": mean,
            "actions": action,
            "rewards": next_state.reward,
            "dones": next_state.done,
        }
        return next_state, transition

    final_state, arrays = jax.lax.scan(step, state, action_noise)
    return (
        SharedTrajectory(
            noise=arrays["noise"],
            observation_rngs=arrays["observation_rngs"],
            raw_observations=arrays["raw_observations"],
            observations=arrays["observations"],
            normalized_observations=arrays["normalized_observations"],
            means=arrays["means"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            dones=arrays["dones"],
            initial_phase=initial_phase,
        ),
        final_state,
    )


def rollout_batched_environments(
    actor_params: PyTree,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    env: Any,
    *,
    normalizer: Any,
    normalizer_state: PyTree,
    initial_states: PyTree,
    action_noise: jax.Array,
    sigma: float | jax.Array,
) -> tuple[SharedTrajectory, PyTree]:
    """Vectorizes :func:`rollout_one_environment` over carried states."""

    return jax.vmap(
        lambda state, noise: rollout_one_environment(
            actor_params,
            actor_apply,
            env,
            normalizer=normalizer,
            normalizer_state=normalizer_state,
            initial_state=state,
            action_noise=noise,
            sigma=sigma,
        )
    )(initial_states, action_noise)


def pathwise_negative_objective(
    rewards: jax.Array,
    dones: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    """Returns exact SHAC negative return with zero actor bootstrap.

    Discounts start at one, advance after each transition, and reset to one
    after a true episode boundary.  The fragment's incomplete final episode is
    included without a value bootstrap.
    """

    if rewards.ndim != 1 or dones.shape != rewards.shape or rewards.shape[0] < 1:
        raise ValueError("rewards and dones must be nonempty one-dimensional arrays")

    def accumulate(carry, transition):
        total, running, discount = carry
        reward, done = transition
        running = running + discount * reward
        total = total + jp.where(done, running, jp.zeros_like(running))
        running = jp.where(done, jp.zeros_like(running), running)
        discount = jp.where(done, jp.ones_like(discount), discount * gamma)
        return (total, running, discount), None

    zero = jp.zeros_like(rewards[0])
    one = jp.ones_like(rewards[0])
    (total, running, _), _ = jax.lax.scan(
        accumulate,
        (zero, zero, one),
        (rewards, dones),
    )
    return -(total + running) / rewards.shape[0]


def _pathwise_effective_per_environment(
    gradients: PyTree,
    *,
    max_norm: float,
) -> tuple[PyTree, jax.Array, jax.Array]:
    """Applies the production whole-environment sanitization and norm clip."""

    leaves = jax.tree_util.tree_leaves(gradients)
    if not leaves:
        raise ValueError("gradient tree must not be empty")
    population = leaves[0].shape[0]
    finite = jp.ones((population,), dtype=jp.bool_)
    squared_norm = jp.zeros((population,), dtype=jp.float32)
    for leaf in leaves:
        if leaf.shape[0] != population:
            raise ValueError("gradient leaves must share the environment axis")
        axes = tuple(range(1, leaf.ndim))
        finite = finite & jp.all(jp.isfinite(leaf), axis=axes)
        safe = jp.where(jp.isfinite(leaf), leaf, 0.0).astype(jp.float32)
        squared_norm = squared_norm + jp.sum(jp.square(safe), axis=axes)
    raw_norms = jp.sqrt(squared_norm)
    scales = jp.minimum(1.0, max_norm / jp.maximum(raw_norms, 1e-12))
    scales = jp.where(finite, scales, 0.0)

    def make_effective(leaf):
        safe = jp.where(jp.isfinite(leaf), leaf, 0.0)
        shape = (population,) + (1,) * (leaf.ndim - 1)
        return safe * scales.reshape(shape)

    return jax.tree_util.tree_map(make_effective, gradients), raw_norms, scales


def estimate_shared_gradients(
    actor_params: PyTree,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    env: Any,
    *,
    normalizer: Any,
    normalizer_state: PyTree,
    initial_states: PyTree,
    action_noise: jax.Array,
    sigma: float | jax.Array,
    gamma: float,
    pathwise_clip_norm: float,
) -> SharedGradientResult:
    """Computes vmapped pathwise and detached score gradients from one rollout."""

    _require_unbounded_environment(env)
    if action_noise.ndim != 3:
        raise ValueError("batched action_noise must have shape (env, time, action)")

    def pathwise_loss(params, state, noise):
        trajectory, final_state = rollout_one_environment(
            params,
            actor_apply,
            env,
            normalizer=normalizer,
            normalizer_state=normalizer_state,
            initial_state=state,
            action_noise=noise,
            sigma=sigma,
        )
        loss = pathwise_negative_objective(
            trajectory.rewards,
            trajectory.dones,
            gamma=gamma,
        )
        return loss, (trajectory, final_state)

    value_and_gradient = jax.value_and_grad(pathwise_loss, has_aux=True)
    (
        (losses, (trajectory, final_states)),
        pathwise_raw_gradients,
    ) = jax.vmap(value_and_gradient, in_axes=(None, 0, 0))(
        actor_params,
        initial_states,
        action_noise,
    )
    score_trajectory = jax.tree_util.tree_map(
        jax.lax.stop_gradient, trajectory
    )

    @jax.custom_vjp
    def attach_actor_derivative(params, observations, stored_means):
        return stored_means

    def attach_actor_derivative_fwd(params, observations, stored_means):
        return stored_means, (params, observations, stored_means)

    def attach_actor_derivative_bwd(residual, cotangent):
        params, observations, stored_means = residual

        def current_policy(candidate_params):
            return jax.vmap(lambda obs: actor_apply(candidate_params, obs))(
                observations
            ).astype(stored_means.dtype)

        _, pullback = jax.vjp(current_policy, params)
        return (
            pullback(cotangent)[0],
            jax.tree_util.tree_map(jp.zeros_like, observations),
            jax.tree_util.tree_map(jp.zeros_like, stored_means),
        )

    attach_actor_derivative.defvjp(
        attach_actor_derivative_fwd, attach_actor_derivative_bwd
    )

    def score_inputs(params, data):
        stopped_normalized_observations = jax.lax.stop_gradient(
            data.normalized_observations
        )
        means = attach_actor_derivative(
            params,
            stopped_normalized_observations,
            data.means,
        )
        returns = discounted_return_to_go(
            data.rewards,
            data.dones,
            gamma=gamma,
        )
        return means, returns

    def score_loss(params, data):
        means, returns = score_inputs(params, data)
        return detached_gaussian_score_loss(
            means,
            data.actions,
            returns,
            std=sigma,
        )

    score_losses, score_gradients = jax.vmap(
        jax.value_and_grad(score_loss), in_axes=(None, 0)
    )(actor_params, score_trajectory)
    score_means, score_returns_to_go = jax.vmap(
        score_inputs, in_axes=(None, 0)
    )(actor_params, score_trajectory)
    pathwise_returns_to_go = jax.vmap(
        lambda rewards, dones: discounted_return_to_go(
            rewards, dones, gamma=gamma
        )
    )(trajectory.rewards, trajectory.dones)
    pathwise_score_losses = jax.vmap(
        lambda means, actions, returns: detached_gaussian_score_loss(
            means, actions, returns, std=sigma
        )
    )(trajectory.means, trajectory.actions, pathwise_returns_to_go)
    score_objectives = jax.vmap(
        lambda rewards, dones: pathwise_negative_objective(
            rewards, dones, gamma=gamma
        )
    )(score_trajectory.rewards, score_trajectory.dones)
    (
        pathwise_effective_gradients,
        pathwise_raw_norms,
        pathwise_clip_scales,
    ) = _pathwise_effective_per_environment(
        pathwise_raw_gradients,
        max_norm=pathwise_clip_norm,
    )
    return SharedGradientResult(
        losses=losses,
        trajectory=trajectory,
        score_trajectory=score_trajectory,
        final_states=final_states,
        pathwise_raw_gradients=pathwise_raw_gradients,
        pathwise_effective_gradients=pathwise_effective_gradients,
        pathwise_raw_norms=pathwise_raw_norms,
        pathwise_clip_scales=pathwise_clip_scales,
        pathwise_returns_to_go=pathwise_returns_to_go,
        pathwise_score_losses=pathwise_score_losses,
        score_means=score_means,
        score_returns_to_go=score_returns_to_go,
        score_objectives=score_objectives,
        score_losses=score_losses,
        score_gradients=score_gradients,
    )


def prepare_compiled_estimator_core(
    actor_params: PyTree,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    env: Any,
    *,
    normalizer: Any,
    normalizer_state: PyTree,
    initial_states: PyTree,
    sigma: float,
    gamma: float,
    pathwise_clip_norm: float,
) -> Callable[[jax.Array], SharedGradientResult]:
    """Builds a same-shape JIT core with no host hashing or validation."""

    return jax.jit(
        lambda action_noise: estimate_shared_gradients(
            actor_params,
            actor_apply,
            env,
            normalizer=normalizer,
            normalizer_state=normalizer_state,
            initial_states=initial_states,
            action_noise=action_noise,
            sigma=sigma,
            gamma=gamma,
            pathwise_clip_norm=pathwise_clip_norm,
        )
    )


def prepare_compiled_rollout_core(
    actor_params: PyTree,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    env: Any,
    *,
    normalizer: Any,
    normalizer_state: PyTree,
    initial_states: PyTree,
    sigma: float,
) -> Callable[[jax.Array], tuple[SharedTrajectory, PyTree]]:
    """Builds a reusable candidate rollout from frozen state and normalization."""

    return jax.jit(
        lambda action_noise: rollout_batched_environments(
            actor_params,
            actor_apply,
            env,
            normalizer=normalizer,
            normalizer_state=normalizer_state,
            initial_states=initial_states,
            action_noise=action_noise,
            sigma=sigma,
        )
    )


def _require_host_tree(tree: PyTree, *, label: str) -> None:
    if any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(tree)
    ):
        raise TypeError(f"{label} validation is host-only and must run before jax.jit")


def validate_identity_observation_handling(env: Any, initial_states: Any) -> None:
    """Exercises the live observation hook and requires exact identity output."""

    _require_host_tree(initial_states, label="observation")
    raw_observations = _field(initial_states, "obs")
    state_info = _field(initial_states, "info")
    observation_rngs = jax.vmap(lambda key: jax.random.split(key)[0])(
        state_info["rng"]
    )
    handled = jax.vmap(env._apply_obs_noise)(
        raw_observations, observation_rngs
    )
    if not np.array_equal(
        np.asarray(jax.device_get(handled)),
        np.asarray(jax.device_get(raw_observations)),
    ):
        raise ValueError("E064 observation handling must be exactly identity")


def validate_e064_live_semantics(
    contract: ValidatedE064Contract,
    env: Any,
    normalizer: Any,
    normalizer_state: Any,
    initial_states: Any,
) -> None:
    """Host-validates the live environment and normalization implementations."""

    _require_host_tree(initial_states, label="E064 live semantics")
    if type(normalizer) is not Normalizer:
        raise ValueError("E064 requires the exact Normalizer implementation")
    if (normalizer.size, normalizer.eps) != (154, 1e-4):
        raise ValueError("E064 Normalizer must use size 154 and eps 1e-4")
    if type(env) is not G1TrackingRMR50HzValidatedEnv:
        raise ValueError("E064 requires G1TrackingRMR50HzValidatedEnv exactly")
    causal_methods = {
        name
        for cls in type(env).__mro__
        for name, value in vars(cls).items()
        if callable(value)
    }
    shadowed = sorted(causal_methods.intersection(vars(env)))
    if shadowed:
        raise ValueError(
            "E064 live environment has instance-overridden methods: "
            + ", ".join(shadowed)
        )
    if not hasattr(env, "xml_path") or sha256_file(env.xml_path) != E064_XML_SHA256:
        raise ValueError("E064 physical XML SHA-256 differs from registration")
    if (
        not hasattr(env, "controller_path")
        or sha256_file(env.controller_path) != E064_CONTROLLER_SHA256
    ):
        raise ValueError("E064 controller SHA-256 differs from registration")
    for name, expected in (
        ("body_mass_scale", 1.0),
        ("effort_limit_scale", 1.0),
        ("termination_margin_weight", 0.0),
        ("reference_reset_noise_scale", 0.0),
        ("n_frames", 4),
        ("reward_scale", 0.02),
        ("dt", 0.02),
        ("control_reference_dt", 0.02),
        ("clip_actions", False),
        ("squash_actor_actions", False),
        ("reference_length", 500),
        ("mj_model.nq", 36),
        ("mj_model.nv", 35),
        ("action_dim", 29),
        ("actor_frame_obs_dim", 154),
        ("actor_obs_dim", 154),
        ("critic_obs_dim", 286),
        ("actor_history_len", 1),
        ("reference_stride", 1),
        ("reference_transitions", 499),
        ("max_episode_length", 499),
        ("body_ids", (1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30)),
        ("anchor_body_id", 1),
        ("distal_body_slots", (3, 6, 10, 13)),
    ):
        if name.startswith("mj_model."):
            actual = getattr(env.mj_model, name.removeprefix("mj_model."), None)
        else:
            actual = getattr(env, name, None)
        if actual != expected:
            raise ValueError(f"E064 live environment requires {name}={expected!r}")
    if sha256_file(env.reference_path) != contract.reference_sha256:
        raise ValueError("E064 live reference SHA-256 differs from registration")
    reference = env.reference
    if reference.qpos.shape[0] != 500 or float(reference.fps) != 50.0:
        raise ValueError("E064 live reference must be the 500-state 50 Hz archive")
    model = env.mj_model
    if (model.opt.iterations, model.opt.ls_iterations) != (4, 5):
        raise ValueError("E064 live solver must be exactly 4/5")
    if float(model.opt.timestep) != 0.005:
        raise ValueError("E064 live physics timestep must be exactly 0.005")
    runtime_tree = {
        name: getattr(env, name)
        for name in (
            "qpos_reference",
            "qvel_reference",
            "body_pos_reference",
            "body_quat_reference",
            "body_lin_vel_reference",
            "body_ang_vel_reference",
            "kp",
            "kd",
            "effort_limit",
            "default_joints",
            "action_scales",
            "actor_to_model_permutation",
            "model_to_actor_permutation",
            "soft_joint_lower",
            "soft_joint_upper",
            "_foot_body_ids",
        )
    }
    runtime_tree["mjx_model"] = env.mjx_model
    runtime_tree["mj_model_body_rootid"] = jp.asarray(
        np.asarray(env.mj_model.body_rootid)
    )
    if stable_pytree_sha256(runtime_tree) != E064_LIVE_RUNTIME_SHA256:
        raise ValueError(
            "E064 live controller, reference, or MJX runtime identity drifted"
        )

    validate_identity_observation_handling(env, initial_states)
    observations = _field(initial_states, "obs")
    actual = env.normalize_actor_obs(
        normalizer, normalizer_state, observations
    )
    mean = _field(normalizer_state, "mean")
    var = _field(normalizer_state, "var")
    expected = (observations - mean) / jp.sqrt(var + normalizer.eps)
    if not np.allclose(
        np.asarray(jax.device_get(actual)),
        np.asarray(jax.device_get(expected)),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("E064 live actor normalization semantics differ")


class ExactE064AuditResult(NamedTuple):
    """Validated numerical output and the two matching identity receipts."""

    gradients: SharedGradientResult
    pathwise_receipt: dict[str, Any]
    score_receipt: dict[str, Any]
    contract: ValidatedE064Contract


def _snapshot_pytree(tree: PyTree) -> PyTree:
    """Deep-copies leaves, containers, and registered static PyTree metadata."""

    return copy.deepcopy(tree)


def _assert_runtime_snapshot_identity(
    contract: ValidatedE064Contract,
    hparams: Mapping[str, Any],
    actor_params: PyTree,
    normalizer_state: PyTree,
    initial_state: PyTree,
) -> None:
    identities = {
        "hparams": (
            stable_mapping_sha256(hparams),
            contract.hparams_sha256,
        ),
        "actor parameters": (
            stable_pytree_sha256(actor_params),
            contract.actor_parameters_sha256,
        ),
        "normalizer": (
            stable_pytree_sha256(_normalizer_identity_tree(normalizer_state)),
            contract.normalizer_sha256,
        ),
        "initial state": (
            stable_pytree_sha256(initial_state),
            contract.initial_state_sha256,
        ),
    }
    for label, (actual, expected) in identities.items():
        if actual != expected:
            raise ValueError(f"literal E064 {label} snapshot identity drifted")


class PreparedE064Estimator:
    """Host-validated wrapper around one reusable same-shape compiled core."""

    def __init__(
        self,
        *,
        contract: ValidatedE064Contract,
        actor_params: PyTree,
        normalizer_state: PyTree,
        initial_states: PyTree,
        checkpoint_path: str | Path,
        hparams: Mapping[str, Any],
        env: Any,
        normalizer: Normalizer,
        actor_apply: Callable[[PyTree, jax.Array], jax.Array],
        compiled_core: Callable[[jax.Array], SharedGradientResult],
    ):
        self.contract = contract
        self._actor_params = actor_params
        self._normalizer_state = normalizer_state
        self._initial_states = initial_states
        self._checkpoint_path = Path(checkpoint_path)
        self._hparams = E064_FROZEN_HPARAMS
        self._env = env
        self._normalizer = normalizer
        self._actor_apply = actor_apply
        self._compiled_core = compiled_core

    @property
    def actor_apply(self) -> Callable[[PyTree, jax.Array], jax.Array]:
        """The exact frozen actor apply used for candidate evaluation."""

        return self._actor_apply

    def prepare_candidate_rollout(
        self, actor_params: PyTree
    ) -> Callable[[jax.Array], tuple[SharedTrajectory, PyTree]]:
        """Compiles a candidate on the frozen state without claiming its hash."""

        _require_host_tree(actor_params, label="candidate actor")
        if pytree_shape_signature(actor_params) != self.contract.actor_parameter_signature:
            raise ValueError("candidate actor shape differs from the E064 actor")
        if not all(
            np.all(np.isfinite(np.asarray(jax.device_get(leaf))))
            for leaf in jax.tree_util.tree_leaves(actor_params)
        ):
            raise ValueError("candidate actor must be finite")
        return prepare_compiled_rollout_core(
            actor_params,
            self._actor_apply,
            self._env,
            normalizer=self._normalizer,
            normalizer_state=self._normalizer_state,
            initial_states=self._initial_states,
            sigma=self.contract.sigma,
        )

    def __call__(self, action_noise: jax.Array) -> ExactE064AuditResult:
        _require_host_tree(action_noise, label="exact E064 execution")
        if action_noise.shape != (64, 48, 29):
            raise ValueError("E064 action_noise must have exact shape (64, 48, 29)")
        if sha256_file(self._checkpoint_path) != E064_CHECKPOINT_SHA256:
            raise ValueError("literal E064 checkpoint changed after preparation")
        _assert_runtime_snapshot_identity(
            self.contract,
            self._hparams,
            self._actor_params,
            self._normalizer_state,
            self._initial_states,
        )
        validate_e064_live_semantics(
            self.contract,
            self._env,
            self._normalizer,
            self._normalizer_state,
            self._initial_states,
        )
        result = self._compiled_core(action_noise)
        if not np.array_equal(
            np.asarray(jax.device_get(result.trajectory.raw_observations)),
            np.asarray(jax.device_get(result.trajectory.observations)),
        ):
            raise ValueError("E064 rollout observation hook was not identity")
        expected_normalized = self._env.normalize_actor_obs(
            self._normalizer,
            self._normalizer_state,
            result.trajectory.observations,
        ).astype(jp.float32)
        if not np.array_equal(
            np.asarray(jax.device_get(expected_normalized)),
            np.asarray(
                jax.device_get(result.trajectory.normalized_observations)
            ),
        ):
            raise ValueError("E064 rollout normalization semantics drifted")
        pathwise_receipt, score_receipt = (
            build_and_validate_estimator_receipts(
                checkpoint_sha256=E064_CHECKPOINT_SHA256,
                hparams=self._hparams,
                actor_params=self._actor_params,
                actor_apply=self._actor_apply,
                normalizer_state=_normalizer_identity_tree(
                    self._normalizer_state
                ),
                initial_state=self._initial_states,
                action_noise=action_noise,
                result=result,
                gamma=self.contract.gamma,
                sigma=self.contract.sigma,
                pathwise_clip_norm=self.contract.pathwise_clip_norm,
            )
        )
        return ExactE064AuditResult(
            gradients=result,
            pathwise_receipt=pathwise_receipt,
            score_receipt=score_receipt,
            contract=self.contract,
        )


def prepare_e064_estimator_engine(
    checkpoint_state: Any,
    hparams: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    env: Any,
    normalizer: Normalizer,
) -> PreparedE064Estimator:
    """Host-validates literal E064 identities, then builds one JIT core."""

    contract = validate_e064_checkpoint_contract(
        checkpoint_state, hparams, checkpoint_path=checkpoint_path
    )
    validate_e064_live_semantics(
        contract,
        env,
        normalizer,
        checkpoint_state.normalizer,
        checkpoint_state.env_state,
    )
    actor_params = _snapshot_pytree(checkpoint_state.actor_params)
    normalizer_state = copy.deepcopy(checkpoint_state.normalizer)
    initial_states = _snapshot_pytree(checkpoint_state.env_state)
    _assert_runtime_snapshot_identity(
        contract,
        E064_FROZEN_HPARAMS,
        actor_params,
        normalizer_state,
        initial_states,
    )
    actor = Actor(
        29,
        hidden=(512, 512),
        squash=False,
        layer_norm=False,
        zero_output=False,
    )
    compiled_core = prepare_compiled_estimator_core(
        actor_params,
        actor.apply,
        env,
        normalizer=normalizer,
        normalizer_state=normalizer_state,
        initial_states=initial_states,
        sigma=contract.sigma,
        gamma=contract.gamma,
        pathwise_clip_norm=contract.pathwise_clip_norm,
    )
    return PreparedE064Estimator(
        contract=contract,
        actor_params=actor_params,
        normalizer_state=normalizer_state,
        initial_states=initial_states,
        checkpoint_path=checkpoint_path,
        hparams=hparams,
        env=env,
        normalizer=normalizer,
        actor_apply=actor.apply,
        compiled_core=compiled_core,
    )


def estimate_e064_shared_gradients(
    checkpoint_state: Any,
    hparams: Mapping[str, Any],
    env: Any,
    *,
    checkpoint_path: str | Path,
    normalizer: Normalizer,
    action_noise: jax.Array,
) -> ExactE064AuditResult:
    """One-shot convenience wrapper; use prepare for four-shard reuse."""

    prepared = prepare_e064_estimator_engine(
        checkpoint_state,
        hparams,
        checkpoint_path=checkpoint_path,
        env=env,
        normalizer=normalizer,
    )
    return prepared(action_noise)


def pytree_shape_signature(tree: PyTree) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Returns a deterministic full leaf path/shape/dtype signature."""

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    signature = []
    for path, leaf in leaves_with_paths:
        array = np.asarray(jax.device_get(leaf))
        signature.append(
            (jax.tree_util.keystr(path), tuple(array.shape), array.dtype.str)
        )
    return tuple(signature)


def stable_pytree_sha256(tree: PyTree) -> str:
    """Hashes PyTree structure, paths, shapes, dtypes, and canonical bytes."""

    leaves_with_paths, tree_definition = jax.tree_util.tree_flatten_with_path(tree)
    digest = sha256()
    digest.update(b"open-diffloco-g1-gradient-audit-pytree-v3\0")
    digest.update(_stable_treedef_schema(tree_definition))
    digest.update(len(leaves_with_paths).to_bytes(8, "little"))
    for path, leaf in leaves_with_paths:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot receive a stable identity hash")
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        for component in (
            jax.tree_util.keystr(path),
            canonical.dtype.str,
            repr(tuple(canonical.shape)),
        ):
            encoded = component.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        payload = canonical.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _qualified_type_name(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _stable_aux_schema(value: Any) -> bytes:
    """Encodes static PyTree metadata without process-specific object reprs."""

    if value is None:
        return b"none"
    if isinstance(value, type):
        return b"type:" + _qualified_type_name(value).encode()
    if isinstance(value, bool):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, int):
        return b"int:" + str(value).encode()
    if isinstance(value, float):
        return b"float:" + value.hex().encode()
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"str:" + len(encoded).to_bytes(8, "little") + encoded
    if isinstance(value, bytes):
        return b"bytes:" + len(value).to_bytes(8, "little") + value
    if isinstance(value, tuple):
        parts = [_stable_aux_schema(item) for item in value]
        return b"tuple:" + b"".join(
            len(part).to_bytes(8, "little") + part for part in parts
        )
    if isinstance(value, list):
        parts = [_stable_aux_schema(item) for item in value]
        return b"list:" + b"".join(
            len(part).to_bytes(8, "little") + part for part in parts
        )
    if isinstance(value, Mapping):
        parts = sorted(
            (
                _stable_aux_schema(key),
                _stable_aux_schema(item),
            )
            for key, item in value.items()
        )
        return b"mapping:" + b"".join(
            len(key).to_bytes(8, "little")
            + key
            + len(item).to_bytes(8, "little")
            + item
            for key, item in parts
        )
    if isinstance(value, np.generic):
        return _stable_aux_schema(value.item())
    if isinstance(value, np.dtype):
        return b"dtype:" + value.str.encode()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be stable PyTree metadata")
        dtype = value.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(value.astype(dtype, copy=False))
        return (
            b"ndarray:"
            + canonical.dtype.str.encode()
            + b":"
            + repr(tuple(canonical.shape)).encode()
            + b":"
            + canonical.tobytes(order="C")
        )
    wrapped_array = getattr(value, "array", None)
    wrapper_type = _qualified_type_name(type(value))
    if (
        wrapper_type == "mujoco.mjx._src.dataclasses._NumPyArrayHashWrapper"
        and getattr(type(value), "__slots__", None) == ("_hash_key", "array")
        and isinstance(wrapped_array, np.ndarray)
    ):
        return (
            b"array-wrapper:"
            + wrapper_type.encode()
            + b":"
            + _stable_aux_schema(wrapped_array)
            + b":"
            + _stable_aux_schema(value._hash_key)
        )
    raise TypeError(
        "unsupported static PyTree metadata type: "
        + _qualified_type_name(type(value))
    )


def _stable_treedef_schema(tree_definition: Any) -> bytes:
    node_data = tree_definition.node_data()
    if node_data is None:
        return b"leaf"
    node_type, aux_data = node_data
    children = tree_definition.children()
    header = (
        b"node:"
        + _qualified_type_name(node_type).encode()
        + b":"
        + str(len(children)).encode()
        + b":"
        + _stable_aux_schema(aux_data)
    )
    return header + b"".join(
        len(child_schema).to_bytes(8, "little") + child_schema
        for child_schema in map(_stable_treedef_schema, children)
    )


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Returns the SHA-256 of an immutable checkpoint or other artifact."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_mapping_sha256(mapping: Mapping[str, Any]) -> str:
    """Hashes a JSON-compatible complete hparams/configuration mapping."""

    try:
        payload = json.dumps(
            dict(mapping),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("identity mapping must be finite and JSON-compatible") from error
    return sha256(payload).hexdigest()


def identity_receipt(
    *,
    checkpoint_sha256: str,
    hparams: Mapping[str, Any],
    gamma: float,
    sigma: float,
    pathwise_clip_norm: float,
    actor_params: PyTree,
    normalizer_state: PyTree,
    initial_state: PyTree,
    action_noise: jax.Array,
    trajectory: SharedTrajectory,
    objective_values: jax.Array,
    returns_to_go: jax.Array,
    score_losses: jax.Array,
    independent_means: jax.Array | None = None,
    independent_objective_values: jax.Array | None = None,
    independent_score_losses: jax.Array | None = None,
    numeric_equivalence_evidence: Mapping[str, Any] | None = None,
    engine_estimator_values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Builds the binding identity receipt shared by estimator consumers."""

    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must contain 64 hexadecimal characters")
    try:
        int(checkpoint_sha256, 16)
    except ValueError as error:
        raise ValueError("checkpoint_sha256 must be hexadecimal") from error
    if independent_means is None:
        independent_means = trajectory.means
    if independent_objective_values is None:
        independent_objective_values = objective_values
    if independent_score_losses is None:
        independent_score_losses = score_losses
    if numeric_equivalence_evidence is None:
        numeric_equivalence_evidence = {
            "mean_max_abs_error": 0.0,
            "mean_rms_error": 0.0,
            "mean_relative_rms_error": 0.0,
            "mean_cosine": 1.0,
            "objective_max_abs_error": 0.0,
            "score_loss_max_abs_error": 0.0,
            "mean_max_abs_gate": _ACTOR_RECONSTRUCTION_MAX_ABSOLUTE_ERROR,
            "mean_rms_gate": _ACTOR_RECONSTRUCTION_MAX_RMS_ERROR,
            "mean_minimum_cosine": _ACTOR_RECONSTRUCTION_MINIMUM_COSINE,
            "objective_gate_ulps": 16,
            "score_loss_gate_ulps": 256,
        }
    if engine_estimator_values is None:
        engine_estimator_values = {
            "objective_values": stable_pytree_sha256(objective_values),
            "score_losses": stable_pytree_sha256(score_losses),
        }
    return {
        "checkpoint": checkpoint_sha256.lower(),
        "hparams": stable_mapping_sha256(hparams),
        "gamma": stable_pytree_sha256(jp.asarray(gamma)),
        "sigma": stable_pytree_sha256(jp.asarray(sigma)),
        "pathwise_clip_norm": stable_pytree_sha256(
            jp.asarray(pathwise_clip_norm)
        ),
        "actor_contract": stable_mapping_sha256(
            {
                "architecture": [154, 512, 512, 29],
                "layer_norm": False,
                "squashed": False,
            }
        ),
        "actor_parameters": stable_pytree_sha256(actor_params),
        "normalizer": stable_pytree_sha256(normalizer_state),
        "initial_state": stable_pytree_sha256(initial_state),
        "noise": stable_pytree_sha256(action_noise),
        "observation_rngs": stable_pytree_sha256(
            trajectory.observation_rngs
        ),
        "raw_observations": stable_pytree_sha256(
            trajectory.raw_observations
        ),
        "observations": stable_pytree_sha256(trajectory.observations),
        "normalized_observations": stable_pytree_sha256(
            trajectory.normalized_observations
        ),
        "means": stable_pytree_sha256(trajectory.means),
        "independent_means": stable_pytree_sha256(independent_means),
        "actions": stable_pytree_sha256(trajectory.actions),
        "rewards": stable_pytree_sha256(trajectory.rewards),
        "dones": stable_pytree_sha256(trajectory.dones),
        "initial_phases": stable_pytree_sha256(trajectory.initial_phase),
        "objective_contract": stable_mapping_sha256(
            {
                "name": "negative_episode_start_discounted_return",
                "horizon_divisor": "trajectory_length",
                "done_discount_reset": True,
                "bootstrap": 0.0,
                "score": "detached_gaussian_likelihood_ratio",
            }
        ),
        "objective_values": stable_pytree_sha256(objective_values),
        "independent_objective_values": stable_pytree_sha256(
            independent_objective_values
        ),
        "returns_to_go": stable_pytree_sha256(returns_to_go),
        "score_losses": stable_pytree_sha256(score_losses),
        "independent_score_losses": stable_pytree_sha256(
            independent_score_losses
        ),
        "numeric_equivalence_evidence": stable_mapping_sha256(
            numeric_equivalence_evidence
        ),
        "numeric_equivalence_metrics": dict(numeric_equivalence_evidence),
        "engine_estimator_values": stable_mapping_sha256(
            engine_estimator_values
        ),
        "trajectory": stable_pytree_sha256(trajectory),
    }


_IDENTITY_RECEIPT_FIELDS = frozenset({
    "checkpoint",
    "hparams",
    "gamma",
    "sigma",
    "pathwise_clip_norm",
    "actor_contract",
    "actor_parameters",
    "normalizer",
    "initial_state",
    "noise",
    "observation_rngs",
    "raw_observations",
    "observations",
    "normalized_observations",
    "means",
    "independent_means",
    "actions",
    "rewards",
    "dones",
    "initial_phases",
    "objective_contract",
    "objective_values",
    "independent_objective_values",
    "returns_to_go",
    "score_losses",
    "independent_score_losses",
    "numeric_equivalence_evidence",
    "numeric_equivalence_metrics",
    "engine_estimator_values",
    "trajectory",
})


_ACTOR_RECONSTRUCTION_MAX_ABSOLUTE_ERROR = 0.005
_ACTOR_RECONSTRUCTION_MAX_RMS_ERROR = 0.0005
_ACTOR_RECONSTRUCTION_MINIMUM_COSINE = 0.9999999
_PATHWISE_CLIPPING_GATE_ULPS = 256
_SCORE_GRADIENT_MAX_RELATIVE_L2_ERROR = 0.005
_SCORE_GRADIENT_MINIMUM_COSINE = 0.99998
_SCORE_MEAN_GRADIENT_MAX_RELATIVE_L2_ERROR = 0.02
_SCORE_MEAN_GRADIENT_MINIMUM_COSINE = 0.9998


def _numeric_equivalence_metrics(actual: Any, expected: Any) -> dict[str, float]:
    """Return finite host metrics for two nonempty same-shaped arrays."""

    actual_array = np.asarray(jax.device_get(actual), dtype=np.float64)
    expected_array = np.asarray(jax.device_get(expected), dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise ValueError("numeric equivalence arrays have different shape")
    if actual_array.size == 0:
        raise ValueError("numeric equivalence arrays must be nonempty")
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise ValueError("numeric equivalence arrays must be finite")
    difference = actual_array - expected_array
    maximum_absolute_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(np.square(difference))))
    expected_rms = float(np.sqrt(np.mean(np.square(expected_array))))
    relative_rms_error = rms_error / max(expected_rms, np.finfo(np.float64).tiny)
    actual_norm = float(np.linalg.norm(actual_array.ravel()))
    expected_norm = float(np.linalg.norm(expected_array.ravel()))
    if actual_norm == 0.0 or expected_norm == 0.0:
        cosine = 1.0 if np.array_equal(actual_array, expected_array) else 0.0
    else:
        cosine = float(
            np.vdot(actual_array.ravel(), expected_array.ravel()).real
            / (actual_norm * expected_norm)
        )
    metrics = {
        "maximum_absolute_error": maximum_absolute_error,
        "rms_error": rms_error,
        "relative_rms_error": relative_rms_error,
        "cosine": cosine,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("numeric equivalence metrics must be finite")
    return metrics


def _validate_actor_reconstruction(actual: Any, expected: Any) -> dict[str, float]:
    """Validate standalone actor arithmetic at a functional, not ULP, scale.

    The actor fused into the 64-by-48 differentiable MJX graph and the same
    float32 actor compiled alone can use different reduction orders on GPU.
    The RMS bound is five percent of the frozen 0.01 output-RMS audit step; the
    pointwise bound is half that functional step and five percent of exploration
    sigma 0.1.  Exact identity between the two in-engine estimator consumers is
    enforced separately.
    """

    try:
        metrics = _numeric_equivalence_metrics(actual, expected)
    except ValueError as error:
        raise ValueError(f"independent actor reconstruction {error}") from error
    if (
        metrics["maximum_absolute_error"]
        > _ACTOR_RECONSTRUCTION_MAX_ABSOLUTE_ERROR
    ):
        raise ValueError(
            "independent actor reconstruction maximum absolute error exceeds gate"
        )
    if metrics["rms_error"] > _ACTOR_RECONSTRUCTION_MAX_RMS_ERROR:
        raise ValueError("independent actor reconstruction RMS error exceeds gate")
    if metrics["cosine"] < _ACTOR_RECONSTRUCTION_MINIMUM_COSINE:
        raise ValueError("independent actor reconstruction cosine is below gate")
    return metrics


def _numeric_pytree_equivalence_metrics(
    actual: PyTree,
    expected: PyTree,
    *,
    label: str,
) -> dict[str, float]:
    """Returns finite whole-tree geometry without concatenating large leaves."""

    actual_leaves, actual_definition = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_definition = jax.tree_util.tree_flatten(expected)
    if actual_definition != expected_definition or len(actual_leaves) != len(
        expected_leaves
    ):
        raise ValueError(f"{label} have different tree structure")
    if not actual_leaves:
        raise ValueError(f"{label} must not be empty")

    maximum_absolute_error = 0.0
    difference_squared = 0.0
    actual_squared = 0.0
    expected_squared = 0.0
    inner_product = 0.0
    element_count = 0
    for actual_leaf, expected_leaf in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        actual_source = np.asarray(jax.device_get(actual_leaf))
        expected_source = np.asarray(jax.device_get(expected_leaf))
        if (
            actual_source.shape != expected_source.shape
            or actual_source.dtype != expected_source.dtype
            or actual_source.size == 0
        ):
            raise ValueError(f"{label} have incompatible leaves")
        actual_array = actual_source.astype(np.float64)
        expected_array = expected_source.astype(np.float64)
        if not np.isfinite(actual_array).all() or not np.isfinite(
            expected_array
        ).all():
            raise ValueError(f"{label} must be finite")
        difference = actual_array - expected_array
        maximum_absolute_error = max(
            maximum_absolute_error, float(np.max(np.abs(difference)))
        )
        difference_squared += float(np.vdot(difference, difference).real)
        actual_squared += float(np.vdot(actual_array, actual_array).real)
        expected_squared += float(np.vdot(expected_array, expected_array).real)
        inner_product += float(np.vdot(actual_array, expected_array).real)
        element_count += actual_array.size

    actual_norm = float(np.sqrt(actual_squared))
    expected_norm = float(np.sqrt(expected_squared))
    difference_norm = float(np.sqrt(difference_squared))
    if actual_norm == 0.0 or expected_norm == 0.0:
        cosine = 1.0 if difference_norm == 0.0 else 0.0
    else:
        cosine = inner_product / (actual_norm * expected_norm)
    metrics = {
        "maximum_absolute_error": maximum_absolute_error,
        "rms_error": float(np.sqrt(difference_squared / element_count)),
        "relative_l2_error": difference_norm
        / max(expected_norm, np.finfo(np.float64).tiny),
        "cosine": float(cosine),
        "actual_norm": actual_norm,
        "expected_norm": expected_norm,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"{label} metrics must be finite")
    return metrics


def _validate_score_gradient_reconstruction(
    actual: PyTree,
    expected: PyTree,
) -> dict[str, float]:
    """Validates independent score gradients at measured GPU fusion bounds."""

    full = _numeric_pytree_equivalence_metrics(
        actual, expected, label="score gradients"
    )
    actual_mean = jax.tree_util.tree_map(lambda leaf: jp.mean(leaf, axis=0), actual)
    expected_mean = jax.tree_util.tree_map(
        lambda leaf: jp.mean(leaf, axis=0), expected
    )
    mean = _numeric_pytree_equivalence_metrics(
        actual_mean, expected_mean, label="score mean gradients"
    )
    if (
        full["relative_l2_error"] > _SCORE_GRADIENT_MAX_RELATIVE_L2_ERROR
        or full["cosine"] < _SCORE_GRADIENT_MINIMUM_COSINE
    ):
        raise ValueError("score gradients fail independent functional gate")
    if (
        mean["relative_l2_error"]
        > _SCORE_MEAN_GRADIENT_MAX_RELATIVE_L2_ERROR
        or mean["cosine"] < _SCORE_MEAN_GRADIENT_MINIMUM_COSINE
    ):
        raise ValueError("score gradients fail aggregate-mean functional gate")
    return {
        "score_gradient_max_abs_error": full["maximum_absolute_error"],
        "score_gradient_rms_error": full["rms_error"],
        "score_gradient_relative_l2_error": full["relative_l2_error"],
        "score_gradient_cosine": full["cosine"],
        "score_gradient_actual_norm": full["actual_norm"],
        "score_gradient_expected_norm": full["expected_norm"],
        "score_mean_gradient_max_abs_error": mean["maximum_absolute_error"],
        "score_mean_gradient_rms_error": mean["rms_error"],
        "score_mean_gradient_relative_l2_error": mean["relative_l2_error"],
        "score_mean_gradient_cosine": mean["cosine"],
        "score_mean_gradient_actual_norm": mean["actual_norm"],
        "score_mean_gradient_expected_norm": mean["expected_norm"],
    }


def build_and_validate_estimator_receipts(
    *,
    checkpoint_sha256: str,
    hparams: Mapping[str, Any],
    actor_params: PyTree,
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    normalizer_state: PyTree,
    initial_state: PyTree,
    action_noise: jax.Array,
    result: SharedGradientResult,
    gamma: float,
    sigma: float,
    pathwise_clip_norm: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently receipts both estimator consumers and fails on drift."""

    def exact_equal(left, right, *, label):
        left_array = np.asarray(jax.device_get(left))
        right_array = np.asarray(jax.device_get(right))
        if (
            left_array.shape != right_array.shape
            or left_array.dtype != right_array.dtype
            or not np.array_equal(left_array, right_array)
        ):
            raise ValueError(f"estimator identity mismatch for {label}")

    def tree_ulp_gate(actual, expected, *, ulps, label):
        actual_leaves, actual_definition = jax.tree_util.tree_flatten(actual)
        expected_leaves, expected_definition = jax.tree_util.tree_flatten(expected)
        if actual_definition != expected_definition or len(actual_leaves) != len(
            expected_leaves
        ):
            raise ValueError(f"estimator identity mismatch for {label}")
        return max(
            (
                ulp_gate(
                    actual_leaf,
                    expected_leaf,
                    source_dtype=np.asarray(jax.device_get(actual_leaf)).dtype,
                    ulps=ulps,
                    label=label,
                )
                for actual_leaf, expected_leaf in zip(
                    actual_leaves, expected_leaves, strict=True
                )
            ),
            default=0.0,
        )

    exact_equal(result.trajectory.noise, action_noise, label="noise")
    exact_equal(result.score_trajectory.noise, action_noise, label="noise")
    for field_name in SharedTrajectory._fields:
        exact_equal(
            getattr(result.trajectory, field_name),
            getattr(result.score_trajectory, field_name),
            label=f"engine trajectory {field_name}",
        )
    exact_equal(
        result.trajectory.means,
        result.score_means,
        label="engine means",
    )
    exact_equal(
        result.losses,
        result.score_objectives,
        label="engine objective_values",
    )
    exact_equal(
        result.pathwise_score_losses,
        result.score_losses,
        label="engine score_losses",
    )
    expected_actions = result.trajectory.means + jp.asarray(
        sigma, dtype=result.trajectory.means.dtype
    ) * result.trajectory.noise.astype(result.trajectory.means.dtype)
    exact_equal(
        result.trajectory.actions, expected_actions, label="action equation"
    )

    def independently_reconstruct(trajectory):
        raw_means = jax.vmap(
            lambda observations: jax.vmap(
                lambda observation: actor_apply(actor_params, observation)
            )(observations)
        )(jax.lax.stop_gradient(trajectory.normalized_observations))
        means = raw_means.astype(trajectory.means.dtype)
        returns = jax.vmap(
            lambda rewards, dones: discounted_return_to_go(
                rewards, dones, gamma=gamma
            )
        )(trajectory.rewards, trajectory.dones)
        objectives = jax.vmap(
            lambda rewards, dones: pathwise_negative_objective(
                rewards, dones, gamma=gamma
            )
        )(trajectory.rewards, trajectory.dones)
        losses = jax.vmap(
            lambda canonical_means, actions, candidate_returns: (
                detached_gaussian_score_loss(
                    canonical_means,
                    actions,
                    candidate_returns,
                    std=sigma,
                )
            )
        )(
            jax.lax.stop_gradient(trajectory.means),
            trajectory.actions,
            returns,
        )
        return raw_means, means, returns, objectives, losses

    def independently_reconstruct_score_gradients(trajectory):
        def score_loss(params, data):
            rebuilt_means = jax.vmap(
                lambda observation: actor_apply(params, observation)
            )(
                jax.lax.stop_gradient(data.normalized_observations)
            ).astype(data.means.dtype)
            # Keep the exact in-engine primal while independently rebuilding
            # the actor derivative in a small standalone graph.
            means = rebuilt_means + jax.lax.stop_gradient(
                data.means - rebuilt_means
            )
            returns = discounted_return_to_go(
                data.rewards, data.dones, gamma=gamma
            )
            return detached_gaussian_score_loss(
                means,
                jax.lax.stop_gradient(data.actions),
                jax.lax.stop_gradient(returns),
                std=sigma,
            )

        return jax.vmap(jax.grad(score_loss), in_axes=(None, 0))(
            actor_params, trajectory
        )

    path_rebuilt = independently_reconstruct(result.trajectory)
    score_rebuilt = independently_reconstruct(result.score_trajectory)
    (
        path_raw_means,
        path_means,
        path_returns,
        path_objectives,
        path_losses,
    ) = path_rebuilt
    (
        score_raw_means,
        score_means,
        score_returns,
        score_objectives,
        score_losses,
    ) = score_rebuilt

    def ulp_gate(actual, expected, *, source_dtype, ulps, label):
        actual_array = np.asarray(jax.device_get(actual))
        expected_array = np.asarray(jax.device_get(expected))
        if (
            actual_array.shape != expected_array.shape
            or actual_array.dtype != expected_array.dtype
        ):
            raise ValueError(f"estimator identity mismatch for {label}")
        epsilon = np.finfo(np.dtype(source_dtype)).eps
        errors = np.abs(actual_array.astype(np.float64) - expected_array)
        scales = np.maximum(1.0, np.abs(expected_array.astype(np.float64)))
        if not np.all(errors <= ulps * epsilon * scales):
            raise ValueError(f"estimator identity mismatch for {label}")
        return float(np.max(errors, initial=0.0))

    expected_pathwise_effective, expected_pathwise_norms, expected_clip_scales = (
        _pathwise_effective_per_environment(
            result.pathwise_raw_gradients,
            max_norm=pathwise_clip_norm,
        )
    )
    clipping_errors = (
        tree_ulp_gate(
            result.pathwise_effective_gradients,
            expected_pathwise_effective,
            ulps=_PATHWISE_CLIPPING_GATE_ULPS,
            label="pathwise clipping effective gradients",
        ),
        tree_ulp_gate(
            result.pathwise_raw_norms,
            expected_pathwise_norms,
            ulps=_PATHWISE_CLIPPING_GATE_ULPS,
            label="pathwise clipping raw norms",
        ),
        tree_ulp_gate(
            result.pathwise_clip_scales,
            expected_clip_scales,
            ulps=_PATHWISE_CLIPPING_GATE_ULPS,
            label="pathwise clipping scales",
        ),
    )
    mean_metrics = (
        _validate_actor_reconstruction(result.trajectory.means, path_means),
        _validate_actor_reconstruction(result.score_means, score_means),
    )
    exact_equal(
        result.pathwise_returns_to_go,
        path_returns,
        label="returns_to_go",
    )
    exact_equal(result.score_returns_to_go, score_returns, label="returns_to_go")
    objective_errors = (
        ulp_gate(
            result.losses,
            path_objectives,
            source_dtype=np.asarray(
                jax.device_get(result.trajectory.rewards)
            ).dtype,
            ulps=16,
            label="independent objective_values",
        ),
        ulp_gate(
            result.score_objectives,
            score_objectives,
            source_dtype=np.asarray(
                jax.device_get(result.score_trajectory.rewards)
            ).dtype,
            ulps=16,
            label="independent objective_values",
        ),
    )
    loss_errors = (
        ulp_gate(
            result.pathwise_score_losses,
            path_losses,
            source_dtype=np.asarray(jax.device_get(path_raw_means)).dtype,
            ulps=256,
            label="independent score_losses",
        ),
        ulp_gate(
            result.score_losses,
            score_losses,
            source_dtype=np.asarray(jax.device_get(score_raw_means)).dtype,
            ulps=256,
            label="independent score_losses",
        ),
    )
    independent_score_gradients = independently_reconstruct_score_gradients(
        result.score_trajectory
    )
    score_gradient_evidence = _validate_score_gradient_reconstruction(
        result.score_gradients,
        independent_score_gradients,
    )
    evidence = {
        "mean_max_abs_error": max(
            metrics["maximum_absolute_error"] for metrics in mean_metrics
        ),
        "mean_rms_error": max(metrics["rms_error"] for metrics in mean_metrics),
        "mean_relative_rms_error": max(
            metrics["relative_rms_error"] for metrics in mean_metrics
        ),
        "mean_cosine": min(metrics["cosine"] for metrics in mean_metrics),
        "objective_max_abs_error": max(objective_errors),
        "score_loss_max_abs_error": max(loss_errors),
        "pathwise_clipping_max_abs_error": max(clipping_errors),
        "mean_max_abs_gate": _ACTOR_RECONSTRUCTION_MAX_ABSOLUTE_ERROR,
        "mean_rms_gate": _ACTOR_RECONSTRUCTION_MAX_RMS_ERROR,
        "mean_minimum_cosine": _ACTOR_RECONSTRUCTION_MINIMUM_COSINE,
        "objective_gate_ulps": 16,
        "score_loss_gate_ulps": 256,
        "pathwise_clipping_gate_ulps": _PATHWISE_CLIPPING_GATE_ULPS,
        "score_gradient_max_relative_l2_gate": (
            _SCORE_GRADIENT_MAX_RELATIVE_L2_ERROR
        ),
        "score_gradient_minimum_cosine": _SCORE_GRADIENT_MINIMUM_COSINE,
        "score_mean_gradient_max_relative_l2_gate": (
            _SCORE_MEAN_GRADIENT_MAX_RELATIVE_L2_ERROR
        ),
        "score_mean_gradient_minimum_cosine": (
            _SCORE_MEAN_GRADIENT_MINIMUM_COSINE
        ),
        **score_gradient_evidence,
    }
    engine_values = {
        "pathwise_objectives": stable_pytree_sha256(result.losses),
        "score_objectives": stable_pytree_sha256(result.score_objectives),
        "pathwise_score_losses": stable_pytree_sha256(
            result.pathwise_score_losses
        ),
        "score_score_losses": stable_pytree_sha256(result.score_losses),
        "pathwise_raw_gradients": stable_pytree_sha256(
            result.pathwise_raw_gradients
        ),
        "pathwise_effective_gradients": stable_pytree_sha256(
            result.pathwise_effective_gradients
        ),
        "pathwise_raw_norms": stable_pytree_sha256(result.pathwise_raw_norms),
        "pathwise_clip_scales": stable_pytree_sha256(
            result.pathwise_clip_scales
        ),
        "score_gradients": stable_pytree_sha256(result.score_gradients),
        "independent_score_gradients": stable_pytree_sha256(
            independent_score_gradients
        ),
    }
    pathwise_receipt = identity_receipt(
        checkpoint_sha256=checkpoint_sha256,
        hparams=hparams,
        gamma=gamma,
        sigma=sigma,
        pathwise_clip_norm=pathwise_clip_norm,
        actor_params=actor_params,
        normalizer_state=normalizer_state,
        initial_state=initial_state,
        action_noise=action_noise,
        trajectory=result.trajectory,
        objective_values=path_objectives,
        returns_to_go=path_returns,
        score_losses=path_losses,
        independent_means=path_means,
        independent_objective_values=path_objectives,
        independent_score_losses=path_losses,
        numeric_equivalence_evidence=evidence,
        engine_estimator_values=engine_values,
    )

    score_trajectory = result.score_trajectory._replace(
        means=result.score_means
    )
    score_receipt = identity_receipt(
        checkpoint_sha256=checkpoint_sha256,
        hparams=hparams,
        gamma=gamma,
        sigma=sigma,
        pathwise_clip_norm=pathwise_clip_norm,
        actor_params=actor_params,
        normalizer_state=normalizer_state,
        initial_state=initial_state,
        action_noise=action_noise,
        trajectory=score_trajectory,
        objective_values=score_objectives,
        returns_to_go=score_returns,
        score_losses=score_losses,
        independent_means=score_means,
        independent_objective_values=score_objectives,
        independent_score_losses=score_losses,
        numeric_equivalence_evidence=evidence,
        engine_estimator_values=engine_values,
    )
    assert_matching_identity_receipts(pathwise_receipt, score_receipt)
    return pathwise_receipt, score_receipt


def assert_matching_identity_receipts(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    """Fails closed when any estimator-consumer identity differs."""

    if set(left) != _IDENTITY_RECEIPT_FIELDS:
        missing = sorted(_IDENTITY_RECEIPT_FIELDS.symmetric_difference(left))
        raise ValueError(f"left identity receipt is incomplete: {missing}")
    if set(right) != _IDENTITY_RECEIPT_FIELDS:
        missing = sorted(_IDENTITY_RECEIPT_FIELDS.symmetric_difference(right))
        raise ValueError(f"right identity receipt is incomplete: {missing}")
    mismatches = [key for key in sorted(left) if left[key] != right[key]]
    if mismatches:
        raise ValueError(
            "estimator identity mismatch for " + ", ".join(mismatches)
        )


_EXPECTED_ACTOR_SHAPES = {
    "['params']['Dense_0']['bias']": (512,),
    "['params']['Dense_0']['kernel']": (154, 512),
    "['params']['Dense_1']['bias']": (512,),
    "['params']['Dense_1']['kernel']": (512, 512),
    "['params']['Dense_2']['bias']": (29,),
    "['params']['Dense_2']['kernel']": (512, 29),
}


def _shape_map(tree: PyTree) -> dict[str, tuple[int, ...]]:
    return {path: shape for path, shape, _dtype in pytree_shape_signature(tree)}


def _field(state: Any, name: str) -> Any:
    return state[name] if isinstance(state, Mapping) else getattr(state, name)


def _normalizer_identity_tree(normalizer: Any) -> dict[str, Any]:
    return {
        "mean": _field(normalizer, "mean"),
        "var": _field(normalizer, "var"),
        "count": _field(normalizer, "count"),
    }


def _state_shape(
    state_shapes: Mapping[str, tuple[int, ...]], *paths: str
) -> tuple[int, ...] | None:
    for path in paths:
        if path in state_shapes:
            return state_shapes[path]
    return None


def validate_e064_checkpoint_shapes(
    checkpoint_state: Any,
    *,
    expected_initial_state_signature: tuple | None = None,
) -> ValidatedCheckpointShapes:
    """Checks exact actor/normalizer shapes and carried population geometry."""

    actor_params = checkpoint_state.actor_params
    actor_signature = pytree_shape_signature(actor_params)
    if _shape_map(actor_params) != _EXPECTED_ACTOR_SHAPES:
        raise ValueError(
            "actor parameter shapes must be exactly 154->512->512->29 with no LayerNorm"
        )
    if any(
        not np.issubdtype(np.dtype(dtype), np.floating)
        for _, _, dtype in actor_signature
    ):
        raise ValueError("actor parameter leaves must all be floating-point arrays")

    normalizer_tree = _normalizer_identity_tree(checkpoint_state.normalizer)
    normalizer_signature = pytree_shape_signature(normalizer_tree)
    if _shape_map(normalizer_tree) != {
        "['count']": (),
        "['mean']": (154,),
        "['var']": (154,),
    }:
        raise ValueError(
            "actor normalizer shapes must be mean/var (154,) and scalar count"
        )

    initial_state = checkpoint_state.env_state
    initial_state_signature = pytree_shape_signature(initial_state)
    if (
        expected_initial_state_signature is not None
        and initial_state_signature != tuple(expected_initial_state_signature)
    ):
        raise ValueError(
            "initial-state shape signature does not match the frozen checkpoint"
        )
    state_shapes = _shape_map(initial_state)
    for path, shape in state_shapes.items():
        if not shape or shape[0] != 64:
            raise ValueError(
                f"initial-state shape {path} must carry exactly 64 environments"
            )
    required_shapes = (
        (("['obs']", ".obs"), (64, 154), "['obs']"),
        (("['reward']", ".reward"), (64,), "['reward']"),
        (("['done']", ".done"), (64,), "['done']"),
        (("['info']['rng']", ".info['rng']"), (64, 2), "info.rng"),
        (("['info']['phase']", ".info['phase']"), (64,), "info.phase"),
    )
    for paths, expected_shape, label in required_shapes:
        if _state_shape(state_shapes, *paths) != expected_shape:
            raise ValueError(
                f"initial-state shape {label} must be exactly {expected_shape}"
            )
    return ValidatedCheckpointShapes(
        population=64,
        horizon=48,
        sigma=0.1,
        actor_parameter_signature=actor_signature,
        normalizer_signature=normalizer_signature,
        initial_state_signature=initial_state_signature,
    )


def validate_e064_checkpoint_contract(
    checkpoint_state: Any,
    hparams: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
) -> ValidatedE064Contract:
    """Host-validates the literal registered E064 artifact and state."""

    if (
        set(hparams) != set(E064_FROZEN_HPARAMS)
        or stable_mapping_sha256(hparams) != E064_HPARAMS_SHA256
    ):
        changed = sorted(
            key
            for key in set(hparams).union(E064_FROZEN_HPARAMS)
            if hparams.get(key) != E064_FROZEN_HPARAMS.get(key)
        )
        label = changed[0] if changed else "mapping"
        raise ValueError(
            f"{label}: hparams must equal the literal frozen E064 hparams"
        )
    if sha256_file(checkpoint_path) != E064_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not equal literal E064 checkpoint")

    shapes = validate_e064_checkpoint_shapes(checkpoint_state)
    actor_params = checkpoint_state.actor_params
    normalizer_tree = _normalizer_identity_tree(checkpoint_state.normalizer)
    initial_state = checkpoint_state.env_state
    if stable_pytree_sha256(actor_params) != E064_ACTOR_PARAMETERS_SHA256:
        raise ValueError("actor parameters do not equal literal E064 actor")
    if stable_pytree_sha256(normalizer_tree) != E064_NORMALIZER_SHA256:
        raise ValueError("normalizer does not equal literal E064 normalizer")
    if stable_pytree_sha256(initial_state) != E064_INITIAL_STATE_SHA256:
        raise ValueError("initial state does not equal literal E064 carried state")

    return ValidatedE064Contract(
        population=64,
        horizon=48,
        sigma=0.1,
        gamma=0.99,
        pathwise_clip_norm=1.0,
        hparams_sha256=E064_HPARAMS_SHA256,
        actor_parameters_sha256=E064_ACTOR_PARAMETERS_SHA256,
        normalizer_sha256=E064_NORMALIZER_SHA256,
        initial_state_sha256=E064_INITIAL_STATE_SHA256,
        env_variant=str(E064_FROZEN_HPARAMS["env_variant"]),
        reference_sha256=E064_REFERENCE_SHA256,
        actor_parameter_signature=shapes.actor_parameter_signature,
        normalizer_signature=shapes.normalizer_signature,
        initial_state_signature=shapes.initial_state_signature,
    )
