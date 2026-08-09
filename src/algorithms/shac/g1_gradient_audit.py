"""Exact shared-rollout runtime for the frozen E064 G1 gradient audit.

This module is intentionally separate from the production SHAC trainer.  It
materializes one trajectory and derives both the reparameterized pathwise and
detached Gaussian score gradients from that trajectory.
"""

from __future__ import annotations

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
from src.core.networks import Actor

PyTree = Any


# A caller must pass this complete audit-runtime mapping, not an unchecked
# subset of a training hparams file.  Fields that are non-causal once their
# parent switch is off (for example terrain bump magnitude) are deliberately
# absent.
E064_RUNTIME_HPARAMS: Mapping[str, Any] = MappingProxyType({
    "algorithm": "shac",
    "env_variant": "g1_tracking_rmr_50hz_validated",
    "unroll_length": 48,
    "num_envs": 64,
    "effective_num_envs": 64,
    "gradient_accumulation_steps": 1,
    "gamma": 0.99,
    "action_noise_std_end": 0.1,
    "actor_per_env_grad_clip": 1.0,
    "actor_bootstrap_scale": 0.0,
    "actor_bootstrap_delay_steps": 0,
    "squash_actor_actions": False,
    "action_scale": 1.0,
    "friction_range": (1.0, 1.0),
    "mass_range": (1.0, 1.0),
    "effort_limit_scale": 1.0,
    "kp_range": (1.0, 1.0),
    "kd_range": (1.0, 1.0),
    "com_offset_range": (0.0, 0.0, 0.0),
    "push_velocity_range": (0.0, 0.0),
    "terrain": False,
    "zero_difficulty_frac": 1.0,
    "actor_history_len": 1,
    "actor_hidden": (512, 512),
    "actor_layer_norm": False,
    "actor_zero_output": False,
    "actor_kind": "flax",
    "source_actor_policy": False,
    "residual_action_scale": 0.0,
})

# Exact key schema written beside the immutable E064 checkpoint.  Callers pass
# the complete mapping and its independently preregistered expectation; this
# schema prevents a hand-picked safety subset from being treated as complete.
E064_REQUIRED_HPARAM_KEYS = frozenset({
    "action_noise_std_end",
    "action_noise_std_start",
    "action_scale",
    "actor_bootstrap_delay_steps",
    "actor_bootstrap_scale",
    "actor_hidden",
    "actor_history_len",
    "actor_kind",
    "actor_layer_norm",
    "actor_lr",
    "actor_per_env_grad_clip",
    "actor_zero_output",
    "algorithm",
    "best_reward",
    "cmd_ctrl_interval_range",
    "cmd_vel_x_range",
    "cmd_vel_y_range",
    "cmd_yaw_rate_range",
    "cmd_zero_prob",
    "com_offset_range",
    "critic_iterations",
    "critic_lr",
    "critic_per_env_grad_clip",
    "curriculum_grace",
    "curriculum_steps",
    "differentiate_source_feedback",
    "effective_num_envs",
    "effort_limit_scale",
    "env_variant",
    "friction_range",
    "gae_lambda",
    "gamma",
    "gradient_accumulation_steps",
    "kd_range",
    "kp_range",
    "mass_range",
    "max_episode_length",
    "num_envs",
    "push_interval_s",
    "push_velocity_range",
    "reference_fps",
    "reference_path",
    "reference_sha256",
    "reference_states",
    "reference_stride",
    "reference_transitions",
    "residual_action_scale",
    "seed",
    "source_actor_policy",
    "squash_actor_actions",
    "steps_per_actor_update",
    "target_update_rate",
    "termination_margin_weight",
    "terrain",
    "terrain_bump_decay",
    "terrain_bump_std",
    "terrain_flat_prob",
    "terrain_slope_max",
    "total_steps",
    "unroll_length",
    "xml_path",
    "zero_difficulty_frac",
})


class SharedTrajectory(NamedTuple):
    """Materialized stochastic inputs and transition data for one population."""

    noise: jax.Array
    observation_rngs: jax.Array
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
        # E064's validated observation transform is exactly identity.  Keep the
        # otherwise-unused split so the carried environment RNG advances in
        # precisely the same order as production.
        actor_observation = state.obs
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

    def score_loss(params, data):
        stopped_normalized_observations = jax.lax.stop_gradient(
            data.normalized_observations
        )
        means = jax.vmap(lambda obs: actor_apply(params, obs))(
            stopped_normalized_observations
        )
        returns = discounted_return_to_go(
            data.rewards,
            data.dones,
            gamma=gamma,
        )
        return detached_gaussian_score_loss(
            means,
            data.actions,
            returns,
            std=sigma,
        )

    score_gradients = jax.vmap(
        jax.grad(score_loss), in_axes=(None, 0)
    )(actor_params, score_trajectory)
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
        score_gradients=score_gradients,
    )


def estimate_e064_shared_gradients(
    contract: ValidatedE064Contract,
    actor_params: PyTree,
    env: Any,
    *,
    normalizer: Any,
    normalizer_state: Any,
    initial_states: PyTree,
    action_noise: jax.Array,
) -> SharedGradientResult:
    """Runs the estimator only after rebinding every frozen E064 input."""

    if action_noise.shape != (64, 48, 29):
        raise ValueError("E064 action_noise must have exact shape (64, 48, 29)")
    if (contract.population, contract.horizon, contract.sigma) != (64, 48, 0.1):
        raise ValueError("validated contract must freeze population 64, horizon 48, sigma 0.1")
    if (contract.gamma, contract.pathwise_clip_norm) != (0.99, 1.0):
        raise ValueError("validated contract must freeze gamma 0.99 and clip norm 1.0")
    if pytree_shape_signature(actor_params) != contract.actor_parameter_signature:
        raise ValueError("execution actor parameter shapes differ from validation")
    if stable_pytree_sha256(actor_params) != contract.actor_parameters_sha256:
        raise ValueError("execution actor parameters differ from validation")
    normalizer_tree = _normalizer_identity_tree(normalizer_state)
    if pytree_shape_signature(normalizer_tree) != contract.normalizer_signature:
        raise ValueError("execution actor normalizer shapes differ from validation")
    if stable_pytree_sha256(normalizer_tree) != contract.normalizer_sha256:
        raise ValueError("execution actor normalizer differs from validation")
    if pytree_shape_signature(initial_states) != contract.initial_state_signature:
        raise ValueError("execution carried-state shapes differ from validation")
    if stable_pytree_sha256(initial_states) != contract.initial_state_sha256:
        raise ValueError("execution carried state differs from validation")
    _require_unbounded_environment(env)
    if getattr(env, "action_dim", None) != 29:
        raise ValueError("E064 environment action_dim must be 29")
    if getattr(env, "actor_obs_dim", None) != 154:
        raise ValueError("E064 environment actor_obs_dim must be 154")
    if getattr(env, "actor_history_len", None) != 1:
        raise ValueError("E064 environment actor_history_len must be one")
    if getattr(env, "reference_stride", None) != 1:
        raise ValueError("E064 environment reference_stride must be one")
    reference = getattr(env, "reference", None)
    if reference is None or reference.qpos.shape[0] != 500:
        raise ValueError("E064 environment must carry the 500-state reference")
    if sha256_file(env.reference_path) != contract.reference_sha256:
        raise ValueError("E064 environment reference SHA-256 differs from validation")
    model = getattr(env, "mj_model", None)
    if model is None or (model.opt.iterations, model.opt.ls_iterations) != (4, 5):
        raise ValueError("E064 environment solver must be exactly 4/5")
    if getattr(env, "reference_reset_noise_scale", 0.0) != 0.0:
        raise ValueError("E064 environment must disable reference reset corruption")
    actor = Actor(
        29,
        hidden=(512, 512),
        squash=False,
        layer_norm=False,
        zero_output=False,
    )
    return estimate_shared_gradients(
        actor_params,
        actor.apply,
        env,
        normalizer=normalizer,
        normalizer_state=normalizer_state,
        initial_states=initial_states,
        action_noise=action_noise,
        sigma=contract.sigma,
        gamma=contract.gamma,
        pathwise_clip_norm=contract.pathwise_clip_norm,
    )


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
    digest.update(b"open-diffloco-g1-gradient-audit-pytree-v1\0")
    digest.update(str(tree_definition).encode("utf-8"))
    digest.update(b"\0")
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
) -> dict[str, str]:
    """Builds the binding identity receipt shared by estimator consumers."""

    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must contain 64 hexadecimal characters")
    try:
        int(checkpoint_sha256, 16)
    except ValueError as error:
        raise ValueError("checkpoint_sha256 must be hexadecimal") from error
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
        "observations": stable_pytree_sha256(trajectory.observations),
        "normalized_observations": stable_pytree_sha256(
            trajectory.normalized_observations
        ),
        "means": stable_pytree_sha256(trajectory.means),
        "actions": stable_pytree_sha256(trajectory.actions),
        "rewards": stable_pytree_sha256(trajectory.rewards),
        "dones": stable_pytree_sha256(trajectory.dones),
        "initial_phases": stable_pytree_sha256(trajectory.initial_phase),
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
    "observations",
    "normalized_observations",
    "means",
    "actions",
    "rewards",
    "dones",
    "initial_phases",
    "trajectory",
})


def assert_matching_identity_receipts(
    left: Mapping[str, str], right: Mapping[str, str]
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


def _require_equal(hparams: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = hparams.get(key)
    if isinstance(expected, tuple) and isinstance(actual, (tuple, list)):
        matches = tuple(actual) == expected
    else:
        matches = actual == expected
    if key not in hparams or not matches:
        raise ValueError(f"{key} must equal the frozen E064 value {expected!r}")


def validate_e064_checkpoint_contract(
    checkpoint_state: Any,
    hparams: Mapping[str, Any],
    *,
    expected_hparams: Mapping[str, Any],
    expected_initial_state_signature: tuple,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> ValidatedE064Contract:
    """Validates the checkpoint, complete hparams, and carried-state shapes.

    ``expected_hparams`` and ``expected_initial_state_signature`` are the
    preregistered caller-owned contracts.  Equality is exact and the causal
    safety requirements below are additionally enforced, so changing both an
    input and its claimed expectation cannot enable a different audit.
    """

    if dict(hparams) != dict(expected_hparams):
        raise ValueError("hparams do not match the complete frozen hparams mapping")
    missing_hparams = E064_REQUIRED_HPARAM_KEYS.difference(hparams)
    if missing_hparams:
        raise ValueError(
            "complete frozen hparams mapping is missing fields: "
            + ", ".join(sorted(missing_hparams))
        )

    for key, expected in E064_RUNTIME_HPARAMS.items():
        _require_equal(hparams, key, expected)
    for key, value in hparams.items():
        normalized_key = key.lower()
        if "reset" in normalized_key and (
            "noise" in normalized_key or "corrupt" in normalized_key
        ) and value not in (0, 0.0, False, None):
            raise ValueError(f"{key} must disable reset corruption")
        if ("obs" in normalized_key or "observation" in normalized_key) and (
            "noise" in normalized_key or "corrupt" in normalized_key
        ) and value not in (0, 0.0, False, None, "identity"):
            raise ValueError(f"{key} must select identity observation handling")

    if (checkpoint_path is None) != (expected_checkpoint_sha256 is None):
        raise ValueError(
            "checkpoint_path and expected_checkpoint_sha256 must be supplied together"
        )
    if checkpoint_path is not None:
        actual_checkpoint_sha256 = sha256_file(checkpoint_path)
        if actual_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 does not match the frozen contract")

    actor_params = checkpoint_state.actor_params
    actor_signature = pytree_shape_signature(actor_params)
    if _shape_map(actor_params) != _EXPECTED_ACTOR_SHAPES:
        raise ValueError(
            "actor parameter shapes must be exactly 154->512->512->29 with no LayerNorm"
        )
    if any(not np.issubdtype(np.dtype(dtype), np.floating) for _, _, dtype in actor_signature):
        raise ValueError("actor parameter leaves must all be floating-point arrays")

    normalizer = checkpoint_state.normalizer
    normalizer_tree = _normalizer_identity_tree(normalizer)
    normalizer_signature = pytree_shape_signature(normalizer_tree)
    expected_normalizer_shapes = {
        "['count']": (),
        "['mean']": (154,),
        "['var']": (154,),
    }
    if _shape_map(normalizer_tree) != expected_normalizer_shapes:
        raise ValueError("actor normalizer shapes must be mean/var (154,) and scalar count")

    initial_state = checkpoint_state.env_state
    initial_state_signature = pytree_shape_signature(initial_state)
    if initial_state_signature != tuple(expected_initial_state_signature):
        raise ValueError("initial-state shape signature does not match the frozen checkpoint")
    state_shapes = _shape_map(initial_state)
    for path, shape in state_shapes.items():
        if not shape or shape[0] != 64:
            raise ValueError(
                f"initial-state shape {path} must carry exactly 64 environments"
            )
    if _state_shape(state_shapes, "['obs']", ".obs") != (64, 154):
        raise ValueError("initial-state shape ['obs'] must be exactly (64, 154)")
    if _state_shape(state_shapes, "['reward']", ".reward") != (64,):
        raise ValueError("initial-state shape ['reward'] must be exactly (64,)")
    if _state_shape(state_shapes, "['done']", ".done") != (64,):
        raise ValueError("initial-state shape ['done'] must be exactly (64,)")
    if _state_shape(
        state_shapes, "['info']['rng']", ".info['rng']"
    ) != (64, 2):
        raise ValueError("initial-state shape info.rng must be exactly (64, 2)")
    if _state_shape(
        state_shapes, "['info']['phase']", ".info['phase']"
    ) != (64,):
        raise ValueError("initial-state shape info.phase must be exactly (64,)")

    return ValidatedE064Contract(
        population=64,
        horizon=48,
        sigma=0.1,
        gamma=0.99,
        pathwise_clip_norm=1.0,
        hparams_sha256=stable_mapping_sha256(hparams),
        actor_parameters_sha256=stable_pytree_sha256(actor_params),
        normalizer_sha256=stable_pytree_sha256(normalizer_tree),
        initial_state_sha256=stable_pytree_sha256(initial_state),
        env_variant=str(hparams["env_variant"]),
        reference_sha256=str(hparams["reference_sha256"]),
        actor_parameter_signature=actor_signature,
        normalizer_signature=normalizer_signature,
        initial_state_signature=initial_state_signature,
    )
