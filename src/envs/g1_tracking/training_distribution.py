"""Training-only reset distribution for the RMR G1 tracking task."""

from typing import NamedTuple

import jax
import jax.numpy as jp
from mujoco import mjx

from src.core.data_structures import EnvState
from src.envs.g1_tracking.environment import _quat_mul


ACTOR_NOISE_SLICES = (
    (slice(58, 64), 0.05),
    (slice(64, 67), 0.2),
    (slice(67, 96), 0.01),
    (slice(96, 125), 0.5),
)

_ROOT_POSITION_BOUNDS = jp.array([0.02, 0.02, 0.005])
_ROOT_EULER_BOUNDS = jp.array([0.1, 0.1, 0.1])
_ROOT_LINEAR_VELOCITY_BOUNDS = jp.array([0.25, 0.25, 0.1])
_ROOT_ANGULAR_VELOCITY_BOUNDS = jp.array([0.26, 0.26, 0.39])


class PhaseSamplerState(NamedTuple):
    """Failure-count EMA over fixed reference-timeline bins."""

    failed_count: jax.Array


class ResetPerturbations(NamedTuple):
    """Independent RMR reset perturbations in MuJoCo model order."""

    root_position: jax.Array
    root_euler_xyz: jax.Array
    root_linear_velocity: jax.Array
    root_angular_velocity: jax.Array
    joint_position: jax.Array


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bin_count(reference_length: int, control_hz: int = 50) -> int:
    reference_length = _positive_integer(reference_length, "reference_length")
    control_hz = _positive_integer(control_hz, "control_hz")
    return reference_length // control_hz + 1


def init_phase_sampler(
    reference_length: int, control_hz: int = 50
) -> PhaseSamplerState:
    """Creates the initially uniform one-frame RMR phase sampler."""
    return PhaseSamplerState(
        failed_count=jp.zeros(
            _bin_count(reference_length, control_hz), dtype=jp.float32
        )
    )


def phase_sampling_probabilities(
    state: PhaseSamplerState, uniform_ratio: float = 0.5
) -> jax.Array:
    """Returns a literal mixture of uniform and failure distributions."""
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError("uniform_ratio must be in [0, 1]")
    failures = jp.maximum(jp.asarray(state.failed_count), 0.0)
    bin_count = failures.shape[0]
    uniform = jp.full_like(failures, 1.0 / float(bin_count))
    total = jp.sum(failures)
    failure_distribution = jp.where(total > 0.0, failures / total, uniform)
    return uniform_ratio * uniform + (1.0 - uniform_ratio) * failure_distribution


def update_phase_sampler(
    state: PhaseSamplerState,
    phases: jax.Array,
    terminals: jax.Array,
    reference_length: int,
    alpha: float = 0.001,
) -> PhaseSamplerState:
    """Applies the upstream EMA update from newly terminated trajectories."""
    reference_length = _positive_integer(reference_length, "reference_length")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    phases = jp.asarray(phases, dtype=jp.int32)
    terminals = jp.asarray(terminals, dtype=jp.float32)
    if phases.shape != terminals.shape:
        raise ValueError("phases and terminals must have identical shapes")
    bin_count = state.failed_count.shape[0]
    bins = jp.clip(
        phases * bin_count // reference_length,
        min=0,
        max=bin_count - 1,
    )
    failures = jp.bincount(
        bins.reshape(-1),
        weights=terminals.reshape(-1),
        length=bin_count,
    )
    return PhaseSamplerState(
        failed_count=(
            alpha * failures + (1.0 - alpha) * state.failed_count
        )
    )


def sample_training_phase(
    key: jax.Array,
    state: PhaseSamplerState,
    reference_length: int,
    uniform_ratio: float = 0.2,
) -> jax.Array:
    """Samples a valid frame by selecting a bin and then a point in that bin."""
    reference_length = _positive_integer(reference_length, "reference_length")
    bin_key, offset_key = jax.random.split(key)
    probabilities = phase_sampling_probabilities(state, uniform_ratio)
    selected_bin = jax.random.categorical(bin_key, jp.log(probabilities))
    offset = jax.random.uniform(offset_key, (), dtype=jp.float32)
    phase = (
        (selected_bin.astype(jp.float32) + offset)
        / float(state.failed_count.shape[0])
        * float(reference_length - 1)
    ).astype(jp.int32)
    return jp.clip(phase, 0, reference_length - 2)


def sample_reset_perturbations(
    key: jax.Array, action_dim: int
) -> ResetPerturbations:
    """Samples the active upstream RMR reset perturbation ranges."""
    action_dim = _positive_integer(action_dim, "action_dim")
    keys = jax.random.split(key, 5)

    def symmetric_uniform(sample_key, bound):
        return jax.random.uniform(
            sample_key,
            bound.shape,
            minval=-bound,
            maxval=bound,
        )

    return ResetPerturbations(
        root_position=symmetric_uniform(keys[0], _ROOT_POSITION_BOUNDS),
        root_euler_xyz=symmetric_uniform(keys[1], _ROOT_EULER_BOUNDS),
        root_linear_velocity=symmetric_uniform(
            keys[2], _ROOT_LINEAR_VELOCITY_BOUNDS
        ),
        root_angular_velocity=symmetric_uniform(
            keys[3], _ROOT_ANGULAR_VELOCITY_BOUNDS
        ),
        joint_position=jax.random.uniform(
            keys[4], (action_dim,), minval=-0.05, maxval=0.05
        ),
    )


def _quaternion_from_euler_xyz(euler_xyz: jax.Array) -> jax.Array:
    half_roll, half_pitch, half_yaw = 0.5 * euler_xyz
    cr, sr = jp.cos(half_roll), jp.sin(half_roll)
    cp, sp = jp.cos(half_pitch), jp.sin(half_pitch)
    cy, sy = jp.cos(half_yaw), jp.sin(half_yaw)
    return jp.array(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def corrupt_actor_observation(
    key: jax.Array, observation: jax.Array
) -> jax.Array:
    """Add RMR corruption while preserving future-reference suffixes."""
    observation = jp.asarray(observation)
    dimension = observation.shape[-1]
    if dimension < 154 or (dimension - 154) % 58 != 0:
        raise ValueError(
            "actor observation trailing dimension must be 154 plus "
            "complete 58-value future-reference commands"
        )
    keys = jax.random.split(key, len(ACTOR_NOISE_SLICES))
    noisy = observation
    for noise_key, (region, bound) in zip(
        keys, ACTOR_NOISE_SLICES, strict=True
    ):
        shape = observation.shape[:-1] + (region.stop - region.start,)
        noise = jax.random.uniform(
            noise_key,
            shape,
            minval=-bound,
            maxval=bound,
            dtype=observation.dtype,
        )
        noisy = noisy.at[..., region].add(noise)
    return noisy


def reset_training_at_phase(
    env,
    key: jax.Array,
    difficulty: jax.Array,
    phase: jax.Array,
) -> EnvState:
    """Creates a noisy RMR RSI state without changing exact evaluation reset."""
    perturbation_key, observation_key, continuation_key = jax.random.split(
        key, 3
    )
    perturbation = sample_reset_perturbations(
        perturbation_key, env.action_dim
    )
    phase = jp.asarray(phase, dtype=jp.int32)

    qpos = env.qpos_reference[phase]
    qvel = env.qvel_reference[phase]
    orientation_delta = _quaternion_from_euler_xyz(
        perturbation.root_euler_xyz
    )
    root_orientation = _quat_mul(orientation_delta, qpos[3:7])
    root_orientation = root_orientation / jp.linalg.norm(root_orientation)
    joint_position = jp.clip(
        qpos[7:] + perturbation.joint_position,
        env.soft_joint_lower,
        env.soft_joint_upper,
    )
    qpos = qpos.at[:3].add(perturbation.root_position)
    qpos = qpos.at[3:7].set(root_orientation)
    qpos = qpos.at[7:].set(joint_position)
    qvel = qvel.at[:3].add(perturbation.root_linear_velocity)
    qvel = qvel.at[3:6].add(perturbation.root_angular_velocity)

    data = mjx.make_data(env.mjx_model).replace(qpos=qpos, qvel=qvel)
    data = mjx.forward(env.mjx_model, data)
    info = env._base_info(
        rng=continuation_key,
        phase=phase,
        difficulty=difficulty,
    )
    actor_frame = corrupt_actor_observation(
        observation_key, env._get_actor_obs(data, info)
    )
    actor_history = jp.repeat(
        actor_frame[None, :], env.actor_history_len, axis=0
    )
    critic_obs = env._get_critic_obs(data, info)
    info = {
        **info,
        "actor_obs_history": actor_history,
        "bootstrap_obs": actor_history.reshape(-1),
        "bootstrap_critic_obs": critic_obs,
    }
    return EnvState(
        data=data,
        obs=actor_history.reshape(-1),
        reward=jp.array(0.0),
        done=jp.array(0.0),
        info=info,
        metrics=env._init_metrics(),
    )
