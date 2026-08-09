"""Pure score-estimator, geometry, and scaling primitives for SHAC audits."""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jp

from src.algorithms.shac.batch_gradients import (
    tree_cosine,
    tree_dot,
    tree_norm,
)


PyTree = Any


PHASE_BINS = (
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
)

_FUNCTIONAL_RMS_CALIBRATION_STEPS = 8
_FUNCTIONAL_RMS_RELATIVE_TOLERANCE = 2e-5


class PhaseBinGradientGeometry(NamedTuple):
    """Scalar geometry for one immutable initial-reference phase interval."""

    start: int
    stop: int
    count: jax.Array
    finite_count: jax.Array
    raw_mean_norm: jax.Array
    clipped_mean_norm: jax.Array
    raw_trace_variance: jax.Array
    clipped_trace_variance: jax.Array
    raw_snr: jax.Array
    clipped_snr: jax.Array
    negative_cosine_fraction: jax.Array


class PerEnvironmentGradientGeometry(NamedTuple):
    """Raw and production-effective geometry for a per-environment tree."""

    raw_mean: PyTree
    clipped_mean: PyTree
    finite_by_env: jax.Array
    finite_fraction: jax.Array
    raw_norm_by_env: jax.Array
    clipped_norm_by_env: jax.Array
    clipping_fraction: jax.Array
    raw_mean_norm: jax.Array
    clipped_mean_norm: jax.Array
    raw_trace_variance: jax.Array
    clipped_trace_variance: jax.Array
    raw_snr: jax.Array
    clipped_snr: jax.Array
    negative_cosine_fraction: jax.Array
    contribution_to_aggregate_cosines: jax.Array
    raw_vs_clipped_cosine: jax.Array
    phase_bins: tuple[PhaseBinGradientGeometry, ...]


class FunctionalActorStepSummary(NamedTuple):
    """Functional-size and exact output-delta measurements for one step."""

    scale: jax.Array
    linearized_rms: jax.Array
    output_rms: jax.Array
    max_action_change: jax.Array


class _PopulationGeometry(NamedTuple):
    mean: PyTree
    mean_norm: jax.Array
    trace_variance: jax.Array
    snr: jax.Array
    negative_cosine_fraction: jax.Array
    contribution_cosines: jax.Array


def pytree_leaf_order(tree: PyTree) -> tuple[str, ...]:
    """Returns JAX's deterministic path order for the leaves of ``tree``."""
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    return tuple(
        jax.tree_util.keystr(path) for path, _ in leaves_with_paths
    )


def assert_matching_pytree_leaf_order(*trees: PyTree) -> None:
    """Fails closed unless all trees have exactly the same PyTree leaf order."""
    if not trees:
        raise ValueError("at least one pytree is required")
    reference_structure = jax.tree_util.tree_structure(trees[0])
    reference_order = pytree_leaf_order(trees[0])
    for tree in trees[1:]:
        if (
            jax.tree_util.tree_structure(tree) != reference_structure
            or pytree_leaf_order(tree) != reference_order
        ):
            raise ValueError("pytree leaf order mismatch")


def _require_per_env_leaves(per_env_grads: PyTree) -> tuple[list[jax.Array], int]:
    leaves = jax.tree_util.tree_leaves(per_env_grads)
    if not leaves:
        raise ValueError("per_env_grads must contain at least one array")
    num_envs = leaves[0].shape[0]
    if num_envs == 0:
        raise ValueError("per_env_grads must contain at least one environment")
    if any(leaf.ndim == 0 or leaf.shape[0] != num_envs for leaf in leaves):
        raise ValueError("all gradient leaves must share the leading env axis")
    return leaves, num_envs


def _env_tree(batched_tree: PyTree, index: int) -> PyTree:
    return jax.tree_util.tree_map(lambda leaf: leaf[index], batched_tree)


def _safe_snr(mean_norm: jax.Array, trace_variance: jax.Array) -> jax.Array:
    """Returns finite SNR using dtype epsilon as the variance floor."""
    dtype_limits = jp.finfo(mean_norm.dtype)
    standard_deviation = jp.sqrt(
        jp.maximum(trace_variance, dtype_limits.eps)
    )
    return jp.nan_to_num(
        mean_norm / standard_deviation,
        nan=0.0,
        posinf=dtype_limits.max,
        neginf=0.0,
    )


def _population_geometry(
    contributions: PyTree,
    selected: jax.Array,
) -> _PopulationGeometry:
    """Computes finite population geometry, treating an empty selection as zero."""
    num_envs = selected.shape[0]
    denominator = jp.maximum(jp.sum(selected.astype(jp.int32)), 1)

    def mean_leaf(leaf):
        mask_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        return jp.sum(
            leaf * selected.reshape(mask_shape), axis=0
        ) / denominator

    mean = jax.tree_util.tree_map(mean_leaf, contributions)
    mean_norm = tree_norm(mean)
    individual = tuple(_env_tree(contributions, index) for index in range(num_envs))
    trace_variance = sum(
        (
            selected[index].astype(mean_norm.dtype)
            * tree_dot(
                jax.tree_util.tree_map(
                    lambda value, average: value - average, item, mean
                ),
                jax.tree_util.tree_map(
                    lambda value, average: value - average, item, mean
                ),
            )
        )
        for index, item in enumerate(individual)
    )
    trace_variance = jp.maximum(trace_variance / denominator, 0.0)
    contribution_cosines = jp.stack(
        tuple(tree_cosine(item, mean) for item in individual)
    )
    negative_cosine_fraction = jp.sum(
        selected.astype(mean_norm.dtype)
        * (contribution_cosines < 0.0).astype(mean_norm.dtype)
    ) / denominator
    return _PopulationGeometry(
        mean=mean,
        mean_norm=mean_norm,
        trace_variance=trace_variance,
        snr=_safe_snr(mean_norm, trace_variance),
        negative_cosine_fraction=negative_cosine_fraction,
        contribution_cosines=jp.where(
            selected, contribution_cosines, jp.zeros_like(contribution_cosines)
        ),
    )


def summarize_per_env_gradient_geometry(
    per_env_grads: PyTree,
    *,
    max_norm: float = 1.0,
    initial_phases: jax.Array,
) -> PerEnvironmentGradientGeometry:
    """Summarizes raw and exact production-effective gradient contributions.

    A nonfinite value removes its *whole* environment contribution.  Finite
    contributions are then clipped independently with the same norm-1 style
    contract as ``aggregate_per_env_gradients`` before the clipped mean and
    geometry are calculated.  ``initial_phases`` selects the immutable five
    reference phase intervals; values outside ``[0, 500)`` are not binned.
    """
    if not jp.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    leaves, num_envs = _require_per_env_leaves(per_env_grads)
    initial_phases = jp.asarray(initial_phases)
    if initial_phases.shape != (num_envs,):
        raise ValueError("initial_phases must have one entry per environment")

    finite_by_env = jp.ones((num_envs,), dtype=jp.bool_)
    for leaf in leaves:
        finite_by_env = finite_by_env & jp.all(
            jp.isfinite(leaf), axis=tuple(range(1, leaf.ndim))
        )

    def sanitize_leaf(leaf):
        safe_leaf = jp.where(jp.isfinite(leaf), leaf, 0.0)
        mask_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        return safe_leaf * finite_by_env.reshape(mask_shape)

    raw_contributions = jax.tree_util.tree_map(sanitize_leaf, per_env_grads)
    raw_individual = tuple(
        _env_tree(raw_contributions, index) for index in range(num_envs)
    )
    raw_norm_by_env = jp.stack(tuple(tree_norm(item) for item in raw_individual))
    scale_by_env = jp.minimum(
        1.0, max_norm / jp.maximum(raw_norm_by_env, 1e-12)
    )

    def clip_leaf(leaf):
        scale_shape = (num_envs,) + (1,) * (leaf.ndim - 1)
        return leaf * scale_by_env.reshape(scale_shape)

    clipped_contributions = jax.tree_util.tree_map(
        clip_leaf, raw_contributions
    )
    clipped_individual = tuple(
        _env_tree(clipped_contributions, index) for index in range(num_envs)
    )
    clipped_norm_by_env = jp.stack(
        tuple(tree_norm(item) for item in clipped_individual)
    )
    full_population = jp.ones((num_envs,), dtype=jp.bool_)
    raw = _population_geometry(raw_contributions, finite_by_env)
    clipped = _population_geometry(clipped_contributions, full_population)
    phase_bins = tuple(
        PhaseBinGradientGeometry(
            start=start,
            stop=stop,
            count=jp.sum(phase_selection.astype(jp.int32)),
            finite_count=jp.sum(
                (phase_selection & finite_by_env).astype(jp.int32)
            ),
            raw_mean_norm=(raw_bin := _population_geometry(
                raw_contributions,
                phase_selection & finite_by_env,
            )).mean_norm,
            clipped_mean_norm=(clipped_bin := _population_geometry(
                clipped_contributions,
                phase_selection,
            )).mean_norm,
            raw_trace_variance=raw_bin.trace_variance,
            clipped_trace_variance=clipped_bin.trace_variance,
            raw_snr=raw_bin.snr,
            clipped_snr=clipped_bin.snr,
            negative_cosine_fraction=clipped_bin.negative_cosine_fraction,
        )
        for start, stop in PHASE_BINS
        for phase_selection in (
            (initial_phases >= start) & (initial_phases < stop),
        )
    )
    finite_count = jp.sum(finite_by_env.astype(jp.int32))
    return PerEnvironmentGradientGeometry(
        raw_mean=raw.mean,
        clipped_mean=clipped.mean,
        finite_by_env=finite_by_env,
        finite_fraction=jp.mean(finite_by_env.astype(jp.float32)),
        raw_norm_by_env=raw_norm_by_env,
        clipped_norm_by_env=clipped_norm_by_env,
        clipping_fraction=jp.sum(
            (
                finite_by_env & (raw_norm_by_env > max_norm)
            ).astype(jp.float32)
        ) / jp.maximum(finite_count, 1),
        raw_mean_norm=raw.mean_norm,
        clipped_mean_norm=clipped.mean_norm,
        raw_trace_variance=raw.trace_variance,
        clipped_trace_variance=clipped.trace_variance,
        raw_snr=raw.snr,
        clipped_snr=clipped.snr,
        negative_cosine_fraction=clipped.negative_cosine_fraction,
        contribution_to_aggregate_cosines=clipped.contribution_cosines,
        raw_vs_clipped_cosine=tree_cosine(raw.mean, clipped.mean),
        phase_bins=phase_bins,
    )


def _require_finite(value: PyTree, *, label: str) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves or any(
        not bool(jp.all(jp.isfinite(leaf))) for leaf in leaves
    ):
        raise ValueError(f"{label} contains nonfinite values")


def _candidate_actor_delta(
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    params: PyTree,
    direction: PyTree,
    observations: jax.Array,
    baseline_outputs: jax.Array,
    scale: jax.Array,
) -> tuple[PyTree, jax.Array, jax.Array]:
    candidate_params = jax.tree_util.tree_map(
        lambda value, delta: value + scale * delta, params, direction
    )
    candidate_outputs = actor_apply(candidate_params, observations)
    _require_finite(candidate_outputs, label="candidate actor output")
    action_delta = candidate_outputs - baseline_outputs
    _require_finite(action_delta, label="exact action delta")
    output_rms = jp.sqrt(jp.mean(jp.square(action_delta)))
    _require_finite(output_rms, label="exact action RMS")
    return candidate_params, action_delta, output_rms


def apply_functional_actor_step(
    actor_apply: Callable[[PyTree, jax.Array], jax.Array],
    params: PyTree,
    direction: PyTree,
    observations: jax.Array,
    *,
    target_rms: float = 0.01,
) -> tuple[PyTree, FunctionalActorStepSummary]:
    """Scales a parameter direction to a target actor-output RMS change.

    A JVP at the frozen parameters initializes the scale.  Eight deterministic
    scalar target/exact-RMS corrections then calibrate nonlinear actor outputs;
    this is not an objective line search.  The final exact RMS must meet the
    target within a relative tolerance of ``2e-5``, including float32 output
    quantization around the frozen baseline.
    """
    if not jp.isfinite(target_rms) or target_rms <= 0.0:
        raise ValueError("target_rms must be finite and positive")
    assert_matching_pytree_leaf_order(params, direction)
    _require_finite(direction, label="parameter direction")
    baseline_outputs = actor_apply(params, observations)
    _require_finite(baseline_outputs, label="baseline actor output")
    _, output_tangent = jax.jvp(
        lambda value: actor_apply(value, observations),
        (params,),
        (direction,),
    )
    direction_rms = jp.sqrt(jp.mean(jp.square(output_tangent)))
    if not bool(jp.isfinite(direction_rms)) or not bool(direction_rms > 0.0):
        raise ValueError("parameter direction must produce a finite nonzero JVP")
    target = jp.asarray(target_rms, dtype=direction_rms.dtype)
    scale = target / direction_rms
    for _ in range(_FUNCTIONAL_RMS_CALIBRATION_STEPS):
        _, _, output_rms = _candidate_actor_delta(
            actor_apply,
            params,
            direction,
            observations,
            baseline_outputs,
            scale,
        )
        if not bool(output_rms > 0.0):
            raise ValueError("exact action RMS must be nonzero")
        scale = scale * target / output_rms
    candidate_params, action_delta, output_rms = _candidate_actor_delta(
        actor_apply,
        params,
        direction,
        observations,
        baseline_outputs,
        scale,
    )
    relative_error = jp.abs(output_rms - target) / target
    if not bool(relative_error <= _FUNCTIONAL_RMS_RELATIVE_TOLERANCE):
        raise ValueError("exact action RMS calibration did not converge")
    return candidate_params, FunctionalActorStepSummary(
        scale=scale,
        linearized_rms=jp.sqrt(jp.mean(jp.square(scale * output_tangent))),
        output_rms=output_rms,
        max_action_change=jp.max(jp.abs(action_delta)),
    )


def discounted_return_to_go(
    rewards: jax.Array,
    dones: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    """Returns actor-loss score coefficients through episode boundaries.

    ``dones[t]`` marks that transition ``t`` as terminal, so its immediate
    reward remains in the return while rewards from later transitions do not.
    Each reward retains the outer discount accumulated from the start of its
    episode, matching the production actor loss; that discount resets after a
    done transition.
    The leading axis is time; all remaining axes are independent trajectories.
    """

    def advance_discount(discount, done):
        next_discount = jp.where(done, 1.0, discount * gamma)
        return next_discount, discount

    _, outer_discounts = jax.lax.scan(
        advance_discount,
        jp.ones_like(rewards[0]),
        dones,
    )
    discounted_rewards = outer_discounts * rewards

    def accumulate(next_return, transition):
        reward, done = transition
        current_return = reward + jp.where(done, 0.0, next_return)
        return current_return, current_return

    _, reversed_returns = jax.lax.scan(
        accumulate,
        jp.zeros_like(rewards[0]),
        (jp.flip(discounted_rewards, axis=0), jp.flip(dones, axis=0)),
    )
    return jp.flip(reversed_returns, axis=0)


def detached_gaussian_score_loss(
    means: jax.Array,
    actions: jax.Array,
    returns_to_go: jax.Array,
    *,
    std: float | jax.Array,
) -> jax.Array:
    """Computes a stopped-data Gaussian likelihood-ratio loss.

    The leading axis is time and the final axis is the action dimension.  Any
    axes between them are preserved, allowing callers to receive one loss per
    environment by passing arrays shaped ``(time, environment, action)``.
    """
    stopped_actions = jax.lax.stop_gradient(actions)
    stopped_returns = jax.lax.stop_gradient(returns_to_go)
    standard_deviation = jp.asarray(std)
    normalized_error = (stopped_actions - means) / standard_deviation
    log_probability = -0.5 * jp.sum(
        jp.square(normalized_error)
        + 2.0 * jp.log(standard_deviation)
        + jp.log(2.0 * jp.pi),
        axis=-1,
    )
    return -jp.mean(stopped_returns * log_probability, axis=0)
