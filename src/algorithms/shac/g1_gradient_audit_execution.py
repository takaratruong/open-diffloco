"""Experiment-facing execution for the frozen E064 gradient audit.

The small helpers in this module keep checkpoint mutation, stochastic-noise
construction, and evidence summaries explicit.  The expensive G1 engine is
assembled by :func:`run_audit` below without changing the training path.
"""

from __future__ import annotations

import dataclasses
import getpass
import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.batch_gradients import tree_norm
from src.algorithms.shac.g1_gradient_audit_runner import (
    OutcomeThresholds,
    aggregate_four_shards,
    build_descent_candidates,
    classify_preregistered_outcome,
    to_finite_json,
)

_POPULATION = 64
_HORIZON = 48
_ACTION_DIMENSION = 29
_SHARD_SEEDS = (0, 1, 2, 3)
_HELD_OUT_SEEDS = (4, 5, 6, 7)
_PHASES = (0, 100, 200, 300, 400)
_E066_BASELINE_SURVIVAL = (135, 236, 152, 83, 74)
_E066_COMPETENCE_FLOORS = (110, 78, 74, 76, 58)
_CANDIDATE_LABELS = ("baseline", "pathwise", "score")
_ORDINARY_ARRAY_FIELDS = (
    "active",
    "rewards",
    "dones",
    "terminals",
    "actions",
    "phases",
    "qpos",
    "qvel",
)
_ENVIRONMENT_LOCK = Path(
    "/home/ubuntu/recovery/diffsim2real-20260710T025000Z/"
    "environments/rl.environment.yml"
)
_ENVIRONMENT_LOCK_SHA256 = (
    "4d986f8fe261de2e9cefa45439a2bf8f4493547dc4de694a7df2c37045c9c7ca"
)
_G1_XML_SHA256 = "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
_RMR_CONTROLLER = Path(
    "/home/ubuntu/projects/diffsim2real/outputs/rmr_torques_iter4999.npz"
)
_RMR_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)
_SOLVER_TRACE_CONTEXT = {
    "gradient_shards": "fixed_mjx_solver_outer_loop",
    "heldout_stochastic": "fixed_mjx_solver_outer_loop",
    "ordinary_phase_grid": "stock_mjx_forward_solver",
}


E064_OUTCOME_THRESHOLDS = OutcomeThresholds(
    minimum_stability=0.2,
    material_stability_advantage=0.1,
    minimum_alignment=0.7,
    minimum_return_improvement=0.001,
    minimum_survival_improvement=5.0,
    stability_tolerance=0.05,
    return_tolerance=0.001,
    survival_tolerance=2.0,
    maximum_phase_survival_loss_fraction=0.10,
    bootstrap_confidence_level=0.95,
)


def _baseline_competence_receipt(
    *,
    phases: Sequence[int],
    actual_survival: Sequence[int],
) -> dict[str, Any]:
    """Admit a baseline by E066's frozen floors, retaining exact replay as telemetry."""
    phase_values = [int(value) for value in phases]
    survival_values = [int(value) for value in actual_survival]
    if tuple(phase_values) != _PHASES:
        raise ValueError("baseline phases do not match the frozen phase grid")
    if len(survival_values) != len(_E066_COMPETENCE_FLOORS):
        raise ValueError("baseline survival does not match the frozen phase grid")
    historical = list(_E066_BASELINE_SURVIVAL)
    floors = list(_E066_COMPETENCE_FLOORS)
    exact_match = survival_values == historical
    return {
        "source_experiment": "E066",
        "phases": phase_values,
        "historical_expected_survival": historical,
        "actual_survival": survival_values,
        "competence_floors": floors,
        "competence_floor_pass": all(
            actual >= floor for actual, floor in zip(survival_values, floors, strict=True)
        ),
        "historical_exact_match": exact_match,
        # Retain the v3 keys so older evidence readers can still consume the
        # diagnostic historical comparison without treating it as admission.
        "expected_survival": historical,
        "exact_match": exact_match,
    }


@dataclass(frozen=True)
class EstimatorShardEvidence:
    """One shared-rollout result and independently matched identity receipts."""

    result: Any
    pathwise_receipt: Mapping[str, Any]
    score_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedAuditExecution:
    """Host-validated live dependencies consumed by the pure audit sequence."""

    checkpoint_state: Any
    actor_apply: Callable[[Any, jax.Array], jax.Array]
    normalizer_state: Any
    estimate_shard: Callable[[int], EstimatorShardEvidence]
    prepare_first_action_objective: Callable[[jax.Array, int], Any]
    stochastic_rollout: Callable[[Any, jax.Array], Any]
    phase_rollout: Callable[..., Mapping[str, Any]]
    validated_contract: Any
    algorithmic_validity: Mapping[str, bool]
    external_inputs: Mapping[str, Any] = field(default_factory=dict)
    runtime_provenance: Mapping[str, Any] = field(default_factory=dict)
    gradient_solver_context: Callable[[], Any] = nullcontext
    render_phase_zero: Callable[..., Mapping[str, Path]] | None = None


def make_frozen_action_noise(seed: int) -> jax.Array:
    """Materialize one preregistered standard-normal action-noise shard."""
    jax.config.update("jax_enable_x64", True)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("action-noise seed must be an integer")
    noise = jax.random.normal(
        jax.random.PRNGKey(int(seed)),
        (_POPULATION, _HORIZON, _ACTION_DIMENSION),
        dtype=jp.float64,
    )
    if not bool(jp.all(jp.isfinite(noise))):
        raise ValueError("materialized action noise must be finite")
    return noise


def replace_actor_parameters(checkpoint_state: Any, actor_params: Any) -> Any:
    """Return a candidate checkpoint that differs only in actor parameters."""
    if not hasattr(checkpoint_state, "replace"):
        raise TypeError("checkpoint state must support immutable replace()")
    return checkpoint_state.replace(actor_params=actor_params)


def summarize_stochastic_rollout(
    trajectory: Any,
    *,
    gamma: float,
    expected_population: int | None = None,
    expected_horizon: int | None = None,
    expected_action_dimension: int | None = None,
    expected_observation_dimension: int | None = None,
) -> dict[str, Any]:
    """Summarize the frozen episode-start-discounted stochastic fragments."""
    required_fields = (
        "noise",
        "observation_rngs",
        "raw_observations",
        "observations",
        "normalized_observations",
        "means",
        "actions",
        "rewards",
        "dones",
        "initial_phase",
    )
    missing = [name for name in required_fields if not hasattr(trajectory, name)]
    if missing:
        raise ValueError(f"stochastic trajectory is incomplete: {missing}")
    rewards = jp.asarray(trajectory.rewards)
    dones = jp.asarray(trajectory.dones)
    if rewards.ndim != 2 or dones.shape != rewards.shape or rewards.shape[1] < 1:
        raise ValueError("reward and done arrays must have matching (env, time) shape")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between zero and one")
    if not bool(jp.all(jp.isfinite(rewards))):
        raise ValueError("rollout rewards must be finite")
    population = expected_population or rewards.shape[0]
    horizon = expected_horizon or rewards.shape[1]
    action_dimension = (
        expected_action_dimension or np.asarray(trajectory.actions).shape[-1]
    )
    observation_dimension = (
        expected_observation_dimension
        or np.asarray(trajectory.normalized_observations).shape[-1]
    )
    expected_shapes = {
        "noise": (population, horizon, action_dimension),
        "observation_rngs": (population, horizon, 2),
        "raw_observations": (population, horizon, observation_dimension),
        "observations": (population, horizon, observation_dimension),
        "normalized_observations": (population, horizon, observation_dimension),
        "means": (population, horizon, action_dimension),
        "actions": (population, horizon, action_dimension),
        "rewards": (population, horizon),
        "dones": (population, horizon),
        "initial_phase": (population,),
    }
    for name, expected_shape in expected_shapes.items():
        array = np.asarray(jax.device_get(getattr(trajectory, name)))
        if array.shape != expected_shape:
            raise ValueError(
                f"stochastic trajectory {name} shape must be {expected_shape}, "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"stochastic trajectory {name} contains nonfinite values")
    initial_phases = np.asarray(jax.device_get(trajectory.initial_phase))
    if not np.issubdtype(initial_phases.dtype, np.integer) or not np.all(
        (initial_phases >= 0) & (initial_phases < 500)
    ):
        raise ValueError(
            "stochastic trajectory initial_phase must contain integer reference phases"
        )

    def one_environment(env_rewards, env_dones):
        def accumulate(carry, transition):
            total, discount = carry
            reward, done = transition
            total = total + discount * reward
            discount = jp.where(done, jp.ones_like(discount), discount * gamma)
            return (total, discount), None

        zero = jp.zeros_like(env_rewards[0])
        one = jp.ones_like(env_rewards[0])
        (total, _), _ = jax.lax.scan(
            accumulate,
            (zero, one),
            (env_rewards, env_dones),
        )
        any_done = jp.any(env_dones)
        first_done = jp.argmax(env_dones.astype(jp.int32)) + 1
        survival = jp.where(any_done, first_done, env_rewards.shape[0])
        return total, survival, any_done

    returns, survivals, terminals = jax.vmap(one_environment)(rewards, dones)
    return {
        "discounted_return_by_env": np.asarray(jax.device_get(returns)).tolist(),
        "survival_by_env": np.asarray(jax.device_get(survivals)).tolist(),
        "mean_discounted_return": float(jp.mean(returns)),
        "mean_survival": float(jp.mean(survivals)),
        "terminal_fraction": float(jp.mean(terminals.astype(jp.float64))),
        "trajectory_complete": True,
        "trajectory_shapes": {
            name: list(shape) for name, shape in expected_shapes.items()
        },
    }


def make_phase_rollout(
    env: Any,
    actor_apply: Callable[[Any, jax.Array], jax.Array],
    *,
    normalizer: Any,
    max_steps: int,
) -> Callable[..., dict[str, Any]]:
    """Build one compiled, strict, replay-free ordinary phase evaluator."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    action_dim = getattr(env, "action_dim", None)
    if not isinstance(action_dim, int) or action_dim < 1:
        raise ValueError("environment action_dim must be a positive integer")

    def compiled_rollout(actor_params, normalizer_state, key, phase):
        state = env.reset_at_phase(
            key,
            jp.array(0.0, dtype=jp.float64),
            phase,
        )

        def scan_step(carry, _):
            carried_state, stopped = carry

            def inactive(current):
                zeros = jp.zeros((action_dim,), dtype=jp.float64)
                record = {
                    "active": jp.array(False),
                    "reward": jp.zeros_like(current.reward),
                    "done": jp.array(True),
                    "terminal": jp.array(True),
                    "action": zeros,
                    "phase": current.info["phase"],
                    "qpos": current.data.qpos,
                    "qvel": current.data.qvel,
                }
                return current, jp.array(True), record

            def advance(current):
                normalized = env.normalize_actor_obs(
                    normalizer,
                    normalizer_state,
                    current.obs,
                ).astype(jp.float32)
                action = actor_apply(actor_params, normalized).astype(jp.float64)
                next_state = env.step(current, action)
                done = next_state.done > 0.5
                terminal = next_state.info.get("terminal", done) > 0.5
                record = {
                    "active": jp.array(True),
                    "reward": next_state.reward,
                    "done": done,
                    "terminal": terminal,
                    "action": action,
                    "phase": next_state.info["phase"],
                    "qpos": next_state.data.qpos,
                    "qvel": next_state.data.qvel,
                }
                return next_state, done, record

            next_state, next_stopped, record = jax.lax.cond(
                stopped,
                inactive,
                advance,
                carried_state,
            )
            return (next_state, next_stopped), record

        (_, stopped), arrays = jax.lax.scan(
            scan_step,
            (state, jp.array(False)),
            xs=None,
            length=max_steps,
        )
        return stopped, arrays

    compiled = jax.jit(compiled_rollout)

    def evaluate(
        actor_params: Any,
        normalizer_state: Any,
        *,
        seed: int,
        phase: int,
    ) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("evaluation seed must be an integer")
        if isinstance(phase, bool) or not isinstance(phase, (int, np.integer)):
            raise TypeError("evaluation phase must be an integer")
        stopped, arrays = compiled(
            actor_params,
            normalizer_state,
            jax.random.PRNGKey(int(seed)),
            jp.asarray(int(phase), dtype=jp.int32),
        )
        host = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value)), arrays
        )
        active = host["active"].astype(bool)
        rewards = host["reward"].astype(np.float64)
        finite_arrays = tuple(
            np.asarray(value)
            for key, value in host.items()
            if key not in {"active", "done", "terminal"}
        )
        if not all(np.isfinite(value).all() for value in finite_arrays):
            raise ValueError("ordinary phase rollout contains nonfinite arrays")
        survival = int(np.sum(active))
        terminal = bool(np.any(host["terminal"] & active))
        reward_sum = float(np.sum(rewards[active]))
        mean_reward = reward_sum / survival if survival else 0.0
        return {
            "phase": int(phase),
            # The decision runner calls this common scalar ``return``.  Use
            # mean ordinary reward so variable survival is judged separately.
            "return": mean_reward,
            "reward_sum": reward_sum,
            "survival": survival,
            "terminal": terminal,
            "complete": bool(stopped) or survival == max_steps,
            "replay_free": True,
            "active": active,
            "rewards": rewards,
            "dones": host["done"].astype(bool),
            "terminals": host["terminal"].astype(bool),
            "actions": host["action"].astype(np.float64),
            "phases": host["phase"].astype(np.int64),
            "qpos": host["qpos"].astype(np.float64),
            "qvel": host["qvel"].astype(np.float64),
        }

    return evaluate


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_execution_contract(contract: Any) -> None:
    expected = {
        "shard_seeds": _SHARD_SEEDS,
        "held_out_seeds": _HELD_OUT_SEEDS,
        "phases": _PHASES,
        "population": _POPULATION,
        "horizon": _HORIZON,
        "sigma": 0.1,
        "gamma": 0.99,
        "per_env_clip": 1.0,
        "functional_rms": 0.01,
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
    }
    for name, frozen in expected.items():
        actual = getattr(contract, name, None)
        if isinstance(frozen, tuple):
            try:
                actual = tuple(actual)
            except TypeError:
                pass
        if actual != frozen:
            raise ValueError(
                f"execution contract {name} must equal {frozen!r}, got {actual!r}"
            )


def _tree_is_finite_nonzero(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(
        leaves
        and all(bool(jp.all(jp.isfinite(leaf))) for leaf in leaves)
        and bool(jp.isfinite(tree_norm(tree)))
        and bool(tree_norm(tree) > 0.0)
    )


def _candidate_delta_is_finite_nonzero(baseline: Any, candidate: Any) -> bool:
    try:
        delta = jax.tree_util.tree_map(
            lambda candidate_leaf, baseline_leaf: candidate_leaf - baseline_leaf,
            candidate,
            baseline,
        )
    except (TypeError, ValueError):
        return False
    return _tree_is_finite_nonzero(delta)


def _validate_ordinary_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    missing = sorted(set(_ORDINARY_ARRAY_FIELDS) - set(row))
    if missing:
        raise ValueError(f"ordinary rollout is missing arrays: {missing}")
    arrays = {field: np.asarray(row[field]) for field in _ORDINARY_ARRAY_FIELDS}
    lengths = {array.shape[0] for array in arrays.values() if array.ndim > 0}
    if any(array.ndim == 0 for array in arrays.values()) or len(lengths) != 1:
        raise ValueError("ordinary rollout arrays must share a leading time axis")
    for field_name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(
                f"ordinary rollout array {field_name!r} contains nonfinite values"
            )

    active = arrays["active"].astype(bool)
    rewards = arrays["rewards"].astype(np.float64)
    survival = int(np.sum(active))
    canonical_return = float(np.mean(rewards[active])) if survival else 0.0
    if int(row.get("survival", -1)) != survival:
        raise ValueError("ordinary rollout survival must equal active steps")
    if not np.isclose(
        float(row.get("return", np.nan)),
        canonical_return,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("ordinary rollout return must equal mean active reward")
    summary = {
        str(key): value
        for key, value in row.items()
        if key not in _ORDINARY_ARRAY_FIELDS
    }
    summary["return"] = canonical_return
    summary["survival"] = survival
    return to_finite_json(summary), arrays


def _run_algorithmic_validity_checks() -> dict[str, bool]:
    """Execute the small estimator proofs required by the decision receipt."""
    from src.algorithms.shac.gradient_audit import (
        assert_matching_pytree_leaf_order,
        detached_gaussian_score_loss,
        discounted_return_to_go,
    )

    mean = jp.array([[0.25]], dtype=jp.float64)
    action = jp.array([[0.75]], dtype=jp.float64)
    returns = jp.array([2.0], dtype=jp.float64)
    score_gradient = jax.grad(
        lambda value: detached_gaussian_score_loss(value, action, returns, std=0.5)
    )(mean)
    analytic_gaussian_sign = bool(
        np.array_equal(np.asarray(score_gradient), np.array([[-4.0]]))
    )
    action_gradient = jax.grad(
        lambda sampled_action: detached_gaussian_score_loss(
            mean, sampled_action, returns, std=0.5
        )
    )(action)
    return_gradient = jax.grad(
        lambda coefficient: detached_gaussian_score_loss(
            mean, action, coefficient, std=0.5
        )
    )(returns)
    detachment = bool(
        np.array_equal(np.asarray(action_gradient), np.zeros((1, 1)))
        and np.array_equal(np.asarray(return_gradient), np.zeros((1,)))
    )

    boundary_returns = discounted_return_to_go(
        jp.array([1.0, 2.0, 100.0, 4.0], dtype=jp.float64),
        jp.array([False, True, False, False]),
        gamma=0.5,
    )
    done_boundary_return = bool(
        np.array_equal(np.asarray(boundary_returns), np.array([2.0, 1.0, 102.0, 2.0]))
    )

    stopped_action = jax.lax.stop_gradient(action)
    stopped_return = jax.lax.stop_gradient(returns)

    def log_probability(value):
        return -0.5 * jp.sum(jp.square((stopped_action - value) / 0.5), axis=-1)

    old_log_probability = jax.lax.stop_gradient(log_probability(mean))

    def ppo_loss(value):
        ratio = jp.exp(log_probability(value) - old_log_probability)
        unclipped = ratio * stopped_return
        clipped = jp.clip(ratio, 0.8, 1.2) * stopped_return
        return -jp.mean(jp.minimum(unclipped, clipped))

    ppo_gradient = jax.grad(ppo_loss)(mean)
    ppo_ratio_one = bool(
        np.array_equal(np.asarray(ppo_gradient), np.asarray(score_gradient))
    )

    pytree_order = True
    try:
        assert_matching_pytree_leaf_order(
            {"a": jp.array([1.0]), "b": (jp.array([2.0]),)},
            {"b": (jp.array([3.0]),), "a": jp.array([4.0])},
        )
    except ValueError:
        pytree_order = False

    # A symmetric, finite smooth toy makes both negative-objective estimators
    # converge to the same analytic derivative without an extra RNG contract.
    sigma = 0.1
    parameter = 0.4
    target = -0.2
    toy_noise = np.tile(np.array([-1.0, 1.0], dtype=np.float64), 4096)
    toy_action = parameter + sigma * toy_noise
    toy_pathwise = float(np.mean(2.0 * (toy_action - target)))
    toy_reward = -np.square(toy_action - target)
    toy_score = float(np.mean(-toy_reward * toy_noise / sigma))
    smooth_toy_convergence = bool(
        np.isfinite([toy_pathwise, toy_score]).all()
        and np.isclose(toy_pathwise, 2.0 * (parameter - target), atol=1e-12)
        and np.isclose(toy_score, toy_pathwise, rtol=0.01, atol=0.01)
    )
    return {
        "analytic_gaussian_sign": analytic_gaussian_sign,
        "detachment": detachment,
        "done_boundary_return": done_boundary_return,
        "ppo_ratio_one": ppo_ratio_one,
        "pytree_order": pytree_order,
        "smooth_toy_convergence": smooth_toy_convergence,
    }


def _render_phase_zero_videos(
    env: Any,
    *,
    rows: Mapping[tuple[int, str, int], Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Render required phase-zero videos from already validated numeric arrays."""
    import imageio.v2 as imageio
    import mujoco

    from tools.evaluate_g1_tracking import _render_pair

    paths: dict[str, Path] = {}
    for label in _CANDIDATE_LABELS:
        row = rows[(0, label, 0)]
        active = np.asarray(row["active"], dtype=bool)
        qpos = np.asarray(row["qpos"])[active]
        qvel = np.asarray(row["qvel"])[active]
        phases = np.asarray(row["phases"], dtype=np.int64)[active]
        actual_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        reference_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        actual_data = mujoco.MjData(env.mj_model)
        reference_data = mujoco.MjData(env.mj_model)
        try:
            frames = [
                _render_pair(
                    env,
                    qpos[index],
                    qvel[index],
                    int(phases[index]),
                    actual_renderer,
                    reference_renderer,
                    actual_data,
                    reference_data,
                )
                for index in range(0, len(qpos), 2)
            ]
        finally:
            actual_renderer.close()
            reference_renderer.close()
        if not frames:
            raise ValueError(f"{label} phase-zero rollout has no active video frames")
        path = output_dir / "videos" / f"{label}_phase0.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        try:
            imageio.mimsave(temporary, frames, fps=25)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        paths[label] = path
    return paths


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_external_inputs(hparams: Mapping[str, Any]) -> dict[str, Any]:
    registered = {
        "environment_lock": (_ENVIRONMENT_LOCK, _ENVIRONMENT_LOCK_SHA256),
        "plant_xml": (Path(hparams["xml_path"]), _G1_XML_SHA256),
        "rmr_controller": (_RMR_CONTROLLER, _RMR_CONTROLLER_SHA256),
    }
    receipt = {}
    for label, (path, expected) in registered.items():
        if not path.is_file():
            raise ValueError(f"registered {label} does not exist: {path}")
        actual = _sha256_path(path)
        if actual != expected:
            raise ValueError(
                f"registered {label} SHA-256 differs: {actual} != {expected}"
            )
        receipt[label] = {"path": str(path), "sha256": actual}
    return receipt


def _capture_runtime_provenance() -> dict[str, Any]:
    import flax
    import mujoco

    repository = Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "process_index": int(device.process_index),
        }
        for device in jax.devices()
    ]
    return {
        "repository": str(repository),
        "code_commit": git("rev-parse", "HEAD"),
        "git_clean": not bool(status),
        "git_status": status.splitlines(),
        "dirty_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": flax.__version__,
            "mujoco": mujoco.__version__,
        },
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "devices": devices,
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "executable": sys.executable,
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _prepare_e064_execution(contract: Any) -> PreparedAuditExecution:
    """Load and host-validate E064, then bind the shared compiled runtime."""
    from src.algorithms.shac.g1_gradient_audit import (
        prepare_e064_estimator_engine,
        stable_pytree_sha256,
    )
    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.fixed_solver import fixed_mjx_solver_outer_loop
    from tools.evaluate_g1_tracking import make_evaluation_env

    with Path(contract.checkpoint).open("rb") as stream:
        checkpoint_state = pickle.load(stream)
    with Path(contract.hparams_path).open("r", encoding="utf-8") as stream:
        hparams = json.load(stream)

    runtime_provenance = _capture_runtime_provenance()
    if not runtime_provenance["git_clean"]:
        raise ValueError("E064 audit requires a clean registered code worktree")
    external_inputs = _validate_external_inputs(hparams)

    env = make_evaluation_env(
        hparams["env_variant"],
        solver_iterations=contract.solver_iterations,
        solver_ls_iterations=contract.solver_ls_iterations,
        reference_path=contract.reference,
        reference_stride=hparams["reference_stride"],
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)
    prepared_engine = prepare_e064_estimator_engine(
        checkpoint_state,
        hparams,
        checkpoint_path=contract.checkpoint,
        env=env,
        normalizer=normalizer,
    )
    validated = prepared_engine.contract
    actor_apply = prepared_engine.actor_apply

    def estimate_shard(seed: int) -> EstimatorShardEvidence:
        action_noise = make_frozen_action_noise(seed)
        exact = prepared_engine(action_noise)
        return EstimatorShardEvidence(
            exact.gradients,
            exact.pathwise_receipt,
            exact.score_receipt,
        )

    candidate_cores: dict[str, Callable[[jax.Array], Any]] = {}

    def stochastic_rollout(actor_params: Any, action_noise: jax.Array) -> Any:
        identity = stable_pytree_sha256(actor_params)
        core = candidate_cores.get(identity)
        if core is None:
            core = prepared_engine.prepare_candidate_rollout(actor_params)
            candidate_cores[identity] = core
        trajectory, _final_states = core(action_noise)
        return trajectory

    phase_rollout = make_phase_rollout(
        env,
        actor_apply,
        normalizer=normalizer,
        max_steps=env.reference_transitions,
    )
    return PreparedAuditExecution(
        checkpoint_state=checkpoint_state,
        actor_apply=actor_apply,
        normalizer_state=checkpoint_state.normalizer,
        estimate_shard=estimate_shard,
        prepare_first_action_objective=(
            prepared_engine.prepare_first_action_objective
        ),
        stochastic_rollout=stochastic_rollout,
        phase_rollout=phase_rollout,
        validated_contract=validated,
        algorithmic_validity=_run_algorithmic_validity_checks(),
        external_inputs=external_inputs,
        runtime_provenance=runtime_provenance,
        gradient_solver_context=fixed_mjx_solver_outer_loop,
        render_phase_zero=lambda *, rows, output_dir: _render_phase_zero_videos(
            env, rows=rows, output_dir=output_dir
        ),
    )


def run_audit(contract: Any) -> dict[str, Any]:
    """Execute the frozen E064 audit and atomically publish its evidence bundle."""
    from tools.audit_g1_shac_gradient_quality import (
        sha256_file,
        write_json_atomically,
        write_npz_atomically,
        write_pickle_atomically,
    )

    jax.config.update("jax_enable_x64", True)
    _validate_execution_contract(contract)
    if (Path(contract.output_dir) / "manifest.json").exists():
        raise FileExistsError("audit manifest already exists; refusing to overwrite it")
    prepared = _prepare_e064_execution(contract)
    estimator_receipts: dict[str, Any] = {}

    def estimate_for_runner(*, seed: int) -> Any:
        with prepared.gradient_solver_context():
            evidence = prepared.estimate_shard(seed)
        pathwise = to_finite_json(dict(evidence.pathwise_receipt))
        score = to_finite_json(dict(evidence.score_receipt))
        if pathwise != score:
            raise ValueError(f"shard {seed} estimator identity receipts differ")
        estimator_receipts[str(seed)] = {
            "pathwise": pathwise,
            "score": score,
        }
        return evidence.result

    aggregation = aggregate_four_shards(
        shard_seeds=contract.shard_seeds,
        estimate_shared_gradients=estimate_for_runner,
        estimate_kwargs={},
        pathwise_clip_norm=contract.per_env_clip,
        bootstrap_confidence_level=E064_OUTCOME_THRESHOLDS.bootstrap_confidence_level,
    )
    candidates = build_descent_candidates(
        actor_apply=prepared.actor_apply,
        actor_params=prepared.checkpoint_state.actor_params,
        pathwise_gradient=aggregation.pathwise_mean,
        score_gradient=aggregation.score_mean,
        normalized_observations=aggregation.normalized_observations,
        target_rms=contract.functional_rms,
    )
    candidate_parameters = {
        label: getattr(candidates, label) for label in _CANDIDATE_LABELS
    }

    stochastic_per_seed = []
    for seed in contract.held_out_seeds:
        action_noise = make_frozen_action_noise(seed)
        summaries = {}
        for label in _CANDIDATE_LABELS:
            with prepared.gradient_solver_context():
                trajectory = prepared.stochastic_rollout(
                    candidate_parameters[label], action_noise
                )
            if hasattr(trajectory, "trajectory"):
                trajectory = trajectory.trajectory
            summaries[label] = summarize_stochastic_rollout(
                trajectory,
                gamma=contract.gamma,
                expected_population=contract.population,
                expected_horizon=contract.horizon,
                expected_action_dimension=_ACTION_DIMENSION,
                expected_observation_dimension=154,
            )
        stochastic_per_seed.append(
            {
                "seed": int(seed),
                "action_noise_sha256": _array_sha256(action_noise),
                "candidates": summaries,
            }
        )
    heldout_stochastic_finite_complete = bool(
        len(stochastic_per_seed) == 4
        and all(
            summary["trajectory_complete"] is True
            for seed_row in stochastic_per_seed
            for summary in seed_row["candidates"].values()
        )
    )
    stochastic_evaluation = to_finite_json(
        {
            "held_out_seeds": contract.held_out_seeds,
            "common_random_numbers": True,
            "finite_complete": heldout_stochastic_finite_complete,
            "per_seed": stochastic_per_seed,
        }
    )

    ordinary_rows: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    ordinary_arrays: dict[str, np.ndarray] = {}

    # The ordinary transfer gate is one deterministic grid, not four
    # pseudo-replicates of identical evidence.  Seed 0 is the established
    # exact-phase reset key; stochastic CRN seeds 4--7 remain separate above.
    ordinary_seed = 0
    ordinary_candidates = {}
    for label in _CANDIDATE_LABELS:
        rows = []
        for phase in contract.phases:
            raw_row = prepared.phase_rollout(
                candidate_parameters[label],
                prepared.normalizer_state,
                seed=ordinary_seed,
                phase=phase,
            )
            summary, arrays = _validate_ordinary_row(raw_row)
            ordinary_rows[(ordinary_seed, label, phase)] = dict(raw_row)
            for field_name, array in arrays.items():
                key = f"seed_{ordinary_seed}/{label}/phase_{phase}/{field_name}"
                ordinary_arrays[key] = array
            rows.append(summary)
        ordinary_candidates[label] = rows
    ordinary_evaluation = to_finite_json(
        {
            "mode": "single-deterministic-five-phase-grid",
            "seed": ordinary_seed,
            "phases": contract.phases,
            "per_seed": [{"seed": ordinary_seed, "candidates": ordinary_candidates}],
        }
    )
    actual_baseline_survival = [
        int(row["survival"]) for row in ordinary_candidates["baseline"]
    ]
    baseline_reproduction = {
        **_baseline_competence_receipt(
            phases=contract.phases,
            actual_survival=actual_baseline_survival,
        ),
        "seed": ordinary_seed,
    }

    proof_keys = {
        "analytic_gaussian_sign",
        "detachment",
        "done_boundary_return",
        "ppo_ratio_one",
        "pytree_order",
        "smooth_toy_convergence",
    }
    if set(prepared.algorithmic_validity) != proof_keys:
        raise ValueError("algorithmic validity receipt key mismatch")
    validity = {
        "frozen_hashes": True,
        "aggregate_gradients_finite_nonzero": (
            _tree_is_finite_nonzero(aggregation.pathwise_mean)
            and _tree_is_finite_nonzero(aggregation.score_mean)
        ),
        "candidate_trees_finite_nonzero": all(
            _candidate_delta_is_finite_nonzero(
                candidates.baseline, candidate_parameters[label]
            )
            for label in ("pathwise", "score")
        ),
        **{key: value is True for key, value in prepared.algorithmic_validity.items()},
        "rollouts_fresh_replay_free_complete_finite": bool(
            heldout_stochastic_finite_complete
            and baseline_reproduction["competence_floor_pass"]
            and all(
                bool(row["complete"]) and bool(row["replay_free"])
                for rows in ordinary_candidates.values()
                for row in rows
            )
        ),
    }
    outcome = classify_preregistered_outcome(
        geometry=aggregation.geometry,
        evaluation=ordinary_evaluation,
        thresholds=E064_OUTCOME_THRESHOLDS,
        validity=validity,
    )

    output_dir = Path(contract.output_dir)
    documents = {
        "gradient_geometry.json": aggregation.geometry,
        "estimator_receipts.json": {
            "shared_rollout_identity": True,
            "per_shard": estimator_receipts,
        },
        "functional_step_receipt.json": candidates.functional_steps,
        "heldout_stochastic.json": stochastic_evaluation,
        "ordinary_phase_grid.json": ordinary_evaluation,
        "ordinary_baseline_reproduction.json": baseline_reproduction,
        "solver_trace_context.json": _SOLVER_TRACE_CONTEXT,
        "validity.json": validity,
        "outcome.json": outcome,
    }
    artifact_paths: list[Path] = []
    for label in _CANDIDATE_LABELS:
        candidate_path = output_dir / f"{label}_candidate.pkl"
        candidate_state = replace_actor_parameters(
            prepared.checkpoint_state, candidate_parameters[label]
        )
        write_pickle_atomically(candidate_path, candidate_state)
        artifact_paths.append(candidate_path)
    for relative, document in documents.items():
        path = output_dir / relative
        write_json_atomically(path, to_finite_json(document))
        artifact_paths.append(path)
    arrays_path = output_dir / "ordinary_phase_grid_arrays.npz"
    write_npz_atomically(arrays_path, ordinary_arrays)
    artifact_paths.append(arrays_path)

    if prepared.render_phase_zero is None:
        raise ValueError("required phase-zero renderer is not configured")
    videos = prepared.render_phase_zero(
        rows=ordinary_rows,
        output_dir=output_dir,
    )
    video_paths = {str(label): Path(path) for label, path in videos.items()}
    if set(video_paths) != set(_CANDIDATE_LABELS):
        raise ValueError("phase-zero renderer must return all candidate videos")
    if any(not path.is_file() for path in video_paths.values()):
        raise ValueError("phase-zero renderer returned a missing video")
    relative_video_paths = {
        label: str(path.resolve().relative_to(output_dir.resolve()))
        for label, path in video_paths.items()
    }
    artifact_paths.extend(video_paths.values())
    video_receipt = {
        "status": "produced",
        "seed": 0,
        "phase": 0,
        "paths": relative_video_paths,
    }

    artifacts = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(artifact_paths)
    }
    manifest = to_finite_json(
        {
            "schema_version": "g1-shac-gradient-audit/v1",
            "status": "complete",
            "contract": {
                "checkpoint": str(contract.checkpoint),
                "checkpoint_sha256": contract.checkpoint_sha256,
                "reference": str(contract.reference),
                "reference_sha256": contract.reference_sha256,
                "hparams_path": str(contract.hparams_path),
                "hparams_file_sha256": sha256_file(contract.hparams_path),
                "shard_seeds": contract.shard_seeds,
                "held_out_seeds": contract.held_out_seeds,
                "phases": contract.phases,
                "population": contract.population,
                "horizon": contract.horizon,
                "sigma": contract.sigma,
                "gamma": contract.gamma,
                "per_env_clip": contract.per_env_clip,
                "functional_rms": contract.functional_rms,
                "solver_iterations": contract.solver_iterations,
                "solver_ls_iterations": contract.solver_ls_iterations,
            },
            "validated_e064_contract": prepared.validated_contract,
            "external_inputs": prepared.external_inputs,
            "runtime_provenance": prepared.runtime_provenance,
            "solver_trace_context": _SOLVER_TRACE_CONTEXT,
            "thresholds": dataclasses.asdict(E064_OUTCOME_THRESHOLDS),
            "decision_rules": {
                "resampling_interval": (
                    "deterministic exhaustive four-shard resampling-stability interval"
                ),
                "pathwise_quality_limited_stability_gate": (
                    "score lower bound - pathwise upper bound >= 0.1"
                ),
                "ordinary_transfer_evidence": (
                    "one deterministic seed-0 grid; held-out stochastic fragments "
                    "are descriptive"
                ),
            },
            "canonical_metrics": {
                "return": "mean ordinary reward over active steps",
                "survival": "active steps",
            },
            "ordinary_baseline_reproduction": baseline_reproduction,
            "heldout_stochastic_finite_complete": heldout_stochastic_finite_complete,
            "outcome": outcome,
            "videos": video_receipt,
            "artifacts": artifacts,
        }
    )
    write_json_atomically(output_dir / "manifest.json", manifest)
    return manifest
