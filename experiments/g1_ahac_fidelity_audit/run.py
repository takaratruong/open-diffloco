"""Audit official AHAC contact semantics on the retained E002 state bank.

This diagnostic performs no optimization.  It advances the exact saved
effective-512 E002 population with one prospective H64 random tape, compares
the legacy root generalized-force proxy with the official all-body spatial
metric, and measures the projected horizon response implied by each scale.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import pickle
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.ahac import duplicate_single_critic_params
from src.algorithms.shac.counterfactual_wrench_distillation import (
    parameter_tree_sha256,
)
from src.core.data_structures import Normalizer
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.evaluate_g1_flax_phase_grid import (
    load_checkpoint_environment_contract,
)
from tools.evaluate_g1_tracking import (
    _load_policy,
    make_evaluation_env,
    training_action_noise_at_step,
)
from tools.run_g1_dual_scale_root_position import (
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_HPARAMS_SHA256,
    sha256_file,
    validate_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
EFFECTIVE_NUM_ENVS = 512
MEASURED_STEPS = 24
PROSPECTIVE_MAX_HORIZON = 64
OFFICIAL_MIN_HORIZON = 8
PRESERVED_START_HORIZON = 24
OFFICIAL_CONTACT_THRESHOLD = 500.0
OFFICIAL_DUAL_LR = 5e-4
LOCAL_UPDATE_BUDGET = 128
UPSTREAM_DEFAULT_UPDATE_BUDGET = 2_000
UPSTREAM_AHAC_SOURCE_SHA256 = (
    "e363f854a943ca43f39fca4ac33deaae70fcb67921f0136492760dd5e48cf1c1"
)
UPSTREAM_AHAC_CONFIG_SHA256 = (
    "5d07ea0b3f8cfaae11064e09f318c404c36c88d1fa4ac3bb389fad1d6f293483"
)
AHAC_PAPER_SHA256 = (
    "8f5624c8a3ec43e05d1c01159c82a6d72563064d671035036032b0a4ed92dae7"
)
DETERMINISTIC_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops"
REWARD_PARITY_ATOL = 1e-12
GRADIENT_PROBE_ENVS = 8
CONTACT_VALUE_PARITY_ATOL = 1e-10


def _percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("contact distribution must be finite and nonempty")
    quantiles = np.percentile(array, (0, 10, 25, 50, 75, 90, 95, 99, 100))
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
            quantiles,
            strict=True,
        )
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if not math.isfinite(float(denominator)) or denominator <= 0.0:
        raise ValueError("contact signals require positive finite norms")
    return float(np.dot(left, right) / denominator)


def summarize_contact_action_gradients(
    values: np.ndarray,
    gradients: np.ndarray,
    *,
    trace_values: np.ndarray,
) -> dict[str, object]:
    """Validate a bounded action-Jacobian probe of the spatial metric."""

    contact = np.asarray(values, dtype=np.float64)
    gradient = np.asarray(gradients, dtype=np.float64)
    trace = np.asarray(trace_values, dtype=np.float64)
    if (
        contact.ndim != 1
        or gradient.ndim != 2
        or gradient.shape[0] != contact.size
        or gradient.shape[1] < 1
        or trace.shape != contact.shape
        or contact.size < 2
        or not np.isfinite(contact).all()
        or not np.isfinite(gradient).all()
        or not np.isfinite(trace).all()
    ):
        raise ValueError("contact action-gradient probe is invalid")
    value_delta = np.abs(contact - trace)
    if np.any(value_delta > CONTACT_VALUE_PARITY_ATOL):
        raise ValueError("contact gradient values do not match the paired trace")
    squared_norms = np.sum(np.square(gradient), axis=1)
    rms_norm = float(np.sqrt(np.mean(squared_norms)))
    mean_gradient = np.mean(gradient, axis=0)
    mean_norm = float(np.linalg.norm(mean_gradient))
    variance_trace = max(rms_norm * rms_norm - mean_norm * mean_norm, 0.0)
    if rms_norm <= 0.0 or mean_norm <= 0.0:
        raise ValueError("contact action gradients must carry nonzero signal")
    return {
        "protocol": "g1-ahac-contact-action-gradient-v1",
        "valid": True,
        "probe_envs": int(contact.size),
        "action_dim": int(gradient.shape[1]),
        "finite_fraction": 1.0,
        "nonzero_fraction": float(np.mean(gradient != 0.0)),
        "contact_value_max_abs_trace_delta": float(np.max(value_delta)),
        "per_environment_norm": _percentiles(np.sqrt(squared_norms)),
        "population_mean_norm": mean_norm,
        "population_rms_norm": rms_norm,
        "population_variance_trace": variance_trace,
        "population_cancellation_ratio": mean_norm / rms_norm,
        "population_noise_scale": variance_trace / (mean_norm * mean_norm),
        "interpretation_boundary": (
            "Eight action Jacobians establish differentiability and a bounded "
            "signal check; they do not estimate the full actor-gradient variance."
        ),
    }


def inspect_upstream_ahac_semantics(source_text: str) -> dict[str, object]:
    """Locate the pinned code/paper actor-objective divergence structurally."""

    try:
        module = ast.parse(source_text)
    except SyntaxError as error:
        raise ValueError("upstream AHAC source is not valid Python") from error
    ahac_class = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "AHAC"
        ),
        None,
    )
    if ahac_class is None:
        raise ValueError("upstream AHAC class is missing")
    methods = {
        node.name: node
        for node in ahac_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not {"compute_actor_loss", "train"}.issubset(methods):
        raise ValueError("upstream AHAC methods are incomplete")

    def self_attributes(node: ast.AST) -> set[str]:
        return {
            child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "self"
        }

    actor_attributes = self_attributes(methods["compute_actor_loss"])
    train_attributes = self_attributes(methods["train"])
    actor_reads_contact = "cfs" in actor_attributes
    actor_reads_dual = "lambd" in actor_attributes
    actor_reads_threshold = "C" in actor_attributes
    train_updates_dual_and_horizon = {"cfs", "lambd", "C", "H"}.issubset(
        train_attributes
    )
    if not actor_reads_contact or not train_updates_dual_and_horizon:
        raise ValueError("upstream AHAC source layout is unexpected")
    return {
        "protocol": "pinned-upstream-ahac-semantics-audit-v1",
        "valid": True,
        "paper_actor_objective": "equation-10-contact-dual",
        "upstream_actor_reads_contact_signal": actor_reads_contact,
        "upstream_actor_reads_dual": actor_reads_dual,
        "upstream_actor_reads_contact_threshold": actor_reads_threshold,
        "upstream_train_updates_dual_and_horizon": (
            train_updates_dual_and_horizon
        ),
        "paper_source_actor_objective_divergence": bool(
            not actor_reads_dual or not actor_reads_threshold
        ),
        "selected_local_semantics": "paper_equation_10_no_target",
    }


def stationary_projected_horizon_response(
    contact_by_step: np.ndarray,
    *,
    threshold: float,
    initial_horizon: int,
    update_budget: int,
    minimum: int = OFFICIAL_MIN_HORIZON,
    maximum: int = PROSPECTIVE_MAX_HORIZON,
    learning_rate: float = OFFICIAL_DUAL_LR,
) -> dict[str, object]:
    """Replay the local projected dual rule until measurement support ends.

    This is explicitly a stationary-signal approximation, not a prediction of
    learned behavior.  It stops before an unmeasured rollout slot can affect an
    update.
    """

    contact = np.asarray(contact_by_step, dtype=np.float64)
    if (
        contact.ndim != 1
        or contact.size < 1
        or not np.isfinite(contact).all()
        or np.any(contact < 0.0)
        or isinstance(initial_horizon, bool)
        or not minimum <= initial_horizon <= min(maximum, contact.size)
        or isinstance(update_budget, bool)
        or update_budget < 1
        or not math.isfinite(float(threshold))
        or threshold <= 0.0
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0.0
    ):
        raise ValueError("stationary horizon response inputs are invalid")

    dual = np.zeros((maximum,), dtype=np.float64)
    padded_contact = np.zeros((maximum,), dtype=np.float64)
    padded_contact[: contact.size] = contact
    horizon = float(initial_horizon)
    first_rounded_extension_update = None
    simulated_updates = 0
    measurement_support_exhausted = False
    for update in range(1, update_budget + 1):
        rounded = int(np.floor(horizon + 0.5))
        if rounded > contact.size:
            measurement_support_exhausted = True
            break
        active = np.arange(maximum) < rounded
        dual = np.where(
            active,
            np.maximum(
                dual + learning_rate * (padded_contact - threshold),
                0.0,
            ),
            0.0,
        )
        horizon = float(
            np.clip(
                horizon + learning_rate * np.sum(dual),
                minimum,
                maximum,
            )
        )
        simulated_updates = update
        if (
            first_rounded_extension_update is None
            and int(np.floor(horizon + 0.5)) > initial_horizon
        ):
            first_rounded_extension_update = update

    return {
        "protocol": "g1-ahac-stationary-projected-response-v1",
        "approximation": "frozen-contact-population-means",
        "threshold": float(threshold),
        "learning_rate": float(learning_rate),
        "initial_horizon": initial_horizon,
        "requested_updates": update_budget,
        "simulated_updates": simulated_updates,
        "end_horizon": horizon,
        "end_rounded_horizon": int(np.floor(horizon + 0.5)),
        "first_rounded_extension_update": first_rounded_extension_update,
        "measurement_support_exhausted": measurement_support_exhausted,
        "measured_steps": int(contact.size),
        "dual_mean": float(np.mean(dual)),
        "dual_max": float(np.max(dual)),
    }


def summarize_contact_population(
    root_contact: np.ndarray,
    spatial_contact: np.ndarray,
    *,
    parity: Mapping[str, np.ndarray | bool],
) -> dict[str, object]:
    """Validate paired traces and expose the aggregation-scale mismatch."""

    root = np.asarray(root_contact, dtype=np.float64)
    spatial = np.asarray(spatial_contact, dtype=np.float64)
    if (
        root.ndim != 2
        or root.shape != spatial.shape
        or root.size < 1
        or not np.isfinite(root).all()
        or not np.isfinite(spatial).all()
        or np.any(root < 0.0)
        or np.any(spatial < 0.0)
    ):
        raise ValueError("paired contact traces must be matching finite matrices")
    required_parity = {
        "action_exact",
        "qpos_exact",
        "qvel_exact",
        "obs_exact",
        "reward_exact",
        "done_exact",
        "phase_exact",
        "rng_exact",
        "action_max_abs_delta",
        "qpos_max_abs_delta",
        "qvel_max_abs_delta",
        "obs_max_abs_delta",
        "reward_max_abs_delta",
    }
    if set(parity) != required_parity:
        raise ValueError("paired trajectory parity evidence is incomplete")
    exact_names = tuple(
        name for name in required_parity if name.endswith("_exact")
    )
    delta_names = tuple(name for name in required_parity if name.endswith("_delta"))
    exact = {
        name: np.asarray(parity[name], dtype=bool)
        for name in exact_names
    }
    deltas = {
        name: np.asarray(parity[name], dtype=np.float64)
        for name in delta_names
    }
    if any(
        value.shape != (root.shape[0],)
        for value in (*exact.values(), *deltas.values())
    ):
        raise ValueError("paired trajectory parity rows must match contact steps")
    physical_exact_names = tuple(
        name for name in exact_names if name != "reward_exact"
    )
    if any(not np.all(exact[name]) for name in physical_exact_names) or any(
        not np.isfinite(value).all() or np.any(value != 0.0)
        for name, value in deltas.items()
        if name != "reward_max_abs_delta"
    ):
        raise ValueError("contact metric selection changed the paired trajectory")
    reward_delta = deltas["reward_max_abs_delta"]
    if (
        not np.isfinite(reward_delta).all()
        or np.any(reward_delta > REWARD_PARITY_ATOL)
    ):
        raise ValueError("contact metric selection changed reward beyond roundoff")

    root_step_means = np.mean(root, axis=1)
    spatial_step_means = np.mean(spatial, axis=1)
    official_mean_exceedances = int(
        np.sum(spatial_step_means > OFFICIAL_CONTACT_THRESHOLD)
    )
    aggregation_matched_threshold = float(np.median(spatial_step_means))
    individual_sample_median = float(np.median(spatial))
    classification = (
        "official-C500-inert-in-mjx-scale"
        if official_mean_exceedances == 0
        else "official-C500-active-in-mjx-scale"
    )

    response_cases = {}
    for label, threshold in (
        ("official_C500", OFFICIAL_CONTACT_THRESHOLD),
        ("aggregation_matched_step_mean_median", aggregation_matched_threshold),
        ("individual_sample_median_heuristic", individual_sample_median),
    ):
        response_cases[label] = {
            "initial_H8_local_128": stationary_projected_horizon_response(
                spatial_step_means,
                threshold=threshold,
                initial_horizon=OFFICIAL_MIN_HORIZON,
                update_budget=LOCAL_UPDATE_BUDGET,
            ),
            "initial_H8_upstream_default_2000": (
                stationary_projected_horizon_response(
                    spatial_step_means,
                    threshold=threshold,
                    initial_horizon=OFFICIAL_MIN_HORIZON,
                    update_budget=UPSTREAM_DEFAULT_UPDATE_BUDGET,
                )
            ),
            "initial_H24_local_128": stationary_projected_horizon_response(
                spatial_step_means,
                threshold=threshold,
                initial_horizon=PRESERVED_START_HORIZON,
                update_budget=LOCAL_UPDATE_BUDGET,
            ),
        }

    return {
        "protocol": "g1-ahac-fidelity-audit-v1",
        "valid": True,
        "classification": classification,
        "optimizer_updates": 0,
        "policy_retained": False,
        "retained_policy": None,
        "population_shape": list(root.shape),
        "population_reduction": "mean-over-environments-for-each-rollout-step",
        "paired_physical_trajectory_exact": True,
        "paired_reward_bit_exact": bool(np.all(exact["reward_exact"])),
        "paired_reward_parity_atol": REWARD_PARITY_ATOL,
        "paired_reward_within_tolerance": True,
        "paired_trajectory_max_abs_delta": {
            name: float(np.max(value)) for name, value in deltas.items()
        },
        "root_generalized": {
            "individual_distribution": _percentiles(root),
            "step_mean_distribution": _percentiles(root_step_means),
            "step_means": root_step_means.tolist(),
        },
        "all_body_spatial": {
            "individual_distribution": _percentiles(spatial),
            "step_mean_distribution": _percentiles(spatial_step_means),
            "step_means": spatial_step_means.tolist(),
        },
        "root_spatial_flat_cosine": _cosine(root, spatial),
        "official_threshold": OFFICIAL_CONTACT_THRESHOLD,
        "official_threshold_individual_exceedance_count": int(
            np.sum(spatial > OFFICIAL_CONTACT_THRESHOLD)
        ),
        "official_threshold_step_mean_exceedance_count": official_mean_exceedances,
        "aggregation_matched_calibration_proposal": {
            "statistic": "median-of-per-step-effective-512-means",
            "threshold": aggregation_matched_threshold,
            "status": "mechanism-scale-proposal-not-performance-result",
        },
        "individual_sample_median": {
            "threshold": individual_sample_median,
            "status": "heuristic-only-aggregation-mismatched",
        },
        "stationary_projected_horizon_responses": response_cases,
        "interpretation_boundary": (
            "Stationary responses test mechanical activation only; changing "
            "policies and state distributions make them non-predictive of behavior."
        ),
    }


def critic_migration_receipt(state: object) -> dict[str, object]:
    """Verify exact duplication of both legacy single-critic trees."""

    source = state.critic_params
    source_target = state.target_critic_params
    migrated = duplicate_single_critic_params(source)
    migrated_target = duplicate_single_critic_params(source)
    source_hash = parameter_tree_sha256(source["params"])
    target_hash = parameter_tree_sha256(source_target["params"])
    head_hashes = [
        parameter_tree_sha256(migrated["params"][name])
        for name in ("critic_0", "critic_1")
    ]
    target_head_hashes = [
        parameter_tree_sha256(migrated_target["params"][name])
        for name in ("critic_0", "critic_1")
    ]
    valid = (
        head_hashes == [source_hash, source_hash]
        and target_head_hashes == [source_hash, source_hash]
    )
    if not valid:
        raise ValueError("legacy critic duplication is not exact")
    return {
        "protocol": "g1-legacy-shac-critic-duplication-audit-v1",
        "valid": True,
        "source_critic_sha256": source_hash,
        "source_target_critic_sha256": target_hash,
        "critic_head_sha256": head_hashes,
        "target_critic_head_sha256": target_head_hashes,
        "critic_optimizer_would_reset": True,
        "actor_and_actor_optimizer_would_be_preserved": True,
        "ahac_semantics": "paper_equation_10_no_target",
        "uses_delayed_target_critic": False,
        "legacy_target_would_be_discarded": True,
        "migration_executed": False,
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _validate_external_evidence(
    *, source: Path, config: Path, paper: Path
) -> dict[str, object]:
    expected = {
        source: UPSTREAM_AHAC_SOURCE_SHA256,
        config: UPSTREAM_AHAC_CONFIG_SHA256,
        paper: AHAC_PAPER_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"external AHAC evidence hash mismatch: {path}")
    return {
        "upstream_ahac_source": str(source),
        "upstream_ahac_source_sha256": expected[source],
        "upstream_ahac_config": str(config),
        "upstream_ahac_config_sha256": expected[config],
        "ahac_paper": str(paper),
        "ahac_paper_sha256": expected[paper],
        "upstream_default_update_budget": UPSTREAM_DEFAULT_UPDATE_BUDGET,
        "upstream_semantics": inspect_upstream_ahac_semantics(
            source.read_text(encoding="utf-8")
        ),
    }


def _validate_runtime() -> dict[str, object]:
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    if DETERMINISTIC_XLA_FLAG not in xla_flags:
        raise ValueError("AHAC audit requires deterministic XLA reductions")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("AHAC audit requires exactly one visible GPU")
    if not bool(jax.config.jax_enable_x64):
        raise ValueError("AHAC audit requires JAX float64 mode")
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        "jax_enable_x64": True,
        "xla_deterministic_reductions": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _make_metric_env(
    hparams: Mapping[str, object],
    reference_path: Path,
    *,
    metric: str,
):
    return make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        reference_path=reference_path,
        reference_stride=int(hparams["reference_stride"]),
        actor_history_len=int(hparams["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            int(value) for value in hparams["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(hparams["actor_reference_preview_mode"]),
        actor_observe_motion_anchor_position=bool(
            hparams.get("actor_observe_motion_anchor_position", False)
        ),
        tracking_velocity_kernel=str(hparams["tracking_velocity_kernel"]),
        tracking_anchor_position_kernel=str(
            hparams.get("tracking_anchor_position_kernel", "exponential")
        ),
        tracking_torso_orientation_weight=float(
            hparams["tracking_torso_orientation_weight"]
        ),
        tracking_root_velocity_weight=float(hparams["tracking_root_velocity_weight"]),
        actor_observation_noise=bool(hparams["actor_observation_noise"]),
        domain_randomization=bool(hparams["domain_randomization"]),
        friction_range=tuple(float(value) for value in hparams["friction_range"]),
        kp_range=tuple(float(value) for value in hparams["kp_range"]),
        kd_range=tuple(float(value) for value in hparams["kd_range"]),
        reference_reset_noise_scale=float(hparams["reference_reset_noise_scale"]),
        reference_residual_control=bool(hparams["reference_residual_control"]),
        reference_residual_scale=float(hparams["reference_residual_scale"]),
        contact_stiffness_metric=metric,
    )


def collect_paired_contact_trace(
    *, checkpoint: Path, reference_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Advance both metric modes on one exact prospective H64 random tape."""

    hparams_path = checkpoint.with_name("hparams.json")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    contract = load_checkpoint_environment_contract(checkpoint)
    expected_hparams = {
        "effective_num_envs": EFFECTIVE_NUM_ENVS,
        "unroll_length": MEASURED_STEPS,
        "solver_profile": "g1-4x5",
        "actor_observation_noise": False,
        "domain_randomization": False,
        "friction_range": [1.0, 1.0],
        "kp_range": [35.0, 35.0],
        "kd_range": [0.5, 0.5],
        "reference_reset_noise_scale": 0.0,
        "tracking_root_velocity_weight": 1.0,
        "ahac": False,
    }
    if any(hparams.get(name) != value for name, value in expected_hparams.items()):
        raise ValueError("retained E002 hparams do not match the audit boundary")
    if contract["env_variant"] != hparams["env_variant"]:
        raise ValueError("checkpoint environment contract does not close")

    root_env = _make_metric_env(
        hparams, reference_path, metric="root_generalized"
    )
    spatial_env = _make_metric_env(
        hparams, reference_path, metric="all_body_spatial"
    )
    if (
        root_env.contact_stiffness_metric != "root_generalized"
        or spatial_env.contact_stiffness_metric != "all_body_spatial"
        or root_env.actor_obs_dim != spatial_env.actor_obs_dim
        or root_env.action_dim != spatial_env.action_dim
    ):
        raise ValueError("paired metric environments are incompatible")

    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != START_STEP
        or np.asarray(state.env_state.obs).shape[0] != EFFECTIVE_NUM_ENVS
    ):
        raise ValueError("retained E002 checkpoint state is invalid")
    actor, actor_params, normalizer_state = _load_policy(root_env, checkpoint, 0)
    actor_normalizer = Normalizer(root_env.actor_frame_obs_dim)
    action_noise_std = jnp.asarray(
        training_action_noise_at_step(
            hparams,
            START_STEP,
            action_dim=root_env.action_dim,
        ),
        dtype=jnp.float64,
    )
    split_keys = jax.random.split(state.key, 6)
    prospective_noise = jax.random.normal(
        split_keys[1],
        (EFFECTIVE_NUM_ENVS, PROSPECTIVE_MAX_HORIZON, root_env.action_dim),
        dtype=jnp.float64,
    )
    scan_noise = jnp.swapaxes(
        prospective_noise[:, :MEASURED_STEPS, :], 0, 1
    )

    def prepare_action(environment, env_state, epsilon):
        rng_pairs = jax.vmap(lambda key: jax.random.split(key, 2))(
            env_state.info["rng"]
        )
        obs_rng = rng_pairs[:, 0]
        env_rng = rng_pairs[:, 1]
        env_state = env_state.replace(
            info={**env_state.info, "rng": env_rng}
        )
        actor_obs = jax.vmap(environment._apply_obs_noise)(
            env_state.obs, obs_rng
        )
        normalized = environment.normalize_actor_obs(
            actor_normalizer,
            normalizer_state,
            actor_obs,
        ).astype(jnp.float32)
        action = jax.vmap(
            lambda observation: actor.apply(actor_params, observation)
        )(normalized).astype(jnp.float64)
        noisy_action = action + epsilon * action_noise_std
        if environment.clip_sampled_actor_actions:
            noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
        return env_state, noisy_action

    def scan_step(carry, epsilon):
        root_state, spatial_state = carry
        root_state, root_action = prepare_action(
            root_env, root_state, epsilon
        )
        spatial_state, spatial_action = prepare_action(
            spatial_env, spatial_state, epsilon
        )
        next_root = jax.vmap(root_env.step)(root_state, root_action)
        next_spatial = jax.vmap(spatial_env.step)(
            spatial_state, spatial_action
        )

        def exact(left, right):
            return jnp.all(left == right)

        def max_delta(left, right):
            return jnp.max(jnp.abs(left - right))

        row = {
            "root_contact_stiffness": next_root.info[
                "transition_contact_stiffness"
            ],
            "all_body_spatial_contact_stiffness": next_spatial.info[
                "transition_contact_stiffness"
            ],
            "action_exact": exact(root_action, spatial_action),
            "qpos_exact": exact(next_root.data.qpos, next_spatial.data.qpos),
            "qvel_exact": exact(next_root.data.qvel, next_spatial.data.qvel),
            "obs_exact": exact(next_root.obs, next_spatial.obs),
            "reward_exact": exact(next_root.reward, next_spatial.reward),
            "done_exact": exact(next_root.done, next_spatial.done),
            "phase_exact": exact(
                next_root.info["phase"], next_spatial.info["phase"]
            ),
            "rng_exact": exact(
                next_root.info["rng"], next_spatial.info["rng"]
            ),
            "action_max_abs_delta": max_delta(root_action, spatial_action),
            "qpos_max_abs_delta": max_delta(
                next_root.data.qpos, next_spatial.data.qpos
            ),
            "qvel_max_abs_delta": max_delta(
                next_root.data.qvel, next_spatial.data.qvel
            ),
            "obs_max_abs_delta": max_delta(
                next_root.obs, next_spatial.obs
            ),
            "reward_max_abs_delta": max_delta(
                next_root.reward, next_spatial.reward
            ),
        }
        return (next_root, next_spatial), row

    paired_scan = jax.jit(
        lambda initial, noise: jax.lax.scan(
            scan_step,
            (initial, initial),
            noise,
            length=MEASURED_STEPS,
        )
    )
    spatial_probe_state, spatial_probe_action = prepare_action(
        spatial_env,
        state.env_state,
        scan_noise[0],
    )
    spatial_probe_state = jax.tree.map(
        lambda value: value[:GRADIENT_PROBE_ENVS],
        spatial_probe_state,
    )
    spatial_probe_action = spatial_probe_action[:GRADIENT_PROBE_ENVS]

    def spatial_contact_from_action(env_state, action):
        return spatial_env.step(env_state, action).info[
            "transition_contact_stiffness"
        ]

    spatial_value_and_action_gradient = jax.jit(
        jax.vmap(
            jax.value_and_grad(spatial_contact_from_action, argnums=1),
            in_axes=(0, 0),
        )
    )
    spatial_probe_forward = jax.jit(
        jax.vmap(spatial_contact_from_action, in_axes=(0, 0))
    )
    profile = get_solver_profile(str(hparams["solver_profile"]))
    with solver_context(profile):
        (_, _), trace = paired_scan(state.env_state, scan_noise)
        (
            spatial_gradient_values,
            spatial_action_gradients,
        ) = spatial_value_and_action_gradient(
            spatial_probe_state,
            spatial_probe_action,
        )
        spatial_forward_values = spatial_probe_forward(
            spatial_probe_state,
            spatial_probe_action,
        )
    trace = jax.tree_util.tree_map(lambda value: np.asarray(value), trace)
    arrays = {
        **trace,
        "spatial_gradient_probe_contact": np.asarray(
            spatial_gradient_values
        ),
        "spatial_gradient_probe_forward_contact": np.asarray(
            spatial_forward_values
        ),
        "spatial_contact_action_gradient": np.asarray(
            spatial_action_gradients
        ),
        "prospective_action_noise": np.asarray(prospective_noise),
        "action_noise_std": np.asarray(action_noise_std),
        "start_phase": np.asarray(state.env_state.info["phase"]),
        "checkpoint_key": np.asarray(state.key),
    }
    migration = critic_migration_receipt(state)
    return arrays, migration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--upstream-ahac-source", type=Path, required=True)
    parser.add_argument("--upstream-ahac-config", type=Path, required=True)
    parser.add_argument("--ahac-paper", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("AHAC fidelity audit seed must equal zero")
    configure_jax()
    runtime = _validate_runtime()
    repository = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_path.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=args.code_commit,
    )
    preflight.update(
        protocol="g1-ahac-fidelity-preflight-v1",
        hparams_sha256=SOURCE_HPARAMS_SHA256,
        runtime=runtime,
        measured_steps=MEASURED_STEPS,
        prospective_max_horizon=PROSPECTIVE_MAX_HORIZON,
        gradient_probe_envs=GRADIENT_PROBE_ENVS,
        optimizer_updates=0,
        external_evidence=_validate_external_evidence(
            source=args.upstream_ahac_source.resolve(),
            config=args.upstream_ahac_config.resolve(),
            paper=args.ahac_paper.resolve(),
        ),
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    arrays, migration = collect_paired_contact_trace(
        checkpoint=checkpoint,
        reference_path=reference,
    )
    parity = {
        name: arrays[name]
        for name in (
            "action_exact",
            "qpos_exact",
            "qvel_exact",
            "obs_exact",
            "reward_exact",
            "done_exact",
            "phase_exact",
            "rng_exact",
            "action_max_abs_delta",
            "qpos_max_abs_delta",
            "qvel_max_abs_delta",
            "obs_max_abs_delta",
            "reward_max_abs_delta",
        )
    }
    result = summarize_contact_population(
        arrays["root_contact_stiffness"],
        arrays["all_body_spatial_contact_stiffness"],
        parity=parity,
    )
    result.update(
        source_step=START_STEP,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_hparams_sha256=SOURCE_HPARAMS_SHA256,
        prospective_noise_shape=list(arrays["prospective_action_noise"].shape),
        critic_migration=migration,
        contact_action_gradient=summarize_contact_action_gradients(
            arrays["spatial_gradient_probe_contact"],
            arrays["spatial_contact_action_gradient"],
            trace_values=arrays["spatial_gradient_probe_forward_contact"],
        ),
        external_evidence=preflight["external_evidence"],
    )
    paired_gradient_value_delta = np.abs(
        arrays["spatial_gradient_probe_contact"]
        - arrays["all_body_spatial_contact_stiffness"][
            0, :GRADIENT_PROBE_ENVS
        ]
    )
    result["contact_action_gradient"].update(
        paired_graph_contact_value_max_abs_delta=float(
            np.max(paired_gradient_value_delta)
        ),
        paired_graph_contact_value_comparison=(
            "reported-cross-compilation-diagnostic-not-an-equality-gate"
        ),
    )

    trace_path = output_root / "ahac_fidelity_trace.npz"
    result_path = output_root / "ahac_fidelity_audit.json"
    _atomic_npz(trace_path, arrays)
    result["trace_sha256"] = sha256_file(trace_path)
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": "g1-ahac-fidelity-completion-v1",
        "valid": True,
        "classification": result["classification"],
        "optimizer_updates": 0,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "ahac_fidelity_trace.npz": sha256_file(trace_path),
            "ahac_fidelity_audit.json": sha256_file(result_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(output_root)


if __name__ == "__main__":
    main()
