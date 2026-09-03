"""Materialize and reject-by-default the first retained-E002 AHAC proposal.

The proposal and evaluation execute in separate child processes so the large
training executable is released before paired carried-state and replay-free
behavior are evaluated.  No candidate is published as resumable or retained.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experiments.g1_e002_critic_calibration_audit.run import (
    _distribution,
    _make_environment,
    first_terminal_returns,
)
from src.algorithms.shac.algorithm import (
    resolve_core_optimizer_resume_settings,
    train,
)
from src.algorithms.shac.frozen_controller_residual import (
    FrozenControllerResidualOptState,
    FrozenControllerResidualParams,
)
from src.core.data_structures import Normalizer
from src.envs.g1_tracking.solver_profiles import (
    get_solver_profile,
    solver_context,
)
from tools.compare_g1_tracking_residual import rollout
from tools.evaluate_g1_e038_recovery_transfer import parameter_tree_sha256
from tools.evaluate_g1_flax_phase_grid import (
    load_checkpoint_environment_contract,
)
from tools.evaluate_g1_rmr_phase_grid import build_phase_grid_summary
from tools.evaluate_g1_tracking import (
    _load_policy,
    build_compiled_step,
    make_evaluation_env,
    prepare_evaluation_action,
)
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_ahac_walk import validate_ahac_telemetry_row
from tools.run_g1_dual_scale_root_position import (
    SOURCE_CHECKPOINT_SHA256,
    build_arm_kwargs,
    validate_preflight as validate_e002_preflight,
)
from tools.run_g1_tracking_shac import configure_jax
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


START_STEP = 1_867_776
TRANSITIONS_PER_UPDATE = 512 * 24
END_STEP = START_STEP + TRANSITIONS_PER_UPDATE
CONTACT_THRESHOLD = 116.33409086875099
CANDIDATE_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
PHASES = (0, 25, 50, 75, 100)
E002_SURVIVAL = (136, 144, 84, 90, 79)
DETERMINISTIC_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops"
RESUME_AUDIT_SHA256 = (
    "a80e87ceddd8ad97fe98884158fe5855181f4a7ee2b9a30837e1ccffbd2a8425"
)
AHAC_FIDELITY_AUDIT_SHA256 = (
    "986d7466daa9016cf52c82d0de26b239234561fcfa01b37d1825e9aa4aa66dbf"
)
CRITIC_CALIBRATION_AUDIT_SHA256 = (
    "6fdd17559bec03ac0739ca26c9aaefcf5711d0406f1e8e7fdd7b906ae53f8cf9"
)
_CARRIED_KEYS = (
    "mean_return",
    "survival_p10",
    "survival_median",
    "survival_mean",
)


def _tree_exact(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    return left_structure == right_structure and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_delta(source: Any, target: Any) -> dict[str, float | int]:
    source_leaves, source_structure = jax.tree_util.tree_flatten(source)
    target_leaves, target_structure = jax.tree_util.tree_flatten(target)
    if source_structure != target_structure:
        raise ValueError("parameter structures differ")
    source_sq = 0.0
    delta_sq = 0.0
    maximum = 0.0
    changed = 0
    count = 0
    for source_leaf, target_leaf in zip(
        source_leaves, target_leaves, strict=True
    ):
        start = np.asarray(source_leaf, dtype=np.float64)
        end = np.asarray(target_leaf, dtype=np.float64)
        if start.shape != end.shape or not (
            np.isfinite(start).all() and np.isfinite(end).all()
        ):
            raise ValueError("parameter leaves are incompatible or nonfinite")
        delta = end - start
        source_sq += float(np.sum(np.square(start)))
        delta_sq += float(np.sum(np.square(delta)))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0.0)))
        changed += int(np.any(delta != 0.0))
        count += int(start.size)
    delta_norm = math.sqrt(delta_sq)
    source_norm = math.sqrt(source_sq)
    return {
        "parameter_count": count,
        "changed_leaf_count": changed,
        "delta_l2": delta_norm,
        "relative_delta": delta_norm / max(source_norm, np.finfo(float).tiny),
        "max_abs_delta": maximum,
    }


def interpolate_actor_params(
    source: FrozenControllerResidualParams,
    proposal: FrozenControllerResidualParams,
    *,
    alpha: float,
) -> FrozenControllerResidualParams:
    """Interpolate only E002's newest adapter and freeze its complete parent."""

    if alpha not in CANDIDATE_ALPHAS:
        raise ValueError("alpha is not on the registered candidate grid")
    if not isinstance(source, FrozenControllerResidualParams) or not isinstance(
        proposal, FrozenControllerResidualParams
    ):
        raise ValueError("proposal audit requires nested frozen-controller actors")
    if not _tree_exact(source.parent, proposal.parent):
        raise ValueError("proposal changed the frozen parent actor")
    if jax.tree_util.tree_structure(source.adapter) != jax.tree_util.tree_structure(
        proposal.adapter
    ):
        raise ValueError("proposal adapter structure differs from source")
    adapter = (
        proposal.adapter
        if alpha == 1.0
        else jax.tree_util.tree_map(
            lambda start, end: start + alpha * (end - start),
            source.adapter,
            proposal.adapter,
        )
    )
    return FrozenControllerResidualParams(parent=source.parent, adapter=adapter)


def build_proposal_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
) -> dict[str, Any]:
    """Build exactly one paper-semantics AHAC migration proposal from E002."""

    kwargs = build_arm_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        kernel="exponential",
    )
    kwargs.update(
        total_steps=END_STEP,
        checkpoint_steps=(END_STEP,),
        ahac=True,
        allow_resume_ahac_change=True,
        ahac_horizon_min=8,
        ahac_horizon_max=24,
        ahac_contact_threshold=CONTACT_THRESHOLD,
        ahac_dual_lr=5e-4,
        ahac_critic_max_iterations=64,
        ahac_critic_tolerance=0.2,
        ahac_contact_metric="all_body_spatial",
        ahac_semantics="paper_equation_10_no_target",
        actor_bootstrap_scale=1.0,
        actor_bootstrap_delay_steps=0,
        allow_resume_actor_bootstrap_scale_change=True,
        use_lr_decay=False,
        allow_resume_core_optimizer_change=False,
    )
    return kwargs


def classify_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_phase_survival: Sequence[int],
    source_carried: Mapping[str, float],
) -> dict[str, Any]:
    """Require simultaneous non-regression on ordinary and carried states."""

    if tuple(source_phase_survival) != E002_SURVIVAL:
        raise ValueError("contemporaneous source does not reproduce retained E002")
    if any(
        key not in source_carried or not math.isfinite(float(source_carried[key]))
        for key in _CARRIED_KEYS
    ):
        raise ValueError("source carried summary is incomplete")
    if tuple(row.get("alpha") for row in rows) != CANDIDATE_ALPHAS:
        raise ValueError("candidate rows do not match the registered alpha grid")

    candidates: list[dict[str, Any]] = []
    for row in rows:
        phase = row.get("phase_survival")
        carried = row.get("carried")
        if (
            not isinstance(phase, list)
            or len(phase) != len(E002_SURVIVAL)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in phase
            )
            or not isinstance(carried, Mapping)
            or any(
                key not in carried or not math.isfinite(float(carried[key]))
                for key in _CARRIED_KEYS
            )
        ):
            raise ValueError("candidate behavior row is invalid")
        phase_safe = all(
            value >= source
            for value, source in zip(phase, source_phase_survival, strict=True)
        )
        carried_safe = all(
            float(carried[key]) >= float(source_carried[key])
            for key in _CARRIED_KEYS
        )
        advances = any(
            value > source
            for value, source in zip(phase, source_phase_survival, strict=True)
        ) or any(
            float(carried[key]) > float(source_carried[key])
            for key in _CARRIED_KEYS
        )
        candidates.append(
            {
                **dict(row),
                "phase_safe": phase_safe,
                "carried_safe": carried_safe,
                "eligible": phase_safe and carried_safe,
                "advances": advances,
            }
        )

    eligible = [row for row in candidates if row["eligible"]]
    selected = eligible[0] if eligible else None
    if selected is not None:
        outcome = (
            "ahac-proposal-safe-advances"
            if selected["advances"]
            else "ahac-proposal-preserves-only"
        )
    elif any(row["carried_safe"] for row in candidates):
        outcome = "ahac-proposal-carried-only"
    elif any(row["phase_safe"] for row in candidates):
        outcome = "ahac-proposal-phase-only"
    else:
        outcome = "ahac-proposal-unsafe"
    return {
        "protocol": "g1-e002-ahac-first-proposal-selection-v1",
        "valid": True,
        "source_phase_survival": list(source_phase_survival),
        "source_carried": {key: float(source_carried[key]) for key in _CARRIED_KEYS},
        "outcome": outcome,
        "selected_alpha": (
            float(selected["alpha"]) if selected is not None else None
        ),
        "candidates": candidates,
        "policy_retained": False,
        "retained_policy": None,
    }


def _validate_runtime() -> dict[str, Any]:
    if DETERMINISTIC_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise ValueError("proposal audit requires deterministic XLA reductions")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("proposal audit requires exactly one visible GPU")
    if not bool(jax.config.jax_enable_x64):
        raise ValueError("proposal audit requires JAX float64 mode")
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        "jax_enable_x64": True,
        "xla_deterministic_reductions": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_json_array(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return payload


def validate_preflight(
    *,
    repository: Path,
    checkpoint: Path,
    reference: Path,
    resume_audit: Path,
    ahac_fidelity_audit: Path,
    critic_calibration_audit: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Bind code, retained policy, calibrated contact, and failed value gate."""

    base = validate_e002_preflight(
        repository=repository,
        checkpoint=checkpoint,
        reference=reference,
        code_commit=code_commit,
    )
    evidence = (
        (resume_audit, RESUME_AUDIT_SHA256, "resume-state audit"),
        (
            ahac_fidelity_audit,
            AHAC_FIDELITY_AUDIT_SHA256,
            "AHAC fidelity audit",
        ),
        (
            critic_calibration_audit,
            CRITIC_CALIBRATION_AUDIT_SHA256,
            "critic calibration audit",
        ),
    )
    for path, expected_sha256, label in evidence:
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
    resume_payload = _read_json(resume_audit)
    fidelity = _read_json(ahac_fidelity_audit)
    calibration = _read_json(critic_calibration_audit)
    if (
        resume_payload.get("optimizer_updates") != 0
        or resume_payload.get("policy_retained") is not False
        or fidelity.get("valid") is not True
        or float(
            fidelity["aggregation_matched_calibration_proposal"]["threshold"]
        )
        != CONTACT_THRESHOLD
        or calibration.get("classification") != "online-critic-not-predictive"
        or calibration.get("policy_retained") is not False
    ):
        raise ValueError("bound AHAC evidence does not support this discriminator")

    kwargs = build_proposal_kwargs("g1-4x5", reference, 0, checkpoint)
    source_hparams = _read_json(checkpoint.with_name("hparams.json"))
    core_requested = {
        key: kwargs[key]
        for key in (
            "actor_lr",
            "critic_lr",
            "gamma",
            "gae_lambda",
            "target_update_rate",
            "critic_iterations",
            "use_lr_decay",
        )
    }
    core = resolve_core_optimizer_resume_settings(
        requested=core_requested,
        resumed_hparams=source_hparams,
        is_resume=True,
        allow_change=False,
    )
    return {
        **base,
        "protocol": "g1-e002-ahac-first-proposal-preflight-v1",
        "runtime": _validate_runtime(),
        "source_step": START_STEP,
        "proposal_step": END_STEP,
        "optimizer_updates": 1,
        "policy_retention_authorized": False,
        "core_optimizer_resume_contract": core,
        "resume_audit": str(resume_audit.resolve()),
        "resume_audit_sha256": RESUME_AUDIT_SHA256,
        "ahac_fidelity_audit": str(ahac_fidelity_audit.resolve()),
        "ahac_fidelity_audit_sha256": AHAC_FIDELITY_AUDIT_SHA256,
        "critic_calibration_audit": str(critic_calibration_audit.resolve()),
        "critic_calibration_audit_sha256": CRITIC_CALIBRATION_AUDIT_SHA256,
        "candidate_alphas": list(CANDIDATE_ALPHAS),
        "ordinary_phases": list(PHASES),
        "ordinary_source_floor": list(E002_SURVIVAL),
    }


def validate_training_artifacts(
    run_directory: Path,
    *,
    source_checkpoint: Path,
) -> dict[str, Any]:
    """Verify proposal-one order, migration, symmetry, and frozen actor parent."""

    hparams = _read_json(run_directory / "hparams.json")
    expected = {
        "algorithm": "ahac",
        "total_steps": END_STEP,
        "ahac": True,
        "allow_resume_ahac_change": True,
        "ahac_horizon_min": 8,
        "ahac_horizon_max": 24,
        "ahac_contact_threshold": CONTACT_THRESHOLD,
        "ahac_contact_metric": "all_body_spatial",
        "ahac_semantics": "paper_equation_10_no_target",
        "actor_bootstrap_scale": 1.0,
        "actor_bootstrap_delay_steps": 0,
        "allow_resume_actor_bootstrap_scale_change": True,
        "allow_resume_core_optimizer_change": False,
        "use_lr_decay": False,
    }
    for key, expected_value in expected.items():
        if hparams.get(key) != expected_value:
            raise ValueError(f"proposal hparam {key} does not match")
    rows = _read_json_array(run_directory / "checkpoint_phase_metrics.json")
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("proposal must contain exactly one checkpoint row")
    row = rows[0]
    if row.get("step") != END_STEP:
        raise ValueError("proposal checkpoint row has the wrong step")
    validate_ahac_telemetry_row(row, threshold=CONTACT_THRESHOLD)
    preview_drift = (
        "actor_preview_frozen_parameter_drift_max_abs",
        "actor_preview_frozen_moment_drift_max_abs",
        "actor_preview_normalizer_drift_max_abs",
    )
    preview_motion = (
        "actor_preview_gradient_norm",
        "actor_preview_update_norm",
    )
    if (
        float(row["ahac_horizon_before_update"]) != 8.0
        or float(row["ahac_actor_constraint_penalty"]) != 0.0
        or float(row["ahac_critic_head_disagreement"]) != 0.0
        or float(row.get("actor_bootstrap_scale_current", -1.0)) != 1.0
        or row.get("actor_cagrad_valid") is not True
        or row.get("actor_preview_valid") is not True
        or any(float(row.get(key, math.inf)) != 0.0 for key in preview_drift)
        or any(
            not math.isfinite(float(row.get(key, math.nan)))
            or float(row[key]) <= 0.0
            for key in preview_motion
        )
    ):
        raise ValueError("proposal one does not expose the audited initial order")

    migration_path = run_directory / "ahac_resume_migration.json"
    migration = _read_json(migration_path)
    head_hashes = migration.get("critic_head_sha256")
    if (
        migration.get("valid") is not True
        or migration.get("source_step") != START_STEP
        or migration.get("critic_optimizer_reset") is not True
        or migration.get("actor_and_actor_optimizer_preserved") is not True
        or migration.get("legacy_target_discarded") is not True
        or migration.get("horizon") != 8.0
        or migration.get("dual_all_zero") is not True
        or not isinstance(head_hashes, list)
        or len(head_hashes) != 2
        or head_hashes[0] != head_hashes[1]
    ):
        raise ValueError("SHAC-to-AHAC migration artifact is invalid")

    proposal_checkpoint = run_directory / f"checkpoint_step_{END_STEP}.pkl"
    if not proposal_checkpoint.is_file():
        raise ValueError("proposal checkpoint is missing")
    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    with proposal_checkpoint.open("rb") as stream:
        proposal = pickle.load(stream)
    if (
        int(proposal.step) != END_STEP
        or not isinstance(source.actor_params, FrozenControllerResidualParams)
        or not isinstance(proposal.actor_params, FrozenControllerResidualParams)
        or not isinstance(source.actor_opt, FrozenControllerResidualOptState)
        or not isinstance(proposal.actor_opt, FrozenControllerResidualOptState)
        or not _tree_exact(source.actor_params.parent, proposal.actor_params.parent)
        or not _tree_exact(
            source.actor_opt.parent_optimizer_state,
            proposal.actor_opt.parent_optimizer_state,
        )
        or not _tree_exact(source.normalizer, proposal.normalizer)
        or _tree_exact(source.actor_params.adapter, proposal.actor_params.adapter)
        or _tree_exact(
            source.actor_opt.adapter_optimizer_state,
            proposal.actor_opt.adapter_optimizer_state,
        )
    ):
        raise ValueError("proposal actor continuity contract is invalid")
    critic_heads = proposal.critic_params["params"]
    if set(critic_heads) != {"critic_0", "critic_1"} or not _tree_exact(
        critic_heads["critic_0"], critic_heads["critic_1"]
    ):
        raise ValueError("migrated critic heads are not exactly symmetric")
    return {
        "protocol": "g1-e002-ahac-first-proposal-training-v1",
        "valid": True,
        "run_directory": str(run_directory.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "proposal_checkpoint": str(proposal_checkpoint.resolve()),
        "proposal_checkpoint_sha256": sha256_file(proposal_checkpoint),
        "hparams": str((run_directory / "hparams.json").resolve()),
        "hparams_sha256": sha256_file(run_directory / "hparams.json"),
        "checkpoint_metrics": str(
            (run_directory / "checkpoint_phase_metrics.json").resolve()
        ),
        "checkpoint_metrics_sha256": sha256_file(
            run_directory / "checkpoint_phase_metrics.json"
        ),
        "migration_artifact": str(migration_path.resolve()),
        "migration_artifact_sha256": sha256_file(migration_path),
        "actor_delta": _tree_delta(source.actor_params, proposal.actor_params),
        "actor_parent_exact": True,
        "actor_parent_optimizer_exact": True,
        "actor_adapter_optimizer_changed": True,
        "actor_normalizer_exact": True,
        "critic_heads_exact": True,
        "initial_actor_constraint_penalty": 0.0,
        "telemetry": row,
        "policy_retained": False,
    }


def _make_phase_environment(checkpoint: Path, reference: Path):
    contract = load_checkpoint_environment_contract(checkpoint)
    profile = get_solver_profile(str(contract["solver_profile"]))
    environment = make_evaluation_env(
        str(contract["env_variant"]),
        solver_iterations=profile.iterations,
        solver_ls_iterations=profile.ls_iterations,
        reference_path=reference,
        reference_stride=int(contract["reference_stride"]),
        actor_history_len=int(contract["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            contract["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(
            contract["actor_reference_preview_mode"]
        ),
        actor_observe_motion_anchor_position=bool(
            contract["actor_observe_motion_anchor_position"]
        ),
        tracking_velocity_kernel=str(contract["tracking_velocity_kernel"]),
        tracking_anchor_position_kernel=str(
            contract["tracking_anchor_position_kernel"]
        ),
        tracking_torso_orientation_weight=float(
            contract["tracking_torso_orientation_weight"]
        ),
        tracking_root_velocity_weight=float(
            contract["tracking_root_velocity_weight"]
        ),
        reference_residual_control=bool(contract["reference_residual_control"]),
        reference_residual_scale=float(contract["reference_residual_scale"]),
    )
    return environment, profile


def evaluate_phase_grid(
    *,
    checkpoint: Path,
    reference: Path,
    actor_params: Sequence[FrozenControllerResidualParams],
    alphas: Sequence[float],
) -> dict[str, Any]:
    """Evaluate source and candidates with the established replay-free boundary."""

    environment, profile = _make_phase_environment(checkpoint, reference)
    actor, source_params, normalizer_state = _load_policy(environment, checkpoint, 0)
    if not _tree_exact(source_params, actor_params[0]):
        raise ValueError("loaded source actor does not match candidate source")
    normalizer = Normalizer(environment.actor_frame_obs_dim)
    compiled_step = build_compiled_step(environment)
    rows: list[dict[str, Any]] = []
    with solver_context(profile):
        for alpha, params in zip(alphas, actor_params, strict=True):
            def action(state):
                normalized = environment.normalize_actor_obs(
                    normalizer, normalizer_state, state.obs
                ).astype(jnp.float32)
                raw = actor.apply(params, normalized)
                return prepare_evaluation_action(
                    raw,
                    squash=bool(environment.clip_sampled_actor_actions),
                ).astype(jnp.float64)

            results = [
                {
                    "phase": phase,
                    **rollout(
                        environment,
                        action,
                        phase=phase,
                        seed=0,
                        max_steps=int(environment.reference_transitions) - phase,
                        step_fn=compiled_step,
                    ),
                }
                for phase in PHASES
            ]
            rows.append(
                {
                    "alpha": float(alpha),
                    "actor_params_sha256": parameter_tree_sha256(params),
                    "results": results,
                    "summary": build_phase_grid_summary(
                        results,
                        phases=PHASES,
                        reference_transitions=int(environment.reference_transitions),
                    ),
                }
            )
    if rows[0]["summary"]["survival"] != list(E002_SURVIVAL):
        raise ValueError("source phase grid does not reproduce retained E002")
    return {
        "protocol": "g1-e002-ahac-proposal-phase-grid-v1",
        "valid": True,
        "phases": list(PHASES),
        "source_floor": list(E002_SURVIVAL),
        "rows": rows,
    }


def evaluate_carried_population(
    *,
    checkpoint: Path,
    reference: Path,
    actor_params: Sequence[FrozenControllerResidualParams],
    alphas: Sequence[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Roll every actor from the exact same 512 carried simulator states."""

    hparams = _read_json(checkpoint.with_name("hparams.json"))
    with checkpoint.open("rb") as stream:
        source_state = pickle.load(stream)
    environment = _make_environment(hparams, reference)
    actor, source_params, normalizer_state = _load_policy(environment, checkpoint, 0)
    if (
        not _tree_exact(source_params, actor_params[0])
        or np.asarray(source_state.env_state.obs).shape[0] != 512
    ):
        raise ValueError("carried source state does not match retained E002")
    normalizer = Normalizer(environment.actor_frame_obs_dim)
    horizon = int(environment.max_episode_length)

    def carried_rollout(params, initial_state):
        def step(env_state, _):
            rng_pairs = jax.vmap(lambda key: jax.random.split(key, 2))(
                env_state.info["rng"]
            )
            obs_rng = rng_pairs[:, 0]
            env_rng = rng_pairs[:, 1]
            env_state = env_state.replace(info={**env_state.info, "rng": env_rng})
            actor_obs = jax.vmap(environment._apply_obs_noise)(
                env_state.obs, obs_rng
            )
            normalized = environment.normalize_actor_obs(
                normalizer, normalizer_state, actor_obs
            ).astype(jnp.float32)
            action = jax.vmap(lambda observation: actor.apply(params, observation))(
                normalized
            ).astype(jnp.float64)
            if environment.clip_sampled_actor_actions:
                action = jnp.clip(action, -1.0, 1.0)
            next_state = jax.vmap(environment.step)(env_state, action)
            return next_state, {
                "action": action,
                "reward": next_state.reward,
                "done": next_state.done,
                "terminal": next_state.info["terminal"],
                "xfrc_max": jnp.max(jnp.abs(env_state.data.xfrc_applied)),
            }

        return jax.lax.scan(step, initial_state, None, length=horizon)

    compiled = jax.jit(carried_rollout)
    summaries: list[dict[str, Any]] = []
    first_done_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    terminal_rows: list[np.ndarray] = []
    first_action_rows: list[np.ndarray] = []
    profile = get_solver_profile(str(hparams["solver_profile"]))
    with solver_context(profile):
        for alpha, params in zip(alphas, actor_params, strict=True):
            _, raw = compiled(params, source_state.env_state)
            trace = jax.tree.map(np.asarray, raw)
            if (
                not all(np.isfinite(value).all() for value in trace.values())
                or float(np.max(trace["xfrc_max"])) != 0.0
            ):
                raise ValueError("carried rollout is nonfinite or assisted")
            done = np.asarray(trace["done"], dtype=bool)
            if not np.all(np.any(done, axis=0)):
                raise ValueError("carried rollout did not reach a boundary")
            first_done = np.argmax(done, axis=0) + 1
            columns = np.arange(done.shape[1])
            terminal = np.asarray(trace["terminal"], dtype=bool)[
                first_done - 1, columns
            ]
            realized, alive = first_terminal_returns(
                trace["reward"], done, gamma=0.99
            )
            survival_distribution = _distribution(first_done)
            summaries.append(
                {
                    "alpha": float(alpha),
                    "actor_params_sha256": parameter_tree_sha256(params),
                    "mean_return": float(np.mean(realized[0])),
                    "survival_p10": survival_distribution["p10"],
                    "survival_median": survival_distribution["median"],
                    "survival_mean": survival_distribution["mean"],
                    "survival": survival_distribution,
                    "natural_terminal_count": int(np.sum(terminal)),
                    "truncation_count": int(np.sum(~terminal)),
                    "alive_transition_count": int(np.sum(alive)),
                    "mean_alive_reward": float(np.mean(trace["reward"][alive])),
                }
            )
            first_done_rows.append(first_done)
            return_rows.append(realized[0])
            terminal_rows.append(terminal)
            first_action_rows.append(trace["action"][0])

    first_actions = np.stack(first_action_rows)
    source_action = first_actions[0]
    for index, summary in enumerate(summaries):
        delta = first_actions[index] - source_action
        summary["initial_action_delta_rms"] = float(
            np.sqrt(np.mean(np.square(delta)))
        )
        summary["initial_action_delta_max_abs"] = float(np.max(np.abs(delta)))
    return (
        {
            "protocol": "g1-e002-ahac-proposal-carried-population-v1",
            "valid": True,
            "population_size": 512,
            "source_phase_min": int(np.min(source_state.env_state.info["phase"])),
            "source_phase_max": int(np.max(source_state.env_state.info["phase"])),
            "source_phase_unique": int(
                np.unique(source_state.env_state.info["phase"]).size
            ),
            "rows": summaries,
        },
        {
            "alphas": np.asarray(alphas, dtype=np.float64),
            "first_done": np.stack(first_done_rows),
            "return_from_start": np.stack(return_rows),
            "first_done_terminal": np.stack(terminal_rows),
            "first_action": first_actions,
        },
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_source_and_candidates(
    source_checkpoint: Path, proposal_checkpoint: Path
) -> tuple[list[float], list[FrozenControllerResidualParams], dict[str, Any]]:
    with source_checkpoint.open("rb") as stream:
        source = pickle.load(stream)
    with proposal_checkpoint.open("rb") as stream:
        proposal = pickle.load(stream)
    source_params = source.actor_params
    proposal_params = proposal.actor_params
    if not isinstance(source_params, FrozenControllerResidualParams) or not isinstance(
        proposal_params, FrozenControllerResidualParams
    ):
        raise ValueError("proposal checkpoints do not contain nested residual actors")
    candidates = [
        interpolate_actor_params(source_params, proposal_params, alpha=alpha)
        for alpha in CANDIDATE_ALPHAS
    ]
    alphas = [0.0, *CANDIDATE_ALPHAS]
    params = [source_params, *candidates]
    manifest = {
        "protocol": "g1-e002-ahac-proposal-parameter-path-v1",
        "valid": True,
        "evaluation_only": True,
        "resumable": False,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "proposal_checkpoint": str(proposal_checkpoint.resolve()),
        "proposal_checkpoint_sha256": sha256_file(proposal_checkpoint),
        "interpolation_scope": "newest-frozen-controller-residual-adapter-only",
        "source_parent_exact": True,
        "rows": [
            {
                "alpha": alpha,
                "actor_params_sha256": parameter_tree_sha256(candidate),
                "delta": _tree_delta(source_params, candidate),
            }
            for alpha, candidate in zip(alphas, params, strict=True)
        ],
    }
    return alphas, params, manifest


def run_proposal(args: argparse.Namespace) -> None:
    configure_jax()
    repository = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        raise ValueError("proposal output already contains preflight.json")
    preflight = validate_preflight(
        repository=repository,
        checkpoint=args.checkpoint.resolve(),
        reference=args.reference_path.resolve(),
        resume_audit=args.resume_audit.resolve(),
        ahac_fidelity_audit=args.ahac_fidelity_audit.resolve(),
        critic_calibration_audit=args.critic_calibration_audit.resolve(),
        code_commit=args.code_commit,
    )
    _write_json_atomically(preflight_path, preflight)
    kwargs = build_proposal_kwargs(
        "g1-4x5",
        args.reference_path.resolve(),
        args.seed,
        args.checkpoint.resolve(),
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(get_solver_profile("g1-4x5")):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_directory = (output_root / relative_save_dir).resolve()
    validation = validate_training_artifacts(
        run_directory,
        source_checkpoint=args.checkpoint.resolve(),
    )
    _write_json_atomically(
        output_root / "training_validation.json", validation
    )
    print(run_directory)


def run_evaluation(args: argparse.Namespace) -> None:
    configure_jax()
    _validate_runtime()
    output_root = args.output_root.resolve()
    training = _read_json(output_root / "training_validation.json")
    proposal_checkpoint = Path(training["proposal_checkpoint"])
    if (
        training.get("valid") is not True
        or not proposal_checkpoint.is_file()
        or sha256_file(proposal_checkpoint)
        != training.get("proposal_checkpoint_sha256")
    ):
        raise ValueError("proposal training validation is invalid")
    alphas, params, manifest = _load_source_and_candidates(
        args.checkpoint.resolve(), proposal_checkpoint
    )
    _write_json_atomically(output_root / "candidate_manifest.json", manifest)

    carried, carried_arrays = evaluate_carried_population(
        checkpoint=args.checkpoint.resolve(),
        reference=args.reference_path.resolve(),
        actor_params=params,
        alphas=alphas,
    )
    carried_npz = output_root / "carried_evaluation.npz"
    _atomic_npz(carried_npz, carried_arrays)
    carried["arrays_path"] = str(carried_npz.resolve())
    carried["arrays_sha256"] = sha256_file(carried_npz)
    _write_json_atomically(output_root / "carried_evaluation.json", carried)

    jax.clear_caches()
    gc.collect()
    phase_grid = evaluate_phase_grid(
        checkpoint=args.checkpoint.resolve(),
        reference=args.reference_path.resolve(),
        actor_params=params,
        alphas=alphas,
    )
    _write_json_atomically(output_root / "phase_grid.json", phase_grid)

    carried_by_alpha = {row["alpha"]: row for row in carried["rows"]}
    phase_by_alpha = {row["alpha"]: row for row in phase_grid["rows"]}
    source_carried = carried_by_alpha[0.0]
    source_phase = phase_by_alpha[0.0]["summary"]["survival"]
    rows = [
        {
            "alpha": alpha,
            "actor_params_sha256": carried_by_alpha[alpha][
                "actor_params_sha256"
            ],
            "phase_survival": phase_by_alpha[alpha]["summary"]["survival"],
            "phase_summary": phase_by_alpha[alpha]["summary"],
            "carried": {
                key: carried_by_alpha[alpha][key] for key in _CARRIED_KEYS
            },
            "initial_action_delta_rms": carried_by_alpha[alpha][
                "initial_action_delta_rms"
            ],
            "initial_action_delta_max_abs": carried_by_alpha[alpha][
                "initial_action_delta_max_abs"
            ],
        }
        for alpha in CANDIDATE_ALPHAS
    ]
    selection = classify_candidates(
        rows,
        source_phase_survival=source_phase,
        source_carried=source_carried,
    )
    selection.update(
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        proposal_checkpoint_sha256=training["proposal_checkpoint_sha256"],
        proposal_initial_horizon=training["telemetry"][
            "ahac_horizon_before_update"
        ],
        proposal_initial_constraint_penalty=training[
            "initial_actor_constraint_penalty"
        ],
        proposal_critic_heads_exact=training["critic_heads_exact"],
        interpretation_boundary=(
            "This is a one-proposal reject-by-default discriminator. It tests "
            "the current migrated first update; it neither validates later "
            "dual adaptation nor retains an AHAC policy."
        ),
    )
    selection_path = output_root / "selection.json"
    _write_json_atomically(selection_path, selection)
    artifacts = {
        name: sha256_file(output_root / name)
        for name in (
            "preflight.json",
            "training_validation.json",
            "candidate_manifest.json",
            "carried_evaluation.npz",
            "carried_evaluation.json",
            "phase_grid.json",
            "selection.json",
        )
    }
    completion = {
        "protocol": "g1-e002-ahac-first-proposal-completion-v1",
        "valid": True,
        "outcome": selection["outcome"],
        "optimizer_updates": 1,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": artifacts,
        "training_artifacts": {
            "hparams": {
                "path": training["hparams"],
                "sha256": training["hparams_sha256"],
            },
            "checkpoint_metrics": {
                "path": training["checkpoint_metrics"],
                "sha256": training["checkpoint_metrics_sha256"],
            },
            "migration": {
                "path": training["migration_artifact"],
                "sha256": training["migration_artifact_sha256"],
            },
            "proposal_checkpoint": {
                "path": training["proposal_checkpoint"],
                "sha256": training["proposal_checkpoint_sha256"],
                "retained": False,
            },
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


def _child_command(args: argparse.Namespace, stage: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.g1_ahac_first_proposal_audit.run",
        "--stage",
        stage,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--reference-path",
        str(args.reference_path.resolve()),
        "--resume-audit",
        str(args.resume_audit.resolve()),
        "--ahac-fidelity-audit",
        str(args.ahac_fidelity_audit.resolve()),
        "--critic-calibration-audit",
        str(args.critic_calibration_audit.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--code-commit",
        args.code_commit,
        "--seed",
        str(args.seed),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "proposal", "evaluate"), default="all"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-audit", type=Path, required=True)
    parser.add_argument("--ahac-fidelity-audit", type=Path, required=True)
    parser.add_argument("--critic-calibration-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("proposal audit seed must equal zero")
    if args.stage == "proposal":
        run_proposal(args)
    elif args.stage == "evaluate":
        run_evaluation(args)
    else:
        subprocess.run(_child_command(args, "proposal"), check=True)
        subprocess.run(_child_command(args, "evaluate"), check=True)


if __name__ == "__main__":
    main()
