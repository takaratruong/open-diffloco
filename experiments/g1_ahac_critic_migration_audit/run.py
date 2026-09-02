"""Audit candidate double-critic migrations from the retained E002 state.

This diagnostic performs no simulation step and no optimization.  It evaluates
the retained online and delayed SHAC critics on the exact carried effective-512
state population, then compares an exact duplicated-online AHAC migration with
an online-plus-delayed-target migration.  It also records the pinned public
AHAC implementation's accidental parameter sharing between its named critics.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.ahac import duplicate_single_critic_params
from src.algorithms.shac.counterfactual_wrench_distillation import (
    parameter_tree_sha256,
)
from src.core.data_structures import Normalizer
from src.core.networks import Critic, DoubleCritic
from tools.evaluate_g1_flax_phase_grid import (
    load_checkpoint_environment_contract,
)
from tools.evaluate_g1_tracking import make_evaluation_env
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
UPSTREAM_CRITIC_SOURCE_SHA256 = (
    "0fd50db670fa57eb2bcad3cd7d59ac2bc39b1b5ec239c71074177f10d61aa346"
)
UPSTREAM_AHAC_CONFIG_SHA256 = (
    "5d07ea0b3f8cfaae11064e09f318c404c36c88d1fa4ac3bb389fad1d6f293483"
)
AHAC_PAPER_SHA256 = (
    "8f5624c8a3ec43e05d1c01159c82a6d72563064d671035036032b0a4ed92dae7"
)
DETERMINISTIC_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops"
MIN_DISTINCT_FRACTION = 0.95
MIN_PEARSON_CORRELATION = 0.95
MAX_LAG_NRMSE = 0.25
MAX_CONSERVATIVE_SHIFT_NRMSE = 0.25


def _single_critic_tree(params: object) -> Mapping[str, object]:
    if not isinstance(params, Mapping) or set(params) != {"params"}:
        raise ValueError("migration requires a single critic parameter tree")
    single = params["params"]
    if (
        not isinstance(single, Mapping)
        or not single
        or "critic_0" in single
        or "critic_1" in single
    ):
        raise ValueError("migration requires a single critic parameter tree")
    return single


def combine_single_critic_params(first: object, second: object) -> dict[str, object]:
    """Place two trained single critics into ordered independent AHAC heads."""

    first_tree = _single_critic_tree(first)
    second_tree = _single_critic_tree(second)
    return {
        "params": {
            "critic_0": jax.tree.map(lambda leaf: leaf, first_tree),
            "critic_1": jax.tree.map(lambda leaf: leaf, second_tree),
        }
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("critic values must be finite and nonempty")
    quantiles = np.percentile(array, (0, 10, 25, 50, 75, 90, 100))
    return {
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def summarize_migration_candidates(
    online_values: np.ndarray,
    delayed_values: np.ndarray,
    duplicate_values: np.ndarray,
    diverse_values: np.ndarray,
) -> dict[str, object]:
    """Classify whether the trained delayed critic is a safe second head."""

    online = np.asarray(online_values, dtype=np.float64).reshape(-1)
    delayed = np.asarray(delayed_values, dtype=np.float64).reshape(-1)
    duplicate = np.asarray(duplicate_values, dtype=np.float64)
    diverse = np.asarray(diverse_values, dtype=np.float64)
    if (
        online.size < 2
        or delayed.shape != online.shape
        or duplicate.shape != (online.size, 2)
        or diverse.shape != (online.size, 2)
        or not all(
            np.isfinite(value).all()
            for value in (online, delayed, duplicate, diverse)
        )
    ):
        raise ValueError("critic migration predictions are incompatible")

    duplicate_online_error = float(
        np.max(np.abs(duplicate - online[:, None]))
    )
    diverse_online_error = float(np.max(np.abs(diverse[:, 0] - online)))
    diverse_delayed_error = float(
        np.max(np.abs(diverse[:, 1] - delayed))
    )
    if max(
        duplicate_online_error,
        diverse_online_error,
        diverse_delayed_error,
    ) != 0.0:
        raise ValueError("double-critic heads do not preserve source predictions")

    online_std = float(np.std(online))
    delayed_std = float(np.std(delayed))
    if online_std <= 1e-8 or delayed_std <= 1e-8:
        raise ValueError("trained critic predictions are degenerate")
    lag = delayed - online
    lag_rms = float(np.sqrt(np.mean(np.square(lag))))
    lag_nrmse = lag_rms / online_std
    correlation = float(np.corrcoef(online, delayed)[0, 1])
    if not math.isfinite(correlation):
        raise ValueError("critic correlation is nonfinite")

    duplicate_disagreement = np.abs(duplicate[:, 0] - duplicate[:, 1])
    diverse_disagreement = np.abs(diverse[:, 0] - diverse[:, 1])
    conservative = np.min(diverse, axis=1)
    conservative_shift = conservative - online
    conservative_shift_nrmse = float(
        np.sqrt(np.mean(np.square(conservative_shift))) / online_std
    )
    distinct_fraction = float(np.mean(lag != 0.0))
    safe = bool(
        float(np.max(duplicate_disagreement)) == 0.0
        and distinct_fraction >= MIN_DISTINCT_FRACTION
        and correlation >= MIN_PEARSON_CORRELATION
        and lag_nrmse <= MAX_LAG_NRMSE
        and conservative_shift_nrmse <= MAX_CONSERVATIVE_SHIFT_NRMSE
    )
    return {
        "protocol": "g1-ahac-critic-migration-candidate-audit-v1",
        "valid": True,
        "classification": (
            "reuse-legacy-target-as-second-online-head"
            if safe
            else "legacy-target-lag-too-large"
        ),
        "optimizer_updates": 0,
        "policy_retained": False,
        "retained_policy": None,
        "safe_to_reuse_legacy_target": safe,
        "population_size": int(online.size),
        "selection_thresholds": {
            "minimum_distinct_fraction": MIN_DISTINCT_FRACTION,
            "minimum_pearson_correlation": MIN_PEARSON_CORRELATION,
            "maximum_lag_nrmse": MAX_LAG_NRMSE,
            "maximum_conservative_shift_nrmse": (
                MAX_CONSERVATIVE_SHIFT_NRMSE
            ),
        },
        "online": _distribution(online),
        "legacy_delayed_target": _distribution(delayed),
        "online_target_relationship": {
            "distinct_fraction": distinct_fraction,
            "pearson_correlation": correlation,
            "lag": _distribution(lag),
            "lag_nrmse_by_online_std": lag_nrmse,
        },
        "duplicate_online": {
            "head_disagreement_max": float(np.max(duplicate_disagreement)),
            "source_prediction_max_abs_error": duplicate_online_error,
            "functional_double_critic": False,
        },
        "online_plus_legacy_target": {
            "head_disagreement": _distribution(diverse_disagreement),
            "head_disagreement_max": float(np.max(diverse_disagreement)),
            "pearson_correlation": correlation,
            "online_head_max_abs_error": diverse_online_error,
            "delayed_head_max_abs_error": diverse_delayed_error,
            "target_selected_by_min_fraction": float(
                np.mean(delayed < online)
            ),
            "conservative_value": _distribution(conservative),
            "conservative_shift_from_online": _distribution(
                conservative_shift
            ),
            "conservative_shift_nrmse_by_online_std": (
                conservative_shift_nrmse
            ),
            "functional_double_critic": True,
        },
        "interpretation_boundary": (
            "This verifies a continuity-preserving resume migration on the "
            "exact carried state population; it does not establish critic "
            "calibration or behavioral improvement under AHAC training."
        ),
    }


def inspect_upstream_double_critic(source_text: str) -> dict[str, object]:
    """Detect the public implementation's reuse of one module list."""

    try:
        module = ast.parse(source_text)
    except SyntaxError as error:
        raise ValueError("upstream critic source is invalid Python") from error
    critic_class = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DoubleCriticMLP"
        ),
        None,
    )
    if critic_class is None:
        raise ValueError("upstream DoubleCriticMLP is missing")
    constructor = next(
        (
            node
            for node in critic_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        raise ValueError("upstream DoubleCriticMLP constructor is missing")

    branch_spreads: dict[str, str] = {}
    for node in ast.walk(constructor):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            not isinstance(target, ast.Attribute)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "self"
            or target.attr not in {"critic_1", "critic_2"}
            or not isinstance(node.value, ast.Call)
        ):
            continue
        spread = next(
            (
                argument.value.id
                for argument in node.value.args
                if isinstance(argument, ast.Starred)
                and isinstance(argument.value, ast.Name)
            ),
            None,
        )
        if spread is not None:
            branch_spreads[target.attr] = spread
    declares_two = set(branch_spreads) == {"critic_1", "critic_2"}
    shared = bool(
        declares_two
        and branch_spreads["critic_1"] == branch_spreads["critic_2"]
    )
    return {
        "protocol": "pinned-public-ahac-double-critic-source-audit-v1",
        "valid": True,
        "declares_two_critic_branches": declares_two,
        "branch_module_spread_names": branch_spreads,
        "branches_reuse_same_module_container": shared,
        "effective_parameter_independence": not shared,
        "paper_public_code_double_critic_divergence": shared,
        "interpretation": (
            "Both nn.Sequential containers receive the same instantiated "
            "module objects, so PyTorch registers aliases to one parameter set "
            "and both branches return identical values."
            if shared
            else "The two branches do not structurally reuse one module list."
        ),
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _validate_runtime() -> dict[str, object]:
    if DETERMINISTIC_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise ValueError("critic audit requires deterministic XLA reductions")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("critic audit requires exactly one visible GPU")
    if not bool(jax.config.jax_enable_x64):
        raise ValueError("critic audit requires JAX float64 mode")
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        "jax_enable_x64": True,
        "xla_deterministic_reductions": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _make_environment(hparams: Mapping[str, object], reference: Path):
    return make_evaluation_env(
        str(hparams["env_variant"]),
        solver_iterations=int(hparams["solver_iterations"]),
        solver_ls_iterations=int(hparams["solver_ls_iterations"]),
        reference_path=reference,
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
        contact_stiffness_metric="all_body_spatial",
    )


def evaluate_migration_candidates(
    *, checkpoint: Path, reference: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    hparams_path = checkpoint.with_name("hparams.json")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    contract = load_checkpoint_environment_contract(checkpoint)
    if (
        int(hparams["effective_num_envs"]) != EFFECTIVE_NUM_ENVS
        or int(hparams["unroll_length"]) != 24
        or hparams.get("ahac") is not False
        or float(hparams.get("actor_bootstrap_scale", -1.0)) != 0.0
        or contract["env_variant"] != hparams["env_variant"]
    ):
        raise ValueError("retained E002 boundary does not match the audit")
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    if (
        int(state.step) != START_STEP
        or np.asarray(state.env_state.obs).shape[0] != EFFECTIVE_NUM_ENVS
    ):
        raise ValueError("retained E002 checkpoint state is invalid")

    environment = _make_environment(hparams, reference)
    raw_observations = jax.vmap(environment._get_critic_obs)(
        state.env_state.data, state.env_state.info
    )
    if raw_observations.shape != (
        EFFECTIVE_NUM_ENVS,
        environment.critic_obs_dim,
    ):
        raise ValueError("critic observation population has the wrong shape")
    normalizer = Normalizer(environment.critic_obs_dim)
    observations = normalizer.normalize(
        state.critic_normalizer, raw_observations
    ).astype(jnp.float32)
    single = Critic()
    double = DoubleCritic()
    duplicate_params = duplicate_single_critic_params(state.critic_params)
    diverse_params = combine_single_critic_params(
        state.critic_params, state.target_critic_params
    )

    @jax.jit
    def predict_single(params, inputs):
        return single.apply(params, inputs).squeeze(-1)

    @jax.jit
    def predict_double(params, inputs):
        return double.apply(params, inputs)

    online_values = predict_single(state.critic_params, observations)
    delayed_values = predict_single(state.target_critic_params, observations)
    duplicate_values = predict_double(duplicate_params, observations)
    diverse_values = predict_double(diverse_params, observations)
    arrays = {
        "critic_observations": np.asarray(observations),
        "phase": np.asarray(state.env_state.info["phase"]),
        "online_values": np.asarray(online_values),
        "legacy_delayed_target_values": np.asarray(delayed_values),
        "duplicate_online_values": np.asarray(duplicate_values),
        "online_plus_legacy_target_values": np.asarray(diverse_values),
    }
    summary = summarize_migration_candidates(
        arrays["online_values"],
        arrays["legacy_delayed_target_values"],
        arrays["duplicate_online_values"],
        arrays["online_plus_legacy_target_values"],
    )
    online_hash = parameter_tree_sha256(state.critic_params["params"])
    target_hash = parameter_tree_sha256(state.target_critic_params["params"])
    head_hashes = [
        parameter_tree_sha256(diverse_params["params"][name])
        for name in ("critic_0", "critic_1")
    ]
    if head_hashes != [online_hash, target_hash] or online_hash == target_hash:
        raise ValueError("candidate head identity does not close")
    phases = arrays["phase"]
    summary.update(
        source_step=START_STEP,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        source_hparams_sha256=SOURCE_HPARAMS_SHA256,
        actor_bootstrap_scale_in_source=float(hparams["actor_bootstrap_scale"]),
        source_online_critic_sha256=online_hash,
        source_legacy_delayed_target_sha256=target_hash,
        candidate_head_sha256=head_hashes,
        critic_optimizer_would_reset=True,
        delayed_target_role_would_be_removed=True,
        legacy_target_weights_would_be_reused_as_online_head=True,
        actor_and_actor_optimizer_would_be_preserved=True,
        migration_executed=False,
        phase_population={
            "minimum": int(np.min(phases)),
            "maximum": int(np.max(phases)),
            "unique_count": int(np.unique(phases).size),
        },
    )
    return arrays, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--upstream-critic-source", type=Path, required=True)
    parser.add_argument("--upstream-ahac-config", type=Path, required=True)
    parser.add_argument("--ahac-paper", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("critic migration audit seed must equal zero")
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
    evidence_paths = {
        "upstream_critic_source": (
            args.upstream_critic_source.resolve(),
            UPSTREAM_CRITIC_SOURCE_SHA256,
        ),
        "upstream_ahac_config": (
            args.upstream_ahac_config.resolve(),
            UPSTREAM_AHAC_CONFIG_SHA256,
        ),
        "ahac_paper": (args.ahac_paper.resolve(), AHAC_PAPER_SHA256),
    }
    for path, digest in evidence_paths.values():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"external AHAC evidence hash mismatch: {path}")
    upstream_audit = inspect_upstream_double_critic(
        evidence_paths["upstream_critic_source"][0].read_text(encoding="utf-8")
    )
    if not upstream_audit["paper_public_code_double_critic_divergence"]:
        raise ValueError("pinned public double-critic discrepancy is absent")
    preflight.update(
        protocol="g1-ahac-critic-migration-preflight-v1",
        runtime=runtime,
        optimizer_updates=0,
        external_evidence={
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in evidence_paths.items()
        },
        upstream_double_critic=upstream_audit,
    )
    preflight_path = output_root / "preflight.json"
    _write_json_atomically(preflight_path, preflight)

    arrays, result = evaluate_migration_candidates(
        checkpoint=checkpoint, reference=reference
    )
    result["upstream_double_critic"] = upstream_audit
    values_path = output_root / "critic_migration_values.npz"
    result_path = output_root / "critic_migration_audit.json"
    _atomic_npz(values_path, arrays)
    result["values_sha256"] = sha256_file(values_path)
    _write_json_atomically(result_path, result)
    completion = {
        "protocol": "g1-ahac-critic-migration-completion-v1",
        "valid": True,
        "classification": result["classification"],
        "optimizer_updates": 0,
        "policy_retained": False,
        "retained_policy": None,
        "artifacts": {
            "preflight.json": sha256_file(preflight_path),
            "critic_migration_values.npz": sha256_file(values_path),
            "critic_migration_audit.json": sha256_file(result_path),
        },
    }
    _write_json_atomically(output_root / "completion.json", completion)
    print(output_root)


if __name__ == "__main__":
    main()
