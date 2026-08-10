"""Execute one real canonical G1 SHAC update and publish its evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jp
import numpy as np

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REFERENCE_PATH,
    G1TrackingRMR50HzSourceStepEnv,
)
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_canonical_g1_shac import build_canonical_kwargs
from tools.run_g1_tracking_shac import configure_jax


REQUIRED_FINITE_FIELDS = (
    "zero_head_reference_target_max_error",
    "reward",
    "actor_grad_finite_fraction",
    "critic_grad_finite_fraction",
    "actor_grad_raw_median",
    "actor_grad_raw_max",
    "critic_grad_raw_median",
    "critic_grad_raw_max",
    "optimizer_update_norm",
)


def build_smoke_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
) -> dict:
    """Bound execution cost while retaining the canonical scientific path."""
    kwargs = build_canonical_kwargs(profile_name, reference_path, seed)
    kwargs.update(
        total_steps=2,
        num_envs=2,
        unroll_length=1,
        checkpoint_interval=2,
    )
    return kwargs


def validate_smoke_receipt(receipt: dict) -> None:
    """Reject a receipt that does not prove a differentiated real update."""
    required = {
        "schema_version",
        "solver_profile",
        "num_envs",
        "unroll_length",
        "actor_updates",
        "critic_iterations",
        "distinct_model_count",
        "sampled_model_sha256",
        *REQUIRED_FINITE_FIELDS,
    }
    missing = sorted(required - receipt.keys())
    if missing:
        raise ValueError(f"smoke receipt is missing fields: {missing}")
    if receipt["num_envs"] != 2 or receipt["unroll_length"] != 1:
        raise ValueError("smoke must use exactly two one-step environments")
    if receipt["actor_updates"] != 1 or receipt["critic_iterations"] != 16:
        raise ValueError("smoke must execute one actor and 16 critic updates")
    if receipt["distinct_model_count"] != 2:
        raise ValueError("smoke did not exercise two distinct physical models")
    if len(set(receipt["sampled_model_sha256"])) != 2:
        raise ValueError("sampled model hashes are not distinct")
    for name in REQUIRED_FINITE_FIELDS:
        value = receipt[name]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"smoke field {name} must be finite")
    if receipt["zero_head_reference_target_max_error"] > 1e-12:
        raise ValueError("zero actor head does not target the reference pose")
    if receipt["actor_grad_finite_fraction"] != 1.0:
        raise ValueError("actor gradients were not finite for both rollouts")
    if receipt["critic_grad_finite_fraction"] != 1.0:
        raise ValueError("critic gradients were not finite for both rollouts")
    if receipt["optimizer_update_norm"] <= 0.0:
        raise ValueError("actor optimizer produced no parameter update")


def _sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_sha256(env, info: dict) -> str:
    model = env._get_randomized_model(info)
    digest = hashlib.sha256()
    for value in (
        model.geom_friction,
        model.body_mass,
        model.body_inertia,
        model.body_ipos,
    ):
        array = np.asarray(jax.device_get(value))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_sha(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_smoke(
    *,
    profile_name: str,
    reference_path: Path,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run and validate one bounded but genuine SHAC optimizer update."""
    configure_jax()
    profile = get_solver_profile(profile_name)
    kwargs = build_smoke_kwargs(profile_name, reference_path, seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    env = G1TrackingRMR50HzSourceStepEnv(
        xml_path=kwargs["xml_path"],
        reference_path=kwargs["reference_path"],
        reference_stride=kwargs["reference_stride"],
        actor_history_len=kwargs["actor_history_len"],
        actor_observation_noise=kwargs["actor_observation_noise"],
        reference_reset_noise_scale=kwargs["reference_reset_noise_scale"],
        reference_residual_control=kwargs["reference_residual_control"],
        reference_residual_scale=kwargs["reference_residual_scale"],
        domain_randomization=kwargs["domain_randomization"],
        friction_range=kwargs["friction_range"],
        mass_range=kwargs["mass_range"],
        kp_range=kwargs["kp_range"],
        kd_range=kwargs["kd_range"],
        com_offset_range=kwargs["com_offset_range"],
        solver_iterations=kwargs["solver_iterations"],
        solver_ls_iterations=kwargs["solver_ls_iterations"],
    )
    reset_keys = jax.random.split(jax.random.PRNGKey(seed + 10_000), 2)
    reset_states = [
        env.reset(key, jp.array(0.0)) for key in reset_keys
    ]
    model_hashes = [_model_sha256(env, state.info) for state in reset_states]
    phase_state = env.reset_at_phase(
        jax.random.PRNGKey(seed + 20_000),
        jp.array(0.0),
        jp.array(0),
    )
    zero_target = env.position_target(
        phase_state, jp.zeros(env.action_dim)
    )
    target_error = float(
        jp.max(jp.abs(zero_target - env.qpos_reference[0, 7:]))
    )

    previous_directory = Path.cwd()
    try:
        os.chdir(output_dir)
        with solver_context(profile):
            final_state, relative_run_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    run_dir = output_dir / relative_run_dir
    diagnostics = json.loads(
        (run_dir / "diag_log.json").read_text(encoding="utf-8")
    )
    if len(diagnostics) != 1:
        raise ValueError("smoke must publish exactly one diagnostic update")
    metric = diagnostics[0]
    repository = Path(__file__).resolve().parents[1]
    receipt = {
        "schema_version": 1,
        "solver_profile": profile_name,
        "solver_iterations": profile.iterations,
        "solver_ls_iterations": profile.ls_iterations,
        "fixed_scan": profile.fixed_scan,
        "num_envs": 2,
        "unroll_length": 1,
        "actor_updates": int(final_state.step) // 2,
        "critic_iterations": kwargs["critic_iterations"],
        "distinct_model_count": len(set(model_hashes)),
        "sampled_model_sha256": model_hashes,
        "sampled_randomization": [
            {
                name: np.asarray(jax.device_get(state.info[name])).tolist()
                for name in (
                    "friction_scale",
                    "mass_scale",
                    "kp_scale",
                    "kd_scale",
                    "com_offset",
                )
            }
            for state in reset_states
        ],
        "zero_head_reference_target_max_error": target_error,
        "reward": metric["reward"],
        "actor_grad_finite_fraction": metric[
            "actor_grad_finite_fraction"
        ],
        "critic_grad_finite_fraction": metric[
            "critic_grad_finite_fraction"
        ],
        "actor_grad_raw_median": metric["actor_grad_raw_median"],
        "actor_grad_raw_max": metric["actor_grad_raw_max"],
        "critic_grad_raw_median": metric["critic_grad_raw_median"],
        "critic_grad_raw_max": metric["critic_grad_raw_max"],
        "optimizer_update_norm": metric["actor_update_norm"],
        "repository_git_sha": _git_sha(repository),
        "model_sha256": _sha256_file(DEFAULT_MODEL_PATH),
        "reference_sha256": _sha256_file(reference_path),
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "mujoco", "numpy", "flax", "optax")
        },
        "training_artifact_dir": str(run_dir.resolve()),
        "effective_hyperparameters": kwargs,
    }
    validate_smoke_receipt(receipt)
    receipt_path = output_dir / "smoke_receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=tuple(sorted(SOLVER_PROFILES)),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipt = run_smoke(
        profile_name=args.solver_profile,
        reference_path=args.reference_path.resolve(),
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
