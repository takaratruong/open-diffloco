"""Run the bounded carried phase-105 G1 LAFAN action shooting gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco.mjx._src import solver as _mjx_solver

from src.core.data_structures import Normalizer
from src.envs.g1_tracking.action_shooting import (
    ShootingConfig,
    canonical_forward_gradient,
    capture_actor_window_without_reset,
    directional_fd_audit,
    rollout_actions_without_reset,
    run_projected_armijo,
    shooting_objective,
    support_trace_from_states,
)
from src.envs.g1_tracking.fixed_solver import (
    CONVERGENCE_SCAN,
    FIXED_SOLVER_AND_LINESEARCH_SCAN,
    _solve_with_fixed_outer_loop,
    active_solver_gradient_semantic,
    fixed_mjx_solver_outer_loop,
)
from tools.evaluate_g1_tracking import (
    _load_policy,
    configure_jax,
    make_evaluation_env,
)
from tools.prepare_g1_rmr_reference import sha256_file

REFERENCE_PATH = Path(
    "/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/"
    "artifacts/E-20260808-000/reference/"
    "dance1_subject2_f122_422_50hz.npz"
)
REFERENCE_SHA256 = "bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db"
CHECKPOINT_PATH = Path(
    "/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/"
    "artifacts/E-20260808-012/carried_run_root/training_runs/"
    "shac_20260807_234306/checkpoint_step_688128.pkl"
)
CHECKPOINT_SHA256 = "12198b38443c2705da5e26a58ddd320f4d5837880b32f7404db428b7220164d4"
CONFIG_PATH = CHECKPOINT_PATH.with_name("hparams.json")
CONFIG_SHA256 = "a03053410c21c54d4175c7634eaf77f7886b4ec9a23e515952d0ad5d7380c3cf"
MODEL_SHA256 = "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
CONTROLLER_SHA256 = "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HASHED_ARTIFACTS = (
    "preflight.json",
    "initial_rollout.npz",
    "gradient_gate.json",
    "candidate_rollout.npz",
    "optimization_trace.json",
)
SUCCESS_ARTIFACTS = frozenset((*HASHED_ARTIFACTS, "summary.json"))


@dataclass(frozen=True)
class PhysicalEvaluation:
    objective: float
    feasible: bool
    summary: dict
    arrays: dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--reference-sha256", default=REFERENCE_SHA256)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--start-phase", type=int, default=105)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--solver-ls-iterations", type=int, default=5)
    parser.add_argument(
        "--solver-gradient-semantic",
        choices=(CONVERGENCE_SCAN, FIXED_SOLVER_AND_LINESEARCH_SCAN),
        default=CONVERGENCE_SCAN,
    )
    parser.add_argument("--gradient-repeat-count", type=int, default=2)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-3)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--trust-radius", type=float, default=0.02)
    parser.add_argument(
        "--line-search-alphas",
        type=float,
        nargs=4,
        default=[1.0, 0.5, 0.25, 0.125],
    )
    parser.add_argument("--minimum-mean-reward-gain", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_registered_args(args: argparse.Namespace) -> None:
    expected = {
        "reference_sha256": REFERENCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "start_phase": 105,
        "horizon": 12,
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
        "gradient_repeat_count": 2,
        "finite_difference_epsilon": 1e-3,
        "action_deviation_weight": 1e-3,
        "max_iterations": 3,
        "trust_radius": 0.02,
        "line_search_alphas": [1.0, 0.5, 0.25, 0.125],
        "minimum_mean_reward_gain": 0.001,
        "seed": 0,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"{name.replace('_', ' ')} must remain fixed at {value}")
    if len(args.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.code_commit
    ):
        raise ValueError("code commit must be 40 lowercase hex characters")


def validate_training_hparams(hparams: dict) -> None:
    expected = {
        "env_variant": "g1_tracking_rmr_50hz_validated",
        "reference_sha256": REFERENCE_SHA256,
        "reference_fps": 50.0,
        "reference_stride": 1,
        "reference_states": 500,
        "reference_transitions": 499,
        "termination_margin_weight": 0.0,
        "actor_history_len": 1,
        "residual_action_scale": 0.0,
    }
    for name, value in expected.items():
        if hparams.get(name) != value:
            raise ValueError(f"hparams {name} must equal {value}")


def verify_sha256(path: Path, *, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {digest}")
    return digest


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp.npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_output_directory(path: Path) -> None:
    """Create a pristine evidence root without following a symlink."""
    if path.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {path}")
    if path.exists():
        raise ValueError(f"output directory must not be pre-existing: {path}")
    path.mkdir(parents=True, exist_ok=False)


def validate_success_artifacts(path: Path) -> None:
    """Require the closed six-file success evidence set."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("success evidence root must be a regular directory")
    entries = tuple(path.iterdir())
    names = {entry.name for entry in entries}
    if names != SUCCESS_ARTIFACTS:
        raise ValueError(
            "success evidence must contain exactly "
            f"{sorted(SUCCESS_ARTIFACTS)}, got {sorted(names)}"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("all success artifacts must be regular files")


def required_support_transition_present(
    state_phases: np.ndarray,
    support: np.ndarray,
    *,
    required_phase: int = 106,
) -> bool:
    """Require the complete observed right-only to bilateral chronology."""
    phases = np.asarray(state_phases)
    support_array = np.asarray(support, dtype=bool)
    if phases.ndim != 1 or support_array.shape != (phases.size, 2):
        raise ValueError("support chronology must align with state phases")
    expected_phases = np.arange(
        required_phase - 1,
        required_phase + 12,
        dtype=phases.dtype,
    )
    expected_support = np.ones((13, 2), dtype=bool)
    expected_support[0] = [False, True]
    return bool(
        np.array_equal(phases, expected_phases)
        and np.array_equal(support_array, expected_support)
    )


def classify_gate(
    gradient: dict,
    initial: dict,
    candidate: dict,
    *,
    accepted_steps: int,
    minimum_mean_reward_gain: float = 0.001,
) -> str:
    if not gradient.get("passed", False):
        return "action-gradient-identity-blocked"
    if (
        initial.get("terminal_count", 1) != 0
        or initial.get("support_switch_count", 0) < 1
        or not initial.get("required_support_switch_present", False)
    ):
        return "contact-window-invalid"
    reward_gain = candidate.get("mean_reward", -np.inf) - initial.get(
        "mean_reward", np.inf
    )
    if (
        accepted_steps >= 1
        and candidate.get("terminal_count", 1) == 0
        and candidate.get("support_switch_count", 0) >= 1
        and candidate.get("required_support_switch_present", False)
        and reward_gain >= minimum_mean_reward_gain
    ):
        return "contact-shooting-authorized"
    return "finite-contact-no-material-step"


def execute_gate(args: argparse.Namespace, *, runtime) -> dict:
    """Execute the fixed preflight and bounded optimizer through a runtime."""
    validate_registered_args(args)
    output_dir = args.output_dir
    prepare_output_directory(output_dir)

    preflight = runtime.preflight()
    if not preflight.get("passed", False):
        raise ValueError("runtime preflight must pass before physical evaluation")
    atomic_write_json(output_dir / "preflight.json", preflight)

    nominal_actions = np.asarray(runtime.nominal_actions(), dtype=np.float64)
    initial = runtime.evaluate(nominal_actions)
    atomic_savez(output_dir / "initial_rollout.npz", **initial.arrays)

    gradient_gate = runtime.gradient_preflight(nominal_actions)
    atomic_write_json(output_dir / "gradient_gate.json", gradient_gate)
    config = ShootingConfig(
        start_phase=args.start_phase,
        horizon=args.horizon,
        iterations=args.max_iterations,
        trust_radius=args.trust_radius,
        line_search_alphas=tuple(args.line_search_alphas),
    )
    if gradient_gate.get("passed", False) and initial.feasible:

        def objective_and_gate(actions):
            evaluation = runtime.evaluate(actions)
            return evaluation.objective, evaluation.feasible

        selected_actions, trace = run_projected_armijo(
            nominal_actions,
            objective_and_gate=objective_and_gate,
            gradient_fn=runtime.gradient,
            config=config,
        )
    else:
        selected_actions = np.array(nominal_actions, copy=True)
        trace = ()
    candidate = runtime.evaluate(selected_actions)
    atomic_savez(output_dir / "candidate_rollout.npz", **candidate.arrays)

    accepted_steps = sum(row.accepted for row in trace)
    trace_document = {
        "iterations": [asdict(row) for row in trace],
        "accepted_steps": accepted_steps,
        "trust_radius": args.trust_radius,
        "line_search_alphas": args.line_search_alphas,
    }
    atomic_write_json(output_dir / "optimization_trace.json", trace_document)
    classification = classify_gate(
        gradient_gate,
        initial.summary,
        candidate.summary,
        accepted_steps=accepted_steps,
        minimum_mean_reward_gain=args.minimum_mean_reward_gain,
    )
    artifact_sha256 = {
        name: sha256_file(output_dir / name) for name in HASHED_ARTIFACTS
    }
    summary = {
        "protocol": "g1-lafan-carried-action-shooting-gate-v1",
        "classification": classification,
        "initial": initial.summary,
        "candidate": candidate.summary,
        "mean_reward_gain": (
            candidate.summary["mean_reward"] - initial.summary["mean_reward"]
        ),
        "accepted_steps": accepted_steps,
        "gradient_gate_passed": bool(gradient_gate.get("passed", False)),
        "artifact_sha256": artifact_sha256,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    validate_success_artifacts(output_dir)
    return summary


class G1PhysicalRuntime:
    """Pinned physical implementation behind the bounded runner contract."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env = None
        self.phase_zero_state = None
        self.initial_data = None
        self.initial_previous_action = None
        self._nominal_actions = None
        self._rollout_fn = None
        self._values_fn = None
        self._directional_jvp = None
        self._gradient_cache: dict[str, np.ndarray] = {}
        self._preflight_document = None

    @staticmethod
    def _action_key(actions: np.ndarray) -> str:
        array = np.ascontiguousarray(actions, dtype=np.float64)
        return hashlib.sha256(array.tobytes()).hexdigest()

    def _verify_repository(self) -> str:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != self.args.code_commit:
            raise ValueError(
                f"code commit mismatch: expected {self.args.code_commit}, got {head}"
            )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise ValueError("scientific runner requires a clean worktree")
        return head

    def preflight(self) -> dict:
        """Verify immutable inputs and construct the exact validated task."""
        if self._preflight_document is not None:
            return dict(self._preflight_document)
        validate_registered_args(self.args)
        code_commit = self._verify_repository()
        reference_sha = verify_sha256(
            self.args.reference_path,
            expected=self.args.reference_sha256,
            label="reference",
        )
        checkpoint_sha = verify_sha256(
            self.args.checkpoint_path,
            expected=self.args.checkpoint_sha256,
            label="checkpoint",
        )
        config_sha = verify_sha256(
            self.args.config_path,
            expected=self.args.config_sha256,
            label="checkpoint hparams",
        )
        hparams = json.loads(self.args.config_path.read_text())
        validate_training_hparams(hparams)
        if not bool(jax.config.jax_enable_x64):
            raise ValueError("JAX float64 must be enabled")
        if _mjx_solver.solve is not _solve_with_fixed_outer_loop:
            raise ValueError("fixed MJX solver outer loop is not active")
        if active_solver_gradient_semantic() != self.args.solver_gradient_semantic:
            raise ValueError(
                "active solver gradient semantic does not match the request"
            )

        env = make_evaluation_env(
            "g1_tracking_rmr_50hz_validated",
            solver_iterations=self.args.solver_iterations,
            solver_ls_iterations=self.args.solver_ls_iterations,
            reference_path=self.args.reference_path,
            reference_stride=1,
        )
        model_sha = verify_sha256(
            Path(env.xml_path), expected=MODEL_SHA256, label="G1 dynamics model"
        )
        controller_sha = verify_sha256(
            Path(env.controller_path),
            expected=CONTROLLER_SHA256,
            label="G1 controller",
        )
        physical_contract = {
            "action_dim": env.action_dim,
            "actor_history_len": env.actor_history_len,
            "clip_actions": env.clip_actions,
            "control_dt": env.dt,
            "model_nq": env.mj_model.nq,
            "model_nv": env.mj_model.nv,
            "physics_substeps": env.n_frames,
            "physics_timestep": float(env.mj_model.opt.timestep),
            "reference_fps": env.reference.fps,
            "reference_states": env.reference_length,
            "reference_stride": env.reference_stride,
            "reward_scale": env.reward_scale,
            "solver_iterations": int(env.mj_model.opt.iterations),
            "solver_ls_iterations": int(env.mj_model.opt.ls_iterations),
            "squash_actor_actions": env.squash_actor_actions,
            "termination_margin_weight": env.termination_margin_weight,
        }
        expected_contract = {
            "action_dim": 29,
            "actor_history_len": 1,
            "clip_actions": False,
            "control_dt": 0.02,
            "model_nq": 36,
            "model_nv": 35,
            "physics_substeps": 4,
            "physics_timestep": 0.005,
            "reference_fps": 50.0,
            "reference_states": 500,
            "reference_stride": 1,
            "reward_scale": 0.02,
            "solver_iterations": 4,
            "solver_ls_iterations": 5,
            "squash_actor_actions": False,
            "termination_margin_weight": 0.0,
        }
        if physical_contract != expected_contract:
            raise ValueError(
                "validated physical contract mismatch: "
                f"expected {expected_contract}, got {physical_contract}"
            )
        self.env = env
        self._preflight_document = {
            "passed": True,
            "protocol": "g1-lafan-carried-action-shooting-preflight-v1",
            "code_commit": code_commit,
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": True,
            "fixed_solver_outer_loop": True,
            "solver_gradient_semantic": self.args.solver_gradient_semantic,
            "reference_path": str(self.args.reference_path.resolve()),
            "reference_sha256": reference_sha,
            "checkpoint_path": str(self.args.checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "config_path": str(self.args.config_path.resolve()),
            "config_sha256": config_sha,
            "model_sha256": model_sha,
            "controller_sha256": controller_sha,
            "physical_contract": physical_contract,
            "start_phase": self.args.start_phase,
            "end_phase": self.args.start_phase + self.args.horizon,
            "required_support_switch_phase": self.args.start_phase + 1,
            "initial_state_origin_phase": 0,
            "initial_state_prefix_transitions": self.args.start_phase,
            "initial_state_source": "full-carried-mjx-data",
            "horizon": self.args.horizon,
            "decision_shape": [self.args.horizon, env.action_dim],
            "action_deviation_weight": self.args.action_deviation_weight,
        }
        return dict(self._preflight_document)

    def _ensure_nominal(self) -> None:
        if self._nominal_actions is not None:
            return
        if self.env is None:
            raise ValueError("runtime preflight must run before actor capture")
        actor, actor_params, normalizer_state = _load_policy(
            self.env,
            self.args.checkpoint_path,
            self.args.seed,
        )
        normalizer = Normalizer(self.env.actor_frame_obs_dim)

        def policy(obs):
            normalized = self.env.normalize_actor_obs(
                normalizer, normalizer_state, obs
            ).astype(jnp.float32)
            return actor.apply(actor_params, normalized).astype(jnp.float64)

        self.phase_zero_state = self.env.reset_at_phase(
            jax.random.PRNGKey(self.args.seed),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0, dtype=jnp.int32),
        )
        if int(np.asarray(self.phase_zero_state.info["phase"])) != 0:
            raise ValueError("actor initializer must originate at phase zero")

        actor_window = jax.jit(
            lambda: capture_actor_window_without_reset(
                self.env,
                self.phase_zero_state,
                policy,
                start_phase=self.args.start_phase,
                horizon=self.args.horizon,
            )
        )()
        if (
            np.any(np.asarray(actor_window.prefix_done) > 0.5)
            or np.any(np.asarray(actor_window.prefix_terminal) > 0.5)
            or np.any(np.asarray(actor_window.done) > 0.5)
            or np.any(np.asarray(actor_window.terminal) > 0.5)
        ):
            raise ValueError("phase-zero actor carry terminated before phase 117")
        actions = np.asarray(actor_window.actions, dtype=np.float64)
        phases = np.asarray(actor_window.phases, dtype=np.int32)
        if actions.shape != (self.args.horizon, self.env.action_dim) or not (
            np.isfinite(actions).all()
        ):
            raise ValueError("actor must produce a finite (12, 29) tape")
        expected_phases = np.arange(
            self.args.start_phase + 1,
            self.args.start_phase + self.args.horizon + 1,
            dtype=np.int32,
        )
        if not np.array_equal(phases, expected_phases):
            raise ValueError("actor tape does not span exact phases 105 through 117")
        self.initial_data = actor_window.initial_data
        self.initial_previous_action = actor_window.initial_previous_action
        self._nominal_actions = np.asarray(actions, dtype=np.float64)
        nominal_jax = jnp.asarray(self._nominal_actions, dtype=jnp.float64)

        def rollout_fn(actions_value):
            return rollout_actions_without_reset(
                self.env,
                self.initial_data,
                start_phase=self.args.start_phase,
                initial_previous_action=self.initial_previous_action,
                actions=actions_value,
            )

        def values_fn(actions_value):
            rollout = rollout_fn(actions_value)
            objective = shooting_objective(
                rollout,
                actions_value,
                nominal_jax,
                action_deviation_weight=self.args.action_deviation_weight,
            )
            physical = jnp.concatenate((rollout.qpos, rollout.qvel), axis=1)
            return objective, physical

        self._rollout_fn = jax.jit(rollout_fn)
        self._values_fn = jax.jit(values_fn)
        self._directional_jvp = jax.jit(
            lambda actions_value, direction: jax.jvp(
                values_fn,
                (actions_value,),
                (direction,),
            )
        )

    def nominal_actions(self) -> np.ndarray:
        self._ensure_nominal()
        return np.array(self._nominal_actions, copy=True)

    def _objective(self, actions: jax.Array) -> jax.Array:
        return self._values_fn(actions)[0]

    def _physical(self, actions: jax.Array) -> jax.Array:
        return self._values_fn(actions)[1]

    def evaluate(self, actions) -> PhysicalEvaluation:
        """Evaluate one action tape through the uninterrupted carried plant."""
        self._ensure_nominal()
        action_array = np.asarray(actions, dtype=np.float64)
        expected_shape = (self.args.horizon, self.env.action_dim)
        if action_array.shape != expected_shape or not np.isfinite(action_array).all():
            raise ValueError(
                f"physical evaluation actions must be finite {expected_shape}"
            )
        rollout = self._rollout_fn(jnp.asarray(action_array))
        transition_phases = np.asarray(rollout.phases, dtype=np.int32)
        state_phases = np.concatenate(
            (
                np.asarray([self.args.start_phase], dtype=np.int32),
                transition_phases,
            )
        )
        qpos = np.concatenate(
            (
                np.asarray(self.initial_data.qpos)[None, :],
                np.asarray(rollout.qpos),
            ),
            axis=0,
        )
        qvel = np.concatenate(
            (
                np.asarray(self.initial_data.qvel)[None, :],
                np.asarray(rollout.qvel),
            ),
            axis=0,
        )
        support = support_trace_from_states(self.env.mj_model, qpos, qvel)
        switch_mask = np.any(support.support[1:] != support.support[:-1], axis=1)
        support_switch_phases = state_phases[1:][switch_mask].tolist()
        required_support_switch_phase = self.args.start_phase + 1
        required_support_switch_present = required_support_transition_present(
            state_phases,
            support.support,
            required_phase=required_support_switch_phase,
        )
        arrays = {
            "state_phases": state_phases,
            "transition_phases": transition_phases,
            "qpos": qpos,
            "qvel": qvel,
            "actions": action_array,
            "prepared_actions": np.asarray(rollout.prepared_actions),
            "raw_torques": np.asarray(rollout.raw_torques),
            "rewards": np.asarray(rollout.rewards),
            "terminal": np.asarray(rollout.terminal),
            "anchor_z_error": np.asarray(rollout.anchor_z_error),
            "anchor_xy_error": np.asarray(rollout.anchor_xy_error),
            "gravity_z_error": np.asarray(rollout.gravity_z_error),
            "distal_z_error": np.asarray(rollout.distal_z_error),
            "support": support.support,
        }
        all_finite = all(
            np.isfinite(value).all()
            for value in arrays.values()
            if value.dtype != np.bool_
        )
        expected_state_phases = np.arange(
            self.args.start_phase,
            self.args.start_phase + self.args.horizon + 1,
            dtype=np.int32,
        )
        chronology_valid = np.array_equal(state_phases, expected_state_phases)
        terminal_count = int(np.count_nonzero(arrays["terminal"] > 0.5))
        deviation = action_array - self._nominal_actions
        objective = float(np.asarray(self._objective(jnp.asarray(action_array))))
        summary = {
            "objective": objective,
            "mean_reward": float(np.mean(arrays["rewards"])),
            "terminal_count": terminal_count,
            "support_switch_count": support.switch_count,
            "support_switch_phases": support_switch_phases,
            "required_support_switch_phase": required_support_switch_phase,
            "required_support_switch_present": bool(required_support_switch_present),
            "required_support_before": support.support[0].tolist(),
            "required_support_after": support.support[1].tolist(),
            "phase_start": int(state_phases[0]),
            "phase_end": int(state_phases[-1]),
            "state_count": int(state_phases.size),
            "transition_count": int(transition_phases.size),
            "chronology_valid": bool(chronology_valid),
            "all_finite": bool(all_finite and np.isfinite(objective)),
            "initial_state_origin_phase": 0,
            "initial_state_prefix_transitions": self.args.start_phase,
            "initial_state_source": "full-carried-mjx-data",
            "phase_zero_reset_count": 1,
            "task_reset_calls_after_phase_zero": 0,
            "reconstructed_state_count": 0,
            "minimum_action": float(np.min(action_array)),
            "maximum_action": float(np.max(action_array)),
            "maximum_action_deviation": float(np.max(np.abs(deviation))),
            "rms_action_deviation": float(np.sqrt(np.mean(deviation**2))),
            "max_anchor_z_error": float(np.max(arrays["anchor_z_error"])),
            "max_anchor_xy_error": float(np.max(arrays["anchor_xy_error"])),
            "max_gravity_z_error": float(np.max(arrays["gravity_z_error"])),
            "max_distal_z_error": float(np.max(arrays["distal_z_error"])),
        }
        feasible = bool(
            summary["all_finite"]
            and chronology_valid
            and terminal_count == 0
            and required_support_switch_present
        )
        return PhysicalEvaluation(
            objective=objective,
            feasible=feasible,
            summary=summary,
            arrays=arrays,
        )

    def _canonical_gradient(self, actions: np.ndarray):
        actions_jax = jnp.asarray(actions, dtype=jnp.float64)
        return canonical_forward_gradient(
            self._objective,
            actions_jax,
            physical_fn=self._physical,
            identity_tolerance=1e-8,
            directional_jvp=self._directional_jvp,
        )

    def gradient_preflight(self, actions) -> dict:
        """Require two full forward sweeps and one centered FD audit."""
        self._ensure_nominal()
        action_array = np.asarray(actions, dtype=np.float64)
        try:
            reports = [
                self._canonical_gradient(action_array)
                for _ in range(self.args.gradient_repeat_count)
            ]
            maximum_repeat_error = max(
                float(np.max(np.abs(report.gradient - reports[0].gradient)))
                for report in reports[1:]
            )
            fd_audit = directional_fd_audit(
                self._objective,
                jnp.asarray(action_array),
                reports[0].gradient,
                epsilon=self.args.finite_difference_epsilon,
                seed=20260808,
                support_gate=lambda probe: self.evaluate(np.asarray(probe)).feasible,
            )
            maximum_primal_error = max(
                report.maximum_primal_error for report in reports
            )
            scalar_jvps = reports[0].scalar_jvps
            gradients_finite = all(
                np.isfinite(report.gradient).all() for report in reports
            )
            passed = bool(
                scalar_jvps == self.args.horizon * self.env.action_dim
                and gradients_finite
                and maximum_primal_error <= 1e-8
                and maximum_repeat_error <= 1e-6
                and fd_audit.relative_error <= 0.05
                and fd_audit.positive_support_safe
                and fd_audit.negative_support_safe
            )
            self._gradient_cache[self._action_key(action_array)] = np.array(
                reports[0].gradient, copy=True
            )
            return {
                "passed": passed,
                "scalar_jvps": scalar_jvps,
                "repeat_count": len(reports),
                "maximum_primal_error": float(maximum_primal_error),
                "maximum_repeat_error": maximum_repeat_error,
                "gradients_finite": bool(gradients_finite),
                "fd_epsilon": self.args.finite_difference_epsilon,
                "fd_seed": 20260808,
                "fd_autodiff_directional_derivative": (
                    fd_audit.autodiff_directional_derivative
                ),
                "fd_finite_difference_directional_derivative": (
                    fd_audit.finite_difference_directional_derivative
                ),
                "fd_relative_error": fd_audit.relative_error,
                "fd_positive_support_safe": fd_audit.positive_support_safe,
                "fd_negative_support_safe": fd_audit.negative_support_safe,
                "physical_primal_shape": [self.args.horizon, 71],
                "identity_tolerance": 1e-8,
                "repeat_tolerance": 1e-6,
                "fd_relative_error_tolerance": 0.05,
            }
        except (FloatingPointError, ValueError) as error:
            return {
                "passed": False,
                "scalar_jvps": 0,
                "repeat_count": 0,
                "failure": str(error),
                "identity_tolerance": 1e-8,
                "repeat_tolerance": 1e-6,
                "fd_relative_error_tolerance": 0.05,
            }

    def gradient(self, actions) -> np.ndarray:
        action_array = np.asarray(actions, dtype=np.float64)
        key = self._action_key(action_array)
        if key not in self._gradient_cache:
            report = self._canonical_gradient(action_array)
            self._gradient_cache[key] = np.array(report.gradient, copy=True)
        return np.array(self._gradient_cache[key], copy=True)


def main() -> None:
    configure_jax()
    args = build_parser().parse_args()
    output_preexisting = args.output_dir.is_symlink() or args.output_dir.exists()
    try:
        with fixed_mjx_solver_outer_loop(semantic=args.solver_gradient_semantic):
            summary = execute_gate(args, runtime=G1PhysicalRuntime(args))
    except Exception as error:
        failure = {
            "protocol": "g1-lafan-carried-action-shooting-gate-v1",
            "classification": "invalid-execution",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if (
            not output_preexisting
            and args.output_dir.is_dir()
            and not args.output_dir.is_symlink()
        ):
            atomic_write_json(args.output_dir / "summary.json", failure)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
