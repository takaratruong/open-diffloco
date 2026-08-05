"""Run semantic and production MJX gradient gates for the G1 tracking task."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.g1_tracking.environment import G1TrackingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = G1TrackingEnv(actor_history_len=1)
    state = env.reset(jax.random.PRNGKey(args.seed), jnp.array(0.0))
    reset_reward, reset_components = env._tracking_reward(state.data, state.info)
    semantic = {
        "seed": args.seed,
        "phase": int(state.info["phase"]),
        "reference_reward": float(reset_reward),
        "reference_components": {
            key: float(value) for key, value in reset_components.items()
        },
        "action_dim": env.action_dim,
        "actor_obs_dim": env.actor_obs_dim,
        "critic_obs_dim": env.critic_obs_dim,
        "physics_dt": float(env.mj_model.opt.timestep),
        "control_dt": env.dt,
        "physics_substeps": env.n_frames,
        "reference_frames": env.reference_length,
        "finite_reset_qpos": bool(jnp.isfinite(state.data.qpos).all()),
        "finite_reset_qvel": bool(jnp.isfinite(state.data.qvel).all()),
    }
    (args.output_dir / "semantic_gate.json").write_text(
        json.dumps(semantic, indent=2, sort_keys=True) + "\n"
    )

    def reward_for_action(action):
        return env.step(state, action).reward

    value_and_grad = jax.jit(jax.value_and_grad(reward_for_action))
    action = jnp.zeros(env.action_dim, dtype=jnp.float64)
    started = time.perf_counter()
    reward, gradient = value_and_grad(action)
    jax.block_until_ready((reward, gradient))
    compile_and_execute_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reward_repeat, gradient_repeat = value_and_grad(action)
    jax.block_until_ready((reward_repeat, gradient_repeat))
    cached_execute_seconds = time.perf_counter() - started

    gradient_np = np.asarray(gradient)
    gradient_repeat_np = np.asarray(gradient_repeat)
    gradient_gate = {
        "seed": args.seed,
        "phase": int(state.info["phase"]),
        "reward": float(reward),
        "repeat_reward": float(reward_repeat),
        "gradient": gradient_np.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient_np)),
        "gradient_max_abs": float(np.max(np.abs(gradient_np))),
        "gradient_finite": bool(np.isfinite(gradient_np).all()),
        "gradient_nonzero": bool(np.linalg.norm(gradient_np) > 0.0),
        "repeat_exact": bool(np.array_equal(gradient_np, gradient_repeat_np)),
        "repeat_max_abs_error": float(
            np.max(np.abs(gradient_np - gradient_repeat_np))
        ),
        "compile_and_execute_seconds": compile_and_execute_seconds,
        "cached_execute_seconds": cached_execute_seconds,
    }
    (args.output_dir / "gradient_gate.json").write_text(
        json.dumps(gradient_gate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"semantic": semantic, "gradient": gradient_gate}, indent=2))
    if not (
        semantic["finite_reset_qpos"]
        and semantic["finite_reset_qvel"]
        and abs(semantic["reference_reward"] - 5.0) < 1e-5
        and gradient_gate["gradient_finite"]
        and gradient_gate["gradient_nonzero"]
        and gradient_gate["repeat_exact"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
