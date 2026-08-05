"""Paired replay-free evaluation and side-by-side G1/reference video."""

import argparse
import pickle
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.envs.g1_tracking.environment import G1TrackingEnv


def _load_policy(
    env: G1TrackingEnv, checkpoint: Path | None, seed: int
):
    actor = Actor(env.action_dim)
    if checkpoint is not None:
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        return actor, state.actor_params, state.normalizer

    params = actor.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, env.actor_obs_dim), dtype=jnp.float32),
    )
    return actor, params, Normalizer(env.actor_frame_obs_dim).init()


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.7
    camera.azimuth = 120.0
    camera.elevation = -15.0
    return camera


def _render_pair(
    env: G1TrackingEnv,
    actual_qpos: np.ndarray,
    actual_qvel: np.ndarray,
    phase: int,
    actual_renderer: mujoco.Renderer,
    reference_renderer: mujoco.Renderer,
    actual_data: mujoco.MjData,
    reference_data: mujoco.MjData,
) -> np.ndarray:
    actual_data.qpos[:] = actual_qpos
    actual_data.qvel[:] = actual_qvel
    mujoco.mj_forward(env.mj_model, actual_data)
    reference_data.qpos[:] = env.reference.qpos[phase]
    reference_data.qvel[:] = env.reference.qvel[phase]
    mujoco.mj_forward(env.mj_model, reference_data)
    camera = _camera()
    camera.lookat[:] = env.reference.body_pos[phase, 0]
    actual_renderer.update_scene(actual_data, camera)
    reference_renderer.update_scene(reference_data, camera)
    return np.concatenate(
        (actual_renderer.render(), reference_renderer.render()), axis=1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rmr-action-tape", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = G1TrackingEnv(actor_history_len=1)
    if args.checkpoint is not None and args.rmr_action_tape is not None:
        parser.error("--checkpoint and --rmr-action-tape are mutually exclusive")
    actor = actor_params = normalizer_state = None
    action_tape = None
    if args.rmr_action_tape is not None:
        with np.load(args.rmr_action_tape, allow_pickle=False) as archive:
            source_names = tuple(map(str, archive["joint_names"]))
            permutation = np.array(
                [
                    source_names.index(name)
                    for name in env.controller.joint_names
                ]
            )
            action_tape = np.asarray(archive["action"][:, permutation])
    else:
        actor, actor_params, normalizer_state = _load_policy(
            env, args.checkpoint, args.seed
        )
    state = env.reset_at_phase(
        jax.random.PRNGKey(args.seed),
        jnp.array(0.0),
        jnp.array(args.phase),
    )

    actual_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    reference_renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    actual_data = mujoco.MjData(env.mj_model)
    reference_data = mujoco.MjData(env.mj_model)
    frames = []
    records = []

    for step in range(args.max_steps):
        phase = int(state.info["phase"])
        if step % args.render_every == 0:
            frames.append(
                _render_pair(
                    env,
                    np.asarray(state.data.qpos),
                    np.asarray(state.data.qvel),
                    phase,
                    actual_renderer,
                    reference_renderer,
                    actual_data,
                    reference_data,
                )
            )
        if action_tape is not None:
            # The logged RMR controller runs at 50 Hz; the grounded reference
            # and MJX task run at 100 Hz. Zero-order hold each logged action for
            # two reference frames without using it for policy training.
            action = jnp.asarray(
                action_tape[min(phase // 2, len(action_tape) - 1)]
            )
        else:
            normalized = env.normalize_actor_obs(
                Normalizer(env.actor_frame_obs_dim),
                normalizer_state,
                state.obs,
            ).astype(jnp.float32)
            action = actor.apply(actor_params, normalized).astype(jnp.float64)
        state = env.step(state, action)
        records.append(
            (
                step,
                phase,
                float(state.reward),
                float(state.done),
                float(state.info["terminal"]),
                float(state.metrics["anchor_position_error"]),
                float(state.metrics["anchor_orientation_error"]),
                float(state.metrics["body_position_error"]),
                float(state.metrics["body_orientation_error"]),
                float(state.metrics["body_linear_velocity_error"]),
                float(state.metrics["body_angular_velocity_error"]),
            )
        )
        if float(state.done) > 0.5:
            break

    columns = (
        "step",
        "phase",
        "reward",
        "done",
        "terminal",
        "anchor_position_error",
        "anchor_orientation_error",
        "body_position_error",
        "body_orientation_error",
        "body_linear_velocity_error",
        "body_angular_velocity_error",
    )
    values = np.asarray(records, dtype=np.float64)
    np.savez_compressed(
        args.output_dir / "evaluation.npz",
        columns=np.asarray(columns),
        values=values,
    )
    imageio.mimsave(
        args.output_dir / "evaluation.mp4",
        frames,
        fps=round(1.0 / (env.dt * args.render_every)),
        quality=8,
    )
    summary = {
        "steps": len(records),
        "terminal": bool(values[-1, 4] > 0.5),
        "mean_reward": float(np.mean(values[:, 2])),
        "mean_anchor_position_error": float(np.mean(values[:, 5])),
        "mean_anchor_orientation_error": float(np.mean(values[:, 6])),
        "mean_body_position_error": float(np.mean(values[:, 7])),
        "mean_body_orientation_error": float(np.mean(values[:, 8])),
        "mean_body_linear_velocity_error": float(np.mean(values[:, 9])),
        "mean_body_angular_velocity_error": float(np.mean(values[:, 10])),
    }
    (args.output_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
