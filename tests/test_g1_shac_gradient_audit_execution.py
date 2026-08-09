import json
import pickle
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flax
import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.shac.g1_gradient_audit_execution import (
    EstimatorShardEvidence,
    PreparedAuditExecution,
    make_frozen_action_noise,
    make_phase_rollout,
    replace_actor_parameters,
    run_audit,
    summarize_stochastic_rollout,
)


@flax.struct.dataclass
class FakeTrainState:
    actor_params: object
    untouched: jax.Array


def complete_stochastic_trajectory(
    rewards,
    dones,
    *,
    action_dimension=1,
    observation_dimension=2,
):
    rewards = jnp.asarray(rewards)
    population, horizon = rewards.shape
    return SimpleNamespace(
        noise=jnp.zeros((population, horizon, action_dimension)),
        observation_rngs=jnp.zeros((population, horizon, 2), dtype=jnp.uint32),
        raw_observations=jnp.zeros((population, horizon, observation_dimension)),
        observations=jnp.zeros((population, horizon, observation_dimension)),
        normalized_observations=jnp.zeros((population, horizon, observation_dimension)),
        means=jnp.zeros((population, horizon, action_dimension)),
        actions=jnp.zeros((population, horizon, action_dimension)),
        rewards=rewards,
        dones=jnp.asarray(dones),
        initial_phase=jnp.zeros((population,), dtype=jnp.int32),
    )


class FrozenNoiseTest(unittest.TestCase):
    def test_is_deterministic_distinct_and_exact_shape(self):
        first = make_frozen_action_noise(3)
        repeated = make_frozen_action_noise(3)
        other = make_frozen_action_noise(4)

        self.assertEqual(first.shape, (64, 48, 29))
        self.assertEqual(first.dtype, jnp.float64)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.isfinite(np.asarray(first)).all())


class CandidateCheckpointTest(unittest.TestCase):
    def test_replaces_only_actor_parameters(self):
        state = FakeTrainState(
            actor_params={"w": jnp.array([1.0])},
            untouched=jnp.array([7.0]),
        )
        replacement = {"w": jnp.array([2.0])}

        candidate = replace_actor_parameters(state, replacement)

        np.testing.assert_array_equal(candidate.actor_params["w"], [2.0])
        np.testing.assert_array_equal(candidate.untouched, state.untouched)
        np.testing.assert_array_equal(state.actor_params["w"], [1.0])


class StochasticRolloutSummaryTest(unittest.TestCase):
    def test_reports_discounted_return_survival_and_terminal_fraction(self):
        trajectory = complete_stochastic_trajectory(
            jnp.array([[1.0, 2.0, 50.0], [4.0, 5.0, 6.0]], dtype=jnp.float64),
            jnp.array([[False, True, False], [False, False, False]]),
        )

        summary = summarize_stochastic_rollout(trajectory, gamma=0.5)

        # Episode-start discount resets after done, matching the actor objective.
        np.testing.assert_allclose(summary["discounted_return_by_env"], [52.0, 8.0])
        np.testing.assert_array_equal(summary["survival_by_env"], [2, 3])
        self.assertAlmostEqual(summary["mean_discounted_return"], 30.0)
        self.assertAlmostEqual(summary["terminal_fraction"], 0.5)

    def test_rejects_shape_mismatch_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            summarize_stochastic_rollout(
                complete_stochastic_trajectory(jnp.ones((2, 3)), jnp.ones((2, 2))),
                gamma=0.99,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_stochastic_rollout(
                complete_stochastic_trajectory(
                    jnp.array([[jnp.nan]]), jnp.array([[False]])
                ),
                gamma=0.99,
            )

    def test_rejects_incomplete_or_nonfinite_trajectory_evidence(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize_stochastic_rollout(
                SimpleNamespace(rewards=jnp.ones((2, 3)), dones=jnp.zeros((2, 3))),
                gamma=0.99,
            )
        trajectory = complete_stochastic_trajectory(
            jnp.ones((2, 3)), jnp.zeros((2, 3), dtype=bool)
        )
        trajectory.noise = trajectory.noise.at[0, 0, 0].set(jnp.nan)
        with self.assertRaisesRegex(ValueError, "noise.*nonfinite"):
            summarize_stochastic_rollout(trajectory, gamma=0.99)


@flax.struct.dataclass
class FakeData:
    qpos: jax.Array
    qvel: jax.Array


@flax.struct.dataclass
class FakeEvaluationState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict
    data: FakeData


class FakeEvaluationEnv:
    action_dim = 1
    actor_frame_obs_dim = 1

    def reset_at_phase(self, key, _difficulty, phase):
        del key
        return FakeEvaluationState(
            obs=jnp.array([1.0]),
            reward=jnp.array(0.0),
            done=jnp.array(False),
            info={"phase": phase, "terminal": jnp.array(False), "step": jnp.array(0)},
            data=FakeData(qpos=jnp.array([0.0]), qvel=jnp.array([0.0])),
        )

    def normalize_actor_obs(self, _normalizer, norm_state, obs):
        return obs * norm_state["scale"]

    def step(self, state, action):
        step = state.info["step"] + 1
        done = step == 2
        return state.replace(
            obs=state.obs + action,
            reward=action[0],
            done=done,
            info={
                **state.info,
                "phase": state.info["phase"] + 1,
                "terminal": done,
                "step": step,
            },
            data=FakeData(qpos=state.data.qpos + action, qvel=action),
        )


class PhaseRolloutTest(unittest.TestCase):
    def test_is_replay_free_stops_after_done_and_materializes_arrays(self):
        rollout = make_phase_rollout(
            FakeEvaluationEnv(),
            lambda params, obs: params["gain"] * obs,
            normalizer=object(),
            max_steps=4,
        )

        result = rollout(
            {"gain": jnp.array([0.5])},
            {"scale": jnp.array(2.0)},
            seed=7,
            phase=10,
        )

        self.assertEqual(result["phase"], 10)
        self.assertEqual(result["survival"], 2)
        self.assertTrue(result["terminal"])
        self.assertTrue(result["complete"])
        self.assertTrue(result["replay_free"])
        np.testing.assert_allclose(result["rewards"], [1.0, 2.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["active"], [True, True, False, False])
        self.assertAlmostEqual(result["return"], 1.5)
        self.assertAlmostEqual(result["reward_sum"], 3.0)
        self.assertEqual(result["qpos"].shape, (4, 1))
        self.assertEqual(result["actions"].shape, (4, 1))


class RunAuditTest(unittest.TestCase):
    @staticmethod
    def _gradient_result(seed):
        value = float(seed + 1)
        gradients = {"w": jnp.full((64, 1), value, dtype=jnp.float64)}
        score_gradients = {"w": jnp.full((64, 1), value + 0.25, dtype=jnp.float64)}
        trajectory = SimpleNamespace(
            initial_phase=jnp.arange(64, dtype=jnp.int32) % 5 * 100,
            normalized_observations=jnp.ones((64, 48, 1), dtype=jnp.float64),
            rewards=jnp.ones((64, 48), dtype=jnp.float64) * value,
            dones=jnp.zeros((64, 48), dtype=bool),
        )
        return SimpleNamespace(
            pathwise_effective_gradients=gradients,
            pathwise_raw_gradients=gradients,
            score_gradients=score_gradients,
            trajectory=trajectory,
        )

    @staticmethod
    def _ordinary_row(*, seed, phase, gain):
        rewards = np.array([gain + seed * 0.001, gain + 0.5], dtype=np.float64)
        return {
            "phase": phase,
            "return": float(np.mean(rewards)),
            "reward_sum": float(np.sum(rewards)),
            "survival": 2,
            "terminal": True,
            "complete": True,
            "replay_free": True,
            "active": np.array([True, True]),
            "rewards": rewards,
            "dones": np.array([False, True]),
            "terminals": np.array([False, True]),
            "actions": np.full((2, 1), gain),
            "phases": np.array([phase + 1, phase + 2]),
            "qpos": np.full((2, 1), phase),
            "qvel": np.full((2, 1), gain),
        }

    def test_runs_exact_e064_transaction_and_hashes_every_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint_final.pkl"
            reference = root / "reference.npz"
            hparams_path = root / "hparams.json"
            checkpoint.write_bytes(b"checkpoint input")
            reference.write_bytes(b"reference input")
            hparams_path.write_text('{"frozen": true}\n')
            contract = SimpleNamespace(
                checkpoint=checkpoint,
                checkpoint_sha256="a" * 64,
                reference=reference,
                reference_sha256="b" * 64,
                hparams_path=hparams_path,
                output_dir=root / "audit",
                shard_seeds=(0, 1, 2, 3),
                held_out_seeds=(4, 5, 6, 7),
                phases=(0, 100, 200, 300, 400),
                horizon=48,
                population=64,
                sigma=0.1,
                gamma=0.99,
                per_env_clip=1.0,
                functional_rms=0.01,
                solver_iterations=4,
                solver_ls_iterations=5,
            )
            state = FakeTrainState(
                actor_params={"w": jnp.array([1.0], dtype=jnp.float64)},
                untouched=jnp.array([7.0]),
            )
            shard_calls = []
            stochastic_calls = []
            phase_calls = []
            events = []
            solver_context_active = [False]

            @contextmanager
            def gradient_solver_context():
                self.assertFalse(solver_context_active[0])
                solver_context_active[0] = True
                events.append("fixed-solver-enter")
                try:
                    yield
                finally:
                    solver_context_active[0] = False
                    events.append("fixed-solver-exit")

            def estimate_shard(seed):
                self.assertTrue(solver_context_active[0])
                shard_calls.append(seed)
                result = self._gradient_result(seed)
                receipt = {"seed": str(seed), "identity": "same"}
                return EstimatorShardEvidence(result, receipt, receipt)

            def stochastic_rollout(actor_params, action_noise):
                self.assertTrue(solver_context_active[0])
                stochastic_calls.append(
                    (
                        float(actor_params["w"][0]),
                        np.asarray(action_noise).copy(),
                    )
                )
                return complete_stochastic_trajectory(
                    jnp.ones((64, 48), dtype=jnp.float64) * actor_params["w"][0],
                    jnp.zeros((64, 48), dtype=bool),
                    action_dimension=29,
                    observation_dimension=154,
                )

            def phase_rollout(actor_params, normalizer_state, *, seed, phase):
                del normalizer_state
                self.assertFalse(solver_context_active[0])
                events.append("numeric")
                phase_calls.append((seed, phase, float(actor_params["w"][0])))
                return self._ordinary_row(
                    seed=seed, phase=phase, gain=float(actor_params["w"][0])
                )

            def render_phase_zero(*, rows, output_dir):
                self.assertEqual(len(phase_calls), 5 * 3)
                self.assertTrue(
                    (output_dir / "ordinary_phase_grid_arrays.npz").is_file()
                )
                events.append("video")
                paths = {}
                for label in ("baseline", "pathwise", "score"):
                    self.assertIn((0, label, 0), rows)
                    path = output_dir / "videos" / f"{label}_phase0.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(label.encode())
                    paths[label] = path
                return paths

            prepared = PreparedAuditExecution(
                checkpoint_state=state,
                actor_apply=lambda params, observations: observations * params["w"],
                normalizer_state={"mean": jnp.array([0.0])},
                estimate_shard=estimate_shard,
                stochastic_rollout=stochastic_rollout,
                phase_rollout=phase_rollout,
                validated_contract={"hparams_sha256": "c" * 64},
                algorithmic_validity={
                    "analytic_gaussian_sign": True,
                    "detachment": True,
                    "done_boundary_return": True,
                    "ppo_ratio_one": True,
                    "pytree_order": True,
                    "smooth_toy_convergence": True,
                },
                external_inputs={"plant_xml": {"sha256": "d" * 64}},
                runtime_provenance={"code_commit": "e" * 40, "git_clean": True},
                gradient_solver_context=gradient_solver_context,
                render_phase_zero=render_phase_zero,
            )

            with patch(
                "src.algorithms.shac.g1_gradient_audit_execution._prepare_e064_execution",
                return_value=prepared,
            ):
                manifest = run_audit(contract)

            self.assertEqual(shard_calls, [0, 1, 2, 3])
            self.assertEqual(len(stochastic_calls), 4 * 3)
            for offset in range(0, len(stochastic_calls), 3):
                np.testing.assert_array_equal(
                    stochastic_calls[offset][1], stochastic_calls[offset + 1][1]
                )
                np.testing.assert_array_equal(
                    stochastic_calls[offset][1], stochastic_calls[offset + 2][1]
                )
            self.assertEqual(len(phase_calls), 5 * 3)
            self.assertEqual(events[-1], "video")

            output = contract.output_dir
            loaded_manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(
                loaded_manifest["thresholds"],
                {
                    "bootstrap_confidence_level": 0.95,
                    "material_stability_advantage": 0.1,
                    "maximum_phase_survival_loss_fraction": 0.1,
                    "minimum_alignment": 0.7,
                    "minimum_return_improvement": 0.001,
                    "minimum_stability": 0.2,
                    "minimum_survival_improvement": 5.0,
                    "return_tolerance": 0.001,
                    "stability_tolerance": 0.05,
                    "survival_tolerance": 2.0,
                },
            )
            self.assertEqual(
                loaded_manifest["ordinary_baseline_reproduction"]["expected_survival"],
                [135, 236, 152, 83, 74],
            )
            self.assertFalse(
                loaded_manifest["ordinary_baseline_reproduction"]["exact_match"]
            )
            self.assertEqual(loaded_manifest["outcome"]["verdict"], "invalid")
            self.assertEqual(
                loaded_manifest["external_inputs"]["plant_xml"]["sha256"],
                "d" * 64,
            )
            self.assertTrue(loaded_manifest["runtime_provenance"]["git_clean"])
            self.assertTrue(loaded_manifest["heldout_stochastic_finite_complete"])
            self.assertEqual(
                loaded_manifest["solver_trace_context"],
                {
                    "gradient_shards": "fixed_mjx_solver_outer_loop",
                    "heldout_stochastic": "fixed_mjx_solver_outer_loop",
                    "ordinary_phase_grid": "stock_mjx_forward_solver",
                },
            )
            self.assertEqual(
                sorted(
                    path
                    for path in loaded_manifest["artifacts"]
                    if path.endswith("_candidate.pkl")
                ),
                [
                    "baseline_candidate.pkl",
                    "pathwise_candidate.pkl",
                    "score_candidate.pkl",
                ],
            )
            for relative, digest in loaded_manifest["artifacts"].items():
                artifact = output / relative
                self.assertTrue(artifact.is_file(), relative)
                import hashlib

                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(), digest
                )

            for label in ("baseline", "pathwise", "score"):
                with (output / f"{label}_candidate.pkl").open("rb") as stream:
                    candidate = pickle.load(stream)
                np.testing.assert_array_equal(candidate.untouched, state.untouched)
            with np.load(output / "ordinary_phase_grid_arrays.npz") as archive:
                self.assertIn("seed_0/baseline/phase_0/rewards", archive.files)
                rewards = archive["seed_0/baseline/phase_0/rewards"]
                summary = json.loads((output / "ordinary_phase_grid.json").read_text())
                row = summary["per_seed"][0]["candidates"]["baseline"][0]
                self.assertEqual(summary["seed"], 0)
                self.assertAlmostEqual(row["return"], float(np.mean(rewards)))
                self.assertEqual(row["survival"], 2)

    def test_numeric_failure_prevents_video_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            contract = SimpleNamespace(
                checkpoint=Path(directory) / "checkpoint.pkl",
                checkpoint_sha256="a" * 64,
                reference=Path(directory) / "reference.npz",
                reference_sha256="b" * 64,
                hparams_path=Path(directory) / "hparams.json",
                output_dir=output,
                shard_seeds=(0, 1, 2, 3),
                held_out_seeds=(4, 5, 6, 7),
                phases=(0, 100, 200, 300, 400),
                horizon=48,
                population=64,
                sigma=0.1,
                gamma=0.99,
                per_env_clip=1.0,
                functional_rms=0.01,
                solver_iterations=4,
                solver_ls_iterations=5,
            )
            for path in (
                contract.checkpoint,
                contract.reference,
                contract.hparams_path,
            ):
                path.write_bytes(b"input")
            rendered = []
            state = FakeTrainState(
                actor_params={"w": jnp.array([1.0])}, untouched=jnp.array([7.0])
            )
            receipt = {"identity": "same"}
            prepared = PreparedAuditExecution(
                checkpoint_state=state,
                actor_apply=lambda params, observations: observations * params["w"],
                normalizer_state={},
                estimate_shard=lambda seed: EstimatorShardEvidence(
                    self._gradient_result(seed), receipt, receipt
                ),
                stochastic_rollout=lambda params, noise: complete_stochastic_trajectory(
                    jnp.ones((64, 48)),
                    jnp.zeros((64, 48), dtype=bool),
                    action_dimension=29,
                    observation_dimension=154,
                ),
                phase_rollout=lambda *args, **kwargs: {
                    **self._ordinary_row(
                        seed=kwargs["seed"], phase=kwargs["phase"], gain=1.0
                    ),
                    "rewards": np.array([np.nan, 1.0]),
                },
                validated_contract={},
                algorithmic_validity={
                    "analytic_gaussian_sign": True,
                    "detachment": True,
                    "done_boundary_return": True,
                    "ppo_ratio_one": True,
                    "pytree_order": True,
                    "smooth_toy_convergence": True,
                },
                render_phase_zero=lambda **kwargs: rendered.append(kwargs),
            )
            with (
                patch(
                    "src.algorithms.shac.g1_gradient_audit_execution._prepare_e064_execution",
                    return_value=prepared,
                ),
                self.assertRaisesRegex(ValueError, "nonfinite"),
            ):
                run_audit(contract)

            self.assertEqual(rendered, [])
            self.assertFalse((output / "manifest.json").exists())

    def test_rejects_nonfrozen_heldout_seeds_before_preparation(self):
        contract = SimpleNamespace(
            output_dir=Path("unused"),
            shard_seeds=(0, 1, 2, 3),
            held_out_seeds=(4, 5, 6, 8),
            phases=(0, 100, 200, 300, 400),
            population=64,
            horizon=48,
            sigma=0.1,
            gamma=0.99,
            per_env_clip=1.0,
            functional_rms=0.01,
            solver_iterations=4,
            solver_ls_iterations=5,
        )
        with (
            patch(
                "src.algorithms.shac.g1_gradient_audit_execution._prepare_e064_execution"
            ) as prepare,
            self.assertRaisesRegex(ValueError, "held_out_seeds"),
        ):
            run_audit(contract)

        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
