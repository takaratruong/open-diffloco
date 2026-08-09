import hashlib
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
)


@flax.struct.dataclass
class FakeTrainState:
    actor_params: object
    untouched: jax.Array


def complete_stochastic_trajectory(value):
    population, horizon = 64, 48
    return SimpleNamespace(
        noise=jnp.zeros((population, horizon, 29), dtype=jnp.float64),
        observation_rngs=jnp.zeros((population, horizon, 2), dtype=jnp.uint32),
        raw_observations=jnp.zeros((population, horizon, 154), dtype=jnp.float64),
        observations=jnp.zeros((population, horizon, 154), dtype=jnp.float64),
        normalized_observations=jnp.zeros(
            (population, horizon, 154), dtype=jnp.float64
        ),
        means=jnp.zeros((population, horizon, 29), dtype=jnp.float64),
        actions=jnp.zeros((population, horizon, 29), dtype=jnp.float64),
        rewards=jnp.full((population, horizon), value, dtype=jnp.float64),
        dones=jnp.zeros((population, horizon), dtype=bool),
        initial_phase=jnp.arange(population, dtype=jnp.int32) % 5 * 100,
    )


class FailureWeightedExecutionTest(unittest.TestCase):
    @staticmethod
    def _gradient_result(seed):
        environment = jnp.arange(64, dtype=jnp.float64)
        losses = (environment * (seed + 3)) % 17
        phases = jnp.asarray(
            [
                *range(4),
                *range(100, 117),
                *range(200, 219),
                *range(300, 308),
                *range(400, 416),
            ],
            dtype=jnp.int32,
        )
        gradients = {
            "w": jnp.stack(
                (
                    1.0 + 0.01 * environment + seed,
                    0.5 + ((environment + 2 * seed) % 11) / 10.0,
                ),
                axis=1,
            ).reshape((64, 2, 1))
        }
        trajectory = SimpleNamespace(
            initial_phase=phases,
            normalized_observations=jnp.tile(
                jnp.asarray([[[1.0, 0.5]]], dtype=jnp.float64), (64, 48, 1)
            ),
        )
        return SimpleNamespace(
            losses=losses,
            trajectory=trajectory,
            pathwise_effective_gradients=gradients,
        )

    @staticmethod
    def _ordinary_row(*, phase, gain, survival):
        rewards = np.full(survival, 0.08 + gain * 1e-4, dtype=np.float64)
        dones = np.zeros(survival, dtype=bool)
        dones[-1] = True
        return {
            "phase": phase,
            "return": float(np.mean(rewards)),
            "reward_sum": float(np.sum(rewards)),
            "survival": survival,
            "terminal": True,
            "complete": True,
            "replay_free": True,
            "active": np.ones(survival, dtype=bool),
            "rewards": rewards,
            "dones": dones,
            "terminals": dones.copy(),
            "actions": np.full((survival, 1), gain),
            "phases": np.arange(phase + 1, phase + survival + 1),
            "qpos": np.full((survival, 1), phase),
            "qvel": np.full((survival, 1), gain),
        }

    @staticmethod
    def _contract(root):
        checkpoint = root / "checkpoint.pkl"
        reference = root / "reference.npz"
        hparams = root / "hparams.json"
        checkpoint.write_bytes(b"checkpoint")
        reference.write_bytes(b"reference")
        hparams.write_text('{"frozen": true}\n')
        return SimpleNamespace(
            checkpoint=checkpoint,
            checkpoint_sha256="a" * 64,
            reference=reference,
            reference_sha256="b" * 64,
            hparams_path=hparams,
            output_dir=root / "evidence",
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

    def test_publishes_complete_manifest_bound_failure_weighted_evidence(self):
        from src.algorithms.shac.g1_failure_weighted_audit_execution import run_audit

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._contract(root)
            state = FakeTrainState(
                actor_params={"w": jnp.asarray([[1.0], [0.5]], dtype=jnp.float64)},
                untouched=jnp.asarray([7.0]),
            )
            shard_calls = []
            stochastic_calls = []
            phase_calls = []
            solver_active = [False]

            @contextmanager
            def solver_context():
                self.assertFalse(solver_active[0])
                solver_active[0] = True
                try:
                    yield
                finally:
                    solver_active[0] = False

            def estimate_shard(seed):
                self.assertTrue(solver_active[0])
                shard_calls.append(seed)
                receipt = {"identity": "same", "seed": seed}
                return EstimatorShardEvidence(
                    self._gradient_result(seed), receipt, receipt
                )

            def stochastic_rollout(actor_params, action_noise):
                self.assertTrue(solver_active[0])
                stochastic_calls.append(
                    (float(jnp.sum(actor_params["w"])), np.asarray(action_noise))
                )
                return complete_stochastic_trajectory(float(jnp.sum(actor_params["w"])))

            survivals = {0: 135, 100: 230, 200: 157, 300: 82, 400: 75}

            def phase_rollout(actor_params, _normalizer, *, seed, phase):
                self.assertFalse(solver_active[0])
                gain = float(jnp.sum(actor_params["w"]))
                phase_calls.append((seed, phase, gain))
                return self._ordinary_row(
                    phase=phase,
                    gain=gain,
                    survival=survivals[phase],
                )

            def render_phase_zero(*, rows, output_dir):
                # The reused E064 renderer consumes aliases; published names must not.
                self.assertIn((0, "pathwise", 0), rows)
                self.assertIn((0, "score", 0), rows)
                paths = {}
                for label in ("baseline", "pathwise", "score"):
                    path = output_dir / "videos" / f"{label}_phase0.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(label.encode())
                    paths[label] = path
                return paths

            prepared = PreparedAuditExecution(
                checkpoint_state=state,
                actor_apply=lambda params, observations: observations @ params["w"],
                normalizer_state={"mean": jnp.asarray([0.0])},
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
                gradient_solver_context=solver_context,
                render_phase_zero=render_phase_zero,
            )

            with patch(
                "src.algorithms.shac.g1_failure_weighted_audit_execution._prepare_e064_execution",
                return_value=prepared,
            ):
                manifest = run_audit(contract)

            self.assertEqual(shard_calls, [0, 1, 2, 3])
            self.assertEqual(len(stochastic_calls), 12)
            for offset in range(0, 12, 3):
                np.testing.assert_array_equal(
                    stochastic_calls[offset][1], stochastic_calls[offset + 1][1]
                )
                np.testing.assert_array_equal(
                    stochastic_calls[offset][1], stochastic_calls[offset + 2][1]
                )
            self.assertEqual(len(phase_calls), 15)

            output = contract.output_dir
            loaded = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest, loaded)
            self.assertEqual(
                loaded["schema_version"], "g1-shac-failure-weighted-audit/v1"
            )
            self.assertEqual(
                set(loaded["validity"]),
                {
                    "frozen_hashes",
                    "weight_receipts_exact",
                    "uniform_reproduction_exact",
                    "tail_reproduction_exact",
                    "stability_evidence_exact",
                    "aggregate_gradients_finite_nonzero",
                    "candidate_trees_finite_nonzero",
                    "functional_steps_valid",
                    "rollouts_fresh_replay_free_complete_finite",
                },
            )
            self.assertTrue(all(loaded["validity"].values()))
            self.assertEqual(
                sorted(
                    path
                    for path in loaded["artifacts"]
                    if path.endswith("_candidate.pkl")
                ),
                [
                    "baseline_candidate.pkl",
                    "tail_candidate.pkl",
                    "uniform_candidate.pkl",
                ],
            )
            self.assertEqual(
                sorted(path for path in loaded["artifacts"] if path.endswith(".mp4")),
                [
                    "videos/baseline_phase0.mp4",
                    "videos/tail_phase0.mp4",
                    "videos/uniform_phase0.mp4",
                ],
            )
            for relative, digest in loaded["artifacts"].items():
                artifact = output / relative
                self.assertTrue(artifact.is_file(), relative)
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(), digest
                )

            weighting = json.loads(
                (output / "failure_weight_receipts.json").read_text()
            )
            self.assertTrue(
                weighting["independent_recomputation"]["weight_receipts_exact"]
            )
            self.assertTrue(
                weighting["independent_recomputation"]["uniform_reproduction_exact"]
            )
            self.assertTrue(
                weighting["independent_recomputation"]["tail_reproduction_exact"]
            )
            self.assertTrue(
                weighting["independent_recomputation"]["stability_evidence_exact"]
            )
            host_recomputation = weighting["independent_recomputation"][
                "independent_host_recomputation"
            ]
            self.assertEqual(host_recomputation["tolerance_units"], 256)
            self.assertIn("uniform_maximum_absolute_error", host_recomputation)
            self.assertIn("tail_maximum_absolute_error", host_recomputation)
            self.assertEqual(len(weighting["gradient_hashes"]["per_shard"]), 4)
            self.assertIn(
                "per_environment_clipped", weighting["gradient_hashes"]["per_shard"][0]
            )
            self.assertIn("uniform", weighting["gradient_hashes"]["aggregate"])
            self.assertIn("tail", weighting["gradient_hashes"]["aggregate"])

            functional = json.loads(
                (output / "functional_step_receipt.json").read_text()
            )
            self.assertTrue(functional["candidate_reconstruction"]["exact"])
            self.assertTrue(functional["candidate_reconstruction"]["dtype_tree_exact"])
            with np.load(output / "ordinary_phase_grid_arrays.npz") as arrays:
                self.assertIn("seed_0/tail/phase_400/qvel", arrays.files)
            for label in ("baseline", "uniform", "tail"):
                with (output / f"{label}_candidate.pkl").open("rb") as stream:
                    candidate = pickle.load(stream)
                self.assertEqual(
                    candidate.actor_params["w"].dtype, state.actor_params["w"].dtype
                )
                np.testing.assert_array_equal(candidate.untouched, state.untouched)

    def test_rejects_tampered_functional_step_summary(self):
        from src.algorithms.shac.g1_failure_weighted_audit import (
            build_failure_weighted_candidates,
        )
        from src.algorithms.shac.g1_failure_weighted_audit_execution import (
            _candidate_reconstruction,
        )

        actor_params = {
            "w": jnp.asarray([[1.0], [0.5]], dtype=jnp.float64)
        }
        aggregation = SimpleNamespace(
            uniform_mean={
                "w": jnp.asarray([[0.5], [0.25]], dtype=jnp.float64)
            },
            tail_mean={
                "w": jnp.asarray([[0.25], [0.75]], dtype=jnp.float64)
            },
            normalized_observations=jnp.asarray(
                [[1.0, 0.5]], dtype=jnp.float64
            ),
        )
        actor_apply = lambda params, observations: observations @ params["w"]
        candidates = build_failure_weighted_candidates(
            actor_apply=actor_apply,
            actor_params=actor_params,
            uniform_gradient=aggregation.uniform_mean,
            tail_gradient=aggregation.tail_mean,
            normalized_observations=aggregation.normalized_observations,
            target_rms=0.01,
        )
        tampered_steps = {
            **candidates.functional_steps,
            "uniform": {
                **candidates.functional_steps["uniform"],
                "output_rms": candidates.functional_steps["uniform"][
                    "output_rms"
                ]
                + 1.0,
            },
        }
        tampered_candidates = SimpleNamespace(
            baseline=candidates.baseline,
            uniform=candidates.uniform,
            tail=candidates.tail,
            functional_steps=tampered_steps,
        )

        with self.assertRaisesRegex(
            ValueError, "uniform functional step summary.*exactly"
        ):
            _candidate_reconstruction(
                actor_apply=actor_apply,
                actor_params=actor_params,
                aggregation=aggregation,
                candidates=tampered_candidates,
                target_rms=0.01,
            )

    def test_refuses_to_overwrite_completed_manifest_before_preparation(self):
        from src.algorithms.shac.g1_failure_weighted_audit_execution import run_audit

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._contract(root)
            contract.output_dir.mkdir()
            (contract.output_dir / "manifest.json").write_text("{}\n")
            with (
                patch(
                    "src.algorithms.shac.g1_failure_weighted_audit_execution._prepare_e064_execution"
                ) as prepare,
                self.assertRaisesRegex(FileExistsError, "refusing to overwrite"),
            ):
                run_audit(contract)
            prepare.assert_not_called()

    def test_consumes_authoritative_core_recomputation_receipt(self):
        from src.algorithms.shac.g1_failure_weighted_audit_execution import (
            _validate_aggregation,
        )

        authoritative = {
            "weight_receipts_exact": True,
            "uniform_reproduction_exact": True,
            "tail_reproduction_exact": True,
            "stability_evidence_exact": True,
            "independent_host_recomputation": {
                "tolerance_units": 256,
                "tolerance_model": (
                    "absolute_error <= tolerance_units * dtype_epsilon * "
                    "max(1, abs(expected))"
                ),
                "uniform_maximum_absolute_error": 1e-16,
                "tail_maximum_absolute_error": 2e-16,
            },
        }
        results = (object(),) * 4
        aggregation = object()
        with patch(
            "src.algorithms.shac.g1_failure_weighted_audit_execution.validate_failure_weighted_aggregation",
            return_value=authoritative,
        ) as validate:
            receipt = _validate_aggregation(results, aggregation)

        self.assertEqual(receipt, authoritative)
        validate.assert_called_once_with(results, aggregation)

        missing_host_metric = dict(authoritative)
        missing_host_metric["independent_host_recomputation"] = {
            **authoritative["independent_host_recomputation"]
        }
        del missing_host_metric["independent_host_recomputation"][
            "tail_maximum_absolute_error"
        ]
        with (
            patch(
                "src.algorithms.shac.g1_failure_weighted_audit_execution.validate_failure_weighted_aggregation",
                return_value=missing_host_metric,
            ),
            self.assertRaisesRegex(ValueError, "authoritative.*receipt"),
        ):
            _validate_aggregation(results, aggregation)

        wrong_tolerance = dict(authoritative)
        wrong_tolerance["independent_host_recomputation"] = {
            **authoritative["independent_host_recomputation"],
            "tolerance_units": 257,
        }
        with (
            patch(
                "src.algorithms.shac.g1_failure_weighted_audit_execution.validate_failure_weighted_aggregation",
                return_value=wrong_tolerance,
            ),
            self.assertRaisesRegex(ValueError, "frozen tolerance"),
        ):
            _validate_aggregation(results, aggregation)

    def test_estimator_identity_mismatch_aborts_before_publication(self):
        from src.algorithms.shac.g1_failure_weighted_audit_execution import run_audit

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._contract(root)

            def must_not_run(*_args, **_kwargs):
                self.fail("downstream evaluation ran after identity mismatch")

            prepared = PreparedAuditExecution(
                checkpoint_state=FakeTrainState(
                    actor_params={"w": jnp.ones((2, 1), dtype=jnp.float64)},
                    untouched=jnp.asarray([7.0]),
                ),
                actor_apply=must_not_run,
                normalizer_state={},
                estimate_shard=lambda seed: EstimatorShardEvidence(
                    self._gradient_result(seed),
                    {"identity": "pathwise"},
                    {"identity": "score"},
                ),
                stochastic_rollout=must_not_run,
                phase_rollout=must_not_run,
                validated_contract={},
                algorithmic_validity={
                    "analytic_gaussian_sign": True,
                    "detachment": True,
                    "done_boundary_return": True,
                    "ppo_ratio_one": True,
                    "pytree_order": True,
                    "smooth_toy_convergence": True,
                },
                render_phase_zero=must_not_run,
            )
            with (
                patch(
                    "src.algorithms.shac.g1_failure_weighted_audit_execution._prepare_e064_execution",
                    return_value=prepared,
                ),
                self.assertRaisesRegex(ValueError, "identity receipts differ"),
            ):
                run_audit(contract)

            self.assertFalse(contract.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
