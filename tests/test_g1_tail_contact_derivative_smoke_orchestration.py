import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class FakeComparison:
    forward_valid: bool = True
    reverse_parity_valid: bool = False


@dataclass(frozen=True)
class FakeSmoke:
    comparison: FakeComparison
    probes_preserve_done_and_support: bool = True
    execution_valid: bool = True
    reverse_compile_duration_seconds: float = 2.0
    forward_compile_duration_seconds: float = 3.0
    reverse_cached_duration_seconds: float = 4.0
    forward_cached_sweep_durations_seconds: tuple[float, ...] = (5.0, 6.0, 7.0)
    probe_durations_seconds: tuple[float, ...] = (8.0, 9.0)


def _source(losses, phases, identity):
    return SimpleNamespace(
        run_dir=Path("/frozen/e011"),
        evidence_dir=Path("/frozen/e011/seed-1/evidence"),
        outcome={"verdict": "failure-aware-unstable"},
        validity={"valid": True},
        failure_weight_receipts={},
        estimator_receipts={
            "per_shard": {"0": {"pathwise": identity, "score": identity}}
        },
        losses_by_shard={0: tuple(losses)},
        initial_phases_by_shard={0: tuple(phases)},
    )


class OneCaseSmokeOrchestrationTest(unittest.TestCase):
    def test_replays_only_shard_zero_selects_bin_zero_and_writes_one_receipt(self):
        from src.algorithms.shac.g1_tail_contact_derivative_smoke import (
            run_one_case_smoke,
        )

        losses = np.linspace(-3.0, -1.0, 64, dtype=np.float64)
        phases = np.arange(64, dtype=np.int64) % 100
        losses[5] = 10.0
        losses[7] = 10.0  # exact tie must keep lower environment index
        identity = {"trajectory": "same"}
        source = _source(losses, phases, identity)
        trajectory = SimpleNamespace(initial_phase=phases)
        result = SimpleNamespace(losses=losses, trajectory=trajectory)
        calls = []

        @contextmanager
        def solver_context():
            calls.append("solver-enter")
            yield
            calls.append("solver-exit")

        def estimate_shard(seed):
            calls.append(("estimate", seed))
            return SimpleNamespace(
                result=result,
                pathwise_receipt=identity,
                score_receipt=dict(identity),
            )

        def prepare_objective(noise, env_index, *, expected_shared_trajectory):
            self.assertIs(expected_shared_trajectory, trajectory)
            self.assertEqual(env_index, 5)
            calls.append(("prepare-objective", env_index))
            return SimpleNamespace(
                nominal_first_action=np.zeros(29), nominal_objective=np.asarray(0.0)
            )

        prepared = SimpleNamespace(
            estimate_shard=estimate_shard,
            prepare_first_action_objective=prepare_objective,
            gradient_solver_context=solver_context,
            runtime_provenance={"git_clean": True},
            external_inputs={"xml": "bound"},
        )
        smoke = FakeSmoke(comparison=FakeComparison())

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "smoke"
            contract = SimpleNamespace(
                output_dir=output_dir,
                checkpoint=Path("/checkpoint.pkl"),
                checkpoint_sha256="checkpoint-sha",
                reference=Path("/reference.npz"),
                reference_sha256="reference-sha",
            )
            receipt = run_one_case_smoke(
                contract,
                Path("/frozen/e011"),
                load_source_receipts_impl=lambda _: source,
                prepare_e064_execution_impl=lambda _: prepared,
                make_action_noise_impl=lambda seed: ("noise", seed),
                compile_case_kernels_impl=lambda diagnostic: (
                    calls.append(("compile", "diagnostic")) or "compiled"
                ),
                run_compiled_case_smoke_impl=lambda compiled: (
                    calls.append(("run", compiled)) or smoke
                ),
                memory_snapshot_impl=lambda: {"available": False},
                source_hashes_impl=lambda _: {"manifest.json": "manifest-sha"},
            )

            files = tuple(path.name for path in output_dir.iterdir())
            on_disk = json.loads((output_dir / "smoke_receipt.json").read_text())

        self.assertEqual(files, ("smoke_receipt.json",))
        self.assertEqual(receipt, on_disk)
        self.assertEqual(receipt["decision"], "authorize-forward-shac-method")
        self.assertEqual(receipt["selected_case"]["environment_index"], 5)
        self.assertTrue(receipt["authoritative_binding"]["all_exact"])
        self.assertEqual(calls.count(("estimate", 0)), 1)
        self.assertNotIn(("estimate", 1), calls)
        self.assertIn(("prepare-objective", 5), calls)
        self.assertEqual(calls.count("solver-enter"), 2)
        self.assertEqual(calls.count("solver-exit"), 2)
        self.assertLess(calls.index("solver-enter"), calls.index(("estimate", 0)))
        self.assertLess(calls.index(("prepare-objective", 5)), calls.index(("compile", "diagnostic")))

    def test_abandons_on_forward_failure_but_still_publishes_classification(self):
        from src.algorithms.shac.g1_tail_contact_derivative_smoke import (
            run_one_case_smoke,
        )

        losses = np.arange(64, dtype=np.float64)
        phases = np.arange(64, dtype=np.int64) % 100
        identity = {"trajectory": "same"}
        trajectory = SimpleNamespace(initial_phase=phases)
        evidence = SimpleNamespace(
            result=SimpleNamespace(losses=losses, trajectory=trajectory),
            pathwise_receipt=identity,
            score_receipt=identity,
        )
        prepared = SimpleNamespace(
            estimate_shard=lambda _: evidence,
            prepare_first_action_objective=lambda *args, **kwargs: SimpleNamespace(
                nominal_first_action=np.zeros(29), nominal_objective=np.asarray(0.0)
            ),
            gradient_solver_context=lambda: _null_context(),
            runtime_provenance={},
            external_inputs={},
        )
        failed = FakeSmoke(comparison=FakeComparison(forward_valid=False))

        with tempfile.TemporaryDirectory() as temporary:
            contract = SimpleNamespace(
                output_dir=Path(temporary) / "smoke",
                checkpoint=Path("/checkpoint.pkl"),
                checkpoint_sha256="checkpoint-sha",
                reference=Path("/reference.npz"),
                reference_sha256="reference-sha",
            )
            receipt = run_one_case_smoke(
                contract,
                Path("/frozen/e011"),
                load_source_receipts_impl=lambda _: _source(losses, phases, identity),
                prepare_e064_execution_impl=lambda _: prepared,
                make_action_noise_impl=lambda _: object(),
                compile_case_kernels_impl=lambda _: object(),
                run_compiled_case_smoke_impl=lambda _: failed,
                memory_snapshot_impl=lambda: {"available": False},
                source_hashes_impl=lambda _: {},
            )

        self.assertEqual(receipt["decision"], "abandon-forward-shac-mechanism")
        self.assertFalse(receipt["decision_gates"]["forward_valid"])

    def test_rejects_e012_output_and_existing_receipt_before_live_work(self):
        from src.algorithms.shac.g1_tail_contact_derivative_smoke import (
            run_one_case_smoke,
        )

        live_calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                root / "runs" / "E-20260809-012" / "forbidden",
                root / "existing",
            )
            cases[1].mkdir()
            (cases[1] / "smoke_receipt.json").write_text("{}")
            for output_dir in cases:
                with self.subTest(output_dir=output_dir), self.assertRaises(
                    (ValueError, FileExistsError)
                ):
                    run_one_case_smoke(
                        SimpleNamespace(output_dir=output_dir),
                        Path("/frozen/e011"),
                        load_source_receipts_impl=lambda _: live_calls.append("load"),
                    )
        self.assertEqual(live_calls, [])

    def test_rejects_mismatched_authoritative_replay(self):
        from src.algorithms.shac.g1_tail_contact_derivative_smoke import (
            run_one_case_smoke,
        )

        losses = np.arange(64, dtype=np.float64)
        phases = np.arange(64, dtype=np.int64) % 100
        source = _source(losses, phases, {"identity": "source"})
        evidence = SimpleNamespace(
            result=SimpleNamespace(losses=losses, trajectory=SimpleNamespace(initial_phase=phases)),
            pathwise_receipt={"identity": "different"},
            score_receipt={"identity": "different"},
        )
        prepared = SimpleNamespace(
            estimate_shard=lambda _: evidence,
            gradient_solver_context=lambda: _null_context(),
        )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ValueError, "estimator.*source"
        ):
            run_one_case_smoke(
                SimpleNamespace(output_dir=Path(temporary) / "smoke"),
                Path("/frozen/e011"),
                load_source_receipts_impl=lambda _: source,
                prepare_e064_execution_impl=lambda _: prepared,
                make_action_noise_impl=lambda _: object(),
            )


@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    unittest.main()
