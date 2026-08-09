import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

_VALIDITY_KEYS = (
    "frozen_hashes",
    "weight_receipts_exact",
    "uniform_reproduction_exact",
    "tail_reproduction_exact",
    "stability_evidence_exact",
    "aggregate_gradients_finite_nonzero",
    "candidate_trees_finite_nonzero",
    "functional_steps_valid",
    "rollouts_fresh_replay_free_complete_finite",
)


def valid_documents():
    weighting = []
    gradient_hashes = []
    estimator_shards = {}
    for seed in range(4):
        weighting.append(
            {
                "seed": seed,
                "losses": [float(seed * 100 + index) for index in range(64)],
                "initial_phases": [(index % 5) * 100 for index in range(64)],
            }
        )
        gradient_hashes.append(
            {
                "seed": seed,
                "per_environment_clipped": f"per-env-{seed}",
                "uniform": f"uniform-{seed}",
                "tail": f"tail-{seed}",
            }
        )
        identity = {"identity": f"shard-{seed}"}
        estimator_shards[str(seed)] = {
            "pathwise": identity,
            "score": dict(identity),
        }
    return {
        "outcome.json": {
            "verdict": "failure-aware-unstable",
            "reason": "lower-tail direction regresses a frozen stability gate",
            "decision_metrics": {"stability_checks": {}},
        },
        "validity.json": {key: True for key in _VALIDITY_KEYS},
        "failure_weight_receipts.json": {
            "weighting": weighting,
            "independent_recomputation": {
                "weight_receipts_exact": True,
                "uniform_reproduction_exact": True,
                "tail_reproduction_exact": True,
                "stability_evidence_exact": True,
                "independent_host_recomputation": {},
            },
            "gradient_hashes": {
                "per_shard": gradient_hashes,
                "aggregate": {"uniform": "uniform", "tail": "tail"},
            },
        },
        "estimator_receipts.json": {
            "shared_rollout_identity": True,
            "algorithmic_validity": {"pathwise_matches_score": True},
            "per_shard": estimator_shards,
        },
    }


def write_run_dir(root: Path, documents=None) -> Path:
    run_dir = root / "e011"
    evidence_dir = run_dir / "seed-1" / "evidence"
    evidence_dir.mkdir(parents=True)
    artifacts = {}
    for name, document in (documents or valid_documents()).items():
        path = evidence_dir / name
        path.write_text(json.dumps(document), encoding="utf-8")
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (evidence_dir / "manifest.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return run_dir


class E011SourceReceiptTest(unittest.TestCase):
    def assert_rejected(self, documents, message):
        from src.algorithms.shac.g1_tail_contact_derivative_source import (
            load_e011_source_receipts,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = write_run_dir(Path(temporary), documents)
            with self.assertRaisesRegex(ValueError, message):
                load_e011_source_receipts(run_dir)

    def test_loads_all_receipts_and_extracts_four_shards(self):
        from src.algorithms.shac.g1_tail_contact_derivative_source import (
            load_e011_source_receipts,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = write_run_dir(Path(temporary))

            source = load_e011_source_receipts(run_dir)

        self.assertEqual(source.run_dir, run_dir.resolve())
        self.assertEqual(
            source.evidence_dir, (run_dir / "seed-1" / "evidence").resolve()
        )
        self.assertEqual(source.outcome["verdict"], "failure-aware-unstable")
        self.assertTrue(all(source.validity.values()))
        self.assertEqual(tuple(source.losses_by_shard), (0, 1, 2, 3))
        self.assertEqual(tuple(source.initial_phases_by_shard), (0, 1, 2, 3))
        self.assertEqual(source.losses_by_shard[2][7], 207.0)
        self.assertEqual(source.initial_phases_by_shard[3][7], 200)
        self.assertEqual(
            set(source.estimator_receipts["per_shard"]), {"0", "1", "2", "3"}
        )

    def test_rejects_wrong_outcome_or_incomplete_validity(self):
        cases = []
        wrong_outcome = valid_documents()
        wrong_outcome["outcome.json"]["verdict"] = "failure-aware-supported"
        cases.append((wrong_outcome, "failure-aware-unstable"))
        malformed_outcome = valid_documents()
        malformed_outcome["outcome.json"].pop("decision_metrics")
        cases.append((malformed_outcome, "outcome.*decision_metrics"))
        false_validity = valid_documents()
        false_validity["validity.json"]["weight_receipts_exact"] = False
        cases.append((false_validity, "validity.*weight_receipts_exact"))
        missing_validity = valid_documents()
        missing_validity["validity.json"].pop("functional_steps_valid")
        cases.append((missing_validity, "validity.*key mismatch"))
        extra_validity = valid_documents()
        extra_validity["validity.json"]["unregistered"] = True
        cases.append((extra_validity, "validity.*key mismatch"))

        for documents, message in cases:
            with self.subTest(message=message):
                self.assert_rejected(documents, message)

    def test_rejects_missing_or_malformed_failure_weight_structures(self):
        cases = []
        missing_losses = valid_documents()
        missing_losses["failure_weight_receipts.json"]["weighting"][0].pop("losses")
        cases.append((missing_losses, "weighting.*losses"))
        malformed_losses = valid_documents()
        malformed_losses["failure_weight_receipts.json"]["weighting"][0][
            "losses"
        ] = [0.0] * 63
        cases.append((malformed_losses, "losses.*64"))
        duplicate_weighting_seed = valid_documents()
        duplicate_weighting_seed["failure_weight_receipts.json"]["weighting"][3][
            "seed"
        ] = 2
        cases.append((duplicate_weighting_seed, "weighting.*shard.*0, 1, 2, 3"))
        duplicate_hash_seed = valid_documents()
        duplicate_hash_seed["failure_weight_receipts.json"]["gradient_hashes"][
            "per_shard"
        ][3]["seed"] = 2
        cases.append((duplicate_hash_seed, "gradient hashes.*shard.*0, 1, 2, 3"))
        failed_recomputation = valid_documents()
        failed_recomputation["failure_weight_receipts.json"][
            "independent_recomputation"
        ]["tail_reproduction_exact"] = False
        cases.append((failed_recomputation, "recomputation.*tail_reproduction_exact"))

        for documents, message in cases:
            with self.subTest(message=message):
                self.assert_rejected(documents, message)

    def test_rejects_malformed_or_incomplete_estimator_structures(self):
        cases = []
        wrong_shard_keys = valid_documents()
        per_shard = wrong_shard_keys["estimator_receipts.json"]["per_shard"]
        per_shard["4"] = per_shard.pop("3")
        cases.append((wrong_shard_keys, "estimator.*shard.*0, 1, 2, 3"))
        missing_pathwise = valid_documents()
        missing_pathwise["estimator_receipts.json"]["per_shard"]["0"].pop(
            "pathwise"
        )
        cases.append((missing_pathwise, "estimator shard 0.*pathwise"))
        mismatched_estimators = valid_documents()
        mismatched_estimators["estimator_receipts.json"]["per_shard"]["1"][
            "score"
        ]["identity"] = "different"
        cases.append((mismatched_estimators, "estimator shard 1.*differ"))
        failed_algorithmic_validity = valid_documents()
        failed_algorithmic_validity["estimator_receipts.json"][
            "algorithmic_validity"
        ]["pathwise_matches_score"] = False
        cases.append(
            (failed_algorithmic_validity, "algorithmic validity.*pathwise_matches_score")
        )
        false_shared_identity = valid_documents()
        false_shared_identity["estimator_receipts.json"][
            "shared_rollout_identity"
        ] = False
        cases.append((false_shared_identity, "shared rollout identity"))

        for documents, message in cases:
            with self.subTest(message=message):
                self.assert_rejected(documents, message)

    def test_read_failures_name_the_required_receipt(self):
        from src.algorithms.shac.g1_tail_contact_derivative_source import (
            load_e011_source_receipts,
        )

        for failure in ("missing", "malformed"):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(
                failure=failure
            ):
                documents = deepcopy(valid_documents())
                if failure == "missing":
                    documents.pop("outcome.json")
                run_dir = write_run_dir(Path(temporary), documents)
                if failure == "malformed":
                    evidence_dir = run_dir / "seed-1" / "evidence"
                    (evidence_dir / "outcome.json").write_text("{", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "outcome.json"):
                    load_e011_source_receipts(run_dir)

    def test_rejects_receipt_tampering_against_frozen_manifest(self):
        from src.algorithms.shac.g1_tail_contact_derivative_source import (
            load_e011_source_receipts,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = write_run_dir(Path(temporary))
            outcome_path = run_dir / "seed-1" / "evidence" / "outcome.json"
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome["reason"] = "tampered after the manifest was frozen"
            outcome_path.write_text(json.dumps(outcome), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outcome.json.*SHA-256"):
                load_e011_source_receipts(run_dir)


if __name__ == "__main__":
    unittest.main()
