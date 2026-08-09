import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class G1ShacGradientQualityCliTest(unittest.TestCase):
    def _inputs(self, directory: Path) -> tuple[list[str], Path, Path]:
        from tools.audit_g1_shac_gradient_quality import (
            E064_CHECKPOINT_SHA256,
            E064_REFERENCE_SHA256,
            FROZEN_E064_HPARAMS,
        )

        checkpoint = directory / "checkpoint_final.pkl"
        checkpoint.write_bytes(b"immutable E064 checkpoint")
        reference = directory / "lafan1_reference.npz"
        reference.write_bytes(b"immutable 500-state reference")
        hparams = dict(FROZEN_E064_HPARAMS)
        hparams["reference_path"] = str(reference.resolve())
        (directory / "hparams.json").write_text(json.dumps(hparams))
        return (
            [
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                E064_CHECKPOINT_SHA256,
                "--reference",
                str(reference),
                "--reference-sha256",
                E064_REFERENCE_SHA256,
                "--output-dir",
                str(directory / "audit"),
            ],
            checkpoint,
            reference,
        )

    def _pinned_file_sha(self, path: Path) -> str:
        from tools.audit_g1_shac_gradient_quality import (
            E064_CHECKPOINT_SHA256,
            E064_REFERENCE_SHA256,
        )

        if path.suffix == ".pkl":
            return E064_CHECKPOINT_SHA256
        return E064_REFERENCE_SHA256

    def test_parser_requires_immutable_inputs_and_uses_frozen_defaults(self):
        from tools.audit_g1_shac_gradient_quality import (
            FIXED_HELD_OUT_SEEDS,
            FIXED_PHASES,
            FIXED_SHARD_SEEDS,
            build_parser,
        )

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

        with tempfile.TemporaryDirectory() as directory:
            argv, _, _ = self._inputs(Path(directory))
            args = parser.parse_args(argv)

        self.assertEqual(tuple(args.shard_seeds), FIXED_SHARD_SEEDS)
        self.assertEqual(tuple(args.held_out_seeds), FIXED_HELD_OUT_SEEDS)
        self.assertEqual(tuple(args.phases), FIXED_PHASES)
        self.assertEqual(args.horizon, 48)
        self.assertEqual(args.population, 64)
        self.assertEqual(args.sigma, 0.1)
        self.assertEqual(args.gamma, 0.99)
        self.assertEqual(args.per_env_clip, 1.0)
        self.assertEqual(args.functional_rms, 0.01)
        self.assertEqual(args.solver_iterations, 4)
        self.assertEqual(args.solver_ls_iterations, 5)

    def test_contract_rejects_unpinned_cli_hashes(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        for attribute, label in (
            ("checkpoint_sha256", "checkpoint SHA-256"),
            ("reference_sha256", "reference SHA-256"),
        ):
            with (
                self.subTest(attribute=attribute),
                tempfile.TemporaryDirectory() as directory,
            ):
                argv, _, _ = self._inputs(Path(directory))
                args = build_parser().parse_args(argv)
                setattr(args, attribute, "0" * 64)
                with self.assertRaisesRegex(ValueError, label):
                    validate_audit_contract(args)

    def test_contract_rejects_recomputed_file_hash_mismatch(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            argv, _, _ = self._inputs(Path(directory))
            with self.assertRaisesRegex(ValueError, "checkpoint.*SHA-256"):
                validate_audit_contract(build_parser().parse_args(argv))

    def test_contract_rejects_nonfrozen_arguments(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            argv, _, _ = self._inputs(Path(directory))
            parser = build_parser()
            args = parser.parse_args(argv)
            args.horizon = 47
            with (
                patch(
                    "tools.audit_g1_shac_gradient_quality.sha256_file",
                    side_effect=self._pinned_file_sha,
                ),
                self.assertRaisesRegex(ValueError, "horizon"),
            ):
                validate_audit_contract(args)

            args = parser.parse_args(argv)
            args.shard_seeds = [0, 1, 2, 4]
            with (
                patch(
                    "tools.audit_g1_shac_gradient_quality.sha256_file",
                    side_effect=self._pinned_file_sha,
                ),
                self.assertRaisesRegex(ValueError, "shard seeds"),
            ):
                validate_audit_contract(args)

    def test_contract_rejects_hparams_that_violate_the_frozen_protocol(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            argv, _, _ = self._inputs(path)
            hparams_path = path / "hparams.json"
            hparams = json.loads(hparams_path.read_text())
            hparams["squash_actor_actions"] = True
            hparams_path.write_text(json.dumps(hparams))

            with (
                patch(
                    "tools.audit_g1_shac_gradient_quality.sha256_file",
                    side_effect=self._pinned_file_sha,
                ),
                self.assertRaisesRegex(ValueError, "squash_actor_actions"),
            ):
                validate_audit_contract(build_parser().parse_args(argv))

    def test_contract_rejects_changed_kp_or_kd_range(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        for name in ("kp_range", "kd_range"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                argv, _, _ = self._inputs(path)
                hparams_path = path / "hparams.json"
                hparams = json.loads(hparams_path.read_text())
                hparams[name] = [0.9, 1.1]
                hparams_path.write_text(json.dumps(hparams))

                with (
                    patch(
                        "tools.audit_g1_shac_gradient_quality.sha256_file",
                        side_effect=self._pinned_file_sha,
                    ),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    validate_audit_contract(build_parser().parse_args(argv))

    def test_contract_rejects_extra_hparam_key(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            argv, _, _ = self._inputs(path)
            hparams_path = path / "hparams.json"
            hparams = json.loads(hparams_path.read_text())
            hparams["unregistered_override"] = True
            hparams_path.write_text(json.dumps(hparams))

            with (
                patch(
                    "tools.audit_g1_shac_gradient_quality.sha256_file",
                    side_effect=self._pinned_file_sha,
                ),
                self.assertRaisesRegex(ValueError, "extra.*unregistered_override"),
            ):
                validate_audit_contract(build_parser().parse_args(argv))

    def test_contract_rejects_nonobject_hparams(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            argv, _, _ = self._inputs(path)
            (path / "hparams.json").write_text("[]")

            with (
                patch(
                    "tools.audit_g1_shac_gradient_quality.sha256_file",
                    side_effect=self._pinned_file_sha,
                ),
                self.assertRaisesRegex(TypeError, "JSON object"),
            ):
                validate_audit_contract(build_parser().parse_args(argv))

    def test_contract_rejects_missing_hparam_key(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            argv, _, _ = self._inputs(path)
            hparams_path = path / "hparams.json"
            hparams = json.loads(hparams_path.read_text())
            del hparams["actor_kind"]
            hparams_path.write_text(json.dumps(hparams))

            with (
                patch(
                    "tools.audit_g1_shac_gradient_quality.sha256_file",
                    side_effect=self._pinned_file_sha,
                ),
                self.assertRaisesRegex(ValueError, "missing.*actor_kind"),
            ):
                validate_audit_contract(build_parser().parse_args(argv))

    def test_atomic_helpers_reject_nonfinite_json_and_replace_outputs(self):
        import numpy as np

        from tools.audit_g1_shac_gradient_quality import (
            assert_finite_json,
            write_json_atomically,
            write_npz_atomically,
            write_pickle_atomically,
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            assert_finite_json({"nested": [float("nan")]})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            json_path = path / "evidence" / "manifest.json"
            json_path.parent.mkdir()
            json_path.write_text('{"old": true}\n')
            write_json_atomically(json_path, {"finite": [1.0, 2]})
            self.assertEqual(json.loads(json_path.read_text()), {"finite": [1.0, 2]})
            self.assertFalse((json_path.parent / ".manifest.json.tmp").exists())

            pickle_path = path / "evidence" / "candidate.pkl"
            write_pickle_atomically(pickle_path, {"candidate": "score"})
            with pickle_path.open("rb") as stream:
                self.assertEqual(pickle.load(stream), {"candidate": "score"})
            self.assertFalse((pickle_path.parent / ".candidate.pkl.tmp").exists())

            npz_path = path / "evidence" / "rollouts.npz"
            write_npz_atomically(npz_path, {"rewards": np.array([1.0, 2.0])})
            with np.load(npz_path) as archive:
                np.testing.assert_array_equal(archive["rewards"], [1.0, 2.0])
            self.assertFalse((npz_path.parent / ".rollouts.npz.tmp").exists())

    def test_default_loader_points_to_live_execution_engine(self):
        from src.algorithms.shac.g1_gradient_audit_execution import run_audit
        from tools.audit_g1_shac_gradient_quality import _load_future_run_audit

        self.assertIs(_load_future_run_audit(), run_audit)

    def test_main_validates_then_calls_injected_future_audit(self):
        from tools.audit_g1_shac_gradient_quality import main

        with tempfile.TemporaryDirectory() as directory:
            argv, checkpoint, reference = self._inputs(Path(directory))
            calls = []

            def future_run_audit(*, contract):
                calls.append(contract)
                return {"status": "wired"}

            with patch(
                "tools.audit_g1_shac_gradient_quality.sha256_file",
                side_effect=self._pinned_file_sha,
            ):
                result = main(argv, run_audit_impl=future_run_audit)

        self.assertEqual(result, {"status": "wired"})
        self.assertEqual(calls[0].checkpoint, checkpoint.resolve())
        self.assertEqual(calls[0].reference, reference.resolve())


if __name__ == "__main__":
    unittest.main()
