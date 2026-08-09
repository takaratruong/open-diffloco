import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path


class G1ShacGradientQualityCliTest(unittest.TestCase):
    def _inputs(self, directory: Path) -> tuple[list[str], Path, Path]:
        checkpoint = directory / "checkpoint_final.pkl"
        checkpoint.write_bytes(b"immutable E064 checkpoint")
        reference = directory / "lafan1_reference.npz"
        reference.write_bytes(b"immutable 500-state reference")
        hparams = {
            "algorithm": "shac",
            "env_variant": "g1_tracking_rmr_50hz_validated",
            "unroll_length": 48,
            "num_envs": 64,
            "gamma": 0.99,
            "action_noise_std_start": 0.1,
            "action_noise_std_end": 0.1,
            "actor_per_env_grad_clip": 1.0,
            "actor_bootstrap_scale": 0.0,
            "squash_actor_actions": False,
            "friction_range": [1.0, 1.0],
            "mass_range": [1.0, 1.0],
            "com_offset_range": [0.0, 0.0, 0.0],
            "push_velocity_range": [0.0, 0.0],
            "terrain": False,
            "reference_reset_noise_scale": 0.0,
            "reference_path": str(reference.resolve()),
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        }
        (directory / "hparams.json").write_text(json.dumps(hparams))
        return (
            [
                "--checkpoint", str(checkpoint),
                "--checkpoint-sha256",
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "--reference", str(reference),
                "--reference-sha256",
                hashlib.sha256(reference.read_bytes()).hexdigest(),
                "--output-dir", str(directory / "audit"),
            ],
            checkpoint,
            reference,
        )

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

    def test_contract_rejects_hash_mismatch_and_nonfrozen_arguments(self):
        from tools.audit_g1_shac_gradient_quality import (
            build_parser,
            validate_audit_contract,
        )

        with tempfile.TemporaryDirectory() as directory:
            argv, _, _ = self._inputs(Path(directory))
            parser = build_parser()
            args = parser.parse_args(argv)
            args.checkpoint_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                validate_audit_contract(args)

            args = parser.parse_args(argv)
            args.horizon = 47
            with self.assertRaisesRegex(ValueError, "horizon"):
                validate_audit_contract(args)

            args = parser.parse_args(argv)
            args.shard_seeds = [0, 1, 2, 4]
            with self.assertRaisesRegex(ValueError, "shard seeds"):
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

            with self.assertRaisesRegex(ValueError, "squash_actor_actions"):
                validate_audit_contract(build_parser().parse_args(argv))

    def test_atomic_helpers_reject_nonfinite_json_and_replace_outputs(self):
        from tools.audit_g1_shac_gradient_quality import (
            assert_finite_json,
            write_json_atomically,
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
            self.assertEqual(
                json.loads(json_path.read_text()), {"finite": [1.0, 2]}
            )
            self.assertFalse(
                (json_path.parent / ".manifest.json.tmp").exists()
            )

            pickle_path = path / "evidence" / "candidate.pkl"
            write_pickle_atomically(pickle_path, {"candidate": "score"})
            with pickle_path.open("rb") as stream:
                self.assertEqual(pickle.load(stream), {"candidate": "score"})
            self.assertFalse(
                (pickle_path.parent / ".candidate.pkl.tmp").exists()
            )

    def test_main_validates_then_calls_injected_future_audit(self):
        from tools.audit_g1_shac_gradient_quality import main

        with tempfile.TemporaryDirectory() as directory:
            argv, checkpoint, reference = self._inputs(Path(directory))
            calls = []

            def future_run_audit(*, contract):
                calls.append(contract)
                return {"status": "wired"}

            result = main(argv, run_audit_impl=future_run_audit)

        self.assertEqual(result, {"status": "wired"})
        self.assertEqual(calls[0].checkpoint, checkpoint.resolve())
        self.assertEqual(calls[0].reference, reference.resolve())


if __name__ == "__main__":
    unittest.main()
