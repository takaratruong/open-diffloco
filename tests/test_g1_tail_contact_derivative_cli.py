import argparse
import unittest
from pathlib import Path


class TailContactDerivativeCliTest(unittest.TestCase):
    @staticmethod
    def _parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        return parser

    def test_validates_source_before_lazy_loading_and_delegates(self):
        from tools.audit_g1_tail_contact_derivatives import main

        calls = []

        def validate_contract(args):
            calls.append(("contract", args.output_dir))
            return "frozen-contract"

        def validate_e011(path):
            calls.append(("e011", path))
            return Path("/verified/e011")

        def load_execution():
            calls.append(("load",))

            def run(*, contract, e011_run_dir):
                calls.append(("run", contract, e011_run_dir))
                return "complete"

            return run

        result = main(
            [
                "--output-dir",
                "evidence",
                "--e011-run-dir",
                "source",
            ],
            build_parser_impl=self._parser,
            validate_contract_impl=validate_contract,
            validate_e011_impl=validate_e011,
            load_run_audit_impl=load_execution,
        )

        self.assertEqual(result, "complete")
        self.assertEqual(
            calls,
            [
                ("e011", Path("source")),
                ("contract", Path("evidence")),
                ("load",),
                ("run", "frozen-contract", Path("/verified/e011")),
            ],
        )

    def test_invalid_source_does_not_load_execution(self):
        from tools.audit_g1_tail_contact_derivatives import main

        loads = []

        def reject_e011(_path):
            raise ValueError("wrong E011 receipt")

        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "--output-dir",
                    "evidence",
                    "--e011-run-dir",
                    "source",
                ],
                build_parser_impl=self._parser,
                validate_contract_impl=lambda _args: "unused",
                validate_e011_impl=reject_e011,
                load_run_audit_impl=lambda: loads.append(True),
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(loads, [])

    def test_e011_validation_binds_every_frozen_file(self):
        from tools.audit_g1_tail_contact_derivatives import (
            E011_FROZEN_FILE_SHA256,
            validate_e011_run_dir,
        )

        calls = []
        run_dir = Path("/frozen/e011")

        def hash_file(path):
            calls.append(path)
            return E011_FROZEN_FILE_SHA256[path.relative_to(run_dir)]

        source = validate_e011_run_dir(
            run_dir,
            is_dir=lambda path: path == run_dir,
            is_file=lambda path: path.relative_to(run_dir) in E011_FROZEN_FILE_SHA256,
            sha256_file_impl=hash_file,
        )

        self.assertEqual(source, run_dir)
        self.assertEqual(
            calls,
            [run_dir / relative for relative in E011_FROZEN_FILE_SHA256],
        )

    def test_e011_validation_rejects_each_changed_frozen_file(self):
        from tools.audit_g1_tail_contact_derivatives import (
            E011_FROZEN_FILE_SHA256,
            validate_e011_run_dir,
        )

        run_dir = Path("/frozen/e011")
        for changed in E011_FROZEN_FILE_SHA256:

            def hash_file(path, changed=changed):
                relative = path.relative_to(run_dir)
                if relative == changed:
                    return "0" * 64
                return E011_FROZEN_FILE_SHA256[relative]

            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, changed.as_posix()),
            ):
                validate_e011_run_dir(
                    run_dir,
                    is_dir=lambda _path: True,
                    is_file=lambda _path: True,
                    sha256_file_impl=hash_file,
                )

    def test_legacy_source_evidence_flag_is_not_supported(self):
        from tools.audit_g1_tail_contact_derivatives import main

        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "--output-dir",
                    "evidence",
                    "--source-evidence-dir",
                    "source",
                ],
                build_parser_impl=self._parser,
                validate_contract_impl=lambda _args: "unused",
                validate_e011_impl=lambda path: path,
                load_run_audit_impl=lambda: self.fail("must not load"),
            )

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
