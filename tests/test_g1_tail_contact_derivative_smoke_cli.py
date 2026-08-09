import argparse
import unittest
from pathlib import Path


class TailContactDerivativeSmokeCliTest(unittest.TestCase):
    @staticmethod
    def _parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        return parser

    def test_validates_all_host_contracts_before_loading_runtime(self):
        from tools.smoke_g1_tail_contact_derivatives import main

        calls = []

        def load_runtime():
            calls.append(("load",))

            def run(contract, e011_run_dir):
                calls.append(("run", contract, e011_run_dir))
                return {"decision": "authorize-forward-shac-method"}

            return run

        result = main(
            ["--output-dir", "smoke", "--e011-run-dir", "source"],
            build_parser_impl=self._parser,
            validate_contract_impl=lambda args: (
                calls.append(("contract", args.output_dir)) or "contract"
            ),
            validate_e011_impl=lambda path: (
                calls.append(("e011", path)) or Path("/verified/e011")
            ),
            load_run_smoke_impl=load_runtime,
        )

        self.assertEqual(result["decision"], "authorize-forward-shac-method")
        self.assertEqual(
            calls,
            [
                ("e011", Path("source")),
                ("contract", Path("smoke")),
                ("load",),
                ("run", "contract", Path("/verified/e011")),
            ],
        )

    def test_invalid_source_or_e012_output_never_loads_runtime(self):
        from tools.smoke_g1_tail_contact_derivatives import main

        cases = (
            (["--output-dir", "smoke", "--e011-run-dir", "bad"], True),
            (
                [
                    "--output-dir",
                    "/tmp/runs/E-20260809-012/forbidden",
                    "--e011-run-dir",
                    "source",
                ],
                False,
            ),
        )
        for argv, reject_source in cases:
            loads = []

            def validate_source(path, reject_source=reject_source):
                if reject_source:
                    raise ValueError("wrong E011 receipt")
                return path

            with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                main(
                    argv,
                    build_parser_impl=self._parser,
                    validate_contract_impl=lambda args: args,
                    validate_e011_impl=validate_source,
                    load_run_smoke_impl=lambda loads=loads: loads.append(True),
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(loads, [])


if __name__ == "__main__":
    unittest.main()
