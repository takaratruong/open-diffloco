import unittest


class FailureWeightedAuditCliTest(unittest.TestCase):
    def test_delegates_to_existing_validated_main_with_failure_weighted_execution(self):
        from tools.audit_g1_shac_failure_weighting import main

        calls = []

        def shared_main(argv, *, run_audit_impl):
            calls.append((argv, run_audit_impl))
            return "delegated"

        marker = object()
        result = main(
            ["--output-dir", "evidence"],
            shared_main_impl=shared_main,
            run_audit_impl=marker,
        )

        self.assertEqual(result, "delegated")
        self.assertEqual(calls, [(["--output-dir", "evidence"], marker)])

    def test_invalid_cli_does_not_load_default_execution(self):
        from tools.audit_g1_shac_failure_weighting import main

        loads = []

        def load_execution():
            loads.append(True)
            return lambda *, contract: ("ran", contract)

        with self.assertRaises(SystemExit) as raised:
            main(
                ["--invalid"],
                load_run_audit_impl=load_execution,
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(loads, [])

    def test_default_execution_is_loaded_when_validated_main_delegates(self):
        from tools.audit_g1_shac_failure_weighting import main

        loads = []

        def load_execution():
            loads.append(True)
            return lambda *, contract: ("ran", contract)

        def accepting_shared_main(_argv, *, run_audit_impl):
            return run_audit_impl(contract="validated")

        result = main(
            [],
            shared_main_impl=accepting_shared_main,
            load_run_audit_impl=load_execution,
        )
        self.assertEqual(result, ("ran", "validated"))
        self.assertEqual(loads, [True])


if __name__ == "__main__":
    unittest.main()
