import inspect
import json
from pathlib import Path
import tempfile
import unittest


class ShacExactResumeTest(unittest.TestCase):
    def test_future_reference_train_settings_are_default_off(self):
        from src.algorithms.shac.algorithm import train

        parameters = inspect.signature(train).parameters

        self.assertEqual(
            parameters["actor_reference_lookahead_steps"].default, ()
        )
        self.assertIs(
            parameters[
                "allow_resume_actor_reference_lookahead_upgrade"
            ].default,
            False,
        )

    def test_migration_report_is_persisted_as_sorted_json(self):
        from src.algorithms.shac.algorithm import (
            persist_future_reference_migration_report,
        )

        report = {
            "valid": True,
            "max_action_absolute_error": 0.0,
            "max_action_relative_error": 0.0,
            "z": 2,
            "a": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = persist_future_reference_migration_report(
                directory, report
            )
            content = Path(path).read_text(encoding="utf-8")

        self.assertEqual(path.name, "migration_equivalence.json")
        self.assertEqual(json.loads(content), report)
        self.assertLess(content.index('"a"'), content.index('"z"'))

    def test_future_reference_upgrade_requires_explicit_resume_authority(self):
        from src.algorithms.shac.algorithm import (
            resolve_future_reference_resume_settings,
        )

        self.assertEqual(
            resolve_future_reference_resume_settings(
                {"actor_reference_lookahead_steps": []},
                requested_steps=(4, 8, 12),
                allow_upgrade=True,
            ),
            ((4, 8, 12), True),
        )
        with self.assertRaisesRegex(ValueError, "explicit upgrade authority"):
            resolve_future_reference_resume_settings(
                {"actor_reference_lookahead_steps": []},
                requested_steps=(4, 8, 12),
                allow_upgrade=False,
            )

    def test_future_reference_exact_resume_does_not_migrate_twice(self):
        from src.algorithms.shac.algorithm import (
            resolve_future_reference_resume_settings,
        )

        self.assertEqual(
            resolve_future_reference_resume_settings(
                {"actor_reference_lookahead_steps": [4, 8, 12]},
                requested_steps=(4, 8, 12),
                allow_upgrade=False,
            ),
            ((4, 8, 12), False),
        )
        self.assertEqual(
            resolve_future_reference_resume_settings(
                {}, requested_steps=(), allow_upgrade=False
            ),
            ((), False),
        )

    def test_future_reference_resume_rejects_changed_or_removed_layout(self):
        from src.algorithms.shac.algorithm import (
            resolve_future_reference_resume_settings,
        )

        for requested in ((), (4, 12), (2, 4, 8)):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, "must match"):
                    resolve_future_reference_resume_settings(
                        {"actor_reference_lookahead_steps": [4, 8, 12]},
                        requested_steps=requested,
                        allow_upgrade=True,
                    )

    def test_fresh_future_reference_run_needs_no_migration(self):
        from src.algorithms.shac.algorithm import (
            resolve_future_reference_resume_settings,
        )

        self.assertEqual(
            resolve_future_reference_resume_settings(
                None,
                requested_steps=(4, 8, 12),
                allow_upgrade=False,
            ),
            ((4, 8, 12), False),
        )

    def test_margin_change_requires_explicit_resume_treatment(self):
        from src.algorithms.shac.algorithm import (
            validate_termination_margin_resume,
        )

        resumed = {"termination_margin_weight": 0.0}
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            validate_termination_margin_resume(
                resumed,
                requested_weight=0.5,
                allow_change=False,
            )
        validate_termination_margin_resume(
            resumed,
            requested_weight=0.5,
            allow_change=True,
        )

    def test_legacy_checkpoint_keeps_original_noise_schedule_endpoint(self):
        from src.algorithms.shac.algorithm import (
            resolve_action_noise_schedule_steps,
        )

        schedule_steps = resolve_action_noise_schedule_steps(
            total_steps=589_824,
            resumed_step=393_216,
            resumed_hparams={"total_steps": 393_216},
        )

        self.assertEqual(schedule_steps, 393_216)

    def test_fresh_run_uses_requested_total_for_noise_schedule(self):
        from src.algorithms.shac.algorithm import (
            resolve_action_noise_schedule_steps,
        )

        self.assertEqual(
            resolve_action_noise_schedule_steps(
                total_steps=393_216,
                resumed_step=0,
                resumed_hparams=None,
            ),
            393_216,
        )

    def test_resume_rejects_target_before_checkpoint(self):
        from src.algorithms.shac.algorithm import (
            resolve_action_noise_schedule_steps,
        )

        with self.assertRaisesRegex(ValueError, "total_steps"):
            resolve_action_noise_schedule_steps(
                total_steps=300_000,
                resumed_step=393_216,
                resumed_hparams={"total_steps": 393_216},
            )

    def test_resumed_training_reuses_complete_state_object(self):
        from src.algorithms.shac.algorithm import select_initial_training_state

        initialized = object()
        resumed = object()

        self.assertIs(
            select_initial_training_state(
                initialized_state=initialized,
                resumed_state=resumed,
            ),
            resumed,
        )
        self.assertIs(
            select_initial_training_state(
                initialized_state=initialized,
                resumed_state=None,
            ),
            initialized,
        )


if __name__ == "__main__":
    unittest.main()
