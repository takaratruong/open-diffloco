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
        self.assertEqual(
            parameters["actor_reference_preview_mode"].default, "absolute"
        )
        self.assertIs(
            parameters[
                "allow_resume_actor_reference_lookahead_upgrade"
            ].default,
            False,
        )
        self.assertIs(parameters["actor_preview_adapter"].default, False)
        self.assertIs(
            parameters["actor_residual_preview_adapter"].default, False
        )
        self.assertEqual(
            parameters["actor_residual_preview_hidden"].default, 256
        )
        self.assertEqual(
            parameters["actor_residual_preview_optimizer"].default, "adam"
        )

    def test_residual_preview_can_only_start_during_explicit_legacy_upgrade(self):
        from src.algorithms.shac.algorithm import (
            resolve_residual_preview_adapter_resume_setting,
        )

        self.assertEqual(
            resolve_residual_preview_adapter_resume_setting(
                {},
                requested=True,
                requested_hidden=256,
                requested_optimizer="muon",
                future_reference_upgrade=True,
            ),
            (True, 256, "muon"),
        )
        with self.assertRaisesRegex(ValueError, "future-reference upgrade"):
            resolve_residual_preview_adapter_resume_setting(
                {},
                requested=True,
                requested_hidden=256,
                requested_optimizer="muon",
                future_reference_upgrade=False,
            )

    def test_residual_preview_resume_requires_exact_saved_flag_and_width(self):
        from src.algorithms.shac.algorithm import (
            resolve_residual_preview_adapter_resume_setting,
        )

        metadata = {
            "actor_residual_preview_adapter": True,
            "actor_residual_preview_hidden": 256,
            "actor_residual_preview_optimizer": "muon",
        }
        self.assertEqual(
            resolve_residual_preview_adapter_resume_setting(
                metadata,
                requested=True,
                requested_hidden=256,
                requested_optimizer="muon",
                future_reference_upgrade=False,
            ),
            (True, 256, "muon"),
        )
        for requested, hidden, optimizer in (
            (False, 256, "muon"),
            (True, 128, "muon"),
            (True, 256, "adam"),
        ):
            with self.subTest(
                requested=requested,
                hidden=hidden,
                optimizer=optimizer,
            ):
                with self.assertRaisesRegex(ValueError, "must match"):
                    resolve_residual_preview_adapter_resume_setting(
                        metadata,
                        requested=requested,
                        requested_hidden=hidden,
                        requested_optimizer=optimizer,
                        future_reference_upgrade=False,
                    )

        self.assertEqual(
            resolve_residual_preview_adapter_resume_setting(
                {
                    "actor_residual_preview_adapter": True,
                    "actor_residual_preview_hidden": 256,
                },
                requested=True,
                requested_hidden=256,
                requested_optimizer="adam",
                future_reference_upgrade=False,
            ),
            (True, 256, "adam"),
        )

    def test_residual_preview_resume_rejects_invalid_metadata(self):
        from src.algorithms.shac.algorithm import (
            resolve_residual_preview_adapter_resume_setting,
        )

        with self.assertRaisesRegex(ValueError, "boolean"):
            resolve_residual_preview_adapter_resume_setting(
                {"actor_residual_preview_adapter": "yes"},
                requested=True,
                requested_hidden=256,
                requested_optimizer="adam",
                future_reference_upgrade=False,
            )
        with self.assertRaisesRegex(ValueError, "optimizer"):
            resolve_residual_preview_adapter_resume_setting(
                {},
                requested=True,
                requested_hidden=256,
                requested_optimizer="sgd",
                future_reference_upgrade=True,
            )

    def test_delta_preview_mode_requires_upgrade_or_exact_resume(self):
        from src.algorithms.shac.algorithm import (
            resolve_future_reference_preview_mode,
        )

        self.assertEqual(
            resolve_future_reference_preview_mode(
                {"actor_reference_lookahead_steps": []},
                requested_mode="delta",
                future_reference_upgrade=True,
            ),
            "delta",
        )
        self.assertEqual(
            resolve_future_reference_preview_mode(
                {
                    "actor_reference_lookahead_steps": [4, 8, 12],
                    "actor_reference_preview_mode": "delta",
                },
                requested_mode="delta",
                future_reference_upgrade=False,
            ),
            "delta",
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            resolve_future_reference_preview_mode(
                {"actor_reference_lookahead_steps": [4, 8, 12]},
                requested_mode="delta",
                future_reference_upgrade=False,
            )

    def test_preview_mode_defaults_legacy_preview_to_absolute(self):
        from src.algorithms.shac.algorithm import (
            resolve_future_reference_preview_mode,
        )

        self.assertEqual(
            resolve_future_reference_preview_mode(
                {"actor_reference_lookahead_steps": [4, 8, 12]},
                requested_mode="absolute",
                future_reference_upgrade=False,
            ),
            "absolute",
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            resolve_future_reference_preview_mode(
                {
                    "actor_reference_lookahead_steps": [4],
                    "actor_reference_preview_mode": "relative",
                },
                requested_mode="relative",
                future_reference_upgrade=False,
            )

    def test_preview_adapter_resume_is_explicit_and_exact(self):
        from src.algorithms.shac.algorithm import (
            resolve_preview_adapter_resume_setting,
        )

        self.assertTrue(
            resolve_preview_adapter_resume_setting({}, requested=True)
        )
        self.assertTrue(
            resolve_preview_adapter_resume_setting(
                {"actor_preview_adapter": True}, requested=True
            )
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            resolve_preview_adapter_resume_setting(
                {"actor_preview_adapter": True}, requested=False
            )

    def test_preview_adapter_resume_rejects_invalid_metadata(self):
        from src.algorithms.shac.algorithm import (
            resolve_preview_adapter_resume_setting,
        )

        with self.assertRaisesRegex(ValueError, "boolean"):
            resolve_preview_adapter_resume_setting(
                {"actor_preview_adapter": "yes"}, requested=True
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
