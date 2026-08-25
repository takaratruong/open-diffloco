import inspect
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import dataclass, replace

import jax.numpy as jnp


@dataclass(frozen=True)
class _EnvState:
    qpos: object
    metrics: dict

    def replace(self, **changes):
        return replace(self, **changes)


class ShacExactResumeTest(unittest.TestCase):
    def test_torso_orientation_weight_resume_is_explicit_and_fail_closed(self):
        from src.algorithms.shac.algorithm import (
            resolve_tracking_torso_orientation_resume_weight,
        )

        self.assertEqual(
            resolve_tracking_torso_orientation_resume_weight(
                None,
                requested=1.0,
                allow_change=False,
                is_resume=False,
            ),
            1.0,
        )
        self.assertEqual(
            resolve_tracking_torso_orientation_resume_weight(
                {},
                requested=0.0,
                allow_change=False,
                is_resume=True,
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "hparams"):
            resolve_tracking_torso_orientation_resume_weight(
                None,
                requested=0.0,
                allow_change=False,
                is_resume=True,
            )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_tracking_torso_orientation_resume_weight(
                {"tracking_torso_orientation_weight": 0.0},
                requested=1.0,
                allow_change=False,
                is_resume=True,
            )
        self.assertEqual(
            resolve_tracking_torso_orientation_resume_weight(
                {"tracking_torso_orientation_weight": 0.0},
                requested=1.0,
                allow_change=True,
                is_resume=True,
            ),
            1.0,
        )
        for invalid in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "tracking_torso_orientation_weight"
                ):
                    resolve_tracking_torso_orientation_resume_weight(
                        {},
                        requested=invalid,
                        allow_change=False,
                        is_resume=False,
                    )

    def test_torso_orientation_train_settings_are_default_off(self):
        from src.algorithms.shac.algorithm import train

        parameters = inspect.signature(train).parameters

        self.assertEqual(
            parameters["tracking_torso_orientation_weight"].default, 0.0
        )
        self.assertIs(
            parameters[
                "allow_resume_tracking_torso_orientation_change"
            ].default,
            False,
        )

    def test_tracking_velocity_kernel_resume_is_explicit_and_fail_closed(self):
        from src.algorithms.shac.algorithm import (
            resolve_tracking_velocity_kernel_resume_setting,
        )

        self.assertEqual(
            resolve_tracking_velocity_kernel_resume_setting(
                None,
                requested="pseudo_huber",
                allow_change=False,
                is_resume=False,
            ),
            "pseudo_huber",
        )
        self.assertEqual(
            resolve_tracking_velocity_kernel_resume_setting(
                {},
                requested="exponential",
                allow_change=False,
                is_resume=True,
            ),
            "exponential",
        )
        with self.assertRaisesRegex(ValueError, "hparams"):
            resolve_tracking_velocity_kernel_resume_setting(
                None,
                requested="exponential",
                allow_change=False,
                is_resume=True,
            )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_tracking_velocity_kernel_resume_setting(
                {"tracking_velocity_kernel": "exponential"},
                requested="pseudo_huber",
                allow_change=False,
                is_resume=True,
            )
        self.assertEqual(
            resolve_tracking_velocity_kernel_resume_setting(
                {"tracking_velocity_kernel": "exponential"},
                requested="pseudo_huber",
                allow_change=True,
                is_resume=True,
            ),
            "pseudo_huber",
        )
        for invalid in ("unknown", 1, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "velocity kernel"):
                    resolve_tracking_velocity_kernel_resume_setting(
                        {"tracking_velocity_kernel": invalid},
                        requested="exponential",
                        allow_change=False,
                        is_resume=True,
                    )

    def test_tracking_velocity_kernel_train_settings_preserve_legacy_default(self):
        from src.algorithms.shac.algorithm import train

        parameters = inspect.signature(train).parameters

        self.assertEqual(
            parameters["tracking_velocity_kernel"].default, "exponential"
        )
        self.assertIs(
            parameters[
                "allow_resume_tracking_velocity_kernel_change"
            ].default,
            False,
        )

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
        self.assertIs(
            parameters[
                "allow_resume_actor_residual_preview_adapter_start"
            ].default,
            False,
        )
        self.assertIs(
            parameters["allow_resume_reference_path_change"].default,
            False,
        )

    def test_motion_anchor_position_observation_resume_is_fail_closed(self):
        from src.algorithms.shac.algorithm import (
            resolve_actor_observe_motion_anchor_position_resume_setting,
        )

        assert (
            resolve_actor_observe_motion_anchor_position_resume_setting(
                None, requested=False
            )
            is False
        )
        assert (
            resolve_actor_observe_motion_anchor_position_resume_setting(
                {"actor_observe_motion_anchor_position": True},
                requested=True,
            )
            is True
        )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_actor_observe_motion_anchor_position_resume_setting(
                None, requested=True
            )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_actor_observe_motion_anchor_position_resume_setting(
                {"actor_observe_motion_anchor_position": False},
                requested=True,
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            resolve_actor_observe_motion_anchor_position_resume_setting(
                {"actor_observe_motion_anchor_position": 1},
                requested=False,
            )

    def test_expected_actor_input_guard_accepts_only_the_3310_contract(self):
        from src.core.actor_input_contract import validate_actor_input_contract

        report = validate_actor_input_contract(
            expected_input_dim=3310,
            environment_input_dim=3310,
            first_layer_input_dim=3310,
        )
        self.assertTrue(report["valid"])
        with self.assertRaises(ValueError):
            validate_actor_input_contract(
                expected_input_dim=3310,
                environment_input_dim=3280,
                first_layer_input_dim=3310,
            )

    def test_reference_path_change_requires_explicit_resume_authority(self):
        from src.algorithms.shac.algorithm import (
            resolve_reference_path_resume_setting,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "saved.npz"
            requested = root / "requested.npz"
            saved.write_bytes(b"saved")
            requested.write_bytes(b"requested")
            metadata = {"reference_path": str(saved)}

            with self.assertRaisesRegex(ValueError, "explicit authority"):
                resolve_reference_path_resume_setting(
                    metadata,
                    requested_path=str(requested),
                    allow_change=False,
                    is_resume=True,
                )
            resolved, report = resolve_reference_path_resume_setting(
                metadata,
                requested_path=str(requested),
                allow_change=True,
                is_resume=True,
            )

        self.assertEqual(resolved, str(requested.resolve()))
        self.assertIsNotNone(report)
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["previous_reference_path"], str(saved.resolve())
        )
        self.assertEqual(
            report["requested_reference_path"], str(requested.resolve())
        )
        self.assertTrue(report["environment_state_reinitialized"])

    def test_reference_path_resume_fails_closed_without_metadata(self):
        from src.algorithms.shac.algorithm import (
            resolve_reference_path_resume_setting,
        )

        with self.assertRaisesRegex(ValueError, "metadata"):
            resolve_reference_path_resume_setting(
                None,
                requested_path="/tmp/new.npz",
                allow_change=True,
                is_resume=True,
            )

    def test_reference_path_exact_resume_and_fresh_run_do_not_migrate(self):
        from src.algorithms.shac.algorithm import (
            resolve_reference_path_resume_setting,
        )

        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.npz"
            reference.write_bytes(b"reference")
            exact = resolve_reference_path_resume_setting(
                {"reference_path": str(reference)},
                requested_path=None,
                allow_change=False,
                is_resume=True,
            )
            fresh = resolve_reference_path_resume_setting(
                None,
                requested_path=str(reference),
                allow_change=False,
                is_resume=False,
            )

        self.assertEqual(exact, (str(reference.resolve()), None))
        self.assertEqual(fresh, (str(reference), None))

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

    def test_residual_preview_plain_actor_start_requires_explicit_authority(self):
        from src.algorithms.shac.algorithm import (
            resolve_residual_preview_adapter_resume_setting,
        )

        metadata = {
            "actor_residual_preview_adapter": False,
            "actor_residual_preview_hidden": 256,
            "actor_residual_preview_optimizer": "adam",
        }
        with self.assertRaisesRegex(ValueError, "explicit start authority"):
            resolve_residual_preview_adapter_resume_setting(
                metadata,
                requested=True,
                requested_hidden=256,
                requested_optimizer="adam",
                future_reference_upgrade=False,
                allow_start=False,
            )
        self.assertEqual(
            resolve_residual_preview_adapter_resume_setting(
                metadata,
                requested=True,
                requested_hidden=256,
                requested_optimizer="adam",
                future_reference_upgrade=False,
                allow_start=True,
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

    def test_carried_reset_change_requires_explicit_resume_treatment(self):
        from src.algorithms.shac.algorithm import (
            resolve_carried_reset_resume_settings,
        )

        resumed = {
            "carried_reset_bank_path": "/tmp/old-bank.npz",
            "carried_reset_probability": 0.25,
            "carried_reset_bank_start": 7,
        }
        self.assertEqual(
            resolve_carried_reset_resume_settings(
                resumed,
                requested_bank_path=None,
                requested_probability=0.0,
                requested_start=0,
                allow_change=False,
            ),
            ("/tmp/old-bank.npz", 0.25, 7),
        )
        with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
            resolve_carried_reset_resume_settings(
                resumed,
                requested_bank_path="/tmp/new-bank.npz",
                requested_probability=0.5,
                requested_start=0,
                allow_change=False,
            )
        self.assertEqual(
            resolve_carried_reset_resume_settings(
                resumed,
                requested_bank_path="/tmp/new-bank.npz",
                requested_probability=0.5,
                requested_start=0,
                allow_change=True,
            ),
            ("/tmp/new-bank.npz", 0.5, 0),
        )
        self.assertEqual(
            resolve_carried_reset_resume_settings(
                resumed,
                requested_bank_path=None,
                requested_probability=0.0,
                requested_start=0,
                allow_change=True,
            ),
            (None, 0.0, 0),
        )

    def test_fresh_carried_reset_settings_need_no_resume_override(self):
        from src.algorithms.shac.algorithm import (
            resolve_carried_reset_resume_settings,
        )

        self.assertEqual(
            resolve_carried_reset_resume_settings(
                None,
                requested_bank_path="/tmp/bank.npz",
                requested_probability=0.5,
                requested_start=0,
                allow_change=False,
            ),
            ("/tmp/bank.npz", 0.5, 0),
        )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            resolve_carried_reset_resume_settings(
                None,
                requested_bank_path=None,
                requested_probability=0.0,
                requested_start=0,
                allow_change=1,
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

    def test_resume_migrates_environment_metric_schema_only(self):
        from src.algorithms.shac.algorithm import migrate_env_state_metrics

        qpos = jnp.asarray([[1.0, 2.0]])
        resumed = _EnvState(
            qpos=qpos,
            metrics={
                "rew_action_rate": jnp.asarray([3.0]),
                "legacy_removed_metric": jnp.asarray([9.0]),
            },
        )
        initialized = _EnvState(
            qpos=jnp.zeros_like(qpos),
            metrics={
                "rew_action_rate": jnp.asarray([0.0]),
                "rew_action_magnitude": jnp.asarray([0.0]),
            },
        )

        migrated = migrate_env_state_metrics(resumed, initialized)

        self.assertIs(migrated.qpos, qpos)
        self.assertEqual(
            set(migrated.metrics),
            {"rew_action_rate", "rew_action_magnitude"},
        )
        self.assertEqual(float(migrated.metrics["rew_action_rate"][0]), 3.0)
        self.assertEqual(
            float(migrated.metrics["rew_action_magnitude"][0]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
