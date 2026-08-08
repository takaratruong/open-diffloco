import unittest


class ShacExactResumeTest(unittest.TestCase):
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
