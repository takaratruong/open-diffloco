import unittest


class G1PairedCheckpointComparisonTest(unittest.TestCase):
    def test_classifies_material_recovery_gain_against_control(self):
        from tools.compare_g1_checkpoint_grids import classify_pair

        self.assertEqual(
            classify_pair(
                control_steps={1: 140, 2: 130, 3: 150, 4: 120},
                treatment_steps={1: 160, 2: 176, 3: 150, 4: 140},
                control_completed={step: False for step in range(1, 5)},
                treatment_completed={step: False for step in range(1, 5)},
                minimum_gain=25,
            ),
            "material-recovery-gain",
        )

    def test_requires_completion_advantage_or_fixed_gain(self):
        from tools.compare_g1_checkpoint_grids import classify_pair

        incomplete = {step: False for step in range(1, 5)}
        treatment_complete = {**incomplete, 3: True}
        self.assertEqual(
            classify_pair(
                control_steps={1: 140, 2: 130, 3: 150, 4: 120},
                treatment_steps={1: 160, 2: 174, 3: 499, 4: 140},
                control_completed=incomplete,
                treatment_completed=treatment_complete,
                minimum_gain=25,
            ),
            "recovery-completion-advantage",
        )
        self.assertEqual(
            classify_pair(
                control_steps={1: 140, 2: 130, 3: 150, 4: 120},
                treatment_steps={1: 160, 2: 174, 3: 150, 4: 140},
                control_completed=incomplete,
                treatment_completed=incomplete,
                minimum_gain=25,
            ),
            "no-material-recovery-gain",
        )

    def test_rejects_mismatched_checkpoint_grids(self):
        from tools.compare_g1_checkpoint_grids import validate_pair

        with self.assertRaisesRegex(ValueError, "checkpoint steps"):
            validate_pair(
                {"checkpoint_steps": [1, 2, 3, 4]},
                {"checkpoint_steps": [1, 2, 3, 5]},
            )


if __name__ == "__main__":
    unittest.main()
