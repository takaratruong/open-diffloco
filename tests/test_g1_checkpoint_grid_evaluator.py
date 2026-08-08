import unittest
from pathlib import Path


class G1CheckpointGridEvaluatorTest(unittest.TestCase):
    def test_requires_four_ordered_checkpoint_steps_and_unique_gpus(self):
        from tools.evaluate_g1_checkpoint_grid import validate_grid

        checkpoints = tuple(Path(f"/tmp/{step}.pkl") for step in (1, 2, 3, 4))
        validate_grid(
            checkpoints=checkpoints,
            checkpoint_steps=(638976, 688128, 737280, 786432),
            checkpoint_sha256=("a", "b", "c", "d"),
            gpu_ids=("1", "2", "3", "4"),
        )
        for steps, gpus in (
            ((688128, 638976, 737280, 786432), ("1", "2", "3", "4")),
            ((638976, 688128, 737280), ("1", "2", "3", "4")),
            ((638976, 688128, 737280, 786432), ("1", "1", "3", "4")),
        ):
            with self.subTest(steps=steps, gpus=gpus):
                with self.assertRaises(ValueError):
                    validate_grid(
                        checkpoints=checkpoints,
                        checkpoint_steps=steps,
                        checkpoint_sha256=("a", "b", "c", "d"),
                        gpu_ids=gpus,
                    )

    def test_classification_uses_requested_material_gate(self):
        from tools.evaluate_g1_checkpoint_grid import classify_checkpoint_grid

        self.assertEqual(
            classify_checkpoint_grid(
                survival={442368: 90, 491520: 121, 540672: 100, 589824: 99},
                completed={step: False for step in (442368, 491520, 540672, 589824)},
                material_survival=120,
            ),
            "material-continuation-gain",
        )
        self.assertEqual(
            classify_checkpoint_grid(
                survival={442368: 90, 491520: 100, 540672: 119, 589824: 99},
                completed={step: False for step in (442368, 491520, 540672, 589824)},
                material_survival=120,
            ),
            "no-material-continuation-gain",
        )
        self.assertEqual(
            classify_checkpoint_grid(
                survival={442368: 90, 491520: 100, 540672: 499, 589824: 99},
                completed={
                    442368: False,
                    491520: False,
                    540672: True,
                    589824: False,
                },
                material_survival=120,
            ),
            "complete-long-reference-tracking",
        )

    def test_selects_earliest_checkpoint_at_best_survival(self):
        from tools.evaluate_g1_checkpoint_grid import select_checkpoint

        summaries = {
            442368: {"steps": 100, "mean_reward": 0.08},
            491520: {"steps": 130, "mean_reward": 0.07},
            540672: {"steps": 130, "mean_reward": 0.09},
            589824: {"steps": 120, "mean_reward": 0.10},
        }

        self.assertEqual(select_checkpoint(summaries), 540672)

    def test_builds_existing_evaluator_command_at_phase_zero(self):
        from tools.evaluate_g1_checkpoint_grid import build_evaluator_command

        command = build_evaluator_command(
            python=Path("/env/python"),
            evaluator=Path("/repo/evaluate.py"),
            checkpoint=Path("/artifacts/checkpoint.pkl"),
            reference=Path("/artifacts/reference.npz"),
            output_dir=Path("/output/checkpoint"),
        )

        self.assertEqual(command[0], "/env/python")
        self.assertEqual(command[command.index("--phase") + 1], "0")
        self.assertEqual(
            command[command.index("--checkpoint") + 1],
            "/artifacts/checkpoint.pkl",
        )
        self.assertIn("--random-actor-output-head", command)


if __name__ == "__main__":
    unittest.main()
