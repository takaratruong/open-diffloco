import pickle
import tempfile
import unittest
from pathlib import Path

from src.algorithms.shac import algorithm


class PeriodicCheckpointArchivalTest(unittest.TestCase):
    def test_archives_each_step_while_advancing_latest(self):
        self.assertTrue(
            hasattr(algorithm, "save_periodic_checkpoint"),
            "SHAC exposes no durable periodic-checkpoint writer",
        )
        first_state = {"step": 64_512}
        second_state = {"step": 125_952}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first_path = algorithm.save_periodic_checkpoint(
                first_state, output, 64_512
            )
            second_path = algorithm.save_periodic_checkpoint(
                second_state, output, 125_952
            )

            self.assertEqual(
                first_path, output / "checkpoint_step_064512.pkl"
            )
            self.assertEqual(
                second_path, output / "checkpoint_step_125952.pkl"
            )
            with first_path.open("rb") as stream:
                self.assertEqual(pickle.load(stream), first_state)
            with second_path.open("rb") as stream:
                self.assertEqual(pickle.load(stream), second_state)
            with (output / "checkpoint_latest.pkl").open("rb") as stream:
                self.assertEqual(pickle.load(stream), second_state)


if __name__ == "__main__":
    unittest.main()
