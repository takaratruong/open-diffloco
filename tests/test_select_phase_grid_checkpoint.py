import unittest


class SelectPhaseGridCheckpointTest(unittest.TestCase):
    def _summary(self, values, sha="a" * 64):
        return {
            "phases": [0, 100, 200, 300, 400],
            "steps": {
                str(phase): value
                for phase, value in zip(
                    (0, 100, 200, 300, 400), values, strict=True
                )
            },
            "checkpoint_sha256": sha,
        }

    def test_selects_minimum_then_median_then_mean_then_earliest(self):
        from tools.select_phase_grid_checkpoint import select_checkpoint

        summaries = {
            10: self._summary([27, 59, 81, 54, 44], "a" * 64),
            20: self._summary([27, 88, 38, 53, 44], "b" * 64),
            30: self._summary([27, 55, 81, 55, 44], "c" * 64),
            40: self._summary([27, 55, 81, 55, 44], "d" * 64),
        }

        result = select_checkpoint(summaries)

        self.assertEqual(result["selected_step"], 30)
        self.assertEqual(result["selected_survival"], [27, 55, 81, 55, 44])
        self.assertEqual(result["selected_minimum"], 27)
        self.assertEqual(result["selected_median"], 55)
        self.assertEqual(result["selected_mean"], 52.4)
        self.assertEqual(len(result["checkpoints"]), 4)

    def test_rejects_incomplete_or_nonpositive_survival(self):
        from tools.select_phase_grid_checkpoint import select_checkpoint

        incomplete = self._summary([1, 2, 3, 4, 5])
        del incomplete["steps"]["400"]
        with self.assertRaises(ValueError):
            select_checkpoint({10: incomplete})
        with self.assertRaises(ValueError):
            select_checkpoint({10: self._summary([1, 2, 0, 4, 5])})

    def test_accepts_provenance_rich_flax_summary(self):
        from tools.select_phase_grid_checkpoint import select_checkpoint

        summary = {
            "checkpoint_sha256": "e" * 64,
            "summary": {
                "phases": [0, 100, 200, 300, 400],
                "survival": [68, 62, 89, 56, 58],
            },
        }

        result = select_checkpoint({1_867_776: summary})

        self.assertEqual(result["selected_step"], 1_867_776)
        self.assertEqual(result["selected_survival"], [68, 62, 89, 56, 58])


if __name__ == "__main__":
    unittest.main()
