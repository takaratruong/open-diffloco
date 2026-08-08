import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare_g1_rmr_reference import prepare_reference


JOINT_NAMES = tuple(f"joint_{index}" for index in range(29))


def make_rmr_fixture(frames: int = 4) -> dict[str, np.ndarray]:
    identity = np.zeros((frames, 2, 4), dtype=np.float32)
    identity[..., 0] = 1.0
    return {
        "fps": np.asarray([50], dtype=np.int32),
        "joint_pos": np.arange(frames * 29, dtype=np.float32).reshape(frames, 29),
        "joint_vel": np.zeros((frames, 29), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 2, 3), dtype=np.float32),
        "body_quat_w": identity,
        "body_lin_vel_w": np.zeros((frames, 2, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 2, 3), dtype=np.float32),
    }


class PrepareReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.input_path = root / "raw.npz"
        self.output_path = root / "prepared.npz"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_preserves_source_arrays_and_adds_explicit_order_metadata(self):
        source_arrays = make_rmr_fixture()
        np.savez(self.input_path, **source_arrays)

        manifest = prepare_reference(
            self.input_path,
            self.output_path,
            joint_names=JOINT_NAMES,
            source_metadata={"revision": "abc", "frame_range": [122, 422]},
        )

        with np.load(self.output_path, allow_pickle=False) as prepared:
            for key, expected in source_arrays.items():
                np.testing.assert_array_equal(prepared[key], expected)
            self.assertEqual(tuple(map(str, prepared["joint_names"])), JOINT_NAMES)
            self.assertEqual(str(prepared["root_body_name"]), "pelvis")
            self.assertEqual(int(prepared["root_body_index"]), 0)
        self.assertEqual(manifest["frames"], 4)
        self.assertEqual(manifest["fps"], 50)
        manifest_path = self.output_path.with_suffix(".npz.manifest.json")
        self.assertEqual(json.loads(manifest_path.read_text()), manifest)

    def test_rejects_missing_required_array(self):
        source_arrays = make_rmr_fixture()
        source_arrays.pop("joint_vel")
        np.savez(self.input_path, **source_arrays)

        with self.assertRaisesRegex(ValueError, "joint_vel"):
            prepare_reference(
                self.input_path,
                self.output_path,
                joint_names=JOINT_NAMES,
                source_metadata={},
            )

    def test_rejects_nonfinite_source_data(self):
        source_arrays = make_rmr_fixture()
        source_arrays["body_pos_w"][0, 0, 0] = np.nan
        np.savez(self.input_path, **source_arrays)

        with self.assertRaisesRegex(ValueError, "body_pos_w"):
            prepare_reference(
                self.input_path,
                self.output_path,
                joint_names=JOINT_NAMES,
                source_metadata={},
            )

    def test_refuses_to_overwrite_prepared_reference(self):
        np.savez(self.input_path, **make_rmr_fixture())
        self.output_path.write_bytes(b"already here")

        with self.assertRaises(FileExistsError):
            prepare_reference(
                self.input_path,
                self.output_path,
                joint_names=JOINT_NAMES,
                source_metadata={},
            )


if __name__ == "__main__":
    unittest.main()
