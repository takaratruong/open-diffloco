# G1 LAFAN1 Long-Reference DiffSim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the proven compact from-random-weights G1 DiffSim policy on a single RMR-preprocessed ten-second LAFAN1 dance slice without changing any other learning mechanism.

**Architecture:** Preserve RMR's clean CSV-to-NPZ converter as the only interpolation and velocity authority, add a small immutable metadata wrapper around its output, and extend the existing MuJoCo reference loader to accept that named RMR schema alongside legacy `X`/`V`. Thread an explicit reference path and timebase through SHAC and evaluation, then register one seed-zero experiment whose only scientific change from E006 is the 499-transition reference.

**Tech Stack:** Python 3.10/3.11, NumPy, MuJoCo 3.9, MJX/JAX, Flax/Optax, `unittest`, RMR Isaac Lab converter, YAML research registry.

## Global Constraints

- Source dataset revision is `ce1572906efe6157840e8474d5a0d7aa87481e74`, licensed CC BY-NC-ND 4.0.
- Source is `g1/dance1_subject2.csv`, inclusive frames 122–422, input 30 FPS, output 50 FPS.
- Run clean RMR `scripts/csv_to_npz.py` at repository commit `8a886f3a9df561df3454e6b6233ab8c54d66f097`; converter SHA-256 is `1724fe91ee4da6d1136510db83598c4875d20297cf0f56dd55c13326122ea4f5`.
- Preserve seed zero, random 512-by-512 actor, nonzero random output head, no LayerNorm, 256 environments, horizon 12, 128 updates, 393,216 transitions, actor LR 1e-3, actor bootstrap zero, and action noise 1.0→0.1.
- Keep exact RSI; do not add reset perturbations, observation noise, pushes, terrain, adaptive phase sampling, extra environments, a longer differentiable horizon, PPO weights, replay, or a residual controller.
- The RMR slice produces 500 states and 499 carried control transitions. Never reintroduce the old 60-step cap.
- Use one seed only. Do not infer repeatability or sim-to-real validity.
- Preserve unrelated dirty files in `/home/ubuntu/projects/rmr_tracking` and `/home/ubuntu/worktrees/diffsim2real-lab/slice3-implementation-20260730`.

---

## File Map

- Create `tools/prepare_g1_rmr_reference.py`: add immutable joint/body-order and provenance metadata to the byte-preserved arrays emitted by RMR.
- Create `tests/test_prepare_g1_rmr_reference.py`: validate metadata wrapping without invoking Isaac Lab.
- Modify `src/envs/g1_tracking/reference.py`: load either legacy `X`/`V` or named RMR NPZ and reconstruct MuJoCo generalized state.
- Modify `tests/test_g1_tracking_reference.py`: cover RMR schema, Jacobian-derived root velocity, and legacy compatibility.
- Modify `src/envs/g1_tracking/environment.py`: load the controller before the reference and validate reference FPS/stride against the 20 ms control period.
- Modify `src/algorithms/shac/algorithm.py`: accept, transport, hash, and record the reference path/stride.
- Modify `tools/run_g1_tracking_rmr50_shac.py`: expose reference CLI arguments and derive the reference-length episode contract.
- Modify `tests/test_g1_tracking_runner.py`: prove the new arguments preserve the E006 defaults and transport the LAFAN contract.
- Modify `tools/evaluate_g1_tracking.py`: accept the reference path/stride, default to the complete suffix, and publish strict stability metrics.
- Modify `tests/test_g1_tracking_evaluator.py`: prove the evaluator does not truncate long references and computes strict metrics.
- Create one experiment YAML and one hypothesis YAML in the research registry only after preprocessing passes.

---

### Task 1: Preserve RMR Output With Explicit Metadata

**Files:**
- Create: `tools/prepare_g1_rmr_reference.py`
- Create: `tests/test_prepare_g1_rmr_reference.py`

**Interfaces:**
- Consumes: an RMR NPZ containing `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`.
- Produces: `prepare_reference(input_path: Path, output_path: Path, *, joint_names: tuple[str, ...], source_metadata: dict[str, object]) -> dict[str, object]`; the output NPZ retains every source numeric array exactly and adds `joint_names`, `root_body_name`, and `root_body_index`. The returned manifest contains file hashes, shapes, FPS, frame count, and source metadata.

- [ ] **Step 1: Write failing metadata-preservation tests**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare_g1_rmr_reference import prepare_reference


JOINT_NAMES = tuple(f"joint_{index}" for index in range(29))


def make_rmr_fixture(frames: int) -> dict[str, np.ndarray]:
    identity = np.zeros((frames, 2, 4), dtype=np.float32)
    identity[..., 0] = 1.0
    return {
        "fps": np.asarray([50], dtype=np.int32),
        "joint_pos": np.zeros((frames, 29), dtype=np.float32),
        "joint_vel": np.zeros((frames, 29), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 2, 3), dtype=np.float32),
        "body_quat_w": identity,
        "body_lin_vel_w": np.zeros((frames, 2, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 2, 3), dtype=np.float32),
    }


class PrepareReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.input_path = Path(self.temp.name) / "raw.npz"
        self.output_path = Path(self.temp.name) / "prepared.npz"

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_reference_preserves_numeric_arrays_and_adds_names(self):
        arrays = make_rmr_fixture(frames=4)
        np.savez(self.input_path, **arrays)
        manifest = prepare_reference(
            self.input_path,
            self.output_path,
            joint_names=JOINT_NAMES,
            source_metadata={"revision": "abc", "frame_range": [122, 422]},
        )
        with np.load(self.output_path, allow_pickle=False) as result:
            for key, expected in arrays.items():
                np.testing.assert_array_equal(result[key], expected)
            self.assertEqual(tuple(result["joint_names"]), JOINT_NAMES)
            self.assertEqual(str(result["root_body_name"]), "pelvis")
            self.assertEqual(int(result["root_body_index"]), 0)
        self.assertEqual(manifest["frames"], 4)

    def test_prepare_reference_rejects_missing_arrays(self):
        arrays = make_rmr_fixture(frames=4)
        arrays.pop("joint_vel")
        np.savez(self.input_path, **arrays)
        with self.assertRaisesRegex(ValueError, "joint_vel"):
            prepare_reference(
                self.input_path,
                self.output_path,
                joint_names=JOINT_NAMES,
                source_metadata={},
            )
```

- [ ] **Step 2: Run the focused tests and confirm the import failure**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_prepare_g1_rmr_reference -v`

Expected: FAIL because `tools.prepare_g1_rmr_reference` does not exist.

- [ ] **Step 3: Implement the strict metadata wrapper**

```python
REQUIRED_ARRAYS = (
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)

def prepare_reference(input_path, output_path, *, joint_names, source_metadata):
    with np.load(input_path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_ARRAYS if key not in archive]
        if missing:
            raise ValueError(f"RMR reference missing arrays: {missing}")
        arrays = {key: np.array(archive[key], copy=True) for key in REQUIRED_ARRAYS}
    frames = arrays["joint_pos"].shape[0]
    if frames <= 0 or arrays["joint_pos"].shape != (frames, 29):
        raise ValueError("joint_pos must have shape (T, 29) with T > 0")
    if arrays["joint_vel"].shape != (frames, 29):
        raise ValueError("joint_vel must have shape (T, 29)")
    for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        if arrays[key].ndim != 3 or arrays[key].shape[0] != frames or arrays[key].shape[-1] != 3:
            raise ValueError(f"{key} must have shape (T, B, 3)")
    if arrays["body_quat_w"].ndim != 3 or arrays["body_quat_w"].shape[0] != frames or arrays["body_quat_w"].shape[-1] != 4:
        raise ValueError("body_quat_w must have shape (T, B, 4)")
    if arrays["body_pos_w"].shape[1] == 0:
        raise ValueError("RMR reference must contain at least one rigid body")
    if int(np.asarray(arrays["fps"]).reshape(-1)[0]) != 50:
        raise ValueError("RMR reference FPS must equal 50")
    if len(joint_names) != 29 or len(set(joint_names)) != 29 or any(not name for name in joint_names):
        raise ValueError("joint_names must contain 29 unique nonempty names")
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be finite and numeric")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **arrays,
        joint_names=np.asarray(joint_names),
        root_body_name=np.asarray("pelvis"),
        root_body_index=np.asarray(0, dtype=np.int32),
    )
    manifest = {
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "frames": frames,
        "fps": 50,
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "source": source_metadata,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest
```

Validation must require 50 FPS, `(T, 29)` joint arrays, common positive `T`, at least one rigid body, quaternion trailing size four, finite numeric arrays, 29 unique nonempty joint names, and no pre-existing output. Write the manifest atomically next to the prepared NPZ as `<output>.manifest.json`.
Define `sha256_file(path)` directly in the module as a streaming 1 MiB-block
SHA-256 helper and import `hashlib`, `json`, `Path`, and `numpy` explicitly.

- [ ] **Step 4: Run focused tests**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_prepare_g1_rmr_reference -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the self-contained preparation tool**

```bash
git add tools/prepare_g1_rmr_reference.py tests/test_prepare_g1_rmr_reference.py
git commit -m "feat: preserve named RMR motion references"
```

---

### Task 2: Load Named RMR References Into MuJoCo

**Files:**
- Modify: `src/envs/g1_tracking/reference.py`
- Modify: `tests/test_g1_tracking_reference.py`

**Interfaces:**
- Consumes: `load_mujoco_reference(model, reference_path, body_names=RMR_G1_BODY_NAMES, controller=None)` where `controller` is required only for RMR-format archives.
- Produces: `MujocoReference` with immutable `qpos`, `qvel`, rigid-body targets, and `fps: float | None`; legacy `X`/`V` results remain unchanged with `fps=None`.

- [ ] **Step 1: Add failing RMR-schema and legacy-compatibility tests**

```python
def test_named_rmr_reference_maps_source_joints_and_root_velocity(self):
    frames = 5
    source_joint_pos = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 1000.0
    source_joint_vel = source_joint_pos / 10.0
    root_pos = np.zeros((frames, 1, 3), dtype=np.float32)
    root_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    root_quat[..., 0] = 1.0
    root_lin_vel = np.tile(np.asarray([0.2, -0.1, 0.05], dtype=np.float32), (frames, 1, 1))
    root_ang_vel = np.tile(np.asarray([0.1, 0.2, -0.15], dtype=np.float32), (frames, 1, 1))
    fixture = Path(self.temp.name) / "named_rmr.npz"
    np.savez(
        fixture,
        fps=np.asarray([50]),
        joint_pos=source_joint_pos,
        joint_vel=source_joint_vel,
        body_pos_w=root_pos,
        body_quat_w=root_quat,
        body_lin_vel_w=root_lin_vel,
        body_ang_vel_w=root_ang_vel,
        joint_names=np.asarray(self.controller.actor_joint_names),
        root_body_name=np.asarray("pelvis"),
        root_body_index=np.asarray(0),
    )
    reference = load_mujoco_reference(self.model, fixture, controller=self.controller)
    self.assertEqual(reference.fps, 50.0)
    expected_model_joints = source_joint_pos[:, self.controller.actor_to_model_permutation]
    np.testing.assert_allclose(reference.qpos[:, 7:], expected_model_joints, atol=0.0)
    np.testing.assert_allclose(reference.body_lin_vel[:, 0], root_lin_vel[:, 0], atol=2e-5)
    np.testing.assert_allclose(reference.body_ang_vel[:, 0], root_ang_vel[:, 0], atol=2e-5)

def test_legacy_xv_reference_remains_unchanged(self):
    actual = load_mujoco_reference(self.model, REFERENCE)
    with np.load(REFERENCE, allow_pickle=False) as archive:
        expected_qpos = np.array(archive["X"], dtype=np.float64, copy=True)
        expected_qpos[:, 3:7] /= np.linalg.norm(expected_qpos[:, 3:7], axis=-1, keepdims=True)
        np.testing.assert_array_equal(actual.qpos, expected_qpos)
        np.testing.assert_array_equal(actual.qvel, np.asarray(archive["V"], dtype=np.float64))
    self.assertIsNone(actual.fps)
```

Add `self.temp = tempfile.TemporaryDirectory()` in `setUp` and clean it in
`tearDown`; the existing module already supplies the real model/controller paths.

- [ ] **Step 2: Run the tests and observe the missing schema support**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_reference -v`

Expected: the new RMR test FAILS because the loader requires `X` and `V`; existing tests PASS.

- [ ] **Step 3: Implement dual-schema loading and exact root Jacobian recovery**

```python
@dataclass(frozen=True)
class MujocoReference:
    qpos: np.ndarray
    qvel: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    body_lin_vel: np.ndarray
    body_ang_vel: np.ndarray
    body_ids: tuple[int, ...]
    body_names: tuple[str, ...]
    fps: float | None

def _root_generalized_velocity(model, qpos, root_body_id, linear_w, angular_w):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, root_body_id)
    root_jacobian = np.concatenate((jacp[:, :6], jacr[:, :6]), axis=0)
    target = np.concatenate((linear_w, angular_w))
    return np.linalg.solve(root_jacobian, target)
```

Detect schemas from key sets. For named RMR input, require the stored joint names to equal `controller.actor_joint_names`, root name `pelvis`, root index zero, and FPS 50. Build qpos from logged root pose plus source-to-model joint permutation, normalize only root quaternions, build qvel joint columns with the same permutation, and solve root qvel per frame. After the existing FK loop, require reconstructed root pose and world velocity to match the stored RMR root arrays at `atol=2e-5`, `rtol=0`.

- [ ] **Step 4: Run reference and environment tests**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_reference tests.test_g1_tracking_environment -v`

Expected: all tests PASS, including the unchanged 121-state legacy fixture.

- [ ] **Step 5: Commit the loader seam**

```bash
git add src/envs/g1_tracking/reference.py tests/test_g1_tracking_reference.py
git commit -m "feat: load RMR motion archives in G1 MJX"
```

---

### Task 3: Thread Reference Path and Timebase Through Training

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tools/run_g1_tracking_rmr50_shac.py`
- Modify: `tests/test_g1_tracking_environment.py`
- Modify: `tests/test_g1_tracking_runner.py`

**Interfaces:**
- Consumes: `reference_path: str | None`, `reference_stride: int`, and `reference_fps: float | None` at the runner/train boundary.
- Produces: an environment whose `control_reference_dt`, `reference_length`, and `max_episode_length` are derived and whose hparams include absolute reference path, SHA-256, FPS, stride, states, and transitions.

- [ ] **Step 1: Write failing transport and timebase tests**

```python
def test_native_runner_transports_50hz_named_reference(self):
    kwargs = build_train_kwargs(
        steps=393_216,
        num_envs=256,
        seed=0,
        checkpoint_interval=30_720,
        validated_task=True,
        reference_path="/tmp/dance.npz",
        reference_stride=1,
    )
    self.assertEqual(kwargs["reference_path"], "/tmp/dance.npz")
    self.assertEqual(kwargs["reference_stride"], 1)

def test_environment_rejects_reference_timebase_mismatch(self):
    with self.assertRaisesRegex(ValueError, "reference timebase"):
        G1TrackingRMR50HzValidatedEnv(
            reference_path=self.named_50hz_fixture,
            reference_stride=2,
        )
```

- [ ] **Step 2: Run the focused tests and confirm argument failures**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_runner tests.test_g1_tracking_environment -v`

Expected: new tests FAIL with unexpected/missing reference arguments.

- [ ] **Step 3: Implement the minimal transport**

Add `reference_path` and `reference_stride` to `train()`, pass them only to G1 environments, and preserve old defaults. Load `RMRController` before the reference so named RMR archives can validate their source order. Let `G1TrackingRMR50HzSourceStepEnv` accept a stride override rather than discarding it.

```python
g1_reference_kwargs = {}
if env_variant.startswith("g1_tracking"):
    if reference_path is not None:
        g1_reference_kwargs["reference_path"] = reference_path
    if reference_stride is not None:
        g1_reference_kwargs["reference_stride"] = reference_stride
```

Expand `g1_reference_kwargs` into the existing `Go2Env` constructor call after its
current fixed arguments; do not replace or reorder those arguments.

When `env.reference.fps` is present, require
`env.reference_stride / env.reference.fps == env.dt` within `1e-12`. Derive
`reference_transitions = ceil((reference_length - 1) / reference_stride)` and
set the recorded G1 `max_episode_length` to that value. Hash the reference with
streaming SHA-256 and include all reference fields in `hparams.json`.

- [ ] **Step 4: Add CLI flags and preserve E006 defaults**

```python
parser.add_argument("--reference-path", type=Path)
parser.add_argument("--reference-stride", type=int, default=2)
```

`build_train_kwargs()` must default to the legacy reference and stride two; the
new experiment supplies the prepared RMR NPZ and stride one explicitly.

- [ ] **Step 5: Run focused tests**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_runner tests.test_g1_tracking_environment tests.test_g1_tracking_reference -v`

Expected: all tests PASS and the legacy task still reports 60 transitions.

- [ ] **Step 6: Commit the training plumbing**

```bash
git add src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tools/run_g1_tracking_rmr50_shac.py tests/test_g1_tracking_environment.py tests/test_g1_tracking_runner.py
git commit -m "feat: parameterize G1 tracking references"
```

---

### Task 4: Extend Replay-Free Evaluation to the Full Reference

**Files:**
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tests/test_g1_tracking_evaluator.py`

**Interfaces:**
- Consumes: `--reference-path`, `--reference-stride`, and optional `--max-steps`; omitted max steps means the complete suffix.
- Produces: existing evaluation artifacts plus per-frame termination errors and `summarize_stability_errors(errors: dict[str, np.ndarray]) -> dict[str, float]` maxima for anchor height, anchor XY, projected gravity, and distal height.

- [ ] **Step 1: Write failing parser and summary tests**

```python
def test_parser_allows_complete_reference_evaluation(self):
    args = build_parser().parse_args(["--output-dir", "/tmp/out", "--reference-path", "/tmp/dance.npz", "--reference-stride", "1"])
    self.assertIsNone(args.max_steps)
    self.assertEqual(args.reference_stride, 1)

def test_stability_summary_reports_maximum_termination_errors(self):
    summary = summarize_stability_errors({
        "anchor_z_error": np.asarray([0.01, 0.12]),
        "anchor_xy_error": np.asarray([0.04, 0.08]),
        "gravity_z_error": np.asarray([0.10, 0.31]),
        "distal_z_error": np.asarray([0.03, 0.09]),
    })
    self.assertEqual(summary["max_anchor_z_error"], 0.12)
    self.assertEqual(summary["max_gravity_z_error"], 0.31)
```

- [ ] **Step 2: Run evaluator tests and confirm failures**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_evaluator -v`

Expected: FAIL because the parser lacks reference arguments and max steps defaults to 120.

- [ ] **Step 3: Implement full-suffix bounds and stability metrics**

Pass reference arguments into `make_evaluation_env`. After reset, compute:

```python
remaining = math.ceil((env.reference_length - 1 - args.phase) / env.reference_stride)
step_limit = remaining if args.max_steps is None else min(args.max_steps, remaining)
```

After stepping, recover the advanced carried body state and call:

```python
body_pos, body_quat, _, _ = env._body_state(state.data)
errors = env.termination_errors(
    phase=state.info["phase"],
    body_pos=body_pos,
    body_quat=body_quat,
)
```

Append all four errors to the record. Add their maxima to `summary.json`, along with reference path,
SHA-256, FPS, stride, state count, expected transitions, and whether an
intermediate reset occurred. Keep clip-end `done` distinct from true terminal.

- [ ] **Step 4: Run evaluator and environment tests**

Run: `/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_evaluator tests.test_g1_tracking_environment -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the evaluation gate**

```bash
git add tools/evaluate_g1_tracking.py tests/test_g1_tracking_evaluator.py
git commit -m "feat: evaluate full G1 reference stability"
```

---

### Task 5: Acquire and Preprocess the Pinned Dance Slice

**Files:**
- Generate outside git: `artifacts/E-20260808-000/reference/source/dance1_subject2.csv`
- Generate outside git: `artifacts/E-20260808-000/reference/rmr_motion_raw.npz`
- Generate outside git: `artifacts/E-20260808-000/reference/dance1_subject2_f122_422_50hz.npz`
- Generate outside git: matching `.manifest.json`

**Interfaces:**
- Consumes: the pinned Hugging Face URL and clean RMR converter.
- Produces: the exact prepared reference and provenance manifest used by training.

- [ ] **Step 1: Download and hash the pinned CSV**

Run:

```bash
curl -fL 'https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/ce1572906efe6157840e8474d5a0d7aa87481e74/g1/dance1_subject2.csv' -o artifacts/E-20260808-000/reference/source/dance1_subject2.csv
sha256sum artifacts/E-20260808-000/reference/source/dance1_subject2.csv
wc -l artifacts/E-20260808-000/reference/source/dance1_subject2.csv
```

Expected: 3,945 rows, 1,352,694 bytes, and a stable SHA-256 recorded verbatim in the manifest and experiment YAML.

- [ ] **Step 2: Run clean RMR CSV-to-NPZ offline**

From `/home/ubuntu/projects/rmr_tracking`, run the repository's configured Isaac Lab Python with:

```bash
WANDB_MODE=offline python scripts/csv_to_npz.py \
  --input_file /absolute/path/to/dance1_subject2.csv \
  --input_fps 30 --output_fps 50 --frame_range 122 422 \
  --output_name dance1_subject2_f122_422_50hz --headless
```

Expected: `~/tmp/motion.npz` exists with 500 frames; copy it once to the experiment reference directory and hash it. Do not edit RMR's dirty unrelated files.

- [ ] **Step 3: Add immutable names/provenance and run the loader preflight**

Run `tools/prepare_g1_rmr_reference.py` with the exact RMR controller source
joint names and pinned source metadata, then instantiate
`G1TrackingRMR50HzValidatedEnv(reference_path="/home/ubuntu/worktrees/open-diffloco/g1-rmr-50hz-20260805/artifacts/E-20260808-000/reference/dance1_subject2_f122_422_50hz.npz", reference_stride=1)`
under the fixed solver scope.

Expected: prepared reference has 500 states, 499 transitions, 50 FPS, all finite arrays, unit quaternions, exact source/model joint mapping, and reconstructed root pose/velocity within `2e-5` of RMR output.

- [ ] **Step 4: Render the kinematic reference and inspect it**

Use the existing side-by-side evaluator's reference renderer or a bounded reference-only invocation to produce a ten-second MP4 and contact sheet before training.

Expected: the intended dance slice, correct root orientation, correct limbs, no joint-order scramble, no discontinuous interpolation, and no obvious ground-frame offset. If any condition fails, classify preprocessing invalid and stop before GPU training.

---

### Task 6: Verify, Commit, and Push the Implementation

**Files:**
- All files from Tasks 1–4

- [ ] **Step 1: Run the complete focused suite**

Run:

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_prepare_g1_rmr_reference \
  tests.test_g1_tracking_reference \
  tests.test_g1_tracking_environment \
  tests.test_g1_tracking_runner \
  tests.test_g1_tracking_evaluator -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run repository diff and smoke gates**

Run:

```bash
git diff --check
/home/ubuntu/miniconda3/envs/rl/bin/python tools/run_g1_tracking_rmr50_shac.py --help
/home/ubuntu/miniconda3/envs/rl/bin/python tools/evaluate_g1_tracking.py --help
```

Expected: clean diff check and both CLIs list reference path/stride options.

- [ ] **Step 3: Review the aggregate diff against the design**

Confirm no default task, actor, reward, termination, controller, solver, action support, noise schedule, or legacy reference behavior changed. Confirm no generated artifact is staged.

- [ ] **Step 4: Push the implementation branch**

```bash
git push takaratruong research/g1-rmr-50hz-20260805
```

Expected: remote branch resolves to the verified local HEAD.

---

### Task 7: Register and Execute the One-Seed Experiment

**Files:**
- Create: `/home/ubuntu/worktrees/diffsim2real-lab/slice3-implementation-20260730/research/hypotheses/H-G1-034.yaml`
- Create: `/home/ubuntu/worktrees/diffsim2real-lab/slice3-implementation-20260730/research/experiments/E-20260808-000.yaml`
- Modify after evaluation: `research/state/current.yaml`

**Interfaces:**
- Consumes: clean implementation commit, prepared reference SHA-256/manifest, fixed E006 hyperparameters.
- Produces: one curated one-seed result with fixed-checkpoint full-reference evaluations and videos.

- [ ] **Step 1: Register the single causal question**

Hypothesis: replacing the 1.2-second reference with a continuous ten-second RMR-preprocessed dance, while holding E006 fixed, exposes enough carried-state diversity for the compact direct DiffSim actor to track all 499 transitions without progressive lateral collapse.

The outcome map must exactly match the design's four branches: stable tracking, delayed instability, no basin entry, or invalid conversion/execution.

- [ ] **Step 2: Pin the exact execution contract**

Use this scientific command shape with absolute paths and a single GPU selected at execution time:

```bash
python -u tools/run_g1_tracking_rmr50_shac.py \
  --steps 393216 --num-envs 256 --seed 0 \
  --checkpoint-interval 30720 --actor-lr 1e-3 \
  --action-noise-std 1.0 --action-noise-std-end 0.1 \
  --unroll-length 12 --actor-bootstrap-scale 0.0 \
  --validated-task --actor-hidden 512 512 \
  --no-actor-layer-norm --random-actor-output-head \
  --reference-path /absolute/path/to/dance1_subject2_f122_422_50hz.npz \
  --reference-stride 1
```

Pin code commit, absence of dirty patch, source/model/controller/reference hashes, RMR converter commit/hash, environment lock, solver 4/5, GPU count one, and wall-time budget 120 minutes.

- [ ] **Step 3: Validate and dry-run the registry entry**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260808-000 --dry-run
```

Expected: registry valid and dry run resolves the pinned command without execution.

- [ ] **Step 4: Execute one registered experiment**

Run: `uv run python -m tools.runexp E-20260808-000`

Expected: exactly 128 finite actor updates or a guarded invalid failure. Do not launch another experiment while this one is active.

- [ ] **Step 5: Evaluate six fixed checkpoints over all 499 transitions**

Evaluate updates 10/20/30/40/50/128 with seed zero, phase zero, complete-reference default, validated environment, stride one, compact actor options, and no replay. Render every second control frame at 25 FPS to keep video compact while preserving the full carried rollout.

- [ ] **Step 6: Curate only a preregistered outcome**

Verify manifests, hashes, logs, checkpoint leaves, summaries, NPZ trajectories, videos, and contact sheets. Select by full survival, then reward/errors; enforce all behavioral gates and visual review. Mark invalid if the causal comparison or artifact contract was not executed.

- [ ] **Step 7: Update registry, vault, and Git history**

Run:

```bash
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
```

Run focused registry tests, commit only owned YAML/current-state/generated pages, push the research branch, and report the exact outcome and next decision without claiming repeatability.
