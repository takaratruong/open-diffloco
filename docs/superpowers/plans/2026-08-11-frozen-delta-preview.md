# Frozen Delta-Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate a causally isolated future-minus-current reference preview for the frozen G1 SHAC actor.

**Architecture:** Preserve the existing 174-value preview layout and append-only migration, but make its representation explicit at the environment boundary. Thread the immutable mode through training resume metadata, the experiment runner, and evaluation provenance; default every existing call to `absolute`.

**Tech Stack:** Python 3.11, JAX, Flax, MJX, pytest, Ruff.

## Global Constraints

- `absolute` remains the default and must preserve existing behavior.
- `delta` means each +4/+8/+12 qpos/qvel command minus the current reference qpos/qvel command in actor joint order.
- Resume may introduce `delta` only when upgrading a checkpoint with no preview; an existing preview mode cannot change.
- The scientific runner resumes exact E008, uses 64 effective-512 H12 updates, five-bin CAGrad, zero margin, and eight-update checkpoints.
- Only 89,088 newest-frame preview input weights and their Adam moments are trainable.
- Never bind a server to `0.0.0.0`.

---

### Task 1: Environment and resume contract

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Test: `tests/test_g1_tracking_environment.py`
- Test: `tests/test_shac_exact_resume.py`

**Interfaces:**
- Consumes: `actor_reference_lookahead_steps: tuple[int, ...]`.
- Produces: `actor_reference_preview_mode: str` on `G1TrackingEnv` and `train`; `resolve_future_reference_preview_mode(...) -> str`.

- [ ] **Step 1: Write failing environment tests**

Add tests that instantiate `(4, 8, 12)` with mode `delta`, compare the 174-value
suffix to explicit future-minus-current commands, verify clamping near the end,
and reject `relative`, `True`, or delta-without-lookahead.

```python
def test_actor_observation_supports_delta_future_reference(self):
    env = G1TrackingEnv(
        xml_path=str(MODEL),
        reference_path=str(REFERENCE),
        controller_path=str(CONTROLLER),
        actor_history_len=1,
        actor_reference_lookahead_steps=(4, 8, 12),
        actor_reference_preview_mode="delta",
    )
    phase = 37
    actual = np.asarray(env._future_reference_command(jnp.array(phase))).reshape(3, 58)
    actor_order = np.asarray(env.model_to_actor_permutation)
    current = np.concatenate((
        np.asarray(env.qpos_reference[phase, 7:])[actor_order],
        np.asarray(env.qvel_reference[phase, 6:])[actor_order],
    ))
    expected = []
    for offset in (4, 8, 12):
        future = phase + offset * env.reference_stride
        expected.append(np.concatenate((
            np.asarray(env.qpos_reference[future, 7:])[actor_order],
            np.asarray(env.qvel_reference[future, 6:])[actor_order],
        )) - current)
    np.testing.assert_allclose(actual, np.stack(expected), rtol=0.0, atol=1e-12)
```

- [ ] **Step 2: Verify the environment tests fail for the missing mode**

Run:
`python -m pytest -q tests/test_g1_tracking_environment.py -k 'delta_future_reference or preview_mode'`

Expected: failures because `actor_reference_preview_mode` is not implemented.

- [ ] **Step 3: Implement the minimal environment mode**

Validate `actor_reference_preview_mode in {"absolute", "delta"}` in
`G1TrackingEnv.__init__`. In `_future_reference_command`, construct the existing
`commands` matrix and, only for delta mode, subtract:

```python
current = jp.concatenate(
    (
        self.qpos_reference[phase, 7:][actor_order],
        self.qvel_reference[phase, 6:][actor_order],
    )
)
commands = commands - current[None, :]
```

- [ ] **Step 4: Write failing resume tests**

Test default `train` mode `absolute`, legacy no-preview upgrade to `delta`,
legacy raw-preview metadata defaulting to `absolute`, exact delta resume, invalid
metadata, and rejection of absolute-to-delta changes after preview exists.

```python
def test_delta_preview_mode_requires_legacy_upgrade_or_exact_resume(self):
    assert resolve_future_reference_preview_mode(
        {"actor_reference_lookahead_steps": []},
        requested_mode="delta",
        future_reference_upgrade=True,
    ) == "delta"
    assert resolve_future_reference_preview_mode(
        {
            "actor_reference_lookahead_steps": [4, 8, 12],
            "actor_reference_preview_mode": "delta",
        },
        requested_mode="delta",
        future_reference_upgrade=False,
    ) == "delta"
    with pytest.raises(ValueError, match="must match"):
        resolve_future_reference_preview_mode(
            {"actor_reference_lookahead_steps": [4, 8, 12]},
            requested_mode="delta",
            future_reference_upgrade=False,
        )
```

- [ ] **Step 5: Implement resume and metadata wiring**

Add:

```python
def resolve_future_reference_preview_mode(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_mode: str,
    future_reference_upgrade: bool,
) -> str:
    if requested_mode not in {"absolute", "delta"}:
        raise ValueError("future reference preview mode is invalid")
    if resumed_hparams is None:
        return requested_mode
    saved_steps = tuple(
        resumed_hparams.get("actor_reference_lookahead_steps", ())
    )
    if not saved_steps:
        if requested_mode == "delta" and not future_reference_upgrade:
            raise ValueError("delta preview requires explicit upgrade authority")
        return requested_mode
    saved_mode = resumed_hparams.get(
        "actor_reference_preview_mode", "absolute"
    )
    if saved_mode not in {"absolute", "delta"}:
        raise ValueError("saved future reference preview mode is invalid")
    if saved_mode != requested_mode:
        raise ValueError("future reference preview mode must match the checkpoint")
    return saved_mode
```

Thread the resolved mode to `G1TrackingEnv`, persist it in `hparams.json`, and
retain `absolute` as the backward-compatible saved default.

- [ ] **Step 6: Run focused green tests**

Run:
`python -m pytest -q tests/test_g1_tracking_environment.py -k 'future_reference or preview_mode' tests/test_shac_exact_resume.py`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the environment contract**

```bash
git add src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tests/test_g1_tracking_environment.py tests/test_shac_exact_resume.py
git commit -m "feat: add delta future reference mode"
```

### Task 2: Immutable runner and evaluator provenance

**Files:**
- Create: `tools/run_g1_frozen_delta_preview_continuation.py`
- Create: `tests/test_g1_frozen_delta_preview_runner.py`
- Modify: `tools/evaluate_g1_flax_phase_grid.py`
- Modify: `tests/test_evaluate_g1_flax_phase_grid.py`

**Interfaces:**
- Consumes: `build_frozen_preview_dense_checkpoint_kwargs(...)` and preview mode from Task 1.
- Produces: `build_frozen_delta_preview_kwargs(...)`; evaluator CLI `--actor-reference-preview-mode {absolute,delta}`; artifact field `actor_reference_preview_mode`.

- [ ] **Step 1: Write failing runner tests**

Compare the candidate kwargs with the dense raw-preview parent and assert the
only scientific difference is `actor_reference_preview_mode == "delta"`; also
assert endpoint 1,572,864, H12, effective batch 512, checkpoint interval 49,152,
zero termination margin, and immutable parser surface.

```python
parent = build_frozen_preview_dense_checkpoint_kwargs(
    "g1-4x5", Path("/tmp/dance.npz"), 0, Path("/tmp/e008.pkl")
)
candidate = build_frozen_delta_preview_kwargs(
    "g1-4x5", Path("/tmp/dance.npz"), 0, Path("/tmp/e008.pkl")
)
assert set(candidate) == set(parent) | {"actor_reference_preview_mode"}
assert candidate["actor_reference_preview_mode"] == "delta"
assert {
    key: value for key, value in candidate.items()
    if key != "actor_reference_preview_mode"
} == parent
```

- [ ] **Step 2: Verify runner RED**

Run:
`python -m pytest -q tests/test_g1_frozen_delta_preview_runner.py`

Expected: import failure for the missing runner.

- [ ] **Step 3: Implement the immutable runner**

Build from `build_frozen_preview_dense_checkpoint_kwargs`, set only
`actor_reference_preview_mode = "delta"`, and use the established solver
context/output-root pattern.

- [ ] **Step 4: Write failing evaluator tests**

Require the parser default `absolute`, accept `delta`, and require
`build_payload(..., actor_reference_preview_mode="delta")` to record the exact
mode.

```python
assert build_parser().parse_args(required_args).actor_reference_preview_mode == "absolute"
assert build_parser().parse_args(
    [*required_args, "--actor-reference-preview-mode", "delta"]
).actor_reference_preview_mode == "delta"
assert build_payload(
    results,
    phases=(0, 100, 200, 300, 400),
    reference_transitions=499,
    checkpoint_path="/tmp/checkpoint.pkl",
    checkpoint_sha256="a" * 64,
    reference_path="/tmp/reference.npz",
    reference_sha256="b" * 64,
    solver_profile="g1-4x5",
    actor_reference_preview_mode="delta",
)["actor_reference_preview_mode"] == "delta"
```

- [ ] **Step 5: Implement evaluator mode wiring**

Pass the CLI mode to `make_evaluation_env`, include it in the payload, and
leave existing invocation behavior absolute by default.

- [ ] **Step 6: Run focused green tests**

Run:
`python -m pytest -q tests/test_g1_frozen_delta_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py tests/test_g1_frozen_preview_dense_checkpoint_runner.py`

Expected: all tests pass.

- [ ] **Step 7: Commit runner and evaluator**

```bash
git add tools/run_g1_frozen_delta_preview_continuation.py tests/test_g1_frozen_delta_preview_runner.py tools/evaluate_g1_flax_phase_grid.py tests/test_evaluate_g1_flax_phase_grid.py
git commit -m "feat: run frozen delta preview continuation"
```

### Task 3: Verification and experiment handoff

**Files:**
- Verify: all Task 1 and Task 2 files
- Verify: `docs/superpowers/specs/2026-08-11-frozen-delta-preview-design.md`
- Verify: `docs/superpowers/plans/2026-08-11-frozen-delta-preview.md`

**Interfaces:**
- Consumes: completed implementation and tests.
- Produces: one clean scientific code commit suitable for registry pinning.

- [ ] **Step 1: Run the focused suite**

Run:
`python -m pytest -q tests/test_g1_tracking_environment.py -k 'future_reference or preview_mode' tests/test_shac_exact_resume.py tests/test_shac_future_reference_migration.py tests/test_shac_preview_adapter.py tests/test_g1_frozen_delta_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py`

- [ ] **Step 2: Run static verification**

Run:
`ruff check src/envs/g1_tracking/environment.py src/algorithms/shac/algorithm.py tools/run_g1_frozen_delta_preview_continuation.py tools/evaluate_g1_flax_phase_grid.py tests/test_g1_tracking_environment.py tests/test_shac_exact_resume.py tests/test_g1_frozen_delta_preview_runner.py tests/test_evaluate_g1_flax_phase_grid.py`

Run:
`python -m py_compile tools/run_g1_frozen_delta_preview_continuation.py tools/evaluate_g1_flax_phase_grid.py`

Run: `git diff --check`

- [ ] **Step 3: Self-review the exact diff and commit documentation**

Confirm the default path is unchanged, delta is the only treatment change, no
unrelated files are staged, and commit the spec/plan with the implementation if
they are not already committed.
