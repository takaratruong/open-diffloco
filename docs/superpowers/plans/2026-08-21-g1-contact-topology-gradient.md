# G1 Contact-Topology Gradient Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether dropping only touchdown/liftoff pathwise derivatives makes fresh and retained G1 H24 actor gradients more solver-robust without erasing the useful signal.

**Architecture:** Add a pure grouped contact topology seam to the G1 environment, then add one default-off tree gradient barrier at the existing SHAC transition boundary. A dedicated frozen evaluator reconstructs exact fresh and E023 actors, captures ordinary/truncated H24 gradients under two registered fixed solvers, validates the preregistered robustness gates, and writes manifest-last evidence. No training is launched in this plan.

**Preflight correction (2026-08-21):** The implemented two-bit foot-support
detector produced zero events in all 120 fixed noisy H24 rollouts. Preserve its
pure helper for diagnostics, but drive the barrier from active unordered
body-pair contacts instead. Duplicate geom/contact points for one body pair
collapse to one bit. This correction is required before Task 4 and does not
change the fixed population, actor, loss, solver, or outcome gates.

**Tech Stack:** Python 3.11, JAX x64, MuJoCo/MJX, Flax, NumPy, pytest, Ruff, existing SHAC/CAGrad and research registry tools.

## Global Constraints

- Preserve the raw 125-frame walking reference and exact E023 action/noise/observation/reward contract.
- The event is a grouped active unordered body-pair contact signature; do not add a force or stiffness threshold.
- Preserve all forward state, reward, done, terminal, phase, observation, and action values exactly.
- Apply the barrier to transition reward and carried state only at nonterminal support changes; later direct actor gradients remain live.
- Keep the feature default off and preserve legacy checkpoint/hparams behavior when disabled.
- Run exactly eight scientific captures: two actors by two derivative modes by two solvers. Do not train or tune after the diagnostic.
- Canonical evidence lives under `diffsim2real-lab/runs/`; no scientific claim comes from logs alone.

---

### Task 1: Grouped G1 Foot-Contact Topology

**Files:**
- Create: `src/envs/g1_tracking/contact_topology.py`
- Modify: `src/envs/g1_tracking/environment.py`
- Create: `tests/test_g1_contact_topology.py`
- Modify: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Consumes: MJX `data.contact.geom`, signed `data.contact.dist`, model `geom_bodyid`, and exact left/right ankle-roll body IDs.
- Produces: `grouped_foot_support(contact_geom, contact_distance, geom_bodyid, foot_body_ids) -> jax.Array`; `contact_topology_event(previous, current, *, done) -> jax.Array`; `G1TrackingEnv.foot_support_signature(data) -> jax.Array`; `state.info["transition_contact_topology_event"]`.

- [ ] **Step 1: Write pure RED tests**

```python
def test_grouped_support_uses_all_geometries_and_ignores_inactive_rows():
    geom_bodyid = jp.asarray([0, 7, 7, 13, 18])
    contact_geom = jp.asarray([[1, 0], [2, 4], [3, 0], [4, 0]])
    efc_address = jp.asarray([0, -1, 3, 6])
    actual = grouped_foot_support(
        contact_geom, efc_address, geom_bodyid, jp.asarray([7, 13])
    )
    np.testing.assert_array_equal(actual, np.asarray([True, True]))


def test_topology_event_detects_touchdown_and_liftoff_but_not_reset():
    left = jp.asarray([False, True])
    double = jp.asarray([True, True])
    assert bool(contact_topology_event(left, double, done=False))
    assert bool(contact_topology_event(double, left, done=False))
    assert not bool(contact_topology_event(left, double, done=True))
```

- [ ] **Step 2: Run RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_contact_topology.py
```

Expected: collection fails because `src.envs.g1_tracking.contact_topology` does not exist.

- [ ] **Step 3: Implement the pure topology functions**

```python
def grouped_foot_support(contact_geom, contact_distance, geom_bodyid, foot_body_ids):
    pairs = jp.asarray(contact_geom)
    distances = jp.asarray(contact_distance)
    body_for_geom = jp.asarray(geom_bodyid)
    feet = jp.asarray(foot_body_ids)
    if pairs.ndim != 2 or pairs.shape[-1] != 2:
        raise ValueError("contact_geom must have shape (N, 2)")
    if distances.shape != pairs.shape[:1] or feet.shape != (2,):
        raise ValueError("contact topology inputs have incompatible shapes")
    safe_pairs = jp.clip(pairs, 0, body_for_geom.shape[0] - 1)
    pair_bodies = body_for_geom[safe_pairs]
    active = distances <= 0.0
    return jp.any(
        active[None, :, None]
        & (pair_bodies[None, :, :] == feet[:, None, None]),
        axis=(1, 2),
    )


def contact_topology_event(previous, current, *, done):
    previous = jp.asarray(previous, dtype=jp.bool_)
    current = jp.asarray(current, dtype=jp.bool_)
    if previous.shape != (2,) or current.shape != (2,):
        raise ValueError("foot support signatures must have shape (2,)")
    return jp.any(previous != current) & ~jp.asarray(done, dtype=jp.bool_)
```

- [ ] **Step 4: Integrate the exact G1 geometry boundary**

In `G1TrackingEnv.__init__`, retain the existing four-slot terrain body IDs and add exactly:

```python
self._support_foot_body_ids = jp.asarray(
    [self.body_ids[3], self.body_ids[6]], dtype=jp.int32
)
```

Add:

```python
def foot_support_signature(self, data: mjx.Data) -> jax.Array:
    return grouped_foot_support(
        data.contact.geom,
        data.contact.efc_address,
        self.mjx_model.geom_bodyid,
        self._support_foot_body_ids,
    )
```

In `step`, compute the pre-step signature from `state.data`, the transition signature from pre-reset `data`, and persist a stopped boolean `transition_contact_topology_event`. Add an exact false scalar to `_base_info`; do not reuse the post-reset state to classify the transition.

- [ ] **Step 5: Verify environment integration**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_contact_topology.py \
  tests/test_g1_tracking_environment.py -k 'contact_topology or transition_contact or reset_and_step'
```

Expected: all selected tests pass and an exact terminal reset reports no contact event.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/envs/g1_tracking/contact_topology.py \
  src/envs/g1_tracking/environment.py \
  tests/test_g1_contact_topology.py \
  tests/test_g1_tracking_environment.py
git commit -m "feat: expose grouped G1 contact topology"
```

---

### Task 2: Default-Off SHAC Contact Gradient Barrier

**Files:**
- Create: `src/algorithms/shac/contact_truncation.py`
- Modify: `src/algorithms/shac/algorithm.py`
- Create: `tests/test_shac_contact_topology_truncation.py`
- Modify: `tests/test_shac_gradient_aggregation.py`

**Interfaces:**
- Consumes: `candidate_next_state.info["transition_contact_topology_event"]` from Task 1.
- Produces: `contact_gradient_barrier(tree, event, *, enabled)`; `resolve_contact_topology_truncation_resume_setting(...)`; `train(..., actor_contact_topology_gradient_truncation=False, allow_resume_actor_contact_topology_gradient_truncation_change=False)`; persisted event and treatment telemetry.

- [ ] **Step 1: Write RED derivative and resume tests**

```python
def test_contact_barrier_preserves_primal_and_zeros_event_vjp():
    def objective(value, event):
        tree = {"reward": value**2, "state": 3.0 * value}
        result = contact_gradient_barrier(tree, event, enabled=True)
        return result["reward"] + result["state"]
    assert objective(2.0, True) == 10.0
    assert jax.grad(objective)(2.0, True) == 0.0
    assert jax.grad(objective)(2.0, False) == 7.0


def test_resume_requires_explicit_authority_for_contact_truncation_change():
    with pytest.raises(ValueError, match="explicit resume authority"):
        resolve_contact_topology_truncation_resume_setting(
            requested=True,
            resumed_hparams={"actor_contact_topology_gradient_truncation": False},
            is_resume=True,
            allow_change=False,
        )
```

- [ ] **Step 2: Run RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_shac_contact_topology_truncation.py
```

Expected: collection fails because `src.algorithms.shac.contact_truncation` does not exist.

- [ ] **Step 3: Implement the barrier and fail-closed resolver**

```python
def contact_gradient_barrier(tree, event, *, enabled):
    if not enabled:
        return tree
    event = jax.lax.stop_gradient(jp.asarray(event, dtype=jp.bool_))
    return jax.tree_util.tree_map(
        lambda value: jp.where(event, jax.lax.stop_gradient(value), value),
        tree,
    )


def resolve_contact_topology_truncation_resume_setting(
    *, requested, resumed_hparams, is_resume, allow_change
):
    if not isinstance(requested, bool) or not isinstance(allow_change, bool):
        raise ValueError("contact topology truncation settings must be boolean")
    if not is_resume:
        return requested
    if resumed_hparams is None or (
        "actor_contact_topology_gradient_truncation" not in resumed_hparams
    ):
        raise ValueError("contact topology truncation requires complete resume metadata")
    saved = resumed_hparams["actor_contact_topology_gradient_truncation"]
    if not isinstance(saved, bool):
        raise ValueError("saved contact topology truncation must be boolean")
    if saved != requested and not allow_change:
        raise ValueError("changing contact topology truncation requires explicit resume authority")
    return requested if allow_change else saved
```

- [ ] **Step 4: Wire the barrier at the existing actor-loss seam**

Immediately after `candidate_next_state = env.step(state, noisy_action)`, read and stop the event bit, then create `gradient_next_state = contact_gradient_barrier(candidate_next_state, event, enabled=...)`. Use `gradient_next_state` for reward and the carried `next_state`; use the original candidate only for event telemetry and existing forward diagnostics. Add `contact_topology_event` to the trajectory dictionary. Preserve ordinary done/terminal/reset semantics.

Resolve the two flags after checkpoint loading, serialize both in hparams, and add checkpoint-row fields:

```python
"actor_contact_topology_gradient_truncation": bool(...),
"contact_topology_event_count": int(jp.sum(trajs["contact_topology_event"])),
"contact_topology_event_fraction": float(jp.mean(trajs["contact_topology_event"])),
```

- [ ] **Step 5: Prove direct post-event actor dependence remains live**

Add a two-step synthetic scan test in which step one is detached and step two contains a fresh multiplication by the shared parameter. Assert the gradient contains only step two's derivative, while the forward scalar equals the untruncated result exactly.

- [ ] **Step 6: Run focused and neighboring tests**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_shac_contact_topology_truncation.py \
  tests/test_shac_ahac_integration.py \
  tests/test_shac_gradient_aggregation.py \
  tests/test_shac_microbatch.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  src/algorithms/shac/contact_truncation.py \
  src/algorithms/shac/algorithm.py \
  tests/test_shac_contact_topology_truncation.py
```

Expected: all tests and Ruff pass; the disabled path's exact synthetic gradient and forward value match legacy.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/algorithms/shac/contact_truncation.py \
  src/algorithms/shac/algorithm.py \
  tests/test_shac_contact_topology_truncation.py \
  tests/test_shac_gradient_aggregation.py
git commit -m "feat: truncate SHAC gradients at contact changes"
```

---

### Task 3: Frozen Two-Solver Gradient Discriminator

**Files:**
- Create: `tools/evaluate_g1_contact_topology_gradients.py`
- Create: `tests/test_g1_contact_topology_gradient_evaluator.py`

**Interfaces:**
- Consumes: exact E023 checkpoint/hparams/reference/model/controller hashes, `evaluate_g1_tracking._load_policy(..., training_initialization=True)` for fresh actor reconstruction, Task 2 barrier, `aggregate_audit_direction`, and registered solver profiles.
- Produces: `build_fixed_phase_population(seed)`, `solver_angular_error(cosine)`, `classify_contact_topology_gradient_audit(...)`, `validate_gradient_evidence(...)`, `publish_gradient_evidence(...)`, CLI `python -m tools.evaluate_g1_contact_topology_gradients`.

- [ ] **Step 1: Write RED contract tests**

Cover exact phase counts `(24,24,24,24,24)`, fixed float32 `(120,24,29)` noise, invalid seed, cosine/error math, robust/neutral/destructive/invalid precedence, fewer than three event-bearing bins, fewer than 24 total events, nonfinite arrays, hash tampering, solver drift, and manifest-last atomic publication.

```python
def test_classifier_requires_both_actor_boundaries_to_improve():
    evidence = valid_evidence()
    evidence["fresh"]["angular_error_reduction"] = 0.21
    evidence["e023"]["angular_error_reduction"] = 0.19
    assert classify_contact_topology_gradient_audit(evidence) == (
        "contact-truncation-neutral"
    )
```

- [ ] **Step 2: Run RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_contact_topology_gradient_evaluator.py
```

Expected: collection fails because the evaluator module does not exist.

- [ ] **Step 3: Implement pure evidence and classifier seams**

Use exact constants:

```python
POPULATION = 120
PHASES = (0, 25, 50, 75, 100)
REPLICAS_PER_PHASE = 24
HORIZON = 24
ACTION_DIM = 29
SOLVERS = ("g1-4x5", "diagnostic-10x20")
MODES = ("ordinary", "contact_truncated")
ACTORS = ("fresh", "e023")
```

Classify invalid first, destructive second, robust third, and neutral last. Robust requires both actors to reduce aggregate angular error by at least 20 percent, no phase cosine regression below `-0.02`, nominal ordinary/truncated cosine at least `0.8`, and norm ratio in `[0.25,4.0]`.

- [ ] **Step 4: Reconstruct exact states, actors, and noise**

Instantiate `G1TrackingRMR50HzActionParityEnv` from the exact E023 hparams and raw walk reference. Reset 24 replicas at each fixed phase, restore deterministic ten-frame actor context using the environment's exact reference reset contract, and assert qpos/qvel/phase/history identities. Load E023 from its exact checkpoint and construct fresh parameters with `_load_policy(..., checkpoint=None, seed=0, actor_hidden=(512,256,128), actor_layer_norm=True, actor_zero_output=True, training_initialization=True)`. Require fresh architecture and normalizer contracts to match E023's registered recipe.

- [ ] **Step 5: Capture and aggregate all eight gradients**

Implement one static H24 loss parameterized only by `truncate_contact`. It must mirror normalized observations, reparameterized RMR noise, sampled-action boundary, environment step, discounted done/terminal accumulation, and zero bootstrap. Return event masks as auxiliaries. JIT/vmap per-environment gradients in bounded chunks, then clip at 1.0 and call `aggregate_audit_direction(..., phase_count=125, alpha=0.5, iterations=32)`.

Run both modes inside each registered `solver_context`. Verify ordinary/truncated forward qpos, qvel, rewards, done, terminal, actions, and event masks are bit-identical within a solver before comparing gradients.

- [ ] **Step 6: Publish strict evidence and diagnostics**

Write `preflight.json`, `gradient_evidence.npz`, `gradient_summary.json`, and `gradient_cosines.png` atomically; write `completion.json` last. Bind the exact checkpoint, hparams, reference, model, controller, code commit, dirty patch, environment, both solver profiles, state arrays, noise tape, and every output SHA. Reopen and validate everything before success.

- [ ] **Step 7: Run the compiled one-environment smoke**

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m tools.evaluate_g1_contact_topology_gradients \
  --checkpoint /home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-023/20260815T023359Z/seed-0/training_runs/training_runs/shac_20260814_193402/checkpoint_step_1572864.pkl \
  --hparams /home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-023/20260815T023359Z/seed-0/training_runs/training_runs/shac_20260814_193402/hparams.json \
  --reference-path /home/ubuntu/projects/diffsim2real/outputs/rmr_motion_walk_win137_212_named.npz \
  --repository "$PWD" --code-commit "$(git rev-parse HEAD)" \
  --output-directory /tmp/g1-contact-topology-smoke --seed 0 \
  --smoke --smoke-phase 25
```

Expected: `smoke_summary.json` says `valid: true`, ordinary/truncated forward identity passes, both gradients are finite/nonzero, and at least one contact event is observed.

- [ ] **Step 8: Verify and commit Task 3**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/test_g1_contact_topology_gradient_evaluator.py \
  tests/test_g1_contact_topology.py \
  tests/test_shac_contact_topology_truncation.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  tools/evaluate_g1_contact_topology_gradients.py \
  tests/test_g1_contact_topology_gradient_evaluator.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  tools/evaluate_g1_contact_topology_gradients.py
git diff --check
git add tools/evaluate_g1_contact_topology_gradients.py \
  tests/test_g1_contact_topology_gradient_evaluator.py
git commit -m "feat: audit G1 contact topology gradients"
```

Expected: all commands pass and the worktree is clean.

---

### Task 4: Register, Execute, Curate, and Branch

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260821-004.yaml`
- Modify after evaluation: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Modify after evaluation: `/home/ubuntu/projects/diffsim2real-lab/runtime/vault-previews.yaml`
- Generate: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/Dashboard.md`

**Interfaces:**
- Consumes: clean final code commit from Tasks 1–3 and exact immutable input hashes from E023.
- Produces: one schema-v1 registered experiment, one guarded frozen diagnostic run, one preregistered verdict, canonical raw artifacts, and the generated Dashboard plot.

- [ ] **Step 1: Write the E004 experiment record**

Use parent `E-20260821-003`, thread `g1-tracking-shac`, problem tags `contact-discontinuity`, `gradient-variance`, and `short-horizon-credit`, and method tags `mjx`, `shac`, and `cagrad`. Copy the four outcomes and exact gates from the committed design. Resolve and write the actual clean 40-character code commit and empty dirty-patch SHA; never use a symbolic revision in registry provenance.

- [ ] **Step 2: Validate and dry-run**

```bash
cd /home/ubuntu/projects/diffsim2real-lab
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260821-004 --dry-run
```

Expected: both commands succeed and the dry run prints exactly one seed-zero frozen evaluator command with one GPU and no training command.

- [ ] **Step 3: Execute exactly once**

```bash
uv run python -m tools.runexp E-20260821-004
```

Expected: the guarded run emits exactly eight gradient captures and writes `completion.json` last. Do not relaunch with changed inputs or gates.

- [ ] **Step 4: Independently validate and classify**

Reopen every JSON/NPZ/PNG, recompute all hashes and gradient cosines, require exact ordinary/truncated forward identity, and select only the ordered registered outcome. If any check fails, mark `invalid-execution`; do not infer from logs.

- [ ] **Step 5: Curate the program state**

Record the exact verdict and quantitative fresh/E023 aggregate and phase cosines in E004. Update `research/state/current.yaml` with the resulting branch:

- robust: authorize one separately registered exact fresh training run;
- neutral: reject topology truncation and design a hybrid score-function or paired behavioral accept/reject treatment;
- destructive: reject the barrier;
- invalid: repair execution only.

Add the cosine plot to `runtime/vault-previews.yaml`, rebuild the vault, and keep canonical evidence under `runs/`.

- [ ] **Step 6: Final verification**

```bash
cd /home/ubuntu/projects/diffsim2real-lab
uv run python tools/researchctl.py validate
uv run python tools/researchctl.py build-vault
uv run python tools/researchctl.py validate
test -s docs/vault/attachments/research-previews/E-20260821-004/contact_sheet.png
```

Expected: registry validation passes before and after rebuild, the Dashboard lists E004 first, and the diagnostic plot opens from the Dashboard. Do not publish a rollout video because this experiment does not train or select a policy.

## Plan Self-Review

- Spec coverage: grouped topology, exact forward barrier, default-off resume contract, fresh/E023 actors, two solvers, eight captures, classification, provenance, atomic artifacts, stop rule, and Dashboard projection each map to a task.
- Placeholder scan: no TBD/TODO or unnamed implementation step remains; runtime commit/hash values are explicitly resolved and persisted rather than represented by placeholders.
- Type consistency: Task 1's boolean scalar event is the exact Task 2 input; Task 2's flag and telemetry are the exact Task 3 loss mode/evidence; Task 3's four outcomes are the exact Task 4 registry outcomes.
