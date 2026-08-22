# G1 Backward-Only Contact Compliance Implementation Plan

**Goal:** Decide whether one fixed compliant-contact backward derivative improves
fresh G1 gradient reliability while preserving the exact hard-contact forward
task.

**Architecture:** Add a small custom-VJP tree seam whose primal is the hard MJX
state and whose cotangent is routed through a soft-contact state. Reuse the E006
frozen capture and aggregation machinery in a dedicated evaluator. Keep training
code and default environment behavior unchanged.

## Task 1: Pure straight-through state seam

**Files:**
- Create `src/algorithms/shac/contact_compliance.py`
- Create `tests/test_shac_contact_compliance.py`

- Write failing tests for exact hard primal, zero hard cotangent, compliant
  cotangent, mixed floating/integer pytrees, shape mismatch, and fixed MuJoCo
  `solref` replacement.
- Implement `backward_from_compliant(hard, compliant)` and
  `with_contact_time_constant(model, 0.05)` without mutating inputs.
- Verify focused tests, Ruff, and bytecode compilation.

## Task 2: Frozen fresh-policy evaluator

**Files:**
- Create `tools/evaluate_g1_backward_contact_compliance.py`
- Create `tests/test_g1_backward_contact_compliance_evaluator.py`
- Narrowly extend `tools/evaluate_g1_ivw_h_gradients.py` only if a reusable
  optional transition seam avoids duplicated rollout code.

- Write failing tests for model-delta validation, registered classifier
  precedence, exact hard-forward identity, complete finite five-bin evidence,
  atomic publication, hash reopening, and exact E023/runtime/code provenance.
- Capture hard and backward-compliant fresh gradients for both fixed solvers and
  both fixed tapes.
- Persist action-node/decomposition evidence, per-phase and aggregate vectors,
  forward identity arrays, reliability summary, and one comparison plot.
- Run one small no-claim smoke, then the focused and neighboring suites.

## Task 3: Preregister, execute, and curate

**Files:**
- Create the next experiment record in `diffsim2real-lab`
- Update `research/state/current.yaml`, one finding if warranted, literature
  relationships, and `runtime/vault-previews.yaml`

- Pin the clean code commit, empty dirty patch, E023 hashes, model/controller
  hashes, GPU, fixed `0.02 -> 0.05` contact delta, outcome map, and artifact gates.
- Dry-run and launch exactly one seed-zero frozen audit.
- Inspect the first valid smoke and the final manifest; do not tune.
- Validate the registry, rebuild the vault, and expose the reliability plot at
  the top of the dashboard.

